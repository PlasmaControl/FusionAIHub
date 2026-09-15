"""labelmaker's labels and events -> `labels_wide.parquet` and `events.parquet`.

Two semantics from the plan (section 2 "Label semantics", Appendix C item 7) are enforced here and
are the reason this module exists at all rather than a `pd.concat`:

* **A probability is summarised, never thresholded into truth.** `labels_wide` keeps the
  distribution of a label over a shot -- `max/mean/p95` over the valid samples -- beside whatever
  operating threshold there is and what that threshold did (`frac_above`, `first_above_t_s`,
  `n_intervals`, `longest_interval_s`). Where there is no operating threshold those columns are
  null, never zero: "nobody chose a threshold" and "the alarm never fired" are different facts and
  a stored 0.0 destroys the difference. Every statistic is over the samples labelmaker marked
  valid, because an invalid sample is not a measurement of a low probability.
* **A threshold says whose it is.** `thr_source` is `card` when the number is the model card's own
  operating point and `config` when it is the alarm level `configs/ideate/labels.yaml` chose, and
  `""` where there is no threshold at all. One map (`_threshold_map`, card over config) feeds both
  `labels_wide` and the forecast events, so the two tables cannot disagree about the number an
  alarm was raised at, and each forecast event repeats the pair in its `attrs`.
* **A forecast is not an observation.** The DSM risk heads answer "will this happen within h?", so
  a run of one above its threshold becomes an event with `evidence_kind="forecast"`, a finite
  `horizon_s` and `source="label_forecast"` -- and never `evidence_kind="detector"`. Which label
  series are risks, at what threshold and horizon, is `configs/ideate/labels.yaml`'s `forecasts:`
  block, and the labels.yaml comment there says which numbers are the card's and which are ours.

`db/event_sources.parquet` is the fourth table and the one that makes an empty answer readable:
labelmaker writes `events/<shot>_sources.parquet` saying which detector RAN over which shot, span
and channel -- including the ones that completed with zero detections -- and this join ingests
every file that exists. A shot with no file contributes no rows, which is how the database says
"nobody looked here" rather than returning a silent empty list. See `labels.event_sources`.

The event schema is labelmaker's, imported and reused: `COLUMNS`, `DTYPES`, `Event` (whose
`__post_init__` is what rejects a forecast with no horizon) and `read_events`, which already reads
an absent file as an empty typed frame -- `events/` does not exist until the mask job has run, and
a join must not care.

## The sample-duration convention

A label is a series on a uniform grid: labelmaker writes one value every `time_step_ms`. A run of
samples `i..j` above a threshold is read here as the half-open span `[t[i], t[j] + dt)`, `dt` being
the grid step, so a single sample above the threshold is one time step of alarm rather than an
instant of zero duration. `t_cov` follows the same reading of the same grid. Everything downstream
that measures a forecast's duration measures it this way.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from labelmaker.config import Paths as LabelmakerPaths
from labelmaker.config import atomic_path
from labelmaker.events import schema as events_schema
from labelmaker.labels import store as label_store
from labelmaker.models import registry

from .. import config
from . import claims as claims_mod
from . import event_sources as es
from .claims import CLAIMS_COLUMNS, CLAIMS_DTYPES
from .event_sources import SOURCES_COLUMNS

_NAN = float("nan")

#: `labels_wide.parquet`, in this order (plan section 5.5).
#:
#: `n_intervals` is pandas' *nullable* Int32 rather than plain int32 for the reason in the module
#: docstring: a label with no operating threshold has no interval count, and the only int32 that
#: can say so is a null. Everything else in the row is a real number or a real NaN.
LABELS_WIDE_DTYPES: dict[str, str] = {
    "shot": "int32",
    "slug": "object",
    "label": "object",
    "n_valid": "int32",
    "valid_frac": "float32",
    "max_valid": "float32",
    "mean_valid": "float32",
    "p95_valid": "float32",
    "thr": "float32",
    "thr_source": "object",
    "frac_above": "float32",
    "first_above_t_s": "float32",
    "n_intervals": "Int32",
    "longest_interval_s": "float32",
    "artifact_sha256": "object",
}
LABELS_WIDE_COLUMNS: tuple[str, ...] = tuple(LABELS_WIDE_DTYPES)

#: What `write_tables` merges into `db/manifest.json` under `"labels"`. The full manifest a join
#: returns also names the missing shots; the file keeps the counts, because at full corpus the
#: names are 17,000 integers nobody reads out of a manifest.
MANIFEST_KEYS: tuple[str, ...] = (
    "labelmaker_root", "n_shots", "n_shots_with_labels", "n_shots_with_events",
    "n_labels_wide_rows", "n_events", "n_forecast_events", "n_text_claims",
    "n_shots_missing_labels", "n_shots_missing_events", "labelmaker_git_sha",
    "thresholds_from_card", "thresholds_from_config", "thresholds", "forecast_rules", "lexicon",
    "n_event_source_rows", "n_shots_with_source_rows", "n_shots_with_observed_products",
    "n_shots_unprocessed", "n_sources_ok", "n_sources_skipped", "n_sources_error",
)

#: Where a `thr` came from. `""` is the third value `labels_wide.thr_source` takes and means
#: "there is no threshold", which is not a source.
THR_SOURCES: tuple[str, ...] = ("card", "config")

#: The file the `forecasts:` block lives in.
FORECASTS_CONFIG = "labels.yaml"

_NO_THRESHOLD: tuple[float, str] = (_NAN, "")


@dataclass(frozen=True)
class ForecastRule:
    """One risk label read as a forecast: `configs/ideate/labels.yaml`'s `forecasts:` entries."""

    slug: str
    label: str
    thr: float
    horizon_s: float
    phenomenon: str

    @property
    def key(self) -> str:
        return f"{self.slug}/{self.label}"


@dataclass(frozen=True)
class JoinResult:
    """The three tables and the manifest of one `ideate labels join`."""

    labels_wide: pd.DataFrame
    events: pd.DataFrame
    claims: pd.DataFrame
    #: `db/event_sources.parquet`: which detector RAN over which shot and span, including the
    #: ones that emitted nothing. Empty for a shot labelmaker has not processed, which is what
    #: lets `get_events` say "unprocessed" instead of returning a silent empty list.
    sources: pd.DataFrame = field(default_factory=lambda: es.empty_sources())
    manifest: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------ small helpers


def _paths(labelmaker_root) -> LabelmakerPaths:
    return LabelmakerPaths(root=Path(labelmaker_root))


def _shots(shots: Iterable[int]) -> list[int]:
    return sorted({int(s) for s in shots})


def empty_events() -> pd.DataFrame:
    """No events, labelmaker's columns, labelmaker's dtypes."""
    return pd.DataFrame(
        {name: pd.Series(dtype=events_schema.DTYPES[name]) for name in events_schema.COLUMNS}
    )


def _step(t: np.ndarray) -> float:
    """The label grid's step. Median, not `t[1] - t[0]`: one duplicated timestamp in a file must
    not become the unit every duration in the shot is measured in."""
    return float(np.median(np.diff(t))) if t.size > 1 else 0.0


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of True as inclusive `(first, last)` index pairs."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > 1)
    starts = np.concatenate(([idx[0]], idx[breaks + 1]))
    ends = np.concatenate((idx[breaks], [idx[-1]]))
    return [(int(a), int(b)) for a, b in zip(starts, ends, strict=True)]


def _series(
    path: Path, slug: str, label: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """`(t, y, valid, attrs)` for one label, through labelmaker's own reader.

    A label whose `(C, T)` payload has more than one channel is a profile label; the label schema
    stores it, but there is no honest one-number summary of a profile and averaging the channels
    would invent one. Loud, and only reachable when a model that emits profiles is implemented.
    """
    arr = label_store.read_label(path, slug, label)
    if arr.y.ndim != 2 or arr.y.shape[0] != 1:
        raise ValueError(
            f"{path}: {slug}/{label} is {arr.y.shape}; labels_wide summarises single-channel "
            "labels only (a profile label needs its own summary, not a channel mean)"
        )
    try:
        valid = label_store.read_label(path, slug, f"{label}_valid")
    except KeyError as exc:  # a file not written by labels.store.write_labels
        raise ValueError(f"{path}: {slug}/{label} has no _valid companion") from exc
    return arr.x, arr.y[0], np.asarray(valid.y[0], dtype=bool), arr.attrs


# ------------------------------------------------------------------------------- thresholds


def thresholds_from_card(card: Mapping) -> dict[str, float]:
    """`"<slug>/<label>" -> threshold` for the outputs of one model card that record one.

    A card records its operating point as `threshold:` on an entry of its `labelmaker.outputs`
    list. Absence is the normal case and means "this model's authors did not name an operating
    point", which is a fact about the model, not a reason to invent 0.5.
    """
    block = dict(card.get("labelmaker") or {})
    slug = str(block.get("slug") or "")
    out: dict[str, float] = {}
    for entry in block.get("outputs") or []:
        thr = entry.get("threshold")
        if thr is not None:
            out[f"{slug}/{entry['name']}"] = float(thr)
    return out


def card_thresholds() -> dict[str, float]:
    """Every operating threshold recorded by every model card, via labelmaker's registry.

    None of the five implemented cards records one today -- the tearing DSM card publishes a
    threshold sweep and says in as many words that it is "not a threshold recommendation" -- so
    this is empty and `labels_wide.thr` is NaN for every label the `forecasts:` block does not
    name. That is the honest answer, and it is why `frac_above` and its companions are null rather
    than computed against a number nobody chose.
    """
    out: dict[str, float] = {}
    for slug in registry.model_slugs():
        out.update(thresholds_from_card(registry.read_card(slug)))
    return out


def forecast_rules(rules: Sequence[Mapping] | None = None) -> tuple[ForecastRule, ...]:
    """The `forecasts:` block of `configs/ideate/labels.yaml`, or an explicit list of the same."""
    if rules is None:
        rules = config.load_yaml("labels.yaml").get("forecasts") or []
    return tuple(
        ForecastRule(
            slug=str(r["slug"]), label=str(r["label"]), thr=float(r["thr"]),
            horizon_s=float(r["horizon_s"]), phenomenon=str(r["phenomenon"]),
        )
        for r in rules
    )


def _threshold_map(
    rules: Sequence[ForecastRule], from_card: Mapping[str, float]
) -> dict[str, tuple[float, str]]:
    """`"<slug>/<label>" -> (threshold, source)` for every label anyone has named a level for.

    A card's own operating point wins over the config's alarm level, because a card is the model's
    word about its own model. One map, built once per join and read by both `labels_wide` and
    `label_forecast_events`, is what makes the two tables incapable of disagreeing: if a card ever
    starts recording a `threshold:` on a label the `forecasts:` block also names, the events move
    to it in the same call the `labels_wide` row does.
    """
    return {
        **{r.key: (float(r.thr), "config") for r in rules},
        **{key: (float(thr), "card") for key, thr in from_card.items()},
    }


def _normalise_thresholds(
    thresholds: Mapping[str, tuple[float, str]] | None,
) -> dict[str, tuple[float, str]]:
    """`thresholds` as `{key: (thr, source)}`, or ValueError.

    A bare number is refused rather than defaulted to a source. Silently calling an unattributed
    number a card's operating point is exactly the confusion `thr_source` was added to end, and a
    caller with a threshold in hand always knows where it came from.
    """
    if thresholds is None:
        return _threshold_map((), card_thresholds())
    out: dict[str, tuple[float, str]] = {}
    for key, value in thresholds.items():
        if isinstance(value, str) or not isinstance(value, Sequence) or len(value) != 2:
            raise ValueError(
                f"threshold for {key!r} is {value!r}; a threshold must name its source as "
                f"(thr, source) with source in {THR_SOURCES}"
            )
        thr, source = value
        if source not in THR_SOURCES:
            raise ValueError(f"threshold for {key!r} has source {source!r}, not one of "
                             f"{THR_SOURCES}")
        out[key] = (float(thr), str(source))
    return out


def forecast_rules_path() -> Path:
    """The file the `forecasts:` block is read from."""
    return config.CONFIG_DIR / FORECASTS_CONFIG


def rules_sha256(rules: Sequence[ForecastRule]) -> str:
    """A digest of the rule set itself, in canonical order.

    Of the *rules*, not of the file's bytes: a comment rewrite in `labels.yaml` must not read as a
    changed alarm, and a rule set passed in by a caller with no file behind it still has to be
    identifiable. Two joins whose forecast events could differ have different digests here.
    """
    payload = json.dumps(
        sorted([r.slug, r.label, r.thr, r.horizon_s, r.phenomenon] for r in rules),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ------------------------------------------------------------------------------- labels_wide


def _labels_wide_frame(rows: list[dict]) -> pd.DataFrame:
    cols = {}
    for name, dtype in LABELS_WIDE_DTYPES.items():
        values = [r[name] for r in rows]
        cols[name] = (
            pd.array(values, dtype=dtype)
            if dtype == "Int32"
            else pd.Series(values, dtype=dtype)
        )
    return pd.DataFrame(cols, columns=list(LABELS_WIDE_COLUMNS))


def _summary(
    shot: int, slug: str, label: str, t, y, valid, threshold: tuple[float, str], sha: str
) -> dict:
    """One `labels_wide` row. Every statistic is over `valid`; see the module docstring."""
    thr, thr_source = threshold
    n_valid = int(valid.sum())
    yv = y[valid]
    row = {
        "shot": shot,
        "slug": slug,
        "label": label,
        "n_valid": n_valid,
        "valid_frac": (n_valid / y.size) if y.size else _NAN,
        "max_valid": float(yv.max()) if n_valid else _NAN,
        "mean_valid": float(yv.mean()) if n_valid else _NAN,
        "p95_valid": float(np.percentile(yv, 95)) if n_valid else _NAN,
        "thr": float(thr),
        "thr_source": thr_source if np.isfinite(thr) else "",
        "frac_above": _NAN,
        "first_above_t_s": _NAN,
        "n_intervals": None,
        "longest_interval_s": _NAN,
        "artifact_sha256": sha,
    }
    # A non-finite valid sample stops the alarm columns the same way it stops `max/mean/p95`
    # above, which come out NaN by propagation. A `frac_above` computed by skipping the NaNs would
    # be an alarm rate over a population the row's own summary refuses to describe, and one
    # computed with them would silently read `nan >= thr` as False -- "the alarm did not fire" for
    # a sample where the model emitted a non-number while claiming its inputs were fine.
    if not (np.isfinite(thr) and n_valid and bool(np.isfinite(yv).all())):
        return row
    above = valid & (y >= thr)
    runs = _runs(above)
    dt = _step(t)
    row["frac_above"] = float(above.sum()) / n_valid
    row["first_above_t_s"] = float(t[above][0]) if runs else _NAN
    row["n_intervals"] = len(runs)
    row["longest_interval_s"] = (
        max(float(t[b] - t[a] + dt) for a, b in runs) if runs else 0.0
    )
    return row


def labels_wide(
    shots: Iterable[int], *, labelmaker_root,
    thresholds: Mapping[str, tuple[float, str]] | None = None,
) -> pd.DataFrame:
    """One row per (shot, slug, label) of every shot that has a labels file.

    A shot with no labels file contributes no rows -- not a row of zeros. `join()` names those
    shots in `manifest["labels_missing"]`, which is where "we never labelled this shot" belongs;
    a zero in a table is indistinguishable from a measurement.

    `thresholds` maps `"<slug>/<label>"` to `(threshold, source)` and defaults to whatever the
    model cards record (`card_thresholds()`, all `card`); `join()` passes the cards' plus the
    `forecasts:` block's, so the risk labels an alarm is actually built on carry the alarm's own
    threshold -- and `thr_source` says, row by row, which of the two the number is.
    """
    paths = _paths(labelmaker_root)
    thr_by_key = _normalise_thresholds(thresholds)
    rows: list[dict] = []
    for shot in _shots(shots):
        path = paths.labels_file(shot)
        if not path.exists():
            continue
        for key in sorted(label_store.labelled(path)):
            slug, label = key.split("/", 1)
            t, y, valid, attrs = _series(path, slug, label)
            sha = str(attrs.get("artifact_sha256", ""))
            rows.append(
                _summary(shot, slug, label, t, y, valid,
                         thr_by_key.get(key, _NO_THRESHOLD), sha)
            )
    return _labels_wide_frame(rows)


# --------------------------------------------------------------------------- forecast events


def _run_id() -> str:
    return f"join-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}"


def label_forecast_events(
    shots: Iterable[int], *, labelmaker_root, rules: Sequence[ForecastRule],
    thresholds: Mapping[str, tuple[float, str]] | None = None,
    run_id: str | None = None,
) -> pd.DataFrame:
    """Every maximal run of `p >= thr` in a risk label, as a `forecast` event.

    `confidence` is the largest probability in the run and `horizon_s` the rule's horizon, so a
    reader can tell "0.9 that a tearing mode starts within 250 ms" from "0.9 that one is present",
    which is the whole point of keeping these out of the detector rows.

    `thresholds` is the join's `{key: (thr, source)}` map and overrides the rule's own level where
    it names the rule's label -- the same map `labels_wide` reads, so an event and its row can
    never be raised at two different numbers. Without it every rule stands at its own `thr`, which
    is `config`'s by definition. Both the number and its source go into the event's `attrs`.
    """
    paths = _paths(labelmaker_root)
    thresholds = dict(thresholds or {})
    run_id = run_id or _run_id()
    frames: list[pd.DataFrame] = []
    for shot in _shots(shots):
        path = paths.labels_file(shot)
        if not path.exists():
            continue
        have = label_store.labelled(path)
        events: list[events_schema.Event] = []
        for rule in rules:
            if rule.key not in have:
                continue
            t, y, valid, attrs = _series(path, rule.slug, rule.label)
            if not valid.any():
                continue
            thr, thr_source = thresholds.get(rule.key, (float(rule.thr), "config"))
            dt = _step(t)
            cov0 = float(t[valid][0])
            cov1 = float(t[valid][-1] + dt)
            sha = str(attrs.get("artifact_sha256", ""))
            # The DSM heads are queried one step past their horizon and the label records which
            # millisecond that was; a risk with no "as of when" beside it is not interpretable.
            queried = attrs.get("queried_at_ms")
            for a, b in _runs(valid & (y >= thr)):
                events.append(
                    events_schema.Event(
                        shot=shot,
                        source="label_forecast",
                        evidence_kind="forecast",
                        phenomenon=rule.phenomenon,
                        t0_s=float(t[a]),
                        t1_s=float(t[b] + dt),
                        confidence=float(y[a : b + 1].max()),
                        horizon_s=rule.horizon_s,
                        diag="",
                        channel=-1,
                        pass_name="",
                        attrs={
                            "slug": rule.slug,
                            "label": rule.label,
                            "thr": thr,
                            "thr_source": thr_source,
                            "n_samples": int(b - a + 1),
                            "artifact_sha256": sha,
                            **({"queried_at_ms": str(queried)} if queried is not None else {}),
                        },
                        t_cov0_s=cov0,
                        t_cov1_s=cov1,
                    )
                )
        if events:
            # labelmaker's own row builder, not a copy of it: `event_id` numbering
            # ("{shot}-{source}-{n:05d}", per source in t0 order), the column order and the dtype
            # cast are the schema's, and a second implementation of them here would be exactly the
            # redefinition the schema module exists to prevent.
            frames.append(events_schema._rows(shot, events, run_id=run_id))
    if not frames:
        return empty_events()
    return pd.concat(frames, ignore_index=True).astype(events_schema.DTYPES)


def events_union(
    shots: Iterable[int], *, labelmaker_root, forecasts: pd.DataFrame
) -> pd.DataFrame:
    """Every shot's `events/<shot>_events.parquet` plus the forecast rows, in one typed frame.

    Sorted by `(shot, t0_s)`, with `event_id` as the tie-break so the file is a function of its
    inputs and not of the order the directory happened to list.

    `label_forecast` rows already on disk are dropped: they are this join's to compute, out of the
    labels and this join's threshold map, and keeping a file's copy as well would put two rows
    raised at two different thresholds over the same stretch of the same shot into one table --
    with `event_id`s that collide, since the schema numbers per source.
    """
    paths = _paths(labelmaker_root)
    parts = []
    for shot in _shots(shots):
        disk = events_schema.read_events(paths.events_file(shot))
        parts.append(disk[disk["source"] != "label_forecast"] if not disk.empty else disk)
    parts.append(forecasts)
    kept = [p for p in parts if not p.empty]
    out = pd.concat(kept, ignore_index=True) if kept else empty_events()
    return (
        out.sort_values(["shot", "t0_s", "event_id"], kind="stable")
        .reset_index(drop=True)
        .astype(events_schema.DTYPES)
    )


# --------------------------------------------------------------------------------- the join


def _git_sha_of(events: pd.DataFrame) -> str | None:
    """The labelmaker commit the events on disk were written by, if they say.

    Only the rows labelmaker wrote: the forecast rows carry this checkout's sha, which is ideate's
    provenance, not labelmaker's, and mixing the two would misattribute both.
    """
    if events.empty:
        return None
    theirs = events.loc[events["source"] != "label_forecast", "git_sha"]
    seen = sorted({str(s) for s in theirs if str(s)})
    return ",".join(seen) if seen else None


def join(
    shots: Iterable[int],
    *,
    labelmaker_root,
    rules: Sequence[ForecastRule] | None = None,
    text_root=None,
    lexicon_path=None,
) -> JoinResult:
    """The whole join: `labels_wide`, the events union with its forecasts, and the text claims.

    `text_root=None` means "do not read the operator text at all" and gives an empty claims table;
    it is what a run on a machine without the text corpus asks for.
    """
    shots = _shots(shots)
    paths = _paths(labelmaker_root)
    rules_path = forecast_rules_path() if rules is None else None
    rules = forecast_rules() if rules is None else tuple(rules)
    from_card = card_thresholds()
    thresholds = _threshold_map(rules, from_card)

    wide = labels_wide(shots, labelmaker_root=labelmaker_root, thresholds=thresholds)
    forecasts = label_forecast_events(
        shots, labelmaker_root=labelmaker_root, rules=rules, thresholds=thresholds
    )
    events = events_union(shots, labelmaker_root=labelmaker_root, forecasts=forecasts)

    lexicon = None
    if text_root is None:
        claims = claims_mod.empty_claims()
    else:
        lexicon = claims_mod.load_lexicon(lexicon_path)
        claims = claims_mod.text_claims(shots, text_root=text_root, lexicon_path=lexicon.source)

    # Which detectors RAN, as opposed to what they found. A shot with no source file gets no
    # rows and is `unprocessed`; see `labels.event_sources`. Preserve intervals
    # and min_gap_s unchanged; missing legacy columns remain null, not inferred.
    sources = es.sources_union(shots, events_dir=paths.events)
    by_shot = {int(s): es.shot_summary(sources, s) for s in shots}
    observed = [s for s in shots if by_shot[s]["has_observed_products"]]

    missing_labels = [s for s in shots if not paths.labels_file(s).exists()]
    missing_events = [s for s in shots if not paths.events_file(s).exists()]
    manifest = {
        "labelmaker_root": str(Path(labelmaker_root)),
        "n_shots": len(shots),
        "n_shots_with_labels": len(shots) - len(missing_labels),
        "n_shots_with_events": len(shots) - len(missing_events),
        "n_labels_wide_rows": len(wide),
        "n_events": len(events),
        "n_forecast_events": len(forecasts),
        "n_text_claims": len(claims),
        "n_shots_missing_labels": len(missing_labels),
        "n_shots_missing_events": len(missing_events),
        "labels_missing": missing_labels,
        "events_missing": missing_events,
        "labelmaker_git_sha": _git_sha_of(events),
        "thresholds_from_card": sum(1 for _, src in thresholds.values() if src == "card"),
        "thresholds_from_config": sum(1 for _, src in thresholds.values() if src == "config"),
        # Per key, not just the two counts: a `frac_above` is only readable beside the number it
        # was computed at and the name of whoever chose that number.
        "thresholds": {key: [thr, src] for key, (thr, src) in sorted(thresholds.items())},
        # Which rule set raised the forecasts. `labels.yaml` is a file that will be edited, and a
        # table built from it has to say which version of it it was built from.
        "forecast_rules": {
            "path": str(rules_path) if rules_path else None,
            "sha256": rules_sha256(rules),
            "n_rules": len(rules),
        },
        "lexicon": str(lexicon.source) if lexicon else None,
        "n_event_source_rows": len(sources),
        "n_shots_with_source_rows": sum(1 for s in shots if by_shot[s]["n_sources"]),
        "n_shots_with_observed_products": len(observed),
        # The count that matters when a reply says "no events": these shots have no observed-event
        # product at all, so their emptiness is not evidence of a quiet shot.
        "n_shots_unprocessed": len(shots) - len(observed),
        "n_sources_ok": sum(v["n_sources_ok"] for v in by_shot.values()),
        "n_sources_skipped": sum(v["n_sources_skipped"] for v in by_shot.values()),
        "n_sources_error": sum(v["n_sources_error"] for v in by_shot.values()),
        # Per shot in the RETURNED manifest only. The file keeps the counts: at full corpus this
        # is 17,000 entries, and the module's rule is that the manifest file holds numbers a
        # person reads, not lists nobody does. `db/event_sources.parquet` is the per-shot answer.
        "sources_by_shot": by_shot,
    }
    return JoinResult(
        labels_wide=wide, events=events, claims=claims, sources=sources, manifest=manifest
    )


# ------------------------------------------------------------------------------- the tables


def _write_parquet(path: Path, df: pd.DataFrame) -> None:
    """Rename into place, so a reader sees the whole old table or the whole new one.

    `labelmaker.config.atomic_path` rather than a second copy of it: it is the tested
    implementation of this and it also removes the temporary sibling when the write raises, which
    is what keeps a full disk from leaving `.tmp` files across the DB.
    """
    with atomic_path(path) as tmp:
        df.to_parquet(tmp, index=False)


def write_tables(
    db_dir, labels_wide_df: pd.DataFrame, events_df: pd.DataFrame, claims_df: pd.DataFrame,
    manifest: Mapping, sources_df: pd.DataFrame | None = None,
    join_block: Mapping | None = None,
) -> dict:
    """Write the four tables and merge the join's counts into `db/manifest.json`.

    The manifest is *merged*, never replaced: a database's manifest is written by `build`, and a
    join that clobbered it would take the shot counts, the PCA and the IGNITE block with it.
    Returns the block that was written.

    `join_block` is written under `labels_join` beside it and holds what the join did to the
    database rather than what it read out of labelmaker -- today the refreshed `has_frame_codes`
    count. Two blocks rather than one because `labels` is a description of labelmaker's product
    and a reader comparing two databases' `labels` blocks must not see it move because an encode
    job finished in between.
    """
    db_dir = Path(db_dir)
    db_dir.mkdir(parents=True, exist_ok=True)
    _write_parquet(db_dir / "labels_wide.parquet", labels_wide_df)
    _write_parquet(db_dir / "events.parquet", events_df)
    _write_parquet(db_dir / "text_claims.parquet", claims_df)
    _write_parquet(
        db_dir / "event_sources.parquet",
        es.empty_sources() if sources_df is None else sources_df,
    )

    block = {k: manifest[k] for k in MANIFEST_KEYS if k in manifest}
    block["written_at"] = datetime.now(UTC).isoformat(timespec="seconds")
    path = db_dir / "manifest.json"
    whole = json.loads(path.read_text()) if path.exists() else {}
    whole["labels"] = block
    if join_block is not None:
        whole["labels_join"] = {
            **dict(join_block),
            "written_at": block["written_at"],
        }
    with atomic_path(path) as tmp:
        tmp.write_text(json.dumps(whole, indent=2, default=str))
    return block


__all__ = [
    "CLAIMS_COLUMNS",
    "CLAIMS_DTYPES",
    "FORECASTS_CONFIG",
    "LABELS_WIDE_COLUMNS",
    "LABELS_WIDE_DTYPES",
    "MANIFEST_KEYS",
    "SOURCES_COLUMNS",
    "THR_SOURCES",
    "ForecastRule",
    "JoinResult",
    "card_thresholds",
    "empty_events",
    "events_union",
    "forecast_rules",
    "forecast_rules_path",
    "join",
    "label_forecast_events",
    "labels_wide",
    "rules_sha256",
    "thresholds_from_card",
    "write_tables",
]
