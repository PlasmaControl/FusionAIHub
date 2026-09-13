"""Fusion, reranking, explanation, and the one function that composes them: `search`.

Read `search` first -- it is the whole query path in a dozen lines, and everything else in this
file is one of the steps it names. Channels live in `channels.py`; nothing here knows what a
channel does, only that `CHANNELS` maps a name to one.

`rrf_combine` and `dedup_near_duplicates` are copied, with attribution, from shotsearch (same
author, MIT): `embed/fusion.py` and `query/ranker.py`. Copies, not imports.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import functools
from collections import Counter
from typing import NamedTuple

import numpy as np

from .. import config
from ..flags.rules import evaluate_flags, load_rules
from ..schema import (
    Explanation,
    Flag,
    Labels,
    Outcome,
    QueryState,
    Range,
    ResultItem,
    ShotRecord,
)
from ..shotdb.store import ShotDB
from .channels import (
    CHANNELS,
    hard_filter,
    nan_excluded,
    positive_and_negatives,
    ref_row,
    target_values,
    tokenize,
)


def load_cfg() -> dict:
    """The `retrieval:` block of configs/ideate/retrieval.yaml.

    The YAML is the one source of every tuned number. This used to carry its own fallback for
    each knob "so a config that predates a knob still runs", and the fallbacks drifted: the YAML
    was tuned to `run_diversity_decay: 0.9` (measured, with its reasoning in a comment) while the
    fallback here and `rerank`'s default both still said 0.7. A missing knob is now a KeyError
    that names it, not a second, silent value. `weights` alone may be absent: a channel not named
    votes with weight 1.0, which is documented behaviour rather than a tuned number.
    """
    r = config.load_yaml("retrieval.yaml").get("retrieval") or {}
    out: dict = {"weights": dict(r.get("weights") or {})}
    for key, cast in (
        ("k0", int),
        ("dedup_threshold", float),
        ("run_diversity_decay", float),
        ("outcome_penalty", float),
    ):
        if key not in r:
            raise KeyError(f"configs/ideate/retrieval.yaml: retrieval.{key} is missing")
        out[key] = cast(r[key])
    return out


# ------------------------------------------------------------------------------------- fusion


def rrf_fuse(
    rankings: dict[str, list[tuple[str, float]]],
    weights: dict[str, float] | None = None,
    k0: int = 60,
) -> list[tuple[str, float]]:
    """Weighted reciprocal-rank fusion: score = sum_c w_c / (k0 + rank_c), rank 1-based.

    Ranks, not scores, because the channels are not on one scale and never will be: a MiniLM
    cosine on this corpus spreads over ~0.1, a BM25 score over ~20, and a whitened-PCA cosine
    over the whole [-1, 1]. Any weighted sum of the raw numbers is really the sharpest channel
    deciding alone. A channel that returned nothing simply casts no votes.
    """
    weights = weights or {}
    scores: dict[str, float] = {}
    for name, ranking in rankings.items():
        if not ranking:
            continue
        w = float(weights.get(name, 1.0))
        for rank, (seg_id, _) in enumerate(ranking, start=1):
            scores[seg_id] = scores.get(seg_id, 0.0) + w / (k0 + rank)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def channel_ranks(rankings: dict[str, list[tuple[str, float]]]) -> dict[str, dict[str, int]]:
    """segment id -> {channel: 1-based rank}, for the explanation."""
    out: dict[str, dict[str, int]] = {}
    for name, ranking in rankings.items():
        for rank, (seg_id, _) in enumerate(ranking, start=1):
            out.setdefault(seg_id, {})[name] = rank
    return out


# ------------------------------------------------------------------------------------ reranking


def _shot_of(seg_id: str) -> int:
    return int(seg_id.split(":")[0])


def rerank(
    items: list[tuple[str, float]],
    db: ShotDB,
    dedup_threshold: float | None = None,
    decay: float | None = None,
    outcome_penalty: float = 1.0,
) -> list[tuple[str, float]]:
    """Drop near-duplicates, then spread the results across run days.

    Two things make a top-10 useless on this data. One shot repeated as eight near-identical
    logbook entries is the first; the second is a top-10 that is one afternoon's worth of
    repeat shots, which answers "what else did they do that day" rather than "what else looks
    like this". So: a candidate whose `text_log` vector is within `dedup_threshold` cosine of an
    already-kept one is dropped (a shot with no text vector is always kept -- no vector is not
    evidence of duplication), and the m-th survivor from a run day is multiplied by decay^(m-1).

    `dedup_threshold` and `decay` default to configs/ideate/retrieval.yaml's values, read when they
    are not given -- not to numbers written here. A literal default (0.7) sat beside a YAML tuned
    to 0.9, so a direct call silently reranked with a decay the config had abandoned.
    `outcome_penalty` defaults to 1.0 because 1.0 means "off", which is not a tuned number.
    """
    if dedup_threshold is None or decay is None:
        cfg = load_cfg()
        dedup_threshold = cfg["dedup_threshold"] if dedup_threshold is None else dedup_threshold
        decay = cfg["run_diversity_decay"] if decay is None else decay
    emb = db.emb.get("text_log")
    shot_pos = {int(s): i for i, s in enumerate(db.shots.index)}
    kept: list[tuple[str, float]] = []
    kept_vecs: list[np.ndarray] = []
    for seg_id, score in items:
        vec = None
        if emb is not None and emb.size:
            p = shot_pos.get(_shot_of(seg_id))
            if p is not None and np.any(emb[p]):
                vec = emb[p]
        if vec is not None:
            if any(float(vec @ kv) > dedup_threshold for kv in kept_vecs):
                continue
            kept_vecs.append(vec)
        kept.append((seg_id, score))
    seen: Counter[str] = Counter()
    out: list[tuple[str, float]] = []
    for seg_id, score in kept:
        row = db.segments.loc[seg_id]
        run = row.get("run_id")
        if isinstance(run, str) and run:
            score *= decay ** seen[run]
            seen[run] += 1
        if outcome_penalty != 1.0 and (row.get("verdict") == "bad" or _missed_target(db, seg_id)):
            score *= outcome_penalty
        out.append((seg_id, score))
    return sorted(out, key=lambda kv: -kv[1])


def _missed_target(db: ShotDB, seg_id: str) -> bool:
    hit = db.shots.loc[_shot_of(seg_id), "ip_target_hit"]
    if hit is None or (isinstance(hit, float) and np.isnan(hit)):
        return False  # "not scored" is not a miss; the column is object dtype when any is None
    return not bool(hit)


# --------------------------------------------------------------------------- units for humans


_STAT_SUFFIXES = ("_on_frac", "_slope", "_mean", "_peak", "_min", "_std")
# Declared units where a k/M prefix is meaningful. The prefix is chosen from the magnitude of the
# numbers on the line, never from the unit alone: `units: A` covers both Ip (median 1.14e6 A) and
# an I-coil peak (median 14.6 A), so a blanket /1e6 printed a 14.6 A coil current as
# "1.46e-05 MA". Everything else -- T, m, G, "V (raw valve command)", and the honest "[?]" ones --
# prints in the unit configs/ideate/ declares, unscaled. An unverified unit never gets a prefix
# invented for it.
_PREFIXABLE = frozenset({"A", "V", "W", "J", "G"})


def split_stat(col: str) -> tuple[str, str]:
    for suffix in _STAT_SUFFIXES:
        if col.endswith(suffix):
            return col[: -len(suffix)], suffix[1:]
    return col, ""


@functools.lru_cache(maxsize=1)
def units() -> dict[str, str]:
    """Quantity -> the unit the registry declares, straight from configs/ideate/ -- never a unit
    guessed here. Several are honestly `[?]` (the line-integrated density, the neutron rate) and
    print that way; an unverified unit is not silently upgraded to a plausible one. Shared with
    `ideate show`, so the two cannot disagree about what a column is measured in.

    Shot-independent, so computed once per process: a signal's unit does not change with the
    campaign, and `include_not_installed=True` lists every actuator member whatever the shot.
    Keyed on the shot it was recomputed for every distinct result -- 628 ms over 10 shots on the
    105-shot database, most of a query's time spent re-reading two YAML files.
    """
    out = {
        s.name: s.units for s in config.expand_registry(0, include_not_installed=True) if s.units
    }
    for sysdef in config.actuator_systems(0).values():
        if sysdef.units:
            out[f"{sysdef.prefix}_total"] = sysdef.units
    return out


def _scaling(col: str, magnitude: float | None = None) -> tuple[float, str]:
    """(divide the stored value by this, call the result this), for numbers of this size."""
    base, stat = split_stat(col)
    if stat == "on_frac":
        return 1.0, "fraction of segment"
    unit = units().get(base, "")
    factor, shown = 1.0, unit
    if unit in _PREFIXABLE and magnitude is not None and np.isfinite(magnitude):
        # The prefix is chosen from the magnitude AS IT WILL PRINT, not the raw value: the number
        # is shown to three figures, so 999949 A is 1.00e6 A on the page, and picking "k" from the
        # raw 9.99949e5 printed it as "1e+03 kA" (measured on `display("ip_mean", 999949.0)`).
        rounded = float(f"{abs(magnitude):.3g}")
        if rounded >= 1e6:
            factor, shown = 1e6, f"M{unit}"
        elif rounded >= 1e3:
            factor, shown = 1e3, f"k{unit}"
    if stat == "slope":
        shown = f"{shown}/s" if shown else ""
    return factor, shown


def display(col: str, value: float, magnitude: float | None = None) -> str:
    """A stored value as a physicist would write it: `ip_mean` 1.21e6 -> "1.21 MA", and
    `irmp_C19_peak` 14.6 -> "14.6 A". Pass `magnitude` to fix the prefix from something other
    than this value -- the other number on the same line, so the two can be compared."""
    factor, unit = _scaling(col, value if magnitude is None else magnitude)
    return f"{value / factor:.3g} {unit}".strip()


def display_pair(col: str, value: float, other: float) -> str:
    """ "958 A vs 12 A" -- both numbers in one unit, chosen from the larger of them."""
    m = max(abs(value), abs(other))
    factor, unit = _scaling(col, m)
    return f"{value / factor:.3g} vs {other / factor:.3g} {unit}".strip()


def constraint_line(col: str, rng: Range, value: float) -> str:
    """ "ip_mean 1.21 MA within [1, 1.4] MA" -- the constraint restated in the units it was met in."""
    bounds = [abs(b) for b in (value, rng.lo, rng.hi) if b is not None]
    factor, unit = _scaling(col, max(bounds))
    got = f"{col} {value / factor:.3g} {unit}".rstrip()
    if rng.lo is not None and rng.hi is not None:
        return f"{got} within [{rng.lo / factor:.3g}, {rng.hi / factor:.3g}] {unit}".rstrip()
    if rng.hi is not None:
        return f"{got} <= {rng.hi / factor:.3g} {unit}".rstrip()
    if rng.lo is not None:
        return f"{got} >= {rng.lo / factor:.3g} {unit}".rstrip()
    return got


def feature_scales(
    db: ShotDB, mask: np.ndarray | None = None, max_impute: float = 0.5
) -> dict[str, tuple[float, float]]:
    """column -> (median, robust sd) over `mask`'s rows: the yardstick `explain` measures in.

    Deliberately not the PCA's own mean/std. Retrieval runs in the fitted space and must; the
    *explanation* cannot, because several of these columns have a fitted scale set by one
    outlier. `ne_line_mean` is standardized against 7.3e16 while ordinary shots read ~1e14, and
    `ip_slope` against 2.9e7 A/s fitted across the ramp-down rows -- so in fitted z every pair of
    flat tops agrees on them to 1e-4 and they would head every "most similar" list forever,
    saying nothing. A median and a MAD over the candidates say what "close" means among the shots
    actually on offer.

    Two exclusions, both measured, not listed by hand:

    * `impute_frac > max_impute`: a column the fit had to impute for most rows carries mostly the
      median, so agreement on it is a statement about the imputation, not the plasma.
    * MAD == 0: more than half the rows hold the identical value (every RMP coil peak sits on
      the same digitizer floor when the coils are off), so the column cannot separate them.

    `search` passes the whole segment population rather than the filtered candidates, so the
    sentence "8 sd from the reference" means the same thing whatever filter produced the result.
    """
    cols = db.pca.get("feature_cols") or []
    frac = db.pca.get("impute_frac") or []
    keep = [c for i, c in enumerate(cols) if i >= len(frac) or frac[i] <= max_impute]
    rows = db.segments.loc[mask] if mask is not None and mask.any() else db.segments
    out: dict[str, tuple[float, float]] = {}
    for c in keep:
        if c not in rows.columns:
            continue
        v = rows[c].to_numpy(dtype=float)
        v = v[np.isfinite(v)]
        if v.size < 2:
            continue
        med = float(np.median(v))
        mad = float(np.median(np.abs(v - med)))
        if mad > 0:
            out[c] = (med, 1.4826 * mad)  # 1.4826 * MAD is the normal-consistent sd
    return out


@functools.lru_cache(maxsize=1)
def _system_prefixes() -> tuple[str, ...]:
    """The actuator systems' column prefixes. A system's prefix does not depend on the shot (only
    its installed members do), so this is one registry read per process rather than one per
    distinct result shot."""
    return tuple(sysdef.prefix for sysdef in config.actuator_systems(0).values())


def _group_of(col: str) -> str:
    """Which physical quantity a column belongs to, for the explanation only.

    An actuator array is one fact: `irmp_C19_peak`, `irmp_IL150_peak` and `irmp_total_peak` are
    three of the eighteen RMP coil columns and reporting all three as the "three most similar
    features" tells a reader one thing three times, while the eighteen of them crowd out every
    other quantity. `ip_mean` and `ip_peak` collapse the same way.
    """
    base = split_stat(col)[0]
    for prefix in _system_prefixes():
        if base.startswith(f"{prefix}_") or base == prefix:
            return prefix
    return base


def _one_per_group(
    deltas: list[tuple[str, float, float]], n: int
) -> list[tuple[str, float, float]]:
    seen: set[str] = set()
    out = []
    for item in deltas:
        g = _group_of(item[0])
        if g in seen:
            continue
        seen.add(g)
        out.append(item)
        if len(out) == n:
            break
    return out


def _row_values(db: ShotDB, seg_id: str, cols) -> dict[str, float]:
    row = db.segments.loc[seg_id]
    out = {}
    for c in cols:
        v = row.get(c)
        if v is not None and np.isfinite(float(v)):
            out[c] = float(v)
    return out


def explanation_scales(q: QueryState, db: ShotDB) -> dict[str, tuple[float, float]]:
    """`feature_scales` over the whole population of the segment the query asked for.

    Not over the filtered candidates: a narrow filter can leave three rows whose MAD on some
    column is ~0, and every difference then reads as a thousand sigma. "How unusual is this
    value" is a question about the database, so the sentence "8 sd from the reference" means the
    same thing whatever filter produced the result. `search` and the CLI both go through here so
    they cannot disagree about which columns are comparable.
    """
    return feature_scales(db, (db.segments["segment"] == q.segment).to_numpy(dtype=bool))


def query_values(
    q: QueryState, db: ShotDB, scales: dict[str, tuple[float, float]] | None = None
) -> dict[str, float]:
    """What the query itself says each feature should be: the reference segment's own row, or the
    constraint midpoints and actuator settings. The CLI prints these beside each result's values,
    because "29 sd apart" is not a number a reader can check without the pair it came from."""
    scales = explanation_scales(q, db) if scales is None else scales
    ref = ref_row(q, db)
    if ref is not None:
        return _row_values(db, ref, scales)
    return {c: v for c, v in target_values(q, db).items() if c in scales}


def explain(
    q: QueryState,
    db: ShotDB,
    seg_id: str,
    ranks: dict[str, int],
    scales: dict[str, tuple[float, float]] | None = None,
    n_features: int = 3,
    q_vals: dict[str, float] | None = None,
) -> Explanation:
    """Why this row came back: the constraints it met, the features it matched and missed, and
    where each channel put it.

    `top_similar`/`top_different` are (column, |z_query - z_shot|, the shot's own value), z as
    `feature_scales` defines it. Only features *both* sides actually recorded are ranked -- a
    median-imputed agreement is not a match -- and a feature both sides read as exactly zero is
    skipped, since two idle gyrotrons agreeing says nothing about either shot.

    `scales` and `q_vals` describe the query, not this row; `search` computes them once and
    passes them in rather than paying for them again on every result.
    """
    shot = _shot_of(seg_id)
    row = db.segments.loc[seg_id]
    matched = [
        constraint_line(col, rng, float(row[col]))
        for col, rng in q.constraints.items()
        if col in db.segments.columns and np.isfinite(float(row.get(col, np.nan)))
    ]
    scales = explanation_scales(q, db) if scales is None else scales
    q_vals = query_values(q, db, scales) if q_vals is None else q_vals
    s_vals = _row_values(db, seg_id, list(q_vals))
    deltas = [
        (c, abs(q_vals[c] - s_vals[c]) / scales[c][1], s_vals[c])
        for c in q_vals
        # A column the query constrained is already reported above, in physical units and with
        # the bounds it met. Repeating it here as "differs by 1.4 sd from the middle of the range
        # you asked for" reads as a contradiction of the line right above it.
        if c in s_vals and c not in q.constraints and not (q_vals[c] == 0.0 and s_vals[c] == 0.0)
    ]
    deltas.sort(key=lambda t: t[1])
    # Differences first, and a quantity that lands there is not also reported as a match. The
    # RMP coils are the case that forces the order: when the reference ran the coils off and the
    # candidate ran them on, one coil column agrees to 0.01 sigma and another disagrees by 89 --
    # the disagreement is the fact about the shot, and it has to win the slot.
    different = _one_per_group(list(reversed(deltas)), n_features)
    used = {_group_of(c) for c, _, _ in different}
    similar = _one_per_group([d for d in deltas if _group_of(d[0]) not in used], n_features)
    return Explanation(
        matched_constraints=matched,
        top_similar=similar,
        top_different=different,
        channel_ranks=dict(ranks),
        text_highlight=_highlight(q, db, shot),
    )


def _highlight(q: QueryState, db: ShotDB, shot: int, max_chars: int = 220) -> str | None:
    """The one quotable logbook entry with the most query terms in it, as a prefix of itself.

    Taken from `HumanTier.log_entries` -- what `text.parse_log_entries` produced -- and never
    from the raw `log_text`, which still carries HTML and runs several operators' notes together.
    Which entries qualify, and how their text is shown, is `describe.quotable`'s decision, the
    same one the Operator line in the description makes: no PCS settings dumps or RF status
    tables, entities decoded for display. Scored on the query's POSITIVE text only -- tokenising
    "ELM suppression but no QH-mode shots" whole put 'but', 'no', 'shots' and 'qh-mode' in the
    term set, so the entry about the thing the user excluded won the highlight.
    """
    from .describe import (
        quotable,  # describe imports this module's display; see _enrich
    )

    positive, _ = positive_and_negatives(q)
    terms = set(tokenize(positive)) if positive else set()
    if not terms:
        return None
    try:
        rec = db.get(shot)
    except Exception:  # noqa: BLE001 — a record that will not parse must not lose the result
        return None
    best: tuple[str, str] | None = None
    best_n = 0
    for entry in rec.human.log_entries:
        text = quotable(entry)
        if text is None:
            continue
        n = len(terms & set(tokenize(text)))
        if n > best_n:
            best, best_n = (entry.role, text), n
    if best is None:
        return None
    role, body = best
    if len(body) > max_chars:
        body = body[:max_chars].rstrip() + " ..."
    return f"[{role}] {body}"


# ---------------------------------------------------------------------------------- the search


def _enrich(
    rec: ShotRecord, segment: str, db: ShotDB, seg_id: str, rule_cfg: dict
) -> tuple[str, list]:
    """The result's description and operating-limit flags. Neither is a term in the score.

    The description is `describe.describe`, the one template `ideate show` and `ideate export`
    also print -- there is no second renderer here, so a query result and `show` cannot disagree
    about a shot's numbers. `rule_cfg` is `load_rules()`'s output, loaded once by `search`.
    """
    # Imported here, not at module top: describe.py takes its units from this module's `display`,
    # and the two importing each other at load time would be a cycle.
    from .describe import describe

    row = db.segments.loc[seg_id]
    values = {c: (float(row[c]) if np.isfinite(float(row[c])) else None) for c in _numeric(row)}
    category = {"regime": str(rec.labels.regime), "campaign": str(rec.campaign)}
    return describe(rec, segment, db=db), list(evaluate_flags(values, category, rule_cfg))


def _numeric(row) -> list[str]:
    return [c for c in row.index if isinstance(row[c], (int, float, np.floating)) and c != "shot"]


class SearchResult(NamedTuple):
    """What one `search` produced.

    `rankings` is each channel's own candidate list, before fusion -- the CLI's "channels
    scalar_knn 105, text_knn 105, bm25 0" line comes from here, so it is read off the pass that
    produced the results rather than from running every channel a second time (a `--text` query
    embedded its text with MiniLM twice, ~1.9 s each, measured).

    `proposal_flags` are the operating-limit rules run over the query's OWN `actuators` -- the
    user's proposal, not any result's values. "Can DIII-D do this at all" is a different question
    from "what has it done like this", and it is answered even when no shot resembles the
    proposal. Empty when the query proposed nothing.
    """

    items: list[ResultItem]
    rankings: dict[str, list[tuple[str, float]]]
    proposal_flags: list[Flag]


def search(q: QueryState, db: ShotDB, cfg: dict | None = None) -> SearchResult:
    """Run every registered channel, fuse, rerank, and explain the top n."""
    cfg = cfg or load_cfg()
    weights = q.channel_weights or cfg["weights"]
    rule_cfg = load_rules()
    proposal = list(evaluate_flags(dict(q.actuators), None, rule_cfg)) if q.actuators else []
    rankings = {name: fn(q, db) for name, fn in CHANNELS.items()}
    fused = rrf_fuse(rankings, weights, cfg["k0"])
    ref = ref_row(q, db)
    fused = [(sid, s) for sid, s in fused if sid != ref]  # a shot is not its own neighbour
    penalty = cfg["outcome_penalty"] if q.prefer_outcome == "success" else 1.0
    ranked = rerank(fused, db, cfg["dedup_threshold"], cfg["run_diversity_decay"], penalty)
    top = ranked[: q.n]
    if not top:
        return SearchResult([], rankings, proposal)
    # Everything that describes the QUERY rather than one result is computed here, once: the
    # explanation yardstick, the query's own feature values, and (above) the flag rules.
    ranks, scales = channel_ranks(rankings), explanation_scales(q, db)
    q_vals = query_values(q, db, scales)
    items = [
        _item(sid, score, q, db, ranks.get(sid, {}), scales, q_vals, rule_cfg) for sid, score in top
    ]
    return SearchResult(items, rankings, proposal)


def _blurb(db: ShotDB, shot: int) -> str | None:
    """The stored offline blurb, or None -- for a database built before there were any, and for a
    row an `add()` onto such a database left empty (a float NaN, not a string)."""
    if "blurb" not in db.shots.columns:
        return None
    value = db.shots.loc[shot, "blurb"]
    return value if isinstance(value, str) and value else None


def _item(
    seg_id: str,
    score: float,
    q: QueryState,
    db: ShotDB,
    ranks: dict[str, int],
    scales: dict[str, tuple[float, float]] | None = None,
    q_vals: dict[str, float] | None = None,
    rule_cfg: dict | None = None,
) -> ResultItem:
    shot = _shot_of(seg_id)
    rec = db.get(shot)
    description, flags = _enrich(rec, q.segment, db, seg_id, rule_cfg or load_rules())
    return ResultItem(
        id=seg_id,
        shot=shot,
        segment=q.segment,
        score=score,
        description=description,
        explanation=explain(q, db, seg_id, ranks, scales, q_vals=q_vals),
        blurb=_blurb(db, shot),
        flags=flags,
        labels=rec.labels or Labels(),
        outcome=rec.outcome or Outcome(),
        run_id=rec.human.run_id,
    )


def search_report(q: QueryState, db: ShotDB) -> dict:
    """What the search would filter away, without running it: the candidate count and the rows
    each constraint dropped for having no value. The CLI prints this above the results."""
    mask = hard_filter(q, db)
    return {
        "candidates": int(mask.sum()),
        "segment_rows": int((db.segments["segment"] == q.segment).sum()),
        "nan_excluded": nan_excluded(q, db),
        "caveats": db.label_filter_caveats(q.segment, q.avoid_labels),
    }
