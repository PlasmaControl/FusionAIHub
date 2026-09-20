"""Turn the raw store + text DB into the shot database.

One ShotRecord per shot, flattened into two Parquet tables, plus per-segment scalar embeddings
(median-impute -> z-score -> whitened PCA -> L2), MiniLM text embeddings, and a manifest that
records what was built, from what, and what was skipped. `build()` writes to a sibling `.tmp`
directory and swaps it in; `add()` upserts rows and projects new segments with the frozen PCA.

Two conventions this module is careful about, both inherited from features.py:

* A missing scalar is NaN, never 0.0. features.py distinguishes "not recorded" from "recorded and
  reading zero" (an idle gyrotron/beam/valve) throughout, and flattening into segments.parquet
  must not collapse the two -- so every feature column is a float column and every `None` from
  Segment.raw/derived becomes NaN, including in a column no shot in the build recorded at all.
* An IGNITE embedding is optional. Nothing here imports the encoder eagerly; a build without it
  is the ordinary case and produces a scalar-only database that says so in its manifest.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import csv
import datetime as dt
import errno
import functools
import hashlib
import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .. import config
from ..schema import Labels, Outcome, Provenance, Segment, ShotRecord, Status
from . import features, legacy_raw, text
from .reader import ShotFailed, Signal, SignalReader, Unavailable

_log = logging.getLogger(__name__)

STATUSES: tuple[Status, ...] = ("present", "unavailable", "pending", "not_installed")
# Columns of segments.parquet that identify a row rather than describe the plasma in it.
ID_COLS = ("shot", "segment", "t0_ms", "t1_ms")
# Columns of shots.parquet, in order: the one place they are listed. `_shot_row` builds each row
# with exactly these keys (records_to_tables checks) and `_empty_shots` uses them for a build
# with no records, so the two cannot drift apart the way a hand-copied list did.
SHOT_COLS = (
    "shot", "shot_date", "campaign", "run_id", "mpid", "mp_title", "verdict", "regime",
    "regime_source", "operational", "ip_target_hit", "end_reason", "n_segments",
    "coverage_fraction", "text_mp", "text_log", "record_json", "blurb", "blurb_source",
    "blurb_model", "blurb_prompt_version",
    "reader", "groups_filled", "has_bes", "has_co2", "has_ece", "has_filterscopes", "has_mhr",
    "has_mirnov", "feature_resolvers", "has_frame_codes",
)  # fmt: skip
# The raw groups whose presence gets its own `has_<group>` column in shots.parquet: the corpus's
# five 500 kHz groups plus filterscopes. They are the ones a phenomenon query filters on before
# anything else -- there is no point ranking a shot for an ELM if nothing was watching D-alpha --
# and `groups_filled` (how many groups the reader saw at all) is the same fact at one number.
# A reader that has no such groups (the d3d_fusion_data layout) simply reports them all False.
FAST_GROUPS = ("bes", "co2", "ece", "filterscopes", "mhr", "mirnov")
# The two text embeddings, one per shots.parquet text column: emb_<key>.npy is the MiniLM row
# for shots_df[key], and every place that builds, concatenates or reorders them loops over this.
TEXT_KEYS = ("text_mp", "text_log")
#: The phases `build` times, in the order it runs them, and the keys of the manifest's
#: `phase_seconds`. A build used to report one number -- total elapsed -- which is why three
#: SLURM rebuilds that ran at 65-68 % CPU on 3-6 cores could not be diagnosed: nobody could say
#: which phase was leaving the cores idle. Wall seconds, not CPU seconds, because the question is
#: about idling.
#:
#: The IGNITE encode is deliberately NOT one of these: it already times itself into
#: `manifest["ignite"]["elapsed_s"]`, and counting it again inside `write_tables` would make that
#: phase the answer to every question on an encoding build.
PHASES = (
    "read_records",  # the text subset, then one ShotRecord per shot from the raw layer
    "segment",  # records -> the shots/segments tables and the waveform-shape matrix
    "scalar_embedding",  # the feature matrix, the PCA fit and the projection
    "text_embedding",  # MiniLM over the two text columns
    "write_tables",  # writing the staging directory `_publish` will move
    "publish",  # moving it into place and pruning what this build did not write
)


@contextmanager
def _phase(seconds: dict[str, float], name: str):
    """Add this block's wall time to `seconds[name]`, whether or not the block raises."""
    t0 = time.perf_counter()
    try:
        yield
    finally:
        seconds[name] += time.perf_counter() - t0


@dataclass
class BuildReport:
    shots: list[int]
    n_segments: int = 0
    failed: dict[int, str] = field(default_factory=dict)
    db_dir: Path | None = None
    encoded: bool = False
    elapsed_s: float = 0.0


def load_build_cfg() -> dict:
    r = config.load_yaml("retrieval.yaml")
    return {
        "segments": r["segments"],
        "scalar": r["scalar_embedding"],
        "labels": config.load_yaml("labels.yaml"),
        "waveforms": config.load_yaml("signals.yaml").get("waveforms", []),
    }


def _git_sha() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                check=False,
                cwd=Path(__file__).resolve().parents[3],
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
            or "unknown"
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _config_sha() -> str:
    h = hashlib.sha1()
    for name in ("signals.yaml", "actuators.yaml", "retrieval.yaml", "labels.yaml"):
        p = config.CONFIG_DIR / name
        if p.exists():
            h.update(p.read_bytes())
    return h.hexdigest()[:8]


@functools.lru_cache(maxsize=1)
def _qh_shots(csv_path: str) -> frozenset[int]:
    """Shots the QH_Database labels as QH-mode, keyed by path so tests can point elsewhere.

    The real file (big_d3d_data/0labeled_data/QH_Database.csv) starts with a UTF-8 BOM, hence
    utf-8-sig; without it the first column comes back named "﻿shot" and every lookup misses.
    134 rows, 107 distinct shots, all between 173694 and 175544 -- so this is a narrow, high-
    precision source, not a general regime labeller.
    """
    p = Path(csv_path)
    if not p.exists():
        return frozenset()
    with open(p, encoding="utf-8-sig") as fh:
        return frozenset(
            int(float(r["shot"])) for r in csv.DictReader(fh) if r.get("shot", "").strip()
        )


def _shot_date(run_id: str | None) -> dt.date | None:
    try:
        return dt.date(int(run_id[:4]), int(run_id[4:6]), int(run_id[6:8]))
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------ labels + outcome


def derive_labels(human, flat: Segment | None, cfg: dict, qh: frozenset[int], shot: int) -> Labels:
    """Regime + operational labels: QH database > explicit text > geometry heuristic > unknown."""
    lc = cfg["labels"]
    txt = " ".join(
        filter(
            None,
            [
                human.mp_title,
                human.run_title,
                human.mp_purpose,
                *(e.text for e in human.log_entries),
            ],
        )
    ).lower()
    regime, source = "unknown", "none"
    if shot in qh:
        regime, source = "QH", "database"
    else:
        # Order in labels.yaml is priority (QH, neg_tri, L, H) and it has to be: "qh-mode"
        # contains "h-mode", so checking H first would relabel every QH shot.
        for reg, kws in lc["regime_text"].items():
            if any(k in txt for k in kws):
                regime, source = reg, "text"
                break
    if regime == "unknown" and flat is not None:
        tt, tb = flat.derived.get("tritop_mean"), flat.derived.get("tribot_mean")
        if tt is not None and tb is not None and (tt + tb) / 2 < lc["neg_tri_mean_delta"]:
            regime, source = "neg_tri", "heuristic"
    ops = {op for op, kws in lc["operational_text"].items() if any(k in txt for k in kws)}
    return Labels(regime=regime, regime_source=source, operational=ops)


def derive_outcome(
    ip_out: dict,
    nbi_target: Signal | None,
    pnbi_total: Signal | None,
    flat: Segment | None,
    human,
    cfg: dict,
) -> Outcome:
    out = Outcome(
        **{
            k: ip_out.get(k)
            for k in (
                "ip_target_err",
                "ip_target_hit",
                "flat_top_ms",
                "ended_early",
                "fast_quench",
                "end_reason",
            )
        }
    )
    # quality_comment is null in all 53,179 logbook records and chief_operator_status is a slice of
    # log_text, so in practice the CHIEF_OPERATOR entries are the whole signal here. Both nulls are
    # passed anyway because fault_strings ignores None and a future export may populate them.
    out.fault_strings = text.fault_strings(
        human.chief_operator_status,
        human.quality_comment,
        *(e.text for e in human.log_entries if e.role == "CHIEF_OPERATOR"),
    )
    if flat is not None and nbi_target is not None and pnbi_total is not None:
        t_tgt, tgt = features.window(nbi_target, flat.t0_ms, flat.t1_ms)
        t_act, act = features.window(pnbi_total, flat.t0_ms, flat.t1_ms)
        # The time axes are passed through, not None: stat_peak drops unrecorded samples by
        # indexing *both* arrays, so a None here is a TypeError, not a shortcut.
        tgt_peak = features.stat_peak(t_tgt, tgt, True) if tgt.size else None
        act_peak = features.stat_peak(t_act, act, True) if act.size else None
        tgt_mw, unit = features.infer_nbi_target_mw(tgt_peak, act_peak)
        if tgt_mw and act_peak:
            err = act_peak * 1e-6 / tgt_mw - 1.0
            out.nbi_target_err = err
            out.nbi_target_hit = abs(err) <= cfg["segments"]["target_tolerance"]
            out.nbi_target_unit = unit
    return out


def _provenance(specs, signals: dict[str, Signal | None]) -> dict[str, Provenance]:
    prov: dict[str, Provenance] = {}
    for s in specs:
        if not s.provenance:
            continue
        p = dict(s.provenance)
        # assumed_for_staged is a registry claim about a *location*, not about the quantity: the
        # staged files do not record which EFIT run produced them, so a value read from one is
        # tagged assumed. A value we fetched ourselves came from the declared tree and is not.
        assumed = bool(p.pop("assumed_for_staged", False))
        sig = signals.get(s.name)
        fields = {k: v for k, v in p.items() if k in Provenance.model_fields}
        for k in ("tool", "version", "tree", "run_id"):
            if fields.get(k) is not None:
                fields[k] = str(fields[k])
        # A labeler feature resolved from the archive store carries no EFIT run id either
        # (its `archive_file` attr names the store, not the run), so it makes exactly the claim
        # `assumed_for_staged` describes: the value is real, its provenance is inferred.
        prov[s.name] = Provenance(
            **fields,
            assumed=assumed and sig is not None and sig.source in ("staged", "labelmaker"),
        )
    return prov


def make_reader(kind: str, paths: config.Paths) -> SignalReader:
    """The reader named by a picklable string.

    A worker process is handed `reader_kind` and `paths`, never a constructed reader: an HDF5
    handle must not be forked, and a reader is cheap to build. `corpus_signals` is imported here
    rather than at module scope so that a legacy build does not import labeler.
    """
    if kind == "legacy":
        return legacy_raw.LegacyReader(paths)
    if kind == "corpus":
        from .corpus_signals import CorpusSignalReader

        return CorpusSignalReader(paths)
    raise ValueError(f"unknown reader {kind!r}; expected 'legacy' or 'corpus'")


def frame_codes_dirs(paths: config.Paths) -> tuple[Path, ...]:
    """Where IGNITE frame codes live, in preference order, because three commands ask.

    Production's own cache comes first when the pinned generation names one
    (`model.frame_codes_cache` in configs/shot_design/ignite_modalities.yaml): it is
    thousands of shots the v4 dynamics checkpoint was actually trained on, read-only,
    and cheaper than a re-encode. `<data_root>/frame_codes` is where
    `design.encode_frame_codes` writes shots nobody encoded yet, and
    `<models_dir>/<local_name>/frame_codes` is the pinned bundle's own directory (empty
    for v4, which ships no frame codes -- see `shotdb/ignite.py:bundle_dir`). `build`
    (the `has_frame_codes` column), `corpus select` (its preference for shots already
    encoded) and `design.program_reference._cache_path` all read this one function, so
    they cannot disagree about the same shot the way `build` and `corpus select` used to
    when each kept its own list.

    Left out entirely, never an empty string, when the pinned generation has no
    `frame_codes_cache` key.
    """
    from . import ignite

    production = ignite.model_cfg().get("frame_codes_cache")
    dirs = [Path(production)] if production else []
    dirs.append(Path(paths.data_root) / "frame_codes")
    dirs.append(ignite.bundle_dir(paths) / "frame_codes")
    return tuple(dirs)


def frame_codes_path(shot: int, paths: config.Paths) -> Path | None:
    """This shot's IGNITE frame codes, if either location has them, else None. A build records
    whether the shot has codes at all, not which of the two producers wrote them."""
    for d in frame_codes_dirs(paths):
        p = d / f"{int(shot)}.pt"
        if p.exists():
            return p
    return None


def _bundle_pulse_length_s(shot: int, paths: config.Paths) -> float | None:
    """`PULSE-LENGTH` (seconds) from this shot's text bundle, or None.

    The same file (`per_shot_txt_dir/shot_<N>.txt`) `text.mp_text` reads for the
    mini-proposal title -- read again here because that function returns the parsed
    title, not the raw bundle `text.shot_table_row` needs. None for a missing bundle, a
    bundle with no shot table row, or a row with no PULSE-LENGTH -- absent evidence,
    never a proxy of 0. Parse matches `select._num`'s (space-stripped float, else None).
    """
    p = paths.per_shot_txt_dir / f"shot_{int(shot)}.txt"
    if not p.exists():
        return None
    bundle = p.read_text(encoding="utf-8", errors="replace")
    raw = text.shot_table_row(bundle).get("PULSE-LENGTH")
    if raw is None:
        return None
    try:
        return float(str(raw).replace(" ", ""))
    except ValueError:
        return None


def _raw_groups(reader: SignalReader, shot: int) -> list[str]:
    """Every raw group this reader sees for the shot, or [] when it cannot say.

    One extra call per shot. On the corpus it is one open of a file that is about to be opened
    anyway; on the d3d_fusion_data layout it decodes each group's header, which is the price of
    the same coverage columns being answerable from either reader rather than only from one.
    """
    try:
        return list(reader.groups(shot))
    except (ShotFailed, Unavailable, OSError, KeyError, RuntimeError):
        return []


# ------------------------------------------------------------------------------------ one record


def build_record(
    shot: int, paths: config.Paths, cfg: dict, reader: SignalReader | None = None
) -> tuple[ShotRecord, dict[str, np.ndarray]]:
    """One shot: every registry signal, its segments, their scalars, the human tier, and labels.

    Returns the record and, separately, the per-segment waveform-shape vectors -- they are a
    fixed-width float matrix that belongs next to the scalar features in the embedding, not in a
    record a human reads.

    `reader` is where the raw signals come from, and it is an argument rather than an import so
    that building from another raw layout (the FAITH corpus, via `corpus.CorpusReader`) is a
    caller's decision. It defaults to the d3d_fusion_data layout this database was first built
    from; nothing below knows which layout answered.
    """
    reader = reader or legacy_raw.LegacyReader(paths)
    specs = config.expand_registry(shot, include_not_installed=True)
    systems = config.actuator_systems(shot)
    # One pass over the raw files for both: `present` is the read that succeeded, so the two
    # dicts cannot disagree and nothing is read twice (legacy_raw.read_shot has the measurement).
    signals, coverage = reader.read_shot(shot, specs)
    totals = features.system_totals(signals, systems)
    ip = signals.get("ip")
    # Frontier (task F2b) has no Ip trace anywhere: `ip is None` for every shot there,
    # not just the ones with no raw file at all. A PULSE-LENGTH proxy gives those shots
    # real segments instead of the empty "no_ip_signal" record `_buildable` exists to
    # keep out of a bulk build -- honestly marked as a proxy in `coverage_reasons["ip"]`
    # below, never as if it were a measured flat-top.
    proxy_ip_reason: str | None = None
    if ip is not None:
        segs = features.find_segments(ip, cfg["segments"])
    else:
        pulse_length_s = _bundle_pulse_length_s(shot, paths)
        segs = features.proxy_segments(pulse_length_s, cfg["segments"])
        if segs:
            proxy_ip_reason = (
                "segments from PULSE-LENGTH proxy (ramp-up 1.0 s, ramp-down 0.27 s); "
                "no Ip trace on this cluster"
            )
    installed = [s for s in specs if s.installed]
    n_points = cfg["scalar"]["waveform_points"]
    shapes: dict[str, np.ndarray] = {}
    for seg in segs:
        raw, derived = features.segment_scalars(signals, seg, installed)
        for sysdef in systems.values():
            tot = totals.get(f"{sysdef.prefix}_total")
            for stat in sysdef.stats:
                key = f"{sysdef.prefix}_total_{stat}"
                if tot is None:
                    raw[key] = None
                    continue
                t, y = features.window(tot, seg.t0_ms, seg.t1_ms)
                # Recorded samples, not samples: system_totals leaves a sample NaN when no member
                # recorded there, and three of those are not three measurements.
                raw[key] = (
                    features.STATS[stat](t, y, True) if int(np.isfinite(y).sum()) >= 3 else None
                )
        seg.raw, seg.derived = raw, derived
        shapes[seg.name] = np.concatenate(
            [
                features.waveform_shape(signals.get(w) or totals.get(w), seg, n_points)
                for w in cfg["waveforms"]
            ]
        )
    human = text.human_tier(shot, paths)
    coverage["mp_text"] = "present" if human.mp_purpose else "unavailable"
    coverage["logbook"] = "present" if human.log_entries else "unavailable"
    flat = next((s for s in segs if s.name == "flat_top"), None)
    ip_out = (
        features.ip_outcome(ip, signals.get("ip_target"), segs, cfg["segments"])
        if ip is not None
        else {"end_reason": "no_ip_signal"}
    )
    outcome = derive_outcome(
        ip_out, signals.get("nbi_target_total"), totals.get("pnbi_total"), flat, human, cfg
    )
    labels = derive_labels(human, flat, cfg, _qh_shots(str(paths.qh_database_csv)), shot)
    if outcome.fast_quench:
        labels.operational.add("fast_current_quench")
    if outcome.ended_early:
        labels.operational.add("early_termination")
    # Three-valued on purpose: only a shot we positively know neither quenched nor ended early is
    # labelled clean. `not fast_quench` would label an unknown one clean too.
    if outcome.fast_quench is False and outcome.ended_early is False:
        labels.operational |= {"disruption_free", "controlled_rampdown"}
    if any("dud" in f for f in outcome.fault_strings):
        labels.operational.add("dud")
    if any("locked" in f for f in outcome.fault_strings):
        labels.operational.add("locked_mode")
    # Merge, never overwrite: the reader's own reasons (e.g. a corpus address that
    # missed) are about OTHER signals and must survive; the proxy reason is `ip`'s
    # alone.
    reasons = dict(getattr(reader, "reasons", None) or {})
    if proxy_ip_reason:
        reasons["ip"] = proxy_ip_reason
    return (
        ShotRecord(
            shot=shot,
            shot_date=_shot_date(human.run_id),
            raw_groups=_raw_groups(reader, shot),
            reader=str(getattr(reader, "kind", "legacy")),
            has_frame_codes=frame_codes_path(shot, paths) is not None,
            # Only the signals whose reader has more than one source of its own carry a resolver;
            # on the legacy layout `raw_sources` is already the whole answer and this is empty.
            feature_resolvers={
                n: s.resolver for n, s in signals.items() if s is not None and s.resolver
            },
            # A shot-number band from actuators.yaml, NOT a date range: the bands disagree with
            # the run folders (1,102 logbook records in the 2019_2021 band have 2022 run folders,
            # 53 have 2018 ones; shot 189013 sits in 2019_2021 with run id 20220427). Use
            # shot_date above for anything that needs an actual date.
            campaign=config.campaign_for_shot(shot),
            segments=segs,
            derived_provenance=_provenance(specs, signals),
            raw_sources={n: s.source for n, s in signals.items() if s is not None},
            human=human,
            labels=labels,
            outcome=outcome,
            coverage=coverage,
            # Optional and read defensively: only a reader that can narrow a miss below "the
            # group is not there" has anything to say, and `SignalReader` does not require it.
            coverage_reasons=reasons,
            built_at=dt.datetime.now(dt.UTC),
            builder_sha=f"{_git_sha()}+{_config_sha()}",
        ),
        shapes,
    )


def _safe_build(args: tuple) -> tuple[int, ShotRecord | None, dict | None, str | None]:
    shot, paths, cfg, reader_kind = args
    try:
        rec, shapes = build_record(shot, paths, cfg, make_reader(reader_kind, paths))
        return shot, rec, shapes, None
    except Exception as e:  # noqa: BLE001 — one broken file must not abort the build; the report lists it
        return shot, None, None, f"{type(e).__name__}: {e}"


def _build_many(
    shots: list[int],
    paths: config.Paths,
    cfg: dict,
    workers: int,
    report: BuildReport,
    reader_kind: str = "legacy",
):
    """Records for `shots`, in order, with failures diverted into `report.failed`.

    workers <= 1 runs in-process rather than forking a pool of one: it keeps a monkeypatched
    build_record (and any other test-time patch) visible, and a pool is pure overhead there.

    `reader_kind` is a string and not a reader because this is what crosses a process boundary:
    each worker calls `make_reader` for itself, so no HDF5 handle is ever inherited by a fork.
    """
    args = [(s, paths, cfg, reader_kind) for s in shots]
    if workers <= 1:
        results = (_safe_build(a) for a in args)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        results = ex.map(_safe_build, args)
    records, shapes = [], {}
    try:
        for shot, rec, shp, err in results:
            if rec is None:
                report.failed[shot] = err or "unknown"
            else:
                records.append(rec)
                shapes[shot] = shp
                report.shots.append(shot)
    finally:
        if workers > 1:
            ex.shutdown()
    return records, shapes


# ------------------------------------------------------------------------------------- flattening


def records_to_tables(
    records: list[ShotRecord],
    shapes: dict[int, dict[str, np.ndarray]],
    blurb_client=None,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """(shots table, segments table, waveform-shape matrix) -- the three things that get written.

    Every scalar column of the segments table is forced to float64, so a `None` becomes NaN and a
    column that no shot in this build recorded stays a float column of NaN instead of degrading
    to object dtype and silently dropping out of the feature matrix (which selects numeric columns).
    """
    shot_rows, seg_rows, shape_rows = [], [], []
    for rec in records:
        shot_rows.append(_shot_row(rec, blurb_client))
        for seg in rec.segments:
            seg_rows.append(
                {
                    "id": f"{rec.shot}:{seg.name}",
                    "shot": rec.shot,
                    "segment": seg.name,
                    "t0_ms": seg.t0_ms,
                    "t1_ms": seg.t1_ms,
                    **seg.raw,
                    **seg.derived,
                }
            )
            shape_rows.append(shapes[rec.shot][seg.name])
    shots_df = (
        pd.DataFrame(shot_rows).set_index("shot", drop=False) if shot_rows else _empty_shots()
    )
    # Nullable integers survive an add to a legacy table whose retained rows have no version.
    shots_df["blurb_prompt_version"] = shots_df["blurb_prompt_version"].astype("Int64")
    if seg_rows:
        segments_df = pd.DataFrame(seg_rows).set_index("id")
        feature_cols = [c for c in segments_df.columns if c not in ID_COLS]
        segments_df[feature_cols] = segments_df[feature_cols].astype("float64")
    else:
        segments_df = pd.DataFrame(columns=list(ID_COLS)).set_index(pd.Index([], name="id"))
    shape_mat = (
        np.stack(shape_rows).astype(np.float32) if shape_rows else np.zeros((0, 0), np.float32)
    )
    return shots_df, segments_df, shape_mat


def _shot_row(rec: ShotRecord, blurb_client=None) -> dict:
    from ..retrieval import blurb

    text_mp, text_log = text.compose_texts(rec.human)
    b = blurb.make(rec, blurb_client)
    row = {
        "shot": rec.shot,
        "shot_date": rec.shot_date,
        "campaign": rec.campaign,
        "run_id": rec.human.run_id,
        "mpid": rec.human.mpid,
        "mp_title": rec.human.mp_title,
        "verdict": rec.human.verdict,
        "regime": rec.labels.regime,
        "regime_source": rec.labels.regime_source,
        "operational": sorted(rec.labels.operational),
        "ip_target_hit": rec.outcome.ip_target_hit,
        "end_reason": rec.outcome.end_reason,
        "n_segments": len(rec.segments),
        "coverage_fraction": sum(v == "present" for v in rec.coverage.values())
        / max(1, len(rec.coverage)),
        "text_mp": text_mp,
        "text_log": text_log,
        "record_json": rec.model_dump_json(),
        "blurb": b.text,
        "blurb_source": b.source,
        **{f"blurb_{k}": v for k, v in _blurb_provenance(blurb_client).items()},
        # Which raw layer this row was built from, and what it saw there. `groups_filled` counts
        # every group the reader listed, not only the six that get a column of their own.
        "reader": rec.reader,
        "groups_filled": len(rec.raw_groups),
        **{f"has_{g}": g in set(rec.raw_groups) for g in FAST_GROUPS},
        # JSON, not a column per signal: the set of resolvers is per shot and per signal (one
        # shot's file legitimately mixes archive, corpus and fdp), and a wide table of eighty
        # nullable string columns would be unreadable and mostly empty.
        "feature_resolvers": json.dumps(rec.feature_resolvers, sort_keys=True),
        "has_frame_codes": rec.has_frame_codes,
    }
    if tuple(row) != SHOT_COLS:
        raise RuntimeError(f"shots.parquet row keys {tuple(row)} != SHOT_COLS {SHOT_COLS}")
    return row


def _empty_shots() -> pd.DataFrame:
    return pd.DataFrame(columns=list(SHOT_COLS)).set_index(pd.Index([], name="shot")).astype(
        {"blurb_prompt_version": "Int64"}
    )


def _scalar_matrix(
    segments_df: pd.DataFrame, shape_mat: np.ndarray, feature_cols: list[str] | None = None
) -> tuple[np.ndarray, list[str]]:
    """Segment scalars (+ waveform shapes) as one float32 matrix, NaN where nothing was recorded.

    `feature_cols` is passed on the `add()` path so the frozen PCA sees the columns it was fitted
    on, in that order, regardless of what the new shots happen to carry.
    """
    cols = feature_cols or [
        c
        for c in segments_df.columns
        if c not in ID_COLS and pd.api.types.is_numeric_dtype(segments_df[c])
    ]
    X = segments_df.reindex(columns=cols).to_numpy(dtype=np.float32)
    if shape_mat.size:
        X = np.concatenate([X, shape_mat], axis=1)
    return X, cols


# ------------------------------------------------------------------------------ scalar embedding


Z_CLIP = 5.0  # robust z beyond which a value is just "far"; see fit_scalar_embedding


def fit_scalar_embedding(X: np.ndarray, n_components: int) -> dict:
    """Median-impute -> robust z (median / percentile spread, clipped) -> whitened PCA; JSON for project_scalar.

    `impute_frac` is kept per column because it is the honest caveat on this embedding: a feature
    that is 90 % imputed contributes 90 % median to every distance computed from it, and only the
    fitted artifact can say so.
    """
    from sklearn.decomposition import PCA

    X = np.asarray(X, dtype=np.float64)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        return empty_pca()
    finite = np.isfinite(X)
    with warnings.catch_warnings():
        # An all-NaN column is expected here (a channel no shot in this build recorded) and is
        # handled on the next line; numpy's warning about it is not news.
        warnings.filterwarnings("ignore", "All-NaN slice encountered", RuntimeWarning)
        medians = np.nanmedian(np.where(finite, X, np.nan), axis=0)
    # An all-NaN column has no median at all; 0.0 makes it a constant column, which the z-score
    # below flattens to zero and the PCA then ignores -- the right treatment for a channel nobody
    # in this build recorded.
    medians = np.where(np.isfinite(medians), medians, 0.0)
    Xi = np.where(finite, X, medians)
    # Robust scale: the 5th-95th percentile spread over 3.29 (= sigma for a normal), falling back
    # to 1.4826*MAD, then the std, then 1.0 for a constant column. The plain std was ruled by
    # outliers: in the first 200-shot build ne_line_mean has a median of 6.4e13 and one shot at
    # 1.23e19, so its std-scale was 8.7e17 and every ordinary shot's density collapsed to z ~ 1e-4
    # -- the feature was silently switched off for everyone but the outlier, while the outlier
    # owned a principal component. The percentile spread was chosen over the MAD after measuring
    # both on the 804-row matrix: they agree where the column is unimodal (ne_line 4.8e13 vs 4.4e13,
    # ip_mean 3.0e5 vs 3.1e5) but the MAD collapses on bimodal columns (flat-top vs ramp rows of
    # the same actuator: a 12-row synthetic database put 5 of 6 flat tops beyond the clip) and
    # saturates 12 % of the heavy-tailed dalpha rows. Centring on the median and clipping at
    # +-Z_CLIP keeps an outlier a "very high" row instead of the axis everything else is measured
    # against. The keys stay "mean"/"scale" so project_scalar and query_z read them as before;
    # "mean" now holds the median.
    p5, p95 = np.percentile(Xi, [5, 95], axis=0)
    spread = (p95 - p5) / 3.29
    mad = np.median(np.abs(Xi - medians), axis=0) * 1.4826
    std = Xi.std(axis=0)
    scale = np.where(spread > 0, spread, np.where(mad > 0, mad, np.where(std > 0, std, 1.0)))
    Z = np.clip((Xi - medians) / scale, -Z_CLIP, Z_CLIP)
    k = max(1, int(min(n_components, Z.shape[0] - 1, Z.shape[1])))
    pca = PCA(n_components=k, whiten=True, random_state=0).fit(Z)
    return {
        "medians": medians.tolist(),
        "mean": medians.tolist(),
        "scale": scale.tolist(),
        "scaling": "median / ((p95 - p5) / 3.29), clipped",
        "z_clip": Z_CLIP,
        "components": pca.components_.tolist(),
        "explained_variance": pca.explained_variance_.tolist(),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
        "impute_frac": (1.0 - finite.mean(axis=0)).tolist(),
        "fitted_on_n": int(Z.shape[0]),
        "feature_cols": [],
        "n_shape": 0,
    }


def empty_pca() -> dict:
    return {
        "medians": [], "mean": [], "scale": [], "components": [], "explained_variance": [],
        "explained_variance_ratio": [], "impute_frac": [], "fitted_on_n": 0,
        "feature_cols": [], "n_shape": 0,
    }  # fmt: skip


def project_scalar(X: np.ndarray, pca: dict) -> np.ndarray:
    """Project rows onto the frozen PCA and L2-normalize, so cosine similarity is a dot product."""
    X = np.asarray(X, dtype=np.float64)
    if not pca.get("components"):
        return np.zeros((X.shape[0], 1), np.float32)
    Xi = np.where(np.isfinite(X), X, np.asarray(pca["medians"]))
    Z = (Xi - np.asarray(pca["mean"])) / np.asarray(pca["scale"])
    clip = pca.get("z_clip")  # absent in a pca.json fitted before robust scaling: no clip then
    if clip:
        Z = np.clip(Z, -clip, clip)
    var = np.asarray(pca["explained_variance"], dtype=np.float64)
    # A component with no variance left (more components asked for than the data supports) would
    # divide by zero and put inf into every row; whitening it by 1.0 leaves it the zero column it
    # already is.
    var = np.where(var > 0, var, 1.0)
    E = Z @ np.asarray(pca["components"]).T / np.sqrt(var)
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    return (E / np.where(norms == 0, 1.0, norms)).astype(np.float32)


# ------------------------------------------------------------------------------ coverage report


def coverage_report(records: list[ShotRecord]) -> pd.DataFrame:
    """present / unavailable / pending / not_installed per field, one row per registry field.

    `pending` means "declared fetchable and not on disk yet", which is a to-do; `unavailable`
    means the archive does not have it for this shot, which is not. Keeping them apart is the
    whole point of the table -- a single "missing" count would hide the difference.
    """
    rows: dict[str, dict[str, int]] = {}
    for rec in records:
        for name, status in rec.coverage.items():
            row = rows.setdefault(name, dict.fromkeys(STATUSES, 0))
            row[status] = row.get(status, 0) + 1
    df = (
        pd.DataFrame.from_dict(rows, orient="index")
        .reindex(columns=list(STATUSES))
        .fillna(0)
        .astype(int)
    )
    total = df.sum(axis=1).replace(0, 1)
    df["present_frac"] = (df["present"] / total).round(3)
    return df.sort_index()


def format_coverage(df: pd.DataFrame, title: str = "coverage") -> str:
    """The table Nathan reads: fields grouped by actuator system, worst coverage first."""
    if df.empty:
        return f"{title}: (no records)"

    def group_of(name: str) -> str:
        for prefix, label in (
            ("pnbi_", "nbi"),
            ("pech_", "ech"),
            ("gas_", "gas"),
            ("irmp_", "coil_rmp"),
        ):
            if name.startswith(prefix):
                return label
        return "signals"

    n_shots = int(df[list(STATUSES)].sum(axis=1).max())
    full = int((df["present"] == n_shots).sum())
    lines = [
        (f"{title}: {len(df)} fields over {n_shots} shots "
         f"({full} present on every shot, {int((df['present'] == 0).sum())} on none)"),
        "",
    ]
    for grp in ("signals", "nbi", "ech", "gas", "coil_rmp"):
        sub = df[[group_of(i) == grp for i in df.index]]
        if sub.empty:
            continue
        lines.append(f"[{grp}]")
        lines.append(
            f"  {'field':<22}{'present':>9}{'unavail':>9}{'pending':>9}{'not_inst':>10}{'frac':>8}"
        )
        for name, r in sub.sort_values(["present_frac", "present"]).iterrows():
            # int() on each cell: iterrows() gives one Series per row, so a float present_frac
            # would otherwise widen every count in the row to "70.0".
            lines.append(
                f"  {name:<22}{int(r['present']):>9}{int(r['unavailable']):>9}"
                f"{int(r['pending']):>9}{int(r['not_installed']):>10}{r['present_frac']:>8.2f}"
            )
        lines.append("")
    return "\n".join(lines)


# ------------------------------------------------------------------------------------- writing


def _write_tables(
    out: Path,
    shots_df: pd.DataFrame,
    segments_df: pd.DataFrame,
    shape_mat: np.ndarray,
    emb: dict[str, np.ndarray],
    pca: dict,
    manifest: dict,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    shots_df.to_parquet(out / "shots.parquet")
    segments_df.to_parquet(out / "segments.parquet")
    # The waveform shapes are half the feature vector and are not in segments.parquet; without
    # them the embedding cannot be refitted or audited without re-reading every raw file.
    np.save(out / "shapes.npy", shape_mat)
    for name, mat in emb.items():
        np.save(out / f"emb_{name}.npy", mat)
    (out / "pca.json").write_text(json.dumps(pca))
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))


def _blurb_client():
    """The shared LLM client when one is reachable, else None (every blurb is then the template).

    A build must never wait on a model that is not running, so this asks once and hands `None`
    down rather than letting each record discover the absence for itself.
    """
    from ..llm.client import LLMClient

    client = LLMClient()
    return client if client.available()[0] else None


def _blurb_provenance(client=None) -> dict[str, str | int]:
    """Configuration used for this pass, including when a row falls back to a template."""
    cfg = client.cfg if client is not None else config.load_yaml("llm.yaml")
    key = cfg["blurb"]["model"]
    return {
        "model": str(cfg["models"].get(key, key)),
        "prompt_version": int(cfg["blurb"]["prompt_version"]),
    }


def _blurb_counts(shots_df: pd.DataFrame, client=None) -> dict[str, object]:
    col = shots_df["blurb_source"] if "blurb_source" in shots_df.columns else pd.Series(dtype=str)
    versions = pd.to_numeric(
        shots_df.get("blurb_prompt_version", pd.Series(dtype="Int64")), errors="coerce"
    ).dropna().astype(int).value_counts().sort_index()
    return {
        **{k: int(col.eq(k).sum()) for k in ("llm", "template", "human")},
        **_blurb_provenance(client),
        "prompt_versions": {str(version): int(count) for version, count in versions.items()},
    }


def write_blurbs(
    paths: config.Paths, client, only_missing: bool = True, *,
    limit: int | None = None, dry_run: bool = False, shots: Sequence[int] | None = None,
    workers: int = 1,
) -> int:
    """Backfill blurbs into shots.parquet with the model; rewrite the file atomically.
    Return the number of accepted model replies. Select templates, empty blurbs and stale or
    unknown prompt versions in shot order, or exactly the sorted unique targeted `shots` regardless
    of `only_missing`, then apply `limit`. Dry runs print each candidate, gate verdict and final
    text without writing database files or the client's request cache.

    `workers` runs the `_blurb.make` calls (each an `agy` subprocess through
    `LLMClient.chat`, ~5-10 s) in a `ThreadPoolExecutor` so a large backfill does not
    run one call at a time; the agy provider is a subprocess per call, so it is
    thread-safe with no shared mutable state. Whatever runs the calls, results are
    written into `df` (and printed, for a dry run) from this thread afterwards, in the
    same shot order `workers=1` always used -- so `shots.parquet` and stdout are
    identical regardless of thread scheduling, and one shot's failure (caught the same
    way inside `_blurb.make` either way) never stops another's. `workers=1` takes no
    executor at all: the original sequential loop, unchanged.
    """
    from ..retrieval import blurb as _blurb
    from .store import ShotDB

    if limit is not None and limit < 0:
        raise ValueError("blurb limit must be non-negative")
    if workers < 1:
        raise ValueError("blurb workers must be at least 1")
    db = ShotDB.load(paths.db_dir)
    df = db.shots.copy()
    targeted = sorted({int(shot) for shot in shots}) if shots is not None else None
    if targeted is not None:
        known = {int(shot) for shot in df.index}
        unknown = [shot for shot in targeted if shot not in known]
        if unknown:
            raise ValueError(f"shot not in shots.parquet: {unknown}")
    provenance = _blurb_provenance(client)
    for col, default in (("blurb", ""), ("blurb_source", "template"), ("blurb_model", None)):
        if col not in df:
            df[col] = default
    versions = df.get("blurb_prompt_version", pd.Series(index=df.index, dtype="Int64"))
    df["blurb_prompt_version"] = pd.to_numeric(versions, errors="coerce").astype("Int64")
    source = df["blurb_source"].fillna("template")
    # A hand-written row ("human") was never prompted, so it has no prompt version to be stale;
    # only model rows age out. `shots=` still rewrites one on purpose.
    missing = (
        ~source.isin(["llm", "human"])
        | df["blurb"].fillna("").str.strip().eq("")
        | (source.eq("llm") & df["blurb_prompt_version"].fillna(0).lt(provenance["prompt_version"]))
    )
    todo = targeted if targeted is not None else list(
        (df.index[missing] if only_missing else df.index).sort_values()
    )
    if limit is not None:
        todo = todo[:limit]
    if not len(todo):
        return 0

    def _one(shot: int):
        rec = db.get(int(shot))
        return rec, _blurb.make(rec, client, cache=False if dry_run else None)

    # Rows are consumed in `todo` (shot) order on this thread whatever the workers do,
    # so the parquet is byte-stable; the single-worker path stays a lazy stream, so a
    # dry run prints each shot as it is made instead of after the last one.
    if workers == 1:
        rows = ((shot, _one(shot)) for shot in todo)
    else:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = {shot: ex.submit(_one, shot) for shot in todo}
        rows = ((shot, fut.result()) for shot, fut in futures.items())
    n = 0
    for shot, (rec, b) in rows:
        if dry_run:
            verdict = "PASS" if b.reason is None else f"FAIL ({b.reason})"
            print(
                f"Shot {shot}\nSource text: {len(_blurb.source_text(rec))} chars\n"
                f"Candidate: {b.candidate or '(none)'}\nGate: {verdict}\n"
                f"Final ({b.source}): {b.text}\n"
            )
        else:
            df.loc[shot, ["blurb", "blurb_source", "blurb_model", "blurb_prompt_version"]] = [
                b.text, b.source, provenance["model"], provenance["prompt_version"],
            ]
        n += int(b.source == "llm")
    if dry_run:
        return n
    tmp = paths.db_dir / "shots.parquet.part"
    df.to_parquet(tmp)
    os.replace(tmp, paths.db_dir / "shots.parquet")
    counts = _blurb_counts(df, client)
    # The manifest's counts are what `build()` and `add()` write and what a reader consults to see
    # how much of the database the model wrote; leaving them behind would have a fully backfilled
    # database still claiming 0 llm blurbs. Written the same way as the parquet: tmp + replace.
    mf = paths.db_dir / "manifest.json"
    if mf.exists():
        manifest = json.loads(mf.read_text())
        manifest["blurbs"] = counts
        tmp_mf = paths.db_dir / "manifest.json.part"
        tmp_mf.write_text(json.dumps(manifest, indent=2, default=str))
        os.replace(tmp_mf, mf)
    _log.info(
        "write_blurbs: %d llm, %d template (of %d rows)", counts["llm"], counts["template"], len(df)
    )
    return n


def refresh_frame_codes(db_dir: Path, paths: config.Paths) -> dict:
    """Recompute `shots.parquet`'s `has_frame_codes` from the directories on disk. Returns counts.

    The column is set at BUILD time from the caches that existed then, and encoding is a separate,
    later job -- so a database built before the encode says false for every shot the encode has
    since written. It said 13 true / 487 false while all 500 production caches existed, and a
    reader has no way to tell a stale flag from a shot that really has no codes. `shot_design labels
    join` calls this because the join is the step that runs after the long jobs and republishes,
    and because a rebuild to fix one boolean costs an hour.

    The flag lives twice -- as a column and inside `record_json`, which is what `describe_shot`
    reads back -- so both are refreshed, and only the rows that changed are re-serialised.
    """
    path = Path(db_dir) / "shots.parquet"
    if not path.exists():
        return {"n_shots": 0, "n_has_frame_codes": 0, "n_changed": 0, "refreshed": False,
                "reason": f"no {path}", "frame_codes_dirs": []}
    df = pd.read_parquet(path)
    shots = [int(s) for s in df.index]
    now = [frame_codes_path(shot, paths) is not None for shot in shots]
    was = [bool(v) for v in df["has_frame_codes"]] if "has_frame_codes" in df.columns else \
        [False] * len(shots)
    changed = [i for i, (a, b) in enumerate(zip(was, now, strict=True)) if a != b]
    if changed:
        df["has_frame_codes"] = now
        records = df["record_json"].to_list()
        for i in changed:
            rec = ShotRecord.model_validate_json(records[i])
            rec.has_frame_codes = now[i]
            records[i] = rec.model_dump_json()
        df["record_json"] = records
        tmp = Path(db_dir) / "shots.parquet.part"
        df.to_parquet(tmp)
        os.replace(tmp, path)
    return {
        "n_shots": len(shots),
        "n_has_frame_codes": int(sum(now)),
        "n_was_true": int(sum(was)),
        "n_changed": len(changed),
        "refreshed": bool(changed),
        "frame_codes_dirs": [str(d) for d in frame_codes_dirs(paths)],
    }


def _tmp_dir(db_dir: Path) -> Path:
    return db_dir.parent / f"{db_dir.name}.tmp"


def _move_onto(src: Path, dst: Path) -> None:
    """Rename `src` over `dst`, copying instead when the two are on different filesystems.

    The rename is the path that matters and stays first: it is atomic, so a reader of `dst` sees
    the whole of one file or the whole of the other. But the caller stages the text subset beside
    db_dir while `text_cache_dir` is wherever the paths file says, and an `SHOT_DESIGN_PATHS` file that
    puts them on different mounts -- exactly the scratch-database workflow docs/SHOT_DESIGN.md
    recommends -- makes `os.replace` raise EXDEV. The fallback copies, so it is not atomic; that
    is acceptable here and only here, because what it moves is a cache the next build rewrites.
    """
    try:
        os.replace(src, dst)
    except OSError as exc:
        if exc.errno != errno.EXDEV:
            raise
        shutil.move(str(src), str(dst))


#: Every name `build`/`add` write into db_dir -- the two tables, the shape matrix, the embedding
#: matrices, the PCA and the manifest. `_write_tables` and `ignite.encode_db` between them write
#: exactly these, and `store.ShotDB.load` reads exactly these back.
#:
#: This list is the ONLY thing a publish is allowed to delete. The rule is that way round on
#: purpose: db_dir has more than one producer -- `shot_design corpus scan` writes its census there and
#: `shot_design labels join` writes labels_wide/events/text_claims.parquet -- and a rule phrased as
#: "carry the foreign files across" has to name every one of them correctly or it deletes a
#: table, which is what happened to the join's three. Phrased as "delete only what I write", a
#: producer this module has never heard of is safe by default, and the failure mode of getting
#: THIS list wrong is a stale file left behind rather than someone else's work destroyed.
#:
#: Stale build-owned files must still go: an emb_ignite_*.npy left beside a database built with
#: --no-encode is worse than a missing file, it is a database that lies about what it holds.
#:
#: manifest.json IS build-owned, so a rebuild still drops the `labels` block `shot_design labels join`
#: merges into it while keeping the join's three tables. That asymmetry is deliberate and is the
#: marker: tables with no `labels` block in the manifest were not written for THIS build, and the
#: join has to be re-run before they are quoted against it.
BUILD_FILES = (
    "shots.parquet",
    "segments.parquet",
    "shapes.npy",
    "pca.json",
    "manifest.json",
    # Not written by a publish: it is the staging name `build` renames over manifest.json when it
    # rewrites the timings, and a crash in that window leaves it behind for good. Claiming it here
    # is what lets the next publish prune it.
    "manifest.json.part",
    "windows.parquet",
)
#: Same set, for the matrices whose names depend on what was encoded (emb_scalar, emb_text_mp,
#: emb_text_log, emb_ignite_seg, emb_ignite_win).
BUILD_GLOBS = ("emb_*.npy",)


def _build_owned(db_dir: Path) -> set[str]:
    """The build-owned files that exist in `db_dir` right now."""
    names = {n for n in BUILD_FILES if (db_dir / n).exists()}
    for pattern in BUILD_GLOBS:
        names |= {p.name for p in db_dir.glob(pattern)}
    return names


def _publish(tmp: Path, db_dir: Path) -> None:
    """Move what the build wrote into `db_dir`, then delete the build-owned files it did NOT
    write. Nothing else in the directory is read, moved or removed.

    Per file rather than one directory swap: the swap was atomic for the database as a whole but
    could only preserve foreign files it was told about by name, and the names went stale. Each
    `os.replace` here is itself atomic (tmp is a sibling of db_dir, so same filesystem), the new
    tables land before the stale ones are pruned, and a crash mid-publish leaves a directory whose
    files are each whole -- never a half-written one.
    """
    db_dir.mkdir(parents=True, exist_ok=True)
    written: set[str] = set()
    for src in sorted(tmp.iterdir()):
        dst = db_dir / src.name
        if src.is_dir():  # no producer writes one today; handled so one would not be silently lost
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
        else:
            os.replace(src, dst)
        written.add(src.name)
    unclaimed = written - _build_owned(db_dir)
    if unclaimed:
        # Not fatal, but it means BUILD_FILES no longer enumerates what this module writes, and
        # the next rebuild will leave these behind rather than replacing them.
        _log.warning("published files BUILD_FILES does not claim: %s", ", ".join(sorted(unclaimed)))
    for name in sorted(_build_owned(db_dir) - written):
        (db_dir / name).unlink()
    shutil.rmtree(tmp, ignore_errors=True)


def _embeddings(
    shots_df: pd.DataFrame, X: np.ndarray, pca: dict, seconds: dict[str, float] | None = None
) -> dict[str, np.ndarray]:
    """The three embedding matrices. `seconds`, when a caller passes its phase dict, receives the
    scalar projection and the MiniLM text pass under their own keys -- they are different work on
    different hardware and one number for both cannot say which one idled the cores."""
    seconds = dict.fromkeys(PHASES, 0.0) if seconds is None else seconds
    with _phase(seconds, "scalar_embedding"):
        emb = {"scalar": project_scalar(X, pca)}
    with _phase(seconds, "text_embedding"):
        for key in TEXT_KEYS:
            emb[key] = text.embed_texts(shots_df[key].tolist())
    return emb


def _encode(tmp: Path, records: list[ShotRecord], paths: config.Paths, workers: int = 8) -> dict:
    """Ask shotdb.ignite to add its embeddings, and say plainly what happened either way.

    Without the model bundle on disk the answer is `not_installed`, and that is an ordinary path,
    not an error path: a scalar-only database is fully usable -- it just cannot answer
    waveform-similarity queries -- and the manifest has to make the difference visible rather than
    leave a caller guessing why emb_ignite_* is absent. With the bundle, ignite.encode_db writes
    emb_ignite_seg.npy (segment rows), emb_ignite_win.npy + windows.parquet (fragment windows) into
    `tmp` and returns the block describing the model, the pooling and the per-modality coverage.
    """
    try:
        from . import ignite
    except ImportError as e:
        return {
            "status": "not_installed",
            "reason": f"shotdb.ignite unavailable: {e}",
            "channels": [],
        }
    try:
        result = ignite.encode_db(tmp, records, paths, workers=workers, log=_log.info)
    except Exception as e:  # noqa: BLE001 — a missing checkpoint must not lose a finished scalar build
        return {"status": "failed", "reason": f"{type(e).__name__}: {e}", "channels": []}
    return result or {"status": "failed", "reason": "encode_db returned nothing", "channels": []}


def _check_publish(db_dir: Path, shot_source: str | None, n_shots: int, force: bool):
    """Refuse a different or smaller rebuild, retaining readable history for an override."""
    previous = {}
    problem = None
    try:
        manifest = db_dir / "manifest.json"
        manifest.stat()
        previous = json.loads(manifest.read_text(encoding="utf-8"))
        if not isinstance(previous, dict):
            previous = {}
        if (
            "shot_source" not in previous
            or previous["shot_source"] is not None
            and not isinstance(previous["shot_source"], str)
            or type(previous.get("n_shots")) is not int
            or previous["n_shots"] < 0
        ):
            problem = "invalid manifest fields"
    except FileNotFoundError:
        return None
    except (OSError, ValueError) as exc:
        problem = f"unreadable/corrupt manifest: {exc}"
    history = {key: previous.get(key) for key in ("shot_source", "n_shots", "built_at")}
    if not force and (
        problem or history["shot_source"] != shot_source or n_shots < history["n_shots"]
    ):
        raise ValueError(
            f"refusing to replace database at {db_dir}: existing source "
            f"{history['shot_source']!r}, n_shots={history['n_shots']!r}; "
            f"new source {shot_source!r}, n_shots={n_shots}. "
            f"{problem + '; ' if problem else ''}Use --force to override."
        )
    return history


def build(
    shots: list[int],
    paths: config.Paths,
    cfg: dict,
    workers: int = 8,
    encode: bool = True,
    reuse: bool = True,
    reader_kind: str = "legacy",
    shot_source: str | None = None,
    n_requested: int | None = None,
    limit: int | None = None,
    force: bool = False,
) -> BuildReport:
    """Full rebuild into <db_dir>, via <db_dir>.tmp so a crash leaves the old database intact.

    `reader_kind` picks the raw layer ("legacy" -- the d3d_fusion_data layout this database was
    first built from -- or "corpus"). The other three say where the shots came from:
    `shot_source` is how they were selected (`list:<name>`, `list-file:<path>`, `shots:<n>`),
    `n_requested` how many that selection named and `limit` the `--limit` applied to it, if any.
    All four go in the manifest, because two databases at the same path built from different
    layers or different selections are different databases and have to say which they are -- and
    "the shots are in the manifest's `shots` array" is a reconstruction, not a record.
    """
    t_start = time.perf_counter()
    seconds = dict.fromkeys(PHASES, 0.0)
    shots = sorted(set(shots))
    previous = _check_publish(paths.db_dir, shot_source, len(shots), force)
    report = BuildReport(shots=[])
    with _phase(seconds, "read_records"):
        # Stage the text subset as well: failures can reduce the actual count after the
        # preflight check. A refused rebuild must leave the old cache and DB untouched.
        paths.db_dir.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix=".build-text-", dir=paths.db_dir.parent) as cache:
            read_paths = paths.model_copy(update={"text_cache_dir": Path(cache)})
            old_subset, subset = text.subset_path(paths), text.subset_path(read_paths)
            if old_subset.exists():
                shutil.copy2(old_subset, subset)
            text.build_logs_subset(read_paths, set(shots))
            records, shapes = _build_many(shots, read_paths, cfg, workers, report, reader_kind)
            previous = _check_publish(paths.db_dir, shot_source, len(records), force)
            if subset.exists():
                old_subset.parent.mkdir(parents=True, exist_ok=True)
                _move_onto(subset, old_subset)
    with _phase(seconds, "segment"):
        shots_df, segments_df, shape_mat = records_to_tables(records, shapes, _blurb_client())
    with _phase(seconds, "scalar_embedding"):
        X, feature_cols = _scalar_matrix(segments_df, shape_mat)
        # PCA needs more rows than components to mean anything; below that the embedding is a
        # single zero column and every cosine distance is zero, which the manifest's fitted_on_n
        # makes visible.
        pca = (
            fit_scalar_embedding(X, cfg["scalar"]["n_components"])
            if X.shape[0] > 2
            else empty_pca()
        )
        pca["feature_cols"] = feature_cols
        pca["n_shape"] = int(shape_mat.shape[1]) if shape_mat.size else 0
    emb = _embeddings(shots_df, X, pca, seconds)
    cov = coverage_report(records)
    manifest = {
        "built_at": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(),
        "config_sha": _config_sha(),
        "reader": reader_kind,
        # How the shot list was chosen, how many shots it named, how many were built, and the
        # limit that cut it down. Four facts, because one field cannot carry them: `list` alone
        # was null for a `--list-file`/`--shots` build and named a 500-shot list for a
        # `--limit 20` database.
        "shot_source": shot_source,
        "n_requested": n_requested,
        "n_built": len(records),
        "limit": limit,
        "shots": report.shots,
        "n_shots": len(records),
        "n_segments": len(segments_df),
        "failed": {str(k): v for k, v in report.failed.items()},
        "adds_since_fit": 0,
        "segments_since_fit": 0,
        "n_features": int(X.shape[1]) if X.size else 0,
        "pca_components": len(pca["components"]),
        "pca_fitted_on_n": pca["fitted_on_n"],
        "pca_explained_variance_ratio_sum": round(float(sum(pca["explained_variance_ratio"])), 4),
        "mean_impute_frac": round(float(np.mean(pca["impute_frac"])), 4)
        if pca["impute_frac"]
        else None,
        "blurbs": _blurb_counts(shots_df),
        "campaign_note": "campaign is a shot-number band from actuators.yaml, not a date range",
        "coverage": cov.to_dict(orient="index"),
        # What `encode=False` actually skips, said in full: the IGNITE waveform channel and
        # nothing else. The scalar and MiniLM text embeddings below are part of the database
        # itself and are written either way, so a reader of this manifest does not have to guess
        # whether an emb_*.npy is missing because of it. Said WITHOUT naming a CLI flag: this is
        # a library call, and a programmatic build nobody passed a flag to would otherwise get
        # "--no-encode" written into its manifest. `cli.cmd_build` adds the flag when a flag is
        # what did it.
        "ignite": {
            "status": "disabled",
            "reason": "encoding disabled: the IGNITE waveform channel was skipped; "
            "scalar and text embeddings are built as usual",
            "channels": [],
        },
        # Wall seconds per PHASE, for the question a single elapsed cannot answer: which phase
        # left the cores idle. The dict is written twice -- the publish is the last phase and
        # cannot have timed itself when the manifest it moves is written -- and it is the
        # published copy, rewritten below, that carries every phase. If that rewrite never landed,
        # `write_tables` and `publish` are the two keys left at 0.0.
        "phase_seconds": seconds,
        # The whole of `build`, so the phases can be read against something. They are NOT
        # exhaustive: the coverage report, this manifest, and on an encoding build the entire
        # IGNITE pass (which times itself into `ignite.elapsed_s`) are in no phase, and
        # `build_elapsed_s - sum(phase_seconds.values())` is how much is unaccounted for. Set at
        # the rewrite below, the last moment before the published manifest is serialised.
        "build_elapsed_s": None,
    }
    if force and previous is not None:
        manifest["forced_over"] = previous
    with _phase(seconds, "write_tables"):
        tmp = _tmp_dir(paths.db_dir)
        if tmp.exists():
            shutil.rmtree(tmp)
        _write_tables(tmp, shots_df, segments_df, shape_mat, emb, pca, manifest)
    if encode:
        block = _reuse_encodings(tmp, records, segments_df, paths, workers) if reuse else None
        manifest["ignite"] = block or _encode(tmp, records, paths, workers=workers)
        report.encoded = manifest["ignite"].get("status") == "ok"
        if not report.encoded:
            _log.info("ignite channel skipped: %s", manifest["ignite"]["reason"])
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    with _phase(seconds, "publish"):
        _publish(tmp, paths.db_dir)
    # Rewrite the published manifest now that `publish` has a number, through a sibling temp file
    # renamed over it, so a reader sees the whole of one manifest or the whole of the other.
    manifest["phase_seconds"] = {k: round(v, 6) for k, v in seconds.items()}
    manifest["build_elapsed_s"] = round(time.perf_counter() - t_start, 6)
    part = paths.db_dir / "manifest.json.part"
    part.write_text(json.dumps(manifest, indent=2, default=str))
    os.replace(part, paths.db_dir / "manifest.json")
    # The total on the same line, because the phases only mean something against it: what it
    # exceeds their sum by is the work no phase timed.
    _log.info(
        "build phases (wall s): %s, total %.1f (untimed %.1f)",
        ", ".join(f"{name} {seconds[name]:.1f}" for name in PHASES),
        manifest["build_elapsed_s"],
        manifest["build_elapsed_s"] - sum(seconds.values()),
    )
    report.n_segments, report.db_dir = len(segments_df), paths.db_dir
    report.elapsed_s = time.perf_counter() - t_start
    return report


def _carry_over(
    tmp: Path,
    db,
    new_records: list[ShotRecord],
    segments_df: pd.DataFrame,
    keep_segs: pd.Series,
    paths: config.Paths,
    workers: int,
) -> dict:
    """The IGNITE matrices for a database that shares shots with an existing one: kept rows by
    segment id, new shots freshly encoded. Raises when it cannot (bundle gone, model changed,
    encode error); the caller decides what that means.

    The segment matrix is indexed by segments.parquet's row order, which the caller has just
    rebuilt, so kept rows are placed by segment id rather than position; the window table is the
    kept shots' rows plus the new shots'. Coverage is re-read from the final matrix so a carried
    row counts exactly like a fresh one.
    """
    import time

    from . import ignite

    t0 = time.perf_counter()
    old = dict(db.manifest.get("ignite", {}))
    codecs = ignite.load_codecs(ignite.bundle_dir(paths))
    dims = [int(codecs[n][1].d_model) for n in codecs]
    if dims != list(old.get("dims", [])) or list(codecs) != list(old.get("modalities", [])):
        raise RuntimeError("model changed since the database was built (modalities/dims differ)")
    # v2 identified the weights by a pinned Hub revision. A pinned bundle has
    # none, and its own sha256 table does not answer this question either:
    # re-pinning rewrites the codecs and the manifest together, so a re-pinned
    # bundle passes `load_codecs` above. The digest OF the manifest separates
    # one pin from the next, and mixing two pins in one matrix is exactly what
    # this guard exists to prevent.
    changed = ignite.check_same_bundle(old.get("model", {}), paths)
    if changed:
        raise RuntimeError(changed)
    width = sum(dims)
    embeddings, errors = ignite.encode_records(new_records, codecs, paths, workers=workers)
    seg_ids = segments_df.index.tolist()
    mat = ignite.segment_matrix(embeddings, new_records, seg_ids, width)
    old_pos = {sid: i for i, sid in enumerate(db.segments_base.index)}
    old_mat = db.emb["ignite_seg"]
    kept = 0
    for i, sid in enumerate(seg_ids):
        j = old_pos.get(sid)
        if j is not None and keep_segs.iloc[j]:
            mat[i] = old_mat[j]
            kept += 1
    np.save(tmp / "emb_ignite_seg.npy", mat)
    win_new, wmat_new = ignite.window_table(embeddings, float(old["window_ms"]) / 1000.0, width)
    kept_shots = set(segments_df["shot"].tolist()) - set(embeddings) - {r.shot for r in new_records}
    if db.windows is not None and "ignite_win" in db.emb:
        keep = db.windows["shot"].isin(kept_shots)
        win = pd.concat([db.windows[keep], win_new])
        wmat = np.concatenate([db.emb["ignite_win"][keep.to_numpy()], wmat_new])
    else:
        win, wmat = win_new, wmat_new
    win.to_parquet(tmp / "windows.parquet")
    np.save(tmp / "emb_ignite_win.npy", wmat)
    coverage, n_encoded = ignite.coverage_from_matrix(
        mat, segments_df["shot"].tolist(), dims, list(codecs)
    )
    block = ignite.manifest_block(
        codecs,
        coverage,
        n_encoded,
        {**{int(k): v for k, v in old.get("failed", {}).items()}, **errors},
        f"{len(kept_shots)} shots carried over from the previous database "
        f"({kept} segment rows), {len(embeddings)} encoded now",
        time.perf_counter() - t0,
        paths,
    )
    return block


def _reuse_encodings(
    tmp: Path,
    records: list[ShotRecord],
    segments_df: pd.DataFrame,
    paths: config.Paths,
    workers: int,
) -> dict | None:
    """For `build`: carry the previous database's IGNITE rows over for every shot it already
    encoded with the same model, and encode only the rest. None means "nothing to reuse" (no
    previous database, no ignite block, or the carry-over failed) and the caller encodes everything.

    Encoding is the expensive half of a rebuild -- ~15 s a shot on the V100S, 50 min for 200 --
    while the scalar half takes a minute, so without this every features/flags change would cost
    an hour. `shot_design build --reencode` forces the full path when the raw inputs have changed.
    """
    from .store import ShotDB

    if not (paths.db_dir / "manifest.json").exists():
        return None
    try:
        db = ShotDB.load(paths.db_dir)
    except Exception as e:  # noqa: BLE001 — an unreadable previous database is not a reason to fail the build
        _log.warning("previous database unreadable, encoding everything: %s", e)
        return None
    if db.manifest.get("ignite", {}).get("status") != "ok" or "ignite_seg" not in db.emb:
        return None
    old_shots = set(db.shots.index.tolist())
    reuse = {r.shot for r in records} & old_shots
    if not reuse:
        return None
    new_records = [r for r in records if r.shot not in reuse]
    keep_segs = db.segments_base["shot"].isin(reuse)
    try:
        return _carry_over(tmp, db, new_records, segments_df, keep_segs, paths, workers)
    except Exception as e:  # noqa: BLE001 — optional operation; preserve the fallback contract
        _log.warning("could not reuse the previous encodings (%s); encoding everything", e)
        for name in ("emb_ignite_seg.npy", "emb_ignite_win.npy", "windows.parquet"):
            (tmp / name).unlink(missing_ok=True)
        return None


def add(
    shots: list[int],
    paths: config.Paths,
    cfg: dict,
    workers: int = 1,
    reader_kind: str | None = None,
) -> BuildReport:
    """Incremental upsert: new records, frozen PCA projection, embeddings only for the new texts.

    `reader_kind` defaults to the one the existing database's manifest records, so an add into a
    corpus-built database does not silently produce legacy-built rows beside corpus ones.
    """
    from .store import ShotDB

    t_start = time.perf_counter()
    db = ShotDB.load(paths.db_dir)
    reader_kind = reader_kind or str(db.manifest.get("reader") or "legacy")
    text.build_logs_subset(paths, set(shots))
    report = BuildReport(shots=[])
    new_records, shapes = _build_many(
        sorted(set(shots)), paths, cfg, workers, report, reader_kind
    )
    n_shots, n_segs, shape_mat = records_to_tables(new_records, shapes, _blurb_client())
    if shape_mat.size and shape_mat.shape[1] != db.pca["n_shape"]:
        raise ValueError(
            f"waveform width changed since the fit ({shape_mat.shape[1]} vs {db.pca['n_shape']}); rebuild instead"
        )
    keep_shots = ~db.shots["shot"].isin(report.shots)
    keep_segs = ~db.segments_base["shot"].isin(report.shots)
    shots_df = pd.concat([db.shots[keep_shots], n_shots]).sort_index()
    segments_df = pd.concat([db.segments_base[keep_segs], n_segs])
    X, _ = _scalar_matrix(n_segs, shape_mat, db.pca["feature_cols"])
    kept_shapes = db.shape_matrix()[keep_segs.to_numpy()]
    all_shapes = np.concatenate([kept_shapes, shape_mat]) if shape_mat.size else kept_shapes
    # Kept rows precede new rows in both the frames and the matrices, so the embeddings stay
    # aligned row-for-row without any join. shots_df is sorted afterwards, so its two matrices are
    # reordered to match rather than concatenated blindly.
    kept_scalar = db.emb["scalar"][keep_segs.to_numpy()]
    emb = {
        "scalar": np.concatenate([kept_scalar, project_scalar(X, db.pca)])
        if X.size
        else kept_scalar,
    }
    for key in TEXT_KEYS:
        emb[key] = np.concatenate(
            [db.emb[key][keep_shots.to_numpy()], text.embed_texts(n_shots[key].tolist())]
        )
    order = np.argsort(
        np.concatenate(
            [db.shots.index.to_numpy()[keep_shots.to_numpy()], n_shots.index.to_numpy()]
        ),
        kind="stable",
    )
    for key in TEXT_KEYS:
        emb[key] = emb[key][order]
    manifest = dict(db.manifest)
    manifest.update(
        {
            "built_at": dt.datetime.now(dt.UTC).isoformat(),
            "shots": sorted(set(manifest.get("shots", [])) | set(report.shots)),
            "n_shots": len(shots_df),
            "n_segments": len(segments_df),
            "failed": {
                **manifest.get("failed", {}),
                **{str(k): v for k, v in report.failed.items()},
            },
            "blurbs": _blurb_counts(shots_df),
            "adds_since_fit": int(manifest.get("adds_since_fit", 0)) + len(report.shots),
            "segments_since_fit": int(manifest.get("segments_since_fit", 0)) + len(n_segs),
        }
    )
    fitted_on = max(1, int(db.pca.get("fitted_on_n", 1)))
    if manifest["segments_since_fit"] > 0.2 * fitted_on:
        _log.warning(
            "%d segments added since the PCA was fitted on %d; the embedding is drifting -- rebuild",
            manifest["segments_since_fit"],
            fitted_on,
        )
    tmp = _tmp_dir(paths.db_dir)
    if tmp.exists():
        shutil.rmtree(tmp)
    _write_tables(tmp, shots_df, segments_df, all_shapes, emb, db.pca, manifest)
    if manifest.get("ignite", {}).get("status") == "ok":
        try:
            manifest["ignite"] = _carry_over(
                tmp, db, new_records, segments_df, keep_segs, paths, workers
            )
        except Exception as e:  # noqa: BLE001 — never lose a finished upsert over the optional channel
            _log.warning("ignite matrices dropped by add(): %s", e)
            for name in ("emb_ignite_seg.npy", "emb_ignite_win.npy", "windows.parquet"):
                (tmp / name).unlink(missing_ok=True)
            manifest["ignite"] = {
                "status": "stale",
                "reason": f"dropped by add(): {e}; rebuild to re-encode",
                "channels": [],
            }
        (tmp / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    _publish(tmp, paths.db_dir)
    report.n_segments, report.db_dir = len(segments_df), paths.db_dir
    report.elapsed_s = time.perf_counter() - t_start
    return report
