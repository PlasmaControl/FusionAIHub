"""Curated RWM onsets and an explicitly unconfirmed high-beta magnetic screen.

N1RMS alone cannot distinguish an RWM from tearing or applied fields. The
screen therefore writes uncertain candidates, never present or absent.

Two methods draft the review page's source. `detect` (`rwm_candidates` v1) is
the screen alone; curated onsets keep their times without an end. Since round
five `detect_onsets` (`rwm_onsets`) adds the curated onsets: at each onset o,
`[o - PRE_ONSET_MS, o)` is uncertain and the mode is present from o to its
measured end (`onset_end`), the first of N1RMS back at or below its pre-onset
baseline (both `SMOOTH_MS` centred means), the beta_N collapse and the end of
the plasma current, at most `MAX_PRESENT_MS` after o. v2 counts the N1RMS end
only when the mean rises above the baseline within `RISE_MS` of o; an onset
without that response is present for `MIN_PRESENT_MS` and uncertain to the
other ends (v1 took the first rise however late). Only the onset is curated
evidence; the end is measured on signals that are not RWM-specific.

v3 (round five, R1b): a screen candidate at least `TRACK_COVER` of whose time
lies inside n = 1 Mirnov tracks of the mhd_modes `MHD_CACHE` pipeline cache is a
rotating mode, not an RWM, and is dropped (that time is unassessed); and an
n = 2 onset's end is measured on N2RMS when the shot has it cached, else on
N1RMS, flagged (`end_rms`). v4 (R1b clarified): the tracks are every n = 1
Mirnov track the mhd_n1 parent counts, whatever its resonance status
(`n1_tracks`); only track time inside the Mirnov record counts, and a kept
candidate not wholly inside the record is not assessed (`gate_screen`). An
n = 2 onset also falls back to N1RMS when N2RMS has no sample in its baseline
stretch.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict

import numpy as np
import pandas as pd

from ..config import Paths
from ..mhd import pipeline as mhd_pipeline
from ..mhd import rows as mhd_rows
from ..rwm.features import COLLAPSE_DROP
from ..rwm.labels import ONSET_WINDOW_MS, PRESENT_MS
from . import coverage, equilibrium
from .catalog.states import PRESENT, UNCERTAIN
from .catalog.window import WINDOW_IP_A, _centred_mean
from .interval_tables import validate_intervals
from .verify import NoDataError

UNASSESSED = 4
NO_WALL_FACTOR = 4.0
MIN_MS = 10.0
NOISE_MAD = 6.0
RULE = {
    **equilibrium.RULE,
    "screen": "n1rms_high_beta",
    "beta_n_over_li": NO_WALL_FACTOR,
    "amplitude": "above the flat-top median + 6 median absolute deviations",
    "noise_mad": NOISE_MAD,
    "min_ms": MIN_MS,
    "classification": "uncertain candidates only; the rest is unassessed",
    "database": "onsets only, with no duration or negative coverage inferred",
    "limitations": "N1RMS is not RWM-specific and includes applied-field response",
    "timing": "postprocessed RMS; sampled excursions are not physical growth times",
}

#: `rwm_onsets`: the uncertain stretch before each curated onset (the export's
#: 20 ms, `rwm.labels.ONSET_WINDOW_MS`, an order-of-magnitude wall-time choice).
PRE_ONSET_MS = ONSET_WINDOW_MS
#: N1RMS and its pre-onset baseline are centred means this long; the baseline
#: is the mean of the `SMOOTH_MS` before the uncertain stretch.
SMOOTH_MS = 50.0
#: No present span runs longer than this after its onset ...
MAX_PRESENT_MS = 300.0
#: ... or shorter than one review frame (the v1 minimal slice).
MIN_PRESENT_MS = PRESENT_MS
#: The N1RMS end counts only when the centred mean rises above the baseline
#: within this long of the onset: half the mean's width, by when a rise at the
#: onset has reached it. A later rise is another excursion, not this mode's.
RISE_MS = SMOOTH_MS / 2
#: R1b: the mhd_modes pipeline cache (`labeler.mhd.pipeline`) whose n = 1 tracks
#: gate the screen, pinned so a later cache does not change this table's rule.
MHD_CACHE = "v4"
#: A screen candidate at least this much (in time) inside n = 1 track time
#: within the Mirnov record is dropped.
TRACK_COVER = 0.5
ONSET_RULE = {
    **RULE,
    "database": "curated onsets present from the onset to the measured end",
    "pre_onset_ms": PRE_ONSET_MS,
    "end": ("the first of: the SMOOTH_MS centred mean of the end amplitude "
            "(end_amplitude) back at or below its "
            "mean over [o - PRE_ONSET_MS - SMOOTH_MS, o - PRE_ONSET_MS) after being "
            "above it in [o, o + rise_ms]; the first "
            "beta_N sample after o at most (1 - collapse_drop) of the last one at "
            "or before o; the first |Ip| sample after o below ip_end_a"),
    "end_amplitude": ("N1RMS for an n = 1 onset; N2RMS for an n = 2 onset when the "
                      "shot has it cached with a finite sample in "
                      "[o - PRE_ONSET_MS - SMOOTH_MS, o - PRE_ONSET_MS), else "
                      "N1RMS, the onset's meta record flagged n2_end_on_n1rms"),
    "no_n1rms_response": ("the end amplitude not above the baseline in "
                          "[o, o + rise_ms], or none: present for min_present_ms, "
                          "then uncertain to the first of the other ends"),
    "screen_gate": {
        "drop": ("a screen candidate at least cover of whose time lies inside n = 1 "
                 "track time within the Mirnov record: a rotating mode, not an RWM; "
                 "its time is unassessed"),
        "cover": TRACK_COVER,
        "tracks": (f"the mhd_modes pipeline cache mhd/labeler/{MHD_CACHE}: every "
                   "n = 1 Mirnov track that is not a fixed-frequency line (the "
                   "mhd_n1 parent rule, labeler.mhd.rows), whatever its resonance "
                   "status, P(resonant) or whether the gate modelled it"),
        "spans": ("the tracks' frames padded, merged and kept as the MHD tables' "
                  f"(rows.spans: {mhd_rows.PAD_MS:g} ms pad, {mhd_rows.MERGE_MS:g} ms "
                  f"merge, {mhd_rows.MIN_MS:g} ms minimum)"),
        "record": "the Mirnov record: the cache's mirnov_ms less its mirnov_gaps",
        "assessed": ("dropped, or kept wholly inside the Mirnov record; a kept "
                     "candidate on a shot with no cache record or no Mirnov read, "
                     "or with time outside the record, is not assessed"),
    },
    "smooth_ms": SMOOTH_MS,
    "rise_ms": RISE_MS,
    "collapse_drop": COLLAPSE_DROP,
    "ip_end_a": WINDOW_IP_A,
    "max_present_ms": MAX_PRESENT_MS,
    "min_present_ms": MIN_PRESENT_MS,
    "classification": ("screen candidates and the pre-onset stretches uncertain, "
                       "curated onsets present to the measured end, the rest "
                       "unassessed"),
    "screen_present": "never: screen candidates stay uncertain",
    "windows": ("curated shots outside the cohort: the Ip window "
                "(frames.targets.window_for)"),
}


def onset_table(paths: Paths) -> pd.DataFrame:
    """The latest curated format table, retaining point times and its identity."""
    directory = paths.label_tables / "resistive_wall_mode" / "format"
    path = max(directory.glob("*_format_*.csv"), key=lambda p: p.name, default=None)
    if path is None:
        return pd.DataFrame(columns=["shot", "t_ms", "source"])
    frame = validate_intervals(pd.read_csv(path))
    points = frame[(frame.category == 1) & (frame.t_start == frame.t_end)]
    details = defaultdict(list)
    meta_path = path.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    for source in meta.get("made_from", []):
        raw_path = paths.label_tables / source["raw_file"]
        if not raw_path.is_file():
            continue
        original = pd.read_csv(raw_path)
        for row in original.to_dict("records"):
            key = (int(row["SHOT"]), round(float(row["ONSET_TIME"]), 9))
            details[key].append({"ntor": int(row["NTOR"]),
                                 "mode_type": str(row["MODE_TYPE"]),
                                 "raw_source": source["source"]})
    rows = []
    for row in points.itertuples():
        key = (int(row.shot), round(float(row.t_start), 9))
        attributes = details[key].pop(0) if details[key] else {}
        rows.append({"shot": int(row.shot), "t_ms": float(row.t_start),
                     "source": path.name, **attributes})
    return pd.DataFrame(rows, columns=["shot", "t_ms", "source", "ntor",
                                       "mode_type", "raw_source"])


def onsets(shot: int, paths: Paths) -> list[dict]:
    frame = onset_table(paths)
    found = frame.loc[frame.shot == int(shot)].drop(columns="shot")
    return [{k: int(v) if k == "ntor" else v for k, v in row.items() if pd.notna(v)}
            for row in found.to_dict("records")]


def _nearest(array, target_ms):
    """Nearest finite samples, only within half a native step of their own clock."""
    t = array.x * 1000
    values = np.asarray(array.y[0], dtype=float)
    # sample_runs validates the clock and the scalar shape before sampling.
    equilibrium.sample_runs(t, array.y)
    index = np.searchsorted(t, target_ms)
    left, right = np.maximum(index - 1, 0), np.minimum(index, len(t) - 1)
    index = np.where(np.abs(t[right] - target_ms) < np.abs(t[left] - target_ms),
                     right, left)
    out = values[index].copy()
    out[np.abs(t[index] - target_ms) > 0.5 * np.median(np.diff(t)) + 1e-6] = np.nan
    return out


def detect(shot: int, paths: Paths, window=None):
    from .spans import Found

    info = {"onsets": onsets(shot, paths)}
    try:
        amplitude, measured, inputs = equilibrium.gated(shot, "n1rms", paths, window)
        beta = equilibrium.signal(shot, "betan", paths)
        li = equilibrium.signal(shot, "li", paths)
        inputs.update(betan=dict(beta.attrs), li=dict(li.attrs))
    except (NoDataError, KeyError, ValueError, OSError) as error:
        return Found((), (), {**info, "not_run": str(error)})
    t, y = amplitude.x * 1000, amplitude.y[0]
    beta_y, li_y = _nearest(beta, t), _nearest(li, t)
    valid = np.isfinite(y) & np.isfinite(beta_y) & np.isfinite(li_y) & (li_y > 0)
    for array in (beta, li):
        measured = coverage.intersect_intervals(
            measured, equilibrium.sample_runs(array.x * 1000, array.y)
        )
    inside = np.zeros(len(t), dtype=bool)
    for a, b in measured:
        inside |= (t >= a) & (t <= b)
    baseline = y[inside & valid]
    if not baseline.size:
        return Found((), (), {**info, "not_run": "no overlapping finite screen inputs"})
    median = float(np.median(baseline))
    mad = float(np.median(np.abs(baseline - median)))
    floor = median + NOISE_MAD * max(mad, np.finfo(float).eps * max(1, abs(median)))
    candidate = valid & inside & (y > floor) & (beta_y > NO_WALL_FACTOR * li_y)
    runs = equilibrium.sample_runs(t, np.where(candidate, y, np.nan))
    spans = tuple((a, b, 2) for a, b in runs if b - a >= MIN_MS - 1e-6)
    # Nearest sampling can miss the support of a coarse EFIT sample; do not
    # claim that every N1RMS sample inside the display hull was assessed.
    observed = coverage.intersect_intervals(
        measured, equilibrium.sample_runs(t, np.where(valid, y, np.nan))
    )
    return Found(spans, observed, {**info, "screen": "n1rms_high_beta",
                                  "inputs": inputs,
                                  "amplitude_threshold_g": floor})


def _before(t, y, onset: float):
    """The finite samples of `(t, y)` in the onset's baseline stretch,
    `[onset - PRE_ONSET_MS - SMOOTH_MS, onset - PRE_ONSET_MS)`."""
    return ((t >= onset - PRE_ONSET_MS - SMOOTH_MS) & (t < onset - PRE_ONSET_MS)
            & np.isfinite(y))


def _first_after(t, hit, onset: float) -> float | None:
    """The first time after `onset` at which `hit` holds, or None."""
    found = np.flatnonzero((t > onset) & hit)
    return float(t[found[0]]) if found.size else None


def onset_end(onset: float, *, rms=None, betan=None, ip=None, rise_ms=RISE_MS,
              rms_name="n1rms"):
    """`(end_ms, why, baseline_g, uncertain_to_ms)`: where the mode a curated
    onset starts ends.

    Each input is `(t_ms, y)` or None; `rms` is the mode's RMS amplitude
    (`end_rms`), named `rms_name`. The end is the first of: the `SMOOTH_MS`
    centred mean of `rms` back at or below its pre-onset baseline, the mean of
    the `SMOOTH_MS` before `[onset - PRE_ONSET_MS, onset)`, after it was above
    it in `[onset, onset + rise_ms]` (`rms_name`; `rise_ms` None takes the first
    rise however late, v1's reading); the first beta_N sample at most
    `1 - COLLAPSE_DROP` of the last one at or before the onset ("betan"); the
    first |Ip| sample below the catalog's plasma threshold ("ip");
    `MAX_PRESENT_MS` after the onset ("cap"). An end within `MIN_PRESENT_MS` is
    moved there ("min"). A mean that does not rise in time, or no `rms` before
    the onset, measured nothing of the mode: the end is `MIN_PRESENT_MS` after
    the onset and `uncertain_to_ms` the first of the other ends ("no <rms_name>
    response"); otherwise `uncertain_to_ms` is None. With no input at all the
    onset keeps the v1 minimal slice ("no inputs"). `baseline_g` is None
    without `rms` before the onset."""
    if rms is None and betan is None and ip is None:
        return onset + MIN_PRESENT_MS, "no inputs", None, None
    ends = {"cap": onset + MAX_PRESENT_MS}
    if betan is not None:
        t, y = betan
        at = np.flatnonzero(np.isfinite(y) & (t <= onset))
        if at.size:
            ends["betan"] = _first_after(t, y <= (1 - COLLAPSE_DROP) * y[at[-1]], onset)
    if ip is not None:
        t, y = ip
        ends["ip"] = _first_after(t, np.abs(y) < WINDOW_IP_A, onset)
    ends = {why: end for why, end in ends.items() if end is not None}
    baseline, rose = None, np.array([], dtype=int)
    if rms is not None:
        t, y = rms
        before = _before(t, y, onset)
        if before.any() and len(t) > 1:
            baseline = float(np.mean(y[before]))
            step = float(np.median(np.diff(t)))
            smooth = _centred_mean(y, round(SMOOTH_MS / 2 / step))
            # "Back" below: only once the mean has been above the baseline.
            soon = t >= onset
            if rise_ms is not None:
                soon &= t <= onset + rise_ms
            rose = np.flatnonzero(soon & (smooth > baseline))
            if rose.size:
                back = _first_after(t, smooth <= baseline, t[rose[0]])
                if back is not None:
                    ends[rms_name] = back
    if not rose.size:
        end = onset + MIN_PRESENT_MS
        later = min(ends.values())
        return (end, f"no {rms_name} response", baseline,
                later if later > end else None)
    why = min(ends, key=lambda key: ends[key])
    if ends[why] < onset + MIN_PRESENT_MS:
        return onset + MIN_PRESENT_MS, "min", baseline, None
    return ends[why], why, baseline, None


def end_rms(point: dict, signals: dict) -> tuple[str, dict]:
    """`(name, flag)`: the RMS amplitude an onset's end is measured on, N2RMS
    for an n = 2 onset when the shot has it in `signals` with a finite sample in
    the onset's baseline stretch (`_before`), else N1RMS; `flag` is what the
    onset's meta record gains: nothing for an n = 1 onset, `end_rms` for an
    n = 2 one and `n2_end_on_n1rms` when it falls back to N1RMS."""
    if int(point.get("ntor", 1)) != 2:
        return "n1rms", {}
    if "n2rms" in signals and _before(*signals["n2rms"], float(point["t_ms"])).any():
        return "n2rms", {"end_rms": "n2rms"}
    return "n1rms", {"end_rms": "n1rms", "n2_end_on_n1rms": True}


def end_signals(shot: int, paths: Paths, points) -> dict:
    """`{name: (t_ms, y)}`: the cached N1RMS, beta_N and Ip `onset_end` reads,
    and N2RMS when an onset in `points` is n = 2; a name not cached is left out."""
    names = ["n1rms", "betan", "ip"]
    if any(int(point.get("ntor", 1)) == 2 for point in points):
        names.append("n2rms")
    signals = {}
    for name in names:
        try:
            array = equilibrium.signal(shot, name, paths)
        except (NoDataError, KeyError, ValueError, OSError):
            continue
        signals[name] = (array.x * 1000, np.asarray(array.y[0], dtype=float))
    return signals


def point_end(point: dict, signals: dict, rise_ms=RISE_MS):
    """`onset_end` for a curated onset `point` on its `end_rms` amplitude, and
    the flag its meta record gains."""
    name, flag = end_rms(point, signals)
    ends = {k: signals[k] for k in ("betan", "ip") if k in signals}
    got = onset_end(float(point["t_ms"]), rms=signals.get(name), **ends,
                    rise_ms=rise_ms, rms_name=name)
    return (*got, name, flag)


def _sha256(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _overlap(a: float, b: float, intervals) -> float:
    return sum(max(0.0, min(b, hi) - max(a, lo)) for lo, hi in intervals)


def n1_tracks(shot: int, paths: Paths):
    """`(tracks, record)`: the shot's n = 1 Mirnov tracks in the mhd_modes
    `MHD_CACHE` pipeline cache as the mhd_n1 parent counts them
    (`labeler.mhd.rows`): every n = 1 track that is not a fixed-frequency line,
    whatever its resonance status, P(resonant) or whether the co-rotation gate
    modelled it, its frames spanned as the MHD tables span a track's time
    (`rows.spans`). `tracks` is `(spans, record_ms)`, `record_ms` the Mirnov
    record (`mirnov_ms` less `mirnov_gaps`), or None when the cache has no
    record of the shot, did not read its Mirnov or names no Mirnov stretch
    (nothing assessed). `record` names the cache's files read, with their
    sha256, and the Mirnov status."""
    from .spans import minus

    base = paths.root / "mhd" / "labeler" / MHD_CACHE / str(int(shot))
    files = {"info": base.with_suffix(".json"),
             "tracks": base.parent / f"{base.name}.tracks.parquet",
             "frames": base.parent / f"{base.name}.frames.parquet"}
    if not files["info"].is_file():
        return None, {"status": "no record"}
    info = json.loads(files["info"].read_text())
    record = {"status": info.get("status"),
              "sha256": {k: _sha256(f) for k, f in files.items() if f.is_file()}}
    empty = pd.DataFrame()
    cover = info.get("mirnov_ms")
    if not (mhd_pipeline.Shot(int(shot), info, empty, empty).mirnov and cover):
        return None, record
    gaps = [(float(a), float(b)) for a, b in info.get("mirnov_gaps") or ()]
    seen = minus([(float(cover[0]), float(cover[1]))], gaps)
    if not (files["tracks"].is_file() and files["frames"].is_file()):
        return ([], seen), {**record, "tracks": 0}
    tracks = pd.read_parquet(files["tracks"])
    frames = pd.read_parquet(files["frames"])
    if not len(tracks):
        return ([], seen), {**record, "tracks": 0}
    chosen = (tracks.n.astype(int) == 1) & ~tracks.line.astype(bool)
    times = frames.t_ms[frames.family_id.isin(tracks.family_id[chosen])]
    found = mhd_rows.spans(times.to_numpy(float), np.zeros(len(times)))
    return ([(float(a), float(b)) for a, b, _ in found], seen), {
        **record, "tracks": int(chosen.sum())}


def gate_screen(candidates, tracks) -> tuple[list, list]:
    """`(kept, gated)`: the screen `candidates` `(start, stop, state)` less those
    at least `TRACK_COVER` of whose time lies inside n = 1 track time within the
    Mirnov record (`tracks`: `n1_tracks`'s `(spans, record_ms)`, or None).
    `gated` lists each candidate as `[start, stop, cover, in_record, outcome]`:
    the fractions of its time inside track time within the record and inside
    the record (None when `tracks` is None), and "dropped", "kept" (assessed:
    wholly inside the record) or "not assessed" (kept: no Mirnov read, or time
    outside the record, where a track could not have been seen)."""
    kept, gated = [], []
    for a, b, state in candidates:
        cover = in_record = None
        outcome = "not assessed"
        if tracks is not None and b > a:
            spans, seen = tracks
            inside = coverage.intersect_intervals(spans, seen) if spans and seen else ()
            cover = _overlap(a, b, inside) / (b - a)
            in_record = _overlap(a, b, seen) / (b - a)
            if cover >= TRACK_COVER:
                outcome = "dropped"
            elif in_record >= 1 - 1e-9:
                outcome = "kept"
        if outcome != "dropped":
            kept.append((a, b, state))
        gated.append([a, b, None if cover is None else round(cover, 3),
                      None if in_record is None else round(in_record, 3), outcome])
    return kept, gated


def screen_gate_summary(per_shot: dict, paths: Paths) -> dict:
    """The table meta's `screen_gate`: the gate's outcomes counted over the
    per-shot records (candidates as the screen found them, before the window
    clip) and the MHD cache files read, each file's sha256 and one sha256 over
    the sorted `<shot>.<kind> <sha256>` lines (`spans.run` adds it, so a rerun
    of the table reproduces it)."""
    groups = {"dropped": [], "kept": [], "not assessed": []}
    files = {}
    for shot, info in per_shot.items():
        gate = info.get("screen_gate")
        if gate is None:
            continue
        for kind, sha in gate["mhd"].get("sha256", {}).items():
            files[f"{shot}.{kind}"] = sha
        for a, b, *_, outcome in gate["candidates"]:
            groups[outcome].append((int(shot), b - a))
    lines = "".join(f"{name} {sha}\n" for name, sha in sorted(files.items()))
    names = {"dropped": "dropped", "kept": "kept_assessed",
             "not assessed": "kept_not_assessed"}
    return {"screen_gate": {
        "mhd_source": {
            "cache": str(paths.root / "mhd" / "labeler" / MHD_CACHE),
            "files": len(files),
            "sha256": hashlib.sha256(lines.encode()).hexdigest(),
            "sha256_of": "the sorted '<shot>.<kind> <file sha256>' lines",
            "file_sha256": dict(sorted(files.items())),
        },
        **{names[key]: {"candidates": len(items),
                        "seconds": round(sum(d for _, d in items) / 1000, 3),
                        "shots": len({shot for shot, _ in items})}
           for key, items in groups.items()},
    }}


def detect_onsets(shot: int, paths: Paths, window=None):
    """`rwm_onsets`: `detect`'s screen candidates less those inside n = 1
    Mirnov tracks (`n1_tracks`, `gate_screen`), uncertain, then each curated
    onset's `[o - PRE_ONSET_MS, o)` uncertain, its `[end, uncertain_to)`
    uncertain when its amplitude did not respond, and `[o, end)` present
    (`point_end`), painted in that order, so a present span outlasts another
    onset's uncertain stretches. A dropped candidate's time stays measured:
    unassessed. The onsets are evidence where the screen could not run: their
    stretches join what the method measured. The meta's onsets record each end,
    why it ended, where its uncertain stretch ends and, for n = 2, the amplitude
    measured (`end_rms`); `screen_gate` records each candidate's cover and
    outcome and the MHD files read."""
    from .spans import Found

    screen = detect(shot, paths, window)
    info = dict(screen.info)
    candidates = screen.spans
    if candidates:
        tracks, record = n1_tracks(shot, paths)
        candidates, gated = gate_screen(candidates, tracks)
        info["screen_gate"] = {"mhd": record, "candidates": gated}
    points = info["onsets"]
    if not points:
        return Found(tuple(candidates), screen.measured, info)
    signals = end_signals(shot, paths, points)
    pre, after, present, recorded, seen = [], [], [], [], []
    for point in points:
        onset = float(point["t_ms"])
        end, why, baseline, uncertain_to, name, flag = point_end(point, signals)
        pre.append((onset - PRE_ONSET_MS, onset, UNCERTAIN))
        present.append((onset, end, PRESENT))
        if uncertain_to is not None:
            after.append((end, uncertain_to, UNCERTAIN))
        seen.append((onset - PRE_ONSET_MS, end if uncertain_to is None else uncertain_to))
        recorded.append({**point, "end_ms": round(end, 1), "end": why,
                         **({} if uncertain_to is None
                            else {"uncertain_to_ms": round(uncertain_to, 1)}),
                         **({} if baseline is None
                            else {f"{name}_baseline_g": round(baseline, 3)}),
                         **flag})
    measured = coverage.union_intervals(screen.measured, seen)
    return Found((*candidates, *pre, *after, *present), measured,
                 {**info, "onsets": recorded, "ends_from": sorted(signals)})


def database_windows(paths: Paths) -> pd.DataFrame:
    """Display windows for curated shots missing from the frozen review cohort:
    the Ip window (`frames.targets.window_for`, the catalog's rule 4 on the
    shot's own Ip), or, for a shot with no Ip record, the curated onsets less
    500 ms (from 0 at the latest) to the last onset plus 500 ms.

    These are view extents, not RWM durations or observation coverage.
    """
    from ..frames.targets import window_for  # it imports spans, which imports this

    rows = []
    for shot, frame in onset_table(paths).groupby("shot", sort=True):
        try:
            (lo, hi), _ = window_for(int(shot), paths, None, catalog={})
        except ValueError:
            lo = min(0, int(np.floor(frame.t_ms.min())) - 500)
            hi = int(np.ceil(frame.t_ms.max())) + 500
        rows.append([int(shot), lo, hi])
    return pd.DataFrame(rows, columns=["shot", "window_start_ms", "window_end_ms"])
