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

v5 (round five, delta-Bp; phase4/rwm_inputs.md): N1RMS is an FFT band from
0.5 kHz and is blind to the RWM, which grows in 1-4 ms and rotates below 30 Hz.
The screen and the n = 1 ends now read the RWM's own sensor, the n = 1 delta-Bp
of the midplane probe differences with the coil pickup regressed out
(`labeler.rwm.bp`): a screen candidate is delta-Bp above max(median +
`BP_NOISE_MAD` MAD, `BP_FLOOR_G`) for at least `BP_MIN_MS` (dips shorter than
that bridged) starting with beta_N > `BP_START_LI` l_i (`bp_screen`), and the
Mirnov track gate stays. A curated n = 1 onset is present to delta-Bp back
under the same threshold (`bp_end`); n = 2 onsets, which the probe differences
cannot see, keep v4's N2RMS end.

The v5 follow-up (2026-10-14) checked the rule on the paper's 2003-04 reference
RWMs (Okabayashi et al. 2005) and the Hanson onsets (`round5/rwm/checks`,
`rule_survey.csv`): beta_N > 4 l_i was false at every reference RWM (beta_N/l_i
2.76-5.15 there), the N1RMS rotating-mode check dropped five of the seven (the
crash's N1RMS falls inside the run), and requiring beta_N and l_i in the measured
time cut the terminal ones where EFIT01 ends first. So the measured time is
delta-Bp and the plasma only; the start condition is `BP_START_LI`, untested
(kept, counted) where no EFIT sample lies in the lead-in; N1RMS no longer drops a
candidate, its cover is only recorded (`n1rms_cover`). Below 2021 the four
differences are PTDATA's MPID points; from 2021 they are formed from the single
probes (`bp.differences`).
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
from ..rwm import bp
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
#: v5: the delta-Bp threshold is the measured time's median plus this many
#: median absolute deviations (the N1RMS screen's NOISE_MAD) ...
BP_NOISE_MAD = NOISE_MAD
#: ... and at least this (G): the level above which the check report's 2014
#: bursts stay for 18-32 ms; a quiet shot's MAD (0.05-0.5 G) would otherwise
#: put the threshold inside ordinary flat-top excursions.
BP_FLOOR_G = 5.0
#: A candidate lasts at least this long above the threshold, and a dip below it
#: shorter than this does not end one (the brief's 2-3 ms; the mode grows in
#: 1-4 ms and `bp.SMOOTH_MS` already spreads a single-bin spike over 2 ms).
BP_MIN_MS = 3.0
#: A curated n = 1 onset's delta-Bp response: above the threshold somewhere in
#: [o - PRE_ONSET_MS, o + BP_RISE_MS]. Of the 50 Hanson n = 1 onsets, the first
#: run starts within -21 to +8 ms of 39, at +10.6, +15.5 and +22 ms on three,
#: at +31 to +37 ms on four (another excursion) and not within 100 ms on four;
#: 25 ms keeps the three and is v4's `RISE_MS`.
BP_RISE_MS = 25.0
#: A screen candidate starts with beta_N above this many l_i somewhere in its
#: lead-in [start - PRE_ONSET_MS, start] (nearest EFIT samples). Chosen on the
#: references: the largest one-decimal margin that keeps the seven 2003-04
#: reference RWMs (lowest beta_N/l_i 2.76, 114329) and all 44 Hanson n = 1
#: onsets a run reaches (lowest 2.46, 156785). 4 l_i, v5's first margin, kept
#: none of the references. No EFIT sample in the lead-in (EFIT01 ends before a
#: terminal mode) leaves the condition untested, and the candidate is kept.
BP_START_LI = 2.4
#: The cached inputs `bp.amplitude` reads (the single probes stand in for a
#: missing difference from 2021 on, `bp.differences`).
BP_INPUTS = (*bp.ANGLES_DEG, *bp.SINGLE_NAMES, *bp.COILS)
BP_RULE = {
    **equilibrium.RULE,
    "screen": "bp_n1_high_beta",
    "sensor": ("the n = 1 delta-Bp of MPID067U/097U/127U/157U (labeler.rwm.bp): "
               "0.5 ms bin means less a centred baseline_ms running median, the "
               "I-coil and odd C-coil pickup regressed out per shot (two-pass "
               "trimmed least squares, whole shot, no onset time), a centred "
               "smooth_ms boxcar, a least-squares fit of 2 A cos(phi - phi0)"),
    "baseline_ms": bp.BASELINE_MS,
    "bp_smooth_ms": bp.SMOOTH_MS,
    "pickup_trim_mad": bp.TRIM_MAD,
    "pickup_coils": [*bp.ICOILS, *("-".join(pair) for pair in bp.CCOIL_PAIRS)],
    "measured": "delta-Bp finite and |Ip| at least ip_end_a",
    "probes": ("MPID067U/097U/127U/157U; where a point is missing (2021 on), "
               "difference_gain x (MPI66M{a}F - its partner 180 deg away; "
               "MPI66M340F for 157) on the same scale"),
    "difference_gain": bp.DIFFERENCE_GAIN,
    "threshold": ("max(median + noise_mad median absolute deviations of delta-Bp "
                  "over the measured time, floor_g)"),
    "noise_mad": BP_NOISE_MAD,
    "floor_g": BP_FLOOR_G,
    "min_ms": BP_MIN_MS,
    "runs": ("delta-Bp above the threshold, dips shorter than min_ms inside the "
             "measured time bridged, at least min_ms long"),
    "beta_n_over_li": BP_START_LI,
    "high_beta": ("beta_N > beta_n_over_li l_i (nearest EFIT samples) somewhere in "
                  "[start - pre_onset_ms, start]; untested, and the candidate "
                  "kept, where no EFIT sample lies there"),
    "n1rms_cover": ("recorded, not used: the fraction of the candidate's N1RMS "
                    "samples above N1RMS's median + noise_mad MAD over the "
                    "measured time (it dropped five of the seven paper reference "
                    "RWMs, whose crash falls inside the run)"),
    "ip_end_a": WINDOW_IP_A,
    "pre_onset_ms": PRE_ONSET_MS,
    "classification": "uncertain candidates only; the rest is unassessed",
    "limitations": (
        "delta-Bp also sees a locked or slowly rotating (below ~1 kHz) tearing "
        "mode and residual applied-field pickup; the running median absorbs a "
        "perturbation lasting longer than about half of baseline_ms, so a "
        "saturated mode's run ends by then; n = 2 is invisible to the probe "
        "differences"),
    "timing": "0.5 ms bins; the centred median and boxcar look up to 50 ms ahead",
}
ONSET_RULE = {
    **BP_RULE,
    "database": "curated onsets present from the onset to the measured end",
    "end": ("n = 1: the first of delta-Bp back at or below the screen threshold "
            "(the first sample after max(o, rise) outside the bridged runs) "
            "after it was above it in [o - pre_onset_ms, o + rise_ms]; the first "
            "beta_N sample after o at most (1 - collapse_drop) of the last one at "
            "or before o; the first |Ip| sample after o below ip_end_a"),
    "rise_ms": BP_RISE_MS,
    "no_bp_response": ("delta-Bp not above the threshold in "
                       "[o - pre_onset_ms, o + rise_ms], or no delta-Bp: present "
                       "for min_present_ms, then uncertain to the first of the "
                       "other ends"),
    "n2_end": ("v4's, which the probe differences cannot see: the first of the "
               "SMOOTH_MS centred mean of N2RMS (N1RMS when N2RMS has no finite "
               "sample in [o - pre_onset_ms - n2_smooth_ms, o - pre_onset_ms), "
               "flagged n2_end_on_n1rms) back at or below its mean there after "
               "being above it in [o, o + n2_rise_ms], the beta_N collapse and "
               "the Ip end"),
    "n2_smooth_ms": SMOOTH_MS,
    "n2_rise_ms": RISE_MS,
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
    "collapse_drop": COLLAPSE_DROP,
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


def _other_ends(onset: float, betan=None, ip=None) -> dict:
    """`onset_end`'s ends that are not the amplitude's: the cap, the beta_N
    collapse and the end of the plasma current."""
    ends = {"cap": onset + MAX_PRESENT_MS}
    if betan is not None:
        t, y = betan
        at = np.flatnonzero(np.isfinite(y) & (t <= onset))
        if at.size:
            ends["betan"] = _first_after(t, y <= (1 - COLLAPSE_DROP) * y[at[-1]], onset)
    if ip is not None:
        t, y = ip
        ends["ip"] = _first_after(t, np.abs(y) < WINDOW_IP_A, onset)
    return {why: end for why, end in ends.items() if end is not None}


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
    ends = _other_ends(onset, betan, ip)
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



def bp_signals(shot: int, paths: Paths, names=BP_INPUTS) -> dict:
    """`{name: (t_ms, y)}`: the cached probe differences and coil currents
    `bp.amplitude` reads; a name not cached is left out."""
    signals = {}
    for name in names:
        try:
            array = equilibrium.signal(shot, name, paths)
        except (NoDataError, KeyError, ValueError, OSError):
            continue
        signals[name] = (array.x * 1000, np.asarray(array.y[0], dtype=float))
    return signals


def bp_runs(t, above, finite, min_ms=BP_MIN_MS) -> list[tuple[float, float]]:
    """`(first, last)` sample times of the runs of `above` on the uniform grid
    `t`, a dip shorter than `min_ms` between two runs bridged when every sample
    in it is `finite`, and only runs at least `min_ms` long."""
    above = np.asarray(above, dtype=bool)
    index = np.flatnonzero(above)
    if not index.size:
        return []
    groups = np.split(index, np.flatnonzero(np.diff(index) > 1) + 1)
    merged = [[groups[0][0], groups[0][-1]]]
    for group in groups[1:]:
        last = merged[-1][1]
        if (t[group[0]] - t[last] < min_ms - 1e-9
                and np.all(finite[last + 1:group[0]])):
            merged[-1][1] = group[-1]
        else:
            merged.append([group[0], group[-1]])
    return [(float(t[a]), float(t[b])) for a, b in merged
            if t[b] - t[a] >= min_ms - 1e-6]


def bp_series(shot: int, paths: Paths):
    """`(t_ms, amp_g, measured, threshold_g, info)` of the shot's centred n = 1
    delta-Bp over what it and the plasma (|Ip| >= `WINDOW_IP_A`) measured, with
    the screen threshold; raises `NoDataError` without them."""
    got = bp.amplitude(bp_signals(shot, paths))
    if got is None:
        raise NoDataError(f"shot {shot}: no delta-Bp probes (mpid067u ...)")
    t, y = got["t_ms"], got["amp_g"]
    measured = equilibrium.sample_runs(t, y)
    inputs = {}
    for name in ("ip",):
        array = equilibrium.signal(shot, name, paths)
        inputs[name] = dict(array.attrs)
        values = np.asarray(array.y[0], dtype=float)
        if name == "ip":
            values = np.where(np.abs(values) >= WINDOW_IP_A, values, np.nan)
        measured = coverage.intersect_intervals(
            measured, equilibrium.sample_runs(array.x * 1000, values[None]))
    inside = np.zeros(len(t), dtype=bool)
    for a, b in measured:
        inside |= (t >= a) & (t <= b)
    base = y[inside & np.isfinite(y)]
    if not base.size:
        raise NoDataError(f"shot {shot}: no delta-Bp inside the measured plasma")
    median = float(np.median(base))
    mad = float(np.median(np.abs(base - median)))
    threshold = max(median + BP_NOISE_MAD * mad, BP_FLOOR_G)
    pickup = got["pickup"]
    info = {"inputs": inputs, "probes": got["probes"],
            "bp_median_g": round(median, 4),
            "bp_mad_g": round(mad, 4), "bp_threshold_g": round(threshold, 4),
            "pickup": {"removed": pickup.get("removed", False),
                       "regressors": pickup.get("regressors", []),
                       "variance_removed": [
                           None if c is None else round(c["variance_removed"], 3)
                           for c in pickup.get("channels", [])]}}
    return t, np.where(inside, y, np.nan), measured, threshold, info


def _n1rms_cover(shot: int, paths: Paths, candidates, measured):
    """`[start, stop, cover]` per candidate: the fraction of its N1RMS samples
    above N1RMS's median + `NOISE_MAD` MAD over `measured` (None without N1RMS
    there). Recorded for the reviewer, not used to drop a candidate."""
    try:
        array = equilibrium.signal(shot, "n1rms", paths)
    except (NoDataError, KeyError, ValueError, OSError):
        return [[a, b, None] for a, b, _ in candidates]
    t, y = array.x * 1000, np.asarray(array.y[0], dtype=float)
    inside = np.zeros(len(t), dtype=bool)
    for a, b in measured:
        inside |= (t >= a) & (t <= b)
    base = y[inside & np.isfinite(y)]
    if not base.size:
        return [[a, b, None] for a, b, _ in candidates]
    median = float(np.median(base))
    level = median + NOISE_MAD * float(np.median(np.abs(base - median)))
    records = []
    for a, b, _ in candidates:
        during = (t >= a) & (t <= b) & np.isfinite(y)
        cover = float(np.mean(y[during] > level)) if during.any() else None
        records.append([a, b, None if cover is None else round(cover, 3)])
    return records


def _screen(shot: int, paths: Paths):
    """`(found, series)`: `bp_screen`'s result and the delta-Bp it read,
    `(t_ms, amp_g, threshold_g)`, or None without it."""
    from .spans import Found

    info = {"onsets": onsets(shot, paths), "screen": "bp_n1_high_beta"}
    try:
        t, y, measured, threshold, details = bp_series(shot, paths)
    except (NoDataError, KeyError, ValueError, OSError) as error:
        return Found((), (), {**info, "not_run": str(error)}), None
    info.update(details)
    margin = np.full(len(t), np.nan)
    try:
        beta, li = (_nearest(equilibrium.signal(shot, n, paths), t)
                    for n in ("betan", "li"))
        margin = beta - BP_START_LI * li
    except (NoDataError, KeyError, ValueError, OSError) as error:
        info["start_inputs"] = str(error)
    runs = bp_runs(t, np.isfinite(y) & (y > threshold), np.isfinite(y))
    candidates, start = [], {"passed": 0, "untested": [], "failed": 0}
    for a, b in runs:
        lead = margin[(t >= a - PRE_ONSET_MS) & (t <= a)]
        if not np.isfinite(lead).any():
            start["untested"].append([a, b])
            candidates.append((a, b, UNCERTAIN))
        elif np.nanmax(lead) > 0:
            start["passed"] += 1
            candidates.append((a, b, UNCERTAIN))
        else:
            start["failed"] += 1
    info["bp_runs"] = len(runs)
    info["start"] = start
    if candidates:
        info["n1rms_cover"] = _n1rms_cover(shot, paths, candidates, measured)
    return Found(tuple(candidates), measured, info), (t, y, threshold)


def bp_screen(shot: int, paths: Paths, window=None):
    """The v5 screen: uncertain runs of the n = 1 delta-Bp above the shot's
    threshold that start at beta_N > `BP_START_LI` l_i, or with no EFIT sample
    in the lead-in (untested); the measured time is what delta-Bp and the plasma
    share (`BP_RULE`). `window` is not used: `spans.run` clips to it."""
    return _screen(shot, paths)[0]


def bp_end(onset: float, *, series=None, betan=None, ip=None, rise_ms=BP_RISE_MS):
    """`(end_ms, why, rise_ms, peak_g, uncertain_to_ms)` of a curated n = 1 onset
    on the delta-Bp `series` (`(t_ms, amp_g, threshold_g)`, or None).

    The mode responded when delta-Bp is above the threshold somewhere in
    `[onset - PRE_ONSET_MS, onset + rise_ms]` (`rise_ms` the first such time less
    the onset). Its end is the first of: the first sample after the onset, and
    after that first rise, outside the runs above the threshold (`bp_runs`'
    bridging, no minimum) ("bp_n1"); `_other_ends`. An end within
    `MIN_PRESENT_MS` is moved there ("min"). With no response, or no delta-Bp,
    the end is `MIN_PRESENT_MS` after the onset and `uncertain_to_ms` the first
    of the other ends ("no bp_n1 response", "no bp_n1"); with no input at all,
    "no inputs". `peak_g` is the largest delta-Bp in `[onset, end)`."""
    if series is None and betan is None and ip is None:
        return onset + MIN_PRESENT_MS, "no inputs", None, None, None
    ends = _other_ends(onset, betan, ip)
    rise = None
    why_none = "no bp_n1"
    if series is not None:
        t, y, threshold = series
        finite = np.isfinite(y)
        runs = bp_runs(t, finite & (y > threshold), finite, min_ms=0.0)
        soon = [(a, b) for a, b in runs
                if b >= onset - PRE_ONSET_MS and a <= onset + rise_ms]
        why_none = "no bp_n1 response"
        if soon:
            rise = soon[0][0]
            inside = np.zeros(len(t), dtype=bool)
            for a, b in runs:
                inside |= (t >= a) & (t <= b)
            out = np.flatnonzero((t >= max(onset, rise)) & ~inside)
            if out.size:
                ends["bp_n1"] = float(t[out[0]])
    if rise is None:
        end = onset + MIN_PRESENT_MS
        later = min(ends.values())
        return end, why_none, None, None, later if later > end else None
    why = min(ends, key=lambda key: ends[key])
    end = ends[why]
    if end < onset + MIN_PRESENT_MS:
        end, why = onset + MIN_PRESENT_MS, "min"
    during = (t >= onset) & (t < end) & np.isfinite(y)
    peak = float(np.max(y[during])) if during.any() else None
    return end, why, rise - onset, peak, None


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
    clip), the start condition's (`start`: runs that passed, failed and were
    untested for want of an EFIT sample) and the MHD cache files
    read, each file's sha256 and one sha256 over
    the sorted `<shot>.<kind> <sha256>` lines (`spans.run` adds it, so a rerun
    of the table reproduces it)."""
    groups = {"dropped": [], "kept": [], "not assessed": []}
    start = {"passed": 0, "failed": 0, "untested": []}
    files = {}
    for shot, info in per_shot.items():
        record = info.get("start", {})
        start["passed"] += record.get("passed", 0)
        start["failed"] += record.get("failed", 0)
        start["untested"] += [(int(shot), b - a) for a, b in record.get("untested", ())]
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
        **{names[key]: _counted(items) for key, items in groups.items()},
        "start": {"passed_runs": start["passed"], "failed_runs": start["failed"],
                  "untested": _counted(start["untested"])},
    }}


def _counted(items) -> dict:
    """Candidates, seconds and shots of `(shot, duration_ms)` items."""
    return {"candidates": len(items),
            "seconds": round(sum(d for _, d in items) / 1000, 3),
            "shots": len({shot for shot, _ in items})}


def detect_onsets(shot: int, paths: Paths, window=None):
    """`rwm_onsets` v5: `bp_screen`'s candidates less those inside n = 1 Mirnov
    tracks (`n1_tracks`, `gate_screen`), uncertain, then each curated onset's
    `[o - PRE_ONSET_MS, o)` uncertain, its `[end, uncertain_to)` uncertain when
    it did not respond, and `[o, end)` present, painted in that order, so a
    present span outlasts another onset's uncertain stretches. An n = 1 onset
    ends on delta-Bp (`bp_end`), an n = 2 one on v4's N2RMS (`point_end`). A
    dropped candidate's time stays measured: unassessed. The onsets are
    evidence where the screen could not run: their stretches join what the
    method measured. The meta's onsets record each end, why it ended, where its
    uncertain stretch ends, the delta-Bp rise (ms from the onset) and peak or,
    for n = 2, the amplitude measured (`end_rms`); `start`, `n1rms_cover` and
    `screen_gate` record each candidate's start condition, N1RMS cover, gate
    outcome and the MHD files read."""
    from .spans import Found

    screen, series = _screen(shot, paths)
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
    others = {k: signals[k] for k in ("betan", "ip") if k in signals}
    pre, after, present, recorded, seen = [], [], [], [], []
    for point in points:
        onset = float(point["t_ms"])
        extra = {}
        if int(point.get("ntor", 1)) == 2:
            end, why, baseline, uncertain_to, name, flag = point_end(point, signals)
            extra = {**({} if baseline is None
                        else {f"{name}_baseline_g": round(baseline, 3)}), **flag}
        else:
            end, why, rise, peak, uncertain_to = bp_end(onset, series=series, **others)
            if rise is not None:
                extra = {"bp_rise_ms": round(rise, 1), "bp_peak_g": round(peak, 2)}
        pre.append((onset - PRE_ONSET_MS, onset, UNCERTAIN))
        present.append((onset, end, PRESENT))
        if uncertain_to is not None:
            after.append((end, uncertain_to, UNCERTAIN))
        seen.append((onset - PRE_ONSET_MS, end if uncertain_to is None else uncertain_to))
        recorded.append({**point, "end_ms": round(end, 1), "end": why,
                         **({} if uncertain_to is None
                            else {"uncertain_to_ms": round(uncertain_to, 1)}),
                         **extra})
    measured = coverage.union_intervals(screen.measured, seen)
    ends_from = sorted({*signals, *(("bp_n1",) if series is not None else ())})
    return Found((*candidates, *pre, *after, *present), measured,
                 {**info, "onsets": recorded, "ends_from": ends_from})


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
