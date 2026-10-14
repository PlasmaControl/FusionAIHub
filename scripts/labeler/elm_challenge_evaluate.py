#!/usr/bin/env python
"""Score the ELM methods on the DIII-D ELM detection challenge's answer key.

The key (24 DIII-D hand-timed windows, `elm_challenge_key_b71f991.json`, commit b71f991
of D. Eldon's `elm_detection_challenge`) is held out: none of its shots is one of the 119
reviewed ones and no method is trained or tuned on it. Three views, on the same shots:

* the challenge's own **merit function** (lower is better, unnormalised), run unchanged
  from a pinned clone outside the repository by `elm_challenge_merit.py` in a separate
  venv, on the methods that report spans: no detection, the ELM clock (peak plus its
  half-prominence width, detection-only type), and Eldon's own sample submissions
  (7 shots, read in place from the clone);
* our **boundary matching** (`scoring.events.match`) of detected onsets to the key's ELM
  onsets at 1, 2 and 5 ms, uncertain key events excluded;
* 50 ms **occupancy bins** (an ELM touches the bin), which `elm-ours` (a 1 ms occupancy
  model, not an event detector) can also be scored on: the five frozen fold models of
  run `cv2`, probabilities averaged, at the mean of the fold thresholds.

The review page's own onset detector, **jump-and-hold** (`panels.edge_localized_mode.markers`:
D-alpha summed over the strike-point chords, after D. Eldon's OMFIT module), is scored
once, as it stands (added 2026-10-13, no constant changed for the key). It has no end
either, so each onset is submitted as an `ONSET_SPAN_MS` span, and it enters the merit
function, the onset matching and the crowding view, not the bins. Nothing is fetched
(`LABELER_NO_FETCH` is set): a shot with no strike points reads the spans' filterscope.

The onset heads are the instance-detection baseline (no crowd handling yet): the peaks of
the averaged onset trace of the five `cv2` fold models (`elm-ours-onset-review`, trained
on the reviewed non-crowd span starts) and of the five Smith-only fold models
(`elm-ours-onset-smith`), each at the mean of its folds' onset thresholds. They have no
end head, so each peak is submitted as a fixed `ONSET_SPAN_MS` span; they enter the merit
function and the onset matching, not the bins.

Submissions are written under `$LABELER_ROOT/benchmarks/elm/challenge/submissions/`,
never into the clone. Numbers are "scored with the challenge merit function at commit
b71f991", not official leaderboard results. Output: `outputs/labeler/elm/challenge/
evaluation.json`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import challenge, inputs, onset
from labeler.events.interval_tables import read_label_grid
from labeler.events.panels import edge_localized_mode as elm_panel
from labeler.events.raw import NO_FETCH_ENV
from labeler.events.source_formatters import read_elm_challenge

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/challenge/evaluation.json"
KEY = "edge_localized_mode/raw/elm_challenge_key_b71f991.json"
GRIDS = "edge_localized_mode/format/challenge_shots"
RUN = "round4/elm/cv/cv2"
SMITH_RUN = "round4/elm/smith/cv"
ONSET_SPAN_MS = 1.0  # the onset heads have no end: a peak is submitted as this long
INST_RUN = "round4/elm/inst/inst1"
INST_RECORD = REPO / "outputs/labeler/elm/inst/evaluation.json"
INST_METHODS = ("elm-inst", "elm-inst-alt")
ONSET_METHODS = {
    "elm-ours-onset-review": ("review", "cv2 onset head"),
    "elm-ours-onset-smith": ("smith", "Smith-only onset head"),
}
#: The page's jump-and-hold onsets (`panels.edge_localized_mode.markers`), each submitted
#: as an `ONSET_SPAN_MS` event: it has no end.
JUMP_METHOD = "jump-and-hold"
#: The chords of Eldon's module that the challenge's `signal_list.yml` lists: all but
#: FS07 and FS08.
ALLOWED_CHORDS = frozenset(elm_panel.CHORDS) - {"FS07", "FS08"}
TOLERANCES_MS = (1.0, 2.0, 5.0)
# distance to the nearest other key ELM onset; a lone onset counts as 1e9 ms away
GAP_EDGES_MS = (0.0, 10.0, 20.0, 50.0, 1e12)
NEAR_MS = 20.0
DRAWS = 2000
BUSY_EVENTS = 10
SEED = 20261008
METHODS_INPUT_NOTE = {
    "no-detection": "no input",
    "always-present": "no input",
    "eldon-samples": "the challenge author's own detector; filterscopes_used per file",
    "elm-clock": "one filterscope FS01-FS06 (on the list), else FS07/FS08 (not on it)",
    JUMP_METHOD: "the strike-point filterscopes (FS07/FS08 are not on the list), chosen "
    "from EFIT01 RVSOUT/ZVSOUT (on it) and RVSIN (not on it), and Ip: partly; "
    "per shot in `jump_and_hold_inputs`",
    "elm-ours": "FS02-FS04 (on the list) and the fast interferometer chords "
    "DENV2F/DENV3F (the list has PCDENV2/PCDENV3, a different pointname): partly",
    "elm-ours-onset-review": "as elm-ours (the same inputs): partly",
    "elm-ours-onset-smith": "as elm-ours (the same inputs): partly",
    "elm-inst": "as elm-ours (the same inputs): partly",
    "elm-inst-alt": "as elm-ours (the same inputs): partly",
    "elm-elmo": "needs BES, which is not on the list: not run",
}


def write_folder(folder: Path, records: dict[int, dict]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for old in folder.glob("*.json"):
        old.unlink()
    for shot, record in records.items():
        (folder / f"DIII-D_{shot}.json").write_text(json.dumps(record) + "\n")


def fold_traces(paths: Paths, shots, signals: Path, run: str, threshold_keys):
    """`{shot: ((event, onset) traces per 1 ms from -50 ms, valid starts, valid stops)}`
    of the five frozen fold models of `run`, probabilities averaged, and for each of
    `threshold_keys` (fold.json keys) the mean of the folds' thresholds and the list."""
    import torch

    from labeler.elm import net, train

    torch.set_num_threads(8)
    root = paths.root / run
    record = json.loads((root / "run.json").read_text())
    models, folds = [], []
    for k in range(len(record["folds"])):
        model = net.ElmUNet(dropout=record.get("config", {}).get("dropout", 0.1))
        model.load_state_dict(
            torch.load(
                root / f"fold{k}/model.pt", map_location="cpu", weights_only=True
            )
        )
        models.append(model)
        folds.append(json.loads((root / f"fold{k}/fold.json").read_text()))
    out = {}
    for shot in shots:
        path = signals / f"{shot}.npz"
        if not path.is_file():
            continue
        x = inputs.read_channels(path)
        traces = np.mean([train.predict(m, x, "cpu") for m in models], axis=0)
        out[shot] = (traces, *inputs.valid_intervals(x[inputs.VALID]))
    thresholds = {k: [f[k] for f in folds] for k in threshold_keys}
    return out, {k: float(np.mean(v)) for k, v in thresholds.items()}, thresholds


def onset_spans(trace, threshold, starts, stops, window_ms):
    """The peaks of an onset `trace` (per 1 ms from -50 ms) at or above `threshold`, inside
    the shot's valid input and the key window, as `(n, 2)` spans of `ONSET_SPAN_MS`."""
    t = inputs.GRID0_MS + onset.peaks(trace, threshold) + 0.5
    t = t[covered(starts, stops, t, 0.0) & (t >= window_ms[0]) & (t <= window_ms[1])]
    return np.column_stack([t, t + ONSET_SPAN_MS]) if len(t) else np.zeros((0, 2))


def inst_found(paths: Paths, shots, window):
    """`elm-inst` on the key from the five-model mean traces `elm_inst.py predict`
    wrote: onsets at the mean fold threshold, each spanning the mean fold event
    duration. `elm-inst` uses the decoding its development record chose (gated by
    occupancy or not) before the key was scored; `elm-inst-alt` is the other one."""
    from labeler.elm import inst

    directory = paths.root / INST_RUN
    folds = [
        json.loads((directory / f"fold{k}/fold.json").read_text()) for k in range(5)
    ]
    thr = float(np.mean([f["onset_threshold"] for f in folds]))
    gate = float(np.mean([f["occ_threshold"] for f in folds]))
    duration = float(np.mean([f["event_duration_ms"] for f in folds]))
    primary = json.loads(INST_RECORD.read_text())["primary_decoding"]
    out = {m: {} for m in INST_METHODS}
    traces = {}
    for shot in shots:
        path = directory / "key" / f"{shot}.npz"
        if not path.is_file():
            continue
        z = np.load(path)
        c0, c1 = inputs.valid_intervals(z["valid"])
        traces[shot] = (z["occ"], c0, c1)
        for method, gated in zip(
            INST_METHODS, (primary == "gated", primary != "gated"), strict=True
        ):
            t = inst.onsets(
                z["inst"], thr, z["occ"] if gated else None, gate if gated else None
            )
            t = t[
                covered(c0, c1, t, 0.0)
                & (t >= window[shot][0])
                & (t <= window[shot][1])
            ]
            out[method][shot] = (
                np.column_stack([t, t + duration]) if len(t) else np.zeros((0, 2))
            )
    info = {
        "run": INST_RUN,
        "onset_threshold": thr,
        "occupancy_threshold": gate,
        "event_duration_ms": duration,
        "primary_decoding": primary,
        "decodings": {
            "elm-inst": primary,
            "elm-inst-alt": "ungated" if primary == "gated" else "gated",
        },
        "development_record_sha256": sha256_of(INST_RECORD),
    }
    return out, traces, info


def jump_spans(shot: int, paths: Paths, window_ms) -> tuple[np.ndarray, dict]:
    """The page's jump-and-hold onsets inside the key window, as `ONSET_SPAN_MS` spans,
    and what they were read from: the D-alpha row (`dalpha`), its chords, the side,
    and whether the chords are on the challenge's signal list (the strike-point
    choice also reads RVSIN, which is not)."""
    found = elm_panel.markers(shot, t_range=tuple(window_ms), paths=paths)
    t = np.array(sorted(x for m in found for x in m["t_ms"]), dtype=float)
    d = elm_panel.dalpha(shot, paths=paths)
    strike = d.title == elm_panel.DALPHA_STRIKE_TITLE
    chords = d.caveat.split(" summed", 1)[0] if strike else d.legend
    chords = chords.split("+")
    side = "fallback"
    if strike:
        side = "upper" if any(c.endswith("UP") for c in chords) else "lower"
    info = {
        "dalpha": d.title,
        "chords": chords,
        "side": side,
        "chords_on_signal_list": set(chords) <= ALLOWED_CHORDS,
        "reads_rvsin": strike,
        "n_onsets": len(t),
    }
    return (
        np.column_stack([t, t + ONSET_SPAN_MS]) if len(t) else np.zeros((0, 2))
    ), info


def covered(starts, stops, times, bin_ms):
    """Bins `[t, t + bin_ms)` that lie inside one valid interval."""
    t = np.asarray(times, dtype=float)
    ok = np.zeros(len(t), dtype=bool)
    for a, b in zip(starts, stops, strict=True):
        ok |= (t >= a) & (t + bin_ms <= b)
    return ok


def ours_bins(trace, times, thr):
    """Mean of the 1 ms trace (cell `i` is `[-50 + i, -49 + i)` ms) over each bin >= thr."""
    n = int(challenge.BIN_MS)
    called = np.zeros(len(times), dtype=bool)
    have = np.zeros(len(times), dtype=bool)
    for i, t in enumerate(times):
        a = int(t) - int(inputs.GRID0_MS)
        if 0 <= a and a + n <= len(trace):
            called[i] = trace[a : a + n].mean() >= thr
            have[i] = True
    return called, have


def pooled_with_ci(per_shot: dict[int, dict], rng) -> dict:
    """Pooled counts, precision/recall/F1 and a 95% shot-bootstrap interval of F1."""
    shots = sorted(per_shot)
    tp = np.array([per_shot[s]["tp"] for s in shots])
    fp = np.array([per_shot[s]["fp"] for s in shots])
    fn = np.array([per_shot[s]["fn"] for s in shots])

    def f1(i):
        d = 2 * tp[i].sum() + fp[i].sum() + fn[i].sum()
        return 2 * tp[i].sum() / d if d else np.nan

    point = challenge.bin_metrics_from_counts(tp.sum(), fp.sum(), fn.sum())
    draws = [f1(rng.integers(0, len(shots), len(shots))) for _ in range(DRAWS)]
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return {
        **point,
        "tp": int(tp.sum()),
        "fp": int(fp.sum()),
        "fn": int(fn.sum()),
        "bins": int(sum(per_shot[s]["bins"] for s in shots)),
        "present_bins": int(sum(per_shot[s]["present_bins"] for s in shots)),
        "f1_ci95": [float(lo), float(hi)],
        "n_shots": len(shots),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--venv-python", type=Path)
    ap.add_argument("--clone", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--skip-ours", action="store_true")
    args = ap.parse_args(argv)

    # a benchmark run reads what is stored: a missing input is never fetched
    os.environ.setdefault(NO_FETCH_ENV, "1")
    paths = Paths.from_env()
    root = paths.root
    clone = args.clone or root / "benchmarks/elm/challenge/elm_detection_challenge"
    venv_python = args.venv_python or root / "envs/elm_challenge/bin/python"
    key_path = paths.label_tables / KEY
    cases = {c["shot"]: c for c in read_elm_challenge(key_path)}
    shots = sorted(cases)
    work = root / "benchmarks/elm/challenge"
    submissions = work / "submissions"
    signals = root / "benchmarks/elm/elmo/signals"

    # --- submissions of the methods that report spans -------------------------------
    window = {s: (cases[s]["tstart"], cases[s]["tend"]) for s in shots}
    found: dict[str, dict[int, np.ndarray]] = {
        "no-detection": {s: np.zeros((0, 2)) for s in shots}
    }
    clock_info, clock_missing = {}, {}
    found["elm-clock"] = {}
    for shot in shots:
        try:
            found["elm-clock"][shot], clock_info[shot] = challenge.clock_spans(
                shot, paths
            )
        except Exception as error:  # noqa: BLE001 - reported, never hidden
            clock_missing[shot] = f"{type(error).__name__}: {str(error)[:160]}"
    jump_info, jump_missing = {}, {}
    found[JUMP_METHOD] = {}
    for shot in shots:
        try:
            found[JUMP_METHOD][shot], jump_info[shot] = jump_spans(
                shot, paths, window[shot]
            )
        except Exception as error:  # noqa: BLE001 - reported, never hidden
            jump_missing[shot] = f"{type(error).__name__}: {str(error)[:160]}"
    ours, ours_threshold, fold_thresholds = ({}, None, None)
    smith_traces, onset_thresholds = {}, {}
    if not args.skip_ours:
        ours, thr, fold_thr = fold_traces(
            paths, shots, signals, RUN, ("threshold", "onset_threshold")
        )
        ours_threshold = thr["threshold"]
        fold_thresholds = fold_thr["threshold"]
        onset_thresholds["elm-ours-onset-review"] = {
            "run": RUN,
            "channel": "onset",
            "threshold": thr["onset_threshold"],
            "fold_thresholds": fold_thr["onset_threshold"],
        }
        smith_traces, thr, fold_thr = fold_traces(
            paths, shots, signals, SMITH_RUN, ("threshold",)
        )
        onset_thresholds["elm-ours-onset-smith"] = {
            "run": SMITH_RUN,
            "channel": "onset",
            "threshold": thr["threshold"],
            "fold_thresholds": fold_thr["threshold"],
        }
        for method, (family, _) in ONSET_METHODS.items():
            source = ours if family == "review" else smith_traces
            found[method] = {
                s: onset_spans(
                    traces[1],
                    onset_thresholds[method]["threshold"],
                    c0,
                    c1,
                    window[s],
                )
                for s, (traces, c0, c1) in source.items()
            }
    inst_traces, inst_info = {}, None
    if not args.skip_ours and INST_RECORD.is_file():
        inst_spans, inst_traces, inst_info = inst_found(paths, shots, window)
        found.update(inst_spans)
    for method, comment in (
        ("no-detection", "reports no event on any shot"),
        *(
            (
                method,
                (
                    "FusionAIHub elm-inst: peaks of the five fold models' mean 0.1 ms "
                    "onset heatmap, each event the training shots' median clock "
                    "event duration long"
                ),
            )
            for method in INST_METHODS
            if method in found
        ),
        *(
            (
                method,
                (
                    f"FusionAIHub {label}: peaks of the five fold models' mean onset "
                    f"trace, each submitted as a {ONSET_SPAN_MS:g} ms event (no end head)"
                ),
            )
            for method, (_, label) in ONSET_METHODS.items()
            if method in found
        ),
        (
            "elm-clock",
            (
                "FusionAIHub ELM clock: peaks of one smoothed filterscope "
                "(prominence 0.03, 3 ms apart, width <= 5 ms); each event spans the "
                "peak's half-prominence width"
            ),
        ),
        (
            JUMP_METHOD,
            (
                "FusionAIHub review page ELM onsets: jump-and-hold (rise over the next "
                "0.25 ms over its p99.5 in the Ip flat-top, threshold 0.12, after D. "
                "Eldon's OMFIT module) on D-alpha summed over the strike-point chords; "
                f"each onset submitted as a {ONSET_SPAN_MS:g} ms event (no end)"
            ),
        ),
    ):
        records = {
            shot: challenge.submission(
                shot,
                spans,
                window[shot],
                method=method,
                comment=comment,
                filterscopes_used=[clock_info[shot]["filterscope"]]
                if method == "elm-clock"
                else jump_info[shot]["chords"]
                if method == JUMP_METHOD
                else [],
            )
            for shot, spans in found[method].items()
        }
        write_folder(submissions / method, records)
    samples = {}
    for path in sorted((clone / "samples").glob("*.json")):
        record = json.loads(path.read_text())
        if record["device"] == challenge.DEVICE and record["shot"] in cases:
            samples[record["shot"]] = record
    found["eldon-samples"] = {s: challenge.flag_spans(r) for s, r in samples.items()}

    # --- the merit function, unchanged, from the pinned clone --------------------------
    scores_file = work / "merit_scores.json"
    command = [
        str(venv_python),
        str(REPO / "scripts/labeler/elm_challenge_merit.py"),
        "--clone",
        str(clone),
        "--key",
        str(key_path),
        "--out",
        str(scores_file),
        "--folder",
        f"no-detection={submissions / 'no-detection'}",
        "--folder",
        f"elm-clock={submissions / 'elm-clock'}",
        "--folder",
        f"{JUMP_METHOD}={submissions / JUMP_METHOD}",
        *(
            arg
            for method in (*ONSET_METHODS, *INST_METHODS)
            if method in found
            for arg in ("--folder", f"{method}={submissions / method}")
        ),
        "--folder",
        f"eldon-samples={clone / 'samples'}",
    ]
    subprocess.run(command, check=True)
    merit = json.loads(scores_file.read_text())

    def merit_rows(name, shot_set):
        rows = {
            c: v
            for c, v in merit["folders"][name].items()
            if c not in ("total", "total_not_counting_missing")
            and not v["missing"]
            and _shot_of(v["label"]) in shot_set
        }
        keys = (
            "total",
            "timing_mismatch",
            "false_positive",
            "overcounting",
            "false_negative",
            "incorrect_classification",
            "real_events",
            "detected_events",
        )
        return {
            "n_cases": len(rows),
            **{k: float(sum(v[k] for v in rows.values())) for k in keys},
            "per_case": {
                str(_shot_of(v["label"])): {k: float(v[k]) for k in keys}
                for v in rows.values()
            },
        }

    # the shots every method ran on: the clock had its filterscopes, elm-ours its inputs
    ran = set(found["elm-clock"]) & (set(ours) if ours else set(shots))
    sample_shots = set(samples) & ran
    # split by what the key holds (events of any type), not by any method's result
    busy = {s for s in ran if len(cases[s]["events"]) >= BUSY_EVENTS}
    scopes = {
        "all_graded_cases": ran,
        "on_the_samples_shots": sample_shots,
        f"windows_with_{BUSY_EVENTS}_or_more_key_events": busy,
        f"windows_with_fewer_than_{BUSY_EVENTS}_key_events": ran - busy,
    }
    merit_view = {}
    for name in merit["folders"]:
        merit_view[name] = {}
        for scope, shot_set in scopes.items():
            rows = merit_rows(name, shot_set)
            if rows["n_cases"] == len(shot_set):
                merit_view[name][scope] = rows

    # --- our boundary matching ----------------------------------------------------------
    onset_view = {}
    for method, per_shot in found.items():
        if method == "no-detection":
            continue
        onset_view[method] = {}
        for scope, shot_set in scopes.items():
            if not shot_set <= set(per_shot):
                continue
            onset_view[method][scope] = {}
            for tol in TOLERANCES_MS:
                rows = {
                    s: challenge.onset_match(cases[s], per_shot[s], tol)
                    for s in sorted(shot_set)
                }
                tp = sum(r["tp"] for r in rows.values())
                fp = sum(r["fp"] for r in rows.values())
                fn = sum(r["fn"] for r in rows.values())
                offsets = np.array([d for r in rows.values() for d in r["offsets_ms"]])
                onset_view[method][scope][f"{tol:g} ms"] = {
                    **challenge.bin_metrics_from_counts(tp, fp, fn),
                    "tp": tp,
                    "fp": fp,
                    "fn": fn,
                    "excluded_uncertain": sum(r["excluded"] for r in rows.values()),
                    "median_abs_offset_ms": float(np.median(offsets))
                    if len(offsets)
                    else None,
                    "n_shots": len(rows),
                }

    # where detections fall relative to the key's ELMs, by crowding: all 24 shots
    gap_view = {}
    for method, per_shot in found.items():
        if method == "no-detection" or not set(shots) <= set(per_shot):
            continue
        gaps, offsets = [], []
        for shot in shots:
            onsets, _ = challenge.key_onsets(cases[shot])
            onsets, gap = challenge.nearest_other_gap_ms(onsets)
            starts = np.asarray(per_shot[shot]).reshape(-1, 2)[:, 0]
            gaps.append(gap)
            offsets.append(challenge.nearest_signed_offsets_ms(onsets, starts, NEAR_MS))
        gap, offset = np.concatenate(gaps), np.concatenate(offsets)
        found_2ms = np.abs(offset) <= TOLERANCES_MS[1]
        signed = offset[np.isfinite(offset)]
        gap_view[method] = {
            "by_distance_to_nearest_other_key_elm_ms": [
                {
                    "from_ms": float(lo),
                    "to_ms": float(hi),
                    "key_elms": int(((gap >= lo) & (gap < hi)).sum()),
                    "detection_within_2_ms": int(
                        (found_2ms & (gap >= lo) & (gap < hi)).sum()
                    ),
                }
                for lo, hi in pairwise(GAP_EDGES_MS)
            ],
            f"signed_offset_to_nearest_detection_within_{NEAR_MS:g}_ms": {
                "n_key_elms": len(signed),
                "median_ms": float(np.median(signed)) if len(signed) else None,
            },
        }

    # --- 50 ms occupancy bins -----------------------------------------------------------
    grids = {s: read_label_grid(paths.label_tables / GRIDS / f"{s}.npz") for s in shots}
    bin_calls: dict[str, dict[int, np.ndarray]] = {}
    truth, have_all = {}, {}
    for shot, grid in grids.items():
        times = grid["time_ms"]
        truth[shot] = challenge.bin_truth(grid)
        have_all[shot] = np.ones(len(times), dtype=bool)
        for method, per_shot in found.items():
            if (
                method not in (*ONSET_METHODS, *INST_METHODS, JUMP_METHOD)
                and shot in per_shot
            ):
                bin_calls.setdefault(method, {})[shot] = challenge.bins_touched(
                    per_shot[shot], times
                )
        bin_calls.setdefault("always-present", {})[shot] = np.ones(
            len(times), dtype=bool
        )
        if shot in ours:
            traces, c0, c1 = ours[shot]
            called, have = ours_bins(traces[0], times, ours_threshold)
            bin_calls.setdefault("elm-ours", {})[shot] = called
            have_all[shot] = have & covered(c0, c1, times, challenge.BIN_MS)
        if shot in inst_traces:
            occ, c0, c1 = inst_traces[shot]
            called, have = ours_bins(occ, times, inst_info["occupancy_threshold"])
            bin_calls.setdefault("elm-inst (occupancy)", {})[shot] = called
            have_all[shot] &= have & covered(c0, c1, times, challenge.BIN_MS)
    rng = np.random.default_rng(SEED)
    # every method on the same bins: assessed by the key and, for elm-ours, inside its
    # inputs; a method is scored on a scope only if it ran on all of its shots
    bin_view = {}
    for scope, shot_set in scopes.items():
        bin_view[scope] = {}
        for method, per_shot in bin_calls.items():
            if not shot_set <= set(per_shot):
                continue
            rows = {
                s: challenge.bin_metrics(
                    np.where(have_all[s], truth[s], np.nan), per_shot[s]
                )
                for s in sorted(shot_set)
            }
            bin_view[scope][method] = pooled_with_ci(rows, rng)

    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_sha(full=True),
        "scripts_sha256": {
            p.name: sha256_of(p)
            for p in (
                Path(__file__),
                REPO / "scripts/labeler/elm_challenge_merit.py",
                REPO / "src/labeler/elm/challenge.py",
            )
        },
        "key": {
            "file": KEY,
            "sha256": sha256_of(key_path),
            "n_cases": len(cases),
            "clone_commit": merit["clone_commit"],
        },
        "protocol": "Held out: no method trained or tuned on the key; its shots are not "
        "among the 119 reviewed. Merit scores are the challenge's own, lower is better, "
        "run from the pinned clone; not official leaderboard results.",
        "shots": shots,
        "shots_clock_failed": clock_missing,
        "shots_jump_and_hold_failed": jump_missing,
        "jump_and_hold_inputs": jump_info,
        "shots_without_ours_inputs": sorted(set(shots) - set(ours)) if ours else None,
        "shots_scored_by_every_method": sorted(ran),
        "shots_on_sample_submissions": sorted(sample_shots),
        "clock_inputs": clock_info,
        "inputs_within_signal_list": METHODS_INPUT_NOTE,
        "merit": merit_view,
        "onset_matching": onset_view,
        "onset_by_crowding": gap_view,
        "bins": {
            "bin_ms": challenge.BIN_MS,
            "elm_ours": {
                "run": RUN,
                "threshold": ours_threshold,
                "fold_thresholds": fold_thresholds,
            },
            "methods": bin_view,
        },
        "elm_inst": inst_info,
        "onset_heads": {
            "span_ms": ONSET_SPAN_MS,
            "note": "the onset heads have no end head: each peak is a fixed-length event; "
            "none of the key's shots is in any fold's test list or in the reviewed set; "
            "min peak separation 10 ms (`onset.MIN_SEP_MS`); no crowd handling",
            "methods": onset_thresholds,
            "n_detections": {
                m: {str(s): len(v) for s, v in found[m].items()}
                for m in ONSET_METHODS
                if m in found
            },
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(finite(record), indent=1, default=float) + "\n")
    print(
        json.dumps(
            {
                "merit": {
                    k: {
                        w: {m: v[m] for m in ("n_cases", "total")} for w, v in d.items()
                    }
                    for k, d in merit_view.items()
                },
                "bins": {
                    sc: {
                        k: {m: v[m] for m in ("precision", "recall", "f1")}
                        for k, v in d.items()
                    }
                    for sc, d in bin_view.items()
                },
            },
            indent=1,
            default=float,
        )
    )
    return 0


def finite(value):
    """`value` with every NaN (a metric with an empty denominator) as null."""
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite(v) for v in value]
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _shot_of(label: str) -> int:
    """The shot of a merit-function case label `DIII-D#<shot>: <description>`."""
    return int(label.split("#", 1)[1].split(":", 1)[0])


if __name__ == "__main__":
    raise SystemExit(main())
