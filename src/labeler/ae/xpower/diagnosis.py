"""The whole-window version's diagnosis, and the cross-validation ablation's
summary, from records already made (post hoc: no decision depends on either).

    python -m labeler.ae.xpower.diagnosis --version v3  # models/ae_xpower/v3/diagnosis.md
    python -m labeler.ae.xpower.diagnosis --ablation    # runs/ae_xpower/ablation/summary.md

Nothing here trains or runs a model, and nothing scores the test again: the
test's numbers are `evaluation.json`'s, per shot the gallery's `index.csv`'s,
and the out-of-fold ones the fold records' (`cv.candidate_frames`, each record
checked as `--choose` checks it).

**The diagnosis** (`diagnosis`, `diagnosis_md`): the present share of the
out-of-fold frames and of the test's, and `always`'s F1 (2s / (1 + s) at a
present share s; the test's from its record); recall before and after 2 s
(frames 0-199, and after); the lowest-F1 test shots; how far F1 must rise for
A3 (over `always`) and for A1's paired clause (over SELDNet) if the bootstrap
intervals keep their width; where the chosen threshold sits on the
out-of-fold F1 curve; and one line for the owner's list. Out of fold, recall
before and after 2 s is counted from the frames. The test's is bounded: its
record gives recall over the whole window (`methods`) and over v2's 0-2 s table
(`window_0_2s`), which is not quite the whole-window frames before 2 s (v1's
masks cover it and the source table's window filters it). So the frames of both
are counted without the model (`shot_counts`: the snapshot's states, TokEye's
coverage and the store's, checked against the record's counts), and the model's
true positives before 2 s are the 0-2 s table's, give or take the present frames
the two do not share. The model says the same of a frame in both, as its
receptive field (about 40 ms either side) lies well inside the 20 context
frames either side of each call.

**The ablation's summary** (`ablation_summary`, `summary_md`): the out-of-fold
frontier (F1 and MHD FP per threshold) of each run in `RUNS`, the versions' own
candidates and the ablation's (`cv.ABLATIONS`) on the same folds; a run not yet
there is shown as not run or incomplete. Each own candidate's rows are checked
against its version's `cv/choice.json`. The whole-window runs are also scored
on their frames before 2 s. Then the factors (`FACTORS`): the band (0-250
against 80-250 kHz: same folds, frames and weight) and the training window
(whole window against 0-2 s, both scored on frames before 2 s: same band and
weight), each by its effect on peak F1 and on MHD FP at `COMMON` thresholds
(v2's and v3's chosen). v2 and v3 share their folds (`folds_sha256`); the
window's pairs also differ in the label snapshot (counted over the pool's 0-2 s
frames) and TokEye's record (v1's masks against masks-full). Last, the
out-of-fold MHD false positives at the version's chosen threshold, split by how
much of the frame TokEye's AE band (`data.AE_BINS`, 80-250 kHz, on two chords)
is lit (`AE_LIT`): at least half, where TokEye itself says AE; some; none.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from ...config import Paths, atomic_path, git_sha
from ...events.catalog.states import ABSENT, PRESENT
from . import (
    EVENT,
    LABEL_SNAPSHOTS,
    WHOLE_WINDOW_VERSIONS,
    cv,
    evaluate,
    gallery_dir,
    model_dir,
    read_snapshot,
    tokeye_masks,
    train,
)
from .data import (
    MIN_FRACTION,
    clean_path,
    frame_covered,
    frame_inputs,
    store_rows,
    targets,
    tokeye_frames,
    window_frames,
)

#: Frames before this one lie before 2 s (10 ms frames).
SPLIT_FRAME = 200
#: The ablation's groups: the folds' version, and its window.
GROUPS = {"v3": "whole window", "v2": "0-2 s"}
#: The ablation's runs per group: the version's own candidates, then its
#: ablation candidates.
RUNS = {v: [*train.candidates(v), *cv.ABLATIONS[v]] for v in GROUPS}
#: (factor, what is held, run a, run b, b's frames): the effect is b minus a.
#: a is scored on all its frames; b on all ("all") or its frames before 2 s.
FACTORS = (
    (
        "band",
        "v3's folds, weight 3",
        ("v3", "band80-mhd3"),
        ("v3", "band0-mhd3"),
        "all",
    ),
    (
        "band",
        "v3's folds, weight 10",
        ("v3", "band80-mhd10"),
        ("v3", "band0-mhd10"),
        "all",
    ),
    (
        "band",
        "v2's folds, weight 3",
        ("v2", "band80-mhd3"),
        ("v2", "band0-mhd3"),
        "all",
    ),
    (
        "window",
        "80-250 kHz, weight 3",
        ("v2", "band80-mhd3"),
        ("v3", "band80-mhd3"),
        "before",
    ),
    (
        "window",
        "80-250 kHz, weight 10",
        ("v2", "band80-mhd10"),
        ("v3", "band80-mhd10"),
        "before",
    ),
    (
        "window",
        "0-250 kHz, weight 3",
        ("v2", "band0-mhd3"),
        ("v3", "band0-mhd3"),
        "before",
    ),
)
#: The thresholds the factors' MHD FP is compared at: v2's and v3's chosen.
COMMON = (0.70, 0.80)
#: The MHD split's classes by the share of the frame TokEye's AE band is lit.
AE_LIT = ("at least half", "some", "none")
#: The judge's worst test shots, listed with the lowest-F1 ones.
NAMED_SHOTS = (170792, 170798, 170793, 170667)
WORST = 5


def ablation_root(paths: Paths) -> Path:
    return paths.runs / "ae_xpower" / "ablation"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _counts(frames) -> dict:
    n = evaluate.frame_counts(frames)
    return {k: n[k] for k in ("shots", "scored", "present", "mhd_absent")}


def _windows(paths: Paths, version: str, shots) -> dict[int, tuple[int, int]]:
    """Per shot, the first frame and number of the version's out-of-fold
    frames: the snapshot's window for a whole-window version, else 0-2 s."""
    if version not in WHOLE_WINDOW_VERSIONS:
        return dict.fromkeys(shots, evaluate.EVAL_FRAMES)
    saved = read_snapshot(paths, version)[1]
    return {s: window_frames(saved[s].window) for s in shots}


def _before(frames, windows) -> list:
    """The frames, scored only before 2 s."""
    out = []
    for f in frames:
        first, n = windows[f.shot]
        if len(f.owner) != n:
            raise ValueError(f"shot {f.shot}: {len(f.owner)} frames, its window {n}")
        early = first + np.arange(n) < SPLIT_FRAME
        out.append(replace(f, scored=f.scored & early, said={}))
    return out


def _after(frames, windows) -> list:
    out = []
    for f in frames:
        first, n = windows[f.shot]
        late = first + np.arange(n) >= SPLIT_FRAME
        out.append(replace(f, scored=f.scored & late, said={}))
    return out


def _peak(rows: list[dict]) -> dict:
    scored = [r for r in rows if r["f1"] is not None]
    return max(scored, key=lambda r: (r["f1"], -r["threshold"]))


def _at(rows: list[dict], threshold: float) -> dict:
    (row,) = [r for r in rows if round(r["threshold"] * 100) == round(threshold * 100)]
    return row


def _ae_lit(paths: Paths, version: str, windows) -> dict[int, np.ndarray]:
    """Per shot, the share of each frame's TokEye columns with an AE-band line
    on two chords (`tokeye_frames`' "ae"), from the version's record."""
    masks = tokeye_masks(paths, version)
    return {
        s: tokeye_frames(clean_path(masks, s), first, n)["ae"]
        for s, (first, n) in windows.items()
    }


def mhd_split(frames, lit: dict[int, np.ndarray], threshold: float) -> dict:
    """The MHD frames the owner calls absent, and the model's false positives on
    them at `threshold`, per `AE_LIT` class."""
    out = {}
    for name in AE_LIT:
        frames_n = fp = 0
        for f in frames:
            share = lit[f.shot]
            where = {
                "at least half": share >= MIN_FRACTION,
                "some": (share > 0) & (share < MIN_FRACTION),
                "none": share == 0,
            }[name]
            mask = f.scored & evaluate.mhd_absent(f) & where
            frames_n += int(mask.sum())
            fp += int((mask & (f.prob >= threshold)).sum())
        out[name] = {
            "frames": frames_n,
            "fp": fp,
            "rate": fp / frames_n if frames_n else None,
        }
    return out


def _run_status(paths: Paths, version: str, name: str) -> tuple[str, list[int]]:
    if name in train.candidates(version):
        return "done", []
    where = cv.ablation_dir(paths, version, name)
    lacking = cv._lacking(where, name)
    if len(lacking) == cv.N_FOLDS:
        return "not run", lacking
    return ("incomplete", lacking) if lacking else ("done", [])


def _check_choice(rows: list[dict], choice: dict, name: str) -> None:
    """A version's own candidate's rows are its `cv/choice.json`'s."""
    theirs = [r for r in choice["table"] if r["candidate"] == name]
    keys = ("threshold", "f1", "precision", "recall", "fp_rate_mhd", "cells")
    if [{k: r[k] for k in keys} for r in rows] != [
        {k: r[k] for k in keys} for r in theirs
    ]:
        raise ValueError(f"{name}: its out-of-fold rows differ from choice.json's")


def snapshot_difference(paths: Paths, shots) -> dict:
    """Over the pool's 0-2 s frames: the shots and frames whose owner states
    differ between v2's and v3's label snapshots."""
    old, new = read_snapshot(paths, "v2")[1], read_snapshot(paths, "v3")[1]
    first, n = evaluate.EVAL_FRAMES
    differ = {
        s: int((targets(old[s], first, n) != targets(new[s], first, n)).sum())
        for s in shots
    }
    return {
        "shots": len(differ),
        "shots_differing": sum(v > 0 for v in differ.values()),
        "frames_differing": sum(differ.values()),
        "frames": len(differ) * n,
    }


def ablation_summary(paths: Paths) -> dict:
    """Every run's out-of-fold rows (the whole-window ones also before 2 s),
    the factors' effects, and the MHD split, as the module's docstring says."""
    runs, frames_by, windows_by, lit_by = {}, {}, {}, {}
    for version in GROUPS:
        models = model_dir(paths, version)
        choice = json.loads((models / "cv" / "choice.json").read_text())
        folds = cv.checked_folds(paths, models, version)
        windows_by[version] = _windows(paths, version, sorted(folds.folds))
        threshold = float(choice["threshold"])
        for name in RUNS[version]:
            status, lacking = _run_status(paths, version, name)
            own = name in train.candidates(version)
            if own:
                spec = train.candidate_spec(version, name)
            else:
                spec = cv.ablation_spec(version, name)
            run = {
                "version": version,
                "candidate": name,
                "own": own,
                "band_khz": list(spec["band"]),
                "mhd_weight": spec["mhd_weight"],
                "status": status,
                "lacking": lacking,
                "records": str(
                    (models if own else cv.ablation_dir(paths, version, name))
                    / "cv"
                    / name
                ),
            }
            runs[f"{version}/{name}"] = run
            if status != "done":
                continue
            frames = cv.candidate_frames(paths, version, name)
            frames_by[(version, name)] = frames
            weight = spec["mhd_weight"]
            run["rows"] = cv._rows(name, weight, frames)
            if own:
                _check_choice(run["rows"], choice, name)
            run["frames"] = _counts(frames)
            if version in WHOLE_WINDOW_VERSIONS:
                early = _before(frames, windows_by[version])
                run["rows_before_2s"] = cv._rows(name, weight, early)
                run["frames_before_2s"] = _counts(early)
            if version not in lit_by:
                lit_by[version] = _ae_lit(paths, version, windows_by[version])
            run["mhd_split"] = {
                "threshold": threshold,
                **mhd_split(frames, lit_by[version], threshold),
            }
    factors = []
    for factor, held, a, b, which in FACTORS:
        ra, rb = runs[f"{a[0]}/{a[1]}"], runs[f"{b[0]}/{b[1]}"]
        entry = {"factor": factor, "held": held, "a": a, "b": b, "b_frames": which}
        rows_b = rb.get("rows_before_2s" if which == "before" else "rows")
        if "rows" not in ra or rows_b is None:
            entry["status"] = "not run"
            factors.append(entry)
            continue
        pa, pb = _peak(ra["rows"]), _peak(rows_b)
        entry |= {
            "status": "done",
            "peak_f1": [pa["f1"], pb["f1"]],
            "peak_threshold": [pa["threshold"], pb["threshold"]],
            "d_peak_f1": pb["f1"] - pa["f1"],
            "at": {},
        }
        for t in COMMON:
            xa, xb = _at(ra["rows"], t), _at(rows_b, t)
            entry["at"][f"{t:.2f}"] = {
                "f1": [xa["f1"], xb["f1"]],
                "fp_rate_mhd": [xa["fp_rate_mhd"], xb["fp_rate_mhd"]],
                "d_f1": _diff(xb["f1"], xa["f1"]),
                "d_fp_rate_mhd": _diff(xb["fp_rate_mhd"], xa["fp_rate_mhd"]),
            }
        factors.append(entry)
    pool = sorted(cv.checked_folds(paths, model_dir(paths, "v3"), "v3").folds)
    return {
        "runs": runs,
        "factors": factors,
        "verdict": _verdict(factors),
        "snapshots": snapshot_difference(paths, pool),
        "folds_sha256": {
            v: cv.checked_folds(paths, model_dir(paths, v), v).sha256 for v in GROUPS
        },
        "git_sha": git_sha(),
    }


def _diff(b, a):
    return None if a is None or b is None else b - a


def _verdict(factors: list[dict]) -> dict:
    """Per measure, each factor's mean |effect| over its pairs, and the factor
    with the larger; None while a factor has no pair done."""
    measures = {
        "peak F1": lambda e: [e["d_peak_f1"]],
        "MHD FP": lambda e: [e["at"][f"{t:.2f}"]["d_fp_rate_mhd"] for t in COMMON],
    }
    out = {}
    for measure, get in measures.items():
        means = {}
        for factor in ("band", "window"):
            values = [
                abs(v)
                for e in factors
                if e["factor"] == factor and e["status"] == "done"
                for v in get(e)
                if v is not None
            ]
            means[factor] = float(np.mean(values)) if values else None
        done = all(v is not None for v in means.values())
        larger = max(means, key=lambda k: means[k]) if done else None
        out[measure] = {"mean_abs": means, "larger": larger}
    return out


def _f(value, digits: int = 4) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def _signed(value) -> str:
    return "n/a" if value is None else f"{value:+.4f}"


def _frontier(runs: list[dict], key: str) -> list[str]:
    names = [f"{r['version']} {r['candidate']}" for r in runs]
    lines = [
        "| threshold | " + " | ".join(names) + " |",
        "|---:|" + "---|" * len(runs),
    ]
    for t in train.THRESHOLDS:
        cells = []
        for r in runs:
            if key not in r:
                cells.append(r["status"])
                continue
            row = _at(r[key], float(t))
            cells.append(f"{_f(row['f1'])} / {_f(row['fp_rate_mhd'])}")
        lines.append(f"| {t:.2f} | " + " | ".join(cells) + " |")
    return lines


def summary_md(summary: dict) -> str:
    """The summary as Markdown."""
    runs = summary["runs"]
    lines = [
        "# ae_xpower: the cross-validation ablation",
        "",
        (
            "Post hoc, cross-validation only: out-of-fold frames of the pool's "
            "folds, no test shot. Each version's own candidates from its cv/; the "
            "ablation's from runs/ae_xpower/ablation/<version>/<candidate>/cv/, "
            "trained on the same folds, stop rule, epoch cap, TrainConfig and "
            "data, with only the band and MHD weight changed. v2's and v3's folds "
            "are the same (folds_sha256 "
            + ", ".join(f"{v} {s[:12]}" for v, s in summary["folds_sha256"].items())
            + ")."
        ),
        "",
        "## Runs",
        "",
        "| run | band (kHz) | MHD weight | status | frames | present | MHD absent |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    for key, r in runs.items():
        n = r.get("frames", {})
        band = "-".join(f"{b:g}" for b in r["band_khz"])
        status = r["status"] + (
            f" (folds {r['lacking']} missing)" if r["lacking"] else ""
        )
        own = "" if r["own"] else " (ablation)"
        lines.append(
            f"| {key}{own} | {band} | {r['mhd_weight']:g} | {status} "
            f"| {n.get('scored', '-')} | {n.get('present', '-')} "
            f"| {n.get('mhd_absent', '-')} |"
        )
    for version, window in GROUPS.items():
        group = [r for r in runs.values() if r["version"] == version]
        lines += [
            "",
            f"## Frontier on {version}'s folds ({window}): F1 / MHD FP",
            "",
            *_frontier(group, "rows"),
        ]
        if version in WHOLE_WINDOW_VERSIONS:
            lines += [
                "",
                f"## {version}'s runs scored on their frames before 2 s: F1 / MHD FP",
                "",
                *_frontier(group, "rows_before_2s"),
            ]
    lines += [
        "",
        "## Peak F1",
        "",
        "| run | frames | peak F1 | at | recall there | MHD FP there |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for key, r in runs.items():
        for which, name in (("rows", "all"), ("rows_before_2s", "before 2 s")):
            if which in r:
                p = _peak(r[which])
                lines.append(
                    f"| {key} | {name} | {_f(p['f1'])} | {p['threshold']:.2f} "
                    f"| {_f(p['recall'])} | {_f(p['fp_rate_mhd'])} |"
                )
    at = " | ".join(f"d F1 at {t:.2f} | d MHD FP at {t:.2f}" for t in COMMON)
    lines += [
        "",
        "## Factors (b minus a)",
        "",
        (
            "Band: 0-250 against 80-250 kHz on the same folds and frames. Window: "
            "trained over whole windows (v3's folds) against over 0-2 s (v2's), "
            "both scored on frames before 2 s; these pairs also differ in the "
            "label snapshot and in TokEye's record (v1's masks against "
            "masks-full). Snapshot: "
            f"{summary['snapshots']['shots_differing']} of "
            f"{summary['snapshots']['shots']} pool shots, "
            f"{summary['snapshots']['frames_differing']} of "
            f"{summary['snapshots']['frames']} 0-2 s frames, differ between v2's "
            "and v3's labels."
        ),
        "",
        f"| factor | held | a | b | b's frames | d peak F1 | {at} |",
        "|---|---|---|---|---|---:|" + "---:|---:|" * len(COMMON),
    ]
    for e in summary["factors"]:
        a, b = "/".join(e["a"]), "/".join(e["b"])
        head = f"| {e['factor']} | {e['held']} | {a} | {b} | {e['b_frames']} "
        if e["status"] != "done":
            lines.append(head + "| not run |" + " |" * (2 * len(COMMON)))
            continue
        cells = " | ".join(
            f"{_signed(e['at'][f'{t:.2f}']['d_f1'])} "
            f"| {_signed(e['at'][f'{t:.2f}']['d_fp_rate_mhd'])}"
            for t in COMMON
        )
        lines.append(head + f"| {_signed(e['d_peak_f1'])} | {cells} |")
    lines.append("")
    for measure, v in summary["verdict"].items():
        means = ", ".join(f"{k} {_f(x)}" for k, x in v["mean_abs"].items())
        if v["larger"] is None:
            lines.append(f"- {measure}: not yet decided (a factor has no pair run).")
        else:
            lines.append(
                f"- {measure} moves more with the {v['larger']} "
                f"(mean |effect| over its pairs: {means})."
            )
    lines += [
        "",
        "## MHD false positives by TokEye's AE band",
        "",
        (
            "Out-of-fold frames TokEye flags MHD (a 0-60 kHz line for half the "
            "frame) and the owner calls absent, at the version's chosen threshold, "
            "by the share of the frame with an 80-250 kHz TokEye line on two chords "
            "(at least half: where TokEye itself says AE)."
        ),
        "",
        "| run | threshold | "
        + " | ".join(f"AE band lit {c}: FP / frames (rate)" for c in AE_LIT)
        + " |",
        "|---|---:|" + "---|" * len(AE_LIT),
    ]
    for key, r in runs.items():
        if "mhd_split" not in r:
            continue
        m = r["mhd_split"]
        cells = " | ".join(
            f"{m[c]['fp']} / {m[c]['frames']} ({_f(m[c]['rate'], 3)})" for c in AE_LIT
        )
        lines.append(f"| {key} | {m['threshold']:.2f} | {cells} |")
    lines += ["", f"Made by labeler.ae.xpower.diagnosis at {summary['git_sha']}.", ""]
    return "\n".join(lines)


def write_summary(paths: Paths) -> Path:
    out = ablation_root(paths) / "summary.md"
    text = summary_md(ablation_summary(paths))
    with atomic_path(out) as tmp:
        tmp.write_text(text)
    return out


# ---------------------------------------------------------------- the diagnosis


def shot_counts(paths: Paths, version: str, shot: int, label, source) -> dict:
    """One test shot's frames, counted without the model: the whole-window
    frames the test scores (`evaluate.whole_frames`' rule) and v2's 0-2 s
    table's (`evaluate.shot_frames`' with the source table's window), both over
    frames 0-199 of the shot, and the whole window's before and after 2 s."""
    first, n = window_frames(label.window)
    grid, values, _, _ = store_rows(paths.spectrogram_file(EVENT, shot))
    one = values[:, :1]  # a frame is observed by its columns, not their bins
    observed_w = frame_inputs(one, grid, first, n)[1]
    e_first, e_n = evaluate.EVAL_FRAMES
    observed_e = frame_inputs(one, grid, e_first, e_n)[1]
    with np.load(clean_path(tokeye_masks(paths, version), shot)) as z:
        t_full = np.asarray(z["t_ms"], dtype=np.float64)
    with np.load(clean_path(tokeye_masks(paths), shot)) as z:
        t_v1 = np.asarray(z["t_ms"], dtype=np.float64)
    owner_w, owner_e = targets(label, first, n), targets(label, e_first, e_n)
    whole = np.isin(owner_w, (ABSENT, PRESENT)) & observed_w
    whole &= frame_covered(t_full, first, n)
    table = np.isin(owner_e, (ABSENT, PRESENT)) & observed_e
    table &= frame_covered(t_v1, e_first, e_n) & (targets(source, e_first, e_n) >= 0)
    index = first + np.arange(n)
    early = (index >= e_first) & (index < e_first + e_n)
    before = np.zeros(e_n, dtype=bool)  # the whole window's, on the table's frames
    before[index[early] - e_first] = whole[early]
    present_w, present_e = owner_w == PRESENT, owner_e == PRESENT
    late = index >= SPLIT_FRAME
    return {
        "whole": int(whole.sum()),
        "whole_present": int((whole & present_w).sum()),
        "table": int(table.sum()),
        "table_present": int((table & present_e).sum()),
        "before": int(before.sum()),
        "before_present": int((before & present_e).sum()),
        "after": int((whole & late).sum()),
        "after_present": int((whole & late & present_w).sum()),
        "table_only_present": int((table & ~before & present_e).sum()),
        "before_only_present": int((before & ~table & present_e).sum()),
    }


def _true_positives(recall: float, present: int, where: str) -> int:
    tp = recall * present
    if abs(tp - round(tp)) > 1e-6:
        raise ValueError(f"{where}: recall {recall} of {present} is not a count")
    return round(tp)


def _index_rows(paths: Paths, version: str, chosen: dict) -> dict[int, dict]:
    """The gallery's test rows, refused unless drawn from the chosen model."""
    file = gallery_dir(paths, version) / "index.csv"
    with file.open(newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r["split"] == "test"]
    for r in rows:
        made = (r["version"], r["candidate"], float(r["threshold"]))
        if made != (version, chosen["candidate"], float(chosen["threshold"])):
            raise ValueError(f"{file}: shot {r['shot']} is not the chosen model's")
        if r["snapshot_sha256"] != LABEL_SNAPSHOTS[version]:
            raise ValueError(f"{file}: shot {r['shot']} names another snapshot")
    return {int(r["shot"]): r for r in rows}


def _need(difference: dict, bound: float) -> float:
    """How far the difference must rise for its lower bound to reach `bound`,
    if the interval keeps its width."""
    return bound - difference["low"]


def diagnosis(paths: Paths, version: str) -> dict:
    """The numbers of `diagnosis.md`, as the module's docstring says."""
    if version not in WHOLE_WINDOW_VERSIONS:
        raise ValueError(f"version {version} is not scored over whole windows")
    models = model_dir(paths, version)
    evaluation_file = models / "evaluation.json"
    choice_file = models / "cv" / "choice.json"
    evaluation = json.loads(evaluation_file.read_text())
    choice = json.loads(choice_file.read_text())
    chosen = json.loads((models / "chosen.json").read_text())
    meta = evaluation["meta"]
    if (meta["candidate"], meta["threshold"]) != (
        chosen["candidate"],
        chosen["threshold"],
    ) or meta.get("choice_sha256") != chosen["choice_sha256"]:
        raise ValueError(f"{evaluation_file}: not the chosen model's test")
    name, threshold = chosen["candidate"], float(chosen["threshold"])
    folds = cv.checked_folds(paths, models, version)
    saved = folds.saved

    # Out of fold: exact, from the chosen candidate's fold records.
    frames = cv.candidate_frames(paths, version, name)
    windows = _windows(paths, version, [f.shot for f in frames])
    weight = train.candidate_spec(version, name)["mhd_weight"]
    oof = {}
    for part, subset in (
        ("all", frames),
        ("before", _before(frames, windows)),
        ("after", _after(frames, windows)),
    ):
        (row,) = cv._rows(name, weight, subset, (threshold,))
        n = _counts(subset)
        oof[part] = {**n, "recall": row["recall"], "f1": row["f1"]}
    if (oof["all"]["scored"], oof["all"]["present"]) != (
        choice["frames"]["scored"],
        choice["frames"]["present"],
    ) or oof["all"]["recall"] != choice["chosen_row"]["recall"]:
        raise ValueError(f"{choice_file}: its frames are not the fold records'")

    # The test: its record's cells, and its frames counted without the model.
    test = sorted(s for s, v in folds.split.items() if v == "test")
    source = evaluate.source_table(paths)
    if meta.get("source_sha256") != source.sha256:
        raise ValueError(f"{evaluation_file}: scored with another source table")
    per_shot = {
        s: shot_counts(paths, version, s, saved[s], source.labels[s]) for s in test
    }
    total = {
        k: sum(c[k] for c in per_shot.values()) for k in next(iter(per_shot.values()))
    }
    early_frames = evaluation["window_0_2s"]["frames"]
    if (total["whole"], total["whole_present"]) != (
        evaluation["frames"]["scored"],
        evaluation["frames"]["present"],
    ) or (total["table"], total["table_present"]) != (
        early_frames["scored"],
        early_frames["present"],
    ):
        raise ValueError(f"{evaluation_file}: the counted test frames differ from its")
    model = evaluation["methods"]["ae_xpower"]
    model_early = evaluation["window_0_2s"]["methods"]["ae_xpower"]
    tp_whole = _true_positives(
        model["recall"]["value"], total["whole_present"], "the whole window"
    )
    tp_table = _true_positives(
        model_early["recall"]["value"], total["table_present"], "the 0-2 s table"
    )
    lo = max(0, tp_table - total["table_only_present"])
    hi = min(total["before_present"], tp_table + total["before_only_present"])
    recall_test = {
        "whole": model["recall"]["value"],
        "table_0_2s": model_early["recall"]["value"],
        "before": [lo / total["before_present"], hi / total["before_present"]],
        "after": [
            (tp_whole - hi) / total["after_present"],
            (tp_whole - lo) / total["after_present"],
        ],
    }

    index = _index_rows(paths, version, chosen)
    if sorted(index) != test:
        raise ValueError("the gallery's test shots are not the split's")
    ranked = sorted(test, key=lambda s: float(index[s]["f1_scored"]))
    named = [s for s in NAMED_SHOTS if s in index]
    listed = list(dict.fromkeys([*ranked[:WORST], *named]))
    long = [s for s in test if float(index[s]["window_end_ms"]) > 2000]
    worse = [
        s for s in long if float(index[s]["f1_scored"]) < float(index[s]["f1_0_2s"])
    ]
    diffs = evaluation["differences"]
    table = sorted(
        (r for r in choice["table"] if r["candidate"] == name),
        key=lambda r: r["threshold"],
    )
    return {
        "version": version,
        "candidate": name,
        "threshold": threshold,
        "oof": oof,
        "shot_counts": total,
        "test": {
            "frames": evaluation["frames"],
            "frames_0_2s": early_frames,
            "f1": model["f1"],
            "precision": model["precision"],
            "always_f1": evaluation["methods"]["always"]["f1"],
            "always_f1_0_2s": evaluation["window_0_2s"]["methods"]["always"]["f1"],
            "seldnet_f1": evaluation["methods"]["seldnet"]["f1"],
            "f1_minus_always": diffs["f1_minus_always"],
            "f1_minus_seldnet": diffs["f1_minus_seldnet"],
            "tp_whole": tp_whole,
            "tp_table": tp_table,
            "recall": recall_test,
        },
        "shots": [
            {
                "shot": s,
                "rank": ranked.index(s) + 1,
                "window_end_ms": float(index[s]["window_end_ms"]),
                "f1_whole": float(index[s]["f1_scored"]),
                "f1_0_2s": float(index[s]["f1_0_2s"]),
                "present_before": per_shot[s]["before_present"],
                "present_after": per_shot[s]["after_present"],
            }
            for s in listed
        ],
        "long_window": {"shots": len(long), "lower_whole": len(worse), "of": len(test)},
        "need": {
            "a3": _need(diffs["f1_minus_always"], 0.0),
            "a1_seldnet": _need(
                diffs["f1_minus_seldnet"], evaluate.BAR["f1_vs_seldnet_low"]
            ),
        },
        "curve": [
            {k: r[k] for k in ("threshold", "f1", "precision", "recall", "fp_rate_mhd")}
            for r in table
        ],
        "sources": {
            "evaluation.json": _sha(evaluation_file),
            "cv/choice.json": _sha(choice_file),
            "chosen.json": _sha(models / "chosen.json"),
            "gallery index.csv": _sha(gallery_dir(paths, version) / "index.csv"),
        },
        "git_sha": git_sha(),
    }


def _share(counts: dict, present: str, of: str) -> float:
    return counts[present] / counts[of] if counts[of] else float("nan")


def _always(share: float) -> float:
    return 2 * share / (1 + share)


def _bounds(pair) -> str:
    lo, hi = pair
    return f"{lo:.3f}" if lo == hi else f"{lo:.3f} to {hi:.3f}"


def _ci(e: dict) -> str:
    return f"{e['value']:.3f} [{e['low']:.3f}, {e['high']:.3f}]"


def diagnosis_md(d: dict) -> str:
    """The diagnosis as Markdown: numbers and short statements."""
    oof, test, total = d["oof"], d["test"], d["shot_counts"]
    t = d["threshold"]
    share_oof = _share(oof["all"], "present", "scored")
    share_before = _share(oof["before"], "present", "scored")
    share_after = _share(oof["after"], "present", "scored")
    share_test = test["frames"]["present"] / test["frames"]["scored"]
    share_table = test["frames_0_2s"]["present"] / test["frames_0_2s"]["scored"]
    t_before = total["before_present"] / total["before"]
    t_after = total["after_present"] / total["after"]
    rb, ra = test["recall"]["before"], test["recall"]["after"]
    curve = d["curve"]
    at = next(
        i for i, r in enumerate(curve) if round(r["threshold"] * 100) == round(t * 100)
    )
    peak = max((r for r in curve if r["f1"] is not None), key=lambda r: r["f1"])
    above = curve[at + 1] if at + 1 < len(curve) else None
    need_a3, need_a1 = d["need"]["a3"], d["need"]["a1_seldnet"]
    f1 = test["f1"]["value"]
    lines = [
        f"# ae_xpower {d['version']}: diagnosis",
        "",
        (
            "Post hoc, from records already made: evaluation.json (the one test), "
            "cv/choice.json and the fold records (out of fold), the gallery's "
            "index.csv (per test shot). No model was run and nothing was scored "
            "again; the test's frame counts below are counted without the model "
            "and match evaluation.json's."
        ),
        "",
        f"Model: {d['candidate']} at threshold {t:.2f}.",
        "",
        "## Present share and always",
        "",
        "| frames | scored | present | share | always F1 |",
        "|---|---:|---:|---:|---:|",
        (
            f"| out of fold, whole window | {oof['all']['scored']} "
            f"| {oof['all']['present']} | {share_oof:.3f} | {_always(share_oof):.3f} |"
        ),
        (
            f"| out of fold, before 2 s | {oof['before']['scored']} "
            f"| {oof['before']['present']} | {share_before:.3f} "
            f"| {_always(share_before):.3f} |"
        ),
        (
            f"| out of fold, after 2 s | {oof['after']['scored']} "
            f"| {oof['after']['present']} | {share_after:.3f} "
            f"| {_always(share_after):.3f} |"
        ),
        (
            f"| test, whole window | {test['frames']['scored']} "
            f"| {test['frames']['present']} | {share_test:.3f} "
            f"| {_ci(test['always_f1'])} |"
        ),
        (
            f"| test, before 2 s | {total['before']} | {total['before_present']} "
            f"| {t_before:.3f} | {_always(t_before):.3f} |"
        ),
        (
            f"| test, after 2 s | {total['after']} | {total['after_present']} "
            f"| {t_after:.3f} | {_always(t_after):.3f} |"
        ),
        (
            f"| test, v2's 0-2 s table | {test['frames_0_2s']['scored']} "
            f"| {test['frames_0_2s']['present']} | {share_table:.3f} "
            f"| {_ci(test['always_f1_0_2s'])} |"
        ),
        "",
        (
            f"The test's present share is {share_test:.1%} against {share_oof:.1%} "
            f"out of fold; {d['long_window']['shots']} of {d['long_window']['of']} "
            "test shots have a window past 2 s."
        ),
        "",
        "## Recall before and after 2 s",
        "",
        "| frames | recall | F1 |",
        "|---|---|---|",
        f"| out of fold, whole window | {oof['all']['recall']:.3f} | {oof['all']['f1']:.3f} |",
        (
            f"| out of fold, before 2 s | {oof['before']['recall']:.3f} "
            f"| {oof['before']['f1']:.3f} |"
        ),
        (
            f"| out of fold, after 2 s | {oof['after']['recall']:.3f} "
            f"| {oof['after']['f1']:.3f} |"
        ),
        f"| test, whole window | {test['recall']['whole']:.3f} | {_ci(test['f1'])} |",
        f"| test, v2's 0-2 s table | {test['recall']['table_0_2s']:.3f} | - |",
        f"| test, before 2 s | {_bounds(rb)} | - |",
        f"| test, after 2 s | {_bounds(ra)} | - |",
        "",
        (
            f"Test before and after 2 s: {test['tp_whole']} true positives over "
            f"the whole window and {test['tp_table']} over the 0-2 s table (recall "
            "times present frames); the whole window's frames before 2 s share all "
            f"but {total['table_only_present']} (table only) and "
            f"{total['before_only_present']} (whole window only) of their present "
            "frames with the table, "
            + (
                "so both are exact."
                if rb[0] == rb[1]
                else "so both are bounds (lowest to highest)."
            )
        ),
        "",
        "## Lowest test F1",
        "",
        (
            f"| shot | rank of {d['long_window']['of']} | window end (ms) "
            "| F1 whole window | F1 0-2 s | present before 2 s | present after 2 s |"
        ),
        "|---:|---:|---:|---:|---:|---:|---:|",
        *(
            f"| {s['shot']} | {s['rank']} | {s['window_end_ms']:.0f} "
            f"| {s['f1_whole']:.3f} | {s['f1_0_2s']:.3f} | {s['present_before']} "
            f"| {s['present_after']} |"
            for s in d["shots"]
        ),
        "",
        (
            f"{d['long_window']['lower_whole']} of the {d['long_window']['shots']} "
            "test shots with a window past 2 s score lower over the whole window "
            "than over 0-2 s."
        ),
        "",
        "## What the bar needs on this split",
        "",
        (
            f"- A3: F1 minus always is {_ci(test['f1_minus_always'])}; its lower "
            f"bound reaches 0 if F1 rises by {need_a3:.3f}, to about "
            f"{f1 + need_a3:.3f} (always {test['always_f1']['value']:.3f}), if the "
            "interval keeps its width."
        ),
        (
            f"- The out-of-fold curve's peak F1 is {peak['f1']:.3f} (at "
            f"{peak['threshold']:.2f}), "
            f"{'below' if peak['f1'] < f1 + need_a3 else 'at or above'} the "
            f"{f1 + need_a3:.3f} A3 needs."
        ),
        (
            f"- A1's paired clause: F1 minus SELDNet is "
            f"{_ci(test['f1_minus_seldnet'])}; its lower bound reaches "
            f"{evaluate.BAR['f1_vs_seldnet_low']:.2f} if F1 rises by {need_a1:.3f}, "
            f"to about {f1 + need_a1:.3f} (SELDNet {test['seldnet_f1']['value']:.3f})."
        ),
        "",
        "## The chosen threshold on the out-of-fold curve",
        "",
        "| threshold | F1 | precision | recall | MHD FP |",
        "|---:|---:|---:|---:|---:|",
        *(
            f"| {r['threshold']:.2f}{' (chosen)' if i == at else ''} | {_f(r['f1'])} "
            f"| {_f(r['precision'])} | {_f(r['recall'])} | {_f(r['fp_rate_mhd'])} |"
            for i, r in enumerate(curve)
        ),
        "",
        (
            f"{t:.2f} has out-of-fold F1 {curve[at]['f1']:.4f}, "
            f"{curve[at]['f1'] - cv.F1_MIN:+.4f} from branch 2's {cv.F1_MIN:.2f}"
            + (
                f"; the next threshold, {above['threshold']:.2f}, has {above['f1']:.4f}"
                if above
                else ""
            )
            + f". The curve peaks at {peak['f1']:.4f} at {peak['threshold']:.2f}. "
            f"Out-of-fold recall at {t:.2f} is {curve[at]['recall']:.3f}; the "
            f"test's is {test['recall']['whole']:.3f}."
        ),
        "",
        "## For the owner's list (Task 3.1)",
        "",
        (
            "- v4's test design: stratify the test split by window length, or "
            "score the frames after 2 s separately."
        ),
        "",
        "Records: "
        + ", ".join(f"{k} {v[:12]}" for k, v in d["sources"].items())
        + f". Made by labeler.ae.xpower.diagnosis at {d['git_sha']}.",
        "",
    ]
    return "\n".join(lines)


def write_diagnosis(paths: Paths, version: str) -> Path:
    out = model_dir(paths, version) / "diagnosis.md"
    text = diagnosis_md(diagnosis(paths, version))
    with atomic_path(out) as tmp:
        tmp.write_text(text)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--version", help="a whole-window version: its diagnosis.md")
    mode.add_argument(
        "--ablation", action="store_true", help="runs/ae_xpower/ablation/summary.md"
    )
    args = p.parse_args(argv)
    paths = Paths.from_env()
    try:
        out = (
            write_summary(paths)
            if args.ablation
            else write_diagnosis(paths, args.version)
        )
    except (OSError, ValueError, KeyError) as error:
        p.error(str(error))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
