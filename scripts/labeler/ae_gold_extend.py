#!/usr/bin/env python
"""Label the gold shots that have no Alfven eigenmode label with the SELDnet.

Runs the published ``d3d_ae_activity_seldnet`` (``labeler.ae.model.AeSeldNet``) over
the whole CO2 record of each gold shot (``data/events/gold/gold_shots.csv``) that has
no row in any Alfven eigenmode table, and writes the result as
``data/events/alfven_eigenmode/extend_ae_seldnet/gold_shots.csv`` and its
``.meta.json``.

**Chain** (the model card's own, ``labeler.models.d3d_ae_activity_seldnet.spec``): the
CO2 chords R0, V1, V2, V3 through ``labeler.ae.full.input_signals`` (500 kHz), the
80-250 kHz spectrogram (``labeler.ae.transform.model_input``), 1,024-column windows
with 128 columns of context, sigmoid, then the card's 5-column moving average. The
spectrogram statistics are taken over the assessed span, as the card's
``clip_to_grid`` does.

**Frames.** The owner's 10 ms grid: frame ``k`` covers ``[10k, 10k + 10)`` ms and is
present when at least half of its STFT columns have a smoothed probability >= 0.5
(the protocol of ``docs/labeler/ae_baselines_benchmark.md``), otherwise absent.
Intervals are runs of equal calls, category 1 present, 0 absent. The confidence of an
interval is the mean share of its frames' columns that agree with the call: a model
score, not calibrated.

**Window.** ``[0, plasma end]`` with the end snapped down to a frame edge: the end of
the longest stretch with Ip >= 50 kA (``labeler.events.catalog.window``), else the
last EFIT02 time. Frames outside it are unassessed, not absent.

A shot whose CO2 is on neither the corpus nor the raw cache is skipped and listed in
the sidecar; nothing is fetched (set ``LABELER_NO_FETCH=1`` to make that certain).
The per-shot probabilities are cached in ``--work-dir`` (``<shot>.npz``) so the table
can be rebuilt, or one shot added, without rerunning the others.

    pixi run --frozen -e labeler python scripts/labeler/ae_gold_extend.py [--shots N ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.ae import full
from labeler.ae import transform as tr
from labeler.ae.model import AeSeldNet, AeSeldNetConfig
from labeler.config import Paths, git_dirty, git_sha, sha256_of
from labeler.events import raw
from labeler.events.catalog import window as win
from labeler.events.interval_tables import validate_intervals, write_interval_table
from labeler.events.verify import NoDataError
from labeler.models.d3d_ae_activity_seldnet import spec as S
from labeler.models.runners import torch_pt

CKPT = S.UPSTREAM / S.ARTIFACTS[0]
FRAME_MS = 10
THRESHOLD, MIN_SHARE = 0.5, 0.5
TABLE = Path("alfven_eigenmode/extend_ae_seldnet/gold_shots.csv")
# The first 36 shots were run with three scratch scripts that this file merges without
# changing the chain (their rows are reproduced byte for byte); their sha256:
FIRST_RUN_SCRIPTS = {
    "aeseld.py": "142481b469a413487e765d54e66ebfaad38fd68e0d2fc2822923dae69586972c",
    "run_shots.py": "3e8ff3af489ca63a6ba1f79eabca8113304052bef7cb368fcc515e80b221fa5b",
    "build_table.py": "e21571dbb965f1f4bb24c10dbf10a91d8380f37d1209527d55a5f53957411cf7",
}


def load():
    return torch_pt.load_module(CKPT, lambda cfg: AeSeldNet(AeSeldNetConfig.from_dict(cfg)))


def columns(module, shot, paths, crop_end_ms):
    """-> t_ms of each STFT column, its smoothed probability, record end ms, fs Hz."""
    old = full.CROP_END_MS
    full.CROP_END_MS = float(crop_end_ms)
    try:
        sig, t0_ms, fs_hz = full.input_signals(shot, paths=paths)
    finally:
        full.CROP_END_MS = old
    spec = tr.model_input(sig)  # (4, 348, frames)
    t_ms = full.frame_times(sig.shape[1], t0_ms, fs_hz)
    out = torch_pt.run_windows(
        module,
        np.ascontiguousarray(spec.transpose(0, 2, 1)),
        window=S.WINDOW_FRAMES,
        context=S.CONTEXT_FRAMES,
    )
    prob = S.moving_average(1.0 / (1.0 + np.exp(-out[:, 0].astype(np.float64))))
    return t_ms, prob, float(t0_ms + (sig.shape[1] - 1) * 1000 / fs_hz), fs_hz


def frames(t_ms, prob, n):
    """-> share of columns >= 0.5, mean probability, column count, per 10 ms frame."""
    k = np.floor(t_ms / FRAME_MS).astype(np.int64)
    ok = (k >= 0) & (k < n)
    cnt = np.bincount(k[ok], minlength=n)
    share = np.bincount(k[ok], weights=(prob[ok] >= THRESHOLD).astype(float), minlength=n)
    mean = np.bincount(k[ok], weights=prob[ok], minlength=n)
    return share / np.maximum(cnt, 1), mean / np.maximum(cnt, 1), cnt


def plasma_end(shot, paths):
    try:
        rec = raw.raw_signal(shot, "ip", paths=paths)
    except NoDataError:
        rec = None
    if rec is not None:
        line = win.summarise(shot, rec.x, rec.y[0])
        if line["status"] == "ok":
            tier = rec.attrs.get("tier")
            return float(line["window_end_ms"]), f"Ip >= 50 kA window (labeler.events.catalog.window), {tier} tier"
    z = np.load(paths.root / "mhd" / "efit02" / f"{shot}.npz")
    return float(z["times"][-1]), "last EFIT02 time (no Ip on disk)"


def shots_to_label(paths):
    """Gold shots with no row in any AE table other than this script's own."""
    ae = paths.label_tables / "alfven_eigenmode"
    tables = [
        ae / "review" / "labels.csv",
        *sorted(ae.glob("format/*.csv")),
        *[t for t in sorted(ae.glob("extend_*/**/*.csv")) if t != paths.label_tables / TABLE],
    ]
    have = set()
    for t in tables:
        have |= set(pd.read_csv(t, usecols=["shot"]).shot.astype(int))
    gold = pd.read_csv(paths.label_tables / "gold" / "gold_shots.csv").shot.astype(int)
    return [s for s in gold.tolist() if s not in have]


def run_shot(module, shot, paths, work):
    """Stage 1: cache the shot's column probabilities, or note why it cannot run."""
    if (work / f"{shot}.npz").exists() or (work / f"{shot}.skip.json").exists():
        return
    t0 = time.time()
    if raw.record_tier(shot, "co2", paths=paths) is None:
        reason = "no CO2 record in the corpus (sentinel) or the raw cache"
        (work / f"{shot}.skip.json").write_text(json.dumps({"skipped": reason}))
        print(shot, "SKIP", reason, flush=True)
        return
    end, how = plasma_end(shot, paths)
    t_ms, prob, rec_end, fs = columns(module, shot, paths, end)
    np.savez(
        work / f"{shot}.npz", t_ms=t_ms, prob=prob.astype(np.float32),
        plasma_end_ms=end, plasma_end_how=how, input_end_ms=rec_end, fs_hz=fs,
    )
    print(shot, "end", end, "|", t_ms.size, "columns |", f"{time.time() - t0:.0f} s", flush=True)


def build(shots, paths, work):
    """Stage 2: frame calls -> interval rows and per-shot summary (rerunning is exact)."""
    rows, per_shot, skipped = [], {}, {}
    for shot in shots:
        f, skip = work / f"{shot}.npz", work / f"{shot}.skip.json"
        if skip.exists():
            skipped[str(shot)] = json.loads(skip.read_text())["skipped"]
            continue
        if not f.exists():
            skipped[str(shot)] = "not run"
            continue
        z = np.load(f)
        end = float(z["plasma_end_ms"])
        n = int(end // FRAME_MS)  # whole frames inside [0, end]
        share, mean, cnt = frames(z["t_ms"], z["prob"].astype(np.float64), n)
        assert (cnt > 0).all(), (shot, "frame with no STFT column")
        present = share >= MIN_SHARE
        agree = np.where(present, share, 1.0 - share)
        k, runs = 0, 0
        while k < n:
            j = k
            while j < n and present[j] == present[k]:
                j += 1
            rows.append((shot, int(present[k]), float(FRAME_MS * k), float(FRAME_MS * j),
                         round(float(agree[k:j].mean()), 3)))
            runs += 1
            k = j
        per_shot[str(shot)] = {
            "window_ms": [0, FRAME_MS * n],
            "window_end_rule": str(z["plasma_end_how"]),
            "frames": n,
            "present_frames": int(present.sum()),
            "present_fraction": round(float(present.mean()), 4),
            "n_intervals": runs,
            "frames_where_mean_probability_disagrees_with_call": int(((mean >= THRESHOLD) != present).sum()),
        }
    frame = pd.DataFrame(rows, columns=["shot", "category", "t_start", "t_end", "confidence"])
    validate_intervals(frame)
    return frame, per_shot, skipped


def sidecar(shots, per_shot, skipped, paths):
    frs = np.array([v["present_fraction"] for v in per_shot.values()])
    ends = {s: v["window_end_rule"] for s, v in per_shot.items()}
    efit = sorted(int(s) for s, r in ends.items() if r.startswith("last EFIT02"))
    return {
        "category": "alfven_eigenmode",
        "categories": {"0": "absent", "1": "present"},
        "producer": "ae-seldnet",
        "status": "experimental",
        "review_status": "unreviewed model output; nobody has looked at these frames",
        "made_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "made_by": "scripts/labeler/ae_gold_extend.py (shared repo code only)",
        "scripts": {"scripts/labeler/ae_gold_extend.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "first_run_scripts": FIRST_RUN_SCRIPTS,
        "git_sha": git_sha(),
        "git_sha_full": git_sha(full=True),
        "git_dirty": bool(git_dirty()),
        "git_note": "the worktree had many unrelated uncommitted changes; the files this run imports (labeler/ae/transform.py, labeler/models/d3d_ae_activity_seldnet/spec.py) differ from HEAD only in a docstring path and the UPSTREAM path spelling",
        "shot_list": "gold set: gold shots with no row in any alfven_eigenmode table (saved review/labels.csv, format/*.csv, other extend_*/*.csv) on the day it ran (made_at)",
        "shots_requested": list(shots),
        "shots_labelled": sorted(int(s) for s in per_shot),
        "shots_skipped": skipped,
        "model": {
            "name": "d3d_ae_activity_seldnet",
            "role": "stable and latest AE model in docs/labeler/events/alfven_eigenmode/README.md Models",
            "network": "labeler.ae.model.AeSeldNet (440,514 parameters), run as src/labeler/models/d3d_ae_activity_seldnet/spec.py runs it",
            "checkpoint": str(CKPT),
            "checkpoint_sha256": sha256_of(CKPT),
            "training": "task 7b threeway_sce run, best epoch 6; 180 hand-annotated shots 170659-178879 (training_shots.txt), none of the shots here is among them",
            "card": "src/labeler/models/d3d_ae_activity_seldnet/README.md",
        },
        "inputs": "CO2 interferometer chords R0, V1, V2, V3 over [0, window end] ms through labeler.ae.full.input_signals (the page's own resampler to 500 kHz); 199597, 199603 and 199607 read the corpus (500 kHz), the others the raw cache (1.667 MHz); 195037's CO2 was fetched into the raw cache after the first 36 shots were labeled",
        "band": "80-250 kHz: STFT bins 164:512 of hann 1024, hop 128 at 500 kHz (0.256 ms columns); 250 kHz is the Nyquist of the CO2 record",
        "method": {
            "frame_probability": "sigmoid of the activity logit per STFT column, then the card's 5-column centred moving average, in 1,024-column windows with 128 columns of context (the adapter's own chain, not its 25 ms aggregation)",
            "threshold": THRESHOLD,
            "frame_call": "a 10 ms frame [10k, 10k+10) is present when at least half of its STFT columns have probability >= 0.5, as in docs/labeler/ae_baselines_benchmark.md; otherwise absent",
            "intervals": "runs of equal consecutive frame calls; category 1 present, 0 absent; no smoothing, merging or minimum length applied",
            "confidence": "mean over the interval's frames of the share of the frame's columns that agree with the call; a model score, not calibrated, not a probability the owner would agree",
            "window": f"[0, end] with end the plasma end snapped down to a 10 ms frame edge: the end of the longest stretch with Ip >= 50 kA (labeler.events.catalog.window.summarise, from the cached Ip) for {len(per_shot) - len(efit)} shots; the last EFIT02 time for {', '.join(str(s) for s in efit)}, which have no Ip on disk. The spectrogram statistics are taken over this same span, as the card's clip_to_grid does. Frames outside the window are unassessed, not absent.",
        },
        "per_shot": per_shot,
        "present_fraction_summary": {
            "min": float(frs.min()), "median": float(np.median(frs)), "max": float(frs.max()),
            "pooled_frames": int(sum(v["frames"] for v in per_shot.values())),
            "pooled_present_frames": int(sum(v["present_frames"] for v in per_shot.values())),
        },
        "caveats": [
            "These frames are model output, unreviewed. They are not labels the owner made and do not count as ground truth.",
            "Absent only means absent within the 80-250 kHz band the model sees: an Alfven eigenmode below 80 kHz (BAE, some RSAE) or above 250 kHz is reported as absent, not unknown. Low-frequency modes (LFM) are excluded by construction.",
            "The model over-calls: on its 60 validation shots it marks about two thirds of frames active (precision 0.26 against the raw annotation); against the owner's reviewed labels it scored F1 0.944 (README Models). Expect present frames the owner would not accept, and a long record is 3-4 times the 2 s it trained on (an un-priced approximation in the card).",
            "The model reads CO2 only; it does not see beams, ECH or Ip, so a quiet CO2 record before the beams are on or after they stop reads absent.",
        ],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--work-dir", type=Path, help="probability cache (default $LABELER_ROOT/runs/ae_gold_extend)")
    ap.add_argument("--shots", type=int, nargs="*", help="run only these shots' inference; the table is always rebuilt over all")
    ap.add_argument("--no-write", action="store_true", help="build and validate, write nothing")
    args = ap.parse_args()
    paths = Paths.from_env()
    work = args.work_dir or paths.root / "runs" / "ae_gold_extend"
    work.mkdir(parents=True, exist_ok=True)
    shots = shots_to_label(paths)
    todo = args.shots or shots
    if any(not (work / f"{s}.npz").exists() and not (work / f"{s}.skip.json").exists() for s in todo):
        assert os.environ.get("LABELER_NO_FETCH") == "1", "set LABELER_NO_FETCH=1: this script never fetches"
        module, _ = load()
        for shot in todo:
            run_shot(module, shot, paths, work)
    frame, per_shot, skipped = build(shots, paths, work)
    meta = sidecar(shots, per_shot, skipped, paths)
    if args.no_write:
        print(len(frame), "rows", frame.shot.nunique(), "shots (not written)")
    else:
        out = paths.label_tables / TABLE
        out.parent.mkdir(parents=True, exist_ok=True)
        write_interval_table(frame, out, meta)
        print("wrote", out, len(frame), "rows", frame.shot.nunique(), "shots")
    print(json.dumps(meta["present_fraction_summary"]), "skipped:", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
