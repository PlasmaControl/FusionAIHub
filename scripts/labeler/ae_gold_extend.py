#!/usr/bin/env python
"""Label a shot list with the SELDnet: by default the gold shots with no Alfven eigenmode label.

Runs the published ``d3d_ae_activity_seldnet`` (``labeler.ae.model.AeSeldNet``) over
the whole CO2 record of each listed shot, and writes the result as
``<events>/alfven_eigenmode/extend_ae_seldnet/<name>.csv`` and its ``.meta.json``.

**Lists.** The default list is the gold set (``<events>/gold/gold_shots.csv``), taken
only where a gold shot has no row in any Alfven eigenmode table but this producer's own,
and the default table is ``gold_shots.csv``. ``--shots-file`` names another list (one
shot a line, or a CSV with a ``shot`` column). Every shot on that list is run, labelled
elsewhere or not: on the page a saved label outranks an extend table anyway. ``--name``
is the table's stem, by default the list file's stem.

**Chain** (the model card's own, ``labeler.models.d3d_ae_activity_seldnet.spec``): the
CO2 chords R0, V1, V2, V3 through ``labeler.ae.full.input_signals`` (500 kHz), the
80-250 kHz spectrogram (``labeler.ae.transform.model_input``), 1,024-column windows
with 128 columns of context, sigmoid, then the card's 5-column moving average. The
spectrogram statistics are taken over the assessed span, as the card's
``clip_to_grid`` does. ``--device cuda`` runs the network on a GPU in IEEE float32 (no
TF32); the rest of the chain stays on the CPU. ``--chain gpu`` runs the whole chain in
torch on ``--device`` instead (``labeler.ae.gpu``: the same resample, STFT, clip,
standardisation, band and windows, checked against the CPU chain in
``tests/labeler/test_ae_gpu.py``), with ``--workers`` DataLoader processes reading the
next shots' CO2 and window ends while the GPU computes; it writes the same files.

**Frames.** The owner's 10 ms grid: frame ``k`` covers ``[10k, 10k + 10)`` ms and is
present when at least half of its STFT columns have a smoothed probability >= 0.5
(the protocol of ``docs/labeler/ae_baselines_benchmark.md``), otherwise absent.
Intervals are runs of equal calls, category 1 present, 0 absent. The confidence of an
interval is the mean share of its frames' columns that agree with the call: a model
score, not calibrated.

**Window.** ``[0, plasma end]`` with the end snapped down to a frame edge: the end of
the longest stretch with Ip >= 50 kA (``labeler.events.catalog.window``), else the
last EFIT02 time. A shot with neither is skipped. If the CO2 record stops giving STFT
columns before that end, the window ends at the last frame that has them. Frames
outside the window are unassessed, not absent.

A shot whose CO2 is on neither the corpus nor the raw cache is skipped and listed in
the sidecar; nothing is fetched (set ``LABELER_NO_FETCH=1`` to make that certain).
The per-shot probabilities are cached in ``--work-dir`` (``<shot>.npz``, written
atomically) so the table can be rebuilt, or one shot added, without rerunning the
others. ``--build-only`` builds the table from that cache and never loads the model.

    pixi run --frozen -e labeler python scripts/labeler/ae_gold_extend.py [--shots N ...]
    python scripts/labeler/ae_gold_extend.py --shots-file $LABELER_LABEL_TABLES/silver/silver_shots.csv \\
        --work-dir DIR --chain gpu --device cuda --no-write       # the list, in one GPU job
    python scripts/labeler/ae_gold_extend.py --shots-file $LABELER_LABEL_TABLES/silver/silver_shots.csv \\
        --work-dir DIR --build-only                                # then the table, from the cache

The silver run's jobs: ``scripts/labeler/silver_ae_seldnet.sbatch``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from labeler.ae import full, gpu
from labeler.ae import transform as tr
from labeler.ae.model import AeSeldNet, AeSeldNetConfig
from labeler.ae.xpower.data import seldnet_split
from labeler.config import Paths, atomic_path, events_ref, git_dirty, git_sha, sha256_of
from labeler.events import raw
from labeler.events.catalog import window as win
from labeler.events.interval_tables import validate_intervals, write_interval_table
from labeler.events.verify import NoDataError
from labeler.models.d3d_ae_activity_seldnet import spec as S
from labeler.models.runners import torch_pt

CKPT = S.UPSTREAM / S.ARTIFACTS[0]
FRAME_MS = 10
THRESHOLD, MIN_SHARE = 0.5, 0.5
REPO = Path(__file__).resolve().parents[2]
GOLD = "gold_shots"
GOLD_LIST = Path("gold/gold_shots.csv")  # under the label tables


def table_path(name: str = GOLD) -> Path:
    """The table for a list stem, under the label tables."""
    return Path("alfven_eigenmode/extend_ae_seldnet") / f"{name}.csv"


TABLE = table_path()
# The first 36 shots were run with three scratch scripts that this file merges without
# changing the chain (their rows are reproduced byte for byte); their sha256:
FIRST_RUN_SCRIPTS = {
    "aeseld.py": "142481b469a413487e765d54e66ebfaad38fd68e0d2fc2822923dae69586972c",
    "run_shots.py": "3e8ff3af489ca63a6ba1f79eabca8113304052bef7cb368fcc515e80b221fa5b",
    "build_table.py": "e21571dbb965f1f4bb24c10dbf10a91d8380f37d1209527d55a5f53957411cf7",
}
# What the gold table's sidecar says about its own run (2026-10-12).
GOLD_SHOT_LIST = "gold set: gold shots with no row in any alfven_eigenmode table (saved review/labels.csv, format/*.csv, other extend_*/*.csv) on the day it ran (made_at)"
GOLD_GIT_NOTE = "the worktree had many unrelated uncommitted changes; the files this run imports (labeler/ae/transform.py, labeler/models/d3d_ae_activity_seldnet/spec.py) differ from HEAD only in a docstring path and the UPSTREAM path spelling"
GOLD_INPUTS = "CO2 interferometer chords R0, V1, V2, V3 over [0, window end] ms through labeler.ae.full.input_signals (the page's own resampler to 500 kHz); 199597, 199603 and 199607 read the corpus (500 kHz), the others the raw cache (1.667 MHz); 195037's CO2 was fetched into the raw cache after the first 36 shots were labeled"
TRAINING = "task 7b threeway_sce run, best epoch 6; 180 hand-annotated shots 170659-178879 (training_shots.txt)"
ENV_NOTE = f"torch {torch.__version__}, numpy {np.__version__}, python {sys.version.split()[0]}"
GPU_CHAIN = "chain gpu (labeler.ae.gpu)"
SKIP_NO_CO2 = "no CO2 record in the corpus (sentinel) or the raw cache"
SKIP_NO_END = ("no window end: no Ip >= 50 kA window could be measured from the Ip on disk, "
               "and no EFIT02 file (mhd/efit02)")
# Appended to an Ip window end read from an untagged cached Ip. Until 2026-10-16 it was
# appended to every cached Ip (raw.raw_signal's attrs never carry the group's resolver
# tag); build() drops it where the cached Ip is tagged.
UNTAGGED_IP = "; the cached Ip's clock is untagged (archive or native times, one 25 ms step apart)"
#: The EFIT02 fallback's reason as written before 2026-10-16, and as written since.
OLD_EFIT_REASON = "last EFIT02 time (the Ip on disk has no >= 50 kA window: "
EFIT_REASON = ("last EFIT02 time (the Ip >= 50 kA window could not be measured from the Ip on disk, "
               "so the window falls back to the EFIT02 end; catalog.window status: ")


class OnDevice(torch.nn.Module):
    """The network on `device` behind `torch_pt.run_windows`, which hands it CPU
    tensors and reads what it returns with `.numpy()`."""

    def __init__(self, module, device):
        super().__init__()
        self.inner = module.to(device)
        self.device = torch.device(device)

    def forward(self, x):
        return self.inner(x.to(self.device)).cpu()


def load(device="cpu", wrap=True):
    """The network and its checkpoint blob. On a GPU, wrapped for torch_pt.run_windows
    (CPU tensors in and out) unless `wrap` is false (the gpu chain keeps tensors there)."""
    module, blob = torch_pt.load_module(CKPT, lambda cfg: AeSeldNet(AeSeldNetConfig.from_dict(cfg)))
    if torch.device(device).type == "cpu":
        return module, blob
    # The CPU chain is IEEE float32; cuDNN would run the convolutions and the GRUs in
    # TF32 (a 10-bit mantissa) by default. Only the new API: reading the legacy
    # allow_tf32 flags after setting these raises.
    torch.backends.cudnn.conv.fp32_precision = "ieee"
    torch.backends.cudnn.rnn.fp32_precision = "ieee"
    torch.backends.cuda.matmul.fp32_precision = "ieee"
    if not wrap:
        return module.to(device).eval(), blob
    return OnDevice(module, device).eval(), blob


def device_note(device):
    device = torch.device(device)
    if device.type == "cuda":
        return f"cuda ({torch.cuda.get_device_name(device)}), IEEE float32"
    return "cpu"


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


def ip_clock_tag(shot, paths):
    """The raw cache's tag on its Ip group (`resolver`: archive or fdp), or None: untagged,
    or no cached Ip. Metadata only."""
    try:
        with h5py.File(raw.cache_path(shot, paths=paths), "r") as f:
            tag = f["ip"].attrs.get("resolver") if "ip" in f else None
    except OSError:
        return None
    return tag.decode() if isinstance(tag, bytes) else None if tag is None else str(tag)


def plasma_end(shot, paths):
    """-> (end ms, how), or None when there is neither an Ip window nor an EFIT02 time."""
    try:
        rec = raw.raw_signal(shot, "ip", paths=paths)
    except NoDataError:
        rec = None
    if rec is not None:
        try:
            line = win.summarise(shot, rec.x, rec.y[0])
        except ValueError:  # an Ip record it cannot measure
            line = {"status": "unmeasurable"}
        if line["status"] == "ok":
            tier = rec.attrs.get("tier")
            how = f"Ip >= 50 kA window (labeler.events.catalog.window), {tier} tier"
            if tier == "cache" and ip_clock_tag(shot, paths) not in ("archive", "fdp"):
                # labeler.events.equilibrium refuses such an Ip; here it only sets the end.
                how += UNTAGGED_IP
            return float(line["window_end_ms"]), how
    try:
        with np.load(paths.root / "mhd" / "efit02" / f"{shot}.npz") as z:
            end = float(z["times"][-1])
    except (OSError, KeyError, IndexError, ValueError):
        return None
    if rec is None:
        return end, "last EFIT02 time (no Ip on disk)"
    return end, f"{EFIT_REASON}{line['status']})"


def read_list(path) -> pd.DataFrame:
    """A shot list, each shot once in listed order: a CSV with a `shot` column (its
    other columns kept), or shots separated by blanks, commas or newlines."""
    text = Path(path).read_text()
    lines = [line for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#")]
    if lines and "shot" in [c.strip() for c in lines[0].split(",")]:
        frame = pd.read_csv(path)
    else:
        frame = pd.DataFrame({"shot": [int(t) for line in lines for t in line.replace(",", " ").split()]})
    frame["shot"] = frame.shot.astype(int)
    return frame.drop_duplicates("shot").reset_index(drop=True)


def shots_to_label(paths, listed=None):
    """A `--shots-file` list whole; else the gold shots with no row in any AE table
    but this producer's own (`extend_ae_seldnet/`)."""
    if listed is not None:
        return listed.shot.tolist()
    ae = paths.label_tables / "alfven_eigenmode"
    own = ae / "extend_ae_seldnet"
    tables = [
        ae / "review" / "labels.csv",
        *sorted(ae.glob("format/*.csv")),
        *[t for t in sorted(ae.glob("extend_*/**/*.csv")) if own not in t.parents],
    ]
    have = set()
    for t in tables:
        have |= set(pd.read_csv(t, usecols=["shot"]).shot.astype(int))
    gold = pd.read_csv(paths.label_tables / GOLD_LIST).shot.astype(int)
    return [s for s in gold.tolist() if s not in have]


def skip(work, shot, reason):
    with atomic_path(work / f"{shot}.skip.json") as tmp:
        tmp.write_text(json.dumps({"skipped": reason}))
    print(shot, "SKIP", reason, flush=True)


def done(work, shot):
    """Whether the shot has its npz or its skip file: it is not rerun."""
    return (work / f"{shot}.npz").exists() or (work / f"{shot}.skip.json").exists()


def input_failed(error):
    return f"CO2 input failed: {type(error).__name__}: {error}"


def save(work, shot, t_ms, prob, end, how, rec_end, fs, tier, run_on):
    # Written whole or not at all: a slice job reads every other slice's files.
    with atomic_path(work / f"{shot}.npz") as tmp, open(tmp, "wb") as fh:
        np.savez(
            fh, t_ms=t_ms, prob=prob.astype(np.float32),
            plasma_end_ms=end, plasma_end_how=how, input_end_ms=rec_end, fs_hz=fs,
            co2_tier=tier, run_on=run_on,
        )


def run_shot(module, shot, paths, work, device="cpu"):
    """Stage 1: cache the shot's column probabilities, or note why it cannot run."""
    if done(work, shot):
        return
    t0 = time.time()
    tier = raw.record_tier(shot, "co2", paths=paths)
    if tier is None:
        skip(work, shot, SKIP_NO_CO2)
        return
    found = plasma_end(shot, paths)
    if found is None:
        skip(work, shot, SKIP_NO_END)
        return
    end, how = found
    try:
        t_ms, prob, rec_end, fs = columns(module, shot, paths, end)
    except (NoDataError, ValueError) as error:
        skip(work, shot, input_failed(error))
        return
    save(work, shot, t_ms, prob, end, how, rec_end, fs, tier, f"{device_note(device)}; {ENV_NOTE}")
    print(shot, "end", end, "|", t_ms.size, "columns |", f"{time.time() - t0:.0f} s", flush=True)


class Co2Inputs(Dataset):
    """--chain gpu's loader side, one shot an item, in a DataLoader worker: run_shot's
    steps up to the CO2 resample (the CO2 tier, the window end, the record read and cut:
    gpu.read_co2). Returns the skip reason run_shot would write, or the cut chords as a
    tensor (a worker hands a tensor over in shared memory, an array through a pipe) and
    plain numbers."""

    def __init__(self, shots, paths):
        self.shots, self.paths = list(shots), paths

    def __len__(self):
        return len(self.shots)

    def __getitem__(self, i):
        shot, t0 = self.shots[i], time.perf_counter()
        tier = raw.record_tier(shot, "co2", paths=self.paths)
        if tier is None:
            return {"shot": shot, "skip": SKIP_NO_CO2}
        found = plasma_end(shot, self.paths)
        if found is None:
            return {"shot": shot, "skip": SKIP_NO_END}
        end, how = found
        try:
            co2 = gpu.read_co2(shot, paths=self.paths, end_ms=end)
        except (NoDataError, ValueError) as error:
            return {"shot": shot, "skip": input_failed(error)}
        co2["chords"] = torch.from_numpy(co2["chords"])
        return {"shot": shot, "tier": tier, "end": end, "how": how, "co2": co2,
                "read_s": time.perf_counter() - t0}


def as_is(item):
    """The DataLoader's collate: the item untouched (its default would make tensors of
    the numbers)."""
    return item


def stamp():
    """Local wall-clock time as nvidia-smi's csv writes it, for the job to cut its samples."""
    return datetime.now(UTC).astimezone().strftime("%Y/%m/%d %H:%M:%S.%f")[:-3]


def run_gpu(todo, paths, work, device, workers):
    """Stage 1 with the whole chain in torch on `device` (labeler.ae.gpu): `workers`
    DataLoader processes read and cut the next shots' CO2 (Co2Inputs) while this process
    resamples, transforms and runs the network on the device, then writes each shot's npz
    or skip file as run_shot does. Logs each shot's stage times (s; the GPU stages
    synchronised), the loop's start and end, and when the first shot is written (the
    CUDA start-up is behind it)."""
    device = torch.device(device)
    todo = [s for s in todo if not done(work, s)]
    if not todo:
        return
    loader = DataLoader(Co2Inputs(todo, paths), batch_size=None, collate_fn=as_is, num_workers=workers,
                        prefetch_factor=2 if workers else None, pin_memory=device.type == "cuda",
                        in_order=False)
    items = iter(loader)  # the workers fork here, before this process starts CUDA
    module, _ = load(device, wrap=False)
    run_on = f"{GPU_CHAIN}; network {device_note(device)}; {ENV_NOTE}"
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    print(GPU_CHAIN, "on", device_note(device), "|", workers, "loader workers |", len(todo), "shots |",
          ENV_NOTE, flush=True)
    total, count = Counter(), Counter()
    print("LOOP_START", stamp(), flush=True)
    t_loop = time.perf_counter()
    warm = False
    for _ in range(len(todo)):
        t = time.perf_counter()
        item = next(items)
        took = {"wait": time.perf_counter() - t}
        shot = item["shot"]
        if "skip" in item:
            skip(work, shot, item["skip"])
            continue
        took["read"] = item["read_s"]  # in the worker, overlapped with the loop
        try:
            t = time.perf_counter()
            sig, t0_ms, fs_hz = gpu.input_signals(item["co2"], device=device)  # copy in, resample, cut
            sync()
            took["resample"], t = time.perf_counter() - t, time.perf_counter()
            spec = gpu.model_input(sig)  # (4, 348, frames)
            sync()
            took["stft"], t = time.perf_counter() - t, time.perf_counter()
            out = gpu.run_windows(module, spec.transpose(1, 2).contiguous(),
                                  window=S.WINDOW_FRAMES, context=S.CONTEXT_FRAMES)
            logit = out[:, 0].cpu().numpy()
            took["network"], t = time.perf_counter() - t, time.perf_counter()
        except (NoDataError, ValueError) as error:
            skip(work, shot, input_failed(error))
            continue
        n = int(sig.shape[1])
        del sig, spec, out
        t_ms = full.frame_times(n, t0_ms, fs_hz)
        prob = S.moving_average(1.0 / (1.0 + np.exp(-logit.astype(np.float64))))
        rec_end = float(t0_ms + (n - 1) * 1000 / fs_hz)
        took["post"], t = time.perf_counter() - t, time.perf_counter()
        save(work, shot, t_ms, prob, item["end"], item["how"], rec_end, fs_hz, item["tier"], run_on)
        took["write"] = time.perf_counter() - t
        total.update(took)
        count.update(took.keys())
        print(shot, "end", item["end"], "|", t_ms.size, "columns |",
              " ".join(f"{k} {v:.3f}" for k, v in took.items()), flush=True)
        if not warm:
            print("LOOP_WARM", stamp(), flush=True)
            warm = True
    sync()
    print("LOOP_END", stamp(), flush=True)
    wall = time.perf_counter() - t_loop
    ran = count["write"]
    print(f"loop: {len(todo)} shots ({ran} written) in {wall:.1f} s, {wall / max(len(todo), 1):.2f} s a shot",
          flush=True)
    print("stage totals (s):", " ".join(f"{k} {total[k]:.1f}" for k in total),
          "| means (s):", " ".join(f"{k} {total[k] / max(count[k], 1):.3f}" for k in total), flush=True)
    if device.type == "cuda":
        print(f"GPU memory peak {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB", flush=True)


def build(shots, paths, work):
    """Stage 2: frame calls -> interval rows and per-shot summary (rerunning is exact)."""
    rows, per_shot, skipped = [], {}, {}
    for shot in shots:
        f, skip_file = work / f"{shot}.npz", work / f"{shot}.skip.json"
        if skip_file.exists():
            skipped[str(shot)] = json.loads(skip_file.read_text())["skipped"]
            continue
        if not f.exists():
            skipped[str(shot)] = "not run"
            continue
        z = np.load(f)
        end = float(z["plasma_end_ms"])
        rule = str(z["plasma_end_how"])
        if UNTAGGED_IP in rule and ip_clock_tag(shot, paths) in ("archive", "fdp"):
            rule = rule.replace(UNTAGGED_IP, "")  # written for every cached Ip before 2026-10-16
        rule = rule.replace(OLD_EFIT_REASON, EFIT_REASON)  # the same reason, worded as since 2026-10-16
        n = int(end // FRAME_MS)  # whole frames inside [0, end]
        if n < 1:
            skipped[str(shot)] = f"window end {end:g} ms leaves no whole 10 ms frame"
            continue
        share, mean, cnt = frames(z["t_ms"], z["prob"].astype(np.float64), n)
        if not (cnt > 0).all():
            first = int(np.argmin(cnt > 0))  # the first frame with no STFT column
            if first == 0 or (cnt[first:] > 0).any():
                skipped[str(shot)] = (f"frames with no STFT column inside [0, {FRAME_MS * n}] ms; CO2 input "
                                      f"{float(z['t_ms'][0]):.1f}-{float(z['input_end_ms']):.1f} ms")
                continue
            rule += (f"; cut to {FRAME_MS * first} ms, the last frame with STFT columns: the CO2 record "
                     f"ends at {float(z['input_end_ms']):.1f} ms")
            n, share, mean = first, share[:first], mean[:first]
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
            "window_end_rule": rule,
            "frames": n,
            "present_frames": int(present.sum()),
            "present_fraction": round(float(present.mean()), 4),
            "n_intervals": runs,
            "frames_where_mean_probability_disagrees_with_call": int(((mean >= THRESHOLD) != present).sum()),
            # Recorded since 2026-10-15; the gold run's cache has neither.
            **{key: str(z[key]) for key in ("co2_tier", "run_on") if key in z.files},
        }
    frame = pd.DataFrame(rows, columns=["shot", "category", "t_start", "t_end", "confidence"])
    validate_intervals(frame)
    return frame, per_shot, skipped


def in_training(labelled, paths):
    """-> {"train": [...], "valid": [...]}: labelled shots the checkpoint was fitted
    (train) or had its epoch chosen (valid) on, by its 180-shot split."""
    inside = [s for s in labelled if s in S.TRAINING_SHOTS]
    split = seldnet_split(paths.root / "ae" / "masks")
    found = {}
    for s in inside:
        found.setdefault(split.get(s, "unknown split"), []).append(s)
    return found


def training_note(found, listed):
    if not found:
        return TRAINING + ", none of the shots here is among them"
    inside = sorted(s for shots in found.values() for s in shots)
    parts = [f"{len(v)} {'fitted' if k == 'train' else 'validation (chose the epoch)' if k == 'valid' else k}: "
             f"{', '.join(map(str, sorted(v)))}" for k, v in sorted(found.items())]
    note = (f"{TRAINING}, 120 fitted and 60 validation; {len(inside)} of the shots here are among them, so "
            f"their frames are in-sample: {'; '.join(parts)}")
    if listed is not None and "pool" in listed:
        pools = listed.set_index("shot").pool
        counts = Counter(pools[s] for s in inside)
        note += "; by the list's pool: " + ", ".join(
            f"{n} of the {int((pools == p).sum())} {p}" for p, n in sorted(counts.items()))
    return note


def dirty_note():
    """Which tracked files differ from HEAD now, and which of them this run uses."""
    try:
        out = subprocess.run(["git", "diff", "--name-only", "HEAD"], cwd=REPO, capture_output=True,
                             text=True, timeout=10, check=True).stdout.split()
    except (OSError, subprocess.SubprocessError):
        return "git could not say which tracked files differ from HEAD"
    if not out:
        return "no tracked file differed from HEAD when the table was built"
    loaded = {Path(m.__file__).resolve() for m in list(sys.modules.values()) if getattr(m, "__file__", None)}
    used = [f for f in out if (REPO / f).resolve() in loaded]
    return (f"tracked files that differed from HEAD when the table was built: {', '.join(out)}; of these "
            f"the run imports {', '.join(used) or 'none'} (this script's sha256 is under scripts)")


def list_note(shots_file, listed, shots):
    path = Path(shots_file).resolve()
    name = events_ref(path, Paths.from_env().label_tables.resolve())
    if name == str(path) and path.is_relative_to(REPO):
        name = str(path.relative_to(REPO))
    note = (f"{name}: all {len(shots)} listed shots, whether or not another alfven_eigenmode table labels "
            "them (on the review page the owner's save, a format table and an extend table whose path sorts "
            "first outrank this one)")
    if "pool" in listed:
        note += "; pools " + ", ".join(f"{p} {n}" for p, n in Counter(listed.pool).most_common())
    return note


def sidecar(shots, per_shot, skipped, paths, name=GOLD, shots_file=None, listed=None, notes=()):
    gold = shots_file is None
    frs = np.array([v["present_fraction"] for v in per_shot.values()])
    ends = {s: v["window_end_rule"] for s, v in per_shot.items()}
    efit = sorted(int(s) for s, r in ends.items() if r.startswith("last EFIT02"))
    labelled = sorted(int(s) for s in per_shot)
    found = in_training(labelled, paths)
    tiers = Counter(v.get("co2_tier", "unrecorded") for v in per_shot.values())
    runs = Counter(v["run_on"] for v in per_shot.values() if "run_on" in v)
    meta = {
        "category": "alfven_eigenmode",
        "categories": {"0": "absent", "1": "present"},
        "producer": "ae-seldnet",
        "status": "experimental",
        "review_status": "unreviewed model output; nobody has looked at these frames",
        "made_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "made_by": "scripts/labeler/ae_gold_extend.py (shared repo code only)",
        "scripts": {"scripts/labeler/ae_gold_extend.py": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        "git_sha": git_sha(),
        "git_sha_full": git_sha(full=True),
        "git_dirty": bool(git_dirty()),
        "git_note": GOLD_GIT_NOTE if gold else dirty_note(),
        "shot_list": GOLD_SHOT_LIST if gold else list_note(shots_file, listed, shots),
        "shots_requested": list(shots),
        "shots_labelled": labelled,
        "shots_skipped": skipped,
        "model": {
            "name": "d3d_ae_activity_seldnet",
            "role": "stable and latest AE model in docs/labeler/events/alfven_eigenmode/README.md Models",
            "network": "labeler.ae.model.AeSeldNet (440,514 parameters), run as src/labeler/models/d3d_ae_activity_seldnet/spec.py runs it",
            "checkpoint": str(CKPT),
            "checkpoint_sha256": sha256_of(CKPT),
            "training": training_note(found, listed),
            "card": "src/labeler/models/d3d_ae_activity_seldnet/README.md",
        },
        "inputs": GOLD_INPUTS if gold else (
            "CO2 interferometer chords R0, V1, V2, V3 over [0, window end] ms through labeler.ae.full.input_signals "
            "(the page's own resampler to 500 kHz), or its torch twin labeler.ae.gpu.read_co2 + input_signals where "
            f"per_shot run_on starts with '{GPU_CHAIN}'; by the tier read: "
            + ", ".join(f"{t} {n}" for t, n in tiers.most_common())
            + " (per_shot co2_tier); nothing was fetched by the run"),
        "band": "80-250 kHz: STFT bins 164:512 of hann 1024, hop 128 at 500 kHz (0.256 ms columns); 250 kHz is the Nyquist of the CO2 record",
        "method": {
            "frame_probability": "sigmoid of the activity logit per STFT column, then the card's 5-column centred moving average, in 1,024-column windows with 128 columns of context (the adapter's own chain, not its 25 ms aggregation)",
            "threshold": THRESHOLD,
            "frame_call": "a 10 ms frame [10k, 10k+10) is present when at least half of its STFT columns have probability >= 0.5, as in docs/labeler/ae_baselines_benchmark.md; otherwise absent",
            "intervals": "runs of equal consecutive frame calls; category 1 present, 0 absent; no smoothing, merging or minimum length applied",
            "confidence": "mean over the interval's frames of the share of the frame's columns that agree with the call; a model score, not calibrated, not a probability the owner would agree",
            "window": f"[0, end] with end the plasma end snapped down to a 10 ms frame edge: the end of the longest stretch with Ip >= 50 kA (labeler.events.catalog.window.summarise, from the cached Ip) for {len(per_shot) - len(efit)} shots; the last EFIT02 time for {', '.join(str(s) for s in efit) or 'none'}, where no Ip >= 50 kA window could be measured from the Ip on disk (none cached, or catalog.window formed none: archive Ip at 25 ms steps is coarser than its 10 ms bridge), so the window falls back to the EFIT02 end. Where the CO2 record stops giving STFT columns first, the window ends at the last frame with columns (per_shot window_end_rule). The spectrogram statistics are taken over this same span, as the card's clip_to_grid does. Frames outside the window are unassessed, not absent.",
        },
        "per_shot": per_shot,
        "present_fraction_summary": {
            "min": float(frs.min()) if frs.size else None,
            "median": float(np.median(frs)) if frs.size else None,
            "max": float(frs.max()) if frs.size else None,
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
    if gold:
        meta["first_run_scripts"] = FIRST_RUN_SCRIPTS
    if notes:
        meta["notes"] = list(notes)
    if found:
        meta["shots_in_training"] = {k: sorted(v) for k, v in found.items()}
        meta["caveats"].append(
            f"{sum(map(len, found.values()))} shots here are among the 180 the checkpoint was fitted or validated on "
            "(shots_in_training, model.training): their frames are in-sample, so agreement with the owner's labels "
            "on them says nothing about unseen shots.")
    if runs and any(r.startswith(GPU_CHAIN) for r in runs):
        meta["inference"] = {
            "run_on": dict(runs.most_common()),
            "note": f"where the chain ran, per shot (per_shot run_on). A run_on that starts with '{GPU_CHAIN}' ran "
                    "the whole chain in torch on that device: the resample, STFT, clip, standardisation, band and "
                    "1,024-column windows of labeler.ae.gpu, checked against the CPU chain in "
                    "tests/labeler/test_ae_gpu.py (the resampler equal to scipy's bit for bit on CPU torch, and on CUDA "
                    "by construction: the same multiplies and adds in the same order; spectrogram statistics "
                    "accumulated in float64, the network's full windows batched). The others ran the network there and the rest "
                    "of the chain on the CPU. Either way the column probabilities differ from a CPU run in the last "
                    "digits, so a frame whose share sits at the 0.5 cut can flip. The gold table's network ran on a "
                    "CPU.",
        }
    elif runs:
        meta["inference"] = {
            "run_on": dict(runs.most_common()),
            "note": "where the network ran, per shot (per_shot run_on); the rest of the chain ran on the CPU. A GPU "
                    "sums float32 in another order than a CPU, so its column probabilities differ in the last digits "
                    "and a frame whose share sits at the 0.5 cut can flip. The gold table's network ran on a CPU.",
        }
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots-file", type=Path, help="the shot list (one shot a line, or a CSV with a shot column), every shot of it labelled; default: the gold shots with no AE label")
    ap.add_argument("--name", help="the table's stem under extend_ae_seldnet/ (default: the list file's stem; gold_shots without --shots-file)")
    ap.add_argument("--work-dir", type=Path, help="probability cache (default $LABELER_ROOT/runs/ae_gold_extend, runs/ae_seldnet_<name> for another table)")
    ap.add_argument("--shots", type=int, nargs="*", help="run only these shots' inference; the table is always rebuilt over all")
    ap.add_argument("--device", default="cpu", help="where the network runs: cpu (default) or cuda[:k]; with --chain cpu the rest of the chain is on the CPU")
    ap.add_argument("--chain", choices=("cpu", "gpu"), default="cpu", help="cpu (default): numpy and scipy, the network on --device; gpu: the whole chain in torch on --device (labeler.ae.gpu), fed by --workers loader processes")
    ap.add_argument("--workers", type=int, default=max(len(os.sched_getaffinity(0)) - 1, 0), help="--chain gpu: DataLoader processes reading the next shots (default: the CPUs this process may use, less one)")
    ap.add_argument("--build-only", action="store_true", help="never load the model: build from the cache, a shot without one is 'not run'")
    ap.add_argument("--no-write", action="store_true", help="build and validate, write nothing")
    ap.add_argument("--note", action="append", default=[], help="a provenance line for the sidecar's notes (repeatable)")
    args = ap.parse_args()
    paths = Paths.from_env()
    name = args.name or (args.shots_file.stem if args.shots_file else GOLD)
    work = args.work_dir or paths.root / "runs" / ("ae_gold_extend" if name == GOLD else f"ae_seldnet_{name}")
    work.mkdir(parents=True, exist_ok=True)
    listed = read_list(args.shots_file) if args.shots_file else None
    shots = shots_to_label(paths, listed)
    todo = args.shots or shots
    if not args.build_only and any(
            not (work / f"{s}.npz").exists() and not (work / f"{s}.skip.json").exists() for s in todo):
        assert os.environ.get("LABELER_NO_FETCH") == "1", "set LABELER_NO_FETCH=1: this script never fetches"
        if args.chain == "gpu":
            run_gpu(todo, paths, work, args.device, args.workers)
        else:
            module, _ = load(args.device)
            print("network on", device_note(args.device), "|", ENV_NOTE, flush=True)
            for shot in todo:
                run_shot(module, shot, paths, work, args.device)
    frame, per_shot, skipped = build(shots, paths, work)
    meta = sidecar(shots, per_shot, skipped, paths, name, args.shots_file, listed, args.note)
    if args.no_write:
        print(len(frame), "rows", frame.shot.nunique(), "shots (not written)")
    else:
        out = paths.label_tables / table_path(name)
        out.parent.mkdir(parents=True, exist_ok=True)
        write_interval_table(frame, out, meta)
        print("wrote", out, len(frame), "rows", frame.shot.nunique(), "shots")
    print(json.dumps(meta["present_fraction_summary"]), "skipped:", skipped)
    return 0


if __name__ == "__main__":
    sys.exit(main())
