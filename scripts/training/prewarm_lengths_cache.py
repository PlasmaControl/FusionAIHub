"""Pre-warm the ALL-shots file-length cache for the FSQ Stage-1 chain.

The lengths cache (lengths_e2e_stage1_{train,val}.pt) is keyed by an EXACT
file-list match (see multi_file_dataset._load_or_compute_lengths). A prior run
that used a DIFFERENT file list (e.g. the video-present subset) leaves a cache
whose `paths` don't match the ALL-shots 7878-file list, so a fresh ALL-shots
job recomputes lengths from scratch. Under DDP only rank 0 scans (~1.8 h) while
the other ranks block on the broadcast collective -> the NCCL watchdog fires and
kills all 64 ranks.

This script reproduces the chain's EXACT train/val lists via the trainer's own
`resolve_shot_files` and constructs the datasets SINGLE-PROCESS, which triggers
the same length scan + atomic cache write with no distributed group -> no
watchdog. After it completes, the held chain loads the cache instantly.
"""
import sys
from pathlib import Path

FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from train_e2e_stage1 import build_datasets, resolve_shot_files

DATA = Path("/lustre/orion/fus187/proj-shared/foundation_model")
STATS = "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt"
CACHE = Path("/lustre/orion/fus187/proj-shared/foundation_model_meta")
# Exact diagnostics (14) + actuators (10) of the FSQ chain (from the run banner).
# Length (chunk-count) scanning is signal-independent, but pass the real sets so
# dataset construction matches the run exactly.
DIAG = ["ts_core_density", "ts_core_temp", "ts_tangential_density", "ts_tangential_temp",
        "cer_ti", "cer_rot", "mse", "filterscopes", "ece", "co2", "bes", "mhr",
        "tangtv_lower", "tangtv_upper"]
ACT = ["pin", "beam_voltage", "tin", "ech_power", "ech_tor_angle", "ech_pol_angle",
       "ech_polarization", "gas_flow", "gas_raw", "rmp"]

stats = torch.load(STATS, weights_only=False)
# ALL-shots: no yaml, max_files=None, val_fraction=0.1, seed=42 (matches the chain).
train_files, val_files = resolve_shot_files(DATA, None, None, None, 0.1, 42)
print(f"[prewarm] resolved train={len(train_files)} val={len(val_files)} "
      f"(ALL_SHOTS glob, seed=42, val_fraction=0.1)", flush=True)
print(f"[prewarm] first/last train: {train_files[0].name} .. {train_files[-1].name}", flush=True)
print("[prewarm] constructing datasets single-process -> scan + atomic cache write "
      "(~1-2 h, no NCCL) ...", flush=True)
build_datasets(DATA, train_files, val_files, stats, 0.05, 0.05, 0.01, 1.0, DIAG, ACT, CACHE)
for nm in ("lengths_e2e_stage1_train.pt", "lengths_e2e_stage1_val.pt"):
    fp = CACHE / nm
    print(f"[prewarm] {nm}: exists={fp.exists()} size={fp.stat().st_size if fp.exists() else 0}", flush=True)
print("[prewarm] CACHE PREWARMED — held chain can now be released", flush=True)
