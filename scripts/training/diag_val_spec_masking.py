"""Standalone diagnostic for the CO2/BES = 0 issue in Stage 2 extended val.

Builds the same val-style dataset the extended trainer uses (K_max=80,
prediction_horizon=4.0s, warmup=1.0s), pulls a few batches WITHOUT
running the model, and inspects:
  * batch['targets'][<name>_valid] — the per-sample valid count used
    by ``_spectro_loss_gate`` to mask MAE.
  * batch['targets'][<name>] shape — full spec target time-axis length.
  * Compares the time-axis length against the expected
    ``K_max * trunc_t(name)`` that split_spectro_target_by_step needs.

Runs CPU-only; no GPU forward pass required.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from tokamak_foundation_model.data.data_loader import collate_fn  # noqa: E402
from tokamak_foundation_model.data.multi_file_dataset import (  # noqa: E402
    TokamakMultiFileDataset,
)
from tokamak_foundation_model.e2e.model import DiagnosticConfig  # noqa: E402

CKPT = (
    "/lustre/orion/fus187/proj-shared/models/e2e_stage2_extended_d1024_48L/"
    "e2e_stage2_ext_best.pt"
)
STATS = (
    "/lustre/orion/fus187/proj-shared/foundation_model_meta/"
    "preprocessing_stats.pt"
)
DATA_DIR = "/lustre/orion/fus187/proj-shared/foundation_model"

K_MAX = 80
CHUNK = 0.05
WARMUP = 1.0
N_SHOTS = 6
N_BATCHES = 4
BATCH_SIZE = 2


def _spectro_trunc_t(cfg: DiagnosticConfig) -> int:
    _, T_p = cfg.spectrogram_patch_size
    return (cfg.window_samples // T_p) * T_p


def main() -> None:
    print(f"loading checkpoint diagnostics from {Path(CKPT).name}...",
          flush=True)
    ckpt = torch.load(CKPT, weights_only=False, map_location="cpu")
    diagnostics = [DiagnosticConfig(**d) for d in ckpt["diagnostics"]]
    actuators_cfg = ckpt["actuators"]
    diag_names = [c.name for c in diagnostics]
    act_names = [a["name"] for a in actuators_cfg]

    spec_cfgs = {c.name: c for c in diagnostics if c.kind == "spectrogram"}
    print("\n=== Spec modality config (from checkpoint) ===")
    for n in ("ece", "co2", "bes"):
        c = spec_cfgs.get(n)
        if c is None:
            print(f"  {n}: NOT IN CHECKPOINT DIAGNOSTICS")
            continue
        tt = _spectro_trunc_t(c)
        expected_T = K_MAX * tt
        print(f"  {n:>4s}: n_ch={c.n_channels} "
              f"window_samples={c.window_samples} "
              f"patch={c.spectrogram_patch_size} "
              f"trunc_t={tt} "
              f"K_max*trunc_t={expected_T}")

    print(f"\nloading stats from {Path(STATS).name}...", flush=True)
    stats = torch.load(STATS, weights_only=False)

    # Mix: include shot 200729 (known to have real co2/bes) + first 5
    # alphabetical shots (which happen to be stub shots, per H5 inspection).
    shots = [Path(DATA_DIR) / "200729_processed.h5"] + \
            sorted(Path(DATA_DIR).glob("*_processed.h5"))[:N_SHOTS - 1]
    print(f"using {len(shots)} shots: {[p.stem.split('_')[0] for p in shots]}")

    ds = TokamakMultiFileDataset(
        shots,
        chunk_duration_s=CHUNK,
        prediction_mode=True,
        prediction_horizon_s=K_MAX * CHUNK,
        step_size_s=CHUNK,
        warmup_s=WARMUP,
        preprocessing_stats=stats,
        input_signals=diag_names,
        target_signals=diag_names + act_names,
        lengths_cache_path=None,
    )
    print(f"dataset windows: {len(ds)}")

    loader = DataLoader(
        ds, batch_size=BATCH_SIZE, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    spec_names = ("ece", "co2", "bes")
    # Aggregate stats across batches: how often is each modality valid > 0?
    valid_nonzero: dict[str, int] = {n: 0 for n in spec_names}
    valid_total: dict[str, int] = {n: 0 for n in spec_names}

    for i, batch in enumerate(loader):
        if i >= N_BATCHES:
            break
        print(f"\n=== Batch {i} (batch_size={BATCH_SIZE}) ===")
        for name in spec_names:
            cfg = spec_cfgs.get(name)
            if cfg is None:
                continue
            t = batch["targets"].get(name)
            v = batch["targets"].get(f"{name}_valid")
            shape_str = tuple(t.shape) if t is not None else "MISSING"
            v_list = v.tolist() if v is not None else "MISSING"
            expected_T = K_MAX * _spectro_trunc_t(cfg)
            actual_T = int(t.shape[-1]) if t is not None else 0
            ok = "OK" if actual_T >= expected_T else "TOO SHORT"
            short_by = expected_T - actual_T if actual_T < expected_T else 0
            print(f"  {name:>4s}: target shape={shape_str}  "
                  f"T_actual={actual_T}  T_expected={expected_T}  "
                  f"({ok}, short_by={short_by})")
            print(f"        valid (per sample)={v_list}")
            if v is not None:
                valid_total[name] += int(v.numel())
                valid_nonzero[name] += int((v > 0).sum().item())

    print("\n=== Aggregate over inspected batches ===")
    for name in spec_names:
        tot = valid_total[name]
        nz = valid_nonzero[name]
        frac = (nz / tot * 100) if tot > 0 else 0.0
        print(f"  {name:>4s}: valid>0 in {nz}/{tot} samples ({frac:.1f}%)")


if __name__ == "__main__":
    main()
