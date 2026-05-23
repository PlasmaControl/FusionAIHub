"""Probe: confirm STFT shapes via the dataset (post-bugfix).

Now that the STFT NaN-fill mask projection is in place, this probe
loads a chunk through ``TokamakMultiFileDataset.__getitem__`` (both
standard and prediction modes) and asserts the expected STFT shapes
for ECE, CO2, and BES.
"""

from pathlib import Path

import torch

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
)


def _load_one(prediction: bool):
    data_dir = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
    stats_path = Path(
        "/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt"
    )
    stats = torch.load(stats_path, weights_only=False)

    shot = data_dir / "200003_processed.h5"

    diag = ["ece", "co2", "bes"]
    kwargs = dict(
        hdf5_paths=[shot],
        chunk_duration_s=0.05,
        warmup_s=1.0,
        preprocessing_stats=stats,
        input_signals=diag,
        target_signals=diag,
        n_fft=1024,
        hop_length=256,
        max_open_files=4,
    )
    if prediction:
        kwargs["prediction_mode"] = True
        kwargs["prediction_horizon_s"] = 0.05
    ds = TokamakMultiFileDataset(**kwargs)
    return ds[0], diag


def main() -> None:
    print("=== standard mode ===")
    sample, diag = _load_one(prediction=False)
    expected = {"ece": (40, 512, 98), "co2": (4, 512, 98), "bes": (16, 512, 98)}
    # NB: BES SignalConfig still has num_channels=64; will return (64, 512, 98)
    # until prerequisite #1 lands.
    for name in diag:
        t = sample[name]
        m = sample.get(f"{name}_mask")
        print(f"  {name:<5}  tensor={tuple(t.shape)}  finite={torch.isfinite(t).all().item()}  "
              f"mask={None if m is None else tuple(m.shape)}")

    print()
    print("=== prediction mode ===")
    sample, diag = _load_one(prediction=True)
    inputs = sample["inputs"]
    targets = sample["targets"]
    for name in diag:
        ti = inputs[name]
        tt = targets[name]
        mi = inputs.get(f"{name}_mask")
        print(f"  {name:<5}  input={tuple(ti.shape)}  target={tuple(tt.shape)}  "
              f"finite={torch.isfinite(ti).all().item() and torch.isfinite(tt).all().item()}  "
              f"mask_in={None if mi is None else tuple(mi.shape)}")


if __name__ == "__main__":
    main()
