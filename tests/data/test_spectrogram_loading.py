"""Step 1 (Phase B spectrogram pipeline) tests.

Verifies the data-loader changes that unblock the E2E spectrogram
tokenizer:

* STFT NaN-fill mask shape mismatch is fixed (``_getitem_standard`` and
  ``_getitem_prediction`` both load STFT signals without crashing).
* ``_raw_to_frame_mask`` projects raw-time validity to STFT-frame coords.
* BES SignalConfig slices to channels 49–64 (1-indexed) and uses
  ``log_standardize`` to match ECE/CO2.
* ``<name>_valid`` survives the prediction-mode input/target split and
  reads 0 for shots where the modality isn't present, > 0 otherwise.
* Non-STFT modalities are byte-shape-preserved (no regression on Phase A).

These tests touch real HDF5 fixtures from
``/scratch/gpfs/EKOLEMEN/foundation_model``. They are skipped if that
directory is not present so the suite can run on a stripped-down
checkout.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pytest
import torch

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset,
)


DATA_DIR = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
STATS_PATH = Path(
    "/scratch/gpfs/ps9551/FusionAIHub/scripts/slurm/preprocessing_stats.pt"
)

# Step 0 survey selected these. 200003 has all three modalities; 190000
# has ECE present but CO2/BES absent.
PRESENT_SHOT = DATA_DIR / "200003_processed.h5"
ECE_ONLY_SHOT = DATA_DIR / "190000_processed.h5"

# Plan-locked shape contract.
EXPECTED_C = {"ece": 40, "co2": 4, "bes": 16}
EXPECTED_F = 512
EXPECTED_T = 98


pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists() or not STATS_PATH.exists(),
    reason=(
        f"Fixtures not present: {DATA_DIR} or {STATS_PATH}. "
        "These tests need real shots and preprocessing stats."
    ),
)


@pytest.fixture(scope="module")
def stats() -> dict:
    return torch.load(STATS_PATH, weights_only=False)


def _make_ds(
    shot: Path, prediction: bool, stats: dict, signals: Tuple[str, ...] = (
        "ece", "co2", "bes",
    ),
) -> TokamakMultiFileDataset:
    kwargs = dict(
        hdf5_paths=[shot],
        chunk_duration_s=0.05,
        warmup_s=1.0,
        preprocessing_stats=stats,
        input_signals=list(signals),
        target_signals=list(signals),
        n_fft=1024,
        hop_length=256,
        max_open_files=4,
    )
    if prediction:
        kwargs["prediction_mode"] = True
        kwargs["prediction_horizon_s"] = 0.05
    return TokamakMultiFileDataset(**kwargs)


# ── Shape contract ────────────────────────────────────────────────────


def test_standard_mode_shape_contract(stats):
    """ECE/CO2/BES return ``(C, 512, 98)`` and matching mask."""
    ds = _make_ds(PRESENT_SHOT, prediction=False, stats=stats)
    sample = ds[0]
    for name in ("ece", "co2", "bes"):
        t = sample[name]
        assert t.shape == (EXPECTED_C[name], EXPECTED_F, EXPECTED_T), (
            f"{name}: got {tuple(t.shape)}"
        )
        assert torch.isfinite(t).all(), f"{name}: non-finite values present"
        m = sample.get(f"{name}_mask")
        assert m is not None, f"{name}: no mask emitted"
        assert m.shape == t.shape, (
            f"{name}: mask shape {tuple(m.shape)} != tensor shape {tuple(t.shape)}"
        )


def test_prediction_mode_shape_contract(stats):
    """Input and target halves both ``(C, 512, 98)`` (50 ms each)."""
    ds = _make_ds(PRESENT_SHOT, prediction=True, stats=stats)
    sample = ds[0]
    for name in ("ece", "co2", "bes"):
        ti = sample["inputs"][name]
        tt = sample["targets"][name]
        assert ti.shape == (EXPECTED_C[name], EXPECTED_F, EXPECTED_T)
        assert tt.shape == (EXPECTED_C[name], EXPECTED_F, EXPECTED_T)
        assert torch.isfinite(ti).all() and torch.isfinite(tt).all()


# ── BES SignalConfig (channels + preprocessing) ──────────────────────


def test_bes_channel_slice(stats):
    """BES returns 16 channels (post-slice), not 64."""
    ds = _make_ds(PRESENT_SHOT, prediction=False, stats=stats, signals=("bes",))
    sample = ds[0]
    assert sample["bes"].shape[0] == 16


def test_bes_uses_log_standardize(stats):
    """BES SignalConfig method is now ``log_standardize`` (matching ECE/CO2)."""
    ds = _make_ds(PRESENT_SHOT, prediction=False, stats=stats, signals=("bes",))
    bes_cfg = next(c for c in ds.signal_configs if c.name == "bes")
    assert bes_cfg.preprocess.method == "log_standardize"
    assert bes_cfg.channels_to_use == slice(48, 64)


# ── Per-modality presence indicator (``<name>_valid``) ───────────────


def test_valid_propagates_in_prediction_mode_present(stats):
    """``_valid > 0`` for all three modalities on a shot that has them."""
    ds = _make_ds(PRESENT_SHOT, prediction=True, stats=stats)
    sample = ds[0]
    for name in ("ece", "co2", "bes"):
        iv = int(sample["inputs"][f"{name}_valid"])
        tv = int(sample["targets"][f"{name}_valid"])
        assert iv > 0, f"{name}: input _valid should be > 0 on present shot"
        assert tv > 0, f"{name}: target _valid should be > 0 on present shot"


def test_valid_zero_when_modality_missing(stats):
    """``_valid == 0`` for CO2 and BES on a shot where they're absent."""
    ds = _make_ds(ECE_ONLY_SHOT, prediction=True, stats=stats)
    sample = ds[0]
    assert int(sample["inputs"]["ece_valid"]) > 0, "ECE should be present"
    for missing in ("co2", "bes"):
        iv = int(sample["inputs"][f"{missing}_valid"])
        tv = int(sample["targets"][f"{missing}_valid"])
        assert iv == 0, f"{missing}: input _valid should be 0 (modality absent)"
        assert tv == 0, f"{missing}: target _valid should be 0 (modality absent)"


# ── Bug-fix specific: STFT mask projection ───────────────────────────


def test_raw_to_frame_mask_projection(stats):
    """Helper projects (C, T_raw) → (C, T_frames). Any-NaN-in-source → invalid."""
    ds = _make_ds(PRESENT_SHOT, prediction=False, stats=stats, signals=("ece",))
    # Synthesise a (C=2, T=25_000) mask: first half all-valid, second
    # half has a 1024-sample contiguous NaN block at the start.
    raw_valid = torch.ones((2, 25_000), dtype=torch.bool)
    raw_valid[:, 12_500:13_524] = False  # one full STFT window invalid
    frame_mask = ds._raw_to_frame_mask(raw_valid)
    assert frame_mask.shape == (2, 98)
    # Frames whose source samples land in the invalid window should be False.
    n_false = (~frame_mask[0]).sum().item()
    assert n_false > 0 and n_false < 98, (
        f"Expected some frames invalid, got {n_false}/98"
    )
    # First frame (centred at sample 0) should be valid.
    assert frame_mask[0, 0].item()
    # Last frame should also be valid (its source is past the invalid block).
    assert frame_mask[0, -1].item()


# ── Non-STFT regression ──────────────────────────────────────────────


def test_non_stft_signals_unaffected(stats):
    """Non-STFT signals load with their original shape and dtype."""
    ds = _make_ds(
        PRESENT_SHOT, prediction=True, stats=stats,
        signals=("ts_core_density",),
    )
    sample = ds[0]
    ti = sample["inputs"]["ts_core_density"]
    tt = sample["targets"]["ts_core_density"]
    # ts_core_density is 44 ch × 100 Hz × 50 ms = 5 samples per half.
    assert ti.shape == (44, 5)
    assert tt.shape == (44, 5)
    assert torch.isfinite(ti).all() and torch.isfinite(tt).all()
    # ``_valid`` propagates for non-STFT signals too.
    assert int(sample["inputs"]["ts_core_density_valid"]) > 0