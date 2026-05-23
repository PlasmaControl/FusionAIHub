"""Step 1 (Phase C video pipeline) tests.

Verify the data-loader changes that support the E2E video tokenizer:

* ``MOVIE_CONFIGS`` class attribute updated for tangtv (120x360,
  ``n_output_frames=3``).
* ``_load_movie_raw`` returns ``(data, pixel_valid_mask)``.
* In prediction mode, samples carry ``tangtv``, ``tangtv_channel_mask``, and
  ``tangtv_valid``; the time axis is subsampled from 5 to 3 frames.
* The default ``collate_fn`` batches everything correctly.

These tests touch real HDF5 fixtures from
``/scratch/gpfs/EKOLEMEN/foundation_model``. They are skipped if that
directory is not present so the suite can run on a stripped-down
checkout.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tokamak_foundation_model.data.data_loader import (
    MovieConfig,
    TokamakH5Dataset,
    collate_fn,
)


DATA_DIR = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
# Picked from the 1000-shot Step 0 inspection: tangtv non-empty.
PRESENT_SHOT = DATA_DIR / "191599_processed.h5"
# tangtv group present but raw ``ydata.shape == (7, 1)`` — hits the
# ``n_frames < 2`` early-return path inside ``_load_movie_raw``. The
# raw HDF5 shape is 7-channel (channels_to_use=[4, 6] is applied AFTER
# the early-return check, so this branch never sees the sliced shape).
EMPTY_SHOT = DATA_DIR / "192825_processed.h5"

EXPECTED_C = 2   # tangtv: only ch4 and ch6 carry plasma data
EXPECTED_T = 3
EXPECTED_H = 120
EXPECTED_W = 360


pytestmark = pytest.mark.skipif(
    not DATA_DIR.exists(),
    reason=f"Data fixture directory not present: {DATA_DIR}",
)


def _make_dataset(hdf5_path: Path) -> TokamakH5Dataset:
    """Tangtv-aware prediction-mode dataset over one shot.

    ``input_signals`` and ``target_signals`` both include tangtv so the
    sample dict carries it through the prediction-mode split.
    """
    return TokamakH5Dataset(
        hdf5_path=hdf5_path,
        chunk_duration_s=0.05,
        prediction_mode=True,
        prediction_horizon_s=0.05,
        input_signals=["tangtv"],
        target_signals=["tangtv"],
    )


# ── 1. MOVIE_CONFIGS class-level spec ────────────────────────────────────


def test_movie_configs_tangtv_spec():
    """tangtv must be at 120x360 with n_output_frames=3."""
    by_name = {c.name: c for c in TokamakH5Dataset.MOVIE_CONFIGS}
    assert "tangtv" in by_name
    cfg = by_name["tangtv"]
    assert cfg.height == 120
    assert cfg.width == 360
    assert cfg.n_output_frames == 3
    assert cfg.target_fps == 100  # plan: native 50 fps → resample to 100


# ── 2. ``_load_movie_raw`` signature ─────────────────────────────────────


@pytest.mark.skipif(
    not PRESENT_SHOT.exists(),
    reason=f"Sample shot missing: {PRESENT_SHOT.name}",
)
def test_load_movie_raw_returns_tuple_present():
    """Present-camera path: tensor + per-channel mask."""
    ds = _make_dataset(PRESENT_SHOT)
    cfg = next(c for c in ds.movie_configs if c.name == "tangtv")
    ds._open_hdf5()
    tensor, mask = ds._load_movie_raw(ds.h5_file, cfg, t_start=2.0, t_end=2.1)

    # 100 ms @ target_fps=100 → 10 frames in time before subsample
    assert tensor.shape == (cfg.channels, 10, cfg.height, cfg.width)
    assert tensor.dtype == torch.float32
    assert mask.shape == (cfg.channels,)
    assert mask.dtype == torch.bool
    # Present-camera shot must have at least one active channel.
    assert mask.any()


@pytest.mark.skipif(
    not EMPTY_SHOT.exists(),
    reason=f"Sample shot missing: {EMPTY_SHOT.name}",
)
def test_load_movie_raw_returns_tuple_empty():
    """Empty-camera path: zeros + all-False per-channel mask."""
    ds = _make_dataset(EMPTY_SHOT)
    cfg = next(c for c in ds.movie_configs if c.name == "tangtv")
    ds._open_hdf5()
    tensor, mask = ds._load_movie_raw(ds.h5_file, cfg, t_start=2.0, t_end=2.1)

    assert tensor.shape == (cfg.channels, 10, cfg.height, cfg.width)
    assert torch.all(tensor == 0)
    assert mask.shape == (cfg.channels,)
    assert mask.dtype == torch.bool
    assert not mask.any()


# ── 3. Prediction-mode sample dict ───────────────────────────────────────


@pytest.mark.skipif(
    not PRESENT_SHOT.exists(),
    reason=f"Sample shot missing: {PRESENT_SHOT.name}",
)
def test_sample_present_shapes_and_keys():
    ds = _make_dataset(PRESENT_SHOT)
    sample = ds[len(ds) // 2]   # mid-shot — known to have plasma

    for split in ("inputs", "targets"):
        d = sample[split]
        assert "tangtv" in d
        assert "tangtv_channel_mask" in d
        assert "tangtv_valid" in d

        movie = d["tangtv"]
        assert movie.shape == (EXPECTED_C, EXPECTED_T, EXPECTED_H, EXPECTED_W)
        assert movie.dtype == torch.float32

        mask = d["tangtv_channel_mask"]
        assert mask.shape == (EXPECTED_C,)
        assert mask.dtype == torch.bool

        valid = d["tangtv_valid"]
        assert isinstance(valid, int)
        assert valid == 1   # camera is present in this shot


@pytest.mark.skipif(
    not EMPTY_SHOT.exists(),
    reason=f"Sample shot missing: {EMPTY_SHOT.name}",
)
def test_sample_empty_shapes_and_keys():
    ds = _make_dataset(EMPTY_SHOT)
    sample = ds[len(ds) // 2]

    for split in ("inputs", "targets"):
        d = sample[split]
        movie = d["tangtv"]
        assert movie.shape == (EXPECTED_C, EXPECTED_T, EXPECTED_H, EXPECTED_W)
        assert torch.all(movie == 0)

        mask = d["tangtv_channel_mask"]
        assert mask.shape == (EXPECTED_C,)
        assert mask.dtype == torch.bool
        assert not mask.any()

        assert d["tangtv_valid"] == 0


# ── 4. Channel-mask sanity ───────────────────────────────────────────────


@pytest.mark.skipif(
    not PRESENT_SHOT.exists(),
    reason=f"Sample shot missing: {PRESENT_SHOT.name}",
)
def test_channel_mask_active_subset():
    """For shot 191599, both retained tangtv channels are active.

    The MovieConfig now keeps only raw channels 4 and 6 via
    ``channels_to_use=[4, 6]`` (these are the filters carrying plasma
    data). Channels 0/1/2/3/5 are dropped at load time. On this shot
    both retained channels carry data, so the per-channel availability
    mask should be all-True over the 2-channel output.
    """
    ds = _make_dataset(PRESENT_SHOT)
    sample = ds[len(ds) // 2]
    mask = sample["inputs"]["tangtv_channel_mask"]
    expected = torch.ones(EXPECTED_C, dtype=torch.bool)
    assert torch.equal(mask, expected), (
        f"Both retained channels (raw 4 and 6) should be active for shot "
        f"191599; got mask = {mask.tolist()}"
    )


# ── 5. Collation through default collate_fn ─────────────────────────────


@pytest.mark.skipif(
    not PRESENT_SHOT.exists(),
    reason=f"Sample shot missing: {PRESENT_SHOT.name}",
)
def test_collation_video_keys():
    ds = _make_dataset(PRESENT_SHOT)
    samples = [ds[i] for i in range(min(4, len(ds)))]
    batch = collate_fn(samples)

    inputs = batch["inputs"]
    targets = batch["targets"]

    B = len(samples)
    for d in (inputs, targets):
        assert d["tangtv"].shape == (B, EXPECTED_C, EXPECTED_T, EXPECTED_H, EXPECTED_W)
        assert d["tangtv"].dtype == torch.float32
        assert d["tangtv_channel_mask"].shape == (B, EXPECTED_C)
        assert d["tangtv_channel_mask"].dtype == torch.bool
        # ``_valid`` keys hit the long-tensor path in ``_collate_dict``.
        assert d["tangtv_valid"].shape == (B,)
        assert d["tangtv_valid"].dtype == torch.long


# ── 6. Subsample indices ─────────────────────────────────────────────────


def test_n_output_frames_picks_endpoints_and_centre():
    """For 5 → 3, the linspace round-and-cast strategy picks [0, 2, 4]."""
    idx = torch.linspace(0, 4, 3).round().long().tolist()
    assert idx == [0, 2, 4]