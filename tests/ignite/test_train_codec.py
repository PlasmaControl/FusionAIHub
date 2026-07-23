"""Tests for the streaming, DDP-capable production codec trainer
(``ignite.train_codec``).

Coverage (all CPU; NO SLURM, NO GPU, NO production HDF5):

  * :class:`CodecPairDataset` — a THIN subclass of the production
    :class:`~tokamak_foundation_model.data.multi_file_dataset.TokamakMultiFileDataset` —
    yields correct-shape ``(C, F, T)`` δ-pairs from tiny SYNTHETIC HDF5 shots written in
    the real loader format, and is iterable through a DataLoader under both
    ``num_workers=0`` and ``num_workers>0`` (the parent's per-worker LRU file-handle path).
  * It REUSES the parent's index machinery: it is a ``TokamakMultiFileDataset`` subclass and
    does NOT redefine the ``searchsorted`` global-idx → ``(file, chunk)`` map (it overrides
    only the ``_getitem_standard`` per-item transform hook).
  * A few ``_ddp_codec_train_step`` iterations run in the world_size==1 (single-process)
    path and REDUCE the generator loss on the dataset's data.
  * The DDP init/wrap code path is import-safe + guarded when world_size==1 (``_DDPState``
    is non-distributed, ``wrap`` is a no-op) and the whole ``train_codec`` loop runs.
  * ``modality_channels`` reports the loader's real per-modality channel count.

The synthetic shots are sum-of-sinusoid raw signals resampled to the modality's target_fs,
stored as ``{modality}/xdata`` + ``{modality}/ydata`` so ``TokamakH5Dataset._load_signal_raw``
reads them exactly like a real shot.

Run:
    .pixi/envs/default/bin/python -m pytest tests/ignite/test_train_codec.py -q
"""
from __future__ import annotations

import inspect
import math
from pathlib import Path
from typing import List

import h5py
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from tokamak_foundation_model.ignite import train_codec as tc
from tokamak_foundation_model.ignite.config import CHUNK_S, STFT_FS, SpectroCodecConfig


# --------------------------------------------------------------------------------------- #
# synthetic real-format HDF5 shots
# --------------------------------------------------------------------------------------- #
def _write_synthetic_shot(
    path: Path, modality: str, channels: int, duration_s: float, fs: float, seed: int
) -> None:
    """Write a tiny {modality}/xdata + {modality}/ydata HDF5 shot in the loader's format.

    ydata is (channels, N) sum-of-sinusoid (few modes + noise) so windows are non-degenerate;
    xdata is the time vector [0, duration_s].
    """
    rng = np.random.default_rng(seed)
    n = int(round(duration_s * fs))
    t = np.arange(n) / fs
    freqs = [40e3, 90e3, 150e3]
    ydata = np.zeros((channels, n), dtype=np.float32)
    for c in range(channels):
        sig = np.zeros(n, dtype=np.float64)
        for f in freqs:
            phase = 2 * math.pi * rng.random()
            sig += np.sin(2 * math.pi * (f + c * 1e3) * t + phase)
        sig += 0.01 * rng.standard_normal(n)
        ydata[c] = sig.astype(np.float32)
    with h5py.File(path, "w") as h5:
        grp = h5.create_group(modality)
        grp.create_dataset("xdata", data=t.astype(np.float64))
        grp.create_dataset("ydata", data=ydata)


@pytest.fixture
def synthetic_shots(tmp_path):
    """A handful of tiny synthetic ``co2`` shots (4ch, 500 kHz, ~1.5 s each)."""
    modality = "co2"
    channels = tc.modality_channels(modality)  # 4 for co2
    fs = STFT_FS
    # duration must exceed t0_start (1.0) + CHUNK_S + delta_max, with room for several
    # windows so the map-style dataset has length > 1 per shot.
    duration_s = 1.0 + 5 * CHUNK_S + 0.3
    shots = []
    for i in range(4):
        sid = f"90000{i}"
        _write_synthetic_shot(
            tmp_path / f"{sid}_processed.h5", modality, channels, duration_s, fs, seed=i
        )
        shots.append(sid)
    return {"dir": tmp_path, "modality": modality, "channels": channels, "shots": shots}


def _tiny_cfg(channels: int) -> SpectroCodecConfig:
    """Small codec config for a fast CPU run (matches test_spike_cli shrink)."""
    return SpectroCodecConfig(
        channels=channels,
        freq_bins=64,
        time_frames=32,
        patch_f=32,
        patch_t=16,
        d_model=32,
        enc_depth=1,
        dec_depth=1,
        heads=2,
        fsq_levels=[4, 4, 3],
    )


def _make_ds(synthetic_shots, cfg, **kw):
    return tc.CodecPairDataset(
        synthetic_shots["modality"],
        synthetic_shots["shots"],
        cfg,
        data_dir=synthetic_shots["dir"],
        **kw,
    )


# --------------------------------------------------------------------------------------- #
# modality_channels
# --------------------------------------------------------------------------------------- #
def test_modality_channels_matches_loader_config():
    assert tc.modality_channels("ece") == 40   # channels_to_use=slice(0,40)
    assert tc.modality_channels("co2") == 4     # no channels_to_use
    assert tc.modality_channels("bes") == 16    # slice(48,64)
    assert tc.modality_channels("mhr") == 6     # slice(2,8)
    with pytest.raises(ValueError):
        tc.modality_channels("not_a_modality")


# --------------------------------------------------------------------------------------- #
# CodecPairDataset REUSES the parent's index machinery (the whole point of the refactor)
# --------------------------------------------------------------------------------------- #
def test_codec_dataset_subclasses_production_dataset(synthetic_shots):
    cfg = _tiny_cfg(synthetic_shots["channels"])
    ds = _make_ds(synthetic_shots, cfg)
    # It IS a TokamakMultiFileDataset (reuses idx-map + LRU handles + length cache + pickling).
    assert isinstance(ds, TokamakMultiFileDataset)
    # It reuses the parent's map-style interface: __len__ + integer indexing.
    assert len(ds) > 1
    # It overrides ONLY the per-item transform hook — NOT __getitem__ (the binary-search
    # index map lives in the parent's __getitem__ and is inherited unchanged).
    assert "_getitem_standard" in vars(tc.CodecPairDataset)
    assert "__getitem__" not in vars(tc.CodecPairDataset), (
        "CodecPairDataset must NOT redefine __getitem__ / the searchsorted index map — "
        "it should inherit the parent's global-idx -> (file, chunk) mapping."
    )
    # And it does NOT re-implement the searchsorted global-idx map anywhere in its own body
    # (the mapping is inherited from the parent's __getitem__).
    src = inspect.getsource(tc.CodecPairDataset)
    assert "searchsorted" not in src, (
        "CodecPairDataset re-implemented the binary-search index map; it must reuse the "
        "parent's __getitem__ mapping."
    )
    # The parent's cumulative-length map is present + consistent with __len__.
    assert int(ds._cumulative_lengths[-1]) == len(ds)


def test_codec_dataset_shape_and_delta_pair(synthetic_shots):
    cfg = _tiny_cfg(synthetic_shots["channels"])
    ds = _make_ds(synthetic_shots, cfg, seed=0)
    n = min(6, len(ds))
    for i in range(n):
        spec_a, spec_b = ds[i]
        assert spec_a.shape == (cfg.channels, cfg.freq_bins, cfg.time_frames)
        assert spec_b.shape == (cfg.channels, cfg.freq_bins, cfg.time_frames)
        assert torch.isfinite(spec_a).all() and torch.isfinite(spec_b).all()
        # δ-shift pair: same modes, different sub-window realization -> not identical.
        assert not torch.equal(spec_a, spec_b)


# --------------------------------------------------------------------------------------- #
# DataLoader under num_workers 0 and >0 (parent's per-worker LRU handle path)
# --------------------------------------------------------------------------------------- #
def test_codec_loader_num_workers_0(synthetic_shots):
    cfg = _tiny_cfg(synthetic_shots["channels"])
    ds = _make_ds(synthetic_shots, cfg, seed=1)
    loader = tc.make_codec_loader(ds, batch_size=3, num_workers=0, seed=1)
    spec_a, spec_b = next(iter(loader))
    assert spec_a.shape == (3, cfg.channels, cfg.freq_bins, cfg.time_frames)
    assert spec_b.shape == (3, cfg.channels, cfg.freq_bins, cfg.time_frames)
    assert torch.isfinite(spec_a).all()


def test_codec_loader_num_workers_2(synthetic_shots):
    """num_workers>0: each worker gets its own object copy + LRU handle (no shared-file race).

    This exercises the parent's __getstate__/__setstate__ pickling into worker processes.
    """
    cfg = _tiny_cfg(synthetic_shots["channels"])
    ds = _make_ds(synthetic_shots, cfg, seed=2)
    loader = tc.make_codec_loader(ds, batch_size=2, num_workers=2, seed=2)
    seen = 0
    for spec_a, spec_b in loader:
        assert spec_a.shape[1:] == (cfg.channels, cfg.freq_bins, cfg.time_frames)
        assert torch.isfinite(spec_a).all()
        seen += 1
        if seen >= 3:
            break
    assert seen >= 1


def test_codec_loader_ddp_sampler_shards(synthetic_shots):
    """world_size>1 uses the parent's DistributedTwoLevelSampler (file-level sharding)."""
    cfg = _tiny_cfg(synthetic_shots["channels"])
    ds = _make_ds(synthetic_shots, cfg, seed=0)
    loader0 = tc.make_codec_loader(
        ds, batch_size=1, num_workers=0, rank=0, world_size=2, seed=0
    )
    loader1 = tc.make_codec_loader(
        ds, batch_size=1, num_workers=0, rank=1, world_size=2, seed=0
    )
    assert isinstance(loader0.sampler, tc.DistributedTwoLevelSampler)
    # Disjoint file shards -> disjoint global chunk indices across ranks.
    idx0 = set(iter(loader0.sampler))
    idx1 = set(iter(loader1.sampler))
    assert idx0.isdisjoint(idx1)


# --------------------------------------------------------------------------------------- #
# _ddp_codec_train_step — runs + reduces the generator loss (world_size==1 path)
# --------------------------------------------------------------------------------------- #
def test_codec_train_step_reduces_generator_loss(synthetic_shots):
    torch.manual_seed(0)
    cfg = _tiny_cfg(synthetic_shots["channels"])
    device = torch.device("cpu")

    codec = tc.SpectroCodec(cfg).to(device)
    disc_raw = tc.FreqAwarePatchGAN(cfg).to(device)
    gen_module = tc._GenLossAdapter(codec)  # world_size==1 -> no DDP wrap
    disc = disc_raw
    opt_g = torch.optim.Adam(codec.parameters(), lr=1e-3)
    opt_d = torch.optim.Adam(disc_raw.parameters(), lr=1e-3)

    ds = _make_ds(synthetic_shots, cfg, seed=0)
    # ONE fixed batch, trained repeatedly (an overfit probe): isolates that the shared
    # per-step math actually optimizes the codec, without the GAN adversarial term's
    # step-to-step oscillation against the discriminator on fresh random batches. We assert
    # reduction on the PIXEL reconstruction anchor (the clean learning signal), matching how
    # run_spike proves learning; the adversarial `total` is expected to be non-monotone.
    pairs = [ds[i] for i in range(min(4, len(ds)))]
    a = torch.stack([p[0] for p in pairs]).to(device)
    b = torch.stack([p[1] for p in pairs]).to(device)

    pixel_losses: List[float] = []
    for step in range(20):
        g_terms, d_loss = tc._ddp_codec_train_step(
            gen_module, codec, disc, disc_raw, opt_g, opt_d, a, b, cfg, step=step,
        )
        pixel_losses.append(float(g_terms["pixel"].detach()))
        assert math.isfinite(pixel_losses[-1])
        assert math.isfinite(float(g_terms["total"].detach()))
        assert math.isfinite(float(d_loss.detach()))

    # reconstruction (pixel) anchor should trend down: last-quarter mean < first-quarter mean.
    first = sum(pixel_losses[:5]) / 5
    last = sum(pixel_losses[-5:]) / 5
    assert last < first, f"pixel recon did not reduce: first={first:.4f} last={last:.4f}"


# --------------------------------------------------------------------------------------- #
# _DDPState — import-safe + guarded when world_size==1
# --------------------------------------------------------------------------------------- #
def test_ddp_state_single_process(monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    st = tc._DDPState()
    assert st.distributed is False
    assert st.world_size == 1
    assert st.is_main is True
    # wrap is a no-op in single-process mode
    m = torch.nn.Linear(3, 3)
    assert st.wrap(m) is m
    st.barrier()   # no-op, must not raise
    st.shutdown()  # no-op, must not raise


# --------------------------------------------------------------------------------------- #
# full train_codec loop (single-process CPU) — gate + best ckpt written
# --------------------------------------------------------------------------------------- #
def test_train_codec_loop_writes_artifacts(synthetic_shots, tmp_path, monkeypatch):
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    cfg = _tiny_cfg(synthetic_shots["channels"])
    out_dir = tmp_path / "run"

    final_gate = tc.train_codec(
        cfg,
        synthetic_shots["modality"],
        synthetic_shots["shots"],
        synthetic_shots["shots"],  # eval on the same tiny pool (test only)
        steps=3,
        eval_every=1,
        batch_size=2,
        num_workers=0,
        data_dir=synthetic_shots["dir"],
        lr=1e-3,
        eval_batches=2,
        eval_batch_size=2,
        eval_frames=3,
        out_dir=out_dir,
        seed=0,
    )
    assert final_gate["steps"] == 3
    assert final_gate["global_step"] == 3
    assert 0.0 <= final_gate["stability"] <= 1.0
    assert 0.0 <= final_gate["persistence"] <= 1.0

    # artifacts: gate jsons at steps 0,1,2 + last ckpt + summary written by main(); here
    # we drove train_codec directly, so check gate jsons + codec_last.pt.
    gate_jsons = sorted(out_dir.glob("gate_*.json"))
    assert gate_jsons, "no gate_<step>.json written"
    assert (out_dir / "codec_last.pt").exists()


def test_train_codec_lengths_cache_reused(synthetic_shots, tmp_path):
    """The parent's length-cache sidecar is written + reused (no re-scan) via train_codec."""
    cfg = _tiny_cfg(synthetic_shots["channels"])
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    cache_path = cache_dir / "codec_co2_lengths.pt"

    ds = _make_ds(synthetic_shots, cfg, lengths_cache_path=cache_path)
    assert cache_path.exists(), "length-cache sidecar not written by the parent"
    n = len(ds)
    # Re-open with the cache present: same length, loaded from cache (no HDF5 length scan).
    ds2 = _make_ds(synthetic_shots, cfg, lengths_cache_path=cache_path)
    assert len(ds2) == n


def test_train_codec_cli_main_end_to_end(synthetic_shots, tmp_path, monkeypatch):
    """The argparse main() runs on synthetic shots via --shots (single-process)."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    out_dir = tmp_path / "cli_run"

    # shrink the codec that main() builds internally.
    orig = tc.SpectroCodecConfig

    def _small(**kw):
        kw["freq_bins"] = 64
        kw["time_frames"] = 32
        kw["patch_f"] = 32
        kw["patch_t"] = 16
        kw["d_model"] = 32
        kw["enc_depth"] = 1
        kw["dec_depth"] = 1
        kw["heads"] = 2
        kw["fsq_levels"] = [4, 4, 3]
        return orig(**kw)

    monkeypatch.setattr(tc, "SpectroCodecConfig", _small)

    shots_csv = ",".join(synthetic_shots["shots"])
    argv = [
        "--modality", synthetic_shots["modality"],
        "--shots", shots_csv,
        "--eval_n_shots", "1",
        "--n_shots", "3",
        "--steps", "2",
        "--eval_every", "1",
        "--batch_size", "2",
        "--num_workers", "0",
        "--eval_batches", "2",
        "--eval_batch_size", "2",
        "--eval_frames", "3",
        "--out_dir", str(out_dir),
        "--data_dir", str(synthetic_shots["dir"]),
        "--seed", "0",
    ]
    gate = tc.main(argv)
    assert gate["steps"] == 2
    assert (out_dir / "summary.json").exists()
    assert sorted(out_dir.glob("gate_*.json"))
