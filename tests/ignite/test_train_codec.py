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
    # This end-to-end test shrinks the codec to tiny dims; co2's production per-freq
    # standardization stats are full-size (C, 512) and would mismatch the shrunk config, so
    # opt out of the co2 standardization path here (dispatch is what's under test).
    monkeypatch.setattr(tc, "_SPECTRO_STANDARDIZE_SIGNALS", frozenset())

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


# --------------------------------------------------------------------------------------- #
# fast-TS (filterscopes) is now folded into the standard train_codec dispatch
# --------------------------------------------------------------------------------------- #
def test_train_codec_parser_accepts_filterscopes():
    """The train_codec CLI parser accepts ``--modality filterscopes``."""
    args = tc.build_arg_parser().parse_args(
        ["--modality", "filterscopes", "--out_dir", "/tmp/ignite_fastts_parse"]
    )
    assert args.modality == "filterscopes" == tc.FASTTS_MODALITY


def test_parser_accepts_prod_recipe_overrides():
    """The CLI exposes the prod-recipe knobs (the only spectro recipe that survives the
    multi-shot collapse test): fsq_levels / adaptive_adv_clamp / adv_warmup_steps /
    adversarial_weight / skip_activity_override. Guards the wiring so a launch can reproduce
    prod_ece (cb=32768, entropy 0.1, clamp 10000) without editing the drifted config defaults."""
    args = tc.build_arg_parser().parse_args(
        ["--modality", "bes", "--out_dir", "/tmp/ignite_prod_parse",
         "--fsq_levels", "8,8,8,8,8", "--adaptive_adv_clamp", "10000",
         "--adv_warmup_steps", "0", "--adversarial_weight", "1.0",
         "--entropy_weight", "0.1", "--skip_activity_override"]
    )
    assert args.fsq_levels == "8,8,8,8,8"
    assert args.adaptive_adv_clamp == 10000.0
    assert args.adv_warmup_steps == 0
    assert args.adversarial_weight == 1.0
    assert args.entropy_weight == 0.1
    assert args.skip_activity_override is True
    # the parsed comma-string maps to the cb=32768 prod codebook
    import math
    assert math.prod(int(x) for x in args.fsq_levels.split(",")) == 32768


def test_tangential_thomson_gets_present_fraction_stratification():
    """ts_tangential_{density,temp} carry the same missing-dominated bimodal present-fraction
    as ts_core_density (measured 2026-08-05: median 0.067, 59% of windows <= 0.1 present) but
    were left unstratified in the v6 fleet — ts_tangential_density degraded to 5 codes /
    corr 0.03 over 80k steps. Both must now get the ts_core_density present-fraction override."""
    for modality in ("ts_tangential_density", "ts_tangential_temp"):
        cfg = tc.slowts_codec_cfg(modality, tc.modality_channels(modality))
        assert cfg.min_activity == 0.0 and cfg.active_bias == 0.0
        tc.apply_activity_overrides(cfg, modality)
        assert cfg.min_activity == 0.5, modality
        assert cfg.active_bias == 0.5, modality


def test_refine_depth_is_video_only_cli_override():
    """--refine_depth parses, lands on VideoCodecConfig (which has the field), and must be
    rejected for families whose configs lack it (mirrors main()'s guard)."""
    from tokamak_foundation_model.ignite.config import SpectroCodecConfig, VideoCodecConfig
    args = tc.build_arg_parser().parse_args(
        ["--modality", "tangtv_lower", "--out_dir", "/tmp/x", "--refine_depth", "4"]
    )
    assert args.refine_depth == 4
    vcfg = VideoCodecConfig()
    assert vcfg.refine_depth == 0          # default OFF (byte-identical decoder)
    vcfg.refine_depth = int(args.refine_depth)
    assert vcfg.refine_depth == 4
    assert not hasattr(SpectroCodecConfig(channels=4), "refine_depth")


def test_prod_recipe_overrides_win_over_activity_and_defaults():
    """The prod-recipe CLI overrides are applied AFTER apply_activity_overrides, so they win
    over both the drifted d2 defaults AND the co2/mhr _activity_overrides (which force
    adv_warmup=1500 / adversarial_weight=0.5 and would otherwise fight the prod recipe).
    Replicates main()'s override block against a real config so the precedence is guarded."""
    from tokamak_foundation_model.ignite.config import SpectroCodecConfig
    cfg = SpectroCodecConfig(channels=4)
    # co2's activity overrides would set these (the collapse-inducing values):
    tc.apply_activity_overrides(cfg, "co2")
    assert cfg.adv_warmup_steps == 1500 and cfg.adversarial_weight == 0.5
    # now the prod-recipe CLI block (mirrors main()): CLI wins.
    args = tc.build_arg_parser().parse_args(
        ["--modality", "co2", "--out_dir", "/tmp/x", "--fsq_levels", "8,8,8,8,8",
         "--adaptive_adv_clamp", "10000", "--adv_warmup_steps", "0",
         "--adversarial_weight", "1.0", "--entropy_weight", "0.1"]
    )
    cfg.fsq_levels = [int(x) for x in args.fsq_levels.split(",")]
    for knob in ("adaptive_adv_clamp", "adv_warmup_steps", "adversarial_weight"):
        setattr(cfg, knob, getattr(args, knob))
    cfg.entropy_weight = args.entropy_weight
    import math
    assert math.prod(cfg.fsq_levels) == 32768
    assert cfg.adv_warmup_steps == 0            # prod overrode the activity 1500
    assert cfg.adversarial_weight == 1.0        # prod overrode the activity 0.5
    assert cfg.adaptive_adv_clamp == 10000.0
    assert cfg.entropy_weight == 0.1


def test_logpow_standardization_default_off_and_on():
    """Per-freq log-power z-standardization (the THIN-modality mean-collapse fix):
    (a) DEFAULTS OFF and leaves ``log_power_stft`` byte-identical to the raw baseline;
    (b) when ON with hand-made (C,F) mean/std the output spec is shifted/scaled as
        ``(spec - mean) / std.clamp_min(floor)`` (a constant can no longer minimize recon-MAE)."""
    from tokamak_foundation_model.ignite.config import SpectroCodecConfig
    from tokamak_foundation_model.ignite.data import log_power_stft

    torch.manual_seed(0)
    C = 2
    cfg = SpectroCodecConfig(channels=C, freq_bins=8, time_frames=4, patch_f=4, patch_t=2)
    # (a) default must be OFF and the output must equal the raw baseline (byte-identical).
    assert cfg.logpow_standardize is False
    assert cfg.logpow_freq_mean is None and cfg.logpow_freq_std is None
    W = cfg.window_samples
    raw = torch.randn(1, C, W)
    base = log_power_stft(raw, cfg)  # (1, C, F, T)
    assert base.shape == (1, C, cfg.freq_bins, cfg.time_frames)

    cfg_off2 = SpectroCodecConfig(channels=C, freq_bins=8, time_frames=4, patch_f=4, patch_t=2)
    assert torch.equal(log_power_stft(raw, cfg_off2), base), "OFF must be byte-identical"

    # (b) turn it ON with a small hand-made (C, F) mean/std; std well above the floor so it
    # scales rather than clamps. Expect the exact affine transform of the raw baseline.
    F = cfg.freq_bins
    fmean = (torch.arange(C * F, dtype=torch.float32).reshape(C, F) * 0.1 + 1.0)
    fstd = torch.full((C, F), 2.0)
    cfg.logpow_standardize = True
    cfg.logpow_freq_mean = fmean.tolist()
    cfg.logpow_freq_std = fstd.tolist()
    cfg.logpow_std_floor = 0.25  # 2.0 > floor -> no clamp
    out = log_power_stft(raw, cfg)
    expected = (base - fmean[None, :, :, None]) / fstd[None, :, :, None]
    assert torch.allclose(out, expected, atol=1e-5), "ON must be (spec - mean) / std per (C,F)"
    # the mean-subtraction actually changed the output (not a no-op).
    assert not torch.allclose(out, base)

    # floor: a below-floor std must be clamped up to the floor before dividing.
    cfg.logpow_freq_std = torch.full((C, F), 0.05).tolist()  # < floor 0.25
    out_floor = log_power_stft(raw, cfg)
    expected_floor = (base - fmean[None, :, :, None]) / 0.25
    assert torch.allclose(out_floor, expected_floor, atol=1e-5), "std below floor must clamp to floor"


def test_parser_accepts_logpow_stats_path():
    """The CLI exposes ``--logpow_stats_path`` (enables per-freq log-z, COMPOSED with raw-std)."""
    args = tc.build_arg_parser().parse_args(
        ["--modality", "co2", "--out_dir", "/tmp/ignite_logz_parse",
         "--logpow_stats_path", "/some/codec_co2_perfreq_stats.pt"]
    )
    assert args.logpow_stats_path == "/some/codec_co2_perfreq_stats.pt"
    # default is None (feature OFF) when the flag is absent.
    args2 = tc.build_arg_parser().parse_args(["--modality", "co2", "--out_dir", "/tmp/x"])
    assert args2.logpow_stats_path is None


def test_modality_channels_filterscopes_is_8():
    """``modality_channels('filterscopes') == 8`` (channels_to_use=slice(0,8))."""
    assert tc.modality_channels("filterscopes") == 8


def test_main_dispatches_filterscopes_to_fastts_trainer(tmp_path, monkeypatch):
    """``main(--modality filterscopes)`` builds a FastTSCodecConfig and calls the fast-TS
    trainer (``fastts_train.train_fastts_codec``) with the fast-TS keyword signature — reusing
    that train function rather than duplicating the training logic. No real run: the trainer is
    stubbed and we assert on the dispatch + the args it was handed."""
    monkeypatch.delenv("WORLD_SIZE", raising=False)
    from tokamak_foundation_model.ignite import fastts_train as ft
    from tokamak_foundation_model.ignite.config import FastTSCodecConfig

    captured = {}

    def _stub_train_fastts_codec(cfg, train_shots, eval_shots, **kw):
        # fast-TS signature: modality is a keyword, NOT a positional after cfg.
        captured["cfg"] = cfg
        captured["train_shots"] = list(train_shots)
        captured["eval_shots"] = list(eval_shots)
        captured["kw"] = kw
        return {"steps": kw["steps"], "global_step": kw["steps"], "best_score": 0.0,
                "best_step": 0}

    monkeypatch.setattr(ft, "train_fastts_codec", _stub_train_fastts_codec)

    # main() writes summary.json into out_dir (the real trainer creates it); the stub does not,
    # so pre-create it here — this test only asserts the DISPATCH, not the trainer's own I/O.
    out_dir = tmp_path / "fastts_dispatch"
    out_dir.mkdir(parents=True, exist_ok=True)

    # avoid a real shot scan: pass explicit --shots so discover_shots is never called.
    argv = [
        "--modality", "filterscopes",
        "--shots", "900000,900001,900002,900003",
        "--eval_n_shots", "1",
        "--n_shots", "3",
        "--steps", "2",
        "--eval_every", "1",
        "--batch_size", "2",
        "--num_workers", "0",
        "--out_dir", str(out_dir),
        "--data_dir", str(tmp_path),
        "--seed", "0",
    ]
    gate = tc.main(argv)
    assert gate["steps"] == 2
    # built the fast-TS config (not a spectro/video/slowts one) with the real channel count.
    assert isinstance(captured["cfg"], FastTSCodecConfig)
    assert captured["cfg"].channels == 8
    # dispatched with the fast-TS keyword signature.
    assert captured["kw"]["modality"] == "filterscopes"
    assert captured["kw"]["steps"] == 2
    assert captured["kw"]["batch_size"] == 2
    # train/eval split derived from the shot list (eval = last 1; train = first 3).
    assert captured["eval_shots"] == ["900003"]
    assert captured["train_shots"] == ["900000", "900001", "900002"]
