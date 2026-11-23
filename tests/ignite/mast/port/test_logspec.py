"""CPU-only tests for the centred log10-power Mirnov spectrogram codec (LogBandDCT)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import cache, paths, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.port import framecache, logspec, rollout
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs as tok_codecs
from tokamak_foundation_model.ignite.mast.tokenizers.grid import GridSet

_CACHE_V1 = paths.cache_dir("v1")
_FIT_V1 = paths.stage1_dir() / "fit" / "v1"
_ASSIGN_V1B = paths.stage1_dir() / "assignments" / "v1b.json"
_LOGSPEC_FIT_V1 = logspec.DEFAULT_LOGSPEC_OUT
_PLASMA_SPLITS = paths.mast_root() / "stage2r" / "frame_codes_v1b_plasma" / "splits.json"
_OMV_NAME = "magnetics-b_field_pol_probe_omv_voltage"
_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "mast"


def _load_script_module(script_name: str):
    path = _SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(f"_test_{path.stem}_logspec", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_synthetic_codec(
    *,
    p: int = 2,
    n_bins_freq: int = 64,
    k: list[int] | None = None,
    with_grids: bool = False,
) -> tuple[logspec.LogBandDCT, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(42)
    k = [32, 32] if k is None else k
    st = transform.ChannelStats(
        "none",
        np.ones((p, n_bins_freq), dtype=np.float64),
        np.zeros((p, n_bins_freq), dtype=np.float64),
        np.ones((p, n_bins_freq), dtype=np.float64),
    )
    floor = np.power(10.0, rng.uniform(-12.0, -4.0, size=(p, n_bins_freq)))
    log_mean = np.log10(floor) + rng.uniform(1.0, 4.0, size=(p, n_bins_freq))
    grids = None
    if with_grids:
        n_tok = p * (n_bins_freq // tok_codecs.BAND_BINS)
        n_val = max(k)
        grids = GridSet(
            lo=np.full((n_tok, n_val), -4.0),
            hi=np.full((n_tok, n_val), 4.0),
            width=np.full((n_tok, n_val), 1.0),
            n_bins=64,
        )
    codec = logspec.LogBandDCT(
        "mirnov-synth", 0.001024, (p, n_bins_freq), st, k=k, floor=floor, log_mean=log_mean, grids=grids
    )
    return codec, floor, log_mean


def test_logbanddct_domain_roundtrip_nan_and_positivity():
    """from_domain(to_domain(x)) == max(x, floor) within 1e-9 rel, NaN kept, decoded finite > 0."""
    codec, floor, _log_mean = _make_synthetic_codec(p=2, n_bins_freq=64, k=[32, 16], with_grids=True)
    assert isinstance(codec, tok_codecs.BandDCT)

    rng = np.random.default_rng(1)
    # Mix values above floor, below floor, zero, negative, and NaN.
    x = np.power(10.0, rng.uniform(-14.0, 2.0, size=(2, 2, 64, 15)))
    x[0, 0, 0, 0] = 0.0
    x[0, 0, 1, 1] = -5.0
    x[1, 1, 10, 3] = np.nan
    x[0, 1, 32:64, :5] = np.nan  # whole band missing in frame 0

    u = codec.to_domain(x)
    assert np.array_equal(np.isnan(u), np.isnan(x))

    rec = codec.from_domain(u)
    assert np.array_equal(np.isnan(rec), np.isnan(x))
    expected = np.maximum(x, floor[None, ..., None])
    fin = np.isfinite(x)
    rel_err = np.max(np.abs(rec[fin] - expected[fin]) / expected[fin])
    assert rel_err < 1e-9

    # Decoded values through quantized roundtrip and framecache are strictly positive wherever finite.
    rt = codec.roundtrip(x, mode="nearest")
    fin_rt = np.isfinite(rt)
    assert fin_rt.any()
    assert (rt[fin_rt] > 0.0).all()
    assert np.isnan(rt[0, 1, 32:64, :5]).all()

    cs = cache.CachedSignal(values=x[0].astype(np.float32), t0=0.0, dt=codec.dt)
    codes, present = framecache.encode_signal_codes(codec, cs, t0=0.0, n_frames=3)
    assert present is True
    dec = framecache.decode_signal_codes(codec, codes, n_samples=15)
    fin_dec = np.isfinite(dec)
    assert fin_dec.any()
    assert (dec[fin_dec] > 0.0).all()


def test_logbanddct_unquantized_k32_reproduces_frame_mean_log_power():
    """encode_domain -> decode_domain unquantized with k=32 reproduces each frame's mean log power."""
    codec, floor, _log_mean = _make_synthetic_codec(p=2, n_bins_freq=64, k=[32, 32], with_grids=False)
    rng = np.random.default_rng(7)
    x = np.power(10.0, rng.uniform(-10.0, 1.0, size=(1, 2, 64, 15)))
    u = codec.to_domain(x)
    vals = codec.encode_domain(u)
    u_rec = codec.decode_domain(vals, n=15)

    # Check every frame reproduces the per-bin frame mean of centred log power.
    from tokamak_foundation_model.ignite.mast.tokenizers import frames

    fi = frames.frame_index(15, codec.dt)
    for f in np.unique(fi):
        sel = fi == f
        expected_u = u[..., sel].mean(axis=-1, keepdims=True)
        assert np.allclose(u_rec[..., sel], expected_u, rtol=1e-10, atol=1e-12)
        expected_phys = np.power(10.0, expected_u + codec.log_mean[None, ..., None])
        dec_phys = codec.from_domain(u_rec[..., sel])
        assert np.allclose(dec_phys, expected_phys, rtol=1e-10, atol=0.0)


def test_fit_save_load_and_manifest_sha_checks(tmp_path: Path):
    """fit_logspec -> save/load -> sha check -> port_build_cache manifest safety check."""
    port_build = _load_script_module("port_build_cache.py")
    p_cnt, n_freq = 3, 256
    k = [32, 16, 16, 16, 16, 16, 16, 16]
    sig_names = tasks.SPECTROGRAM_SIGNALS
    rng = np.random.default_rng(123)

    v1_root = tmp_path / "cache_v1"
    (v1_root / "shots").mkdir(parents=True)
    stats = {
        n: transform.ChannelStats(
            "none",
            np.ones((p_cnt, n_freq), dtype=np.float64),
            np.zeros((p_cnt, n_freq), dtype=np.float64),
            np.ones((p_cnt, n_freq), dtype=np.float64),
        )
        for n in sig_names
    }
    transform.save_stats(stats, v1_root / "stats.npz")
    sha = transform.stats_sha256(v1_root / "stats.npz")
    framecache.save_json_atomic({"stats_sha256": sha}, v1_root / "manifest.json")

    shots = ["1001", "1002", "1003"]
    for s in shots:
        with h5py.File(v1_root / "shots" / f"{s}.h5", "w") as f:
            f.attrs["shot"] = int(s)
            f.attrs["split"] = "train"
            f.attrs["cache_version"] = cache.CACHE_VERSION
            f.attrs["stats_sha256"] = sha
            status = {}
            for n in sig_names:
                arr = np.power(10.0, rng.uniform(-8.0, -2.0, size=(p_cnt, n_freq, 15))).astype(
                    np.float32
                )
                # Make bin 0 of probe 0 non-positive in one shot to exercise floor clipping.
                arr[0, 0, :3] = 0.0
                ds = f.create_dataset(n, data=arr, dtype=np.float32)
                ds.attrs["t0"] = 0.0
                ds.attrs["dt"] = 0.001024
                ds.attrs["dead_nan"] = 0
                ds.attrs["lost"] = 0
                ds.attrs["const_channels"] = np.zeros(0, dtype=np.int64)
                status[n] = "ok"
            f.attrs["status"] = json.dumps(status)

    splits_dir = tmp_path / "splits"
    framecache.save_json_atomic({"train": shots, "val": [], "test": []}, splits_dir / "splits.json")
    assign_path = tmp_path / "assign.json"
    assign.save_assignment(
        {
            n: assign.Setting("band_dct", tuple(k), "band", 64, "")
            for n in sig_names
        },
        assign_path,
    )
    specs = {
        n: tasks.SignalSpec(n, 0.001024, (p_cnt, n_freq), 0.0, 1.0) for n in sig_names
    }
    fit_out = tmp_path / "logspec_fit"
    meta = logspec.fit_logspec(
        v1_root,
        splits_dir,
        assign_path,
        fit_out,
        target_shots=10,
        workers=1,
        signal_specs=specs,
    )
    fit_sha = logspec.logspec_fit_sha256(fit_out)
    assert meta["logspec_fit_sha256"] == fit_sha
    assert (fit_out / "logspec_fit.npz").exists()
    assert (fit_out / "logspec_fit.json").exists()

    codecs = logspec.build_logspec_codecs(fit_out, stats)
    assert set(codecs) == set(sig_names)
    for c in codecs.values():
        assert isinstance(c, logspec.LogBandDCT)
        assert c.n_tok == 24 and c.n_val == 32 and c.grids.n_bins == 64

    # Manifest resolution: absent keys -> None; matching sha -> Path; mismatched sha -> ValueError.
    assert logspec.resolve_manifest_logspec_fit({}) is None
    assert logspec.resolve_manifest_logspec_fit(
        {"logspec_fit_path": None, "logspec_fit_sha256": None}
    ) is None
    assert logspec.resolve_manifest_logspec_fit(
        {"logspec_fit_path": str(fit_out), "logspec_fit_sha256": fit_sha}
    ) == fit_out
    with pytest.raises(ValueError, match="logspec_fit_sha256"):
        logspec.resolve_manifest_logspec_fit(
            {"logspec_fit_path": str(fit_out), "logspec_fit_sha256": "0" * 64}
        )

    # port_build_cache existing-manifest check rejects a different logspec_fit_sha256.
    mpath = tmp_path / "_codec_manifest.json"
    framecache.save_json_atomic(
        {
            "assignment_sha256": "a" * 64,
            "fit_sha256": "b" * 64,
            "stats_sha256": "c" * 64,
            "logspec_fit_sha256": fit_sha,
            "plasma_ip_threshold_A": 50_000.0,
        },
        mpath,
    )
    port_build._check_existing_manifest(
        mpath,
        assignment_sha256="a" * 64,
        fit_sha256="b" * 64,
        stats_sha256="c" * 64,
        plasma_ip_threshold_a=50_000.0,
        logspec_fit_sha256=fit_sha,
    )
    with pytest.raises(ValueError, match="logspec_fit_sha256"):
        port_build._check_existing_manifest(
            mpath,
            assignment_sha256="a" * 64,
            fit_sha256="b" * 64,
            stats_sha256="c" * 64,
            plasma_ip_threshold_a=50_000.0,
            logspec_fit_sha256=None,
        )


def test_real_omv_val_shot_roundtrip_and_log_accuracy():
    """Real MAST OMV val shot: encode/decode matches roundtrip, decoded > 0, median |dlog10| < 0.15."""
    if not (_CACHE_V1 / "stats.npz").exists() or not (_CACHE_V1 / "shots").exists():
        pytest.skip(f"MAST cache v1 not found at {_CACHE_V1}")
    if not (_LOGSPEC_FIT_V1 / "logspec_fit.npz").exists():
        pytest.skip(f"logspec fit not found at {_LOGSPEC_FIT_V1}")
    if not _PLASMA_SPLITS.exists():
        pytest.skip(f"plasma splits.json not found at {_PLASMA_SPLITS}")
    if importlib.util.find_spec("tokamark") is None:
        pytest.skip("tokamark is not importable in this environment")

    signals = registry.all_signals()
    codecs, stats, _assignment = framecache.build_mast_codecs(
        signals, _CACHE_V1, _FIT_V1, _ASSIGN_V1B, logspec_fit=_LOGSPEC_FIT_V1
    )
    codec = codecs[_OMV_NAME]
    assert isinstance(codec, logspec.LogBandDCT)
    sha = json.loads((_CACHE_V1 / "manifest.json").read_text())["stats_sha256"]

    val_shots = sorted(json.loads(_PLASMA_SPLITS.read_text())["val"], key=lambda s: int(s))
    chosen_data = None
    chosen_span = None
    for s in val_shots:
        h5 = _CACHE_V1 / "shots" / f"{s}.h5"
        if not h5.exists():
            continue
        sd = cache.read_shot(h5, [_OMV_NAME, framecache.PLASMA_IP_SIGNAL], stats_sha256=sha)
        cs = sd.get(_OMV_NAME)
        if cs is None or cs.values.shape[-1] == 0 or not np.isfinite(cs.values).any():
            continue
        span, reason = framecache.plasma_frame_range(sd, stats[framecache.PLASMA_IP_SIGNAL])
        if reason != "ok" or span is None or span[4] < framecache.MIN_CACHED_FRAMES:
            continue
        chosen_data, chosen_span = sd, span
        break
    assert chosen_data is not None and chosen_span is not None
    _t0_ref, t0, _f_start, _f_end, n_frames = chosen_span

    cs = chosen_data[_OMV_NAME]
    codes, present = framecache.encode_signal_codes(codec, cs, t0=t0, n_frames=n_frames)
    assert present is True
    assert codes.shape == (n_frames, 768)

    truth_phys, _times_s, sample_f = rollout._align_truth_to_window(
        cs, stats[_OMV_NAME], t0=t0, n_frames=n_frames, dt_frame=framecache.DT_FRAME
    )
    assert truth_phys is not None and sample_f is not None

    dec = framecache.decode_signal_codes(
        codec, codes, dt_frame=framecache.DT_FRAME, sample_frame_idx=sample_f
    )
    assert dec.shape == truth_phys.shape

    # Match codec.roundtrip(..., mode="nearest") at every frame's aligned samples.
    dlog_chunks = []
    for f in np.unique(sample_f):
        sel = sample_f == f
        rt_f = codec.roundtrip(truth_phys[None, ..., sel], mode="nearest")[0]
        dec_f = dec[..., sel]
        fin_rt = np.isfinite(rt_f)
        fin_dec = np.isfinite(dec_f)
        assert np.array_equal(fin_rt, fin_dec), f"frame {f}: NaN mismatch"
        if fin_rt.any():
            log_rt = np.log10(rt_f[fin_rt])
            log_dec = np.log10(dec_f[fin_dec])
            assert np.max(np.abs(log_dec - log_rt)) < 1e-6

        # 5 ms frame mean of log10 power (truth clipped at codec.floor).
        t_sel = truth_phys[..., sel]
        ok_t = np.isfinite(t_sel)
        cnt = ok_t.sum(axis=-1)
        has = cnt > 0
        t_log_s = np.where(ok_t, np.log10(np.maximum(t_sel, codec.floor[..., None])), 0.0)
        t_log_mean = np.divide(
            t_log_s.sum(axis=-1),
            cnt,
            out=np.full(cnt.shape, np.nan, dtype=np.float64),
            where=has,
        )
        r_log = np.log10(dec_f[..., 0])
        both = np.isfinite(t_log_mean) & np.isfinite(r_log)
        if both.any():
            dlog_chunks.append(np.abs(r_log[both] - t_log_mean[both]))

    fin_all = np.isfinite(dec)
    assert fin_all.any()
    assert (dec[fin_all] > 0.0).all()

    med_dlog = float(np.median(np.concatenate(dlog_chunks)))
    assert med_dlog < 0.15, f"median |dlog10| = {med_dlog:.4f} >= 0.15"
