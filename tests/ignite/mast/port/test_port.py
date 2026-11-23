"""CPU-only tests for the IGNITE-on-MAST Phase-B dynamics port."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import ModalitySpec
from tokamak_foundation_model.ignite.mast import cache, paths, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.port import framecache, layout, rollout
from tokamak_foundation_model.ignite.mast.port.layout import ev, td
from tokamak_foundation_model.ignite.mast.tokenizers import codecs as tok_codecs
from tokamak_foundation_model.ignite.mast.tokenizers.grid import GridSet

_CACHE_V1 = paths.cache_dir("v1")
_FIT_V1 = paths.stage1_dir() / "fit" / "v1"
_ASSIGN_V1B = paths.stage1_dir() / "assignments" / "v1b.json"
_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "mast"


def _load_script_module(script_name: str):
    path = _SCRIPTS_DIR / script_name
    mod_name = f"_test_{path.stem}_module"
    spec = importlib.util.spec_from_file_location(mod_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _skip_if_no_cache_v1() -> None:
    if not (_CACHE_V1 / "stats.npz").exists() or not (_CACHE_V1 / "shots").exists():
        pytest.skip(f"MAST cache v1 not found at {_CACHE_V1}")
    if not (_FIT_V1 / "fit.npz").exists() or not _ASSIGN_V1B.exists():
        pytest.skip(f"MAST stage-1 fit/assignment not found under {paths.stage1_dir()}")
    if importlib.util.find_spec("tokamark") is None:
        pytest.skip("tokamark is not importable in this environment")


# ----------------------------------------------------------------------- plasma frame range

def test_plasma_frame_range_rules():
    """Plasma frame range selects |frame_mean(summary-ip)| >= 50 kA and skips invalid/short shots."""
    ip_st = transform.ChannelStats("none", np.ones((1,)), np.zeros((1,)), np.ones((1,)))
    dt = 0.00025  # 20 samples per 5 ms frame

    # Absent summary-ip -> no_summary_ip.
    span, reason = framecache.plasma_frame_range({}, ip_st)
    assert span is None and reason == "no_summary_ip"

    # All-NaN summary-ip -> no_summary_ip.
    nan_cs = cache.CachedSignal(np.full((1, 40), np.nan, dtype=np.float32), t0=-0.05, dt=dt)
    span, reason = framecache.plasma_frame_range({"summary-ip": nan_cs}, ip_st)
    assert span is None and reason == "no_summary_ip"

    # Below 50 kA everywhere -> no_plasma_above_ip_threshold.
    low_cs = cache.CachedSignal(np.full((1, 200), 10_000.0, dtype=np.float32), t0=-0.05, dt=dt)
    span, reason = framecache.plasma_frame_range({"summary-ip": low_cs}, ip_st)
    assert span is None and reason == "no_plasma_above_ip_threshold"

    # 5 pre-plasma frames (<50 kA), 12 plasma frames (-80 kA, tests |Ip|), 4 post-termination frames.
    n_tot = 21
    ip_vals = np.full((1, n_tot * 20), 5_000.0, dtype=np.float32)
    ip_vals[0, 5 * 20 : 17 * 20] = -80_000.0
    cs = cache.CachedSignal(ip_vals, t0=-0.05, dt=dt)
    span, reason = framecache.plasma_frame_range({"summary-ip": cs}, ip_st)
    assert reason == "ok" and span is not None
    t0_ref, t0, f_start, f_end, n_frames = span
    assert t0_ref == pytest.approx(-0.05)
    assert f_start == 5 and f_end == 16 and n_frames == 12
    assert t0 == pytest.approx(-0.05 + 5 * framecache.DT_FRAME)


# ----------------------------------------------------------------------- real-shot roundtrip (f_start > 0)

def test_real_shot_entry_and_roundtrip():
    """A real shot with f_start > 0 encodes cleanly and decodes matching codec.roundtrip for every codec type."""
    _skip_if_no_cache_v1()

    signals = registry.all_signals()
    act_specs = framecache.actuator_specs()
    codecs, stats, _assignment = framecache.build_mast_codecs(
        signals, _CACHE_V1, _FIT_V1, _ASSIGN_V1B
    )
    modality_order = tuple(s.name for s in signals if s.name in codecs)
    sha = json.loads((_CACHE_V1 / "manifest.json").read_text())["stats_sha256"]

    # Pick a real shot with f_start > 0 that has FrameDCT, LinearCodec, and a Mirnov BandDCT present.
    shot_files = sorted((_CACHE_V1 / "shots").glob("*.h5"))
    chosen_path = None
    chosen_data = None
    chosen_span = None
    mirnov_names = set(tasks.SPECTROGRAM_SIGNALS)
    for sp in shot_files[:40]:
        sd = cache.read_shot(sp, list(modality_order), stats_sha256=sha)
        span, reason = framecache.plasma_frame_range(sd, stats[framecache.PLASMA_IP_SIGNAL])
        if reason != "ok" or span is None:
            continue
        _t0_ref, _t0, f_start, _f_end, n_frames = span
        if f_start <= 0 or n_frames < framecache.MIN_CACHED_FRAMES:
            continue
        has_mirnov = any(
            sd.get(m) is not None and sd[m].values.shape[-1] > 0 and np.isfinite(sd[m].values).any()
            for m in mirnov_names
        )
        has_linear = any(
            isinstance(codecs[n], tok_codecs.LinearCodec)
            and sd.get(n) is not None
            and sd[n].values.shape[-1] > 0
            and np.isfinite(sd[n].values).any()
            for n in modality_order
        )
        if has_mirnov and has_linear:
            chosen_path, chosen_data, chosen_span = sp, sd, span
            break
    assert chosen_path is not None and chosen_data is not None and chosen_span is not None
    t0_ref, t0, f_start, _f_end, n_frames = chosen_span
    assert f_start > 0

    tm = framecache.shot_training_moments(
        chosen_path, act_specs, modality_order, stats, stats_sha256=sha
    )
    assert tm is not None
    act_moments, sig_moments = tm
    act_stats = framecache.finalize_actuator_stats(act_specs, [act_moments], stats=stats)
    sig_stds = framecache.finalize_signal_physical_std(modality_order, [sig_moments])
    assert sig_stds["summary-ip"] > 0.0

    entry, reason = framecache.encode_shot(
        chosen_path,
        codecs,
        modality_order,
        act_specs,
        act_stats,
        stats,
        stats_sha256=sha,
    )
    assert reason == "ok" and entry is not None
    assert entry["n_frames"] == n_frames
    assert entry["t0"] == pytest.approx(t0)
    assert entry["actuators"].dtype == torch.float16
    assert entry["actuators"].shape == (n_frames, act_stats.total_dim)
    assert torch.isfinite(entry["actuators"]).all()
    assert entry["actuator_valid"].dtype == torch.bool
    invalid_act = ~entry["actuator_valid"]
    if invalid_act.any():
        assert (entry["actuators"][invalid_act] == 0).all()

    checked_types: set[str] = set()
    checked_roundtrip = 0
    for name in modality_order:
        codec = codecs[name]
        codes = entry["codes"][name]
        expected_n_tok = codec.n_tok * codec.n_val
        expected_vocab = codec.grids.n_bins + 1
        assert codes.dtype == torch.int32
        assert codes.shape == (n_frames, expected_n_tok)
        assert entry["vocabs"][name] == expected_vocab
        assert int(codes.min()) >= 0 and int(codes.max()) < expected_vocab

        if not entry["present"][name]:
            assert (codes == framecache.PLACEHOLDER_CODE).all()
            continue

        cs = chosen_data[name]
        assert cs is not None
        truth_phys, times_s, sample_f = rollout._align_truth_to_window(
            cs, stats[name], t0=t0, n_frames=n_frames, dt_frame=framecache.DT_FRAME
        )
        assert truth_phys is not None and times_s is not None and sample_f is not None

        # Every aligned sample's centre time must lie inside its assigned 5 ms frame.
        frame_lo = t0 + sample_f * framecache.DT_FRAME
        frame_hi = frame_lo + framecache.DT_FRAME
        assert np.all(times_s >= frame_lo - 1e-12) and np.all(times_s < frame_hi + 1e-12), (
            f"{name}: sample times fall outside assigned frame boundaries"
        )

        dec, _sf = rollout._decode_aligned(codec, codes, sample_f, n_frames, framecache.DT_FRAME)
        assert dec.shape == truth_phys.shape

        if isinstance(codec, tok_codecs.BandDCT):
            # Mirnov has +1.022 ms start offset and alternating 4/5-sample frames:
            # verify every frame matches codec.roundtrip(..., mode="nearest") on that frame's samples.
            frame_counts = np.bincount(sample_f, minlength=n_frames)
            assert (frame_counts == 4).any() and (frame_counts == 5).any(), (
                f"{name}: expected alternating 4/5-sample Mirnov frames"
            )
            for f in np.unique(sample_f):
                sel = sample_f == f
                rt_f = codec.roundtrip(truth_phys[None, ..., sel], mode="nearest")[0]
                dec_f = dec[..., sel]
                fin_rt = np.isfinite(rt_f)
                fin_dec = np.isfinite(dec_f)
                assert np.array_equal(fin_rt, fin_dec), f"{name} frame {f}: NaN mismatch"
                if fin_rt.any():
                    scale = float(np.std(rt_f[fin_rt])) + float(np.max(np.abs(rt_f[fin_rt]))) + 1e-30
                    rel = float(np.max(np.abs(dec_f[fin_rt] - rt_f[fin_rt]))) / scale
                    assert rel < 1e-6, f"{name} frame {f}: rel diff {rel:.3e}"
        else:
            # FrameDCT and LinearCodec share t0_ref with summary-ip: exact slice match with full roundtrip.
            x_full = transform.to_physical(cs.values, stats[name])[None]
            rt_full = codec.roundtrip(x_full, mode="nearest")[0]
            fi_full = framecache.sample_frame_index(
                cs.values.shape[-1], cs.t0, cs.dt, t0, dt_frame=framecache.DT_FRAME
            )
            keep = (fi_full >= 0) & (fi_full < n_frames)
            rt_win = rt_full[..., keep]
            fin_rt = np.isfinite(rt_win)
            fin_dec = np.isfinite(dec)
            assert np.array_equal(fin_rt, fin_dec), f"{name}: NaN pattern mismatch"
            if fin_rt.any():
                rel = float(np.max(np.abs(dec[fin_rt] - rt_win[fin_rt]))) / (
                    float(np.std(rt_win[fin_rt])) + 1e-12
                )
                assert rel < 1e-6, f"{name}: decode differs from codec.roundtrip by {rel:.3e}"

        checked_types.add(type(codec).__name__)
        checked_roundtrip += 1

    assert checked_roundtrip > 20
    assert {"FrameDCT", "LinearCodec", "BandDCT"} <= checked_types


# ----------------------------------------------------------------------- dropped BandDCT neutralization

def test_decode_neutralizes_dropped_banddct_positions():
    """Missing code in a BandDCT dropped slot does not wipe the band; missing code in a kept slot still gives NaN."""
    p, n_bins_freq = 2, 64  # 2 probes, 2 bands of 32 bins
    st = transform.ChannelStats(
        "none",
        np.ones((p, n_bins_freq), dtype=np.float64),
        np.zeros((p, n_bins_freq), dtype=np.float64),
        np.ones((p, n_bins_freq), dtype=np.float64),
    )
    k = [32, 16]  # band 0 keeps 32 coeffs, band 1 keeps 16 and drops 16..31
    n_tok = p * 2
    n_val = max(k)
    n_grid_bins = 16
    grids = GridSet(
        lo=np.full((n_tok, n_val), -2.0),
        hi=np.full((n_tok, n_val), 2.0),
        width=np.full((n_tok, n_val), 1.0),
        n_bins=n_grid_bins,
    )
    codec = tok_codecs.BandDCT("mirnov-test", 0.001024, (p, n_bins_freq), st, k=k, grids=grids)
    missing_code = n_grid_bins

    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 1.0, size=(p, n_bins_freq, 10)).astype(np.float32)
    cs = cache.CachedSignal(values=x, t0=0.0, dt=0.001024)
    codes, present = framecache.encode_signal_codes(codec, cs, t0=0.0, n_frames=2)
    assert present is True
    clean_dec = framecache.decode_signal_codes(codec, codes, n_samples=10)
    assert np.isfinite(clean_dec).all()

    # Corrupt a DROPPED position: probe 0, band 1 (tok=1), val=20 (>= k[1]=16).
    corrupt_dropped = codes.clone().reshape(2, n_tok, n_val)
    assert not codec._keep[1, 20]
    corrupt_dropped[0, 1, 20] = missing_code
    dec_dropped = framecache.decode_signal_codes(
        codec, corrupt_dropped.reshape(2, n_tok * n_val), n_samples=10
    )
    assert np.isfinite(dec_dropped).all()
    assert np.allclose(dec_dropped, clean_dec)

    # Corrupt a KEPT position: probe 0, band 1 (tok=1), val=0 (< k[1]=16).
    corrupt_kept = codes.clone().reshape(2, n_tok, n_val)
    assert codec._keep[1, 0]
    corrupt_kept[0, 1, 0] = missing_code
    dec_kept = framecache.decode_signal_codes(
        codec, corrupt_kept.reshape(2, n_tok * n_val), n_samples=10
    )
    # Probe 0, band 1 (freq bins 32..63) in frame 0 (first 5 samples) must be NaN;
    # probe 0, band 0 (freq bins 0..31) must remain finite.
    assert np.isnan(dec_kept[0, 32:64, :5]).all()
    assert np.isfinite(dec_kept[0, 0:32, :5]).all()
    assert np.isfinite(dec_kept[0, 32:64, 5:]).all()


# ----------------------------------------------------------------------- counterfactual actuators in physical units

def test_counterfactual_actuators_physical_units(tmp_path: Path):
    """set:/scale:/ramp: act in physical units for linear actuators and reject asinh actuators."""
    cache_dir = tmp_path / "cf_cache"
    cache_dir.mkdir()

    nbi_st = transform.ChannelStats("none", np.array([1.0]), np.array([699_448.0]), np.array([1_088_119.0]))
    pf_st = transform.ChannelStats(
        "asinh",
        np.array([500.0, 100.0]),
        np.array([-0.3, 0.2]),
        np.array([1.1, 0.9]),
    )
    stats = {"summary-power_nbi": nbi_st, "pf_active-coil_voltage": pf_st}
    act_base = framecache.ActuatorStats(
        names=("summary-power_nbi", "pf_active-coil_voltage"),
        channels=(1, 2),
        mu=np.array([0.15, -0.1, 0.2], dtype=np.float64),
        sd=np.array([0.85, 1.05, 0.95], dtype=np.float64),
        n_train_shots=10,
        n_valid_frames=np.array([100, 100, 100], dtype=np.int64),
    )
    comp, mp, sp = framecache.compose_physical_actuator_stats(act_base, stats)
    act_stats = framecache.ActuatorStats(
        names=act_base.names,
        channels=act_base.channels,
        mu=act_base.mu,
        sd=act_base.sd,
        n_train_shots=act_base.n_train_shots,
        n_valid_frames=act_base.n_valid_frames,
        compress=comp,
        mu_phys=mp,
        sd_phys=sp,
    )
    manifest = {
        "t0_start": 0.0,
        "dt_frame": 0.005,
        "modalities": {"summary-ip": {"family": "slowts", "n_tok": 1, "codebook_size": 17}},
        "actuators": act_stats.to_manifest_dict(),
    }
    framecache.save_json_atomic(manifest, cache_dir / "_codec_manifest.json")

    # Build a synthetic actuator sequence with known physical NBI power [1.0, 1.2, ..., 2.8] MW.
    n_frames, k0 = 10, 4
    nbi_phys_orig = np.linspace(1.0e6, 2.8e6, n_frames, dtype=np.float64)
    nbi_model = transform.to_model(nbi_phys_orig[None, :], nbi_st)[0]
    nbi_z = (nbi_model - act_stats.mu[0]) / act_stats.sd[0]
    act = torch.zeros((n_frames, 3), dtype=torch.float32)
    act[:, 0] = torch.from_numpy(nbi_z.astype(np.float32))
    act[:, 1:] = 0.5

    with layout.patched_ignite(cache_dir):
        # set:summary-power_nbi=0@4 -> 0 W from frame 4 on, unchanged in 0..3.
        ed_set0 = ev.apply_actuator_mode(act, f"set:summary-power_nbi=0@{k0}", k0, F=n_frames)
        nbi_set0 = rollout.counterfactual_nbi_physical(ed_set0, act_stats, stats)
        assert nbi_set0 is not None
        assert np.allclose(nbi_set0[:k0], nbi_phys_orig[:k0], rtol=1e-5, atol=1.0)
        assert np.allclose(nbi_set0[k0:], 0.0, atol=1.0)

        # scale:summary-power_nbi=0 -> 0 W from frame k0 on.
        ed_scale0 = ev.apply_actuator_mode(act, "scale:summary-power_nbi=0", k0, F=n_frames)
        nbi_scale0 = rollout.counterfactual_nbi_physical(ed_scale0, act_stats, stats)
        assert nbi_scale0 is not None
        assert np.allclose(nbi_scale0[:k0], nbi_phys_orig[:k0], rtol=1e-5, atol=1.0)
        assert np.allclose(nbi_scale0[k0:], 0.0, atol=1.0)

        # set to 1 MW and scale by 0.5 in physical units.
        ed_set1m = ev.apply_actuator_mode(act, f"set:summary-power_nbi=1e6@{k0}", k0, F=n_frames)
        nbi_set1m = rollout.counterfactual_nbi_physical(ed_set1m, act_stats, stats)
        assert np.allclose(nbi_set1m[k0:], 1.0e6, rtol=1e-5, atol=1.0)

        ed_half = ev.apply_actuator_mode(act, f"scale:summary-power_nbi=0.5@{k0}", k0, F=n_frames)
        nbi_half = rollout.counterfactual_nbi_physical(ed_half, act_stats, stats)
        assert np.allclose(nbi_half[k0:], 0.5 * nbi_phys_orig[k0:], rtol=1e-5, atol=1.0)

        # asinh actuator must raise clear ValueError on scale:, ramp:, set:.
        for bad_mode in (
            "scale:pf_active-coil_voltage=0",
            f"ramp:pf_active-coil_voltage=0.5@{k0}/2",
            f"set:pf_active-coil_voltage=0@{k0}",
        ):
            with pytest.raises(ValueError, match="asinh"):
                ev.apply_actuator_mode(act, bad_mode, k0, F=n_frames)


# ----------------------------------------------------------------------- rollout metrics (global std & all-NaN)

def test_rollout_metrics_global_std_and_all_nan_coverage(tmp_path: Path):
    """Metrics use global physical std, record coverage/missing_frac, and never drop all-NaN rollouts."""
    port_rollout_plots = _load_script_module("port_rollout_plots.py")

    dt = 0.005
    n_frames, k0 = 10, 4
    n_bins = 16
    st = transform.ChannelStats("none", np.array([1.0]), np.array([0.0]), np.array([2.0]))
    grids = GridSet(
        lo=np.full((1, 1), -3.0), hi=np.full((1, 1), 3.0), width=np.full((1, 1), 1.0), n_bins=n_bins
    )
    codec = tok_codecs.FrameDCT("summary-ip", dt, (1,), st, k=1, domain="linear", grids=grids, signal_std=2.0)

    # Constant ground truth in predicted window (per-window std == 0!): global std must keep NRMSE finite.
    h5_path = tmp_path / "shot.h5"
    vals = np.full((1, n_frames), 0.5, dtype=np.float32)  # physical = 1.0
    with h5py.File(h5_path, "w") as f:
        f.attrs["shot"] = 1
        f.attrs["split"] = "val"
        f.attrs["cache_version"] = cache.CACHE_VERSION
        f.attrs["stats_sha256"] = "0" * 64
        ds = f.create_dataset("summary-ip", data=vals, dtype=np.float32)
        ds.attrs["t0"] = 0.0
        ds.attrs["dt"] = dt
        ds.attrs["dead_nan"] = 0
        ds.attrs["lost"] = 0
        ds.attrs["const_channels"] = np.zeros(0, dtype=np.int64)
        f.attrs["status"] = json.dumps({"summary-ip": "ok"})

    cs = cache.CachedSignal(vals, t0=0.0, dt=dt)
    gt_codes, _ = framecache.encode_signal_codes(codec, cs, t0=0.0, n_frames=n_frames)

    # Shot 1: exact prediction -> finite NRMSE using global std = 2.0, coverage = 1.0, missing_frac = 0.0.
    ro1 = ev.Rollout(gt_codes={"summary-ip": gt_codes}, pred_codes={"summary-ip": gt_codes.clone()}, k0=k0, n_frames=n_frames)
    _dec1, m1 = rollout.decode_and_score_rollout(
        ro1, h5_path, {"summary-ip": codec}, {"summary-ip": st}, t0=0.0, signal_stds={"summary-ip": 2.0}
    )
    s1 = m1["summary-ip"]
    assert math.isfinite(s1["rollout_nrmse"]) and math.isfinite(s1["ceiling_nrmse"])
    assert s1["rollout_nrmse"] == pytest.approx(s1["ceiling_nrmse"])
    assert s1["pred_coverage"] == pytest.approx(1.0)
    assert s1["ceil_coverage"] == pytest.approx(1.0)
    assert s1["pred_missing_frac"] == pytest.approx(0.0)
    assert s1["ceil_missing_frac"] == pytest.approx(0.0)

    # Shot 2: all-NaN rollout (missing_code everywhere in predicted frames) -> coverage 0.0, never dropped.
    all_miss = gt_codes.clone()
    all_miss[k0:] = n_bins  # missing_code
    ro2 = ev.Rollout(gt_codes={"summary-ip": gt_codes}, pred_codes={"summary-ip": all_miss}, k0=k0, n_frames=n_frames)
    _dec2, m2 = rollout.decode_and_score_rollout(
        ro2, h5_path, {"summary-ip": codec}, {"summary-ip": st}, t0=0.0, signal_stds={"summary-ip": 2.0}
    )
    s2 = m2["summary-ip"]
    assert math.isnan(s2["rollout_nrmse"])
    assert math.isfinite(s2["ceiling_nrmse"])
    assert s2["pred_coverage"] == pytest.approx(0.0)
    assert s2["pred_missing_frac"] == pytest.approx(1.0)

    summary = port_rollout_plots.summarize_rollout_metrics(
        {"1": {"signals": m1}, "2": {"signals": m2}}
    )
    assert "summary-ip" in summary
    assert summary["summary-ip"]["n_shots"] == 2
    assert summary["summary-ip"]["median_pred_coverage"] == pytest.approx(0.5)
    assert summary["summary-ip"]["median_pred_missing_frac"] == pytest.approx(0.5)

    # All shots all-NaN: signal is still kept with coverage 0.0, never dropped.
    summary_all_nan = port_rollout_plots.summarize_rollout_metrics({"2": {"signals": m2}})
    assert "summary-ip" in summary_all_nan
    assert summary_all_nan["summary-ip"]["n_shots"] == 1
    assert summary_all_nan["summary-ip"]["median_pred_coverage"] == pytest.approx(0.0)
    assert math.isnan(summary_all_nan["summary-ip"]["median_rollout_nrmse"])


# ----------------------------------------------------------------------- layout & split patch

def test_layout_patch_manifest_order(tmp_path: Path):
    """Layout patch supplies MAST ModalitySpecs in manifest order and enforces splits.json."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    manifest = {
        "t0_start": 0.0,
        "dt_frame": 0.005,
        "modalities": {
            "summary-ip": {"family": "slowts", "n_tok": 4, "codebook_size": 65},
            "spectrometer_visible-filter_spectrometer_dalpha_voltage": {
                "family": "fastts",
                "n_tok": 6,
                "codebook_size": 65,
            },
            "magnetics-b_field_pol_probe_omv_voltage": {
                "family": "spectro",
                "n_tok": 768,
                "codebook_size": 65,
            },
        },
        "actuators": {
            "names": ["gas_injection-total_injected", "pf_active-coil_voltage"],
            "channels": [1, 4],
            "total_dim": 5,
            "mu": [0.0, 0.0, 0.0, 0.0, 0.0],
            "sd": [1.0, 1.0, 1.0, 1.0, 1.0],
            "n_train_shots": 2,
            "n_valid_frames": [20, 20, 20, 20, 20],
        },
    }
    (cache_dir / "_codec_manifest.json").write_text(json.dumps(manifest))
    (cache_dir / "splits.json").write_text(
        json.dumps({"train": ["1001", "1002"], "val": ["1003"], "test": ["1004"]})
    )
    for shot in ("1001", "1002", "1003", "1004"):
        torch.save({"n_frames": 12}, cache_dir / f"{shot}.pt")

    orig_fn = td.cache_modality_specs
    with layout.patched_ignite(cache_dir):
        specs = td.cache_modality_specs(cache_dir)
        assert specs == (
            ModalitySpec("summary-ip", "slowts", 4, 65),
            ModalitySpec(
                "spectrometer_visible-filter_spectrometer_dalpha_voltage", "fastts", 6, 65
            ),
            ModalitySpec("magnetics-b_field_pol_probe_omv_voltage", "spectro", 768, 65),
        )
        assert td._ACT_SPEC == (
            ("gas_injection-total_injected", 1),
            ("pf_active-coil_voltage", 4),
        )
        out_dir = tmp_path / "run"
        tr, va, te = td.resolve_split(cache_dir, out_dir, val_n=1, write=True)
        assert tr == ["1001", "1002"]
        assert va == ["1003"]
        assert te == ["1004"]
        assert not (set(te) & (set(tr) | set(va)))
        assert (out_dir / "split_shots.json").exists()
    assert td.cache_modality_specs is orig_fn


# ----------------------------------------------------------------------- port_train --live_end_json refusal

def test_port_train_refuses_live_end_json():
    """port_train.py refuses --live_end_json with a clear error message."""
    port_train = _load_script_module("port_train.py")
    for argv in (
        ["--cache_dir", "/tmp/c", "--live_end_json", "/tmp/live.json"],
        ["--cache_dir", "/tmp/c", "--live_end_json=/tmp/live.json"],
    ):
        with pytest.raises(ValueError, match="live_end_json"):
            port_train.run(argv)


# ----------------------------------------------------------------------- cache safety (manifest mismatch & error exit)

def test_cache_safety_manifest_mismatch_and_error_exit(tmp_path: Path, monkeypatch):
    """Resume without --force rejects manifest mismatches; shot errors exit non-zero and log messages."""
    port_build = _load_script_module("port_build_cache.py")

    base_manifest = {
        "assignment_sha256": "a" * 64,
        "fit_sha256": "b" * 64,
        "stats_sha256": "c" * 64,
        "plasma_ip_threshold_A": 50_000.0,
        "actuators": {"n_train_shots": 100},
    }
    mpath = tmp_path / "_codec_manifest.json"
    framecache.save_json_atomic(base_manifest, mpath)

    # Matching manifest passes cleanly.
    port_build._check_existing_manifest(
        mpath,
        assignment_sha256="a" * 64,
        fit_sha256="b" * 64,
        stats_sha256="c" * 64,
        plasma_ip_threshold_a=50_000.0,
        n_train_shots=100,
    )

    # Every guarded field mismatch must raise ValueError.
    for key, overrides in (
        ("assignment_sha256", {"assignment_sha256": "x" * 64}),
        ("fit_sha256", {"fit_sha256": "y" * 64}),
        ("stats_sha256", {"stats_sha256": "z" * 64}),
        ("plasma_ip_threshold_A", {"plasma_ip_threshold_a": 25_000.0}),
        ("actuators.n_train_shots", {"n_train_shots": 34}),
    ):
        kwargs = dict(
            assignment_sha256="a" * 64,
            fit_sha256="b" * 64,
            stats_sha256="c" * 64,
            plasma_ip_threshold_a=50_000.0,
            n_train_shots=100,
        )
        kwargs.update(overrides)
        with pytest.raises(ValueError, match=key):
            port_build._check_existing_manifest(mpath, **kwargs)

    # _encode_job captures exception type + message (truncated to 200 chars), and main() exits non-zero.
    def _boom(*_a, **_kw):
        raise RuntimeError("x" * 300)

    monkeypatch.setattr(framecache, "encode_shot", _boom)
    port_build._ENCODE_STATE = ({}, (), (), None, {}, None, tmp_path, True)
    res = port_build._encode_job(("9999", "train", str(tmp_path / "9999.h5")))
    assert res[2] == "error:RuntimeError"
    assert res[5] is not None and len(res[5]) == 200 and res[5].startswith("RuntimeError: ")

    monkeypatch.setattr(
        port_build,
        "build_cache",
        lambda **kw: {"counts": {"error:RuntimeError": 1}, "n_errors": 1, "splits": {}, "n_frames": {}},
    )
    rc = port_build.main(["--out", str(tmp_path / "out")])
    assert rc != 0


# ----------------------------------------------------------------------- partial missing & presence

def test_partial_nan_uses_missing_code_and_presence_distinguishes_constant(tmp_path: Path):
    """Partial NaN in a present signal maps to missing_code (not 0), and constant-present stays present."""
    st = transform.ChannelStats("none", np.ones((2,)), np.zeros((2,)), np.ones((2,)))
    grids = GridSet(
        lo=np.full((2, 1), -2.0),
        hi=np.full((2, 1), 2.0),
        width=np.full((2, 1), 1.0),
        n_bins=16,
    )
    codec = tok_codecs.FrameDCT("sig-a", 0.005, (2,), st, k=1, domain="linear", grids=grids)

    vals = np.full((2, 10), 0.5, dtype=np.float32)
    vals[1, :] = np.nan  # channel 1 dead across all frames
    cs = cache.CachedSignal(values=vals, t0=0.0, dt=0.005)

    codes, is_present = framecache.encode_signal_codes(codec, cs, t0=0.0, n_frames=10)
    assert is_present is True
    assert (codes[:, 1] == 16).all()  # missing_code = n_bins = 16, NOT 0
    assert (codes[:, 0] < 16).all()

    dec = framecache.decode_signal_codes(codec, codes, n_samples=10)
    assert np.isfinite(dec[0]).all()
    assert np.isnan(dec[1]).all()

    # Exact presence distinguishes a constant-present shot from an absent shot.
    cdir = tmp_path / "pres"
    cdir.mkdir()
    torch.save(
        {"codes": {"sig-a": torch.full((10, 2), 8, dtype=torch.int32)}, "present": {"sig-a": True}},
        cdir / "1.pt",
    )
    torch.save(
        {"codes": {"sig-a": torch.full((10, 2), 0, dtype=torch.int32)}, "present": {"sig-a": False}},
        cdir / "2.pt",
    )
    with layout.patched_ignite():
        pres = td.build_presence(cdir, out_path=cdir / "_presence.json")
    assert pres == {"1": {"sig-a": True}, "2": {"sig-a": False}}


# ----------------------------------------------------------------------- 3-shot train + rollout + plot

def _make_synthetic_three_shot_fixture(tmp_path: Path):
    """Build a tiny 3-shot MAST cache (2 train, 1 val) + synthetic cache-v1 HDF5s and codecs."""
    n_pre = 3
    n_plasma = 14
    n_tot = n_pre + n_plasma
    dt = 0.005
    n_bins = 16
    rng = np.random.default_rng(7)

    sig_specs = [
        tasks.SignalSpec("summary-ip", dt, (1,), 3e5, 1e5),
        tasks.SignalSpec("interferometer-n_e_line", dt, (1,), 3e19, 1e19),
        tasks.SignalSpec("equilibrium-beta_pol", dt, (1,), 0.2, 0.1),
        tasks.SignalSpec("summary-power_nbi", dt, (1,), 1e6, 5e5),
    ]
    act_specs = (sig_specs[-1],)
    modality_order = tuple(s.name for s in sig_specs)

    stats = {
        s.name: transform.ChannelStats(
            "none",
            np.ones(s.values_shape, dtype=np.float64),
            np.full(s.values_shape, s.mean, dtype=np.float64),
            np.full(s.values_shape, s.std, dtype=np.float64),
        )
        for s in sig_specs
    }
    codecs = {
        s.name: tok_codecs.FrameDCT(
            s.name,
            s.dt,
            s.values_shape,
            stats[s.name],
            k=1,
            domain="linear",
            grids=GridSet(
                lo=np.full((1, 1), -3.5),
                hi=np.full((1, 1), 3.5),
                width=np.full((1, 1), 1.0),
                n_bins=n_bins,
            ),
            signal_mean=s.mean,
            signal_std=s.std,
        )
        for s in sig_specs
    }

    v1_root = tmp_path / "cache_v1"
    (v1_root / "shots").mkdir(parents=True)
    transform.save_stats(stats, v1_root / "stats.npz")
    sha = transform.stats_sha256(v1_root / "stats.npz")

    shot_splits = [("1001", "train"), ("1002", "train"), ("1003", "val")]
    t_axis = np.linspace(0.0, 2.0 * np.pi, n_tot)
    for idx, (shot, split) in enumerate(shot_splits):
        h5_path = v1_root / "shots" / f"{shot}.h5"
        status = {}
        with h5py.File(h5_path, "w") as f:
            f.attrs["shot"] = int(shot)
            f.attrs["split"] = split
            f.attrs["cache_version"] = cache.CACHE_VERSION
            f.attrs["stats_sha256"] = sha
            for s_idx, spec in enumerate(sig_specs):
                if shot == "1002" and spec.name == "equilibrium-beta_pol":
                    status[spec.name] = "absent"
                    continue
                wave = (
                    np.sin(t_axis + 0.4 * idx + 0.3 * s_idx)
                    + 0.1 * rng.standard_normal(n_tot)
                ).astype(np.float32)[None, :]
                if spec.name == "summary-ip":
                    # First n_pre frames below 50 kA (physical = 0 A -> model = -3.0),
                    # remaining n_plasma frames well above 50 kA -> f_start = n_pre > 0.
                    wave[0, :n_pre] = -3.0
                ds = f.create_dataset(spec.name, data=wave, dtype=np.float32)
                ds.attrs["t0"] = -0.015
                ds.attrs["dt"] = dt
                ds.attrs["dead_nan"] = 0
                ds.attrs["lost"] = 0
                ds.attrs["const_channels"] = np.zeros(0, dtype=np.int64)
                status[spec.name] = "ok"
            f.attrs["status"] = json.dumps(status)

    # Fit actuator and physical-std stats on training shots only.
    pairs = [
        framecache.shot_training_moments(
            v1_root / "shots" / f"{shot}.h5",
            act_specs,
            modality_order,
            stats,
            stats_sha256=sha,
        )
        for shot, split in shot_splits
        if split == "train"
    ]
    act_stats = framecache.finalize_actuator_stats(
        act_specs, [a for a, _s in pairs], stats=stats
    )
    sig_stds = framecache.finalize_signal_physical_std(
        modality_order, [s for _a, s in pairs]
    )

    fc_dir = tmp_path / "frame_codes"
    fc_dir.mkdir()
    presence = {}
    for shot, _split in shot_splits:
        entry, reason = framecache.encode_shot(
            v1_root / "shots" / f"{shot}.h5",
            codecs,
            modality_order,
            act_specs,
            act_stats,
            stats,
            stats_sha256=sha,
        )
        assert reason == "ok" and entry is not None
        assert entry["n_frames"] == n_plasma
        assert entry["t0"] == pytest.approx(0.0)
        framecache.save_pt_atomic(entry, fc_dir / f"{shot}.pt")
        presence[shot] = entry["present"]

    manifest = framecache.build_manifest(
        sig_specs,
        codecs,
        act_stats,
        assignment_path=tmp_path / "dummy.json",
        assignment_sha256="0" * 64,
        fit_dir=tmp_path / "fit",
        fit_sha256="1" * 64,
        cache_root=v1_root,
        stats_sha256=sha,
        stats=stats,
        signal_physical_std=sig_stds,
    )
    framecache.save_json_atomic(manifest, fc_dir / "_codec_manifest.json")
    framecache.save_json_atomic(
        {"train": ["1001", "1002"], "val": ["1003"], "test": []},
        fc_dir / "splits.json",
    )
    framecache.save_json_atomic(
        {"cache_dir": str(fc_dir), "presence": presence},
        fc_dir / "_presence.json",
    )
    return v1_root, fc_dir, codecs, stats, act_stats, sig_stds, sha


# train_dynamics.py:2742 calls torch.autocast("cuda", dtype=bfloat16) without enabled=cuda;
# on CPU that unmodified upstream line emits one harmless UserWarning.
@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_end_to_end_train_rollout_plot(tmp_path: Path):
    """Tiny d_model=64 depth=2 trains through port_train.py, rolls out (real & set:0), and writes PNGs."""
    v1_root, fc_dir, codecs, stats, act_stats, sig_stds, sha = _make_synthetic_three_shot_fixture(tmp_path)
    run_dir = tmp_path / "run"
    port_train = _load_script_module("port_train.py")

    try:
        port_train.run(
            [
                "--cache_dir", str(fc_dir),
                "--out_dir", str(run_dir),
                "--d_model", "64",
                "--depth", "2",
                "--n_heads", "4",
                "--k0_seed", "4",
                "--n_predict", "6",
                "--steps", "4",
                "--batch_size", "2",
                "--ckpt_every", "2",
                "--num_workers", "0",
                "--grad_ckpt", "0",
                "--ss_final_frac", "0.0",
            ]
        )

        ckpt_path = run_dir / "dynamics_latest.pt"
        assert ckpt_path.exists()
        assert (run_dir / "dynamics_best.pt").exists()
        assert (run_dir / "split_shots.json").exists()
        split_rec = json.loads((run_dir / "split_shots.json").read_text())
        assert split_rec["train"] == ["1001", "1002"]
        assert split_rec["val"] == ["1003"]
        assert split_rec["test"] == []

        device = torch.device("cpu")
        model, cfg, step = rollout.load_mast_model(ckpt_path, fc_dir, device)
        assert step == 4
        assert cfg.d_model == 64 and cfg.depth == 2
        assert cfg.actuator_dim == 1

        plots_dir = tmp_path / "plots"
        for mode in ("real", "set:summary-power_nbi=0@4"):
            ro, entry, edited_act = rollout.rollout_mast_shot(
                model,
                cfg,
                fc_dir,
                "1003",
                k0=4,
                horizon=6,
                temperature=1.0,
                seed=0,
                device=device,
                actuator_mode=mode,
            )
            assert ro.k0 == 4 and ro.n_frames == 10
            decoded, sig_metrics = rollout.decode_and_score_rollout(
                ro,
                v1_root / "shots" / "1003.h5",
                codecs,
                stats,
                t0=float(entry["t0"]),
                present=entry["present"],
                stats_sha256=sha,
                signal_stds=sig_stds,
            )
            for name in ("summary-ip", "interferometer-n_e_line", "equilibrium-beta_pol"):
                m = sig_metrics[name]
                assert math.isfinite(m["ceiling_nrmse"])
                assert 0.0 <= m["pred_coverage"] <= 1.0
                assert 0.0 <= m["pred_missing_frac"] <= 1.0
                assert 0.0 <= m["ceil_coverage"] <= 1.0
                assert 0.0 <= m["ceil_missing_frac"] <= 1.0

            cf_nbi = (
                rollout.counterfactual_nbi_physical(edited_act, act_stats, stats)
                if mode != "real"
                else None
            )
            if mode.startswith("set:"):
                assert cf_nbi is not None
                assert np.allclose(cf_nbi[4:], 0.0, atol=1.0)
            png = rollout.plot_shot_rollout(
                "1003",
                decoded,
                sig_metrics,
                ro.k0,
                ro.n_frames,
                float(entry["t0"]),
                framecache.DT_FRAME,
                plots_dir / f"rollout_1003_{mode.replace(':', '_')}.png",
                actuator_mode=mode,
                step=step,
                cf_nbi=cf_nbi,
            )
            assert png.exists() and png.stat().st_size > 1000
    finally:
        layout.unpatch_ignite()
