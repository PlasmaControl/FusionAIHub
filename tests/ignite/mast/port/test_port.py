"""CPU-only tests for the IGNITE-on-MAST Phase-B dynamics port."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import ModalitySpec
from tokamak_foundation_model.ignite.mast import cache, paths, registry, tasks, transform
from tokamak_foundation_model.ignite.mast.port import framecache, layout, rollout
from tokamak_foundation_model.ignite.mast.port.layout import td
from tokamak_foundation_model.ignite.mast.tokenizers import codecs as tok_codecs
from tokamak_foundation_model.ignite.mast.tokenizers.grid import GridSet

_CACHE_V1 = paths.cache_dir("v1")
_FIT_V1 = paths.stage1_dir() / "fit" / "v1"
_ASSIGN_V1B = paths.stage1_dir() / "assignments" / "v1b.json"
_PORT_TRAIN_SCRIPT = Path(__file__).resolve().parents[4] / "scripts" / "mast" / "port_train.py"


def _load_port_train_module():
    name = "_test_port_train_module"
    spec = importlib.util.spec_from_file_location(name, _PORT_TRAIN_SCRIPT)
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


# ----------------------------------------------------------------------- real-shot roundtrip

def test_real_shot_entry_and_roundtrip():
    """One real MAST cache-v1 shot encodes to the exact cache format and decodes within roundtrip error."""
    _skip_if_no_cache_v1()

    signals = registry.all_signals()
    act_specs = framecache.actuator_specs()
    act_names = {s.name for s in act_specs}
    plasma_names = tuple(s.name for s in signals if s.name not in act_names)
    codecs, stats, _assignment = framecache.build_mast_codecs(
        signals, _CACHE_V1, _FIT_V1, _ASSIGN_V1B
    )
    modality_order = tuple(s.name for s in signals if s.name in codecs)
    sha = json.loads((_CACHE_V1 / "manifest.json").read_text())["stats_sha256"]

    # Pick a shot that has multiple diagnostics present and at least one absent or partial NaN.
    shot_files = sorted((_CACHE_V1 / "shots").glob("*.h5"))
    shot_path = shot_files[10]
    moments = framecache.shot_actuator_moments(
        shot_path, act_specs, plasma_names, stats_sha256=sha
    )
    assert moments is not None
    act_stats = framecache.finalize_actuator_stats(act_specs, [moments])

    entry, reason = framecache.encode_shot(
        shot_path,
        codecs,
        modality_order,
        act_specs,
        act_stats,
        plasma_names,
        stats_sha256=sha,
    )
    assert reason == "ok" and entry is not None
    n_frames = entry["n_frames"]
    assert n_frames >= framecache.MIN_CACHED_FRAMES
    assert isinstance(entry["t0"], float)
    assert entry["actuators"].dtype == torch.float16
    assert entry["actuators"].shape == (n_frames, act_stats.total_dim)
    assert torch.isfinite(entry["actuators"]).all()
    assert entry["actuator_valid"].dtype == torch.bool
    assert entry["actuator_valid"].shape == (n_frames, act_stats.total_dim)
    # Invalid actuator slots must be 0 after z-scoring.
    invalid_act = ~entry["actuator_valid"]
    if invalid_act.any():
        assert (entry["actuators"][invalid_act] == 0).all()

    shot_data = cache.read_shot(shot_path, list(modality_order), stats_sha256=sha)
    span = framecache.plasma_frame_range(shot_data, plasma_names)
    assert span is not None
    t0_ref, _t0, f_start, _f_end, _nf = span

    checked_present = 0
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

        checked_present += 1
        cs = shot_data[name]
        assert cs is not None
        x_phys = transform.to_physical(cs.values, stats[name])[None]
        t_len = x_phys.shape[-1]
        rt = codec.roundtrip(x_phys, mode="nearest")[0]
        # Align codes to the shot's full record if f_start == 0 and sig_offset == 0.
        sig_offset = int(round((cs.t0 - t0_ref) / framecache.DT_FRAME))
        if f_start == 0 and sig_offset == 0:
            dec = framecache.decode_signal_codes(codec, codes, n_samples=t_len)
            fin_rt = np.isfinite(rt)
            fin_dec = np.isfinite(dec)
            assert np.array_equal(fin_rt, fin_dec), f"{name}: NaN pattern mismatch"
            if fin_rt.any():
                rel = float(np.max(np.abs(dec[fin_rt] - rt[fin_rt]))) / (
                    float(np.std(rt[fin_rt])) + 1e-12
                )
                assert rel < 1e-6, f"{name}: decode differs from codec.roundtrip by {rel:.3e}"
    assert checked_present > 20


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

    codes, is_present = framecache.encode_signal_codes(codec, cs, t0_ref=0.0, f_start=0, n_frames=10)
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
    n_frames = 14
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
    plasma_names = tuple(s.name for s in sig_specs[:-1])

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
    t_axis = np.linspace(0.0, 2.0 * np.pi, n_frames)
    for idx, (shot, split) in enumerate(shot_splits):
        h5_path = v1_root / "shots" / f"{shot}.h5"
        status = {}
        with h5py.File(h5_path, "w") as f:
            f.attrs["shot"] = int(shot)
            f.attrs["split"] = split
            f.attrs["cache_version"] = cache.CACHE_VERSION
            f.attrs["stats_sha256"] = sha
            for s_idx, spec in enumerate(sig_specs):
                # Make equilibrium-beta_pol absent on shot 1002 to exercise --mask_absent.
                if shot == "1002" and spec.name == "equilibrium-beta_pol":
                    status[spec.name] = "absent"
                    continue
                wave = (
                    np.sin(t_axis + 0.4 * idx + 0.3 * s_idx)
                    + 0.1 * rng.standard_normal(n_frames)
                ).astype(np.float32)[None, :]
                ds = f.create_dataset(spec.name, data=wave, dtype=np.float32)
                ds.attrs["t0"] = 0.0
                ds.attrs["dt"] = dt
                ds.attrs["dead_nan"] = 0
                ds.attrs["lost"] = 0
                ds.attrs["const_channels"] = np.zeros(0, dtype=np.int64)
                status[spec.name] = "ok"
            f.attrs["status"] = json.dumps(status)

    # Fit actuator stats on training shots and encode all 3 shots.
    moments = [
        framecache.shot_actuator_moments(
            v1_root / "shots" / f"{shot}.h5", act_specs, plasma_names, stats_sha256=sha
        )
        for shot, split in shot_splits
        if split == "train"
    ]
    act_stats = framecache.finalize_actuator_stats(act_specs, moments)

    fc_dir = tmp_path / "frame_codes"
    fc_dir.mkdir()
    modality_order = tuple(s.name for s in sig_specs)
    presence = {}
    for shot, _split in shot_splits:
        entry, reason = framecache.encode_shot(
            v1_root / "shots" / f"{shot}.h5",
            codecs,
            modality_order,
            act_specs,
            act_stats,
            plasma_names,
            stats_sha256=sha,
        )
        assert reason == "ok" and entry is not None
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
    return v1_root, fc_dir, codecs, stats, act_stats, sha


# train_dynamics.py:2742 calls torch.autocast("cuda", dtype=bfloat16) without enabled=cuda;
# on CPU that unmodified upstream line emits one harmless UserWarning.
@pytest.mark.filterwarnings("ignore:CUDA is not available:UserWarning")
def test_end_to_end_train_rollout_plot(tmp_path: Path):
    """Tiny d_model=64 depth=2 trains through port_train.py, rolls out (real & zero), and writes a PNG."""
    v1_root, fc_dir, codecs, stats, act_stats, sha = _make_synthetic_three_shot_fixture(tmp_path)
    run_dir = tmp_path / "run"
    port_train = _load_port_train_module()

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
        for mode in ("real", "zero"):
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
            )
            for name in ("summary-ip", "interferometer-n_e_line", "equilibrium-beta_pol"):
                assert np.isfinite(sig_metrics[name]["rollout_nrmse"])
                assert np.isfinite(sig_metrics[name]["ceiling_nrmse"])

            cf_nbi = (
                rollout.counterfactual_nbi_physical(edited_act, act_stats, stats)
                if mode != "real"
                else None
            )
            png = rollout.plot_shot_rollout(
                "1003",
                decoded,
                sig_metrics,
                ro.k0,
                ro.n_frames,
                float(entry["t0"]),
                framecache.DT_FRAME,
                plots_dir / f"rollout_1003_{mode}.png",
                actuator_mode=mode,
                step=step,
                cf_nbi=cf_nbi,
            )
            assert png.exists() and png.stat().st_size > 1000
    finally:
        layout.unpatch_ignite()
