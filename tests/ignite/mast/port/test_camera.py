"""CPU-only tests for the MAST 2D visible camera (rgb) IGNITE modality."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import zarr

from tokamak_foundation_model.ignite.mast.port import camera, framecache, layout, rollout
from tokamak_foundation_model.ignite.mast.port.camera import tc, td
from tokamak_foundation_model.ignite.video_codec import VideoCodec

_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts" / "mast"
_REAL_FRAMES_DIR = camera.DEFAULT_FRAMES_DIR
_REAL_FIT_DIR = camera.DEFAULT_CAMERA_FIT_DIR
_REAL_CACHE_DIR = camera.DEFAULT_SPLITS_PATH.parent


def _load_script_module(script_name: str):
    path = _SCRIPTS_DIR / script_name
    spec = importlib.util.spec_from_file_location(f"_test_{path.stem}_camera", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ----------------------------------------------------------------------- standardize_input identity

def test_standardize_input_is_identity_on_preprocessed_frames():
    """VideoCodec.standardize_input is the identity on preprocessed frames (both s >= s_floor and s < s_floor)."""
    rng = np.random.default_rng(123)
    raw = rng.integers(0, 4096, size=(4, 480, 640), dtype=np.uint16)
    # Frame 2 is nearly flat so s < s_floor -> std(z) < 1, testing sd.clamp(min=1.0).
    raw[2] = 100 + rng.integers(0, 2, size=(480, 640), dtype=np.uint16)
    # Frame 3 is completely constant (s == 0).
    raw[3] = 500

    prep = camera.preprocess_frames(raw)
    assert prep["u"].shape == (4, 120, 160)
    assert prep["u"].dtype == np.float16
    assert prep["m"].dtype == np.float32
    assert prep["s"].dtype == np.float32
    assert 0.0 <= prep["saturation_frac"] <= 1.0

    s_floor = 0.05
    z = camera.normalize_u_frames(prep["u_f32"], prep["m"], prep["s"], s_floor=s_floor)
    assert z.shape == (4, 1, 1, 120, 160)
    z_std = VideoCodec.standardize_input(z)
    assert torch.allclose(z_std, z, atol=1e-5, rtol=1e-5)

    # Also verify stored float16 u is exact identity under standardize_input.
    z_f16 = camera.normalize_u_frames(prep["u"], prep["m"], prep["s"], s_floor=s_floor)
    z_f16_std = VideoCodec.standardize_input(z_f16)
    assert torch.allclose(z_f16_std, z_f16, atol=1e-5, rtol=1e-5)


# ----------------------------------------------------------------------- exposure-to-frame mapping

def test_exposure_to_frame_mapping_edge_cases():
    """Exposure mapping covers gaps, pre/post-exposure frames, max-overlap selection, and 2.5 ms rule."""
    # Two exposures with convention="start", E = 9 ms, separated by a large gap:
    # exp 0: [0.010, 0.019], exp 1: [0.0325, 0.0415]
    # 5 ms frames from t0 = 0.0:
    # f=0 [0.000, 0.005): before first -> -1
    # f=1 [0.005, 0.010): touches at 0.010 (0 ms overlap) -> -1
    # f=2 [0.010, 0.015): 5.0 ms with exp 0 -> 0
    # f=3 [0.015, 0.020): 4.0 ms with exp 0 -> 0
    # f=4 [0.020, 0.025): in gap -> -1
    # f=5 [0.025, 0.030): in gap -> -1
    # f=6 [0.030, 0.035): 2.5 ms with exp 1 -> 1 (exact 2.5 ms boundary accepted)
    # f=7 [0.035, 0.040): 5.0 ms with exp 1 -> 1
    # f=8 [0.040, 0.045): 1.5 ms with exp 1 (< 2.5 ms) -> -1
    # f=9 [0.045, 0.050): after last -> -1
    times = np.array([0.010, 0.0325], dtype=np.float64)
    mapping = camera.map_frames_to_exposures(
        times, 0.009, t0=0.0, n_frames=10, dt_frame=0.005, convention="start"
    )
    assert mapping.tolist() == [-1, -1, 0, 0, -1, -1, 1, 1, -1, -1]

    # Just below 2.5 ms (2.49 ms overlap) must be rejected.
    times_just_under = np.array([0.00251], dtype=np.float64)
    m_under = camera.map_frames_to_exposures(
        times_just_under, 0.009, t0=0.0, n_frames=3, dt_frame=0.005, convention="start"
    )
    assert m_under.tolist() == [-1, 0, -1]

    # Adjacent exposures where a frame overlaps both: must pick larger overlap.
    # exp 0 [0.000, 0.007], exp 1 [0.007, 0.014]:
    # f=0 [0.000, 0.005): 5 ms with exp 0 -> 0
    # f=1 [0.005, 0.010): 2 ms with exp 0, 3 ms with exp 1 -> 1
    # f=2 [0.010, 0.015): 4 ms with exp 1 -> 1
    m_split = camera.map_frames_to_exposures(
        np.array([0.000, 0.007]), 0.007, t0=0.0, n_frames=3, dt_frame=0.005, convention="start"
    )
    assert m_split.tolist() == [0, 1, 1]

    # Centre and end conventions shift intervals by -E/2 and -E.
    m_centre = camera.map_frames_to_exposures(
        np.array([0.005]), 0.010, t0=0.0, n_frames=3, dt_frame=0.005, convention="centre"
    )
    assert m_centre.tolist() == [0, 0, -1]
    m_end = camera.map_frames_to_exposures(
        np.array([0.010]), 0.010, t0=0.0, n_frames=3, dt_frame=0.005, convention="end"
    )
    assert m_end.tolist() == [0, 0, -1]


# ----------------------------------------------------------------------- stats encode/decode round trip

def test_stats_codec_roundtrip_and_missing():
    """camera-rgb-stats fits 64-bin grids, round-trips (m, s), and decodes code 64 to NaN."""
    rng = np.random.default_rng(7)
    m_train = rng.uniform(1.0, 7.5, size=2000)
    s_train = np.power(10.0, rng.uniform(-1.5, 0.4, size=2000))
    s_floor, codec, u_std = camera.fit_camera_stats(m_train, s_train)

    assert math.isfinite(s_floor) and s_floor > 0.0
    assert math.isfinite(u_std) and u_std > 0.0
    assert codec.n_bins == 64
    assert codec.missing_code == 64
    assert codec.vocab_size == 65

    m_test = np.array([2.0, 4.5, 6.8, np.nan])
    s_test = np.array([0.1, 0.5, 1.8, np.nan])
    codes = codec.encode(m_test, s_test)
    assert codes.shape == (4, 2)
    assert codes.dtype == np.int64
    assert (codes[:3] >= 0).all() and (codes[:3] < 64).all()
    assert (codes[3] == 64).all()

    m_dec, s_dec = codec.decode(codes)
    assert np.isfinite(m_dec[:3]).all() and np.isfinite(s_dec[:3]).all()
    assert np.isnan(m_dec[3]) and np.isnan(s_dec[3])
    assert np.max(np.abs(m_dec[:3] - m_test[:3])) < 0.25
    assert np.max(np.abs(np.log10(s_dec[:3]) - np.log10(np.maximum(s_test[:3], s_floor)))) < 0.1


# ----------------------------------------------------------------------- zarr completeness check

def test_zarr_completeness_check_rejects_missing_chunk(tmp_path: Path):
    """Completeness check accepts complete zarr and rejects missing/empty chunk or length mismatch."""
    shot_dir = tmp_path / "99999.zarr"
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "no_rgb")

    rgb_dir = shot_dir / "rgb"
    grp = zarr.open_group(str(rgb_dir), mode="w", zarr_format=2)
    grp.attrs["exposure"] = 9090.0
    data = np.arange(6 * 16 * 16, dtype=np.uint16).reshape(6, 16, 16)
    times = np.linspace(0.1, 0.15, 6, dtype=np.float64)
    # Force multiple chunks along axis 0 (chunks=(2, 8, 8) -> 3*2*2 = 12 chunks).
    grp.create_array("data", data=data, chunks=(2, 8, 8))
    grp.create_array("time", data=times, chunks=(6,))

    assert camera.check_rgb_zarr_complete(shot_dir) == (True, "ok")
    d_read, t_read, exp_s, _attrs = camera.read_rgb_zarr(shot_dir)
    assert np.array_equal(d_read, data)
    assert np.allclose(t_read, times)
    assert exp_s == pytest.approx(0.00909)

    # Remove one chunk file -> must be rejected as incomplete and read_rgb_zarr must raise.
    chunk_file = rgb_dir / "data" / "1.0.1"
    assert chunk_file.exists()
    backup = chunk_file.read_bytes()
    chunk_file.unlink()
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "incomplete")
    with pytest.raises(RuntimeError, match="incomplete"):
        camera.read_rgb_zarr(shot_dir)

    # Zero-byte chunk file -> still incomplete.
    chunk_file.write_bytes(b"")
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "incomplete")

    # Restore chunk -> complete again.
    chunk_file.write_bytes(backup)
    assert camera.check_rgb_zarr_complete(shot_dir) == (True, "ok")

    # Mismatched len(time) != data.shape[0] -> incomplete.
    t_meta_path = rgb_dir / "time" / ".zarray"
    t_meta = json.loads(t_meta_path.read_text())
    t_meta["shape"] = [5]
    t_meta_path.write_text(json.dumps(t_meta))
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "incomplete")


# ----------------------------------------------------------------------- patch restoration

def test_patched_video_codec_restores_on_exit_and_error(tmp_path: Path):
    """patched_video_codec restores train_codec globals cleanly after normal exit and exception."""
    orig_ds = tc.VideoCodecPairDataset
    orig_loader = tc.make_video_loader
    with camera.patched_video_codec(frames_dir=tmp_path, s_floor=0.05, plasma_exposures={}):
        assert tc.VideoCodecPairDataset is not orig_ds
        assert tc.make_video_loader is camera.make_mast_video_loader
    assert tc.VideoCodecPairDataset is orig_ds
    assert tc.make_video_loader is orig_loader

    with pytest.raises(RuntimeError, match="boom"):
        with camera.patched_video_codec(frames_dir=tmp_path, s_floor=0.05, plasma_exposures={}):
            assert tc.VideoCodecPairDataset is not orig_ds
            raise RuntimeError("boom")
    assert tc.VideoCodecPairDataset is orig_ds
    assert tc.make_video_loader is orig_loader


# ----------------------------------------------------------------------- CPU smoke of training + encode/decode + plots

def test_cpu_smoke_train_encode_decode_rollout_and_plots(tmp_path: Path):
    """Tiny synthetic frames cache trains via port_train_camera_codec, loads via _load_codec, encodes/decodes, and plots."""
    port_train_cam = _load_script_module("port_train_camera_codec.py")
    port_build = _load_script_module("port_build_cache.py")
    port_recon = _load_script_module("port_recon_plots.py")

    rng = np.random.default_rng(42)
    h, w = 40, 40
    frames_dir = tmp_path / "frames_rgb"
    cache_dir = tmp_path / "frame_codes"
    fit_dir = tmp_path / "camera_fit"
    codec_out = tmp_path / "codec_smoke"
    frames_dir.mkdir()
    cache_dir.mkdir()

    shots_train = ["1001", "1002"]
    shots_val = ["1003"]
    all_shots = shots_train + shots_val
    index_shots = {}
    m_list, s_list = [], []

    # Each shot has 20 exposures at 9 ms cadence from t=0.010..0.181 s,
    # and plasma frame cache covers t0=0.0, n_frames=40 (0.0..0.200 s) so first 2 frames are uncovered!
    yy, xx = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w), indexing="ij")
    for s_idx, shot in enumerate(all_shots):
        n_exp = 20
        times = 0.010 + np.arange(n_exp, dtype=np.float64) * 0.009
        frames = np.empty((n_exp, h, w), dtype=np.float32)
        for k in range(n_exp):
            cx = 0.3 * math.sin(0.4 * k + s_idx)
            cy = 0.3 * math.cos(0.3 * k + s_idx)
            blob = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / 0.25)
            frames[k] = 50.0 + 1500.0 * blob + rng.uniform(0.0, 20.0, size=(h, w))
        u_f32 = np.log1p(frames)
        m = u_f32.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
        s = u_f32.std(axis=(1, 2), ddof=1, dtype=np.float64).astype(np.float32)
        raw_mean = frames.mean(axis=(1, 2), dtype=np.float64).astype(np.float32)
        camera.save_frames_npz_atomic(
            frames_dir / f"{shot}.npz",
            u=u_f32.astype(np.float16),
            time=times,
            m=m,
            s=s,
            raw_mean_counts=raw_mean,
            exposure_s=0.009,
        )
        index_shots[shot] = {"status": "ok", "n_frames": n_exp}
        if shot in shots_train:
            m_list.append(m)
            s_list.append(s)
        torch.save({"t0": 0.0, "n_frames": 40}, cache_dir / f"{shot}.pt")

    framecache.save_json_atomic({"shots": index_shots}, frames_dir / "index.json")
    framecache.save_json_atomic(
        {"train": shots_train, "val": shots_val, "test": []}, cache_dir / "splits.json"
    )
    framecache.save_json_atomic({"dt_frame": 0.005}, cache_dir / "_codec_manifest.json")

    s_floor, stats_codec, u_std = camera.fit_camera_stats(
        np.concatenate(m_list), np.concatenate(s_list)
    )
    fit_sha = camera.save_camera_fit(
        fit_dir,
        s_floor,
        stats_codec,
        {
            "time_convention": "centre",
            "s_floor": s_floor,
            "u_std": u_std,
            "n_bins": 64,
            "margin": 0.05,
        },
    )
    assert fit_sha == camera.camera_fit_sha256(fit_dir)

    # Train tiny codec for 6 CPU steps via port_train_camera_codec.
    gate_res = port_train_cam.train_camera_codec(
        frames_dir=frames_dir,
        fit_dir=fit_dir,
        cache_dir=cache_dir,
        out_dir=codec_out,
        steps=6,
        batch_size=4,
        eval_every=3,
        eval_batches=2,
        eval_batch_size=4,
        eval_frames=4,
        num_workers=0,
        lr=1e-3,
        device_spec="cpu",
        height=h,
        width=w,
        patch_h=20,
        patch_w=20,
        d_model=64,
        enc_depth=2,
        dec_depth=2,
        heads=4,
    )
    assert gate_res["steps"] == 6
    ckpt_path = codec_out / "codec_best.pt"
    if not ckpt_path.exists():
        # If tiny 6-step run stayed below gate_recon_floor, codec_last.pt has the same format.
        ckpt_path = codec_out / "codec_last.pt"
    assert ckpt_path.exists()

    # Must load with dev-peter's train_dynamics._load_codec("video", path).
    loaded_codec, loaded_cfg = td._load_codec("video", ckpt_path)
    assert loaded_cfg.height == h and loaded_cfg.width == w
    assert loaded_cfg.n_tok == 4

    cam_fit = camera.load_camera_fit(fit_dir)
    codes, stats_codes, present = camera.encode_camera_codes(
        loaded_codec,
        cam_fit,
        frames_dir / "1003.npz",
        t0=0.0,
        n_frames=40,
        dt_frame=0.005,
    )
    assert present is True
    assert codes.shape == (40, loaded_cfg.n_tok) and codes.dtype == torch.int64
    assert stats_codes.shape == (40, 2) and stats_codes.dtype == torch.int64
    # Frame 0 [0.000, 0.005) has no exposure (exp 0 centre=0.010 -> [0.0055, 0.0145]) -> missing code!
    assert (codes[0] == camera.CAMERA_MISSING_CODE).all()
    assert (stats_codes[0] == camera.CAMERA_STATS_MISSING_CODE).all()

    counts_rec = camera.decode_camera(loaded_codec, cam_fit, codes, stats_codes)
    assert counts_rec.shape == (40, h, w)
    assert np.isnan(counts_rec[0]).all()
    assert np.isfinite(counts_rec[2:20]).all()

    # Absent shot returns placeholder 0 and present=False.
    abs_c, abs_s, abs_p = camera.encode_camera_codes(
        loaded_codec, cam_fit, None, t0=0.0, n_frames=10
    )
    assert abs_p is False
    assert (abs_c == framecache.PLACEHOLDER_CODE).all()
    assert (abs_s == framecache.PLACEHOLDER_CODE).all()

    # rollout.decode_and_score_rollout handles present and absent camera cleanly.
    ro = SimpleNamespace(
        gt_codes={camera.CAMERA_MODALITY: codes, camera.CAMERA_STATS_MODALITY: stats_codes},
        pred_codes={camera.CAMERA_MODALITY: codes.clone(), camera.CAMERA_STATS_MODALITY: stats_codes.clone()},
        k0=5,
        n_frames=40,
    )
    dec, met = rollout.decode_and_score_rollout(
        ro,
        tmp_path / "1003.h5",
        {},
        {},
        t0=0.0,
        present={camera.CAMERA_MODALITY: True, camera.CAMERA_STATS_MODALITY: True},
        camera_codec=loaded_codec,
        camera_fit=cam_fit,
        camera_frames_dir=frames_dir,
        shot="1003",
    )
    assert dec[camera.CAMERA_MODALITY]["present"] is True
    assert math.isfinite(met[camera.CAMERA_MODALITY]["ceiling_nrmse"])
    assert met[camera.CAMERA_MODALITY]["rollout_nrmse"] == pytest.approx(
        met[camera.CAMERA_MODALITY]["ceiling_nrmse"]
    )

    # Absent camera behaves like any absent modality.
    dec_abs, met_abs = rollout.decode_and_score_rollout(
        ro,
        tmp_path / "1003.h5",
        {},
        {},
        t0=0.0,
        present={camera.CAMERA_MODALITY: False, camera.CAMERA_STATS_MODALITY: False},
        camera_codec=loaded_codec,
        camera_fit=cam_fit,
        camera_frames_dir=frames_dir,
        shot="1003",
    )
    assert dec_abs[camera.CAMERA_MODALITY]["present"] is False
    assert math.isnan(met_abs[camera.CAMERA_MODALITY]["rollout_nrmse"])

    # Camera recon PNG + GIF render cleanly.
    cam_png, cam_gif = port_recon.plot_camera_recon(
        "1003",
        dec[camera.CAMERA_MODALITY],
        k0=5,
        t0=0.0,
        dt_frame=0.005,
        out_png=tmp_path / "recon_camera_1003.png",
        out_gif=tmp_path / "recon_camera_1003.gif",
        step=None,
        with_rollout=False,
    )
    assert cam_png is not None and cam_png.exists() and cam_png.stat().st_size > 1000
    assert cam_gif is not None and cam_gif.exists() and cam_gif.stat().st_size > 1000

    # Manifest camera verification + port_build_cache safety check.
    cam_manifest = {
        "assignment_sha256": "a" * 64,
        "fit_sha256": "b" * 64,
        "stats_sha256": "c" * 64,
        "plasma_ip_threshold_A": 50_000.0,
        "camera_codec_path": str(ckpt_path),
        "camera_codec_sha256": camera.file_sha256(ckpt_path),
        "camera_fit_path": str(fit_dir),
        "camera_fit_sha256": fit_sha,
        "camera_frames_dir": str(frames_dir),
        "camera_frames_index_sha256": camera.file_sha256(frames_dir / "index.json"),
    }
    resolved = camera.resolve_manifest_camera(cam_manifest)
    assert resolved is not None and resolved["codec_path"] == ckpt_path
    with pytest.raises(ValueError, match="camera_fit_sha256"):
        camera.resolve_manifest_camera({**cam_manifest, "camera_fit_sha256": "0" * 64})

    mpath = tmp_path / "cam_manifest.json"
    framecache.save_json_atomic(cam_manifest, mpath)
    with pytest.raises(ValueError, match="camera_codec_sha256"):
        port_build._check_existing_manifest(
            mpath,
            assignment_sha256="a" * 64,
            fit_sha256="b" * 64,
            stats_sha256="c" * 64,
            plasma_ip_threshold_a=50_000.0,
            camera_codec_sha256=None,
            camera_fit_sha256=fit_sha,
            camera_frames_index_sha256=cam_manifest["camera_frames_index_sha256"],
        )

    # Layout supplies camera-rgb and camera-rgb-stats ModalitySpecs in manifest order.
    layout_dir = tmp_path / "layout_cam"
    layout_dir.mkdir()
    framecache.save_json_atomic(
        {
            "t0_start": 0.0,
            "dt_frame": 0.005,
            "modalities": {
                "summary-ip": {"family": "slowts", "n_tok": 4, "codebook_size": 65},
                "camera-rgb": {"family": "video", "n_tok": 48, "codebook_size": 1001},
                "camera-rgb-stats": {"family": "slowts", "n_tok": 2, "codebook_size": 65},
            },
            "actuators": {
                "names": ["summary-power_nbi"],
                "channels": [1],
                "total_dim": 1,
                "mu": [0.0],
                "sd": [1.0],
                "n_train_shots": 1,
                "n_valid_frames": [10],
            },
        },
        layout_dir / "_codec_manifest.json",
    )
    specs = layout.modality_specs_from_manifest(layout_dir)
    assert [s.name for s in specs] == ["summary-ip", "camera-rgb", "camera-rgb-stats"]
    assert specs[1].family == "video" and specs[1].n_tok == 48 and specs[1].codebook_size == 1001
    assert specs[2].n_tok == 2 and specs[2].codebook_size == 65


# ----------------------------------------------------------------------- real-data val shot test

def test_real_val_shot_camera_codes_in_range_and_missing_exact():
    """Real MAST val shot: codes in range and missing codes exactly where no exposure overlaps."""
    if not (_REAL_FRAMES_DIR / "index.json").exists():
        pytest.skip(f"camera frames cache not found at {_REAL_FRAMES_DIR}")
    if not (_REAL_FIT_DIR / "camera_fit.npz").exists():
        pytest.skip(f"camera fit not found at {_REAL_FIT_DIR}")
    if not (_REAL_CACHE_DIR / "splits.json").exists():
        pytest.skip(f"frame cache splits.json not found at {_REAL_CACHE_DIR}")

    index_doc = json.loads((_REAL_FRAMES_DIR / "index.json").read_text())
    shots_index = index_doc["shots"]
    splits = json.loads((_REAL_CACHE_DIR / "splits.json").read_text())
    dt_frame = float(
        json.loads((_REAL_CACHE_DIR / "_codec_manifest.json").read_text()).get(
            "dt_frame", framecache.DT_FRAME
        )
    )
    cam_fit = camera.load_camera_fit(_REAL_FIT_DIR)

    val_shots = sorted((str(s) for s in splits["val"]), key=lambda s: int(s))
    chosen = None
    chosen_entry = None
    chosen_mapping = None
    for s in val_shots:
        if (shots_index.get(s) or {}).get("status") != "ok":
            continue
        npz_path = _REAL_FRAMES_DIR / f"{s}.npz"
        pt_path = _REAL_CACHE_DIR / f"{s}.pt"
        if not npz_path.exists() or not pt_path.exists():
            continue
        entry = torch.load(pt_path, map_location="cpu", weights_only=False)
        fr = camera.load_frames_npz(npz_path, load_u=False)
        mapping = camera.map_frames_to_exposures(
            fr["time"],
            fr["exposure_s"],
            float(entry["t0"]),
            int(entry["n_frames"]),
            dt_frame=dt_frame,
            convention=cam_fit.time_convention,
        )
        if (mapping >= 0).any():
            chosen, chosen_entry, chosen_mapping = s, entry, mapping
            break
    if chosen is None:
        pytest.skip("no ok val shot with overlapping camera exposures found yet")

    codec = VideoCodec(camera.default_video_codec_config()).eval()
    codes, stats_codes, present = camera.encode_camera_codes(
        codec,
        cam_fit,
        _REAL_FRAMES_DIR / f"{chosen}.npz",
        t0=float(chosen_entry["t0"]),
        n_frames=int(chosen_entry["n_frames"]),
        dt_frame=dt_frame,
    )
    assert present is True
    n_f = int(chosen_entry["n_frames"])
    assert codes.shape == (n_f, 48) and codes.dtype == torch.int64
    assert stats_codes.shape == (n_f, 2) and stats_codes.dtype == torch.int64

    ov = chosen_mapping >= 0
    miss = ~ov
    assert (codes[ov] >= 0).all() and (codes[ov] < 1000).all()
    assert (stats_codes[ov] >= 0).all() and (stats_codes[ov] < 64).all()
    if miss.any():
        assert (codes[miss] == camera.CAMERA_MISSING_CODE).all()
        assert (stats_codes[miss] == camera.CAMERA_STATS_MISSING_CODE).all()

    # Decode a slice of overlapping frames and verify finite counts.
    ov_idx = np.flatnonzero(ov)[:4]
    dec_counts = camera.decode_camera(codec, cam_fit, codes[ov_idx], stats_codes[ov_idx])
    assert dec_counts.shape == (len(ov_idx), 120, 160)
    assert np.isfinite(dec_counts).all()
