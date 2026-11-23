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


def test_map_frames_to_exposures_nan_duplicates_and_glitches():
    """NaN times, duplicate times, and real ~0.6 s timestamp glitches never map to frames."""
    # Single NaN timestamp must not wipe out the rest of the shot.
    t_nan = np.array([0.010, np.nan, 0.020], dtype=np.float64)
    m_nan = camera.map_frames_to_exposures(
        t_nan, 0.009, t0=0.0, n_frames=6, dt_frame=0.005, convention="start"
    )
    assert m_nan.tolist() == [-1, -1, 0, 0, 2, 2]
    assert 1 not in m_nan

    # Duplicate timestamps keep the earliest index; later duplicate never maps.
    t_dup = np.array([0.010, 0.010, 0.020], dtype=np.float64)
    assert camera.valid_exposure_indices(t_dup).tolist() == [0, 2]
    m_dup = camera.map_frames_to_exposures(
        t_dup, 0.009, t0=0.0, n_frames=6, dt_frame=0.005, convention="start"
    )
    assert m_dup.tolist() == [-1, -1, 0, 0, 2, 2]

    # Real glitch pattern from shot 19425 indices 1-4: [0.0012, 0.0012, 0.5823, 0.0194]
    # embedded in an otherwise regular 9.1 ms sequence (convention="end", E=9.09 ms).
    t_19425 = np.array(
        [-0.0079, 0.0012, 0.0012, 0.5823, 0.0194, 0.0285, 0.0376], dtype=np.float64
    )
    assert camera.valid_exposure_indices(t_19425).tolist() == [0, 1, 4, 5, 6]
    # Cover both the local window and 0.58 s (where the glitched timestamp jumped).
    m_19425 = camera.map_frames_to_exposures(
        t_19425, 0.00909, t0=-0.015, n_frames=125, dt_frame=0.005, convention="end"
    )
    assert 2 not in m_19425  # duplicate 0.0012 excluded
    assert 3 not in m_19425  # glitched 0.5823 excluded even at t~0.58 s
    # Frame f=2: [-0.005, 0.000) overlaps exp 1 [-0.00789, 0.0012] by 5 ms -> 1.
    assert m_19425[2] == 1
    # Frame f=6: [0.015, 0.020) overlaps exp 4 [0.01031, 0.0194] by 4.4 ms -> 4.
    assert m_19425[6] == 4
    # Frame f=7: [0.020, 0.025) overlaps exp 5 [0.01941, 0.0285] by 5.0 ms -> 5.
    assert m_19425[7] == 5
    # Frame f=118: [0.575, 0.580) would overlap glitched exp 3 -> must be -1.
    assert m_19425[118] == -1

    # Real glitch pattern from shot 19514 indices 14-18: [0.0649, 0.0831, 0.5657, 0.1013, 0.1013]
    # embedded in an otherwise regular 9.1 ms sequence.
    t_19514 = np.array(
        [0.0558, 0.0649, 0.0831, 0.5657, 0.1013, 0.1013, 0.1104], dtype=np.float64
    )
    assert camera.valid_exposure_indices(t_19514).tolist() == [0, 1, 2, 4, 6]
    m_19514 = camera.map_frames_to_exposures(
        t_19514, 0.00909, t0=0.050, n_frames=110, dt_frame=0.005, convention="end"
    )
    assert 3 not in m_19514  # glitched 0.5657 excluded
    assert 5 not in m_19514  # duplicate 0.1013 excluded
    # Frame f=5: [0.075, 0.080) overlaps exp 2 [0.07401, 0.0831] by 5 ms -> 2.
    assert m_19514[5] == 2
    # Frame f=9: [0.095, 0.100) overlaps exp 4 [0.09221, 0.1013] by 5 ms -> 4.
    assert m_19514[9] == 4
    # Frame f=11: [0.105, 0.110) overlaps exp 6 [0.10131, 0.1104] by 5 ms -> 6.
    assert m_19514[11] == 6
    # Frame f=102: [0.560, 0.565) would overlap glitched exp 3 -> must be -1.
    assert m_19514[102] == -1
    # Unique mapped exposure indices are strictly time-ordered.
    uniq = np.unique(m_19514[m_19514 >= 0])
    assert np.all(np.diff(t_19514[uniq]) > 0.0)


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


def test_zarr_completeness_metadata_null_separator_and_strict_exposure(tmp_path: Path):
    """Completeness requires .zgroup/.zattrs + exposure attr and treats null dimension_separator as '.'."""
    shot_dir = tmp_path / "88888.zarr"
    rgb_dir = shot_dir / "rgb"
    (rgb_dir / "data").mkdir(parents=True)
    (rgb_dir / "time").mkdir(parents=True)

    zgroup_path = rgb_dir / ".zgroup"
    zattrs_path = rgb_dir / ".zattrs"
    zgroup_path.write_text(json.dumps({"zarr_format": 2}))
    zattrs_path.write_text(json.dumps({"exposure": 9090.0}))

    data_meta = {
        "zarr_format": 2,
        "shape": [4, 8, 8],
        "chunks": [2, 8, 8],
        "dtype": "<u2",
        "dimension_separator": None,
    }
    time_meta = {
        "zarr_format": 2,
        "shape": [4],
        "chunks": [4],
        "dtype": "<f8",
        "dimension_separator": None,
    }
    (rgb_dir / "data" / ".zarray").write_text(json.dumps(data_meta))
    (rgb_dir / "time" / ".zarray").write_text(json.dumps(time_meta))
    (rgb_dir / "data" / "0.0.0").write_bytes(b"chunk0")
    (rgb_dir / "data" / "1.0.0").write_bytes(b"chunk1")
    (rgb_dir / "time" / "0").write_bytes(b"time0")

    # Explicit "dimension_separator": null must be treated as ".".
    assert camera.check_rgb_zarr_complete(shot_dir) == (True, "ok")

    # Missing or empty .zattrs -> incomplete.
    zattrs_backup = zattrs_path.read_text()
    zattrs_path.unlink()
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "incomplete")
    zattrs_path.write_text("")
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "incomplete")
    zattrs_path.write_text(zattrs_backup)
    assert camera.check_rgb_zarr_complete(shot_dir) == (True, "ok")

    # Missing or empty .zgroup -> incomplete.
    zgroup_backup = zgroup_path.read_text()
    zgroup_path.unlink()
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "incomplete")
    zgroup_path.write_text("")
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "incomplete")
    zgroup_path.write_text(zgroup_backup)
    assert camera.check_rgb_zarr_complete(shot_dir) == (True, "ok")

    # Present .zattrs without exposure attribute -> distinct no_exposure_attr status (no fallback).
    zattrs_path.write_text(json.dumps({"camera": "IPX"}))
    assert camera.check_rgb_zarr_complete(shot_dir) == (False, "no_exposure_attr")
    with pytest.raises(ValueError, match="exposure"):
        camera.extract_exposure_s({"camera": "IPX"}, np.array([0.0, 0.0091]))


def test_make_mast_video_loader_ddp_drop_last_per_rank():
    """Under DDP, drop_last uses per-rank len(sampler) so small datasets never yield 0 batches."""

    class _TinyVideoDataset(torch.utils.data.Dataset):
        def __init__(self, n: int) -> None:
            self.n = n

        def __len__(self) -> int:
            return self.n

        def __getitem__(self, idx: int):
            return torch.full((1, 1, 8, 8), float(idx)), torch.ones(1, dtype=torch.float32)

    # batch_size (4) <= len(dataset) (6) < batch_size * world_size (8) -> 3 samples/rank.
    ds = _TinyVideoDataset(6)
    for rank in (0, 1):
        loader = camera.make_mast_video_loader(
            ds,
            batch_size=4,
            num_workers=0,
            rank=rank,
            world_size=2,
            shuffle=False,
        )
        assert len(loader) == 1
        batches = list(loader)
        assert len(batches) == 1
        assert batches[0][0].shape == (3, 1, 1, 8, 8)


def test_decode_camera_mixed_missing_stats_and_image_tokens():
    """decode_camera returns all-NaN frames with no exception when either stats or image tokens are missing."""
    cfg = camera.default_video_codec_config(
        height=20, width=20, patch_h=20, patch_w=20, d_model=32, enc_depth=1, dec_depth=1, heads=2
    )
    codec = VideoCodec(cfg).eval()
    rng = np.random.default_rng(11)
    s_floor, stats_codec, u_std = camera.fit_camera_stats(
        rng.uniform(1.0, 5.0, size=200), rng.uniform(0.1, 1.5, size=200)
    )
    cam_fit = camera.CameraFit(
        s_floor=s_floor,
        time_convention="end",
        stats_codec=stats_codec,
        u_std=u_std,
        meta={},
    )

    n_tok = int(codec.cfg.n_tok)
    missing_cam = int(codec.codebook_size)
    missing_stat = int(stats_codec.missing_code)

    # 0: both present; 1: stats missing + image present; 2: image missing + stats present;
    # 3: one stats token missing + image present.
    codes = torch.zeros((4, n_tok), dtype=torch.int64)
    codes[2] = missing_cam
    stats_codes = torch.tensor(
        [
            [10, 10],
            [missing_stat, missing_stat],
            [10, 10],
            [10, missing_stat],
        ],
        dtype=torch.int64,
    )

    counts, u = camera.decode_camera(codec, cam_fit, codes, stats_codes, return_u=True)
    assert counts.shape == (4, 20, 20)
    assert np.isfinite(counts[0]).all() and np.isfinite(u[0]).all()
    for f in (1, 2, 3):
        assert np.isnan(counts[f]).all()
        assert np.isnan(u[f]).all()


def test_build_manifest_rejects_empty_camera_sha(tmp_path: Path):
    """framecache.build_manifest raises if camera is present and any camera sha is empty or None."""
    act_stats = framecache.ActuatorStats(
        names=("summary-power_nbi",),
        channels=(1,),
        mu=np.array([0.0], dtype=np.float64),
        sd=np.array([1.0], dtype=np.float64),
        n_train_shots=1,
        n_valid_frames=np.array([10], dtype=np.int64),
    )
    common = dict(
        signals=(),
        codecs={},
        act_stats=act_stats,
        assignment_path=tmp_path / "a.json",
        assignment_sha256="a" * 64,
        fit_dir=tmp_path / "fit",
        fit_sha256="b" * 64,
        cache_root=tmp_path / "cache",
        stats_sha256="c" * 64,
        camera_codec_path=tmp_path / "codec.pt",
        camera_fit_path=tmp_path / "cam_fit",
        camera_frames_dir=tmp_path / "frames",
    )
    for bad_shas in (
        dict(camera_codec_sha256="", camera_fit_sha256="d" * 64, camera_frames_index_sha256="e" * 64),
        dict(camera_codec_sha256="d" * 64, camera_fit_sha256=None, camera_frames_index_sha256="e" * 64),
        dict(camera_codec_sha256="d" * 64, camera_fit_sha256="e" * 64, camera_frames_index_sha256="   "),
    ):
        with pytest.raises(ValueError, match="sha256"):
            framecache.build_manifest(**common, **bad_shas)

    manifest = framecache.build_manifest(
        **common,
        camera_codec_sha256="d" * 64,
        camera_fit_sha256="e" * 64,
        camera_frames_index_sha256="f" * 64,
    )
    assert manifest["camera_codec_sha256"] == "d" * 64
    assert "camera-rgb" in manifest["modalities"]


def test_camera_frames_revalidate_and_dry_run(tmp_path: Path):
    """port_camera_frames --revalidate detects cached exposure mismatch; --dry-run writes nothing."""
    port_frames = _load_script_module("port_camera_frames.py")
    level1 = tmp_path / "level1"
    out_dir = tmp_path / "frames_out"
    splits_path = tmp_path / "splits.json"
    framecache.save_json_atomic({"train": ["77777"], "val": [], "test": []}, splits_path)

    rgb_dir = level1 / "77777.zarr" / "rgb"
    grp = zarr.open_group(str(rgb_dir), mode="w", zarr_format=2)
    grp.attrs["exposure"] = 4545.0  # 4.545 ms
    raw = np.full((4, 16, 16), 200, dtype=np.uint16)
    times = np.array([0.01, 0.0191, 0.0282, 0.0373], dtype=np.float64)
    grp.create_array("data", data=raw, chunks=(4, 16, 16))
    grp.create_array("time", data=times, chunks=(4,))

    port_frames.build_camera_frames_cache(
        level1_root=level1, splits_path=splits_path, out_dir=out_dir, workers=1
    )
    fr0 = camera.load_frames_npz(out_dir / "77777.npz", load_u=True)
    assert fr0["exposure_s"] == pytest.approx(0.004545)

    # Corrupt cached exposure_s to simulate a 2x-wrong fallback cache.
    camera.save_frames_npz_atomic(
        out_dir / "77777.npz",
        u=fr0["u"],
        time=fr0["time"],
        m=fr0["m"],
        s=fr0["s"],
        raw_mean_counts=fr0["raw_mean_counts"],
        exposure_s=0.0091,
    )
    idx_before = (out_dir / "index.json").read_text()

    dry = port_frames.build_camera_frames_cache(
        level1_root=level1,
        splits_path=splits_path,
        out_dir=out_dir,
        workers=1,
        revalidate=True,
        dry_run=True,
    )
    assert dry["dry_run"] is True
    assert "77777" in dry["revalidated_disagreed"]
    assert camera.load_frames_npz(out_dir / "77777.npz", load_u=False)["exposure_s"] == pytest.approx(0.0091)
    assert (out_dir / "index.json").read_text() == idx_before

    live = port_frames.build_camera_frames_cache(
        level1_root=level1,
        splits_path=splits_path,
        out_dir=out_dir,
        workers=1,
        revalidate=True,
        dry_run=False,
    )
    assert "77777" in live["revalidated_disagreed"]
    assert camera.load_frames_npz(out_dir / "77777.npz", load_u=False)["exposure_s"] == pytest.approx(0.004545)


def test_incomplete_camera_manifest_raises_in_plots_and_dlog10_corner_box(tmp_path: Path, monkeypatch):
    """Recon/rollout plots raise on camera modalities without paths; plot_shot_rollout prints dlog10 for LogBandDCT."""
    port_recon = _load_script_module("port_recon_plots.py")
    port_roll = _load_script_module("port_rollout_plots.py")

    for cam_mod in camera.CAMERA_MODALITIES:
        cdir = tmp_path / f"cache_{cam_mod}"
        cdir.mkdir()
        manifest = {
            "t0_start": 0.0,
            "dt_frame": 0.005,
            "cache_v1_root": str(tmp_path),
            "assignment_path": str(tmp_path / "a.json"),
            "fit_dir": str(tmp_path / "fit"),
            "modalities": {cam_mod: {"family": "video", "n_tok": 48, "codebook_size": 1001}},
            "actuators": {
                "names": ["summary-power_nbi"],
                "channels": [1],
                "total_dim": 1,
                "mu": [0.0],
                "sd": [1.0],
                "n_train_shots": 1,
                "n_valid_frames": [10],
            },
        }
        framecache.save_json_atomic(manifest, cdir / "_codec_manifest.json")

        monkeypatch.setattr(framecache, "build_mast_codecs", lambda *a, **kw: ({}, {}, {}))
        with pytest.raises(ValueError, match="camera_codec_path"):
            port_recon.run_recon_plots(
                cache_dir=cdir, out_dir=tmp_path / f"out_recon_{cam_mod}", shots=[], device_spec="cpu"
            )
        with pytest.raises(ValueError, match="camera_codec_path"):
            port_roll.run_rollout_plots(
                ckpt=tmp_path / "dummy.pt",
                cache_dir=cdir,
                out_dir=tmp_path / f"out_roll_{cam_mod}",
                shots=[],
                device_spec="cpu",
            )

    # plot_shot_rollout prints |dlog10| for LogBandDCT (log_floor present) and NRMSE for BandDCT.
    import matplotlib.axes

    captured_texts: list[str] = []
    orig_text = matplotlib.axes.Axes.text

    def _spy_text(self, x, y, s, *args, **kwargs):
        captured_texts.append(str(s))
        return orig_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", _spy_text)

    sig = "magnetics-b_field_pol_probe_omv_voltage"
    arr = np.full((1, 4, 6), 1e-4, dtype=np.float64)
    times = np.arange(6, dtype=np.float64) * 0.005
    base_entry = {"present": True, "times_s": times, "truth": arr, "ceiling": arr, "rollout": arr}
    base_metrics = {
        sig: {
            "rollout_nrmse": 9.876,
            "ceiling_nrmse": 8.765,
            "rollout_dlog10": 0.123,
            "ceiling_dlog10": 0.045,
        }
    }

    # LogBandDCT (with log_floor) -> must print |dlog10| (0.123 / 0.045), not linear NRMSE (9.876).
    captured_texts.clear()
    rollout.plot_shot_rollout(
        "9001",
        {sig: {**base_entry, "log_floor": np.full((1, 4), 1e-8)}},
        base_metrics,
        k0=2,
        n_frames=6,
        t0=0.0,
        dt_frame=0.005,
        out_path=tmp_path / "roll_log.png",
    )
    joined_log = " ".join(captured_texts)
    assert "|dlog10| 0.123 dec" in joined_log and "|dlog10| 0.045 dec" in joined_log
    assert "9.876" not in joined_log

    # BandDCT (no log_floor) -> must print linear NRMSE (9.876 / 8.765).
    captured_texts.clear()
    rollout.plot_shot_rollout(
        "9001",
        {sig: base_entry},
        base_metrics,
        k0=2,
        n_frames=6,
        t0=0.0,
        dt_frame=0.005,
        out_path=tmp_path / "roll_lin.png",
    )
    joined_lin = " ".join(captured_texts)
    assert "roll 9.876" in joined_lin and "ceil 8.765" in joined_lin
    assert "dlog10" not in joined_lin


def test_port_rollout_plots_writes_camera_metrics(tmp_path: Path, monkeypatch):
    """port_rollout_plots resolves manifest camera and writes camera-rgb metrics into rollout_metrics.json."""
    port_roll = _load_script_module("port_rollout_plots.py")

    cfg = camera.default_video_codec_config(
        height=20, width=20, patch_h=20, patch_w=20, d_model=32, enc_depth=1, dec_depth=1, heads=2
    )
    codec = VideoCodec(cfg).eval()
    ckpt_path = tmp_path / "codec_best.pt"
    torch.save({"codec": codec.state_dict(), "cfg": cfg}, ckpt_path)

    frames_dir = tmp_path / "frames"
    frames_dir.mkdir()
    u = np.full((6, 20, 20), 4.0, dtype=np.float16)
    times = 0.010 + np.arange(6, dtype=np.float64) * 0.009
    m = np.full(6, 4.0, dtype=np.float32)
    s = np.full(6, 0.5, dtype=np.float32)
    camera.save_frames_npz_atomic(
        frames_dir / "5555.npz",
        u=u,
        time=times,
        m=m,
        s=s,
        raw_mean_counts=np.full(6, 54.0, dtype=np.float32),
        exposure_s=0.009,
    )
    framecache.save_json_atomic({"shots": {"5555": {"status": "ok"}}}, frames_dir / "index.json")

    fit_dir = tmp_path / "cam_fit"
    rng = np.random.default_rng(5)
    s_floor, stats_codec, u_std = camera.fit_camera_stats(
        rng.uniform(2.0, 6.0, size=100), rng.uniform(0.1, 1.0, size=100)
    )
    fit_sha = camera.save_camera_fit(
        fit_dir,
        s_floor,
        stats_codec,
        {"time_convention": "end", "s_floor": s_floor, "u_std": u_std, "n_bins": 64},
    )
    cam_fit = camera.load_camera_fit(fit_dir)

    codes, stats_codes, present = camera.encode_camera_codes(
        codec, cam_fit, frames_dir / "5555.npz", t0=0.0, n_frames=12, dt_frame=0.005
    )
    assert present is True

    cache_dir = tmp_path / "frame_codes"
    cache_dir.mkdir()
    manifest = {
        "t0_start": 0.0,
        "dt_frame": 0.005,
        "cache_v1_root": str(tmp_path),
        "assignment_path": str(tmp_path / "a.json"),
        "fit_dir": str(tmp_path / "fit"),
        "modalities": {
            camera.CAMERA_MODALITY: {"family": "video", "n_tok": cfg.n_tok, "codebook_size": 1001},
            camera.CAMERA_STATS_MODALITY: {"family": "slowts", "n_tok": 2, "codebook_size": 65},
        },
        "actuators": {
            "names": ["summary-power_nbi"],
            "channels": [1],
            "total_dim": 1,
            "mu": [0.0],
            "sd": [1.0],
            "n_train_shots": 1,
            "n_valid_frames": [12],
        },
        "camera_codec_path": str(ckpt_path),
        "camera_codec_sha256": camera.file_sha256(ckpt_path),
        "camera_fit_path": str(fit_dir),
        "camera_fit_sha256": fit_sha,
        "camera_frames_dir": str(frames_dir),
        "camera_frames_index_sha256": camera.file_sha256(frames_dir / "index.json"),
    }
    framecache.save_json_atomic(manifest, cache_dir / "_codec_manifest.json")

    ro = SimpleNamespace(
        gt_codes={camera.CAMERA_MODALITY: codes, camera.CAMERA_STATS_MODALITY: stats_codes},
        pred_codes={camera.CAMERA_MODALITY: codes.clone(), camera.CAMERA_STATS_MODALITY: stats_codes.clone()},
        k0=2,
        n_frames=12,
    )
    entry = {
        "t0": 0.0,
        "n_frames": 12,
        "present": {camera.CAMERA_MODALITY: True, camera.CAMERA_STATS_MODALITY: True},
    }
    monkeypatch.setattr(framecache, "build_mast_codecs", lambda *a, **kw: ({}, {}, {}))
    monkeypatch.setattr(
        rollout, "load_mast_model", lambda *a, **kw: (None, SimpleNamespace(k0_seed=2), 10)
    )
    monkeypatch.setattr(rollout, "rollout_mast_shot", lambda *a, **kw: (ro, entry, None))
    monkeypatch.setattr(rollout, "plot_shot_rollout", lambda *a, **kw: a[7])

    out_dir = tmp_path / "rollout_out"
    rec = port_roll.run_rollout_plots(
        ckpt=tmp_path / "dyn.pt",
        cache_dir=cache_dir,
        out_dir=out_dir,
        shots=["5555"],
        device_spec="cpu",
    )
    assert camera.CAMERA_MODALITY in rec["per_shot"]["5555"]["signals"]
    assert camera.CAMERA_MODALITY in rec["summary"]
    assert math.isfinite(rec["summary"][camera.CAMERA_MODALITY]["median_rollout_nrmse"])
    disk_rec = json.loads((out_dir / "rollout_metrics.json").read_text())
    assert camera.CAMERA_MODALITY in disk_rec["summary"]


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
