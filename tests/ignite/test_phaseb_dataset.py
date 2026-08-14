"""Phase-B production streaming dataset: window index + shapes + collate over a code cache,
plus the ACTUATOR TIME-BASE guard (see the `actuator_frames` section at the bottom)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec
import tokamak_foundation_model.ignite.train_dynamics as td
from tokamak_foundation_model.ignite.train_dynamics import (
    FrameCodeDataset, _collate_frames, PLACEHOLDER_MODALITIES, load_frozen_codecs,
    precompute_frame_codes)


def _tiny():
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4)),
        d_model=16, depth=1, n_heads=2, k0_seed=2, n_predict=2, actuator_dim=6,  # max_frames=4
    )


def _write_cache(dirpath, shot, n_frames, cfg, drop_last_modality=False):
    mods = cfg.modalities[:-1] if drop_last_modality else cfg.modalities
    codes = {m.name: torch.randint(0, m.codebook_size, (n_frames, m.n_tok), dtype=torch.int16) for m in mods}
    act = torch.randn(n_frames, cfg.actuator_dim, dtype=torch.float16)
    torch.save({"codes": codes, "actuators": act, "n_frames": n_frames}, dirpath / f"{shot}.pt")


def test_window_index_counts(tmp_path):
    cfg = _tiny()                                   # max_frames = 4
    _write_cache(tmp_path, "s1", cfg.max_frames + 3, cfg)   # 3 extra frames -> 4 sliding windows
    _write_cache(tmp_path, "s2", cfg.max_frames, cfg)       # exactly 1 window
    ds = FrameCodeDataset(tmp_path, ["s1", "s2", "does_not_exist"], cfg)
    assert len(ds) == 4 + 1                          # s1:4, s2:1, missing shot skipped


def test_item_and_collate_shapes(tmp_path):
    cfg = _tiny()
    _write_cache(tmp_path, "s1", cfg.max_frames + 1, cfg)
    ds = FrameCodeDataset(tmp_path, ["s1"], cfg)
    codes, act = ds[0]
    for m in cfg.modalities:
        assert codes[m.name].shape == (cfg.max_frames, m.n_tok)
        assert codes[m.name].dtype == torch.long           # decompressed from int16 cache
    assert act.shape == (cfg.max_frames, cfg.actuator_dim)
    cb, ab = _collate_frames([ds[0], ds[1]])
    for m in cfg.modalities:
        assert cb[m.name].shape == (2, cfg.max_frames, m.n_tok)
    assert ab.shape == (2, cfg.max_frames, cfg.actuator_dim)


def test_incomplete_cache_skipped(tmp_path):
    # a shot whose cache lacks a required modality is dropped (frame layout must be complete)
    cfg = _tiny()
    _write_cache(tmp_path, "bad", cfg.max_frames + 2, cfg, drop_last_modality=True)
    _write_cache(tmp_path, "good", cfg.max_frames + 2, cfg)
    ds = FrameCodeDataset(tmp_path, ["bad", "good"], cfg)
    assert len(ds) == 3          # only "good" (3 windows); "bad" skipped entirely


def test_sliding_windows_are_offset(tmp_path):
    cfg = _tiny()
    _write_cache(tmp_path, "s1", cfg.max_frames + 2, cfg)
    ds = FrameCodeDataset(tmp_path, ["s1"], cfg)
    c0, _ = ds[0]
    c1, _ = ds[1]
    # window 1 is window 0 shifted by one frame -> frame1 of w0 == frame0 of w1
    assert torch.equal(c0["a"][1], c1["a"][0])


# --- reserved-slot placeholder (filterscopes codec pending redesign) -------------------------

def test_load_frozen_codecs_skips_placeholders():
    # reserved-slot modalities have no codec to load -> load_frozen_codecs must skip them (never
    # touch the non-existent ckpt path), so asking for a placeholder alone yields an empty dict.
    assert PLACEHOLDER_MODALITIES  # non-empty (filterscopes)
    assert load_frozen_codecs(list(PLACEHOLDER_MODALITIES)) == {}


def test_precompute_emits_constant_placeholder(tmp_path, monkeypatch):
    """precompute fills reserved-slot modalities with a constant code so the frame keeps full width."""
    N = 12                                   # frames the fake real-codec dataset yields

    class _FakeDS:                           # single-shot dataset stand-in
        def __len__(self): return N
        def __getitem__(self, t): return torch.zeros(1, 3)

    monkeypatch.setattr(td, "_single_shot_dataset", lambda *a, **k: _FakeDS())
    monkeypatch.setattr(td, "encode_flat", lambda codec, x: torch.arange(4).view(1, 4))  # (1, n_tok=4)
    monkeypatch.setattr(td, "actuator_frames", lambda shot, n, dd: torch.zeros(n, 70))

    codecs = {"real": (object(), object(), "spectro")}
    n = precompute_frame_codes(["77"], codecs, tmp_path, data_dir="/x",
                               placeholder_specs={"filterscopes": 5})
    assert n == 1
    d = torch.load(tmp_path / "77.pt", weights_only=False)
    assert d["codes"]["real"].shape == (N, 4)                 # real modality encoded
    ph = d["codes"]["filterscopes"]
    assert ph.shape == (N, 5) and ph.dtype == torch.int16     # reserved slot, full width
    assert int(ph.min()) == 0 and int(ph.max()) == 0          # constant placeholder code


# --------------------------------------------------------------------------------------------- #
# actuator_frames TIME BASE — regression guard for the 2026-08-11 misalignment
#
# The bug: actuator_frames indexed ydata by int(t * fs) from the ARRAY START, ignoring xdata[0],
# while every diagnostic goes through data_loader._load_signal_raw, which uses
# round((t - xdata_start_s) * actual_fs). The actuator groups do not start at t=0 (gas -10.0 s,
# beam_voltage ~-6.3 s, rmp ~-1.04 s and shot-dependent), so the model was fed actuators from
# seconds BEFORE the frame — ~83% of the actuator input variance was decorrelated from the true
# trajectory. A second, independent half: xdata is float32, so 1/median(diff(x)) loses the sample
# step to cancellation (off by up to +0.82%), a GROWING drift of ~1.8 frames over an 11 s window.
#
# WHY PULSES, NOT RAMPS: actuator_frames z-scores every channel across frames, so any
# shift-invariant probe is blind to the offset — a ramp displaced by 10 s z-scores IDENTICALLY.
# What survives the z-score is WHICH FRAME a localized feature lands in, so these fixtures plant
# a pulse at a known absolute time and assert its frame index.
# --------------------------------------------------------------------------------------------- #
FS = 10_000.0
FRAME_S = 0.05
SAMPLES_PER_FRAME = int(FS * FRAME_S)          # 500
# Agreement tolerance against the searchsorted oracle. Bounded BELOW by the +-1-sample boundary
# difference between round() and searchsorted (measured ~5e-3 on smooth fixtures) and ABOVE by
# what a misaligned implementation scores (~2.9 here) — i.e. this sits ~57x under the failure it
# must catch, so it discriminates by a wide margin rather than by a hair.
_ALIGN_ATOL = 5e-2


def _write_shot_h5(dirpath, shot, groups):
    """Minimal {shot}_processed.h5. ``groups``: {name: (ydata (C,T), xdata (T,))}."""
    import h5py
    with h5py.File(Path(dirpath) / f"{shot}_processed.h5", "w") as f:
        for name, (y, x) in groups.items():
            g = f.create_group(name)
            g.create_dataset("ydata", data=np.asarray(y, dtype=np.float32))
            g.create_dataset("xdata", data=np.asarray(x, dtype=np.float32))   # float32, as in prod


def _grid(t_start, t_end, fs=FS):
    """float32 timestamp grid — the dtype is deliberate: it is what breaks 1/median(diff(x))."""
    n = int(round((t_end - t_start) * fs)) + 1
    return (np.arange(n, dtype=np.float64) / fs + t_start).astype(np.float32)


def _pulse_channel(x, lo_s, hi_s, n_ch, ch=0, amp=1.0):
    """(n_ch, T) zeros with a unit pulse on ``ch`` over ABSOLUTE time [lo_s, hi_s)."""
    y = np.zeros((n_ch, x.size), dtype=np.float64)
    y[ch, (x >= lo_s) & (x < hi_s)] = amp
    return y


def _col(group):
    """First column index of an actuator group inside the 70-wide vector."""
    off = 0
    for key, nch in td._ACT_SPEC:
        if key == group:
            return off
        off += nch
    raise KeyError(group)


def _reference_frames(shot, n_frames, data_dir, t0_start):
    """Independent oracle: window by ABSOLUTE shot time via searchsorted on xdata.

    A different algorithm from the implementation (which converts time to an index), so agreement
    is real evidence rather than a restatement of the code under test.
    """
    import h5py
    cols = []
    with h5py.File(Path(data_dir) / f"{shot}_processed.h5", "r") as f:
        for key, nch in td._ACT_SPEC:
            if key in f and "ydata" in f[key] and f[key]["ydata"].shape[-1] > 1:
                y = np.nan_to_num(np.asarray(f[key]["ydata"], dtype=np.float64))
                x = np.asarray(f[key]["xdata"], dtype=np.float64)
                fr = []
                for t in range(n_frames):
                    a, b = t0_start + t * FRAME_S, t0_start + (t + 1) * FRAME_S
                    i0, i1 = np.searchsorted(x, a), np.searchsorted(x, b)
                    seg = y[:, i0:i1]
                    fr.append(seg.mean(1) if seg.size else np.zeros(y.shape[0]))
                col = np.stack(fr, 0)
                if col.shape[1] != nch:
                    fixed = np.zeros((n_frames, nch))
                    fixed[:, : min(nch, col.shape[1])] = col[:, :nch]
                    col = fixed
            else:
                col = np.zeros((n_frames, nch))
            cols.append(col)
    a = np.concatenate(cols, axis=1)
    return (a - a.mean(0, keepdims=True)) / (a.std(0, keepdims=True) + 1e-6)


def test_actuator_frames_windows_on_absolute_shot_time(tmp_path):
    """A pulse at a known ABSOLUTE time must land in the frame that absolute time belongs to.

    rmp's record starts at -1.2 s. Indexing from the array start (the old bug) shifts every
    window by +1.2 s, i.e. 24 frames.
    """
    x = _grid(-1.2, 5.0)
    y = _pulse_channel(x, 1.30, 1.35, n_ch=12, ch=0)          # one frame wide, at abs 1.30 s
    _write_shot_h5(tmp_path, "1", {"rmp": (y, x)})
    act = td.actuator_frames("1", 60, tmp_path, t0_start=1.0).numpy()
    c = _col("rmp")
    got = int(act[:, c].argmax())
    assert got == 6, f"pulse at abs 1.30 s with t0=1.0 belongs in frame 6, landed in {got}"
    # and it is a genuine single-frame spike, not a flat/degenerate column
    assert act[6, c] > 3.0 and np.all(act[:6, c] < 0.5) and np.all(act[7:, c] < 0.5)


def test_actuator_frames_rate_comes_from_span_not_median_diff(tmp_path):
    """Sampling rate must come from the span, not from median(diff(xdata)).

    xdata is float32; with a -10 s origin the 1e-4 step is lost to cancellation, so
    1/median(diff(x)) is wrong by ~0.8% — a DRIFT that grows with distance from the record start.

    The grid MUST span the production gas record (-10 .. 94.8576 s): the cancellation is driven by
    the timestamp magnitude / step ratio, i.e. by the sample count, and is non-monotonic in it —
    shorter spans (-10..20, -10..50) reproduce only 0.1% and would not move the pulse a full frame.
    One channel is enough (actuator_frames zero-pads a group to its declared width), which keeps
    the fixture at ~4 MB.
    """
    x = _grid(-10.0, 94.8576)
    fs_span = (x.size - 1) / float(np.float64(x[-1]) - np.float64(x[0]))
    fs_median = 1.0 / np.median(np.diff(np.asarray(x, dtype=np.float64)))
    # fixture precondition: the two estimators genuinely disagree (else the test proves nothing)
    assert abs(fs_median - fs_span) / fs_span > 1e-3, (
        f"fixture failed to reproduce the float32 cancellation: {fs_median} vs {fs_span}")
    y = _pulse_channel(x, 1.00, 1.05, n_ch=1, ch=0)           # abs 1.00 s: 11 s after record start
    _write_shot_h5(tmp_path, "2", {"gas_flow": (y, x)})
    act = td.actuator_frames("2", 40, tmp_path, t0_start=0.0).numpy()
    c = _col("gas_flow")
    got = int(act[:, c].argmax())
    assert got == 20, f"pulse at abs 1.00 s with t0=0.0 belongs in frame 20, landed in {got}"
    # the median-diff rate would have mis-placed it — confirms this test discriminates
    drift_frames = abs(fs_median - fs_span) * 11.0 / SAMPLES_PER_FRAME
    assert drift_frames > 1.0, f"fixture drift {drift_frames:.2f} frames too small to detect"


def test_actuator_frames_matches_absolute_time_reference(tmp_path):
    """Full 70-wide vector agrees with the independent searchsorted oracle, across mixed offsets."""
    specs = {                                     # group -> (record start, n_channels)
        "rmp": (-1.05, 12), "gas_raw": (-10.0, 11), "beam_voltage": (-6.3, 8), "pinj": (0.0, 8),
    }
    groups = {}
    for i, (g, (x0, nch)) in enumerate(specs.items()):
        x = _grid(x0, 6.0)
        # smooth, band-limited content: a +-1 sample difference at a window edge is then
        # negligible, so exact agreement is a meaningful assertion
        t = np.asarray(x, dtype=np.float64)
        y = np.stack([np.sin(2 * np.pi * (0.7 + 0.3 * c + i) * t) for c in range(nch)], 0)
        groups[g] = (y, x)
    _write_shot_h5(tmp_path, "3", groups)
    got = td.actuator_frames("3", 50, tmp_path, t0_start=1.0).numpy()
    exp = _reference_frames("3", 50, tmp_path, t0_start=1.0)
    assert np.allclose(got, exp, atol=_ALIGN_ATOL), f"max deviation {np.abs(got - exp).max():.4g}"
    # the sharp claim: alignment, not merely proximity. Boundary rounding cannot move a
    # correlation off 1.0, but a time shift of even one frame does.
    live = np.where(np.abs(exp).max(0) > 1e-6)[0]
    cors = np.array([np.corrcoef(got[:, j], exp[:, j])[0, 1] for j in live])
    assert cors.min() > 0.999, f"worst channel corr {cors.min():.5f} (col {live[cors.argmin()]})"


def test_actuator_frames_clamps_record_starting_after_window(tmp_path):
    """A record starting AFTER the window origin must zero-fill, never wrap to the array tail.

    Unclamped, round((t - x0) * fs) is negative and numpy slices from the END of the record —
    silently serving late-shot data as if it were the requested early frames.
    """
    x = _grid(2.0, 4.0)                                        # record starts 2 s AFTER t0=0
    y = np.zeros((12, x.size), dtype=np.float64)
    y[:, -12_000:] = 7.0                                       # distinctive tail marker
    _write_shot_h5(tmp_path, "4", {"rmp": (y, x)})
    act = td.actuator_frames("4", 40, tmp_path, t0_start=0.0).numpy()   # window = abs [0, 2) s
    c = _col("rmp")
    block = act[:, c:c + 12]
    assert np.abs(block).max() == 0.0, (
        "frames before the record start picked up data — negative index wrapped to the tail")


def test_legacy_windowing_fails_the_guard(tmp_path):
    """The OLD implementation must FAIL these assertions.

    Without this, the guard above could be passing vacuously. _legacy_actuator_frames is kept in
    the module as the cache-patch integrity check, so the historical behaviour is testable.
    """
    x = _grid(-1.2, 5.0)
    y = _pulse_channel(x, 1.30, 1.35, n_ch=12, ch=0)
    _write_shot_h5(tmp_path, "5", {"rmp": (y, x)})
    legacy = td._legacy_actuator_frames("5", 60, tmp_path, t0_start=1.0).numpy()
    c = _col("rmp")
    assert int(legacy[:, c].argmax()) != 6, (
        "legacy windowing agreed with the fixed one — the fixture no longer exercises the bug")
    exp = _reference_frames("5", 60, tmp_path, t0_start=1.0)
    assert not np.allclose(legacy, exp, atol=_ALIGN_ATOL), (
        "legacy output matched the absolute-time oracle")


_REAL_H5 = Path(td.spike.DEFAULT_DATA_DIR) / "200729_processed.h5"


@pytest.mark.skipif(not _REAL_H5.exists(), reason=f"real-shot data not available: {_REAL_H5}")
def test_actuator_frames_real_shot_matches_reference():
    """Real H5: every live channel tracks the absolute-time oracle.

    Correlation, not equality: on noise-dominated channels (gas_raw) the one-sample difference
    between round() and searchsorted at each window edge moves a 500-sample mean by ~1%, which is
    boundary rounding rather than misalignment. A misaligned implementation scores ~0 here.
    """
    got = td.actuator_frames("200729", 20, td.spike.DEFAULT_DATA_DIR, t0_start=1.0).numpy()
    exp = _reference_frames("200729", 20, td.spike.DEFAULT_DATA_DIR, t0_start=1.0)
    live = np.where(np.abs(exp).max(0) > 1e-6)[0]
    assert live.size >= 20, f"only {live.size} live actuator channels — fixture shot changed?"
    cors = np.array([np.corrcoef(got[:, j], exp[:, j])[0, 1] for j in live])
    assert cors.min() > 0.95, f"worst channel corr {cors.min():.4f} (col {live[cors.argmin()]})"
