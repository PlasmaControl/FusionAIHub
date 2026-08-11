"""Phase-B production streaming dataset: window index + shapes + collate over a code cache."""

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
