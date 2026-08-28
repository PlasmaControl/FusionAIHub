# dataset-integration tests for FrameCodeDataset are appended by a later task.
"""Consolidated text-embedding H5 loader: load_text_embeddings + lookup."""

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from tokamak_foundation_model.ignite.text_embed import load_text_embeddings, lookup

EMBED_DIM = 8
SHOTS = [158103, 158104, 158105]


def _write_h5(path, embed_dim=EMBED_DIM, shots=SHOTS):
    n = len(shots)
    rng = np.random.default_rng(0)
    input_vecs = rng.standard_normal((n, embed_dim)).astype(np.float16)
    total_vecs = rng.standard_normal((n, embed_dim)).astype(np.float16)
    with h5py.File(path, "w") as f:
        f.attrs["embed_dim"] = embed_dim
        f.create_dataset("shots", data=np.array(shots, dtype=np.int64))
        f.create_dataset("input", data=input_vecs)
        f.create_dataset("total", data=total_vecs)
    return input_vecs, total_vecs


def test_load_text_embeddings_returns_unit_norm_prefix_slices(tmp_path):
    path = tmp_path / "text_embeddings.h5"
    _write_h5(path)

    embeds = load_text_embeddings(path, dim=4)

    assert set(embeds.keys()) == {str(s) for s in SHOTS}
    for vec in embeds.values():
        assert vec.shape == (4,)
        assert vec.dtype == torch.float32
        assert torch.isclose(vec.norm(), torch.tensor(1.0), atol=1e-4)


def test_load_text_embeddings_dim_exceeds_embed_dim_raises(tmp_path):
    path = tmp_path / "text_embeddings.h5"
    _write_h5(path)

    with pytest.raises(Exception):
        load_text_embeddings(path, dim=EMBED_DIM + 1)


def test_load_text_embeddings_bogus_key_raises(tmp_path):
    path = tmp_path / "text_embeddings.h5"
    _write_h5(path)

    with pytest.raises(Exception):
        load_text_embeddings(path, dim=4, key="bogus")


def test_load_text_embeddings_total_key_reads_other_dataset(tmp_path):
    path = tmp_path / "text_embeddings.h5"
    input_vecs, total_vecs = _write_h5(path)

    input_embeds = load_text_embeddings(path, dim=EMBED_DIM, key="input")
    total_embeds = load_text_embeddings(path, dim=EMBED_DIM, key="total")

    shot = str(SHOTS[0])
    assert not torch.allclose(input_embeds[shot], total_embeds[shot])

    expected = torch.from_numpy(total_vecs[0].astype(np.float32))
    expected = expected / expected.norm().clamp_min(1e-8)
    assert torch.allclose(total_embeds[shot], expected, atol=1e-4)


def test_lookup_hit_and_miss(tmp_path):
    path = tmp_path / "text_embeddings.h5"
    _write_h5(path)
    embeds = load_text_embeddings(path, dim=4)

    vec, hit = lookup(embeds, SHOTS[0], dim=4)
    assert hit is True
    assert vec.shape == (4,)

    vec, hit = lookup(embeds, 999999, dim=4)
    assert hit is False
    assert torch.equal(vec, torch.zeros(4))


# --------------------------------------------------------------------------------------------- #
# FrameCodeDataset integration: the dataset's 4th tuple element is the per-shot text embedding.
# --------------------------------------------------------------------------------------------- #
from tokamak_foundation_model.ignite.dynamics_config import DynamicsConfig, ModalitySpec  # noqa: E402
from tokamak_foundation_model.ignite.train_dynamics import FrameCodeDataset, _collate_frames  # noqa: E402

DS_DIM = 4


def _tiny_cfg():
    return DynamicsConfig(
        modalities=(ModalitySpec("a", "spectro", 3, 5), ModalitySpec("b", "slowts", 2, 4)),
        d_model=16, depth=1, n_heads=2, k0_seed=2, n_predict=2, actuator_dim=6,  # max_frames=4
    )


def _write_ds_cache(dirpath, shot, n_frames, cfg):
    codes = {m.name: torch.randint(0, m.codebook_size, (n_frames, m.n_tok), dtype=torch.int16)
             for m in cfg.modalities}
    act = torch.randn(n_frames, cfg.actuator_dim, dtype=torch.float16)
    torch.save({"codes": codes, "actuators": act, "n_frames": n_frames}, dirpath / f"{shot}.pt")


def _text_embeds():
    """Covers shot '158103'; '158104' is deliberately left out (a miss)."""
    g = torch.Generator().manual_seed(0)
    return {"158103": torch.randn(DS_DIM, generator=g)}


def test_getitem_is_4tuple_covered_shot_gets_its_embedding(tmp_path):
    cfg = _tiny_cfg()
    _write_ds_cache(tmp_path, "158103", cfg.max_frames + 1, cfg)
    embeds = _text_embeds()
    ds = FrameCodeDataset(tmp_path, ["158103"], cfg, text_embeds=embeds)
    item = ds[0]
    assert len(item) == 4
    codes, act, presence, text = item
    assert text.shape == (DS_DIM,)
    assert torch.equal(text, embeds["158103"])


def test_getitem_missing_shot_gets_zeros(tmp_path):
    cfg = _tiny_cfg()
    _write_ds_cache(tmp_path, "158104", cfg.max_frames + 1, cfg)
    embeds = _text_embeds()             # covers "158103", not "158104"
    ds = FrameCodeDataset(tmp_path, ["158104"], cfg, text_embeds=embeds)
    _, _, _, text = ds[0]
    assert text.shape == (DS_DIM,)
    assert torch.equal(text, torch.zeros(DS_DIM))


def test_getitem_no_text_embeds_gives_zero_length_text(tmp_path):
    cfg = _tiny_cfg()
    _write_ds_cache(tmp_path, "158103", cfg.max_frames + 1, cfg)
    ds = FrameCodeDataset(tmp_path, ["158103"], cfg, text_embeds=None)
    _, _, _, text = ds[0]
    assert text.shape == (0,)


def test_collate_stacks_text_to_batch_dim(tmp_path):
    cfg = _tiny_cfg()
    _write_ds_cache(tmp_path, "158103", cfg.max_frames + 1, cfg)
    embeds = _text_embeds()
    ds = FrameCodeDataset(tmp_path, ["158103"], cfg, text_embeds=embeds)
    codes, act, present, text = _collate_frames([ds[0], ds[1]])
    assert text.shape == (2, DS_DIM)
    assert torch.equal(text[0], text[1])            # same shot, same window -> same embedding


def test_collate_stacks_text_zero_width_with_no_embeds(tmp_path):
    cfg = _tiny_cfg()
    _write_ds_cache(tmp_path, "158103", cfg.max_frames + 1, cfg)
    ds = FrameCodeDataset(tmp_path, ["158103"], cfg, text_embeds=None)
    codes, act, present, text = _collate_frames([ds[0], ds[1]])
    assert text.shape == (2, 0)


def test_train_and_val_style_constructions_both_accept_text_embeds_kwarg(tmp_path):
    """Mirrors train()'s two FrameCodeDataset constructions (train split, val split)."""
    cfg = _tiny_cfg()
    _write_ds_cache(tmp_path, "158103", cfg.max_frames + 1, cfg)
    _write_ds_cache(tmp_path, "158104", cfg.max_frames + 1, cfg)
    embeds = _text_embeds()
    train_ds = FrameCodeDataset(tmp_path, ["158103"], cfg, presence=None, text_embeds=embeds)
    val_ds = FrameCodeDataset(tmp_path, ["158104"], cfg, presence=None, text_embeds=embeds)
    assert len(train_ds) > 0 and len(val_ds) > 0
    # both also work with text_embeds=None (the no-flag default path)
    train_ds0 = FrameCodeDataset(tmp_path, ["158103"], cfg, presence=None, text_embeds=None)
    val_ds0 = FrameCodeDataset(tmp_path, ["158104"], cfg, presence=None, text_embeds=None)
    assert len(train_ds0) > 0 and len(val_ds0) > 0


# --------------------------------------------------------------------------------------------- #
# train()'s --text_embed_path / --text_embed_dim both-or-neither validation: a real 4-case
# truth table calling the actual train() entry point (not a local reimplementation of the
# condition — that is exactly the class of bug a unit test on the boolean alone would miss:
# the original `(text_embed_path is None) != (text_embed_dim > 0)` was inverted in ALL FOUR
# cases and no test caught it because nothing exercised train() itself).
# --------------------------------------------------------------------------------------------- #
from tokamak_foundation_model.ignite.train_dynamics import train as _td_train  # noqa: E402

_TRUTH_TABLE_SHOTS = [190001, 190002, 190003]


def _write_truth_table_cache(cache_dir, shots=_TRUTH_TABLE_SHOTS, n_frames=5,
                             actuator_dim=70):
    """A cache readable by cache_modality_specs (train() calls it BEFORE the flag validation,
    so even the bad-combo cases need a real, loadable cache): one FROZEN_MODALITIES name
    ('mse') is enough — cache_modality_specs derives n_tok/vocab from whatever is present."""
    for s in shots:
        codes = {"mse": torch.randint(0, 50, (n_frames, 4), dtype=torch.int16)}
        act = torch.randn(n_frames, actuator_dim, dtype=torch.float16)
        torch.save({"codes": codes, "actuators": act, "n_frames": n_frames},
                   Path(cache_dir) / f"{s}.pt")


def _write_truth_table_text_h5(path, shots=_TRUTH_TABLE_SHOTS, dim=4):
    rng = np.random.default_rng(1)
    vecs = rng.standard_normal((len(shots), dim)).astype(np.float16)
    with h5py.File(path, "w") as f:
        f.attrs["embed_dim"] = dim
        f.create_dataset("shots", data=np.array(shots, dtype=np.int64))
        f.create_dataset("input", data=vecs)
        f.create_dataset("total", data=vecs)


def _train_kwargs(cache_dir, out_dir, **text_kw):
    """The smallest train() invocation that reaches the validation (and, for the good
    combos, actually completes): tiny model, 1 optimizer step, no worker processes."""
    kw = dict(cache_dir=str(cache_dir), out_dir=str(out_dir), steps=1, batch_size=1,
             d_model=32, depth=1, n_heads=2, k0_seed=2, n_predict=2, num_workers=0,
             val_n=1, ckpt_every=1, val_windows=1, log=lambda *a, **k: None)
    kw.update(text_kw)
    return kw


def test_text_flags_bad_combo_path_only_raises_together(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_truth_table_cache(cache_dir)
    kw = _train_kwargs(cache_dir, tmp_path / "out",
                       text_embed_path=str(tmp_path / "unused.h5"), text_embed_dim=0)
    with pytest.raises(SystemExit) as exc:
        _td_train(**kw)
    assert "TOGETHER" in str(exc.value)


def test_text_flags_bad_combo_dim_only_raises_together(tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_truth_table_cache(cache_dir)
    kw = _train_kwargs(cache_dir, tmp_path / "out", text_embed_path=None, text_embed_dim=4)
    with pytest.raises(SystemExit) as exc:
        _td_train(**kw)
    assert "TOGETHER" in str(exc.value)


def test_text_flags_neither_given_does_not_raise_and_trains(tmp_path):
    """The production default (no text flags): must NOT hit the TOGETHER guard, and a plain
    no-flag training run must actually run to completion."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_truth_table_cache(cache_dir)
    kw = _train_kwargs(cache_dir, tmp_path / "out", text_embed_path=None, text_embed_dim=0)
    try:
        step = _td_train(**kw)
    except SystemExit as e:
        assert "TOGETHER" not in str(e), f"no-flag training run hit the TOGETHER guard: {e}"
        raise
    assert step == 1


def test_text_flags_both_given_does_not_raise_and_trains(tmp_path):
    """The valid enabled case: a real embeddings H5 + matching dim must NOT hit the TOGETHER
    guard, and training with text conditioning enabled must actually run to completion."""
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    _write_truth_table_cache(cache_dir)
    h5 = tmp_path / "text_embeddings.h5"
    _write_truth_table_text_h5(h5, dim=4)
    kw = _train_kwargs(cache_dir, tmp_path / "out",
                       text_embed_path=str(h5), text_embed_dim=4, text_dropout_p=0.1)
    try:
        step = _td_train(**kw)
    except SystemExit as e:
        assert "TOGETHER" not in str(e), f"both-given training run hit the TOGETHER guard: {e}"
        raise
    assert step == 1
