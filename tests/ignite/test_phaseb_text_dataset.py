# dataset-integration tests for FrameCodeDataset are appended by a later task.
"""Consolidated text-embedding H5 loader: load_text_embeddings + lookup."""

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
