"""`SHOT_DESIGN_IGNITE_GENERATION` swaps the pinned `model:` block for another generation's.

Stellar holds the v2 Hub snapshot and none of v4's proj-shared sources, so its wrappers export
the variable; unset, or set to the pinned generation, nothing changes.
"""

from __future__ import annotations

import pytest

from shot_design.config import load_yaml
from shot_design.shotdb import ignite


def _cfg(want: str) -> dict:
    # conftest's autouse fixture replaces `ignite.model_cfg` with a frozen dict, so the
    # selection itself is exercised through the helper `model_cfg` delegates to.
    return ignite._select_model_cfg(load_yaml("ignite_modalities.yaml"), want)


def test_unset_returns_the_pinned_block():
    assert _cfg("")["generation"] == "v4"


def test_pinned_name_is_a_no_op():
    assert _cfg("v4")["generation"] == "v4"


def test_v2_block_is_complete_and_consistent():
    cfg = _cfg("v2")
    pinned = load_yaml("ignite_modalities.yaml")["model"]
    assert cfg["generation"] == "v2" and cfg["local_name"] == "IGNITE"
    for key in (
        "dynamics_file",
        "t0_start_s",
        "window_ms",
        "production_vocabs",
        "families",
        "n_tok",
        "frame_tokens",
    ):
        assert key in cfg, key
    assert set(cfg["families"]) == set(cfg["n_tok"]) == set(cfg["production_vocabs"])
    assert "mirnov" not in cfg["families"] and len(cfg["families"]) == 14
    assert sum(cfg["n_tok"].values()) == cfg["frame_tokens"] == 1017
    assert set(cfg["families"]) < set(pinned["families"])
    assert cfg["t0_start_s"] == 0.0


def test_unknown_generation_is_refused():
    with pytest.raises(KeyError, match="model_generations"):
        _cfg("v9")
