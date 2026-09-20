"""Adding to a legacy database must preserve nullable integer blurb provenance."""

import json

import pandas as pd

from shot_design.shotdb import build

from .conftest import force_ollama_provider
from .test_build_store import stub_embeddings  # noqa: F401


def test_add_to_legacy_database_keeps_unknown_versions_nullable_and_new_versions_integer(
    paths, staged_shot_a, staged_shot_b, text_fixtures, stub_embeddings,  # noqa: F811
):
    # build.build/build.add each build their own LLMClient from the real llm.yaml
    # (provider: agy) when no client is passed; force ollama so a missing endpoint
    # means "no model" instead of a real agy CLI call (see test_blurb.py's docstring).
    cfg = build.load_build_cfg()
    with force_ollama_provider():
        build.build([staged_shot_a], paths, cfg, workers=1, encode=False)
    table = paths.db_dir / "shots.parquet"
    legacy = pd.read_parquet(table).drop(columns=["blurb_model", "blurb_prompt_version"])
    legacy.to_parquet(table)
    with force_ollama_provider():
        build.add([staged_shot_b], paths, cfg)
    updated = pd.read_parquet(table)
    assert pd.api.types.is_integer_dtype(updated["blurb_prompt_version"])
    assert pd.isna(updated.loc[staged_shot_a, "blurb_prompt_version"])
    assert pd.isna(updated.loc[staged_shot_a, "blurb_model"])
    assert updated.loc[staged_shot_b, "blurb_prompt_version"] == 6
    assert updated.loc[staged_shot_b, "blurb_model"] == "gemini-3.8-flash-low"
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["blurbs"] == {
        "llm": 0,
        "template": 2,
        "human": 0,
        "model": "gemini-3.8-flash-low",
        "prompt_version": 6,
        "prompt_versions": {"6": 1},
    }
