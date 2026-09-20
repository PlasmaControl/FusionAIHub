"""The docs state the three things the 2026-09-14 incident turned into code.

A documented escape hatch that has drifted from the code is worse than none: it reads as
authority. So each assertion here is tied to the thing it documents -- the origin labels
`config.data_root_origin` actually returns, the variables `pyproject.toml` actually pins, and
the flag the guard actually names.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

from shot_design import config

REPO = Path(__file__).resolve().parents[2]
DOCS = REPO / "docs" / "shot-design" / "overview.md"
HEADING = "## Scratch databases and the pixi activation env"


def section() -> str:
    body = DOCS.read_text(encoding="utf-8")
    assert HEADING in body, f"{DOCS} has no {HEADING!r} section"
    after = body.split(HEADING, 1)[1]
    return re.split(r"^## ", after, maxsplit=1, flags=re.MULTILINE)[0]


def test_the_section_names_every_variable_the_shot_design_features_pin():
    """`pixi run -e shot-design*` overrides these, whatever the caller exported -- which is what
    replaced the production database with a one-shot one."""
    pinned = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    env = pinned["tool"]["pixi"]["feature"]["shot-design"]["target"]["unix"]["activation"]["env"]
    text = section()
    for name in ("SHOT_DESIGN_DATA_ROOT", "LABELER_ROOT", "SHOT_DESIGN_CORPUS"):
        assert name in env, f"{name} is no longer pinned; the docs section is now wrong"
        assert name in text, f"the docs section does not name {name}"


def test_the_section_gives_both_ways_to_build_a_scratch_database():
    text = section()
    assert ".pixi/envs/shot-design-cpu/bin/python -m shot_design" in text
    assert "SHOT_DESIGN_PATHS=" in text


def test_the_section_quotes_the_origin_labels_the_code_prints(monkeypatch):
    text = section()
    # Not a paraphrase: each label is what `data_root_origin` returns for that precedence.
    monkeypatch.setenv("SHOT_DESIGN_DATA_ROOT", "/tmp/whatever")
    assert config.data_root_origin() in text
    monkeypatch.delenv("SHOT_DESIGN_DATA_ROOT")
    monkeypatch.delenv("SHOT_DESIGN_PATHS", raising=False)
    # The third label names the resolved paths file in full -- an absolute path, and so specific
    # to the checkout -- because `SHOT_DESIGN_CONFIG_DIR` can move it. The docs quote the repo-relative
    # path it is in a plain checkout, which is that label minus the repo root.
    label = config.data_root_origin()
    assert label.endswith(" default"), label
    resolved = Path(label.removesuffix(" default"))
    assert f"{resolved.relative_to(REPO)} default" in text


def test_the_section_says_what_the_publish_guard_refuses_and_how_to_override_it():
    text = section()
    assert "--force" in text
    assert "shot_source" in text and "n_shots" in text
