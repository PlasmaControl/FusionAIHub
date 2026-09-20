"""`shot_design evalsets prompt <name> <slug>`: print one named prompt's text from a
`configs/shot_design/evalsets/<name>.yaml` prompt library, for `demo_frontier.sh` to
redirect into `prompt.md`. Against the real, committed `frontier_demo_prompts.yaml` --
no `paths` fixture needed, since `config.CONFIG_DIR` is fixed to the repo's own configs.
"""

from __future__ import annotations

import pytest

from shot_design import cli, config

SLUGS = ["tearing_eccd", "elm_rmp", "ae_nbi"]


@pytest.mark.parametrize("slug", SLUGS)
def test_prints_each_frontier_demo_prompt(slug, capsys):
    rc = cli.main(["evalsets", "prompt", "frontier_demo_prompts", slug])

    assert rc == 0
    assert capsys.readouterr().out.strip()


def test_unknown_slug_lists_the_known_ones_on_stderr(capsys):
    rc = cli.main(["evalsets", "prompt", "frontier_demo_prompts", "nope"])

    assert rc == 2
    err = capsys.readouterr().err
    assert "nope" in err
    for slug in SLUGS:
        assert slug in err


def test_unknown_evalset_name_fails_cleanly(capsys):
    rc = cli.main(["evalsets", "prompt", "does_not_exist", "x"])

    assert rc == 1
    assert "does_not_exist" in capsys.readouterr().err


def test_malformed_evalset_yaml_fails_cleanly_instead_of_raising(
    monkeypatch, tmp_path, capsys
):
    """A hand-edited evalset with broken YAML must be reported like any other bad
    evalset (`OSError` path), not crash the CLI with an uncaught `yaml.YAMLError`."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    evalsets = tmp_path / "evalsets"
    evalsets.mkdir()
    (evalsets / "broken.yaml").write_text("prompts: [unterminated\n")

    rc = cli.main(["evalsets", "prompt", "broken", "x"])

    assert rc == 1
    assert "broken" in capsys.readouterr().err


def test_evalset_entry_without_text_prints_a_message_instead_of_a_traceback(
    monkeypatch, tmp_path, capsys
):
    """A hand-edited entry that is a mapping without a `text` key raised `KeyError`
    from `entry["text"]` instead of a readable message."""
    monkeypatch.setattr(config, "CONFIG_DIR", tmp_path)
    evalsets = tmp_path / "evalsets"
    evalsets.mkdir()
    (evalsets / "notext.yaml").write_text("prompts:\n  x:\n    goal: no text here\n")

    rc = cli.main(["evalsets", "prompt", "notext", "x"])

    assert rc == 1
    assert "prompt has no text" in capsys.readouterr().err
