"""`shot_design evalsets prompt <name> <slug>`: print one named prompt's text from a
`configs/shot_design/evalsets/<name>.yaml` prompt library, for `demo_frontier.sh` to
redirect into `prompt.md`. Against the real, committed `frontier_demo_prompts.yaml` --
no `paths` fixture needed, since `config.CONFIG_DIR` is fixed to the repo's own configs.
"""

from __future__ import annotations

import pytest

from shot_design import cli

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
