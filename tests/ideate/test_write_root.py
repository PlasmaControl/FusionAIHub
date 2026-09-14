"""Every writing command names the data root it resolved, and where that came from, before it
writes.

The 2026-09-14 incident was a `build` into a root nobody printed: pixi's `[activation.env]`
overrode the exported `IDEATE_DATA_ROOT` and the production database was replaced by a one-shot
one. One line on stderr, emitted before the first write, is what makes that visible at the time
rather than afterwards.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from ideate import cli, config
from ideate.design import seed
from ideate.shotdb import text

from .conftest import write_corpus_group
from .test_select import select_argv, selection_inputs  # noqa: F401

# --------------------------------------------------------------- where the root came from


def test_origin_names_the_env_variable_when_it_is_set(paths, monkeypatch):
    """`IDEATE_DATA_ROOT` wins over the paths file, so it is what the line must name."""
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(paths.data_root))
    assert config.data_root_origin() == "IDEATE_DATA_ROOT env"


def test_origin_names_the_paths_file_when_it_alone_is_set(paths, tmp_path):
    assert config.data_root_origin() == f"IDEATE_PATHS={tmp_path / 'paths.yaml'}"


def test_origin_names_the_repo_default_when_neither_is_set(paths, monkeypatch):
    monkeypatch.delenv("IDEATE_PATHS")
    assert config.data_root_origin() == "configs/ideate/paths.yaml default"


# ------------------------------------------------------------- the line the commands print


@pytest.fixture
def writing_inputs(paths, staged_shot_a, text_fixtures, monkeypatch):
    monkeypatch.setattr(
        text, "embed_texts", lambda texts: np.zeros((len(texts), 384), np.float32)
    )
    monkeypatch.setenv("LABELMAKER_ROOT", str(paths.data_root / "labelmaker"))
    monkeypatch.setenv("IDEATE_CORPUS", str(paths.foundation_model_processed_dir))
    return paths


def watch_first_write(monkeypatch, capsys, root):
    """Capture stderr as it stood when a command first created a directory under `root`.

    Every writer in scope reaches its output through a `mkdir(parents=True, exist_ok=True)` --
    `build` on the root itself and on the `.tmp` publish directory, `add` on the same, `labels
    join` and `corpus scan` on their destination's parent, `encode` on `runs/encode` -- and
    `Path.mkdir` calls `os.mkdir` even when the directory is already there. So the first such
    call is at or before the first byte written under the root, and what stderr held at that
    moment is everything the command had said before it wrote anything.
    """
    seen: list[list[str]] = []
    real = os.mkdir

    def mkdir(path, *args, **kwargs):
        if not seen and str(path).startswith(str(root)):
            seen.append(capsys.readouterr().err.splitlines()[:1])
        return real(path, *args, **kwargs)

    monkeypatch.setattr(os, "mkdir", mkdir)
    return seen


def test_build_names_the_env_root_before_writing(writing_inputs, monkeypatch, capsys):
    paths = writing_inputs
    monkeypatch.setenv("IDEATE_DATA_ROOT", str(paths.data_root))
    expected = (
        f"ideate build: data root {paths.data_root} (IDEATE_DATA_ROOT env) -> db {paths.db_dir}"
    )
    capsys.readouterr()
    seen = watch_first_write(monkeypatch, capsys, paths.data_root)
    assert cli.main(["build", "--shots", "900001", "--workers", "1", "--no-encode"]) == 0
    assert seen == [[expected]]
    assert "data root" not in capsys.readouterr().err  # said once, not once per phase


@pytest.mark.parametrize("command", ["add", "labels join", "encode", "corpus scan"])
def test_writers_name_the_root_before_writing(
    writing_inputs, tmp_path, monkeypatch, capsys, command
):
    paths = writing_inputs
    destination = f"db {paths.db_dir}"
    if command == "add":
        assert cli.main(["build", "--shots", "900001", "--workers", "1", "--no-encode"]) == 0
        argv = ["add", "900001", "--workers", "1"]
    elif command == "labels join":
        lm = paths.data_root / "labelmaker"
        (lm / "labels").mkdir(parents=True)
        argv = ["labels", "join", "--shots", "900001", "--labelmaker-root", str(lm), "--no-text"]
    elif command == "encode":

        def encode_many(shots, *, out_dir, **kwargs):
            out_dir.mkdir(parents=True)
            return {
                "n_encoded": 1, "n_requested": 1, "n_skipped": 0, "failed": {},
                "elapsed_s": 0.1, "s_per_shot_mean": None,
            }

        monkeypatch.setattr(seed, "encode_many", encode_many)
        argv = ["encode", "--shots", "900001"]
        destination = f"frame codes {paths.data_root / 'frame_codes'}"
    else:
        corpus = paths.foundation_model_processed_dir
        corpus.mkdir()
        write_corpus_group(corpus / "900001_processed.h5", "ip", np.arange(4), np.ones((1, 4)))
        argv = ["corpus", "scan", "--workers", "1"]
        destination = f"census {paths.db_dir / 'corpus_coverage.parquet'}"
    expected = (
        f"ideate {command}: data root {paths.data_root} "
        f"(IDEATE_PATHS={tmp_path / 'paths.yaml'}) -> {destination}"
    )
    capsys.readouterr()
    seen = watch_first_write(monkeypatch, capsys, paths.data_root)
    assert cli.main(argv) == 0
    assert seen == [[expected]]
    assert "data root" not in capsys.readouterr().err


def test_select_names_the_root_even_when_every_path_is_explicit(
    writing_inputs, selection_inputs, tmp_path, capsys  # noqa: F811
):
    """`corpus select` may be given every input path and still resolves a root: the shot list it
    writes is what the next `build` will be pointed at."""
    txt, parquet = selection_inputs
    out = writing_inputs.data_root / "selection.yaml"
    expected = (
        f"ideate corpus select: data root {writing_inputs.data_root} "
        f"(IDEATE_PATHS={tmp_path / 'paths.yaml'}) -> shot list {out}"
    )
    capsys.readouterr()
    assert cli.main(select_argv(txt, parquet, tmp_path, **{"--out": str(out)})) == 0
    assert capsys.readouterr().err.splitlines()[0] == expected
    assert out.exists()


def test_a_command_with_no_resolvable_root_still_names_its_destination(
    writing_inputs, selection_inputs, tmp_path, monkeypatch, capsys  # noqa: F811
):
    """An unusable paths config must not make the announcement the thing that fails: the line
    degrades to the destination alone and a fully-explicit run works as it did before."""
    txt, parquet = selection_inputs
    out = tmp_path / "selection.yaml"
    monkeypatch.setenv("IDEATE_DATA_ROOT", "")  # load_paths refuses an empty root
    capsys.readouterr()
    assert cli.main(select_argv(txt, parquet, tmp_path, **{"--out": str(out)})) == 0
    assert capsys.readouterr().err.splitlines()[0] == (
        f"ideate corpus select: no data root resolved -> shot list {out}"
    )
