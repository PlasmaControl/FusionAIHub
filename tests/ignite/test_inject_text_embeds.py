"""Tests for scripts/data_preparation/inject_text_embeds.py.

Builds a tiny consolidated H5 (2 shots, embedding dim 8) and fake per-shot
`{shot}_processed.h5` files in tmp_path, then exercises inject/dry_run/overwrite/verify.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT_PATH = _REPO_ROOT / "scripts" / "data_preparation" / "inject_text_embeds.py"

_spec = importlib.util.spec_from_file_location("inject_text_embeds", _SCRIPT_PATH)
inject_text_embeds = importlib.util.module_from_spec(_spec)
sys.modules["inject_text_embeds"] = inject_text_embeds
_spec.loader.exec_module(inject_text_embeds)

DIM = 8
SHOTS = [111111, 222222, 333333]  # 333333 has no shot file


def _make_consolidated(path: Path, *, input_vals=None, complete=True):
    n = len(SHOTS)
    rng = np.random.default_rng(0)
    input_arr = (input_vals if input_vals is not None else rng.standard_normal((n, DIM))).astype(np.float16)
    total_arr = rng.standard_normal((n, DIM)).astype(np.float16)
    with h5py.File(path, "w") as f:
        f.create_dataset("shots", data=np.array(SHOTS, dtype=np.int64))
        f.create_dataset(
            "run_id",
            data=np.array([str(20240000 + i) for i in range(n)], dtype=object),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        f.create_dataset("input", data=input_arr)
        f.create_dataset("total", data=total_arr)
        f.create_dataset("n_tok_input", data=np.array([10, 20, 30], dtype=np.int32))
        f.create_dataset("n_tok_total", data=np.array([100, 200, 300], dtype=np.int32))
        f.create_dataset("truncated_input", data=np.array([False, True, False], dtype=bool))
        f.create_dataset("truncated_total", data=np.array([True, False, False], dtype=bool))
        f.attrs["model_id"] = "Qwen/Qwen3-Embedding-8B"
        f.attrs["max_length"] = 32768
        f.attrs["pooling"] = "last_token"
        f.attrs["normalized"] = True
        f.attrs["embed_dim"] = DIM
        f.attrs["input_rule"] = "header + general-session-info/metadata + planned(mini-proposal)"
        f.attrs["git_sha"] = "abc123"
        f.attrs["created_utc"] = "2026-08-28T00:00:00+00:00"
        if complete:
            f.attrs["complete"] = True
    return input_arr, total_arr


def _make_shot_file(path: Path, shot: int):
    with h5py.File(path, "w") as f:
        grp = f.create_group("beam_voltage")
        grp.create_dataset("xdata", data=np.arange(10, dtype=np.float64))
        grp.create_dataset("ydata", data=np.arange(10, dtype=np.float64) * 2)


@pytest.fixture
def env(tmp_path):
    embeds_path = tmp_path / "text_embeddings.h5"
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir()
    _make_consolidated(embeds_path)
    _make_shot_file(shot_dir / f"{SHOTS[0]}_processed.h5", SHOTS[0])
    _make_shot_file(shot_dir / f"{SHOTS[1]}_processed.h5", SHOTS[1])
    # SHOTS[2] deliberately has no file.
    return embeds_path, shot_dir


def _keys_snapshot(path):
    with h5py.File(path, "r") as f:
        return set(f.keys())


def test_inject_creates_group_with_data_and_attrs(env):
    embeds_path, shot_dir = env
    shot_path = shot_dir / f"{SHOTS[0]}_processed.h5"
    before_keys = _keys_snapshot(shot_path)

    args = inject_text_embeds._parse_args(["--embeds", str(embeds_path), "--shot_dir", str(shot_dir)])
    counts = inject_text_embeds.run_inject(args)

    assert counts == {"injected": 2, "skipped_existing": 0, "no_shot_file": 1, "error": 0}

    with h5py.File(embeds_path, "r") as ef:
        expected_input = ef["input"][0]
        expected_total = ef["total"][0]

    with h5py.File(shot_path, "r") as sf:
        after_keys = set(sf.keys())
        assert before_keys < after_keys
        assert after_keys - before_keys == {"text_embed"}
        # other groups untouched
        assert "beam_voltage" in sf
        np.testing.assert_array_equal(sf["beam_voltage/xdata"][:], np.arange(10, dtype=np.float64))

        grp = sf["text_embed"]
        assert grp["input"].dtype == np.float16
        assert grp["total"].dtype == np.float16
        np.testing.assert_array_equal(grp["input"][:], expected_input)
        np.testing.assert_array_equal(grp["total"][:], expected_total)

        assert grp.attrs["model_id"] == "Qwen/Qwen3-Embedding-8B"
        assert grp.attrs["embed_dim"] == DIM
        assert grp.attrs["pooling"] == "last_token"
        assert bool(grp.attrs["normalized"]) is True
        assert grp.attrs["input_rule"] == "header + general-session-info/metadata + planned(mini-proposal)"
        assert grp.attrs["max_length"] == 32768
        assert grp.attrs["n_tok_input"] == 10
        assert grp.attrs["n_tok_total"] == 100
        assert bool(grp.attrs["truncated_input"]) is False
        assert bool(grp.attrs["truncated_total"]) is True
        assert grp.attrs["source"] == str(embeds_path.resolve())
        assert grp.attrs["source_created_utc"] == "2026-08-28T00:00:00+00:00"
        assert "injected_utc" in grp.attrs


def test_missing_shot_file_counted_and_exit_zero(env, monkeypatch):
    embeds_path, shot_dir = env
    argv = ["inject_text_embeds.py", "--embeds", str(embeds_path), "--shot_dir", str(shot_dir)]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as exc:
        inject_text_embeds.main()
    assert exc.value.code == 0


def test_rerun_without_overwrite_skips_and_leaves_data_unchanged(env):
    embeds_path, shot_dir = env
    args = inject_text_embeds._parse_args(["--embeds", str(embeds_path), "--shot_dir", str(shot_dir)])
    inject_text_embeds.run_inject(args)

    shot_path = shot_dir / f"{SHOTS[0]}_processed.h5"
    with h5py.File(shot_path, "r") as sf:
        data_before = sf["text_embed/input"][:].copy()
        injected_utc_before = sf["text_embed"].attrs["injected_utc"]
    mtime_before = shot_path.stat().st_mtime_ns

    counts = inject_text_embeds.run_inject(args)
    assert counts == {"injected": 0, "skipped_existing": 2, "no_shot_file": 1, "error": 0}

    with h5py.File(shot_path, "r") as sf:
        np.testing.assert_array_equal(sf["text_embed/input"][:], data_before)
        assert sf["text_embed"].attrs["injected_utc"] == injected_utc_before
    assert shot_path.stat().st_mtime_ns == mtime_before


def test_overwrite_updates_data_after_source_mutation(env):
    embeds_path, shot_dir = env
    args = inject_text_embeds._parse_args(["--embeds", str(embeds_path), "--shot_dir", str(shot_dir)])
    inject_text_embeds.run_inject(args)

    # mutate the source consolidated H5's input row for shot 0
    new_row = np.full(DIM, 9.0, dtype=np.float16)
    with h5py.File(embeds_path, "r+") as ef:
        ef["input"][0] = new_row

    overwrite_args = inject_text_embeds._parse_args(
        ["--embeds", str(embeds_path), "--shot_dir", str(shot_dir), "--overwrite"]
    )
    counts = inject_text_embeds.run_inject(overwrite_args)
    assert counts == {"injected": 2, "skipped_existing": 0, "no_shot_file": 1, "error": 0}

    shot_path = shot_dir / f"{SHOTS[0]}_processed.h5"
    with h5py.File(shot_path, "r") as sf:
        np.testing.assert_array_equal(sf["text_embed/input"][:], new_row)


def test_dry_run_writes_nothing(env):
    embeds_path, shot_dir = env
    shot_path = shot_dir / f"{SHOTS[0]}_processed.h5"
    before_keys = _keys_snapshot(shot_path)
    mtime_before = shot_path.stat().st_mtime_ns

    args = inject_text_embeds._parse_args(
        ["--embeds", str(embeds_path), "--shot_dir", str(shot_dir), "--dry_run"]
    )
    inject_text_embeds.run_inject(args)

    assert _keys_snapshot(shot_path) == before_keys
    assert shot_path.stat().st_mtime_ns == mtime_before


def test_incomplete_source_raises_system_exit(tmp_path):
    embeds_path = tmp_path / "text_embeddings.h5"
    shot_dir = tmp_path / "shots"
    shot_dir.mkdir()
    _make_consolidated(embeds_path, complete=False)

    args = inject_text_embeds._parse_args(["--embeds", str(embeds_path), "--shot_dir", str(shot_dir)])
    with pytest.raises(SystemExit):
        inject_text_embeds.run_inject(args)


def test_verify_passes_after_injection(env):
    embeds_path, shot_dir = env
    args = inject_text_embeds._parse_args(["--embeds", str(embeds_path), "--shot_dir", str(shot_dir)])
    inject_text_embeds.run_inject(args)

    verify_args = inject_text_embeds._parse_args(
        ["--embeds", str(embeds_path), "--shot_dir", str(shot_dir), "--verify"]
    )
    ok = inject_text_embeds.run_verify(verify_args)
    assert ok is True


def test_limit_restricts_shots_processed(env):
    embeds_path, shot_dir = env
    args = inject_text_embeds._parse_args(
        ["--embeds", str(embeds_path), "--shot_dir", str(shot_dir), "--limit", "1"]
    )
    counts = inject_text_embeds.run_inject(args)
    # only SHOTS[0] considered
    assert counts == {"injected": 1, "skipped_existing": 0, "no_shot_file": 0, "error": 0}
