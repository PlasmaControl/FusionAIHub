"""The design layer converts seconds <-> cache frames through the generation's frame origin.

v2 caches start at 0.0 s, v4 caches at 1.0 s (`model.t0_start_s`). Before `frame_origin_s`
every conversion assumed 0.0, so under v4 a design's seed codes were cut one second later
than the actuators drawn and edited beside them.
"""

from __future__ import annotations

import numpy as np
import pytest

from shot_design.design import actuators as act
from shot_design.design import program, program_reference
from shot_design.shotdb import ignite
from shot_design.simulate import cli as simulate_cli


@pytest.fixture
def origin(monkeypatch):
    def set_origin(t0: float):
        cfg = {**ignite.model_cfg(), "t0_start_s": t0}
        monkeypatch.setattr(ignite, "model_cfg", lambda: cfg)
        return t0

    return set_origin


def test_frame_origin_reads_the_selected_generation(origin):
    origin(1.0)
    assert program_reference.frame_origin_s() == 1.0
    origin(0.0)
    assert program_reference.frame_origin_s() == 0.0


@pytest.mark.parametrize("t0", [0.0, 1.0])
def test_window_is_relative_to_the_cache_origin(origin, t0):
    origin(t0)
    prog = program.DesignProgram(reference_shot=1, start_s=t0 + 1.0, end_s=t0 + 4.0)
    errors: list[str] = []
    assert program._window(prog, 100, errors) == (20, 80)
    assert errors == []


def test_a_v4_design_starting_at_one_second_has_no_seed_frames(origin):
    origin(1.0)
    prog = program.DesignProgram(reference_shot=1, start_s=1.0, end_s=4.0)
    errors: list[str] = []
    assert program._window(prog, 100, errors) == (0, 60)
    assert any("20 earlier reference frames" in e for e in errors)


def test_points_carry_shot_seconds_not_frame_indices(origin):
    origin(1.0)
    pts = program._points(np.array([1.0, 2.0]), np.array([0, 20]))
    assert [p["t_s"] for p in pts] == [1.0, 2.0]


def test_windowed_real_actuators_slices_by_cache_frame(origin):
    origin(1.0)
    prog = program.DesignProgram(reference_shot=1, start_s=2.0, end_s=5.0)
    ref_cache = {"actuators": np.arange(100)[:, None]}
    out = simulate_cli._windowed_real_actuators(ref_cache, prog, {})
    assert out["actuators"][0, 0] == 0 and out["actuators"].shape[0] == 80


def test_read_controls_builds_actuators_at_the_cache_origin(origin, monkeypatch, tmp_path):
    origin(1.0)

    class _Reader:
        def __init__(self, root):
            pass

        def read(self, shot, group, channels=None):
            raise program_reference.Unavailable(group)

        def coverage(self, shot, group):
            raise program_reference.Unavailable(group)

    monkeypatch.setattr(program_reference, "CorpusReader", _Reader)
    program_reference._read_controls.cache_clear()
    source = tmp_path / "1_processed.h5"
    source.write_bytes(b"")
    identity = program_reference.file_identity(source)
    controls, available = program_reference._read_controls(1, identity, 100, 1.0)
    assert controls.t0_s == 1.0
    assert controls.frame_times[0] == pytest.approx(1.0)
    assert available.shape == (act.N_CHANNELS, 100) and not available.any()


def test_raw_reference_counts_frames_from_the_cache_origin(origin, monkeypatch, tmp_path):
    origin(1.0)

    class _Reader:
        def __init__(self, root):
            pass

        def read(self, shot, group, channels=None):
            raise program_reference.Unavailable(group)

        def coverage(self, shot, group):
            return 0.0, 5000.0  # ms: the record runs to 5.0 s

    monkeypatch.setattr(program_reference, "CorpusReader", _Reader)
    program_reference._read_controls.cache_clear()
    source = tmp_path / "1_processed.h5"
    source.write_bytes(b"")
    ref = program_reference._raw_reference(1, program_reference.file_identity(source), "e", True)
    assert ref.controls.n_frames == 80  # (5.0 - 1.0) / 0.05, not 100
    assert ref.controls.t0_s == 1.0


def test_frame_origin_requires_the_model_origin(monkeypatch):
    cfg = {k: v for k, v in ignite.model_cfg().items() if k != "t0_start_s"}
    monkeypatch.setattr(ignite, "model_cfg", lambda: cfg)
    with pytest.raises(KeyError, match="t0_start_s"):
        program_reference.frame_origin_s()
