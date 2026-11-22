"""Task specs, split lists and published scores (docs/models/ignite-mast-design.md)."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks


def test_standardize_round_trip_without_tokamark():
    s = tasks.SignalSpec("summary-ip", 0.00025, (1,), mean=3.0, std=2.0)
    x = np.array([1.0, 3.0, 7.0])
    np.testing.assert_allclose(s.standardize(x), [-1.0, 0.0, 2.0])
    np.testing.assert_allclose(s.unstandardize(s.standardize(x)), x)


def test_published_scores():
    assert tasks.PUBLISHED["task_4-5"]["mean"] == 0.653
    assert tasks.PUBLISHED["task_4-4"]["persistence"] == 0.718
    assert tasks.PUBLISHED["task_1-1"]["persistence"] is None
    gentoka = [tasks.PUBLISHED[t]["gentoka"] for t in tasks.ALL_TASKS]
    assert abs(sum(gentoka) / 14 - 0.2033) < 5e-4
    assert tasks.best_published("task_4-1") == 0.2122 and tasks.best_published("task_1-1") == 0.0941


def test_task_1_1_lengths():
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_1-1")
    assert (t.input_length, t.output_length, t.delta, t.stride) == (0.005, 0.005, -0.005, 0.001)
    assert t.markovian
    assert t.actuators == ()


def test_task_4_5_outputs_are_spectrograms():
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_4-5")
    assert {s.name for s in t.outputs} == set(tasks.SPECTROGRAM_SIGNALS)
    assert all(s.values_shape == (3, 256) and t.n_out(s) == 98 for s in t.outputs)
    assert not t.markovian and t.input_length == 0.15


def test_n_in_for_a_non_markovian_task_follows_the_history():
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_4-1")
    s = t.inputs[0]
    assert t.n_in(s) == round(0.15 / s.dt) == t.steps["input"][s.name]
    assert t.n_in(s, history_s=0.05) == round(0.05 / s.dt)


def test_union_of_signals_is_39():
    pytest.importorskip("tokamark")
    names = {s.name for n in tasks.ALL_TASKS for t in [tasks.load_task(n)]
             for s in t.inputs + t.actuators + t.outputs}
    assert len(names) == 39


def test_split_shots_are_disjoint_and_from_the_split_file():
    pytest.importorskip("tokamark")
    tr, va, te = (tasks.split_shots(s) for s in ("train", "val", "test"))
    assert (len(tr), len(va), len(te)) == (8963, 1115, 1110)
    assert not (set(tr) & set(va)) and not (set(tr) & set(te)) and not (set(va) & set(te))
    sel, rep = tasks.split_shots("val_select"), tasks.split_shots("val_report")
    assert sorted(sel + rep) == va and (len(sel), len(rep)) == (558, 557)
    with pytest.raises(ValueError):
        tasks.split_shots("all")


def test_toy_task_steps_cover_its_windows(toy_task):
    t = toy_task
    ip, flux = t.inputs
    assert (t.n_in(ip), t.n_in(flux), t.n_out(t.outputs[0]), t.n_act(t.actuators[0])) == (20, 25, 100, 120)
    assert t.n_in(ip) * ip.dt == pytest.approx(t.input_length)
    assert t.n_out(ip) * ip.dt == pytest.approx(t.output_length)
    assert t.n_act(t.actuators[0]) * t.actuators[0].dt == pytest.approx(t.input_length + t.delta + t.output_length)


def test_steps_of_a_signal_that_is_input_and_output_are_right():
    """TokaMark's metadata overwrites the input length of such a signal with the output length."""
    pytest.importorskip("tokamark")
    t = tasks.load_task("task_3-1")  # markovian, 5 ms in, 50 ms out, dt 5 ms
    s = next(x for x in t.inputs if x.name == "thomson_scattering-t_e")
    assert (t.n_in(s), t.n_out(s)) == (1, 10)
    t = tasks.load_task("task_2-1")  # markovian, 5 ms in, 25 ms out, dt 0.25 ms
    s = next(x for x in t.inputs if x.name == "summary-ip")
    assert (t.n_in(s), t.n_out(s)) == (20, 100)
    t = tasks.load_task("task_3-2")
    s = next(x for x in t.inputs if x.name == "spectrometer_visible-filter_spectrometer_dalpha_voltage")
    assert t.n_in(s) == 250
    t = tasks.load_task("task_4-4")  # non-markovian: 150 ms of history, 100 ms out
    s = next(x for x in t.inputs if x.name == "summary-ip")
    assert (t.n_in(s), t.n_out(s)) == (600, 400)
    assert t.steps["input"]["summary-ip"] == 600


def test_every_count_of_every_task_is_a_positive_integer():
    pytest.importorskip("tokamark")
    for name in tasks.ALL_TASKS:
        t = tasks.load_task(name)
        for role, specs in (("input", t.inputs), ("actuator", t.actuators), ("output", t.outputs)):
            assert set(t.steps[role]) == {s.name for s in specs}, (name, role)
            for s in specs:
                n = t.steps[role][s.name]
                assert isinstance(n, int) and n > 0, (name, role, s.name, n)


def test_n_act_with_a_shorter_history_keeps_the_future_part(toy_task):
    import dataclasses

    t = dataclasses.replace(toy_task, markovian=False)
    coil = t.actuators[0]
    assert t.n_act(coil) == 120 and t.n_act(coil, history_s=0.0025) == 110
    assert toy_task.n_act(coil, history_s=0.0025) == 120  # markovian: history_s is ignored
