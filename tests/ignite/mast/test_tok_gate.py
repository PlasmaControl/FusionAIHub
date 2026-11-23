"""The stage-1 gate: bin usage, the worst channel, and the verdict of each (task, signal) pair."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import tasks, transform
from tokamak_foundation_model.ignite.mast.tokenizers import assign, codecs, gate, grid

SPEC = tasks.SignalSpec("s", 0.00025, (2,), 0.0, 2.0)
ST = transform.ChannelStats(compress="none", scale=np.ones(2), mean=np.zeros(2), std=np.ones(2))


def shot():
    z = np.zeros((2, 40), np.float32)                                         # 2 channels, 2 frames
    z[1, :20:2] = 1.0                                                         # channel 1 alternates in its first frame
    return z


def test_bin_usage_counts_the_bins_each_value_uses():
    idx = np.array([[[0, 5]], [[1, 5]], [[2, 5]], [[-1, 5]]])                 # 4 examples, 1 token, 2 values
    assert gate.bin_usage(idx, 8).tolist() == [[0.375, 0.125]]
    assert np.isnan(gate.bin_usage(np.full((3, 1, 1), -1), 8)).all()          # no example is no data, not 0


def test_channel_errors_are_per_channel_in_tokamark_units():
    c = codecs.FrameDCT("s", 0.00025, (2,), ST, k=1)
    sq, n = gate.channel_errors(c, SPEC, shot())
    assert n.tolist() == [40, 40] and sq[0] == pytest.approx(0.0, abs=1e-12)
    assert sq[1] == pytest.approx(20 * 0.25 / 4.0)                            # 20 samples 0.5 from the frame mean, std 2


def test_the_cache_side_pass_counts_shots_and_names_the_worst_channel():
    g = grid.GridSet(lo=np.full((2, 1), -1.0), hi=np.full((2, 1), 1.0), width=np.ones((2, 1)), n_bins=8)
    c = codecs.FrameDCT("s", 0.00025, (2,), ST, k=1, grids=g)
    out = gate.cache_side(c, SPEC, [shot(), None, shot()])
    assert (out["n_shots"], out["n_skipped"], out["worst_channel"]) == (2, 1, 1)
    assert out["worst_channel_error"] == pytest.approx(np.sqrt(1.25 / 40)) and out["contrast"] is None
    assert out["bins_used_min"] == pytest.approx(1 / 8)                       # channel 0 is always 0: one bin of 8
    none = gate.cache_side(c, SPEC, [None])
    assert (none["n_shots"], none["n_skipped"], none["worst_channel_error"]) == (0, 1, None)


def test_the_verdicts():
    v = gate.verdict
    kw = dict(values=4, reference_values=1)
    assert v(floor=0.010, ranged=0.011, own_mean=0.076, target=0.0176, **kw) == "pass"
    assert v(floor=0.060, ranged=0.061, own_mean=0.098, target=0.0424, **kw) == (
        "pass, misses the fidelity target (0.0610 against 0.0424)")
    assert v(floor=0.100, ranged=0.100, own_mean=0.098, target=0.0424, **kw) == (
        "blocks: does not beat its own mean (0.1000 against 0.0980)")
    assert v(floor=0.098, ranged=0.098, own_mean=0.098, target=None, **kw).startswith("blocks")   # equal is not better
    assert v(values=1, reference_values=1, floor=0.0, ranged=0.0, own_mean=0.0, target=0.0176) == "pass"
    assert v(values=216, reference_values=24, floor=0.676, ranged=0.680, own_mean=0.900, target=None) == (
        "pass (no fidelity target)")
    assert v(floor=None, ranged=None, own_mean=None, target=0.0176, **kw) == "not scored"
    assert v(floor=0.010, ranged=None, own_mean=0.050, target=0.0176, **kw) == "not scored"
    assert v(floor=0.010, ranged=0.011, own_mean=None, target=0.0176, **kw) == "not scored"


def test_gate_rows_judge_each_task_signal_pair_and_keep_missing_scores_missing(toy_task):
    a = {"summary-ip": assign.Setting("frame_dct", (4,))}
    floors = {("task_toy", "unquantized"): {"summary-ip": 0.010}, ("task_toy", "ranged"): {"summary-ip": 0.011},
              ("task_toy", "nearest"): {"summary-ip": 0.020}, ("task_toy", "ref"): {"summary-ip": 0.076},
              ("task_other", "unquantized"): None}
    side = {"summary-ip": {"n_shots": 200, "n_skipped": 0, "bins_used_min": 0.5, "worst_channel": 0,
                           "worst_channel_error": 0.012, "contrast": None}}
    t = gate.gate_rows({"task_toy": toy_task.outputs, "task_other": toy_task.outputs}, a, floors,
                       {"task_toy": 0.0880, "task_other": 0.3116}, side).set_index("task")
    row = t.loc["task_toy"]
    assert row["target"] == pytest.approx(0.0176) and row["verdict"] == "pass" and row["values_per_frame"] == 4
    assert row["own_mean"] == 0.076 and row["nearest"] == 0.020
    assert t.loc["task_other", "verdict"] == "not scored" and np.isnan(t.loc["task_other", "unquantized"])
    text = gate.to_markdown(t.reset_index())
    assert "not scored" in text and "0.0110" in text and "| 0.0000 |" not in text


def test_a_band_codec_does_not_count_the_positions_a_band_drops_as_one_bin():   # ruling B27
    shape = (1, 256)
    st = transform.ChannelStats(compress="asinh", scale=np.full(shape, 2.0), mean=np.zeros(shape), std=np.ones(shape))
    g = grid.GridSet(lo=np.full((8, 4), -20.0), hi=np.full((8, 4), 20.0), width=np.full((8, 4), 0.5), n_bins=8)
    c = codecs.BandDCT("m", 0.001024, shape, st, [4, 1, 1, 1, 1, 1, 1, 1], grids=g)
    amp = np.linspace(0.2, 8.0, 200)                                          # every frame has another level and tilt
    b = (np.arange(256) % 32 + 0.5) * np.pi / 32                              # the cosines of the first four values
    shape_ = 1.0 + np.cos(b) + np.cos(2 * b) + np.cos(3 * b)
    x = 1.0 + amp[None, None, :] * shape_[None, :, None]                     # (1, 256, 200) physical
    spec = tasks.SignalSpec("m", 0.001024, shape, 0.0, 1.0)
    out = gate.cache_side(c, spec, [transform.to_model(x, st).astype(np.float32)])
    assert out["n_shots"] == 1
    assert out["bins_used_min"] > 1 / 8 + 1e-9                                # the 21 dropped positions are not examples
