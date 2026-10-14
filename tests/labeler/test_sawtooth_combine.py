"""saw-ours v2: the derivative edge score, the AND combination and the q gate."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts/labeler"))
import sawtooth_benchmark as benchmark

from labeler.sawtooth import combine


def synthetic_signal(seed=3):
    rng = np.random.default_rng(seed)
    t = np.arange(20000) * 1e-4
    y = np.full((48, len(t)), np.nan, dtype=np.float32)
    core = 2.0 + 0.002 * rng.standard_normal(len(t))
    for crash in (0.3, 0.38, 0.46, 1.4):
        core[t >= crash] -= 0.05 * (1 if crash < 1 else 3)
    y[20] = core
    observable = np.ones(len(t), bool)
    observable[(t > 0.9) & (t < 1.0)] = False
    return {
        "t": t,
        "y": y,
        "central_channel": np.array(20),
        "observable": observable,
        "assessed": observable.copy(),
    }


def test_edge_score_reproduces_the_derivative_presence_at_every_z():
    signal = synthetic_signal()
    peaks, amplitudes = benchmark.derivative_candidates(signal)
    t = signal["t"]
    keep = benchmark.masks_at(signal, t[peaks])[0]
    edge = combine.edge_score(t, t[peaks][keep], amplitudes[keep])
    for z in (2, 5, 10, 50, 1000):
        expected = benchmark.derivative_prediction(signal, z)["presence"]
        np.testing.assert_array_equal(edge >= z, expected >= 0.5)


def test_edge_score_reach_is_closed_like_the_derivative_rule():
    t = np.array([0.0, 0.3749, 0.375, 0.5, 0.625, 0.6251, 1.0])
    score = combine.edge_score(t, [0.5, 0.9], [20.0, 7.0])
    assert score.tolist() == [0, 0, 20, 20, 20, 0, 7]
    assert combine.edge_score(t, [], []).tolist() == [0] * len(t)


def test_derivative_trace_peaks_equal_the_candidate_amplitudes():
    signal = synthetic_signal()
    peaks, amplitudes = benchmark.derivative_candidates(signal)
    z, complete = combine.derivative_trace(signal)
    np.testing.assert_allclose(z[peaks], amplitudes, rtol=1e-12)
    assert complete[peaks].all()
    channels = combine.input_channels(z, np.zeros(len(z)))
    assert channels.shape == (2, len(z)) and np.isfinite(channels).all()


def test_and_combination_and_q_gate():
    presence = np.array([0.9, 0.9, 0.2, 0.9], np.float32)
    edge = np.array([12.0, 3.0, 50.0, 0.0])
    assert combine.combined_presence(presence, edge, 0).tolist() == presence.tolist()
    np.testing.assert_allclose(
        combine.combined_presence(presence, edge, 10), [0.9, 0, 0.2, 0]
    )
    gated = combine.gated_presence(presence, [False, True, False, True])
    np.testing.assert_allclose(gated, [0.9, 0, 0.2, 0])
