"""The floor, the prediction pass-through and the input round-trip, as harness predictors."""

import numpy as np
import pytest

from tokamak_foundation_model.ignite.mast import transform
from tokamak_foundation_model.ignite.mast.tokenizers import codecs, floor


def toy_codecs(task, k):
    out = {}
    for s in (*task.inputs, *task.actuators, *task.outputs):
        st = transform.ChannelStats(compress="none", scale=np.ones(s.values_shape),
                                    mean=np.full(s.values_shape, s.mean), std=np.full(s.values_shape, s.std))
        out[s.name] = codecs.FrameDCT(s.name, s.dt, s.values_shape, st, k=k)
    return out


def test_the_floor_is_the_frame_mean_at_one_value_and_exact_at_all(toy_task, toy_batch):
    b = toy_batch(6)
    exact = floor.floor_predictor(toy_task, toy_codecs(toy_task, 20))(b)["summary-ip"]
    assert exact.dtype == np.float32 and np.allclose(exact, b.targets["summary-ip"], rtol=1e-5)
    coarse = floor.floor_predictor(toy_task, toy_codecs(toy_task, 1))(b)["summary-ip"]
    t = b.targets["summary-ip"].reshape(6, 1, 5, 20)
    assert np.allclose(coarse.reshape(6, 1, 5, 20), t.mean(-1, keepdims=True), rtol=1e-5)


def test_the_floor_is_finite_where_the_target_is_missing(toy_task, toy_batch):
    b = toy_batch(4)
    b.targets["summary-ip"][0, 0, :20] = np.nan                              # a frame without a sample
    b.targets["summary-ip"][1, 0, 25] = np.nan                               # a frame with one missing sample
    out = floor.floor_predictor(toy_task, toy_codecs(toy_task, 4))(b)["summary-ip"]
    assert np.isfinite(out).all()
    assert np.allclose(out[0, 0, :20], toy_task.outputs[0].mean)
    assert np.allclose(out[1, 0, 20:40], np.nanmean(b.targets["summary-ip"][1, 0, 20:40]), rtol=1e-5)


def test_a_task_signal_without_a_codec_is_refused_when_the_predictor_is_built(toy_task):
    with pytest.raises(KeyError, match="summary-ip"):
        floor.floor_predictor(toy_task, {})
    with pytest.raises(KeyError, match="pf_active-coil_voltage"):
        floor.input_roundtrip_predictor(lambda b: {}, toy_task, {"summary-ip": toy_codecs(toy_task, 1)["summary-ip"]})


def test_passthrough_tokenizes_the_prediction_and_never_shows_the_targets(toy_task, toy_batch):
    seen = {}

    def inner(batch):
        seen["blind"] = all(np.isnan(a).all() for a in batch.targets.values())
        return {"summary-ip": np.tile(np.arange(100, dtype=np.float32), (len(batch.shot_ids), 1, 1))}

    inner.history_s = 0.05
    p = floor.passthrough_predictor(inner, toy_task, toy_codecs(toy_task, 1))
    out = p(toy_batch(3))["summary-ip"]
    assert seen["blind"] and p.history_s == 0.05 and out.shape == (3, 1, 100)
    assert np.allclose(out[0, 0, :20], 9.5) and np.allclose(out[2, 0, 80:], 89.5)


def test_the_input_roundtrip_changes_inputs_and_actuators_only_and_keeps_missing_samples(toy_task, toy_batch):
    seen = {}

    def inner(batch):
        seen.update(inputs=batch.inputs, actuators=batch.actuators, last_valid=batch.last_valid, targets=batch.targets)
        return {"summary-ip": np.zeros_like(batch.targets["summary-ip"])}

    b = toy_batch(2)
    b.inputs["summary-ip"][0, 0, 3] = np.nan
    original = b.inputs["summary-ip"].copy()
    floor.input_roundtrip_predictor(inner, toy_task, toy_codecs(toy_task, 1))(b)
    ip = seen["inputs"]["summary-ip"]
    assert np.isnan(ip[0, 0, 3]) and np.allclose(ip[0, 0, :3], np.nanmean(original[0, 0]), rtol=1e-5)
    volts = b.actuators["pf_active-coil_voltage"]
    assert np.allclose(seen["actuators"]["pf_active-coil_voltage"][1, 2, :20], volts[1, 2, :20].mean(), rtol=1e-4, atol=1e-2)
    assert seen["last_valid"] is b.last_valid and np.isnan(seen["targets"]["summary-ip"]).all()
    assert np.array_equal(b.inputs["summary-ip"], original, equal_nan=True)   # the caller's batch is not modified
