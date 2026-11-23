"""The three instruments of the tokenizer comparison, as harness predictors.

The floor answers: what is the best score any model could reach if it predicted the true targets and the
only loss were the tokenizer? It encodes and decodes the true targets of each output signal, so it reads
its targets by definition. The prediction pass-through answers: how much does the tokenizer cost a given
model? It takes the prediction of a model that never sees its targets and sends it through the same codec.
The input round-trip answers: how much does the model lose when its own inputs and actuators arrive
tokenized? It shows the model inputs that went through the codec and leaves everything else as it was.

Every function takes and returns arrays in physical units (the harness's), reads values through
`Codec.roundtrip` (and so through `Codec.read`, never `grids.read`), and keeps NaN where a frame has no
valid input sample.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping

import numpy as np

from tokamak_foundation_model.ignite.mast import harness, tasks
from tokamak_foundation_model.ignite.mast.tokenizers.codecs import Codec


def _require(codecs: Mapping[str, Codec], signals: tuple[tasks.SignalSpec, ...], role: str) -> None:
    missing = [s.name for s in signals if s.name not in codecs]
    if missing:
        raise KeyError(f"no codec for the {role} signals {missing}")


def _keep_attributes(wrapper, inner):
    wrapper.__dict__.update(getattr(inner, "__dict__", {}))
    return wrapper


def floor_predictor(task: tasks.TaskSpec, codecs: Mapping[str, Codec], mode: str = "unquantized") -> harness.PredictFn:
    """The true targets through the codecs. Physical units in and out, float32; a value that is not finite
    (a frame with no sample) is the signal's `SignalSpec.mean`. `KeyError` if an output has no codec."""
    _require(codecs, task.outputs, "output")

    def predict(batch: harness.WindowBatch) -> dict[str, np.ndarray]:
        out = {}
        for s in task.outputs:
            rt = codecs[s.name].roundtrip(batch.targets[s.name], mode)
            out[s.name] = np.where(np.isfinite(rt), rt, s.mean).astype(np.float32)
        return out

    return predict


def blind_to_targets(predict: harness.PredictFn) -> harness.PredictFn:
    """`predict` on a copy of the batch whose targets are NaN arrays of the same shapes and dtypes.

    A deliberate copy of `scripts/mast/run_headroom.py:_blind_to_targets` (package code cannot import a
    script); change one and change the other. The wrapper keeps the attributes of `predict` (`history_s`).
    """
    def blind(batch: harness.WindowBatch) -> dict:
        nan = {name: np.full(a.shape, np.nan, dtype=a.dtype if a.dtype.kind == "f" else np.float32)
               for name, a in batch.targets.items()}
        return predict(dataclasses.replace(batch, targets=nan))

    return _keep_attributes(blind, predict)


def passthrough_predictor(inner: harness.PredictFn, task: tasks.TaskSpec, codecs: Mapping[str, Codec],
                          mode: str = "unquantized") -> harness.PredictFn:
    """The prediction of `blind_to_targets(inner)` through the codecs. Physical units in and out, float32.
    `KeyError` if an output has no codec; the attributes of `inner` are kept."""
    _require(codecs, task.outputs, "output")
    blind = blind_to_targets(inner)

    def predict(batch: harness.WindowBatch) -> dict[str, np.ndarray]:
        pred = blind(batch)
        return {s.name: codecs[s.name].roundtrip(pred[s.name], mode).astype(np.float32) for s in task.outputs}

    return _keep_attributes(predict, inner)


def input_roundtrip_predictor(inner: harness.PredictFn, task: tasks.TaskSpec, codecs: Mapping[str, Codec],
                              mode: str = "unquantized") -> harness.PredictFn:
    """`blind_to_targets(inner)` on a copy of the batch whose inputs and actuators went through the codecs.

    Physical units; a sample that was NaN stays NaN; `last_valid` is passed on unchanged and the caller's
    batch is not modified. `KeyError` if an input or an actuator has no codec; the attributes of `inner`
    are kept.
    """
    _require(codecs, (*task.inputs, *task.actuators), "input and actuator")
    blind = blind_to_targets(inner)

    def through(arrays: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        out = {}
        for name, a in arrays.items():
            rt = codecs[name].roundtrip(a, mode).astype(np.float32)
            out[name] = np.where(np.isnan(a), np.float32(np.nan), rt)
        return out

    def predict(batch: harness.WindowBatch) -> dict[str, np.ndarray]:
        return blind(dataclasses.replace(batch, inputs=through(batch.inputs), actuators=through(batch.actuators)))

    return _keep_attributes(predict, inner)
