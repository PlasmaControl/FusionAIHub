"""d3d_ae_activity_seldnet - Alfven-eigenmode activity, and the mode's
frequency, from the four CO2 interferometer chords.

Upstream is this repository: the network is `labelmaker.ae.model.AeSeldNet`
and the checkpoint was trained by `scripts/labelmaker/ae_train.py` on task
7a's dataset (180 hand-annotated DIII-D shots, 170659-178879, 120 train /
60 validation). Nothing about it is somebody else's artifact, which is why
this is the one adapter whose `upstream` is a path inside our own data root.

The chain, per shot:

    co2 (4, ~4.5e6) at 500 kHz
      -> clipped to the output grid's span
      -> labelmaker.ae.transform.model_input: per-chord STFT (hann 1024,
         hop 128, |.|, log1p, DC dropped, 1/99 percentile clip), per-channel
         standardisation, bins 164:512  ->  (4, 348, frames)
      -> AeSeldNet, in time windows                     ->  (frames, 2)
      -> sigmoid on column 0, 5-frame moving average    ->  p(frame)
      -> denormalise column 1 (v * 170 + 80)            ->  f_khz(frame)
      -> aggregate onto the 25 ms grid                  ->  ae_active,
                                                            ae_frequency

**The spectrogram is never notched here.** Task 7a's per-bin notch (a bin lit
in more than 0.8 of the record is receiver pickup) was a rule over the *mask*
that produced the training label; at inference there is no mask, so there is
nothing to notch and the model sees the un-notched band it was trained to
read. The rule is recorded in `NOTCH_RULE` and written into every label
group's attributes, so a stored series says what the label it was fitted to
had removed. `notched_bins_khz` is `"none"` for the same reason, not because
the rule was forgotten.

Validity is not a `DomainRule`. Those clauses reduce a scalar or a profile on
the 25 ms grid, and this model's one input is a waveform that has no value on
that grid at all - what makes a row trustworthy here is whether the CO2
record covers it. So `InputSpec` rejects every row when `co2` is absent, and
`predict` narrows the mask in place for rows whose 25 ms window holds fewer
than `MIN_FRAMES` STFT frames (see `BuiltInputs`' docstring, which sanctions
exactly this).
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from ...ae import transform as tr
from ...ae.model import FREQ_LO_KHZ, FREQ_SPAN_KHZ, AeSeldNet, AeSeldNetConfig
from ...config import sha256_of
from ...features import namespace as ns
from ..base import (
    BuiltInputs,
    InputField,
    InputSpec,
    ModelAdapter,
    OutputField,
    OutputSpec,
)
from ..runners import torch_pt

SLUG = "d3d_ae_activity_seldnet"
CARD_ID = "plasmacontrol/d3d-ae-activity-seldnet"
UPSTREAM = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_ae_activity_seldnet")
ARTIFACTS = ("ae_seldnet_threeway_sce.pt",)
DT_S = ns.STEP_S

#: Frames the sigmoid is smoothed over before aggregation, as the task 7c
#: brief fixes it. 5 frames is 1.28 ms - shorter than the 25 ms output step,
#: so it removes single-frame flicker without moving an onset across a row.
SMOOTH_FRAMES = 5
#: A 25 ms window holds 97.6 frames at 0.256 ms/frame. Below this many the
#: window is a record edge (or a gap) and the row is marked invalid rather
#: than reported from a handful of frames.
MIN_FRAMES = 50
#: `ae_frequency` is NaN on a row whose `ae_active` is below this: the
#: probability-weighted mean frequency of a window the model calls quiet is
#: the frequency of nothing.
FREQ_MIN_ACTIVITY = 0.5
#: Frames per forward pass, and the extra frames of recurrent context each
#: pass is given on either side (`runners.torch_pt.run_windows`).
WINDOW_FRAMES = 1024
CONTEXT_FRAMES = 128

BAND_KHZ = "80-250"
FREQ_DEFINITION = (
    "probability-weighted mean of the per-frame frequency head over the "
    f"window's frames; NaN where ae_active < {FREQ_MIN_ACTIVITY}"
)
FREQ_CAVEAT = (
    "UNVALIDATED: on the 81,859 annotated-and-active validation frames the "
    "head's MAE is 17.63 kHz against 16.79 kHz for the global median, so it "
    "does not beat a constant. Read it as a diagnostic, never as a measurement"
)
NOTCH_RULE = (
    "training label only: a (channel, bin) row lit in more than 0.8 of the "
    "record was zeroed before the activity decision (labelmaker.ae.labels, "
    "task 7a). Not applied at inference - there is no mask to threshold - so "
    "the spectrogram the network reads here is un-notched"
)

INPUT_SPEC = InputSpec(
    fields=(InputField("co2_density", "co2"),),
    dt_s=DT_S,
    nan_policy="zero",
    # See the module docstring: a waveform cannot carry a DomainRule, and the
    # rules that matter here are enforced in `predict`.
    domain=(),
)

OUTPUT_ATTRS = (
    ("transform", tr.TRANSFORM_NAME),
    ("band_khz", BAND_KHZ),
    ("notch_rule", NOTCH_RULE),
    ("notched_bins_khz", "none"),
    ("aggregation",
     "mean of the 25 ms window's frame probabilities, window (t-25ms, t]"),
    ("min_frames_per_window", str(MIN_FRAMES)),
    ("smooth_frames", str(SMOOTH_FRAMES)),
)

OUTPUT_SPEC = OutputSpec(
    fields=(
        OutputField(
            "ae_active", "binary", column=0, activation="none", units="",
            classes=("quiet", "ae_active"),
            # `activation="none"` because the sigmoid is already applied, per
            # FRAME, before the moving average and the window mean. Letting
            # `OutputSpec.decode` do it instead would sigmoid a mean of
            # logits, which is a different number.
            attrs=OUTPUT_ATTRS,
        ),
        OutputField(
            "ae_frequency", "regression", column=1, activation="none",
            units="kHz",
            attrs=OUTPUT_ATTRS + (
                ("definition", FREQ_DEFINITION),
                ("caveat", FREQ_CAVEAT),
            ),
        ),
    )
)

#: The 180 annotated shots the checkpoint was fitted and validated on, one per
#: line, taken from `aemodes/data/.cache/ae_timeseries/`. Committed for the
#: same reason the survival model's list is: without it no pool number can be
#: split into held-out and in-sample halves. Here the answer is trivial and
#: worth recording anyway - NONE of the 180 has a FAITH corpus file, so every
#: corpus label this model writes is on a held-out shot.
TRAINING_SHOTS = frozenset(
    int(line)
    for line in Path(__file__).with_name("training_shots.txt").read_text().split()
)


def clip_to_grid(x: np.ndarray, y: np.ndarray, grid: np.ndarray):
    """The samples the 25 ms windows of `grid` can possibly need.

    The corpus record runs -1.45 s to 6.5-9.5 s and the grid covers
    0-5.975 s, so a third to a half of every record is outside any window.
    Clipping matters for more than speed: the transform's percentile clip and
    its standardisation are taken over the WHOLE array it is handed, so
    leaving the pre-shot baseline and the long post-shot tail in would move
    every value the network sees. Trained records were 2 s of plasma; this is
    the closest an 8 s corpus record gets to that without inventing a
    plasma-window rule the model was never given.
    """
    lo = float(grid[0]) - DT_S
    hi = float(grid[-1]) + DT_S
    keep = np.flatnonzero((x >= lo) & (x <= hi))
    if keep.size < tr.N_FFT:
        return None
    first, last = int(keep[0]), int(keep[-1]) + 1
    return x[first:last], y[:, first:last]


def moving_average(p: np.ndarray, frames: int = SMOOTH_FRAMES) -> np.ndarray:
    """Centred boxcar over `frames`, with the ends averaged over what exists."""
    p = np.asarray(p, dtype=np.float64)
    if frames <= 1 or p.size == 0:
        return p
    kernel = np.ones(int(frames), dtype=np.float64)
    num = np.convolve(p, kernel, mode="same")
    den = np.convolve(np.ones_like(p), kernel, mode="same")
    return num / den


def aggregate(
    t_frames: np.ndarray, prob: np.ndarray, freq_khz: np.ndarray, grid: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Frame series -> `(ae_active, ae_frequency, n_frames)` on the grid.

    Row `k` summarises the half-open window `(grid[k] - 25 ms, grid[k]]`, the
    causal convention every other labelmaker model's inputs follow. Row 0's
    window therefore ends at t = 0 and normally holds no frames, which is the
    same record-edge outcome the scalar path has at row 0.
    """
    grid = np.asarray(grid, dtype=np.float64)
    n = grid.size
    # ceil(t / step) is the row whose window (t_k - step, t_k] contains t.
    idx = np.ceil(np.asarray(t_frames, dtype=np.float64) / DT_S - 1e-9).astype(np.int64)
    inside = (idx >= 0) & (idx < n)
    idx, prob, freq_khz = idx[inside], prob[inside], freq_khz[inside]
    counts = np.bincount(idx, minlength=n).astype(np.int64)
    prob_sum = np.bincount(idx, weights=prob, minlength=n)
    freq_sum = np.bincount(idx, weights=prob * freq_khz, minlength=n)
    with np.errstate(invalid="ignore", divide="ignore"):
        active = np.where(counts > 0, prob_sum / np.maximum(counts, 1), np.nan)
        frequency = np.where(prob_sum > 0, freq_sum / np.maximum(prob_sum, 1e-12), np.nan)
    frequency = np.where(active >= FREQ_MIN_ACTIVITY, frequency, np.nan)
    return active, frequency, counts


def make_load(artifact: str) -> Callable[[Path], Callable[[BuiltInputs], np.ndarray]]:
    """A `load` for a checkpoint of this architecture under another file name."""

    def load(model_dir: Path) -> Callable[[BuiltInputs], np.ndarray]:
        model_dir = Path(model_dir)
        path = model_dir / artifact
        module, blob = torch_pt.load_module(
            path, lambda cfg: AeSeldNet(AeSeldNetConfig.from_dict(cfg))
        )
        lo = float(blob.get("freq_lo_khz", FREQ_LO_KHZ))
        span = float(blob.get("freq_span_khz", FREQ_SPAN_KHZ))
        checkpoint_sha = sha256_of(path)

        def predict(built: BuiltInputs) -> np.ndarray:
            grid = np.asarray(built.t, dtype=np.float64)
            out = np.full((1, grid.size, 2), np.nan)
            arr = built.waveforms.get("co2")
            if arr is None:
                # Every row is already invalid (`InputSpec._validity`); the
                # label file still gets a full-length NaN series so a
                # consumer sees the shot was attempted.
                return out
            clipped = clip_to_grid(
                np.asarray(arr.x, dtype=np.float64), np.asarray(arr.y), grid
            )
            if clipped is None:
                built.valid[:] = False
                return out
            x, y = clipped
            fs_hz = (x.size - 1) / (x[-1] - x[0])
            spec = tr.model_input(y)                      # (C, 348, frames)
            t_frames = tr.frame_times_s(y.shape[1], fs_hz) + x[0]
            raw = torch_pt.run_windows(
                module,
                np.ascontiguousarray(spec.transpose(0, 2, 1)),   # (C, frames, 348)
                window=WINDOW_FRAMES,
                context=CONTEXT_FRAMES,
            )
            prob = moving_average(1.0 / (1.0 + np.exp(-raw[:, 0].astype(np.float64))))
            freq = raw[:, 1].astype(np.float64) * span + lo
            active, frequency, counts = aggregate(t_frames, prob, freq, grid)
            # In place: `built` is a frozen dataclass, so `&=` would try to
            # rebind the field. Narrowing the array itself is the whole
            # point (see `BuiltInputs`' docstring).
            np.logical_and(built.valid, counts >= MIN_FRAMES, out=built.valid)
            out[0, :, 0] = active
            out[0, :, 1] = frequency
            return out

        predict.output_spec = OutputSpec(tuple(
            OutputField(
                f.name, f.task, f.column, f.activation, f.units, f.classes,
                f.attrs + (("checkpoint_sha256", checkpoint_sha),
                           ("checkpoint", artifact)),
            )
            for f in OUTPUT_SPEC.fields
        ))
        return predict

    return load


load = make_load(ARTIFACTS[0])


ADAPTER = ModelAdapter(
    slug=SLUG,
    card_id=CARD_ID,
    framework="torch_pt",
    time_step_ms=DT_S * 1000.0,
    artifacts=ARTIFACTS,
    upstream=str(UPSTREAM),
    input_spec=INPUT_SPEC,
    output_spec=OUTPUT_SPEC,
    load=load,
    ensemble_n=1,
    training_shots=TRAINING_SHOTS,
)
