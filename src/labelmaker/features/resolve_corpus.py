"""Features from the FAITH corpus itself.

`<shot>_processed.h5` is 32 flat groups, each holding only `xdata` (1-D
float32 seconds) and `ydata` `(C, T)` float32, with no attributes anywhere
in the file - so units are not discoverable from the data and are recorded
here, measured. A group whose `ydata` last axis is shorter than 2 is the
corpus' absent-signal sentinel (see
tokamak_foundation_model/data/multi_file_dataset.py:845-861).

This resolver covers every corpus shot, which the archive resolver does not,
but it can only serve the actuator totals and raw waveforms: the corpus holds
raw diagnostics and actuators, no equilibrium and no fitted profiles.

Three shapes come out of here, decided by the feature's `kind` and by whether
its locator names a channel.

* a `scalar` feature whose locator is a bare group name is the group's channels
  summed into the canonical units and decimated onto the feature's `step`;
* a `scalar` feature whose locator ends in `#<k>` is channel `k` of that group
  ALONE, decimated the same way and never summed. `gas` is PTDATA `gasa`,
  channel 0 of the 11-channel `gas_raw` group, and summing the other ten valves
  into it would be a different quantity; the four `co2_<chord>` features are the
  same story one chord at a time. Only the named channel is read off disk;
* a `profile` feature is every channel of the group, decimated onto `step` and
  kept per channel (`ece`, 48 radiometer channels). The "radial" axis is
  whatever the group's channel order is, so nothing here interpolates it.
* a `waveform` feature (`co2`) is the group itself - every channel, native rate,
  no scale - because the model that consumes it does its own transform; see the
  branch below and the `co2` note in `namespace.py`.

Measured on shot 185945 at t = 1.025 s: summed `pinj` = 1.002e7 where the
model's own training column reads 10,995.6, so the corpus is in W and the
canonical feature is kW. `tinj` needs no scaling (8.43 vs 9.24 N m). The
residual difference is a sampling difference, not a unit one, and Task 15
decides between nearest-sample and window-mean by measurement.

A small but non-negligible fraction of corpus files are truncated on disk
and raise OSError on open: MEASURED 1.33% (8 of a random 600), and 2.0-3.7%
in three independent 300-file draws, so quote it as a range rather than a
point. They are recorded as a per-shot miss, never raised.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import h5py
import numpy as np

from ..timebase import decimate_to_step
from . import namespace as ns
from .store import FeatureArray

SOURCE = "corpus"

#: Multiplier taking a corpus channel sum to the canonical feature's units.
#: A feature resolved from a single channel or kept per channel needs no entry:
#: `_scale_for` returns 1.0 for it, since nothing here converts a raw
#: diagnostic channel into anything but itself.
SCALE_TO_CANONICAL: dict[str, float] = {
    "pinj_total": 1e-3,        # W -> kW
    "tinj_total": 1.0,         # already N m
    "ech_power_total": 1.0,    # already W; see the namespace note, measured
}


def split_locator(locator: str) -> tuple[str, int | None]:
    """`"gas_raw#0"` -> `("gas_raw", 0)`; `"ece"` -> `("ece", None)`."""
    group, sep, channel = locator.partition("#")
    return (group, int(channel)) if sep else (group, None)


def resolve(
    shot: int,
    names: Sequence[str],
    *,
    corpus: Path,
) -> tuple[dict[str, FeatureArray], dict[str, str]]:
    """Sum the channels of each requested corpus group, in canonical units.

    Returns `(arrays, missing)`; `missing` maps a feature name to a short
    cause so a run records why a shot is incomplete instead of failing.
    """
    # KeyError here names the feature and the source, and is raised before
    # any file is touched: asking the corpus for a profile is a programming
    # error, not a per-shot gap.
    specs = {n: ns.by_name(n) for n in names}
    locators = {n: split_locator(spec.locator_for(SOURCE))
                for n, spec in specs.items()}
    path = Path(corpus) / f"{int(shot)}_processed.h5"
    if not path.exists():
        return {}, dict.fromkeys(names, "FileNotFoundError")
    arrays: dict[str, FeatureArray] = {}
    missing: dict[str, str] = {}
    try:
        f = h5py.File(path, "r")
    except OSError:
        # Truncated corpus file; nothing in it is readable for any feature.
        return {}, dict.fromkeys(names, "OSError")
    with f:
        for name, (group, channel) in locators.items():
            if group not in f or "ydata" not in f[group]:
                missing[name] = "KeyError"
                continue
            # Shape checks off the dataset handle, before anything is read:
            # a `co2` group is `(4, ~4.5e6)`, and reading that as float64 to
            # find out it is the wrong shape would cost 144 MB per shot.
            dset = f[group]["ydata"]
            if dset.ndim != 2:
                # Every group measured is (C, T). A 1-D group would make the
                # per-channel reductions below reduce over time instead.
                missing[name] = f"ShapeError(ndim={dset.ndim})"
                continue
            if dset.shape[-1] < 2:
                missing[name] = "SignalAbsent"
                continue
            x = np.asarray(f[group]["xdata"], dtype=np.float64)
            if x.size != dset.shape[-1]:
                missing[name] = "ShapeError"
                continue
            if channel is not None and not 0 <= channel < dset.shape[0]:
                # A locator naming a channel the group does not have is a
                # per-shot gap, not a programming error: the corpus stores
                # different channel counts for the same group on some shots.
                missing[name] = f"ChannelMissing({channel}/{dset.shape[0]})"
                continue
            if specs[name].kind == "waveform":
                # A waveform is the raw record itself: no channel sum, no unit
                # scale, no decimation. The model transforms the signal (the AE
                # adapter takes its own STFT), so anything done here would have
                # to be undone there - and summing four CO2 chords would
                # destroy the very per-chord structure the network reads.
                # Kept float32, as stored: `(4, ~4e6)` in float64 is 128 MB per
                # shot for no gain, and the model casts to float32 anyway.
                wave = np.asarray(dset, dtype=np.float32)
                arrays[name] = FeatureArray(
                    x=x,
                    y=wave,
                    attrs={
                        "resolver": SOURCE,
                        "locator": group,
                        "corpus_file": str(path),
                        "n_channels": str(wave.shape[0]),
                        "nan_channels": str(
                            int((~np.isfinite(wave)).all(axis=1).sum())
                        ),
                        "scale_to_canonical": "1.0",
                        "native_rate": "1",
                        # From the span: `xdata` is float32 and a median diff
                        # quantises (see the namespace note on `co2`).
                        "sample_rate_hz": f"{(x.size - 1) / (x[-1] - x[0]):.1f}",
                    },
                )
                continue
            # Only the named channel is read when the locator names one: the
            # `co2` group is (4, ~4.5e6), and reading all four to keep one
            # would cost 144 MB per chord.
            y = (np.asarray(dset[channel], dtype=np.float64)[None, :]
                 if channel is not None
                 else np.asarray(dset, dtype=np.float64))
            nan_channels = int((~np.isfinite(y)).all(axis=1).sum())
            step = specs[name].step or 0.001
            if specs[name].kind == "profile" or channel is not None:
                # Per channel, not summed: `ece`'s 48 radiometer channels are
                # the model's 48 inputs, and `gas`/`co2_<chord>` are one named
                # channel of a group whose other channels are other quantities.
                scale = 1.0
                xg, yg = decimate_to_step(x, y, step)
            else:
                scale = SCALE_TO_CANONICAL[name]
                total = np.nansum(y, axis=0) * scale
                # A time where every channel is NaN is genuinely unknown;
                # nansum would report 0, which for ECH power is a different
                # claim.
                total[(~np.isfinite(y)).all(axis=0)] = np.nan
                xg, yg = decimate_to_step(x, total[None, :], step)
            arrays[name] = FeatureArray(
                x=xg,
                y=yg,
                attrs={
                    "resolver": SOURCE,
                    "locator": specs[name].locator_for(SOURCE),
                    "corpus_file": str(path),
                    "n_channels": str(y.shape[0]),
                    "nan_channels": str(nan_channels),
                    "scale_to_canonical": str(scale),
                    "decimated_to_s": str(step),
                },
            )
    return arrays, missing
