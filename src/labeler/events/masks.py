"""One corpus channel -> spectrogram -> tiles -> U-Net -> a packed mask file.

This is the data path of a mask run, and every stage of it is pinned by a
measurement rather than chosen:

**The transform is TokEye's own.** `ae/transform.py` already carries it,
ported line for line from `tokeye.transforms.compute_stft` and pinned
against the original - Hann 1024, hop 128, `log1p|STFT|`, DC dropped
(512 bins), clipped to its own 1st/99th percentile - followed by
`model_infer`'s per-channel z-score. `prep` calls exactly those two
functions, so a spectrogram fed to the U-Net here is the spectrogram it was
trained on. It does NOT call `model_input`: that one cuts to bins 164:512
because the AE *label* model was trained on that band, and the event layer
segments the whole spectrogram - EHO and fishbones live at 2-30 kHz, in
bins 4-61.

**Two passes, because four max-pools are four max-pools.** A mode at 5 kHz
occupies ten of 512 rows in the wide pass, which the network's deepest
level sees as less than one pixel. The zoom pass decimates the waveform by
`ZOOM_DECIM` first (`scipy.signal.decimate`, IIR, zero-phase), which puts
the same mode on 0.12207 kHz/bin instead of 0.48828. Measured on shot
198658's CO2: the lit fraction went 0.0220 -> 0.0554 and the centroid
37.3 -> 22.1 kHz, and rows of receiver pickup that the wide pass smeared
became single rows (co2 1.7-2.3 kHz, ece 8.4-9.0 kHz).

**Tiles, because a shot is 16,391 columns wide.** The network takes a
512-column tile; tiles overlap by `OVERLAP` columns and the overlaps are
AVERAGED back, not chosen between, so nothing shows a seam every 448
columns. A mode that straddles a tile boundary is one mode.

**Packed masks, because the raw ones are 30 TB.** 2,000 shots x 11
channels x 2 passes of `(512, T)` float32 is not storable; thresholded at
`PROB_THRESHOLD` and `np.packbits`-ed it is ~1-2 MB per channel, one
`<shot>_masks.npz` per shot (~3 MB), and that plus the per-row and
per-column summaries is everything `events/tracks.py` and
`events/transients.py` need. What is thrown away is the probability
between the threshold and one; `_meta` records the threshold that was
used, so a file never has to be guessed at.

Nothing here decides what a lit region IS. That is `tracks.py`'s job.
"""
from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from scipy import signal

from ..ae.labels import HOP, N_BINS, N_FFT, PROB_THRESHOLD
from ..ae.transform import STD_EPS, compute_stft, standardise
from ..config import atomic_path
from .unet import probabilities

#: Columns per tile - the width the network is run at.
TILE = 512
#: Columns two neighbouring tiles share; averaged back in `stitch`.
OVERLAP = 64
#: New columns per tile: 448, i.e. 114.7 ms wide / 459 ms zoom.
STRIDE = TILE - OVERLAP
#: Decimation of the zoom pass: 0.48828 -> 0.12207 kHz/bin.
ZOOM_DECIM = 4
#: Frequency bands the pre-standardisation log-power is summarised into.
N_BANDS = 16
#: The two passes every planned channel gets. A subset of
#: `events/schema.PASS_NAMES`, which also allows "" for an event that has
#: no spectrogram behind it.
PASS_NAMES = ("wide", "zoom")

#: Suffixes of the seven keys one `(diag, channel, pass)` block writes. The
#: storage contract: `tracks.py`, `transients.py` and the render step read
#: these names, so they are constants rather than f-strings at each site.
KEY_SUFFIXES = (
    "_coh_packed", "_tra_packed", "_row_lit", "_col_act",
    "_band_logpow", "_t_s", "_meta",
)

#: The one key in a masks file that is not part of a block.
SHOT_KEY = "shot"

#: Column 0 of the STFT sits three hops BEFORE sample 0: `ShortTimeFFT`
#: pads the start of the record, and `frame_times_s` is the authority this
#: is pinned against (`test_the_column_times_are_the_transforms_own_frame_grid`).
COL_ORIGIN = -3


# ------------------------------------------------------------------ reading

def read_waveform(corpus_file, group: str, channel: int):
    """One channel of one corpus group -> `(y, fs_hz, t0_s, t1_s)`.

    Reads a single contiguous row: the corpus' datasets are uncompressed and
    unchunked, so `ydata[ch, :]` costs one seek and the row (measured: 0.02 s
    for three ECE channels against 0.31 s for all 48).

    **Non-finite samples are stripped from BOTH ends**, with the matching
    `xdata`, and `t0_s`/`t1_s` are the span of what is left. Every fast
    group is 2^k+1 samples long and the last one is NaN; `mirnov` on shot
    198658 also carries 1,669,828 NaN samples at the FRONT, before the
    digitiser is live. Left in, the transform's percentile is NaN, the
    standardisation is NaN, and the U-Net returns a blank mask with no error
    at all; stripped while keeping the file's own `t[0]`, every column of
    the spectrogram would be reported 3.3 s early. An **interior**
    non-finite sample is a different animal - a gap in the digitiser record,
    not an end artefact - and raises, because filling it would invent
    spectral content and dropping it would misplace every later column in
    time.

    `fs_hz` is `(n - 1) / (t[-1] - t[0])` of what is actually there: 500 kHz
    is nominal, and every kHz this pipeline reports is derived from this
    number.
    """
    with h5py.File(corpus_file, "r", locking=False) as f:
        if group not in f or "ydata" not in f[group]:
            raise KeyError(f"{corpus_file}: no group {group!r}")
        dset = f[group]["ydata"]
        if dset.ndim != 2:
            raise ValueError(f"{group}: expected (channels, samples), got {dset.shape}")
        if dset.shape[-1] < 2:
            raise ValueError(
                f"{group}: absent on this shot (ydata is {dset.shape})"
            )
        if not 0 <= int(channel) < dset.shape[0]:
            raise IndexError(
                f"{group}: channel {channel} outside 0..{dset.shape[0] - 1}"
            )
        y = np.asarray(dset[int(channel), :], dtype=np.float32)
        x = np.asarray(f[group]["xdata"][:], dtype=np.float64)
    if x.size != y.size:
        raise ValueError(
            f"{group}: xdata has {x.size} samples, ydata channel "
            f"{channel} has {y.size}"
        )
    finite = np.flatnonzero(np.isfinite(y))
    if finite.size == 0:
        raise ValueError(f"{group} channel {channel}: no finite samples")
    lo, hi = int(finite[0]), int(finite[-1]) + 1
    y, x = y[lo:hi], x[lo:hi]
    n = hi - lo
    if not np.isfinite(y).all():
        bad = lo + int(np.flatnonzero(~np.isfinite(y))[0])
        raise ValueError(
            f"{group} channel {channel}: interior non-finite sample at "
            f"index {bad} of the finite span {lo}..{hi}"
        )
    if n < 2 or not np.isfinite(x[[0, -1]]).all() or x[-1] == x[0]:
        raise ValueError(
            f"{group} channel {channel}: {n} finite samples spanning "
            f"{x[0]} to {x[-1]} - no sample rate to be had"
        )
    fs_hz = float((n - 1) / (x[-1] - x[0]))
    return y, fs_hz, float(x[0]), float(x[-1])


# ---------------------------------------------------------------- transform

def prep(y, *, fs_hz: float, decim: int = 1):
    """Waveform -> `((512, T) float32, meta)`: the pinned TokEye input.

    `decim=1` is the wide pass and is *exactly*
    `standardise(compute_stft(y))` - the same two functions the AE label
    path uses, in the same order, so the network sees what it was trained
    on. `decim=ZOOM_DECIM` is the zoom pass: the waveform is decimated
    first (`ftype="iir", zero_phase=True`, so the filter adds no delay and
    column `k` still sits at sample `k * hop * decim`), which divides the
    frequency scale by four and multiplies the column duration by four.
    Nothing else may be passed: the two passes are the two that were
    measured, and a third would need its own tile counts and its own axes.

    `meta` is the grid plus the statistics that were applied. `fs_hz` in it
    is the RECORD's rate, not the decimated one - `freq_axis_khz` and
    `col_times_s` both take `(fs_hz, decim)` and divide - and `spec_mean`,
    `spec_std`, `clip_lo`, `clip_hi` describe the standardisation, so a
    caller can recover the pre-standardisation log-power that
    `band_logpow` wants without keeping a second `(512, T)` array alive:

        raw = spec * (meta["spec_std"] + ae.transform.STD_EPS) + meta["spec_mean"]

    (`clip_lo`/`clip_hi` are the percentile bounds `compute_stft` clipped
    to, which are the min and max of what it returned.)
    """
    y = np.asarray(y)
    if y.ndim != 1:
        raise ValueError(f"prep takes one channel, got shape {y.shape}")
    decim = int(decim)
    if decim == 1:
        wave = y
    elif decim == ZOOM_DECIM:
        wave = signal.decimate(y, ZOOM_DECIM, ftype="iir", zero_phase=True)
    else:
        raise ValueError(
            f"decim must be 1 (wide) or {ZOOM_DECIM} (zoom), not {decim}"
        )
    raw = compute_stft(wave)
    spec = standardise(raw[None])[0]
    meta = {
        "fs_hz": float(fs_hz),
        "decim": decim,
        "n_cols": int(spec.shape[1]),
        "hop_s": float(HOP * decim / fs_hz),
        "freq_khz_per_bin": float(fs_hz / decim / 1e3 / N_FFT),
        "spec_mean": float(raw.mean()),
        "spec_std": float(raw.std()),
        "clip_lo": float(raw.min()),
        "clip_hi": float(raw.max()),
    }
    return spec, meta


def unstandardise(spec, meta: Mapping[str, Any]) -> np.ndarray:
    """`prep`'s spectrogram and meta -> the pre-standardisation log-power.

    The inverse of the z-score `prep` applied, and the only place the
    formula is written: `band_logpow` and `tracks.descriptors` both want the
    log-power that `prep` does not return, and a caller retyping
    `spec * (std + eps) + mean` out of a docstring is a caller that can
    retype it wrongly. The percentile clip is NOT undone - it happened
    before the z-score and threw information away - so what comes back is
    `compute_stft`'s output, to float32 rounding.
    """
    spec = np.asarray(spec, dtype=np.float32)
    std = float(meta["spec_std"])
    mean = float(meta["spec_mean"])
    return (spec * (std + STD_EPS) + mean).astype(np.float32)


def freq_axis_khz(fs_hz: float, decim: int = 1) -> np.ndarray:
    """Centre frequency of each of the 512 bins, in kHz.

    Bin `k` (1-based, because the DC bin was dropped) is
    `k * fs / decim / 1024`: 0.48828 kHz per bin on the 500 kHz record,
    0.12207 on the zoom pass, and the top bin is the record's Nyquist.
    """
    k = np.arange(1, N_BINS + 1, dtype=np.float64)
    return k * (float(fs_hz) / int(decim) / 1e3) / N_FFT


def col_times_s(n_cols: int, fs_hz: float, decim: int, t0_s: float) -> np.ndarray:
    """Centre time of each STFT column, in seconds of the shot's clock.

    `t0_s` is the first sample's time, from `read_waveform`. Column 0 sits
    `COL_ORIGIN` hops before it because `ShortTimeFFT` pads the start of the
    record; column `k` is `t0_s + (k - 3) * hop * decim / fs`.
    """
    k = np.arange(int(n_cols), dtype=np.float64) + COL_ORIGIN
    return float(t0_s) + k * (HOP * int(decim) / float(fs_hz))


# ------------------------------------------------------------ tile / stitch

def n_tiles(n_cols: int) -> int:
    """How many tiles `tile` cuts `n_cols` columns into.

    The formula, named, because the batch driver (`events/driver.py`)
    reports tiles per second and has to count a block's tiles WITHOUT
    building the `(n, 512, 512)` array to count them: a `mirnov` wide pass
    is 143 tiles and 150 MB, and counting that way would double the memory
    the driver exists to bound.
    """
    n_cols = int(n_cols)
    return 1 + max(0, -(-(n_cols - TILE) // STRIDE))


def tile(spec):
    """`(512, T)` -> `((n_tiles, 512, 512) float32, meta)`, last one padded.

    Tiles start every `STRIDE` columns, so neighbours share `OVERLAP`
    columns; the last tile is zero-padded to the full width rather than
    shortened, because a batch has to be one array and the network's
    downsampling wants a multiple of 16 anyway. `meta` carries the starts
    and the true column count, which is all `stitch` needs to undo this.
    """
    spec = np.asarray(spec, dtype=np.float32)
    if spec.ndim != 2 or spec.shape[0] != N_BINS:
        raise ValueError(f"expected ({N_BINS}, T), got {spec.shape}")
    n_cols = int(spec.shape[1])
    count = n_tiles(n_cols)
    starts = [i * STRIDE for i in range(count)]
    tiles = np.zeros((count, N_BINS, TILE), dtype=np.float32)
    for i, start in enumerate(starts):
        chunk = spec[:, start:start + TILE]
        tiles[i, :, :chunk.shape[1]] = chunk
    meta = {
        "n_cols": n_cols,
        "n_tiles": count,
        "starts": starts,
        "tile": TILE,
        "overlap": OVERLAP,
        "stride": STRIDE,
    }
    return tiles, meta


def stitch(pred, meta: Mapping[str, Any]) -> np.ndarray:
    """`(n_tiles, C, 512, 512)` -> `(C, 512, T)`, overlaps averaged.

    Averaged, not overwritten: the network sees less context at a tile's
    edge than at its middle, and taking one tile's word for a shared column
    puts a step every `STRIDE` columns straight into the mask. The
    accumulation is float64 so that `k` copies of one value average back to
    that value exactly.
    """
    pred = np.asarray(pred)
    starts = list(meta["starts"])
    n_cols = int(meta["n_cols"])
    if pred.ndim != 4 or pred.shape[0] != len(starts):
        raise ValueError(
            f"expected ({len(starts)}, C, {N_BINS}, {TILE}), got {pred.shape}"
        )
    if pred.shape[-2:] != (N_BINS, TILE):
        raise ValueError(
            f"expected tiles of ({N_BINS}, {TILE}), got {pred.shape[-2:]}"
        )
    n_ch = int(pred.shape[1])
    width = starts[-1] + TILE
    total = np.zeros((n_ch, N_BINS, width), dtype=np.float64)
    count = np.zeros(width, dtype=np.float64)
    for i, start in enumerate(starts):
        total[:, :, start:start + TILE] += pred[i]
        count[start:start + TILE] += 1.0
    return (total[:, :, :n_cols] / count[:n_cols]).astype(np.float32)


def _to_device(x, device):
    """One batch of tiles, on the device. A seam, and a named one.

    `infer`'s out-of-memory retry has to cover this line, and a test can
    only prove that by making the transfer itself raise; patching
    `torch.Tensor.to` would reach every tensor in the process.
    """
    return x.to(device)


@dataclass
class CompactMask:
    """Host result: stored summaries and only descriptor-visible probabilities.

    Sparse coherent values are in row-major lit-pixel order. Components never
    grow the threshold mask, so reconstructing its other pixels as zero leaves
    all descriptor inputs exact. Transient floats have no downstream consumer.
    """

    n_cols: int
    coh_packed: np.ndarray
    tra_packed: np.ndarray
    row_lit: np.ndarray
    col_act: np.ndarray
    coh_values: np.ndarray

    def coherent_inputs(self):
        lit = unpack(self.coh_packed, (N_BINS, self.n_cols))
        prob = np.zeros(lit.shape, dtype=np.float32)
        prob[lit] = self.coh_values
        return prob, lit


def _pack_device(lit):
    # N_BINS is divisible by eight, even for a partial last column.
    bits = lit.reshape(-1, 8).to(torch.uint8)
    shifts = torch.arange(7, -1, -1, device=lit.device, dtype=torch.uint8)
    return (bits << shifts).sum(dim=1).to(torch.uint8)


class DeviceStitch:
    """The reference's ordered float64 overlap sum, retained on the device."""

    def __init__(self, n_cols, device):
        self.n_cols = int(n_cols)
        width = (n_tiles(n_cols) - 1) * STRIDE + TILE
        self.total = torch.zeros((2, N_BINS, width), dtype=torch.float64, device=device)
        self.count = torch.zeros(width, dtype=torch.float64, device=device)

    def add(self, pred, first_tile):
        for i in range(len(pred)):
            start = (first_tile + i) * STRIDE
            self.total[:, :, start : start + TILE].add_(pred[i].float())
            self.count[start : start + TILE].add_(1)

    def finish(self):
        probs = (self.total[:, :, : self.n_cols] / self.count[: self.n_cols]).float()
        lit = probs >= PROB_THRESHOLD
        # NumPy bool.mean accumulates/divides in float64 before casting.
        row_lit = (lit[0].sum(dim=1).double() / self.n_cols).float()
        col_act = (lit[1].sum(dim=0).double() / N_BINS).float()
        return CompactMask(
            self.n_cols,
            _pack_device(lit[0]).to("cpu").numpy(),
            _pack_device(lit[1]).to("cpu").numpy(),
            row_lit.to("cpu").numpy(),
            col_act.to("cpu").numpy(),
            probs[0][lit[0]].to("cpu").numpy(),
        )


def _infer_compact(model, spec, device, *, batch, amp):
    tiles, meta = tile(spec)
    device = torch.device(device)
    x = torch.from_numpy(tiles).unsqueeze(1)
    stitched = DeviceStitch(meta["n_cols"], device)
    i, size = 0, max(1, int(batch))
    while i < len(x):
        chunk = None
        try:
            chunk = _to_device(x[i : i + size], device)
            with (
                torch.autocast("cuda", dtype=torch.float16)
                if amp and device.type == "cuda"
                else nullcontext()
            ):
                probs = probabilities(model, chunk)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            size = max(1, size // 2)
            chunk = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        stitched.add(probs, i)
        i += len(probs)
    return stitched.finish()


@dataclass
class _PooledBlock:
    token: Any
    n_cols: int
    stitch: DeviceStitch | None = None
    error: Exception | None = None


def _fill_tiles(host, spec, used, first, take):
    view = host.numpy()
    for offset in range(take):
        start = (first + offset) * STRIDE
        chunk = spec[:, start : start + TILE]
        view[used + offset, 0, :, : chunk.shape[1]] = chunk
        view[used + offset, 0, :, chunk.shape[1] :] = 0


def _tile_batches(blocks, batch, *, pin_memory):
    """Copy directly into bounded full batches; one partial batch at EOF only."""
    host, spans, used = None, [], 0
    for token, spec in blocks:
        state = _PooledBlock(token, spec.shape[1])
        count = n_tiles(state.n_cols)
        first = 0
        while first < count:
            if host is None:
                host = torch.empty(
                    (batch, 1, N_BINS, TILE), dtype=torch.float32, pin_memory=pin_memory
                )
            take = min(batch - used, count - first)
            _fill_tiles(host, spec, used, first, take)
            spans.append((state, used, take, first, first + take == count))
            used += take
            first += take
            if used == batch:
                yield host, spans
                host, spans, used = None, [], 0
    if used:
        yield host[:used], spans


def _copy_batch(host, device, stream):
    if stream is None:
        return host.to(device), None
    with torch.cuda.stream(stream):
        copied = host.to(device, non_blocking=True)
        ready = torch.cuda.Event()
        ready.record(stream)
    return copied, ready


class _ReferenceBatches:
    """CUDA forward boundaries must match the independent per-block oracle.

    cuDNN AMP rounding changes when a tile moves to another batch shape (and
    can cross the stored threshold). Pool full H2D buffers, but assemble device
    views back into the reference's forward groups. At most one partial group
    waits for the following input buffer. Preserve these boundaries on CPU too:
    its measured single-threaded exactness need not hold for every backend.
    """

    def __init__(self, model, device, batch, amp, forward_context):
        self.model, self.device = model, device
        self.batch, self.amp = batch, amp
        self.forward_context = forward_context
        self.pending = []
        self.state = None
        self.done = 0
        self.size = batch

    def consume(self, copied, spans):
        completed = []
        for state, begin, count, _first, last in spans:
            if self.state is None:
                self.state, self.done, self.size = state, 0, self.batch
            if state is not self.state:
                raise RuntimeError("interleaved blocks in reference forward group")
            if state.error is not None:
                # A failed transfer may interrupt a group spanning two buffers.
                # Drop its retained input and drain just this block's spans.
                self.pending.clear()
                if last:
                    completed.append((state, None))
                    self.state = None
                continue
            self.pending.append(copied[begin : begin + count])
            available = sum(len(x) for x in self.pending)
            total = n_tiles(state.n_cols)
            while available >= min(self.size, total - self.done):
                take = min(self.size, total - self.done)
                joined = (
                    self.pending[0]
                    if len(self.pending) == 1
                    else torch.cat(self.pending, dim=0)
                )
                self.pending = [joined]
                probs = None
                if state.error is None:
                    try:
                        with (
                            self.forward_context([state.token]),
                            (
                                torch.autocast("cuda", dtype=torch.float16)
                                if self.amp and self.device.type == "cuda"
                                else nullcontext()
                            ),
                        ):
                            probs = probabilities(self.model, joined[:take])
                    except torch.cuda.OutOfMemoryError as exc:
                        if self.size > 1:
                            self.size = max(1, self.size // 2)
                            if self.device.type == "cuda":
                                torch.cuda.empty_cache()
                            continue
                        state.error = type(exc)(str(exc))
                    except Exception as exc:  # noqa: BLE001 - one reference block
                        state.error = type(exc)(str(exc))
                if state.error is None:
                    if state.stitch is None:
                        state.stitch = DeviceStitch(state.n_cols, self.device)
                    state.stitch.add(probs, self.done)
                self.done += take
                available -= take
                self.pending = [joined[take:]] if available else []
                del joined, probs
                if self.done == total:
                    if not last or available:
                        raise RuntimeError("invalid end of reference forward group")
                    event = None
                    if self.device.type == "cuda":
                        event = torch.cuda.Event()
                        event.record(torch.cuda.current_stream(self.device))
                    completed.append((state, event))
                    self.state = None
                    break
        return completed


def _preserves_reference_batches(device):
    """Keep oracle forward shapes on every device; CPU pooling is opt-in."""
    return True


def infer_pooled(
    model,
    blocks,
    device,
    *,
    batch=96,
    amp=True,
    forward_context=nullcontext,
    preserve_batches=None,
    remaining=None,
):
    """Yield `(token, CompactMask | Exception)` from consecutive block tiles.

    Two pinned input batches and one unfinished block bound host input memory.
    The copy stream uploads the next batch while the current forward runs.
    A result stream reduces/reads completed blocks beside the next forward.
    Owners survive copies; record_stream protects device input reuse. OOM
    halves the forward size, retaining completed work and retrying only the
    failing slice. Every output remains in input block order.
    """
    from ..run import StageTimeout

    device = torch.device(device)
    cuda = device.type == "cuda"
    if preserve_batches is None:
        preserve_batches = _preserves_reference_batches(device)
    reference = (
        _ReferenceBatches(model, device, max(1, int(batch)), amp, forward_context)
        if preserve_batches
        else None
    )
    copy_stream = torch.cuda.Stream(device=device) if cuda else None
    result_stream = torch.cuda.Stream(device=device) if cuda else None
    batch = max(1, int(batch))
    size, size_state = batch, None
    batches = iter(_tile_batches(blocks, batch, pin_memory=cuda))
    current = next(batches, None)
    if current is None:
        return
    host, spans = current

    def stage(host):
        try:
            return _copy_batch(host, device, copy_stream)
        except torch.cuda.OutOfMemoryError:
            # Retry transfer together with the forward, in smaller slices.
            return None, None

    copied, ready = stage(host)
    completed = []

    def collect():
        for state, event in completed:
            if state.error is not None:
                yield state.token, state.error
            elif cuda:
                result_stream.wait_event(event)
                with torch.cuda.stream(result_stream):
                    state.stitch.total.record_stream(result_stream)
                    state.stitch.count.record_stream(result_stream)
                    yield state.token, state.stitch.finish()
            else:
                yield state.token, state.stitch.finish()
            state.stitch = None

    try:
        while current is not None:
            compute = torch.cuda.current_stream(device) if cuda else None
            if cuda and copied is not None:
                compute.wait_event(ready)
                copied.record_stream(compute)
            offset = 0
            newly_done = []
            if reference is not None:
                if copied is None:
                    # Retry this allocation once after releasing cached memory.
                    # A second failure costs only the blocks in this buffer.
                    torch.cuda.empty_cache()
                    try:
                        copied, ready = _copy_batch(host, device, copy_stream)
                    except torch.cuda.OutOfMemoryError as exc:
                        for state, *_ in spans:
                            state.error = type(exc)(str(exc))
                    if cuda and copied is not None:
                        compute.wait_event(ready)
                        copied.record_stream(compute)
                newly_done = reference.consume(copied, spans)
                offset = len(host)
            while offset < len(host):
                probs, chunk, error = None, None, None
                active, begin, count, _, _ = next(
                    span for span in spans if span[1] <= offset < span[1] + span[2]
                )
                if active is not size_state:
                    size, size_state = batch, active
                stop = min(offset + size, len(host))
                if size < batch or active.error is not None:
                    stop = min(stop, begin + count)
                # Do not forward an already failed span along with healthy ones.
                for state, begin, *_ in spans:
                    if begin > offset and state.error is not None:
                        stop = min(stop, begin)
                try:
                    if copied is None:
                        chunk, event = _copy_batch(
                            host[offset:stop], device, copy_stream
                        )
                        if cuda:
                            compute.wait_event(event)
                            chunk.record_stream(compute)
                    else:
                        chunk = copied[offset:stop]
                    tokens = [
                        state.token
                        for state, begin, count, _, _ in spans
                        if begin < stop and begin + count > offset
                    ]
                    with (
                        forward_context(tokens) if active.error is None
                        else nullcontext(),
                        (
                            torch.autocast("cuda", dtype=torch.float16)
                            if amp and cuda
                            else nullcontext()
                        ),
                    ):
                        if active.error is None:
                            probs = probabilities(model, chunk)
                except torch.cuda.OutOfMemoryError as exc:
                    if size > 1:
                        size = max(1, size // 2)
                        chunk = None
                        if cuda:
                            torch.cuda.empty_cache()
                        continue
                    error = type(exc)(str(exc))
                except StageTimeout as exc:
                    if remaining is None:
                        error = type(exc)(str(exc))
                    else:
                        expired = [
                            state for state, begin, count, _, _ in spans
                            if begin < stop and begin + count > offset
                            and remaining(state.token) <= 0
                        ]
                        if not expired:
                            raise  # An unrelated alarm cannot identify a shot.
                        for state in expired:
                            state.error = type(exc)(str(exc))
                        # Retry healthy spans; failed spans are drained in order.
                        continue
                except Exception as exc:  # noqa: BLE001 - only this batch's blocks
                    error = type(exc)(str(exc))
                for state, begin, count, first, last in spans:
                    lo, hi = max(begin, offset), min(begin + count, stop)
                    if lo >= hi:
                        continue
                    if error is not None:
                        state.error = error
                    if state.error is None:
                        if state.stitch is None:
                            state.stitch = DeviceStitch(state.n_cols, device)
                        state.stitch.add(
                            probs[lo - offset : hi - offset], first + lo - begin
                        )
                    if last and hi == begin + count:
                        event = torch.cuda.Event() if cuda else None
                        if cuda:
                            event.record(compute)
                        newly_done.append((state, event))
                offset = stop
                del probs, chunk
            # GPU work above is enqueued. Host packing and H2D now overlap it.
            following = next(batches, None)
            if following is not None:
                next_host, next_spans = following
                next_copied, next_ready = stage(next_host)
            # Collect the previous batch on its own stream, beside this forward.
            yield from collect()
            completed = newly_done
            current = following
            if following is not None:
                host, spans = next_host, next_spans
                copied, ready = next_copied, next_ready
        yield from collect()
    finally:
        # Also on early generator close/exception: no pinned owner dies while
        # DMA still reads it, and the caller may safely tear down CUDA state.
        if cuda:
            copy_stream.synchronize()
            result_stream.synchronize()


def infer(model, spec, device, *, batch: int = 96, amp: bool = True,
          compact: bool = False) -> np.ndarray | CompactMask:
    """`(512, T)` spectrogram -> `(2, 512, T)` float32 probabilities.

    Channel 0 is coherent activity, channel 1 transient. The vendored
    network's `forward` returns LOGITS (a 1-tuple of them), so the sigmoid
    is applied here, by `unet.probabilities` - the same wrapper the golden
    test pins - rather than assumed.

    `batch` is tiles per forward pass; 8-32 measured flat at 197 tiles/s on
    a V100S, so the number is about memory, not speed (peak allocation 2.61
    GiB at 8, 10.30 at 32). On `torch.cuda.OutOfMemoryError` the batch is
    HALVED and the same tiles retried, down to 1, and only then does the
    error escape: a mask run is 2,000 shots long and a transient neighbour
    on a shared card must cost a slower shot, not a lost one.

    `amp` uses fp16 autocast, and only on a CUDA device - `torch.autocast`
    with "cuda" on a CPU run would be a no-op with a warning.

    The model is expected to be on `device` already (`load_unet(device=...)`);
    only the tiles are moved, one batch at a time, and the move is INSIDE
    the retry: it allocates the whole batch on the card, so it is as likely
    a place to run out of memory as the forward pass is.
    """
    if compact:
        return _infer_compact(model, spec, device, batch=batch, amp=amp)
    tiles, meta = tile(spec)
    device = torch.device(device)
    x = torch.from_numpy(tiles).unsqueeze(1)          # (n_tiles, 1, 512, 512)
    use_amp = bool(amp) and device.type == "cuda"
    out: list[np.ndarray] = []
    i = 0
    size = max(1, int(batch))
    while i < x.shape[0]:
        chunk = None
        try:
            chunk = _to_device(x[i:i + size], device)
            with (
                torch.autocast("cuda", dtype=torch.float16)
                if use_amp else nullcontext()
            ):
                probs = probabilities(model, chunk)
        except torch.cuda.OutOfMemoryError:
            if size == 1:
                raise
            size = max(1, size // 2)
            chunk = None
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue
        out.append(np.asarray(probs.to("cpu", torch.float32)))
        i += size
    return stitch(np.concatenate(out, axis=0), meta)


# ------------------------------------------------------------- summarising

def band_logpow(raw_logpow_spec, n_bands: int = N_BANDS) -> np.ndarray:
    """`(512, T)` pre-standardisation log-power -> `(n_bands, T)` float16.

    The mask says WHERE the network saw activity; this says how much power
    was there, which is what a later step weights a centroid by and what
    tells a strong mode from a faint one. Sixteen bands and float16 because
    it is stored per channel per pass for 2,000 shots and 0.1 % precision
    on a log quantity is far more than any consumer needs.

    Pass the pre-standardisation array: the z-score has already thrown the
    absolute level away, and two channels' standardised spectrograms are
    not comparable.
    """
    raw = np.asarray(raw_logpow_spec, dtype=np.float32)
    n_bands = int(n_bands)
    if raw.ndim != 2 or raw.shape[0] != N_BINS:
        raise ValueError(f"expected ({N_BINS}, T), got {raw.shape}")
    if n_bands <= 0 or N_BINS % n_bands:
        raise ValueError(f"{n_bands} bands do not divide {N_BINS} rows")
    return raw.reshape(n_bands, N_BINS // n_bands, -1).mean(axis=1).astype(np.float16)


def pack(mask_bool) -> np.ndarray:
    """A boolean mask -> `uint8`, eight pixels per byte, row-major.

    `unpack` needs the shape back; it is in the block's `_meta` (`n_cols`)
    and implied by the 512 rows.
    """
    return np.packbits(np.asarray(mask_bool, dtype=bool).ravel())


def unpack(packed, shape) -> np.ndarray:
    """The inverse of `pack`, given the shape it was packed from."""
    shape = tuple(int(s) for s in shape)
    size = int(np.prod(shape))
    bits = np.unpackbits(np.asarray(packed, dtype=np.uint8))
    if bits.size < size or bits.size - size >= 8:
        raise ValueError(
            f"{np.asarray(packed).size} bytes do not hold a {shape} mask"
        )
    return bits[:size].astype(bool).reshape(shape)


# ------------------------------------------------------------------ storage

@dataclass
class MaskBlock:
    """Everything one `(diag, channel, pass)` produced, before packing.

    Held in memory only long enough to become `block_arrays`' seven keys:
    `coh`, `tra` and `raw_logpow` are `(512, T)` float32 and are exactly
    what must not reach disk at full precision.
    """

    diag: str
    channel: int
    pass_name: str
    coh: np.ndarray
    tra: np.ndarray
    raw_logpow: np.ndarray
    t_s: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.pass_name not in PASS_NAMES:
            raise ValueError(
                f"pass_name {self.pass_name!r} is not one of {PASS_NAMES}"
            )
        shapes = {
            "coh": np.shape(self.coh),
            "tra": np.shape(self.tra),
            "raw_logpow": np.shape(self.raw_logpow),
        }
        for name, shape in shapes.items():
            if len(shape) != 2 or shape[0] != N_BINS:
                raise ValueError(f"{name}: expected ({N_BINS}, T), got {shape}")
        if len({s[1] for s in shapes.values()} | {np.shape(self.t_s)[0]}) != 1:
            raise ValueError(
                f"column counts disagree: {shapes}, t_s {np.shape(self.t_s)}"
            )

    @property
    def prefix(self) -> str:
        """`"mhr_00_wide"` - how this block's seven keys are named."""
        return f"{self.diag}_{int(self.channel):02d}_{self.pass_name}"

    @property
    def n_cols(self) -> int:
        return int(np.shape(self.coh)[1])


def block_arrays(
    block: MaskBlock,
    *,
    thr: float = PROB_THRESHOLD,
    unet_sha256: str,
    compact: CompactMask | None = None,
) -> dict[str, np.ndarray | str]:
    """One block -> the seven keys it is stored as.

    The two probability maps become boolean at `thr` and are packbits-ed;
    what survives of the discarded precision is the two summaries a later
    stage would otherwise recompute over the whole mask - `_row_lit`, the
    fraction of columns each frequency row is coherent in (a row lit almost
    everywhere is receiver pickup, not a mode), and `_col_act`, the
    fraction of rows each column is transient in (the ELM/sawtooth trace).
    `_meta` records the threshold and the checkpoint that produced the
    probabilities, so a file always says what it is.
    """
    if compact is not None and thr != PROB_THRESHOLD:
        raise ValueError("a compact mask has already applied PROB_THRESHOLD")
    coh = None if compact else np.asarray(block.coh) >= thr
    tra = None if compact else np.asarray(block.tra) >= thr
    meta = dict(block.meta)
    meta.update(
        diag=block.diag,
        channel=int(block.channel),
        pass_name=block.pass_name,
        n_cols=block.n_cols,
        tile=TILE,
        overlap=OVERLAP,
        thr=float(thr),
        unet_sha256=str(unet_sha256),
    )
    prefix = block.prefix
    return {
        f"{prefix}_coh_packed": compact.coh_packed if compact else pack(coh),
        f"{prefix}_tra_packed": compact.tra_packed if compact else pack(tra),
        f"{prefix}_row_lit": (compact.row_lit if compact else
                              coh.mean(axis=1).astype(np.float32)),
        f"{prefix}_col_act": (compact.col_act if compact else
                              tra.mean(axis=0).astype(np.float32)),
        f"{prefix}_band_logpow": band_logpow(block.raw_logpow),
        f"{prefix}_t_s": np.asarray(block.t_s, dtype=np.float64),
        f"{prefix}_meta": json.dumps(meta, sort_keys=True),
    }


def _prefix_of(key: str) -> str:
    for suffix in KEY_SUFFIXES:
        if key.endswith(suffix):
            return key[:-len(suffix)]
    raise ValueError(f"{key!r} has no known suffix; expected one of {KEY_SUFFIXES}")


def write_masks(path, shot: int, blocks: Sequence[Mapping[str, Any]], *,
                merge: bool = True) -> Path:
    """Write `<shot>_masks.npz` atomically; by default merge into what is there.

    `blocks` are `block_arrays` dicts. `merge=True` carries forward every
    key whose prefix is NOT among the ones being written, and drops the old
    keys of the ones that are - a re-run of one channel replaces that
    channel whole, rather than leaving a stale `_t_s` beside a new
    `_coh_packed`. This is what lets a SLURM array write one channel per
    task and a re-run fix one channel.

    Atomic through `config.atomic_path`: a reader on this file sees the
    whole old version or the whole new one. The temporary is opened as a
    file object rather than handed to `savez_compressed` by name, because
    that function appends ".npz" to any name that lacks it and would write
    beside the temporary instead of into it.
    """
    path = Path(path)
    fresh: dict[str, Any] = {}
    for block in blocks:
        if not isinstance(block, Mapping):
            raise TypeError(
                f"write_masks takes block_arrays() dicts, got {type(block).__name__}"
            )
        for key, value in block.items():
            _prefix_of(key)
            fresh[key] = value
    prefixes = {_prefix_of(k) for k in fresh}

    out: dict[str, Any] = {}
    if merge and path.exists():
        with np.load(path, allow_pickle=False) as old:
            if SHOT_KEY in old.files and int(old[SHOT_KEY]) != int(shot):
                raise ValueError(
                    f"{path} holds shot {int(old[SHOT_KEY])}, not {int(shot)}"
                )
            for key in old.files:
                if key == SHOT_KEY or _prefix_of(key) in prefixes:
                    continue
                out[key] = old[key]
    out.update(fresh)
    out[SHOT_KEY] = np.int32(shot)
    with atomic_path(path) as tmp, open(tmp, "wb") as handle:
        np.savez_compressed(handle, **out)
    return path


def read_mask(path, key: str):
    """One key out of a masks file; `_meta` comes back as a decoded dict."""
    with np.load(path, allow_pickle=False) as z:
        if key not in z.files:
            raise KeyError(f"{path}: no key {key!r}")
        value = z[key]
        return json.loads(str(value)) if key.endswith("_meta") else value


def list_blocks(path) -> list[str]:
    """The `(diag, channel, pass)` prefixes a masks file holds, sorted."""
    with np.load(path, allow_pickle=False) as z:
        return sorted({_prefix_of(k) for k in z.files if k != SHOT_KEY})
