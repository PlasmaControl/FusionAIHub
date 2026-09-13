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


def infer(model, spec, device, *, batch: int = 96, amp: bool = True) -> np.ndarray:
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
    coh = np.asarray(block.coh) >= thr
    tra = np.asarray(block.tra) >= thr
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
        f"{prefix}_coh_packed": pack(coh),
        f"{prefix}_tra_packed": pack(tra),
        f"{prefix}_row_lit": coh.mean(axis=1).astype(np.float32),
        f"{prefix}_col_act": tra.mean(axis=0).astype(np.float32),
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
