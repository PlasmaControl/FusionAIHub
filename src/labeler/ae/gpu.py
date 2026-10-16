"""The AE CO2 chain in torch, for a GPU: resample, spectrogram, network windows.

The CPU chain is ``labeler.ae.full.input_signals`` (the raw CO2 record cut to
[0, end] and resampled to 500 kHz by ``labeler.events.review.alfven.resample``),
``labeler.ae.transform.model_input`` (STFT, clip, standardise, band) and
``labeler.models.runners.torch_pt.run_windows`` (the network in 1,024-frame
windows with 128 frames of context either side). Inside a GPU job it leaves the
GPU idle: nearly all of a shot's time is scipy on one core (``resample_poly``
over ~17 M samples a chord, ``ShortTimeFFT.stft``'s Python loop over ~25,000
frames a chord, ``np.percentile``), and the network is a sliver of it (pilot
job 2965454: 2-8 s a shot, GPU 0 %).

This module is the same arithmetic in torch, on whatever device its input
tensor is on. The CPU chain stays the definition; ``tests/labeler/test_ae_gpu.py``
checks each function against it and states the tolerances.

- :func:`resample` is ``alfven.resample``: the rate from the record's time span,
  the ratio ``Fraction(RATE_HZ / rate).limit_denominator(100)``
  (:func:`resample_ratio`), then :func:`resample_poly`, which is
  ``scipy.signal.resample_poly(x, up, down, axis=-1)``: scipy's own FIR
  (:func:`fir`: ``firwin`` with a Kaiser(5) window, cast to the input's float
  dtype and multiplied by ``up`` in it, exactly as resample_poly builds it), the
  same alignment (output ``i`` centred on upsampled sample ``i * down``), the
  same output length and dtype, and the same sums: ``upfirdn`` adds each
  output's products one at a time in the input's dtype, oldest sample first,
  and so does this, one tap at a time across all outputs of a phase, so the
  result is equal bit for bit (a float32 sum in another order is not: the
  network turns its last-bit differences into probability differences near
  1e-3).
- :func:`spectrogram` is ``transform.spectrogram`` (``compute_stft`` per chord):
  ``ShortTimeFFT(hann 1024, hop 128).stft``'s framing - the zero padding its
  ``p_min``/``p_max`` imply, so a record of ``n`` samples gives
  ``transform.n_frames(n)`` frames - its window, its half-window roll before the
  rfft (``phase_shift=0``), then ``|.|``, ``log1p`` and the DC bin dropped, all
  in float64; then the clip to the array's own 1st and 99th percentiles by
  numpy's default ``linear`` method (:func:`percentiles`: one sort, since
  ``torch.quantile`` refuses inputs this large, and numpy's own index and
  interpolation arithmetic), cast to float32. Shape ``(C, 512, frames)``, before
  standardising.
- :func:`standardise`, :func:`restrict_to_band` and :func:`model_input` are
  transform's: per-chord mean and std over all 512 bins and every frame
  (accumulated in float64 and rounded to float32, where numpy accumulates in
  float32), ``(x - mean) / (std + STD_EPS)`` in float32, then bins 164:512. The
  order is transform's: statistics over the full 512 bins, then the band.
- :func:`run_windows` is ``torch_pt.run_windows``: the same windows and the same
  interior slices kept, the full-size windows (``window + 2 * context`` frames)
  evaluated in batches (:func:`max_batch` sizes them to the GPU's memory) and
  the shorter edge windows one at a time, as there. The network itself is the
  caller's, on the caller's device; keep cuDNN in IEEE float32 (no TF32) to
  match the CPU (``scripts/labeler/ae_gold_extend.py``'s ``load`` does).

For a loader, :func:`read_co2` is ``full.input_signals`` up to the resample (the
record cut to the resampler's span; only that span is read when the record
reaches the window's end) and returns numpy and plain numbers, so a DataLoader
worker can run it; :func:`input_signals` does the rest (resample, cut to
[0, end]) on the device. Frame times stay on the CPU:
``labeler.ae.full.frame_times``.
"""
from __future__ import annotations

import math
from fractions import Fraction
from functools import cache, lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from scipy import signal

from ..config import Paths
from ..events import raw
from ..events.review import alfven
from . import full
from . import transform as tr
from .labels import BAND_HI_BIN, BAND_LO_BIN, N_BINS

#: The share of a GPU's memory :func:`max_batch` lets one batch of windows take.
GPU_MEMORY_SHARE = 0.5
#: :func:`max_batch`'s ceiling on a GPU.
MAX_BATCH = 64
#: :func:`max_batch`'s answer off a GPU.
CPU_BATCH = 8


def _as_tensor(x, device=None, dtype=None) -> torch.Tensor:
    if not torch.is_tensor(x):
        x = np.ascontiguousarray(x)
        x = torch.from_numpy(x if x.flags.writeable else x.copy())  # torch warns on a read-only array
    return x.to(device=device if device is not None else x.device, dtype=dtype or x.dtype)


# ------------------------------------------------------------------ resampling


def resample_ratio(n_samples: int, first_ms: float, last_ms: float) -> tuple[int, int, float]:
    """``alfven.resample``'s ratio for a record of ``n_samples`` from ``first_ms``
    to ``last_ms``: ``(up, down, output rate in Hz)``, by the same float64 lines
    (the rate from the span, not a median step)."""
    first, last = np.float64(first_ms), np.float64(last_ms)
    rate = (int(n_samples) - 1) / ((last - first) / 1000)
    ratio = Fraction(alfven.RATE_HZ / rate).limit_denominator(100)
    return ratio.numerator, ratio.denominator, rate * ratio.numerator / ratio.denominator


@cache
def _fir(up: int, down: int, dtype: str) -> np.ndarray:
    max_rate = max(up, down)
    h = signal.firwin(2 * (10 * max_rate) + 1, 1.0 / max_rate, window=("kaiser", 5.0))
    h = np.asarray(h, dtype=dtype)  # resample_poly: "match dtype of x"
    h *= up
    return h


def fir(up: int, down: int, dtype=np.float32) -> np.ndarray:
    """The filter ``scipy.signal.resample_poly`` builds for a reduced ``up``/``down``
    and an input of float ``dtype``: ``firwin(2 * half_len + 1, 1 / max(up, down),
    window=("kaiser", 5.0))`` with ``half_len = 10 * max(up, down)``, cast to
    ``dtype``, times ``up`` in ``dtype``. A fresh copy."""
    return _fir(int(up), int(down), np.dtype(dtype).name).copy()


def resample_poly(x, up: int, down: int) -> torch.Tensor:
    """``scipy.signal.resample_poly(x, up, down, axis=-1)`` (its defaults: Kaiser(5)
    FIR, zero padding) of a float32 or float64 tensor ``x`` ``(..., n)``, on ``x``'s
    device, equal to it bit for bit.

    Output ``i`` is ``sum_j h[j] * x_up[i * down + half_len - j]``, where ``x_up``
    is ``x`` with ``up - 1`` zeros after each sample and zeros outside it: what
    ``upfirdn`` computes once resample_poly has padded ``h`` and cut the output.
    Split by output phase ``i mod up``, each phase reads every ``up``-th tap
    against consecutive samples of ``x`` itself. ``upfirdn`` accumulates each
    output's products in ``x``'s dtype from the oldest sample to the newest, a
    multiply and an add each rounded; so does this, one tap at a time over all of a
    phase's outputs (a separate multiply and add, never a fused one).
    """
    if up != int(up) or down != int(down):
        raise ValueError("up and down must be integers")
    up, down = int(up), int(down)
    if up < 1 or down < 1:
        raise ValueError("up and down must be >= 1")
    g = math.gcd(up, down)
    up, down = up // g, down // g
    x = _as_tensor(x)
    if x.dtype not in (torch.float32, torch.float64):
        raise TypeError(f"expected a float32 or float64 tensor, got {x.dtype}")
    if up == down == 1:
        return x.clone()
    n_in = int(x.shape[-1])
    n_out = n_in * up
    n_out = n_out // down + bool(n_out % down)
    if n_out == 0:
        return x.new_empty(x.shape)
    h = fir(up, down, np.float32 if x.dtype == torch.float32 else np.float64)
    half_len = (h.size - 1) // 2
    phases = []
    for q in range(min(up, n_out)):
        r = (q * down + half_len) % up  # taps h[r], h[r + up], ... meet real samples
        base = (q * down + half_len - r) // up  # x index tap r meets at the phase's first output
        taps = h[r::up]
        count = (n_out - q + up - 1) // up  # outputs q, q + up, ... below n_out
        first = base - (taps.size - 1)  # the oldest x index the phase's first output reads
        phases.append((q, taps, count, first))
    pad_lo = max(0, max(-first for *_, first in phases))
    reach = max(first + (count - 1) * down + taps.size for _, taps, count, first in phases)
    length = pad_lo + max(reach, n_in)
    length += -length % down
    rows = x.reshape(-1, n_in)
    padded = F.pad(rows, (pad_lo, length - pad_lo - n_in))
    # by_rem[:, s, i] = padded[:, i * down + s]: the samples one tap meets across a
    # phase's outputs, contiguous.
    by_rem = padded.view(rows.shape[0], length // down, down).transpose(1, 2).contiguous()
    del padded
    out = torch.empty((rows.shape[0], n_out), dtype=x.dtype, device=x.device)
    for q, taps, count, first in phases:
        acc = torch.zeros((rows.shape[0], count), dtype=x.dtype, device=x.device)
        prod = torch.empty_like(acc)
        n_taps = taps.size
        for j in range(n_taps):  # the j-th oldest sample of every output meets tap n_taps - 1 - j
            at = first + pad_lo + j
            torch.mul(by_rem[:, at % down, at // down:at // down + count], float(taps[n_taps - 1 - j]), out=prod)
            acc.add_(prod)
        out[:, q::up] = acc
    return out.reshape(*x.shape[:-1], n_out)


def resample(time_ms, chords, *, device=None) -> tuple[torch.Tensor, float]:
    """``alfven.resample`` on ``device`` (default: where ``chords`` is): ``chords``
    ``(C, n)`` at the rate their ``time_ms`` span implies, cast to float32 and
    resampled to ``alfven.RATE_HZ``; the float32 tensor ``(C, m)`` and its exact
    rate in Hz."""
    time_ms = np.asarray(time_ms, dtype=np.float64)
    up, down, fs = resample_ratio(len(time_ms), time_ms[0], time_ms[-1])
    return resample_poly(_as_tensor(chords, device, torch.float32), up, down), fs


# ------------------------------------------------------------------ spectrogram


@lru_cache(maxsize=1)
def _stft() -> signal.ShortTimeFFT:
    return tr.short_time_fft()  # fs = 1: only the framing, the window and the FFT are used


_WINDOWS: dict[str, torch.Tensor] = {}


def _window(device: torch.device) -> torch.Tensor:
    key = str(device)
    if key not in _WINDOWS:
        _WINDOWS[key] = torch.from_numpy(np.array(_stft().win, dtype=np.float64)).to(device)
    return _WINDOWS[key]


def percentiles(values: torch.Tensor, q=(tr.CLIP_LOW, tr.CLIP_HIGH)) -> torch.Tensor:
    """``np.percentile(values, q)`` over all of ``values`` by numpy's default
    ``linear`` method, on ``values``' device: one sort, then numpy's arithmetic -
    the virtual index ``(n - 1) * (q / 100)``, its floor and fractional part
    ``g``, and ``_lerp``: ``a + (b - a) * g``, or ``b - (b - a) * (1 - g)`` when
    ``g >= 0.5``."""
    flat = values.reshape(-1)
    n = int(flat.numel())
    if n == 0:
        raise ValueError("percentiles of an empty array")
    ordered = torch.sort(flat).values
    out = []
    for v in (n - 1) * np.true_divide(np.asarray(q, dtype=np.float64), np.float64(100)):
        if v >= n - 1:  # numpy takes the last element
            out.append(ordered[-1])
            continue
        lo = int(np.floor(v))
        g = float(v - lo)
        a, b = ordered[lo], ordered[lo + 1]
        d = b - a
        out.append(b - d * (1.0 - g) if g >= 0.5 else a + d * g)
    return torch.stack(out)


def compute_stft(x) -> torch.Tensor:
    """One chord ``(n,)`` -> ``(512, frames)`` float32: ``transform.compute_stft``."""
    x = _as_tensor(x)
    if x.ndim != 1:
        raise ValueError(f"expected one channel, got shape {tuple(x.shape)}")
    stft = _stft()
    n = int(x.shape[0])
    least = stft.m_num - stft.m_num_mid
    if n < least:  # ShortTimeFFT.stft's own refusal, for the (1, n) input compute_stft gives it
        raise ValueError(f"of axis=-1 of (1, {n}) must be >= ceil(m_num/2) = {least}!")
    p0, p1 = stft.p_min, stft.p_max(n)
    count = p1 - p0
    k0 = p0 * stft.hop - stft.m_num_mid  # first sample of the first frame
    k1 = k0 + count * stft.hop + stft.m_num
    padded = F.pad(x[max(k0, 0):min(k1, n)].to(torch.float64), (-min(k0, 0), max(k1 - n, 0)))
    seg = padded.unfold(0, stft.m_num, stft.hop)[:count] * _window(x.device)
    if stft.phase_shift is not None:
        seg = torch.roll(seg, -((stft.phase_shift + stft.m_num_mid) % stft.m_num), dims=1)
    mag = torch.fft.rfft(seg, n=stft.mfft, dim=1).abs()
    del seg
    mag = torch.log1p(mag, out=mag)[:, 1:]  # drop DC
    lo, hi = percentiles(mag)
    return torch.clamp(mag, lo, hi).to(torch.float32).T.contiguous()


def spectrogram(signals) -> torch.Tensor:
    """``(C, N)`` waveforms -> ``(C, 512, frames)`` float32, one chord at a time:
    ``transform.spectrogram``."""
    signals = _as_tensor(signals)
    if signals.ndim != 2:
        raise ValueError(f"expected (channels, samples), got {tuple(signals.shape)}")
    return torch.stack([compute_stft(row) for row in signals], dim=0)


def standardise(raw_spec) -> torch.Tensor:
    """Per-chord z-score over the whole ``(bins, frames)`` array: ``transform.standardise``."""
    spec = _as_tensor(raw_spec, dtype=torch.float32)
    if spec.ndim != 3:
        raise ValueError(f"expected (channels, bins, frames), got {tuple(spec.shape)}")
    mean = torch.empty(spec.shape[0], dtype=torch.float64, device=spec.device)
    std = torch.empty_like(mean)
    for c in range(spec.shape[0]):  # one chord's float64 copy at a time
        wide = spec[c].to(torch.float64)
        mean[c] = wide.mean()
        std[c] = (wide - mean[c]).square_().mean().sqrt_()
    mean = mean.to(torch.float32)[:, None, None]
    std = std.to(torch.float32)[:, None, None]
    return (spec - mean) / (std + tr.STD_EPS)


def restrict_to_band(spec) -> torch.Tensor:
    """Bins 164:512 - the 348-bin AE band: ``transform.restrict_to_band``."""
    spec = _as_tensor(spec)
    if spec.shape[-2] != N_BINS:
        raise ValueError(f"expected {N_BINS} bins, got {spec.shape[-2]}")
    return spec[..., BAND_LO_BIN:BAND_HI_BIN, :]


def model_input(signals) -> torch.Tensor:
    """``(C, N)`` CO2 waveforms -> the ``(C, 348, frames)`` float32 the model eats:
    ``transform.model_input``."""
    return restrict_to_band(standardise(spectrogram(signals)))


# ------------------------------------------------------------------ the input


def read_co2(shot: int, *, paths: Paths, end_ms: float | None = None) -> dict:
    """``full.input_signals`` up to the resample, for a loader worker: the four CO2
    chords from the corpus or the raw cache (``raw.raw_signal``), cut to
    ``[-MARGIN_MS, end + MARGIN_MS]`` with ``end = min(record end, end_ms)``
    (``end_ms`` defaults to ``full.CROP_END_MS``).

    The same samples as input_signals' whole read, read faster: first only the
    span ``[-MARGIN_MS, end_ms + MARGIN_MS]`` (``raw_signal``'s ``t_range``, an h5py
    slice of the time-ordered record). When that read reaches ``end_ms``, the
    record does too, so ``end = end_ms`` and its samples are exactly those
    input_signals keeps. When it does not (a record that stops before ``end_ms``),
    the record is read whole and cut by input_signals' own mask. A cut that is one
    run of samples is copied as a slice, not gathered by the mask.

    Returns plain numbers and one array: ``chords`` ``(4, n)`` float32, the
    ``n_samples``, ``first_ms`` and ``last_ms`` of the cut record (all
    ``alfven.resample`` reads of its time vector), ``end_ms`` and ``shot``.
    """
    end_ms = full.CROP_END_MS if end_ms is None else float(end_ms)
    span = (-full.MARGIN_MS, end_ms + full.MARGIN_MS)
    try:
        a = raw.raw_signal(shot, "co2", paths=paths, t_range=span)
    except raw.WindowEmptyError:
        a = None  # nothing in the span: let the whole record say what input_signals would
    if a is None or float(a.x[-1]) < end_ms:
        a = raw.raw_signal(shot, "co2", paths=paths)
    if len(a.y) != len(full.CHANNEL_NAMES):
        raise ValueError(f"shot {int(shot)}: {len(a.y)} CO2 chords, not 4")
    end = min(float(a.x[-1]), end_ms)
    keep = (a.x >= -full.MARGIN_MS) & (a.x <= end + full.MARGIN_MS)
    lo = int(keep.argmax())
    if not keep[lo]:  # input_signals would fail on time_ms[0] (an IndexError there)
        raise ValueError(f"shot {int(shot)}: no CO2 sample in [{-full.MARGIN_MS}, {end + full.MARGIN_MS}] ms")
    hi = keep.size - int(keep[::-1].argmax())
    if keep[lo:hi].all():
        chords, times = a.y[:, lo:hi], a.x[lo:hi]
    else:
        chords, times = a.y[:, keep], a.x[keep]
    return {
        "shot": int(shot),
        "chords": np.ascontiguousarray(chords, dtype=np.float32),
        "n_samples": int(times.size),
        "first_ms": float(times[0]),
        "last_ms": float(times[-1]),
        "end_ms": end,
    }


def input_signals(window: dict, *, device=None) -> tuple[torch.Tensor, float, float]:
    """The rest of ``full.input_signals`` for a :func:`read_co2` result, on
    ``device``: resampled to 500 kHz and cut to ``[0, end]`` ms. Returns the float32
    chords ``(4, m)``, the time of the first sample (ms) and the exact rate (Hz);
    input_signals returns the same samples as float64."""
    up, down, fs = resample_ratio(window["n_samples"], window["first_ms"], window["last_ms"])
    x = resample_poly(_as_tensor(window["chords"], device, torch.float32), up, down)
    t = np.float64(window["first_ms"]) + np.arange(x.shape[1]) * 1000 / fs
    end = window["end_ms"]
    inside = np.flatnonzero((t >= 0) & (t <= end))
    if inside.size == 0:
        raise ValueError(f"shot {int(window['shot'])}: no CO2 sample in [0, {end}] ms")
    lo, hi = int(inside[0]), int(inside[-1]) + 1
    return x[:, lo:hi], float(t[lo]), float(fs)


# ------------------------------------------------------------------ the network


def window_plan(n: int, window: int, context: int) -> list[tuple[int, int, int, int]]:
    """``torch_pt.run_windows``'s windows over ``n`` frames: ``(start, stop, lo, hi)``,
    frames ``[lo, hi)`` evaluated and ``[start, stop)`` kept."""
    if window <= 0 or context < 0:
        raise ValueError(f"window must be positive and context non-negative; got {window}, {context}")
    plan = []
    for start in range(0, n, window):
        stop = min(start + window, n)
        plan.append((start, stop, max(start - context, 0), min(stop + context, n)))
    return plan


def max_batch(module: torch.nn.Module, frames: int, bins: int, device) -> int:
    """How many ``frames x bins`` windows one forward pass takes: ``MAX_BATCH`` at
    most, and on a GPU no more than fit in ``GPU_MEMORY_SHARE`` of its memory at
    three float32 copies of the first convolution's output per window (that
    output, the batch norm's, and room for the next layer; the input and the later,
    pooled layers are small beside them). Off a GPU, ``CPU_BATCH``."""
    device = torch.device(device)
    if device.type != "cuda":
        return CPU_BATCH
    conv = next((m for m in module.modules() if isinstance(m, torch.nn.Conv2d)), None)
    channels = conv.out_channels if conv is not None else 64
    per_window = 3 * channels * frames * bins * 4
    total = torch.cuda.get_device_properties(device).total_memory
    return int(max(1, min(MAX_BATCH, GPU_MEMORY_SHARE * total // per_window)))


def run_windows(
    module: torch.nn.Module,
    x,
    *,
    window: int,
    context: int,
    batch: int | None = None,
) -> torch.Tensor:
    """``torch_pt.run_windows`` on ``x``'s device: a ``(C, T, F)`` record -> ``(T, out)``.

    The same windows and the same kept slices. The windows of full size
    (``window + 2 * context`` frames) go through the network ``batch`` at a time
    (default :func:`max_batch`); the edge windows, shorter because they meet the
    record's ends, one at a time as in run_windows. Each window is an independent
    sample of the batch (the network has no cross-sample layer in eval mode), so
    batching changes only the order of floating-point sums inside a kernel.
    """
    x = _as_tensor(x, dtype=torch.float32)
    if x.ndim != 3:
        raise ValueError(f"expected (channels, frames, bins), got {tuple(x.shape)}")
    n = int(x.shape[1])
    plan = window_plan(n, window, context)
    if not plan:
        raise ValueError("no frames to evaluate")
    size = window + 2 * context
    whole = [p for p in plan if p[3] - p[2] == size]
    edges = [p for p in plan if p[3] - p[2] != size]
    if batch is None:
        batch = max_batch(module, size, int(x.shape[2]), x.device)
    out: torch.Tensor | None = None

    def keep(got: torch.Tensor, start: int, stop: int, lo: int) -> None:
        nonlocal out
        if out is None:
            out = got.new_empty((n, *got.shape[1:]))
        out[start:stop] = got[start - lo:start - lo + (stop - start)]

    with torch.inference_mode():
        for i in range(0, len(whole), max(1, int(batch))):
            group = whole[i:i + max(1, int(batch))]
            got = module(torch.stack([x[:, lo:hi, :] for _, _, lo, hi in group]))
            for b, (start, stop, lo, _) in enumerate(group):
                keep(got[b], start, stop, lo)
        for start, stop, lo, hi in edges:
            keep(module(x[:, lo:hi, :].unsqueeze(0))[0], start, stop, lo)
    return out
