"""Self-contained spectrogram background subtraction (NO external-repo import).

Duplicates the *concept* used in baseline-correction pipelines: estimate the
smooth per-frequency envelope B (the background) and take the residual R = S - B
(the sharp deviations: coherent modes + transients). Recombine exactly with
S = B + R. Working in R-space puts thin modes on a flat background so a codec's
MAE / code budget is no longer dominated by the bright low-frequency envelope.

Baseline estimator = a large Gaussian low-pass ALONG FREQUENCY (fast, separable,
scipy-only). A thin mode (1-2 freq bins) barely moves a wide-sigma Gaussian, so
it survives in the residual; the broad spectral envelope is removed. This is the
simple version — swap in a peak-robust fit (median / grey-opening / ALS) later if
the concept holds. Spectrograms here are already log-standardized, so the residual
is ADDITIVE (S = B + R), not the relative (S-B)/B used on raw log-magnitude.
"""
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d


def baseline_residual(X, sigma: float = 8.0, freq_axis: int = -2):
    """Split a spectrogram into (baseline B, residual R) with R = X - B.

    X : torch.Tensor or np.ndarray, shape (..., F, T) — F on ``freq_axis``.
    sigma : Gaussian std (in frequency bins) of the smooth-envelope low-pass.
    Returns (B, R) matching X's type; torch tensors are returned on CPU float32.
    """
    was_torch = torch.is_tensor(X)
    arr = (X.detach().cpu().numpy() if was_torch else np.asarray(X)).astype(np.float32)
    ax = freq_axis if freq_axis >= 0 else arr.ndim + freq_axis
    B = gaussian_filter1d(arr, sigma=sigma, axis=ax, mode="nearest")
    R = arr - B
    if was_torch:
        return torch.from_numpy(B), torch.from_numpy(R)
    return B, R


# --- GPU version, numerically identical to the scipy call above -------------
# The residual codecs were trained with ``baseline_residual`` (scipy
# ``gaussian_filter1d(mode="nearest")``, truncate=4.0). To run the SAME
# split inside the training/eval forward pass without a per-batch CPU
# round-trip, this reproduces that exact operator on-device: a depthwise
# Gaussian conv along the frequency axis with edge-replicate padding
# (``mode="nearest"`` == replicate) and radius ``int(4*sigma + 0.5)``. For a
# symmetric kernel correlation == convolution, so ``conv1d`` matches scipy's
# ``correlate1d`` to float precision (verified <1e-5 max-abs on real ECE).
_GAUSS_CACHE: dict = {}


def _gauss_kernel(sigma: float, device, dtype):
    key = (round(float(sigma), 4), device, dtype)
    kr = _GAUSS_CACHE.get(key)
    if kr is None:
        r = int(4.0 * float(sigma) + 0.5)
        x = torch.arange(-r, r + 1, device=device, dtype=dtype)
        k = torch.exp(-0.5 * (x / float(sigma)) ** 2)
        k = (k / k.sum()).view(1, 1, -1)
        _GAUSS_CACHE[key] = kr = (k, r)
    return kr


def baseline_residual_torch(X: torch.Tensor, sigma: float = 8.0, freq_axis: int = -2):
    """On-device (B, R) split — the ``baseline_residual`` operator, no CPU hop.

    X : (..., F, T) float tensor on any device. Returns (B, R) same shape/device/dtype.
    """
    ax = freq_axis % X.dim()
    Xf = X.movedim(ax, -1)                        # (..., F) with F last
    shp = Xf.shape
    k, r = _gauss_kernel(sigma, X.device, Xf.dtype)
    xr = Xf.reshape(-1, 1, shp[-1])               # (N, 1, F)
    xr = F.pad(xr, (r, r), mode="replicate")      # scipy mode="nearest"
    B = F.conv1d(xr, k).reshape(shp).movedim(-1, ax)
    return B, X - B


def _box_avg(x: torch.Tensor, wf: int, wt: int) -> torch.Tensor:
    """Same-size neighbourhood average over the last two axes (the expectation E[.]_W)."""
    x4 = x.reshape(-1, 1, x.shape[-2], x.shape[-1])
    x4 = F.pad(x4, (wt // 2, wt - 1 - wt // 2, wf // 2, wf - 1 - wf // 2), mode="replicate")
    x4 = F.avg_pool2d(x4, kernel_size=(wf, wt), stride=1)
    return x4.reshape(x.shape)


def coherence_denoise(S: torch.Tensor, win_f: int = 3, win_t: int = 3, power: float = 1.0):
    """Rung-0 η-removal: multichannel cross-power coherence gate (TRANSPARENT, no training).

    S : complex STFT ``(C, F, T)`` (all channels of ONE modality). Returns
    ``(denoised_magnitude (C,F,T), coherence_gate g (F,T))``.

    A coherent mode adds in-phase across channels (|Σ_c S_c|² ≈ C·Σ|S_c|²); incoherent
    per-channel noise η cancels (|Σ_c S_c|² ≈ Σ|S_c|²). The coherent-power FRACTION
        g = (E[|Σ_c S_c|²] − E[Σ_c|S_c|²]) / ((C−1)·E[Σ_c|S_c|²])   in [0,1]
    (E[.] = box average over a (win_f,win_t) neighbourhood — the cross-power expectation,
    the R_xR_y+I_xI_y mechanism) is ~1 on coherent modes, ~0 on η. Denoised magnitude =
    |S_c|·g^power. Pure down-weighting by a fixed formula → CANNOT hallucinate modes.
    """
    C = S.shape[0]
    mag2 = (S.real ** 2 + S.imag ** 2)                      # (C,F,T) per-channel power
    sumS = S.sum(0)                                         # (F,T) complex coherent sum
    e_sum2 = _box_avg(sumS.real ** 2 + sumS.imag ** 2, win_f, win_t)   # E[|Σ S|²]
    e_powsum = _box_avg(mag2.sum(0), win_f, win_t)          # E[Σ|S|²]
    g = (e_sum2 - e_powsum) / ((C - 1) * e_powsum + 1e-12)
    g = g.clamp(0.0, 1.0)
    return mag2.sqrt() * g.pow(power).unsqueeze(0), g


def channel_coherent_denoise(S: torch.Tensor, k_chan: int = 2, win_f: int = 1, win_t: int = 1):
    """Rung-0b η-removal: LOCAL adjacent-channel coherent integration (TRANSPARENT, no train).

    S : complex STFT ``(C, F, T)``. Returns ``(denoised_magnitude (C,F,T), None)``.

    Global coherence fails for ECE because a mode has RADIAL PHASE STRUCTURE (distant
    channels are out of phase). But ADJACENT channels (neighbouring radii) see the mode
    ~in-phase, while per-channel η is independent. A complex moving-average over the
    +-k_chan neighbours therefore ADDS the coherent mode (amplitude preserved) and
    AVERAGES DOWN incoherent η (~1/sqrt(K)). Optional (win_f,win_t) complex box-avg first.
    Amplitude-preserving (in-phase sum) → passes the A1 amplitude check, unlike a gate.
    """
    if win_f > 1 or win_t > 1:
        S = torch.complex(_box_avg(S.real, win_f, win_t), _box_avg(S.imag, win_f, win_t))
    K = 2 * k_chan + 1
    # complex moving-average along the channel axis (dim 0), replicate-padded edges
    Sr = S.real.permute(1, 2, 0).reshape(-1, 1, S.shape[0])       # (F*T, 1, C)
    Si = S.imag.permute(1, 2, 0).reshape(-1, 1, S.shape[0])
    Sr = F.pad(Sr, (k_chan, k_chan), mode="replicate"); Si = F.pad(Si, (k_chan, k_chan), mode="replicate")
    w = torch.ones(1, 1, K, device=S.device, dtype=S.real.dtype) / K
    ar = F.conv1d(Sr, w).reshape(S.shape[1], S.shape[2], S.shape[0]).permute(2, 0, 1)
    ai = F.conv1d(Si, w).reshape(S.shape[1], S.shape[2], S.shape[0]).permute(2, 0, 1)
    return torch.sqrt(ar * ar + ai * ai), None


def raw_stft_complex(sig: torch.Tensor, n_fft: int = 1024, hop: int = 256, drop_dc: bool = True):
    """Raw ``(C, N)`` time-series -> complex STFT ``(C, F, T)`` (hann, matches the loader).
    DC bin dropped to mirror the dataset. The phase the pipeline normally discards at |·|."""
    w = torch.hann_window(n_fft, device=sig.device, dtype=sig.dtype)
    S = torch.stft(sig, n_fft=n_fft, hop_length=hop, window=w, return_complex=True, center=True)
    return S[:, 1:, :] if drop_dc else S


def smooth_time_mag(X: torch.Tensor, n_frames: int, time_axis: int = -1) -> torch.Tensor:
    """Temporal moving-average of a (magnitude) spectrogram along the TIME axis.

    X : (..., F, T) tensor. Averages ``n_frames`` adjacent STFT frames with a
    stride-1 'same'-length window (replicate-padded edges), so the output keeps the
    original T. Purpose: coherent ridges (tearing modes / AEs) survive frame
    averaging; STFT-phase/realization speckle (which decorrelates in ~1 frame, and
    which a 0.5 ms shift scrambles) is suppressed. ``n_frames<=1`` is a no-op.

    This is the operational form of "encode statistics, not realizations": running a
    codec on ``smooth_time_mag(R, N)`` makes its codes shift-stable (the audit gate).
    Complementary to ``baseline_residual`` (which smooths along FREQUENCY, not time).
    """
    n = int(n_frames)
    if n <= 1:
        return X
    ax = time_axis % X.dim()
    Xt = X.movedim(ax, -1)                        # (..., T) with T last
    shp = Xt.shape
    xr = Xt.reshape(-1, 1, shp[-1])               # (M, 1, T)
    pad_l = n // 2
    pad_r = n - 1 - pad_l
    xr = F.pad(xr, (pad_l, pad_r), mode="replicate")
    w = torch.ones(1, 1, n, device=X.device, dtype=xr.dtype) / n
    out = F.conv1d(xr, w).reshape(shp).movedim(-1, ax)
    return out
