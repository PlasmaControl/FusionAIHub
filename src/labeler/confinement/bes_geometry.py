"""Where the BES channels look, and which 6 x 8 block of the array covers the pedestal.

Gill et al. (2024) train on a 6 x 8 block of the 8 x 8 BES array ("8 x 8 shots truncated
to the first 6 rows") whose radial coverage reaches the edge; other layouts or missing
edge channels cost up to 0.6 F1 in their ablation. The array's layout differs from
shot to shot, so this module places each channel on the plasma from the shot's own
record: ``\\BES::BES_R`` and ``\\BES::BES_Z`` (cm, one entry per channel,
``channel = row * 8 + column``) against the EFIT flux map, as the normalised poloidal
flux ``psi_N`` (0 on axis, 1 on the separatrix).

The block is chosen by geometry alone, never by a classifier score:

* a row (8 channels) covers the pedestal when at least ``ROW_MIN_CHANNELS`` of its
  channels have ``psi_N`` in ``PEDESTAL_BAND`` (the pedestal and the start of its
  gradient, the region the edge fluctuations are read from);
* the block is 6 consecutive rows (the 3D convolution needs neighbouring rows to be
  neighbours) starting at row 0, 1 or 2: the start that covers the most rows, the
  lowest start on a tie (so the paper's "first six rows" stands unless a row is
  displaced from the pedestal);
* the shot is dropped when the block does not reach the separatrix: its outermost
  channel has ``psi_N`` below ``REACH_PSIN``.

The numbers are the physics priors, fixed before any score was looked at.
"""

from __future__ import annotations

import numpy as np

ROWS = 8
COLUMNS = 8
BLOCK_ROWS = 6
#: Normalised flux of the pedestal and the foot of its gradient: a channel inside it
#: reads the edge fluctuations the paper's labels are about.
PEDESTAL_BAND = (0.85, 1.0)
ROW_MIN_CHANNELS = 2
#: The block must reach the last closed flux surface (EFIT puts it within a few mm).
REACH_PSIN = 0.98


def bilinear(
    psirz: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    r_pts: np.ndarray,
    z_pts: np.ndarray,
) -> np.ndarray:
    """Bilinear interpolation of ``(T, nz, nr)`` flux at points ``(T or 1, P)``.

    A point outside the grid takes the nearest edge value (the BES sits well inside).
    """
    t = psirz.shape[0]
    r_pts = np.broadcast_to(r_pts, (t, r_pts.shape[-1]))
    z_pts = np.broadcast_to(z_pts, (t, z_pts.shape[-1]))
    fr = np.clip(
        (r_pts - r_grid[0]) / (r_grid[-1] - r_grid[0]) * (len(r_grid) - 1),
        0,
        len(r_grid) - 1,
    )
    fz = np.clip(
        (z_pts - z_grid[0]) / (z_grid[-1] - z_grid[0]) * (len(z_grid) - 1),
        0,
        len(z_grid) - 1,
    )
    ir = np.minimum(fr.astype(int), len(r_grid) - 2)
    iz = np.minimum(fz.astype(int), len(z_grid) - 2)
    wr, wz = fr - ir, fz - iz
    rows = np.arange(t)[:, None]
    return (
        psirz[rows, iz, ir] * (1 - wr) * (1 - wz)
        + psirz[rows, iz, ir + 1] * wr * (1 - wz)
        + psirz[rows, iz + 1, ir] * (1 - wr) * wz
        + psirz[rows, iz + 1, ir + 1] * wr * wz
    )


def orient_psirz(
    psirz: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    ssimag: np.ndarray,
    rmaxis: np.ndarray,
    zmaxis: np.ndarray,
) -> tuple[np.ndarray, float]:
    """``psirz`` as ``(T, nz, nr)``, and the median |psi(axis) - ssimag| that decided.

    The MDSplus record is square on DIII-D's 65 x 65 grid, so its axis order is read off
    the flux itself: the right order gives ``ssimag`` at the magnetic axis.
    """
    best, best_err = psirz, np.inf
    for cand in (psirz, psirz.transpose(0, 2, 1)):
        at_axis = bilinear(cand, r_grid, z_grid, rmaxis[:, None], zmaxis[:, None])[:, 0]
        err = float(np.nanmedian(np.abs(at_axis - ssimag)))
        if err < best_err:
            best, best_err = cand, err
    return best, best_err


def channel_psin(
    psirz: np.ndarray,
    r_grid: np.ndarray,
    z_grid: np.ndarray,
    ssimag: np.ndarray,
    ssibry: np.ndarray,
    rmaxis: np.ndarray,
    zmaxis: np.ndarray,
    r_cm: np.ndarray,
    z_cm: np.ndarray,
) -> tuple[np.ndarray, float]:
    """``psi_N`` of each channel at each EFIT time, ``(T, P)``, and the axis residual.

    ``r_cm`` and ``z_cm`` are the channels' positions as ``\\BES::BES_R`` and ``BES_Z``
    give them (cm); the flux map is in metres.
    """
    flux, residual = orient_psirz(psirz, r_grid, z_grid, ssimag, rmaxis, zmaxis)
    psi = bilinear(flux, r_grid, z_grid, r_cm[None, :] / 100.0, z_cm[None, :] / 100.0)
    span = (ssibry - ssimag)[:, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        return ((psi - ssimag[:, None]) / span).astype(np.float32), residual


def shot_median_psin(
    times_ms: np.ndarray, psin: np.ndarray, at_ms: np.ndarray
) -> np.ndarray:
    """Each channel's median ``psi_N`` over the times ``at_ms`` (the labelled windows).

    ``psin`` is ``(T, P)`` at EFIT times ``times_ms``; the flux at a time between slices
    is the linear interpolation; times outside the EFIT record are left out. A channel
    with no finite value gives NaN.
    """
    order = np.argsort(times_ms)
    times_ms, psin = times_ms[order], psin[order]
    inside = (at_ms >= times_ms[0]) & (at_ms <= times_ms[-1])
    at_ms = at_ms[inside]
    if at_ms.size == 0:
        return np.full(psin.shape[1], np.nan)
    cols = [np.interp(at_ms, times_ms, psin[:, c]) for c in range(psin.shape[1])]
    return np.nanmedian(np.stack(cols, axis=1), axis=0)


def choose_block(psin_median: np.ndarray) -> dict:
    """The block of ``BLOCK_ROWS`` rows covering the pedestal, and whether it reaches.

    ``psin_median`` is the 64 channels' median ``psi_N``. Returns the first row of the
    block (``start``, None when no flux is known), how many of its rows cover the
    pedestal (``rows_covering``), its channels in the pedestal band
    (``channels_in_band``), the outermost channel's ``psi_N`` (``outer_psin``) and
    ``reaches`` (that is at least ``REACH_PSIN``).
    """
    grid = np.asarray(psin_median, dtype=float).reshape(ROWS, COLUMNS)
    if not np.isfinite(grid).any():
        return {
            "start": None,
            "rows_covering": 0,
            "channels_in_band": 0,
            "outer_psin": float("nan"),
            "reaches": False,
        }
    lo, hi = PEDESTAL_BAND
    in_band = (grid >= lo) & (grid <= hi)
    covered = in_band.sum(axis=1) >= ROW_MIN_CHANNELS
    starts = range(ROWS - BLOCK_ROWS + 1)
    start = max(starts, key=lambda s: (covered[s : s + BLOCK_ROWS].sum(), -s))
    block = slice(start, start + BLOCK_ROWS)
    outer = float(np.nanmax(grid[block]))
    return {
        "start": int(start),
        "rows_covering": int(covered[block].sum()),
        "channels_in_band": int(in_band[block].sum()),
        "outer_psin": outer,
        "reaches": bool(outer >= REACH_PSIN),
    }
