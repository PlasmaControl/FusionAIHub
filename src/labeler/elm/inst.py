"""`elm-inst`: individual ELM onsets at 0.1 ms beside ELMy-period occupancy.

The occupancy model (`elm-ours`) says where ELMing happens; the onset heads said
roughly when one ELM starts, at 1 ms, at least 10 ms apart, and were trained either
on reviewed non-crowd span starts or on Smith's isolated hand-labelled ELMs with no
negative time beyond his short windows. On the DIII-D challenge key they missed most
ELMs of trains and fired in quiet windows. This model is the instance half of an
instance-plus-crowd detector, after `is_crowd` in instance segmentation:

* **occupancy** (1 ms): ELMy time from the review, crowds included, as `elm-ours`;
* **instance** (0.1 ms, the input grid): a heatmap whose peaks are ELM onsets
  (CenterNet's keypoint heatmap and focal loss, Zhou et al. 2019), decoded with a
  `MIN_SEP_MS` minimum separation so ELMs of a train can be told apart.

**Instance targets.** A Gaussian of `SIGMA_MS` with its peak cell at 1, where an
onset is known, and where the loss applies (`known`):

* Smith's windows: his hand-labelled onset (`label_t0_ms`) inside each window; the
  rest of the window is known to hold no other onset;
* the review's absent spans, less `ABSENT_GUARD_MS` at each edge (reviewed edges
  sit a few ms from the D-alpha rise): known, no onset;
* the review's present spans, crowd or not, are `is_crowd` time: their ELMs are not
  individually marked, so no negative is asserted there. Only clock peaks inside
  them become positives (pseudo-onsets: the start of the clock's half-prominence
  span, which on the challenge key's ELMy windows had precision 0.965 at 2 ms), each
  known only within `PSEUDO_HALF_MS`, at weight `PSEUDO_WEIGHT`;
* uncertain, not-observable and unlabelled time is ignored.

An onset's span for a submission is the onset plus a fixed duration taken from the
training shots (no end head).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from . import inputs, net

DT_MS = inputs.DT_MS
GRID0_MS = inputs.GRID0_MS
SIGMA_MS = 0.3
MIN_SEP_MS = 2.0
ABSENT_GUARD_MS = 5.0
PSEUDO_HALF_MS = 1.0
PSEUDO_WEIGHT = 0.5
FINE_WIDTH = 16
HEAD_WIDTH = 32
FINE_KERNEL = 9


def cell_of(t_ms) -> np.ndarray:
    """The 0.1 ms cell holding each time (ms of the shot's clock)."""
    return np.floor((np.asarray(t_ms, dtype=float) - GRID0_MS) / DT_MS + 1e-9).astype(
        np.int64
    )


def time_of(cells) -> np.ndarray:
    """The centre (ms) of each 0.1 ms cell."""
    return GRID0_MS + (np.asarray(cells, dtype=float) + 0.5) * DT_MS


@dataclass(frozen=True)
class Target:
    """One shot's instance targets on the 0.1 ms grid."""

    heat: np.ndarray  # float32 (n_cells,): Gaussian bumps, 1 at onset cells
    known: np.ndarray  # bool (n_cells,): where the instance loss applies
    weight: np.ndarray  # float32 (n_cells,): per-cell loss weight


def _bump(heat, known, weight, t_ms, half_ms, w, n_cells):
    """Add one onset at `t_ms`: Gaussian, peak cell exactly 1, known +/- `half_ms`."""
    c = int(cell_of(t_ms))
    h = int(np.ceil(half_ms / DT_MS))
    lo, hi = max(0, c - h), min(n_cells, c + h + 1)
    if hi <= lo:
        return
    d = time_of(np.arange(lo, hi)) - time_of(c)
    bump = np.exp(-0.5 * (d / SIGMA_MS) ** 2).astype(np.float32)
    heat[lo:hi] = np.maximum(heat[lo:hi], bump)
    known[lo:hi] = True
    weight[lo:hi] = np.maximum(weight[lo:hi], w)


def smith_target(onsets_ms, window_t0, window_t1, n_cells: int) -> Target:
    """Smith's windows: known inside each, with his onset as the only peak."""
    heat = np.zeros(n_cells, np.float32)
    known = np.zeros(n_cells, bool)
    weight = np.zeros(n_cells, np.float32)
    for a, b in zip(window_t0, window_t1, strict=True):
        lo, hi = max(0, int(cell_of(a))), min(n_cells, int(cell_of(b)))
        known[lo:hi] = True
        weight[lo:hi] = 1.0
    for t in onsets_ms:
        _bump(heat, known, weight, t, 5 * SIGMA_MS, 1.0, n_cells)
    return Target(heat, known, weight)


def review_target(spans: pd.DataFrame, pseudo_onsets_ms, n_cells: int) -> Target:
    """Review spans: guarded absent time known empty; clock pseudo-onsets inside
    present spans as down-weighted positives; everything else ignored."""
    heat = np.zeros(n_cells, np.float32)
    known = np.zeros(n_cells, bool)
    weight = np.zeros(n_cells, np.float32)
    present = []
    for r in spans.itertuples():
        if r.kind == "absent":
            a, b = r.t_start + ABSENT_GUARD_MS, r.t_end - ABSENT_GUARD_MS
            if b > a:
                lo, hi = max(0, int(cell_of(a))), min(n_cells, int(cell_of(b)))
                known[lo:hi] = True
                weight[lo:hi] = 1.0
        elif r.kind in ("crowd", "non_crowd"):
            present.append((r.t_start, r.t_end))
    for t in np.asarray(pseudo_onsets_ms, dtype=float):
        if any(a <= t < b for a, b in present):
            _bump(heat, known, weight, t, PSEUDO_HALF_MS, PSEUDO_WEIGHT, n_cells)
    return Target(heat, known, weight)


def focal_loss(logits, heat, known, weight) -> torch.Tensor:
    """CenterNet's penalty-reduced focal loss over the known cells, normalised by the
    number of onset (peak) cells in the batch."""
    p = torch.sigmoid(logits).clamp(1e-4, 1 - 1e-4)
    pos = (heat >= 1.0 - 1e-6) & known
    neg = known & ~pos
    pos_loss = -((1 - p) ** 2) * torch.log(p) * weight
    neg_loss = -((1 - heat) ** 4) * p**2 * torch.log(1 - p) * weight
    n = pos.sum().clamp(min=1)
    return (pos_loss[pos].sum() + neg_loss[neg].sum()) / n


class InstNet(nn.Module):
    """`net.ElmUNet`'s trunk, its 1 ms decoder output read by an occupancy head, and
    upsampled beside a full-rate branch of the input for the 0.1 ms instance head.

    `forward` returns `(occupancy logits (B, n_ms), instance logits (B, n_cells))`.
    """

    def __init__(
        self,
        in_channels: int = inputs.N_CHANNELS,
        widths: tuple[int, ...] = net.WIDTHS,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.trunk = net.ElmUNet(in_channels, widths, dropout)
        k = FINE_KERNEL
        self.fine = nn.Sequential(
            nn.Conv1d(in_channels, FINE_WIDTH, k, padding=k // 2),
            nn.GroupNorm(4, FINE_WIDTH),
            nn.GELU(),
            nn.Conv1d(FINE_WIDTH, FINE_WIDTH, k, padding=k // 2),
            nn.GroupNorm(4, FINE_WIDTH),
            nn.GELU(),
        )
        self.mix = nn.Sequential(
            nn.Conv1d(widths[0] + FINE_WIDTH, HEAD_WIDTH, k, padding=k // 2),
            nn.GroupNorm(8, HEAD_WIDTH),
            nn.GELU(),
            nn.Conv1d(HEAD_WIDTH, HEAD_WIDTH, k, padding=k // 2),
            nn.GroupNorm(8, HEAD_WIDTH),
            nn.GELU(),
        )
        self.occupancy = nn.Conv1d(widths[0], 1, 1)
        self.instance = nn.Conv1d(HEAD_WIDTH, 1, 1)
        nn.init.constant_(self.instance.bias, -4.0)  # onsets are rare

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """The trunk's decoder output at 1 ms, `(B, widths[0], n_ms)`."""
        t = self.trunk
        if x.shape[-1] % net.UNIT:
            raise ValueError(
                f"input length {x.shape[-1]} is not a multiple of {net.UNIT}"
            )
        h = t.stem(x)
        skips = [h]
        for down in t.down:
            h = down(h)
            skips.append(h)
        h = t.drop(h)
        for up, skip in zip(reversed(t.up), reversed(skips[:-1]), strict=True):
            h = F.interpolate(h, size=skip.shape[-1], mode="nearest")
            h = up(torch.cat([h, skip], dim=1))
        return h

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.features(x)
        occ = self.occupancy(h)[:, 0]
        up = F.interpolate(h, size=x.shape[-1], mode="linear", align_corners=False)
        z = self.mix(torch.cat([up, self.fine(x)], dim=1))
        return occ, self.instance(z)[:, 0]


@torch.no_grad()
def predict(model: InstNet, x: np.ndarray, device) -> tuple[np.ndarray, np.ndarray]:
    """Occupancy per 1 ms and instance probability per 0.1 ms of one `(C, cells)` input."""
    model.eval()
    n = x.shape[1]
    buf = np.zeros((x.shape[0], net.pad_to(n)), dtype=np.float32)
    buf[:, :n] = x
    occ, ins = model(torch.from_numpy(buf)[None].to(device))
    occ = torch.sigmoid(occ[0, : n // inputs.CELLS_PER_MS]).float().cpu().numpy()
    return occ, torch.sigmoid(ins[0, :n]).float().cpu().numpy()


def peaks(heat: np.ndarray, threshold: float, min_sep_ms: float = MIN_SEP_MS):
    """Cells of the local maxima of a 0.1 ms heatmap at or above `threshold`, none
    within `min_sep_ms` of a higher one."""
    x = np.asarray(heat, dtype=np.float64)
    if x.size < 3:
        return np.zeros(0, dtype=np.int64)
    inner = (x[1:-1] >= x[:-2]) & (x[1:-1] > x[2:]) & (x[1:-1] >= threshold)
    cand = np.flatnonzero(inner) + 1
    sep = round(min_sep_ms / DT_MS)
    keep: list[int] = []
    taken = np.zeros(x.size, dtype=bool)
    for i in cand[np.argsort(-x[cand], kind="stable")]:
        if not taken[i]:
            keep.append(int(i))
            taken[max(0, i - sep + 1) : i + sep] = True
    return np.array(sorted(keep), dtype=np.int64)


def onsets(heat: np.ndarray, threshold: float, gate=None, gate_threshold=None):
    """Onset times (ms) of a heatmap's peaks; with `gate` (a 1 ms occupancy trace
    from `GRID0_MS`), only those where its 50 ms centred mean reaches `gate_threshold`."""
    t = time_of(peaks(heat, threshold))
    if gate is None or not len(t):
        return t
    from scipy.ndimage import uniform_filter1d

    smooth = uniform_filter1d(np.asarray(gate, dtype=np.float64), 50, mode="nearest")
    i = np.clip(np.floor(t - GRID0_MS).astype(int), 0, len(smooth) - 1)
    return t[smooth[i] >= gate_threshold]


def match(found_ms, truth_ms, tol_ms: float) -> tuple[int, int, int, list[float]]:
    """`(tp, fp, fn, offsets)`: one to one, nearest first (greedy by distance)."""
    found = np.sort(np.asarray(found_ms, dtype=float))
    truth = np.sort(np.asarray(truth_ms, dtype=float))
    if not len(found) or not len(truth):
        return 0, len(found), len(truth), []
    d = np.abs(found[:, None] - truth[None, :])
    pairs = np.argwhere(d <= tol_ms)
    order = np.argsort(d[pairs[:, 0], pairs[:, 1]], kind="stable")
    used_f = np.zeros(len(found), bool)
    used_t = np.zeros(len(truth), bool)
    offsets = []
    for i, j in pairs[order]:
        if not used_f[i] and not used_t[j]:
            used_f[i] = used_t[j] = True
            offsets.append(float(found[i] - truth[j]))
    tp = int(used_t.sum())
    return tp, len(found) - tp, len(truth) - tp, offsets
