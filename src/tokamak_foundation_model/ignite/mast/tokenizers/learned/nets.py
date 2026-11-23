"""The token network of the learned tokenizers and its recoder (plan 1c).

`TokenNet` turns one example (the front values of a frame of one stream, in example units) into
`n_values` numbers in (-1, 1) and back. The first values are meant to carry the most (`quantizer`
zeroes random suffixes in training) and the values survive rounding to a few levels. A stream's identity
(its registry features, and a learned vector of its own) conditions encoder and decoder. `NetRecoder`
wraps a trained network as a `recode.Recoder`. This module imports torch.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from torch import nn

from . import quantizer, recode
from .identity import N_FEAT, StreamTable


def symlog(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) * log1p(|x|)``, elementwise: example units to a scale where 50 is 3.9 (same shape)."""
    return torch.sign(x) * torch.log1p(torch.abs(x))


class _Block(nn.Module):
    """A residual block: ``x + Linear(GELU(Linear(LayerNorm(x))))`` on ``(B, d_model)``."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.up = nn.Linear(d_model, 2 * d_model)
        self.act = nn.GELU()
        self.down = nn.Linear(2 * d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.down(self.act(self.up(self.norm(x))))


class TokenNet(nn.Module):
    """The encoder and decoder between examples ``(B, d_in)`` and values ``(B, n_values)``.

    `n_streams` is the number of rows of the stream table that get a learned identity vector (0 for the
    concatenated control: no identity parts at all, `identity` returns zeros). `d_feat` is the width of
    the table's feature rows. `p_drop_ident` is the share of training examples whose learned vector is
    left out, so a stream the network never saw is not out of distribution. There is no dropout layer.
    """

    def __init__(self, d_in: int, n_values: int, n_streams: int, *, d_feat: int = N_FEAT, d_model: int = 512,
                 depth: int = 4, p_drop_ident: float = 0.1) -> None:
        super().__init__()
        self.d_in, self.n_values, self.n_streams = int(d_in), int(n_values), int(n_streams)
        self.p_drop_ident = float(p_drop_ident)
        if n_streams > 0:
            self.learned = nn.Embedding(n_streams, d_model)
            nn.init.zeros_(self.learned.weight)
            self.features = nn.Sequential(nn.Linear(d_feat, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        else:
            self.learned = None
            self.features = None
        self.enc_in = nn.Linear(d_in, d_model)
        self.enc_blocks = nn.ModuleList(_Block(d_model) for _ in range(depth))
        self.enc_norm = nn.LayerNorm(d_model)
        self.enc_out = nn.Linear(d_model, n_values)
        self.dec_in = nn.Linear(n_values, d_model)
        self.dec_blocks = nn.ModuleList(_Block(d_model) for _ in range(depth))
        self.dec_norm = nn.LayerNorm(d_model)
        self.dec_out = nn.Linear(d_model, d_in)
        self.d_model = int(d_model)

    def identity(self, feats: torch.Tensor, ident: torch.Tensor, drop: torch.Tensor | None = None) -> torch.Tensor:
        """The identity vectors ``(B, d_model)`` of a batch.

        `feats` is float32 ``(B, d_feat)``, the table's feature rows (zeros where `ident` is -1); `ident`
        is int64 ``(B,)``, each example's row in the table (-1: none). The result is the feature
        network's output plus the learned vector of row `ident`; the learned vector is left out where
        `ident == -1` or `drop` (bool ``(B,)``, None: nowhere) is True. Zeros if ``n_streams == 0``.
        """
        if self.learned is None:
            return feats.new_zeros(feats.shape[0], self.d_model)
        use = ident >= 0
        if drop is not None:
            use = use & ~drop
        vec = self.learned(ident.clamp(min=0)) * use[:, None].to(feats.dtype)
        return self.features(feats) + vec

    def encode(self, e: torch.Tensor, feats: torch.Tensor, ident: torch.Tensor,
               drop: torch.Tensor | None = None) -> torch.Tensor:
        """Examples `e` ``(B, d_in)``, example units, to values ``(B, n_values)`` in (-1, 1).

        `feats`, `ident` and `drop` as in `identity`.
        """
        h = self.enc_in(symlog(e)) + self.identity(feats, ident, drop)
        for block in self.enc_blocks:
            h = block(h)
        return torch.tanh(self.enc_out(self.enc_norm(h)))

    def decode(self, v: torch.Tensor, feats: torch.Tensor, ident: torch.Tensor,
               drop: torch.Tensor | None = None) -> torch.Tensor:
        """Values `v` ``(B, n_values)`` in [-1, 1] to examples ``(B, d_in)``, example units.

        `feats`, `ident` and `drop` as in `identity`.
        """
        h = self.dec_in(v) + self.identity(feats, ident, drop)
        for block in self.dec_blocks:
            h = block(h)
        return self.dec_out(self.dec_norm(h))

    def forward(self, e: torch.Tensor, feats: torch.Tensor, ident: torch.Tensor,
                generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Examples `e` ``(B, d_in)``, example units, to ``(e_hat, v)``: the reconstruction ``(B, d_in)``,
        example units, and the values ``(B, n_values)`` in (-1, 1) before any rounding.

        In training mode every random draw comes from `generator` (`ValueError` naming "generator"
        without one): the identity drop (probability `p_drop_ident` per example), then
        `quantizer.draw`, and ``e_hat = decode(quantizer.apply(v, precision, n_keep))``. In eval mode
        the values pass unchanged. `feats` and `ident` as in `identity`.
        """
        if not self.training:
            v = self.encode(e, feats, ident)
            return self.decode(v, feats, ident), v
        if generator is None:
            raise ValueError("a TokenNet in training mode needs a generator: every random draw comes from it")
        b = e.shape[0]
        drop = (torch.rand(b, generator=generator, device=generator.device) < self.p_drop_ident).to(e.device)
        precision, n_keep = quantizer.draw(b, self.n_values, generator, device=e.device)
        v = self.encode(e, feats, ident, drop)
        return self.decode(quantizer.apply(v, precision, n_keep), feats, ident, drop), v


class NetRecoder(recode.Recoder):
    """A trained `TokenNet` as a `recode.Recoder` (kind ``"net"``).

    `table` supplies the feature row of each example's `ident` (zeros for -1, so a concatenated recoder
    with no stream identity works). The net is put in eval mode on `device`. `encode` and `decode` take
    and return numpy, run without gradients in batches of `batch` examples, so the batch size changes
    nothing. `ValueError` if ``n_tok * n_val`` is not the net's `n_values`.
    """

    kind = "net"

    def __init__(self, net: TokenNet, table: StreamTable, *, concat: bool, n_tok: int, n_val: int,
                 device: str = "cpu", batch: int = 65536) -> None:
        if n_tok * n_val != net.n_values:
            raise ValueError(f"n_tok * n_val = {n_tok * n_val} but the network has {net.n_values} values")
        self.net = net.to(device).eval()
        self.table = table
        self.concat = bool(concat)
        self.d_in = int(net.d_in)
        self.n_tok, self.n_val = int(n_tok), int(n_val)
        self.device = device
        self.batch = int(batch)

    def _feats(self, ident: np.ndarray) -> torch.Tensor:
        """Float32 feature rows ``(M, N_FEAT)`` of `ident` (int64 ``(M,)``, -1: zeros) on the device."""
        ident = np.asarray(ident, dtype=np.int64)
        out = np.zeros((ident.shape[0], self.table.features.shape[1]), dtype=np.float32)
        known = ident >= 0
        out[known] = self.table.features[ident[known]]
        return torch.from_numpy(out).to(self.device)

    def _run(self, fn, x: np.ndarray, ident: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32)
        ident = np.asarray(ident, dtype=np.int64)
        parts = []
        with torch.no_grad():
            for i in range(0, x.shape[0], self.batch):
                j = slice(i, i + self.batch)
                ids = torch.from_numpy(ident[j]).to(self.device)
                parts.append(fn(torch.from_numpy(x[j]).to(self.device), self._feats(ident[j]), ids).cpu().numpy())
        return np.concatenate(parts, axis=0) if parts else np.zeros((0, 0), dtype=np.float32)

    def encode(self, e: np.ndarray, ident: np.ndarray) -> np.ndarray:
        """Examples `e` ``(M, d_in)``, example units, to values ``(M, n_values)`` float32 in (-1, 1).

        `ident` is int64 ``(M,)``, each example's row in the table (-1: no identity).
        """
        if len(e) == 0:
            return np.zeros((0, self.n_values), dtype=np.float32)
        return self._run(self.net.encode, e, ident)

    def decode(self, values: np.ndarray, ident: np.ndarray) -> np.ndarray:
        """Values ``(M, n_values)`` in [-1, 1] to examples ``(M, d_in)`` float32, example units (`ident` as in `encode`)."""
        if len(values) == 0:
            return np.zeros((0, self.d_in), dtype=np.float32)
        return self._run(self.net.decode, values, ident)


def save_net(path: Path, net: TokenNet, config: dict) -> None:
    """Write ``{"state": net.state_dict(), "config": config}`` to `path` (a ``best.pt``) atomically.

    `config` holds ``d_in, n_values, n_streams, d_model, depth, p_drop_ident, concat, n_tok, n_val``
    (ints, a float and a bool, so `weights_only` loading reads it). Written to a temporary name, then renamed.
    """
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save({"state": net.state_dict(), "config": dict(config)}, tmp)
    os.replace(tmp, path)


def load_net_recoder(directory: Path, table: StreamTable, device: str = "cpu") -> NetRecoder:
    """The `NetRecoder` of ``directory / "best.pt"`` (`save_net`'s file), on `device`.

    `table` is the recoder's `StreamTable` (features float32 ``(n, N_FEAT)``). The file is read with
    ``weights_only=True``. `FileNotFoundError` if there is no ``best.pt``.
    """
    blob = torch.load(Path(directory) / "best.pt", map_location=device, weights_only=True)
    c = blob["config"]
    net = TokenNet(c["d_in"], c["n_values"], c["n_streams"], d_model=c["d_model"], depth=c["depth"],
                   p_drop_ident=c["p_drop_ident"])
    net.load_state_dict(blob["state"])
    return NetRecoder(net, table, concat=c["concat"], n_tok=c["n_tok"], n_val=c["n_val"], device=device)
