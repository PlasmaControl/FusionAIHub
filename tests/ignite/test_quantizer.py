"""TDD spec for ignite.quantizer.SpectroQuantizer.

Small synthetic CPU-only tensors. Contract (config.py):

    SpectroQuantizer(cfg)
        .quantize(feats: (B,n_tok,d_model)) -> (quant: (B,n_tok,d_model),
                                                codes: (B,n_tok,fsq_dim) long)
        .codebook_size
    - codes[..., i] in [0, fsq_levels[i])
    - straight-through gradient flows from quant back to feats
"""
from __future__ import annotations

from unittest import mock

import torch

from tokamak_foundation_model.ignite import quantizer as quantizer_mod
from tokamak_foundation_model.ignite.config import (
    FastTSCodecConfig,
    SlowTSCodecConfig,
    SpectroCodecConfig,
    VideoCodecConfig,
)
from tokamak_foundation_model.ignite.quantizer import SpectroQuantizer


def _small_cfg() -> SpectroCodecConfig:
    # keep tiny + cheap; only the FSQ / d_model path is exercised here.
    return SpectroCodecConfig(d_model=32, fsq_levels=[4, 4, 3])


def test_shapes() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    B, n_tok = 2, cfg.n_tok
    feats = torch.randn(B, n_tok, cfg.d_model)
    quant, codes = q.quantize(feats)
    assert quant.shape == (B, n_tok, cfg.d_model)
    assert codes.shape == (B, n_tok, cfg.fsq_dim)


def test_codes_are_long() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(2, cfg.n_tok, cfg.d_model)
    _, codes = q.quantize(feats)
    assert codes.dtype == torch.long


def test_codes_in_range() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    # large batch of extreme feats to push FSQ to its bounds.
    feats = torch.randn(8, cfg.n_tok, cfg.d_model) * 10.0
    _, codes = q.quantize(feats)
    levels = torch.tensor(cfg.fsq_levels)
    assert (codes >= 0).all()
    assert (codes < levels).all()  # per-dim upper bound


def test_codebook_size() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    assert q.codebook_size == cfg.codebook_size == 4 * 4 * 3


def test_default_fsq_config_is_right_sized() -> None:
    """FIX 1: the default codec is right-sized to [8, 5, 5, 5] = 1000 codes over 4 FSQ dims.

    The previous default was [8, 8, 8, 8, 8] = 32768 with a spare 5th dim that always died.
    """
    cfg = SpectroCodecConfig()  # real default
    assert cfg.fsq_levels == [8, 5, 5, 5]
    assert cfg.fsq_dim == 4
    assert cfg.codebook_size == 1000


def test_quantizer_roundtrip_at_default_size() -> None:
    """The FSQ quantizer round-trips at the new default size: codes are per-dim in range,
    shape (B, n_tok, fsq_dim=4), and the STE gradient reaches the encoder features."""
    cfg = SpectroCodecConfig(d_model=32)  # default fsq_levels [8, 5, 5, 5]
    q = SpectroQuantizer(cfg)
    assert q.codebook_size == 1000
    feats = torch.randn(4, cfg.n_tok, cfg.d_model, requires_grad=True)
    quant, codes = q.quantize(feats)
    assert quant.shape == (4, cfg.n_tok, cfg.d_model)
    assert codes.shape == (4, cfg.n_tok, cfg.fsq_dim) == (4, cfg.n_tok, 4)
    levels = torch.tensor(cfg.fsq_levels)
    assert (codes >= 0).all()
    assert (codes < levels).all()  # per-dim upper bound honoured at the new size
    quant.sum().backward()
    assert feats.grad is not None and feats.grad.abs().sum() > 0


def test_straight_through_gradient() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(2, cfg.n_tok, cfg.d_model, requires_grad=True)
    quant, _ = q.quantize(feats)
    quant.sum().backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
    assert feats.grad.abs().sum() > 0  # gradient actually reached the input


def test_quant_is_differentiable_continuous() -> None:
    # quant must be a float tensor in d_model space (decoder-facing), not indices.
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(2, cfg.n_tok, cfg.d_model)
    quant, _ = q.quantize(feats)
    assert quant.dtype == torch.float32


# --------------------------------------------------------------------------- #
# anti-collapse entropy_loss
# --------------------------------------------------------------------------- #
def _collapsed_feats(q: SpectroQuantizer, B: int = 16) -> torch.Tensor:
    """Every sample identical -> the encoder maps the whole batch to ONE code."""
    n_tok = q.cfg.n_tok
    one = torch.randn(1, 1, q.cfg.d_model)
    return one.expand(B, n_tok, q.cfg.d_model).contiguous()


def _diverse_feats(q: SpectroQuantizer, B: int = 16) -> torch.Tensor:
    """Large-variance random feats that spread across the FSQ grid -> many codes."""
    n_tok = q.cfg.n_tok
    return torch.randn(B, n_tok, q.cfg.d_model) * 5.0


def test_pre_quant_levels_round_recovers_codes() -> None:
    """round(pre_quant_levels) must equal the integer codes from quantize()."""
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(4, cfg.n_tok, cfg.d_model) * 3.0
    _, codes = q.quantize(feats)
    cont = q.pre_quant_levels(feats)
    assert cont.shape == codes.shape
    levels = torch.tensor(cfg.fsq_levels)
    recovered = cont.round().clamp(min=torch.zeros_like(levels), max=levels - 1).long()
    assert torch.equal(recovered, codes)


def test_entropy_loss_returns_scalar() -> None:
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    loss = q.entropy_loss(_diverse_feats(q))
    assert loss.ndim == 0
    assert torch.isfinite(loss)


def test_entropy_loss_high_for_collapse_low_for_diverse() -> None:
    """Collapse (one code) => HIGH loss; diverse batch (many codes) => LOW loss.

    Correctness of the anti-collapse term: it must penalise the trivial encoder≡const
    minimum (which uses one code) more than a batch that spreads across the grid.
    """
    torch.manual_seed(0)
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    loss_collapsed = q.entropy_loss(_collapsed_feats(q))
    loss_diverse = q.entropy_loss(_diverse_feats(q))
    assert float(loss_collapsed) > float(loss_diverse)


def test_entropy_loss_is_differentiable() -> None:
    """Gradient of entropy_loss must reach the input feats."""
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(8, cfg.n_tok, cfg.d_model, requires_grad=True)
    q.entropy_loss(feats).backward()
    assert feats.grad is not None
    assert torch.isfinite(feats.grad).all()
    assert feats.grad.abs().sum() > 0


# --------------------------------------------------------------------------- #
# FIX 2: entropy_weight default is 1.0 on EVERY codec config
# --------------------------------------------------------------------------- #
def test_entropy_weight_default_is_one_on_all_codec_configs() -> None:
    """The diversity spike that reached healthy min_dim_entropy≈0.65 used entropy_weight=1.0;
    0.1 was the known-collapsing value. Every codec config must now default to 1.0."""
    assert SpectroCodecConfig().entropy_weight == 1.0
    assert VideoCodecConfig().entropy_weight == 1.0
    assert FastTSCodecConfig().entropy_weight == 1.0
    assert SlowTSCodecConfig().entropy_weight == 1.0  # slow-TS uses entropy too — covered
    # diversity_weight stays 1.0 (the spread reward weight is unchanged).
    assert SpectroCodecConfig().diversity_weight == 1.0
    assert VideoCodecConfig().diversity_weight == 1.0
    assert FastTSCodecConfig().diversity_weight == 1.0
    assert SlowTSCodecConfig().diversity_weight == 1.0


# --------------------------------------------------------------------------- #
# FIX 1: all-reduce the batch-diversity statistic across DDP ranks
# --------------------------------------------------------------------------- #
def test_entropy_loss_single_process_is_identical_to_local_computation() -> None:
    """REGRESSION GUARD: with world_size==1 / dist not initialized, entropy_loss must be
    NUMERICALLY IDENTICAL to the previous purely-local computation (p_mean = p.mean(dim=0)).

    We re-derive the old formula by hand from pre_quant_levels and assert bit-for-bit equality
    with the (unmocked, single-process) entropy_loss — the single-process path must not change.
    """
    torch.manual_seed(1)
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    feats = torch.randn(16, cfg.n_tok, cfg.d_model) * 3.0

    # ---- reference: the ORIGINAL local computation, replicated verbatim ----
    levels = list(cfg.fsq_levels)
    max_levels = max(levels)
    x = q.pre_quant_levels(feats).reshape(-1, len(levels))
    grid = torch.arange(max_levels, dtype=x.dtype)
    dist2 = (x[..., None] - grid[None, None, :]) ** 2
    logits = -10.0 * dist2  # beta default = 10.0
    lvl_t = torch.tensor(levels)
    valid = grid[None, :] < lvl_t[:, None]
    logits = logits.masked_fill(~valid[None, :, :], float("-inf"))
    p = torch.softmax(logits, dim=-1)
    eps = 1e-9
    per_sample_entropy = -(p * torch.log(p + eps)).sum(dim=-1)
    per_sample_entropy_mean = per_sample_entropy.mean()
    p_mean_local = p.mean(dim=0)  # the ORIGINAL local batch-mean
    batch_entropy = -(p_mean_local * torch.log(p_mean_local + eps)).sum(dim=-1)
    entropy_of_batch_mean = batch_entropy.mean()
    reference = per_sample_entropy_mean - cfg.diversity_weight * entropy_of_batch_mean

    got = q.entropy_loss(feats)
    assert torch.equal(got, reference), (float(got), float(reference))


def _mock_dist_accumulating(world_size: int, sum_calls: list):
    """Patch quantizer.dist to look distributed with ``world_size``. Each all_reduce(SUM)
    ADDS the previous call's tensor into the current one (simulating summing across the two
    ranks fed sequentially), recording every reduced tensor's value in ``sum_calls`` so the
    test can assert the op and inspect what was reduced.

    The accumulator is keyed by tensor shape so the (fsq_dim, max_levels) prob-sum and the
    (1,) count scalar are reduced independently — exactly two distinct collectives per rank.
    """
    m = mock.MagicMock()
    m.is_available.return_value = True
    m.is_initialized.return_value = True
    m.get_world_size.return_value = world_size

    class _Op:
        SUM = "SUM"

    m.ReduceOp = _Op
    prev: dict = {}

    def _all_reduce(tensor, op=None):
        assert op == _Op.SUM, "batch-diversity reduction must use ReduceOp.SUM"
        key = tuple(tensor.shape)
        if key in prev:
            tensor.add_(prev[key])       # accumulate the earlier rank's contribution
        prev[key] = tensor.detach().clone()
        sum_calls.append((key, op))

    m.all_reduce.side_effect = _all_reduce
    return m


def _batch_entropy_only(q: SpectroQuantizer, feats: torch.Tensor) -> float:
    """The batch-diversity term (entropy of the per-dim batch-mean assignment), isolated.

    entropy_loss = per_sample_entropy_mean - diversity_weight*batch_entropy, and the
    per-sample term is LOCAL/unchanged, so subtracting a same-feats per_sample term is not
    needed for the local-vs-global comparison below — we recompute the batch_entropy directly
    the same way entropy_loss does, so the numbers are the exact quantity the fix strengthens.
    """
    cfg = q.cfg
    levels = list(cfg.fsq_levels)
    max_levels = max(levels)
    x = q.pre_quant_levels(feats).reshape(-1, len(levels))
    grid = torch.arange(max_levels, dtype=x.dtype)
    dist2 = (x[..., None] - grid[None, None, :]) ** 2
    logits = -10.0 * dist2
    lvl_t = torch.tensor(levels)
    valid = grid[None, :] < lvl_t[:, None]
    logits = logits.masked_fill(~valid[None, :, :], float("-inf"))
    p = torch.softmax(logits, dim=-1)
    p_mean = p.mean(dim=0)
    eps = 1e-9
    return float((-(p_mean * torch.log(p_mean + eps)).sum(dim=-1)).mean())


def test_entropy_loss_batch_diversity_is_global_under_ddp() -> None:
    """FIX 1: under DDP the batch-diversity term uses the GLOBAL (all-reduced) p_mean.

    Two 'ranks' with DISJOINT code usage: rank A's feats land on the LOW-index FSQ grid points,
    rank B's on the HIGH-index grid points. Each rank's LOCAL batch-mean is concentrated on its
    own half (low batch-entropy); the GLOBAL batch-mean spreads over BOTH halves (higher
    batch-entropy). We assert the all-reduced batch-entropy is HIGHER than either rank's local
    batch-entropy — proving the global reduction strengthens the spread signal.

    Also asserts: all_reduce is called with SUM, and the per-sample entropy term (local) is
    untouched by the reduction.
    """
    torch.manual_seed(7)
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)

    # Drive the pre-quant level positions toward opposite ends of the grid by pushing the
    # FSQ project_in pre-activations very negative (rank A -> code 0 side) / very positive
    # (rank B -> top-level side). Big-magnitude feats saturate tanh to the grid extremes.
    B = 16
    feats_low = -torch.abs(torch.randn(B, cfg.n_tok, cfg.d_model)) * 20.0
    feats_high = torch.abs(torch.randn(B, cfg.n_tok, cfg.d_model)) * 20.0

    # sanity: the two ranks really do use DISJOINT codes (low half vs high half).
    _, codes_low = q.quantize(feats_low)
    _, codes_high = q.quantize(feats_high)
    assert codes_low.float().mean() < codes_high.float().mean(), "ranks must be disjoint"

    # LOCAL batch-entropy for each rank (single-process, no mock).
    local_be_low = _batch_entropy_only(q, feats_low)
    local_be_high = _batch_entropy_only(q, feats_high)

    # GLOBAL batch-entropy: feed both ranks through entropy_loss under the accumulating mock.
    # The per-sample term is local, so isolate the global batch-entropy by subtracting each
    # rank's per_sample_entropy_mean from its entropy_loss return, then negating/dividing.
    def _per_sample_mean(feats: torch.Tensor) -> float:
        levels = list(cfg.fsq_levels)
        max_levels = max(levels)
        x = q.pre_quant_levels(feats).reshape(-1, len(levels))
        grid = torch.arange(max_levels, dtype=x.dtype)
        logits = -10.0 * (x[..., None] - grid[None, None, :]) ** 2
        lvl_t = torch.tensor(levels)
        valid = grid[None, :] < lvl_t[:, None]
        logits = logits.masked_fill(~valid[None, :, :], float("-inf"))
        p = torch.softmax(logits, dim=-1)
        return float((-(p * torch.log(p + 1e-9)).sum(dim=-1)).mean())

    sum_calls: list = []
    with mock.patch.object(quantizer_mod, "dist", _mock_dist_accumulating(2, sum_calls)):
        loss_a = q.entropy_loss(feats_low)   # rank A: contributes its p_sum, count
        loss_b = q.entropy_loss(feats_high)  # rank B: accumulates A -> sees the GLOBAL p_mean

    # rank B's entropy_loss now reflects the GLOBAL batch-mean; recover its batch-entropy.
    # entropy_loss = per_sample_mean - diversity_weight * batch_entropy_global.
    global_be = (_per_sample_mean(feats_high) - float(loss_b)) / cfg.diversity_weight

    # The global spread must exceed BOTH ranks' local spreads (disjoint halves -> more diverse).
    assert global_be > local_be_low, (global_be, local_be_low)
    assert global_be > local_be_high, (global_be, local_be_high)

    # all_reduce used SUM, and was called for BOTH the prob-sum tensor and the count scalar,
    # per rank (2 ranks x 2 tensors = 4 calls).
    assert all(op == "SUM" for _, op in sum_calls)
    assert len(sum_calls) == 4

    # The per-sample entropy term is LOCAL — the reduction must NOT touch it. Feed a SINGLE
    # rank's feats through the mock (no prior accumulation): the reduced p_sum equals the local
    # p_sum, so p_mean == the local batch-mean and entropy_loss must equal the single-process
    # value. If the reduction had wrongly folded the per-sample term in, this would differ.
    single_process = float(q.entropy_loss(feats_high))
    with mock.patch.object(quantizer_mod, "dist", _mock_dist_accumulating(2, [])):
        under_mock_no_peer = float(q.entropy_loss(feats_high))
    assert abs(under_mock_no_peer - single_process) < 1e-6, (
        under_mock_no_peer, single_process,
    )


# --------------------------------------------------------------------------- #
# FIX 2: the diversity GRADIENT is scaled by world_size under DDP (value preserved)
# --------------------------------------------------------------------------- #
def test_entropy_loss_diversity_gradient_scaled_by_world_size() -> None:
    """dist.all_reduce is not autograd-aware, so each rank's backward carries only its
    local contribution and DDP's gradient averaging attenuates the diversity term by
    1/world_size. FIX 2 scales the term's gradient by world_size (value-preserving via
    w*H - (w-1)*H.detach()). With a single mocked rank (no peer accumulation) the VALUE
    must equal the unmocked loss, while the GRADIENT must equal
    grad(per_sample) + world_size * grad(diversity)."""
    torch.manual_seed(3)
    cfg = _small_cfg()
    q = SpectroQuantizer(cfg)
    W = 4

    def _grad(loss_fn) -> torch.Tensor:
        feats = torch.randn(8, cfg.n_tok, cfg.d_model)
        feats = feats.clone().requires_grad_(True)
        loss_fn(feats).backward()
        return feats.grad.clone()

    # deterministic feats across calls: fix the seed inside each grad computation
    def _feats() -> torch.Tensor:
        g = torch.Generator().manual_seed(11)
        return torch.randn(8, cfg.n_tok, cfg.d_model, generator=g).requires_grad_(True)

    # unmocked full gradient and per-sample-only gradient (diversity_weight = 0)
    f1 = _feats()
    q.entropy_loss(f1).backward()
    grad_full = f1.grad.clone()

    cfg0 = _small_cfg()
    cfg0.diversity_weight = 0.0
    q.cfg = cfg0
    f2 = _feats()
    q.entropy_loss(f2).backward()
    grad_per_sample = f2.grad.clone()
    q.cfg = cfg  # restore

    grad_diversity = grad_full - grad_per_sample

    # mocked single rank at world_size W: value preserved, diversity gradient x W
    f3 = _feats()
    with mock.patch.object(quantizer_mod, "dist", _mock_dist_accumulating(W, [])):
        loss_mock = q.entropy_loss(f3)
        loss_mock.backward()
    grad_mock = f3.grad.clone()

    f4 = _feats()
    loss_plain = q.entropy_loss(f4)
    assert torch.allclose(loss_mock, loss_plain, atol=1e-6)              # value preserved
    expected = grad_per_sample + W * grad_diversity
    assert torch.allclose(grad_mock, expected, atol=1e-5), (
        float((grad_mock - expected).abs().max())
    )
