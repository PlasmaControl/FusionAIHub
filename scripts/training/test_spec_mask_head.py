"""Unit tests for the SpectrogramFlowHead mode-MASK branch + input-conditioning.

Plan B (2026-06-30): the overfit tests proved μ (MAE) and the flow sample
(velocity MSE) both mean-collapse. A SEGMENTATION mask (soft-Dice+BCE) has no
mean-seeking optimum, and input-conditioning adds a PERSISTENCE prior so the head
copies in-window modes forward and learns only the residual. These tests verify
the MECHANISM (shapes, prior application, gradient flow, DDP-safety, sparse init)
— NOT the persistence *quality*, which is a data property (survival τ½ 201 ms) and
is measured by the real-shot benchmark, not synthesizable cleanly here.

Run:  .pixi/envs/default/bin/python scripts/training/test_spec_mask_head.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from tokamak_foundation_model.e2e.output_heads import SpectrogramFlowHead  # noqa: E402
from train_e2e_stage1 import spectro_mask_loss  # noqa: E402

C, DM, PF, PT, NPF, NPT = 4, 32, 8, 4, 2, 3      # F=16, T=12
B, FB, TB = 2, PF * NPF, PT * NPT


def _head(input_cond: bool) -> SpectrogramFlowHead:
    return SpectrogramFlowHead(
        n_channels=C, d_model=DM, patch_f=PF, patch_t=PT,
        n_patches_f=NPF, n_patches_t=NPT,
        enable_mask=True, mask_hidden_ch=16, enable_input_cond=input_cond,
    )


def test_shapes_and_mask_loss():
    head = _head(False)
    tokens = torch.randn(B, NPF * NPT, DM)
    logits = head.mask_logits(tokens)
    assert tuple(logits.shape) == (B, C, FB, TB), logits.shape
    target = torch.randn(B, C, FB, TB)
    target[:, :, 4:6, :] += 6.0
    loss, md = spectro_mask_loss(logits, target, 2.5, gate=torch.ones(B, 1, 1, 1))
    assert loss.item() > 0 and 0.0 <= md.item() <= 1.0
    loss.backward()
    g = max(p.grad.abs().max().item() for p in head.mask_decode.parameters()
            if p.grad is not None)
    assert g > 0, "mask decode must get gradients"
    print("  [ok] shapes + mask loss + gradients")


def test_ddp_safety_when_absent():
    """gate=0 (modality absent) → loss ~0 but every mask param still has a grad
    tensor (participated in the graph) → no DDP unused-parameter error."""
    head = _head(False)
    tokens = torch.randn(B, NPF * NPT, DM)
    target = torch.randn(B, C, FB, TB)
    loss0, md0 = spectro_mask_loss(head.mask_logits(tokens), target, 2.5,
                                   gate=torch.zeros(B, 1, 1, 1))
    loss0.backward()
    assert abs(loss0.item()) < 1e-3
    assert all(p.grad is not None for p in head.mask_decode.parameters())
    print("  [ok] DDP-safe when modality absent (loss 0, grads present)")


def test_sparse_init_without_input_cond():
    """No input-cond → final bias −3 → near-empty initial mask (doesn't flood
    the Dice/BCE before it learns)."""
    head = _head(False)
    tokens = torch.randn(B, NPF * NPT, DM)
    dens = head.mask_prob(tokens).mean().item()
    assert dens < 0.15, f"expected sparse init, got density {dens:.3f}"
    print(f"  [ok] sparse init without input-cond (density {dens:.3f})")


def test_prior_shifts_logits():
    """Input-cond prior BOOSTS logits where prior≈1, SUPPRESSES where prior≈0."""
    head = _head(True)
    tokens = torch.randn(B, NPF * NPT, DM)
    prior = torch.zeros(B, C, FB, TB)
    prior[:, :, 4:6, :] = 0.9
    lg_no = head.mask_logits(tokens)
    lg_pr = head.mask_logits(tokens, prior=prior)
    boost = (lg_pr[:, :, 4:6, :] - lg_no[:, :, 4:6, :]).mean().item()
    supp = (lg_pr[:, :, 0:4, :] - lg_no[:, :, 0:4, :]).mean().item()
    assert boost > 0 and supp < 0, (boost, supp)
    assert head.mask_prior_gain.requires_grad
    print(f"  [ok] prior shifts logits (+{boost:.2f} at modes, {supp:.2f} off)")


def test_predicted_reproduces_prior_at_init():
    """THE key persistence property: at init the predicted hard mask reproduces
    the prior (decode≈0, prior dominates via gain·logit) → the head starts from
    copy-forward persistence, then learns the residual. On real data this means
    maskdice starts ≈ the input→output mode overlap (survival ~0.6)."""
    head = _head(True)
    tokens = torch.randn(B, NPF * NPT, DM)
    prior = torch.zeros(B, C, FB, TB)
    prior[:, :, 5:8, 2:9] = 1.0                    # an arbitrary mode pattern
    ph = (head.mask_prob(tokens, prior=prior) > 0.5).float()
    dice = (2 * (ph * prior).sum() + 1) / (ph.sum() + prior.sum() + 1)
    assert dice.item() > 0.9, f"predicted must reproduce prior at init, dice {dice:.3f}"
    # and the prior gain receives a gradient
    loss, _ = spectro_mask_loss(head.mask_logits(tokens, prior=prior),
                                prior * 6.0, 2.5, gate=torch.ones(B, 1, 1, 1))
    loss.backward()
    assert head.mask_prior_gain.grad is not None
    print(f"  [ok] predicted reproduces prior at init (dice {dice.item():.3f}) + gain grad flows")


def test_no_nan_under_bf16_autocast():
    """Regression for the 4922044 NaN: training runs under bf16 autocast, where
    1−1e-4 rounds to 1.0 → logit(1.0)=+inf → NaN. A HARD (0/1) prior + autocast
    must stay finite (the fp32 + 1e-3-margin fix in mask_logits)."""
    if not torch.cuda.is_available():
        # CPU has no bf16 autocast path here; assert the fp32 fix directly:
        head = _head(True)
        tokens = torch.randn(B, NPF * NPT, DM)
        prior = (torch.rand(B, C, FB, TB) > 0.5).float()   # hard 0/1
        lg = head.mask_logits(tokens, prior=prior)
        assert torch.isfinite(lg).all(), "logits must be finite with a hard 0/1 prior"
        print("  [ok] finite logits with hard 0/1 prior (fp32 path; no CUDA for bf16)")
        return
    head = _head(True).cuda()
    tokens = torch.randn(B, NPF * NPT, DM, device="cuda")
    prior = (torch.rand(B, C, FB, TB, device="cuda") > 0.5).float()
    with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
        lg = head.mask_logits(tokens, prior=prior)
        p = head.mask_prob(tokens, prior=prior)
    assert torch.isfinite(lg).all() and torch.isfinite(p).all(), "bf16 autocast produced NaN/inf"
    print("  [ok] no NaN/inf under bf16 autocast with hard 0/1 prior")


if __name__ == "__main__":
    torch.manual_seed(0)
    print("SpectrogramFlowHead mask-branch + input-conditioning tests:")
    test_shapes_and_mask_loss()
    test_ddp_safety_when_absent()
    test_sparse_init_without_input_cond()
    test_prior_shifts_logits()
    test_predicted_reproduces_prior_at_init()
    test_no_nan_under_bf16_autocast()
    print("ALL TESTS PASSED")
