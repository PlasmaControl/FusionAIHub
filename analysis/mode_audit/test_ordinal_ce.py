"""Unit test for src/.../e2e/ordinal_loss.py — soft/ordinal CE + tol1 metric.
No training run; CPU. Optionally checks against the FROZEN s16 codec's real codes.
Run: pixi run --frozen python analysis/mode_audit/test_ordinal_ce.py
"""
import sys
FMH = "/lustre/orion/fus187/proj-shared/ps9551/Flow/FusionAIHub"
for p in (f"{FMH}/src", f"{FMH}/scripts/training"):
    if p not in sys.path:
        sys.path.insert(0, p)
import torch
from tokamak_foundation_model.e2e.ordinal_loss import (
    build_ordinal_target, soft_ordinal_ce, tol1_codeacc, exact_codeacc)

L, eps = 16, 0.1
ok = []


def check(name, cond):
    ok.append(cond); print(f"  [{'PASS' if cond else 'FAIL'}] {name}", flush=True)


print("== (1) target distribution ==")
q = build_ordinal_target(torch.tensor([5]), L, eps)[0]
check("sums to 1", torch.allclose(q.sum(), torch.tensor(1.0), atol=1e-6))
check("interior [eps,1-2eps,eps]", torch.allclose(q[[4, 5, 6]], torch.tensor([eps, 1 - 2 * eps, eps]), atol=1e-6))
check("no mass elsewhere", q[[0, 1, 2, 3, 7, 8]].sum().item() < 1e-6)
q0 = build_ordinal_target(torch.tensor([0]), L, eps)[0]
check("edge k=0 -> [1-eps, eps]", torch.allclose(q0[[0, 1]], torch.tensor([1 - eps, eps]), atol=1e-6) and abs(q0.sum() - 1) < 1e-6)
qL = build_ordinal_target(torch.tensor([L - 1]), L, eps)[0]
check("edge k=L-1 -> [eps, 1-eps]", torch.allclose(qL[[L - 2, L - 1]], torch.tensor([eps, 1 - eps]), atol=1e-6))

print("== (2) loss behaviour ==")
codes = torch.tensor([5])
def loss_if_peak_at(j):
    lg = torch.full((1, L), -5.0); lg[0, j] = 5.0
    return soft_ordinal_ce(lg, codes, eps).item()
check("loss(peak@k) < loss(peak@k+1)", loss_if_peak_at(5) < loss_if_peak_at(6))
check("loss(peak@k+1) < loss(peak@k+3)", loss_if_peak_at(6) < loss_if_peak_at(8))
check("eps->0 approaches hard CE", abs(
    soft_ordinal_ce(torch.tensor([[0.0, 9.0] + [-9.0] * 14]), torch.tensor([1]), 1e-6).item()
    - torch.nn.functional.cross_entropy(torch.tensor([[0.0, 9.0] + [-9.0] * 14]), torch.tensor([1])).item()) < 1e-2)
lg = torch.randn(4, 7, 48, L, requires_grad=True); cc = torch.randint(0, L, (4, 7, 48))
lo = soft_ordinal_ce(lg, cc, eps); lo.backward()
check("gradient flows, finite", lg.grad is not None and torch.isfinite(lg.grad).all())
check("weighted reduction runs", torch.isfinite(soft_ordinal_ce(lg, cc, eps, weight=torch.rand(4, 7, 48))))

print("== (3) tol1 / exact metric ==")
lg = torch.full((1, 3, 1, L), -5.0); tgt = torch.tensor([[[5], [6], [9]]])   # peaks set below
lg[0, 0, 0, 5] = 5.0; lg[0, 1, 0, 7] = 5.0; lg[0, 2, 0, 12] = 5.0            # off by 0, +1, +3
check("tol1 = 2/3 (0 and +1 within tol; +3 not)", abs(tol1_codeacc(lg, tgt).item() - 2 / 3) < 1e-6)
check("exact = 1/3", abs(exact_codeacc(lg, tgt).item() - 1 / 3) < 1e-6)

print("== (4) against FROZEN s16 codec (real codes) ==")
try:
    from pathlib import Path
    import poc_fsq_stageB as poc
    from poc_fsq_stageB import load_pairs
    from spectro_bg import baseline_residual, smooth_time_mag
    from tokamak_foundation_model.e2e.quantizers.spectro_codec import load_frozen_codec
    cd = "/lustre/orion/fus187/proj-shared/models/fsq_smooth_ece_s16"
    codec, cfg = load_frozen_codec(f"{cd}/spectro_codec_ece.pt", map_location="cpu")
    poc.PATCH_F = cfg["patch_f"]; poc.PATCH_T = cfg["patch_t"]
    X, _ = load_pairs("200729", "/lustre/orion/fus187/proj-shared/foundation_model",
                      "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt",
                      cfg["C"], 4, modality="ece")
    _, R = baseline_residual(X, sigma=8.0); Rc = smooth_time_mag(R, cfg["smooth_frames"])
    codes = codec.encode_codes(Rc[:2]).long()                                   # (2,ntok,dim) real
    Lc = int(cfg["fsq_L"])
    def peak_at(k):                                                            # one-hot logits peaked at level k
        return torch.full((*codes.shape, Lc), -9.0).scatter_(-1, k.clamp(0, Lc - 1).unsqueeze(-1), 9.0)
    onehot = peak_at(codes)
    check("real codes: tol1(one-hot@true)=1.0", abs(tol1_codeacc(onehot, codes).item() - 1.0) < 1e-6)
    check("real codes: tol1(one-hot@true+1)=1.0 (wrap-free)", abs(tol1_codeacc(peak_at(codes + 1), codes).item() - 1.0) < 1e-6)
    check("real codes: tol1(one-hot@true+5) near 0", tol1_codeacc(peak_at(codes + 5), codes).item() < 0.15)
    # ordinal ORDERING: closer prediction -> lower loss; and matching the soft target beats a spike
    ce_true = soft_ordinal_ce(onehot, codes, eps).item()
    ce_far = soft_ordinal_ce(peak_at(codes + 3), codes, eps).item()
    q = build_ordinal_target(codes, Lc, eps); ce_match = soft_ordinal_ce(torch.log(q + 1e-9), codes, eps).item()
    check("real codes: soft-CE(peak@true) < soft-CE(peak@true+3)", ce_true < ce_far)
    check("real codes: soft-CE(match soft-target) < soft-CE(over-confident spike)", ce_match < ce_true)
    print(f"    (ce_match={ce_match:.3f} < ce_true={ce_true:.3f} < ce_far={ce_far:.3f})", flush=True)
except Exception as e:
    import traceback; print(f"  [SKIP] real-codec check: {e}", flush=True); traceback.print_exc()

print(f"\n{'ALL PASS' if all(ok) else 'SOME FAILED'} ({sum(ok)}/{len(ok)})", flush=True)
sys.exit(0 if all(ok) else 1)
