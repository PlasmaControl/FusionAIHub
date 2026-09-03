"""Isolation test: CAN the mask head fit 200729's modes at all?

Freezes the backbone, grabs ONE batch of mode-bearing 200729 windows, and
optimizes ONLY the mask head to fit the target mode-mask — under several loss
variants. This separates three hypotheses for the stuck-at-0.1 overfit:

  * If NO loss can drive maskdice high on a single fixed batch  → the frozen
    backbone forecast tokens don't carry mode info (token bottleneck), OR the
    head architecture can't represent it.
  * If sparse/tversky fits but dice-only doesn't                → loss geometry
    (soft-dice has a vanishing gradient at the diffuse init) is the culprit.
  * If everything fits on one batch                             → the head/loss
    are fine; the real-run problem is cross-batch / backbone-token variation.

Run:  EVAL_CKPT=<overfit latest.pt> sbatch scripts/slurm_frontier/eval_poc_modemask.sh
      (set EVAL_MODE=maskfit to dispatch here; see the sbatch)
"""
import os
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from tokamak_foundation_model.data.data_loader import collate_fn
from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
from torch.utils.data import DataLoader
from train_e2e_stage1 import (
    forward_batch, _spec_mode_arg, _SPEC_STRUCT_GAMMA, _SPEC_STRUCT_CUT,
    _SPEC_STRUCT_K, spectro_mask_loss,
)
from eval_e2e_animation_tokamak import load_model


def _hard(x, k):
    return (_spec_mode_arg(x, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA
            > _SPEC_STRUCT_CUT).float()


def main():
    ckpt_path = Path(os.environ["EVAL_CKPT"])
    data_dir = Path(os.environ.get(
        "EVAL_DATA_DIR", "/lustre/orion/fus187/proj-shared/foundation_model"))
    stats_path = os.environ.get(
        "EVAL_STATS",
        "/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    shot = os.environ.get("EVAL_SHOTS", "200729").split(",")[0].strip()
    steps = int(os.environ.get("FIT_STEPS", "800"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, ckpt = load_model(ckpt_path, device)
    model.eval()
    diag = [d.name for d in model.diagnostics]
    act = [a.name for a in model.actuators]
    spec_mods = [d.name for d in model.diagnostics
                 if getattr(model.diag_heads[d.name], "enable_mask", False)]
    m = spec_mods[0]
    head = model.diag_heads[m]
    k = _SPEC_STRUCT_K.get(m, 2.0)
    has_feat = getattr(head, "enable_input_feat", False)
    has_cond = getattr(head, "enable_input_cond", False)
    print(f"[maskfit] shot={shot} modality={m} input_feat={has_feat} "
          f"input_cond={has_cond} steps={steps}")

    stats = torch.load(stats_path, weights_only=False)
    ds = TokamakMultiFileDataset(
        hdf5_paths=[data_dir / f"{shot}_processed.h5"], chunk_duration_s=0.05,
        prediction_mode=True, prediction_horizon_s=0.05, step_size_s=0.01,
        warmup_s=1.0, n_fft=1024, hop_length=256, preprocessing_stats=stats,
        input_signals=diag, target_signals=diag + act)
    loader = DataLoader(ds, batch_size=64, shuffle=False, collate_fn=collate_fn,
                        num_workers=2)

    # Scan several batches and keep the STRONGEST-mode one (highest mean-per-
    # window persistence). 200729's modes are concentrated in a minority of
    # dense windows, so the first mode-bearing batch is usually weak and
    # uninterpretable — we want the batch where modes are clearly present.
    def _persist_pw(prior_h, gt_h):
        gs = gt_h.sum(dim=(-2, -1)); m_ = gs >= 3
        if int(m_.sum()) == 0:
            return -1.0
        ov = (prior_h * gt_h).sum(dim=(-2, -1))
        d = (2 * ov + 1e-6) / (prior_h.sum(dim=(-2, -1)) + gs + 1e-6)
        return float(d[m_].mean())

    tok_f = tgt_f = prior_f = None
    best_p = -1.0
    with torch.no_grad():
        for bi, batch in enumerate(loader):
            if bi >= 12:
                break
            preds, diag_inputs, targets, masks, tok = forward_batch(
                model, batch, device)
            gt = _hard(targets[m].float(), k)
            per = _hard(diag_inputs[m].float(), k)
            p = _persist_pw(per, gt)
            if p > best_p:
                best_p = p
                tok_f = tok[m].detach().clone()
                tgt_f = targets[m].float().detach().clone()
                prior_f = per.detach().clone()
    if tok_f is None:
        print("[maskfit] ERROR: no mode-bearing batch found"); return
    gt = _hard(tgt_f, k)
    dens = float(gt.mean())
    # persistence ceiling on THIS batch — per-window mean AND pooled
    gsum = gt.sum(dim=(-2, -1)); mb = gsum >= 3
    povl = (prior_f * gt).sum(dim=(-2, -1))
    pdice = ((2 * povl + 1e-6) / (prior_f.sum(dim=(-2, -1)) + gsum + 1e-6))
    persist = float(pdice[mb].mean())
    persist_pool = float((2 * povl[mb].sum() + 1e-6)
                         / (prior_f.sum(dim=(-2, -1))[mb].sum() + gsum[mb].sum() + 1e-6))
    print(f"[maskfit] STRONGEST batch: {tok_f.shape[0]} windows, density={dens:.4f}, "
          f"persistence per-win={persist:.3f} pooled={persist_pool:.3f}")

    def maskdice(logits):
        """Returns (per-window-mean, pooled) dice on mode-bearing windows."""
        p = (torch.sigmoid(logits) > 0.5).float()
        ov = (p * gt).sum(dim=(-2, -1))
        d = (2 * ov + 1e-6) / (p.sum(dim=(-2, -1)) + gsum + 1e-6)
        pw = float(d[mb].mean())
        # pooled = pixel-count-weighted (matches the offline 0.56 measurement)
        pool = float((2 * ov[mb].sum() + 1e-6)
                     / (p.sum(dim=(-2, -1))[mb].sum() + gsum[mb].sum() + 1e-6))
        return pw, pool

    prior = prior_f if (has_feat or has_cond) else None
    import copy
    variants = [
        ("sparse (current)", dict(loss_type="sparse", bce_weight=1.0)),
        ("dice-only", dict(loss_type="dice")),
        ("tversky(.5,.5)", dict(loss_type="tversky", tversky_alpha=0.5, tversky_beta=0.5)),
        ("tversky(.3,.7)", dict(loss_type="tversky", tversky_alpha=0.3, tversky_beta=0.7)),
    ]
    orig_state = copy.deepcopy(head.state_dict())
    for vname, lkw in variants:
        head.load_state_dict(orig_state)      # fresh mask head each variant
        # optimize ONLY the mask-branch params
        mask_params = [p for n, p in head.named_parameters()
                       if any(t in n for t in ("mask_unembed", "mask_decode",
                                               "mask_pre", "mask_prior_gain"))]
        opt = torch.optim.Adam(mask_params, lr=3e-3)
        pw0, pl0 = maskdice(head.mask_logits(tok_f, prior=prior).float())
        for s in range(steps):
            opt.zero_grad()
            logits = head.mask_logits(tok_f, prior=prior)
            # pass the RAW target — spectro_mask_loss binarizes internally
            loss, md = spectro_mask_loss(logits, tgt_f, k, **lkw)
            loss.backward()
            opt.step()
        pwF, plF = maskdice(head.mask_logits(tok_f, prior=prior).float())
        # a genuine FIT = pooled dice clears persistence by a real margin AND
        # reaches a usable absolute value (memorizing a FIXED batch should be easy)
        verdict = "FITS ✓" if (plF > persist_pool + 0.1 and plF > 0.45) else (
            "weak" if plF > persist_pool + 0.05 else "STUCK ✗")
        print(f"[maskfit] {vname:>18}: pooled {pl0:.3f}->{plF:.3f}  "
              f"per-win {pw0:.3f}->{pwF:.3f}  (persist pooled {persist_pool:.3f}) {verdict}")
    head.load_state_dict(orig_state)


if __name__ == "__main__":
    main()
