"""Study C flagship: LoRA-finetune the backbone for ELM prediction.

Task = the frozen-probe task, end-to-end: from one 50 ms input window,
predict whether an ELM occurs in the NEXT 50 ms window (``elm_next`` from
``study_c_make_labels.py``). Trainable parameters: LoRA adapters (rank 16
on every attention block) + a linear head on the pooled diagnostic tokens.
Everything else stays frozen; the comparison against the frozen linear
probe at matched labeled-shot budgets is the "does finetuning buy anything
beyond the frozen representation" measurement.

One process trains ONE (n_shots, seed) cell — the sbatch fans the grid out
across GCDs. A finished cell writes ``result.json`` and is skipped on
rerun. ``--aggregate`` collects every cell + the frozen-probe curves from
``tables/c_sample_efficiency.npz`` into the overlay figure
``figures/study_c/c2b_sample_efficiency_lora``.

During-training model selection uses a fixed --eval-shots subset of val
(full-val forward passes are ~20 min each on one GCD); the same subset is
used for the final number and recorded in the json.

    python scripts/evaluation/study_c_elm_lora.py --n-shots 50 --seed 0
    python scripts/evaluation/study_c_elm_lora.py --aggregate
"""

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))
sys.path.insert(0, str(_HERE.parent))
sys.path.insert(0, str(_HERE.parents[2] / "scripts" / "training"))

from torch.utils.data import DataLoader  # noqa: E402

from tokamak_foundation_model.data.data_loader import (  # noqa: E402
    collate_fn_prediction,
)
from tokamak_foundation_model.e2e.lora import (  # noqa: E402
    apply_lora_to_backbone,
    freeze_non_lora_parameters,
)

from study_b_layerwise import prep_inputs  # noqa: E402
from tfm_eval.ckpt import DEFAULT_CKPT, build_model_from_ckpt, load_ckpt  # noqa: E402
from tfm_eval.data import (  # noqa: E402
    assert_val_matches_training_cache,
    load_stats,
    make_prediction_dataset,
    matched_train_subset,
    resolve_split,
    shot_window_ranges,
)
from tfm_eval.probing import subsample_shots  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", default=DEFAULT_CKPT)
    ap.add_argument("--n-shots", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lora-rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--lr-lora", type=float, default=2e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--eval-shots", type=int, default=200)
    ap.add_argument("--num-workers", type=int, default=5)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--aggregate", action="store_true",
                    help="render the probe-vs-LoRA overlay from run jsons")
    ap.add_argument(
        "--out-root",
        default=str(_HERE.parents[2] / "data" / "outputs" / "eval_suite"
                    / "e2e_stage1_best"),
    )
    return ap.parse_args()


def load_window_labels(labels_csv):
    """{(shot, '%.2f' % t): (elm_next, in_range)} from a windows csv."""
    table = {}
    with open(labels_csv) as f:
        for row in csv.DictReader(f):
            key = (int(row["shot"]), f"{float(row['t_start_s']):.2f}")
            table[key] = (float(row["elm_next"]), int(row["elm_in_range"]))
    return table


def labeled_indices(ds, warmup_s, stride_s, table):
    """(global_idx, label) for windows with a usable elm_next label."""
    out = []
    for path, start, n in shot_window_ranges(ds):
        shot = int(Path(path).stem.split("_")[0])
        for li in range(n):
            t = warmup_s + li * stride_s
            row = table.get((shot, f"{t:.2f}"))
            if row is None:
                continue
            y, in_range = row
            if in_range and np.isfinite(y):
                out.append((start + li, float(y)))
    return out


class ElmLoraModel(nn.Module):
    """Frozen-but-for-LoRA backbone + linear head on pooled diag tokens."""

    def __init__(self, model, rank, d_model):
        super().__init__()
        apply_lora_to_backbone(model.backbone, rank=rank)
        freeze_non_lora_parameters(model)
        if hasattr(model.backbone, "grad_checkpoint"):
            model.backbone.grad_checkpoint = True
        self.model = model
        self.head = nn.Linear(d_model, 1)

    def forward(self, batch, device):
        diag_in, act_in = prep_inputs(self.model, batch, device)
        bsz = next(iter(act_in.values())).shape[0]
        step = torch.zeros(bsz, dtype=torch.long, device=device)
        toff = torch.zeros(bsz, device=device)
        tokens = self.model.tokenize(diag_in, act_in)
        out = self.model.backbone(tokens, step, toff)
        pooled = out[:, : self.model.n_diag_tokens].mean(dim=1)
        return self.head(pooled).squeeze(-1)


def make_loader(ds, pairs, batch_size, num_workers, shuffle, seed=0):
    idx = [g for g, _ in pairs]
    if shuffle:
        rng = np.random.default_rng(seed)
        idx = [idx[i] for i in rng.permutation(len(idx))]
    return DataLoader(
        ds, batch_size=batch_size, sampler=idx, num_workers=num_workers,
        collate_fn=collate_fn_prediction, pin_memory=True,
        prefetch_factor=2 if num_workers > 0 else None,
    )


@torch.no_grad()
def evaluate(net, ds, pairs, device, batch_size, num_workers, t0, tag):
    from sklearn.metrics import average_precision_score, roc_auc_score

    net.eval()
    loader = make_loader(ds, pairs, batch_size, num_workers, shuffle=False)
    ys = np.array([y for _, y in pairs])
    scores = []
    for batch in loader:
        scores.append(net(batch, device).float().cpu().numpy())
    s = np.concatenate(scores)
    net.train()
    auroc = roc_auc_score(ys, s) if len(np.unique(ys)) > 1 else float("nan")
    ap = (average_precision_score(ys, s)
          if len(np.unique(ys)) > 1 else float("nan"))
    print(f"[{time.time()-t0:7.1f}s] {tag}: AUROC {auroc:.4f} AP {ap:.4f} "
          f"(n={len(ys)}, prev={ys.mean():.3f})", flush=True)
    return {"auroc": float(auroc), "ap": float(ap), "n": int(len(ys)),
            "prevalence": float(ys.mean())}


def aggregate(args):
    """Overlay LoRA cells on the frozen-probe sample-efficiency curves."""
    import matplotlib.pyplot as plt

    from tfm_eval.plotting import OKABE_ITO, save_fig, set_style

    root = Path(args.out_root)
    runs = sorted(root.glob("study_c/lora/run_n*_s*/result.json"))
    if not runs:
        print("no finished LoRA runs to aggregate")
        return 1
    cells = [json.loads(p.read_text()) for p in runs
             if not json.loads(p.read_text()).get("degenerate")]
    eff = np.load(root / "tables" / "c_sample_efficiency.npz")
    if "elm_next_latent" not in eff.files:
        print(f"c_sample_efficiency.npz keys {eff.files} — no "
              "'elm_next_latent'; check study_c_probes output naming")
        return 1
    set_style()
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    ns = eff["ns"]
    lat = eff["elm_next_latent"]  # (len(ns), n_seeds) AUROC
    med = np.nanmedian(lat, axis=1)
    lo, hi = np.nanpercentile(lat, [25, 75], axis=1)
    ax.plot(ns, med, color=OKABE_ITO[0], lw=1.8, marker="o", ms=3.5,
            label="frozen linear probe")
    ax.fill_between(ns, lo, hi, color=OKABE_ITO[0], alpha=0.2, lw=0)
    by_n = {}
    for c in cells:
        by_n.setdefault(c["n_shots"], []).append(c["final"]["auroc"])
    xs = sorted(by_n)
    med_l = [float(np.nanmedian(by_n[n])) for n in xs]
    ax.plot(xs, med_l, color=OKABE_ITO[1], lw=1.8, marker="s", ms=4,
            label="LoRA finetune")
    for n in xs:
        ax.plot([n] * len(by_n[n]), by_n[n], "s", ms=2.5,
                color=OKABE_ITO[1], alpha=0.5)
    if "elm_next_persistence_auroc" in eff.files:
        ax.axhline(float(eff["elm_next_persistence_auroc"]), color="0.5",
                   lw=1.0, ls=":", label="persistence")
    ax.set_xscale("log", base=2)
    ax.set_xticks(sorted(set(ns.tolist()) | set(xs)))
    ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax.set_xlabel("labeled shots")
    ax.set_ylabel("val AUROC (elm_next)")
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    ax.set_title("ELM prediction: frozen probe vs LoRA finetune",
                 fontsize=10)
    save_fig(fig, root / "figures" / "study_c"
             / "c2b_sample_efficiency_lora")
    print(f"aggregated {len(cells)} LoRA cells over n_shots={xs}")
    return 0


def main() -> int:
    args = parse_args()
    if args.aggregate:
        return aggregate(args)
    t0 = time.time()
    root = Path(args.out_root)
    run_dir = root / "study_c" / "lora" / f"run_n{args.n_shots}_s{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "result.json").exists():
        print(f"{run_dir}/result.json exists — done, skipping")
        return 0
    device = torch.device(args.device)
    torch.manual_seed(args.seed)

    ckpt = load_ckpt(args.checkpoint)
    ck_args = ckpt.get("args", {}) or {}
    model, diagnostics, actuators = build_model_from_ckpt(
        ckpt, device=args.device
    )
    net = ElmLoraModel(model, args.lora_rank, ck_args["d_model"]).to(device)
    n_train_p = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"[{time.time()-t0:6.1f}s] LoRA rank {args.lora_rank}: "
          f"{n_train_p/1e6:.2f}M trainable")

    train_files, val_files = resolve_split(
        ck_args.get("data_dir", "/lustre/orion/fus187/proj-shared/foundation_model"),
        seed=ck_args.get("seed", 42),
        val_fraction=ck_args.get("val_fraction", 0.1),
    )
    assert_val_matches_training_cache(val_files)
    train_files = matched_train_subset(train_files)
    stats = load_stats(ck_args["stats_path"])
    warmup_s = ck_args.get("warmup_s", 1.0)
    stride_s = 0.25

    labels_dir = root / "labels"
    tr_table = load_window_labels(labels_dir / "windows_train.csv")
    va_table = load_window_labels(labels_dir / "windows_val.csv")

    shot_of = {f: int(Path(f).stem.split("_")[0]) for f in train_files}
    subset = subsample_shots(
        np.array([shot_of[f] for f in train_files]), args.n_shots, args.seed
    )
    tr_files = [f for f in train_files if shot_of[f] in subset]
    va_files = val_files[: args.eval_shots]

    ds_tr = make_prediction_dataset(
        tr_files, stats, ck_args, diagnostics, actuators,
        step_size_s=stride_s,
        lengths_cache_path=run_dir / "lengths_tr.pt", max_open_files=128,
    )
    # per-run cache: parallel GCD cells must not race on one file
    ds_va = make_prediction_dataset(
        va_files, stats, ck_args, diagnostics, actuators,
        step_size_s=stride_s,
        lengths_cache_path=run_dir / f"lengths_va_{args.eval_shots}.pt",
        max_open_files=220,
    )
    tr_pairs = labeled_indices(ds_tr, warmup_s, stride_s, tr_table)
    va_pairs = labeled_indices(ds_va, warmup_s, stride_s, va_table)
    y_tr = np.array([y for _, y in tr_pairs])
    if not len(tr_pairs) or len(np.unique(y_tr)) < 2:
        print("degenerate train subset (no labeled windows or one class)")
        (run_dir / "result.json").write_text(json.dumps(
            {"n_shots": args.n_shots, "seed": args.seed,
             "degenerate": True}))
        return 0
    print(f"[{time.time()-t0:6.1f}s] train {len(tr_pairs)} windows "
          f"({len(tr_files)} shots, prev {y_tr.mean():.3f}); "
          f"eval {len(va_pairs)} windows ({len(va_files)} shots)")

    pos_w = torch.tensor([(1.0 - y_tr.mean()) / max(y_tr.mean(), 1e-3)],
                         device=device)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW([
        {"params": [p for n_, p in net.model.named_parameters()
                    if p.requires_grad], "lr": args.lr_lora},
        {"params": net.head.parameters(), "lr": args.lr_head},
    ], weight_decay=0.01)

    best = {"auroc": -1.0}
    best_state = None
    net.train()
    for epoch in range(args.epochs):
        # shuffle ONCE per epoch; loader and labels read the same list
        ep_pairs = _shuffled_pairs(tr_pairs, args.seed * 100 + epoch)
        loader = make_loader(ds_tr, ep_pairs, args.batch_size,
                             args.num_workers, shuffle=False)
        y_iter = iter([y for _, y in ep_pairs])
        running, n_done = 0.0, 0
        for bi, batch in enumerate(loader):
            bsz = next(iter(batch["inputs"].values())).shape[0]
            ys = torch.tensor([next(y_iter) for _ in range(bsz)],
                              device=device, dtype=torch.float32)
            logit = net(batch, device)
            loss = loss_fn(logit, ys)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in net.parameters() if p.requires_grad], 1.0)
            opt.step()
            running += float(loss); n_done += bsz
            if bi % 50 == 0:
                print(f"[{time.time()-t0:7.1f}s] ep{epoch} "
                      f"{n_done}/{len(tr_pairs)} loss "
                      f"{running/(bi+1):.4f}", flush=True)
        m = evaluate(net, ds_va, va_pairs, device, args.batch_size,
                     args.num_workers, t0, f"ep{epoch} val")
        if np.isfinite(m["auroc"]) and m["auroc"] > best["auroc"]:
            best = dict(m, epoch=epoch)
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in net.state_dict().items()
                if ".lora_" in k or k.startswith("head.")
            }

    if best_state is not None:
        torch.save(best_state, run_dir / "lora_head_best.pt")
    result = {
        "n_shots": args.n_shots, "seed": args.seed,
        "lora_rank": args.lora_rank, "epochs": args.epochs,
        "trainable_params": int(n_train_p),
        "train_windows": len(tr_pairs),
        "train_prevalence": float(y_tr.mean()),
        "eval_shots": len(va_files), "final": best,
        "note": f"eval on first {len(va_files)} val shots "
                "(fixed subset, same for every cell)",
    }
    (run_dir / "result.json").write_text(json.dumps(result, indent=1))
    print(f"[{time.time()-t0:7.1f}s] DONE — best "
          f"AUROC {best['auroc']:.4f} (epoch {best.get('epoch')}) "
          f"→ {run_dir}/result.json")
    return 0


def _shuffled_pairs(pairs, seed):
    rng = np.random.default_rng(seed)
    return [pairs[i] for i in rng.permutation(len(pairs))]


if __name__ == "__main__":
    sys.exit(main())
