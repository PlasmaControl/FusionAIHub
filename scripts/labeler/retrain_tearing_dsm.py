"""Continue training the shipped tearing-survival DSM from its own weights.

The shipped checkpoint (`rt_fixed_rot.pkl`, 2024-11) stopped after 20 epochs at
lr 1e-5 with its validation NLL still falling monotonically, 0.534 -> 0.471:
auton-survival's early stop never fired, so the fit ended on its `iters`
setting rather than on convergence. This script picks that fit up where it
stopped and runs it until the fork's own patience rule ends it.

**Which split the checkpoint was fitted on, measured not assumed.** The config
that names this model (`/projects/EKOLEMEN/survival_tm/outputs/rt_fixed_rotconfig`)
has no `database_shots_list_name`, and the parameter dict inside the pickle has
no `seed`, so the by-shot 80/10/10 split in today's `train_tm_model.py` cannot
be the one that ran in 2024-11. Searching the plausible rules against the
pickle's own stored final validation loss (0.4713212222893277) found the answer
exactly (`runs/task4_split_search.py`, |diff| < 1e-15): the whole 914,898-row
dataset went into `SurvivalModel.fit` with **no `val_data`**, so the estimator
took its own default split,

    data_train = data.sample(frac=1 - 0.15, random_state=0)   # 777,663 rows
    data_val   = data[~data.index.isin(data_train.index)]     # 137,235 rows

which is a **row-level** split: 8,685 of the 8,690 shots with a validation row
also have training rows. The stored validation curve is therefore optimistic,
and so is every continuation number measured on the same rows. It is still the
right split to continue on - it is the only one on which "before" and "after"
mean the same thing - and the honest out-of-sample comparison is labeler's
own 500-shot pool, of which 214 shots contributed training rows. The card says
all of this.

What the script does, in order:

1. reads the upstream array names (x, e, t, shots) from the config, loads them
   from `/projects/EKOLEMEN/survival_tm_2/data/`;
2. rebuilds the split: `--split checkpoint` (default) is the estimator rule
   above; `--split by_shot` is `train_tm_model.py` lines 55-85 (`np.unique`,
   `default_rng(seed)`, 80% of the unique shots then 10% of the remainder),
   kept because it is the leak-free rule and its NLL is printed either way as
   a diagnostic;
3. unpickles the shipped checkpoint with the group's auton-survival fork on
   `sys.path`, so the real classes come back;
4. **gates** on `compute_nll` over the rebuilt validation split reproducing the
   pickle's own stored final validation loss. A mismatch means the split or
   the x file is not the one the checkpoint was fitted on, and every number
   after it would be incomparable, so the script exits non-zero;
5. continues `utilities.train_dsm`'s loop from the existing `torch_model`
   with **no pretraining** - `train_dsm` would first run `pretrain_dsm` and
   overwrite the learned per-component `shape`/`scale` parameters with a
   1-dimensional fit, throwing away part of what the checkpoint learned. Only
   the part from `model.double()` on is reproduced: Adam at `--lr`, per-epoch
   `sklearn.utils.shuffle(..., random_state=epoch)`, `conditional_loss(...,
   elbo=True)` on minibatches, full-set validation `conditional_loss(...,
   elbo=False)`, every epoch's `state_dict` kept, the argmin reloaded at the
   end, and the same "no improvement for `patience` epochs" stop;
6. saves `[[model, train_losses, val_losses, params]]` - the shape upstream's
   own pickle has, so `labeler.models.runners.dsm_pickle.load_dsm` reads it
   - plus `training.json`, `PROVENANCE.json`, `loss_curve.png`, and a copy of
   the normalisation constants beside the weights.

Two continuation choices are recorded rather than hidden:

* the epoch counter continues from the checkpoint's last epoch (20), so the
  per-epoch shuffle seeds are 20, 21, ... and no epoch repeats an ordering the
  shipped fit already used;
* Adam's moment estimates are not in the pickle, so the optimiser restarts
  cold. That is a real difference from an uninterrupted run.

Runs in the Phase 3 uv venv (`scripts/labeler/make_phase3_env.sh`), not in
the labeler pixi env, and imports nothing from `labeler`.
"""
from __future__ import annotations

import argparse
import configparser
import hashlib
import json
import os
import pickle
import socket
import sys
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from labeler.env import getenv

DATA_DIR = Path("/projects/EKOLEMEN/survival_tm_2/data")
CFG = Path("/projects/EKOLEMEN/survival_tm/outputs/rt_fixed_rotconfig")
# The config names x, e and t but not the shots list: the trainer version that
# ran in 2024-11 took its names on the command line (outputs/rt_fixed_rotjob.slurm).
# This is the matching filtered list for that x file (same 914,898 rows).
SHOTS_NAME = "rt_filtered_shots_pcb_rot"
CHECKPOINT = Path(
    "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/"
    "d3d_tearing_time_to_event_dsm/rt_fixed_rot.pkl"
)
NORMALIZATIONS = CHECKPOINT.with_name("rt_normalizations_dict.pkl")
FORK = Path("/projects/EKOLEMEN/survival_tm_2/train_models/auton-survival")
GATE_TOL = 2e-3
VSIZE = 0.15          # SurvivalModel.fit's default, and what this model got


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def array_names(cfg_path: Path) -> dict:
    """The upstream array names, from the config that produced the model."""
    cfg = configparser.ConfigParser()
    cfg.read(cfg_path)
    model = cfg["model"]
    if model["output_filename_base"] != "rt_fixed_rot":
        raise SystemExit(f"{cfg_path}: not the rt_fixed_rot config")
    return {
        "x": model["database_x_name"],
        "e": model["database_e_name"],
        "t": model["database_t_name"],
        "shots": model.get("database_shots_list_name", SHOTS_NAME),
    }


def load_array(name: str) -> np.ndarray:
    with open(DATA_DIR / f"{name}.pkl", "rb") as fh:
        return np.array(pickle.load(fh))


def split_checkpoint(n_rows: int, seed: int, vsize: float = VSIZE) -> tuple:
    """`SurvivalModel.fit`'s own split: a random 1 - vsize row sample.

    Reproduced on an index frame rather than on the joined 40-column frame:
    `DataFrame.sample` draws row positions from `random_state` alone, so the
    columns do not enter, and this keeps a 293 MB copy out of memory. Verified
    against the checkpoint's stored validation loss to machine precision.
    """
    frame = pd.DataFrame(np.arange(n_rows), columns=["i"])
    train = frame.sample(frac=1.0 - vsize, random_state=seed)
    valid = frame[~frame.index.isin(train.index)]
    return train["i"].to_numpy(), valid["i"].to_numpy()


def split_by_shot(shots_list: np.ndarray, seed: int) -> dict:
    """`train_tm_model.py` lines 55-85: 80/10/10 by shot, leak free."""
    unique_shots = np.unique(shots_list)
    first = [np.where(shots_list == shot)[0][0] for shot in unique_shots]
    n_rows = len(shots_list)
    index_of = {shot: np.arange(first[i], first[i + 1] if i < len(first) - 1 else n_rows)
                for i, shot in enumerate(unique_shots)}
    rng = np.random.default_rng(seed)
    n_unique = len(unique_shots)
    shots_train = rng.choice(unique_shots, size=int(n_unique * 0.8), replace=False)
    shots_valid = rng.choice(np.setdiff1d(unique_shots, shots_train),
                             size=int(n_unique * 0.1), replace=False)
    shots_test = np.setdiff1d(unique_shots, np.concatenate((shots_train, shots_valid)))
    return {
        "train": np.concatenate([index_of[s] for s in shots_train]),
        "valid": np.concatenate([index_of[s] for s in shots_valid]),
        "test": np.concatenate([index_of[s] for s in shots_test]),
        "shots": {"unique": n_unique, "train": len(shots_train),
                  "valid": len(shots_valid), "test": len(shots_test)},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=1000)
    ap.add_argument("--max-epochs", type=int, default=300)
    ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--fork", type=Path, default=FORK)
    ap.add_argument("--checkpoint", type=Path, default=CHECKPOINT)
    ap.add_argument("--cfg", type=Path, default=CFG)
    ap.add_argument("--split", choices=("checkpoint", "by_shot"), default="checkpoint",
                    help="'checkpoint' is the split this model was fitted on and the "
                         "one the gate reproduces; 'by_shot' is leak free but its "
                         "validation loss is not comparable to the stored curve")
    args = ap.parse_args()

    t0 = time.time()
    sys.path.insert(0, str(args.fork))
    # Imported HERE and not at module scope: `auton_survival` only exists on
    # `sys.path` once `--fork` has been read, and these have to be the
    # fork's copies. (`PLC0415` is not an enabled rule, so there is nothing
    # to suppress; this comment is the reason the suppression carried.)
    import torch
    from auton_survival.models.dsm.losses import conditional_loss
    from auton_survival.models.dsm.utilities import (
        _reshape_tensor_with_nans,
        get_optimizer,
    )
    from sklearn.utils import shuffle

    names = array_names(args.cfg)
    print(f"arrays: {names}", flush=True)
    x = load_array(names["x"]).astype(np.float64)
    e = load_array(names["e"]).astype(np.float64)
    t = load_array(names["t"]).astype(np.float64)
    shots_list = load_array(names["shots"])
    n_rows = len(x)
    print(f"x {x.shape} e {e.shape} t {t.shape} shots {shots_list.shape} "
          f"({np.unique(shots_list).size} unique), events {e.mean():.4%}", flush=True)
    if not (n_rows == len(e) == len(t) == len(shots_list)):
        raise SystemExit("x, e, t and the shots list disagree on row count")

    ckpt_train, ckpt_valid = split_checkpoint(n_rows, args.seed)
    shot_split = split_by_shot(shots_list, args.seed)
    if args.split == "checkpoint":
        idx_train, idx_valid = ckpt_train, ckpt_valid
    else:
        idx_train, idx_valid = shot_split["train"], shot_split["valid"]
    leak = np.intersect1d(np.unique(shots_list[idx_valid]),
                          np.unique(shots_list[idx_train])).size
    print(f"split '{args.split}': train {idx_train.size} rows, valid {idx_valid.size} "
          f"rows; {leak} shots appear on both sides", flush=True)

    with open(args.checkpoint, "rb") as fh:
        payload = pickle.load(fh)
    survival_model, train_losses_0, val_losses_0, params_0 = payload[0]
    dsm = survival_model._model
    print(f"checkpoint {args.checkpoint}\n  params {params_0}\n"
          f"  stored val NLL: first {val_losses_0[0]:.6f} last {val_losses_0[-1]:.6f} "
          f"({len(val_losses_0)} epochs)", flush=True)

    # ---- gate: the rebuilt validation split must reproduce the stored loss ----
    x_valid, t_valid, e_valid = x[idx_valid], t[idx_valid], e[idx_valid]
    nll = float(dsm.compute_nll(x_valid, t_valid, e_valid))
    stored = float(val_losses_0[-1])
    print(f"GATE: compute_nll on the rebuilt validation split {nll:.6f} vs "
          f"stored final validation loss {stored:.6f} (|diff| {abs(nll - stored):.2e}, "
          f"tolerance {GATE_TOL:.0e})", flush=True)
    if abs(nll - stored) > GATE_TOL:
        print("GATE FAILED: the split or the x file is not the one this checkpoint "
              "was fitted on; nothing computed after this would be comparable.",
              file=sys.stderr, flush=True)
        return 1
    print("GATE PASSED", flush=True)
    shot_valid_nll_before = float(dsm.compute_nll(x[shot_split["valid"]],
                                                  t[shot_split["valid"]],
                                                  e[shot_split["valid"]]))
    print(f"diagnostic: NLL on the by-shot valid shots {shot_valid_nll_before:.6f} "
          "(not leak free for this checkpoint: it trained on 85% of those shots' rows)",
          flush=True)

    # ---- tensors, as auton_survival's _preprocess_training_data builds them ----
    x_train, t_train, e_train = x[idx_train], t[idx_train], e[idx_train]
    order = list(range(x_train.shape[0]))
    np.random.seed(args.seed)
    np.random.shuffle(order)
    x_train = torch.from_numpy(x_train[order]).double()
    t_train = torch.from_numpy(t_train[order]).double()
    e_train = torch.from_numpy(e_train[order]).double()
    x_valid_t = torch.from_numpy(x_valid).double()
    t_valid_ = _reshape_tensor_with_nans(torch.from_numpy(t_valid).double())
    e_valid_ = _reshape_tensor_with_nans(torch.from_numpy(e_valid).double())

    # ---- the train_dsm loop, from `model.double()` on ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    model = dsm.torch_model
    model.double()
    model.train()   # numerically a no-op: this graph has no dropout or norm layers
    optimizer = get_optimizer(model, args.lr)
    start_epoch = len(val_losses_0)
    nbatches = int(x_train.shape[0] / args.batch_size) + 1

    dics, costs, train_losses = [], [], []
    patience, oldcost = 0, stored
    epochs_run, stopped_early = 0, False
    for step in range(args.max_epochs):
        epoch = start_epoch + step
        x_train, t_train, e_train = shuffle(x_train, t_train, e_train, random_state=epoch)
        train_loss = 0.0
        for j in range(nbatches):
            xb = x_train[j * args.batch_size:(j + 1) * args.batch_size]
            tb = t_train[j * args.batch_size:(j + 1) * args.batch_size]
            eb = e_train[j * args.batch_size:(j + 1) * args.batch_size]
            if xb.shape[0] == 0:
                continue
            optimizer.zero_grad()
            loss = 0
            for r in range(model.risks):
                loss += conditional_loss(model, xb, _reshape_tensor_with_nans(tb),
                                         _reshape_tensor_with_nans(eb),
                                         elbo=True, risk=str(r + 1))
            train_loss += float(loss)
            loss.backward()
            optimizer.step()
        train_losses.append(train_loss)

        valid_loss = 0
        for r in range(model.risks):
            valid_loss += conditional_loss(model, x_valid_t, t_valid_, e_valid_,
                                           elbo=False, risk=str(r + 1))
        valid_loss = float(valid_loss.detach().cpu().numpy())
        costs.append(valid_loss)
        dics.append(deepcopy(model.state_dict()))
        epochs_run += 1
        print(f"epoch {epoch} train {train_loss:.4f} valid {valid_loss:.6f} "
              f"patience {patience} [{time.time() - t0:.0f}s]", flush=True)

        if costs[-1] >= oldcost:
            if patience == args.patience:
                stopped_early = True
                break
            patience += 1
        else:
            patience = 0
        oldcost = costs[-1]

    best = int(np.argmin(costs))
    model.load_state_dict(dics[best])
    model.eval()
    del dics
    print(f"best continuation epoch {start_epoch + best} (validation NLL "
          f"{costs[best]:.6f}); stopped early: {stopped_early}", flush=True)
    shot_valid_nll_after = float(dsm.compute_nll(x[shot_split["valid"]],
                                                 t[shot_split["valid"]],
                                                 e[shot_split["valid"]]))
    print(f"diagnostic: NLL on the by-shot valid shots {shot_valid_nll_before:.6f} "
          f"-> {shot_valid_nll_after:.6f}", flush=True)

    # ---- save, in upstream's pickle shape ----
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    params = dict(params_0)
    params.update({
        "iters": int(args.max_epochs), "learning_rate": float(args.lr),
        "batch_size": int(args.batch_size), "seed": int(args.seed),
        "continued_from": str(args.checkpoint), "continued_from_epochs": int(start_epoch),
        "continuation_patience": int(args.patience), "pretrained": False,
        "split": args.split,
    })
    train_all = [float(v) for v in train_losses_0] + [float(v) for v in train_losses]
    val_all = [float(v) for v in val_losses_0] + [float(v) for v in costs]
    weights = out_dir / "rt_fixed_rot_continued.pkl"
    tmp = weights.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump([[survival_model, train_all, val_all, params]], fh)
    os.replace(tmp, weights)
    if NORMALIZATIONS.resolve() != (out_dir / NORMALIZATIONS.name).resolve():
        (out_dir / NORMALIZATIONS.name).write_bytes(NORMALIZATIONS.read_bytes())

    wall_s = time.time() - t0
    job_id = os.environ.get("SLURM_JOB_ID")
    training = {
        "slug": "d3d_tearing_time_to_event_dsm_continued",
        "continued_from": {"path": str(args.checkpoint),
                           "sha256": sha256_of(args.checkpoint),
                           "epochs": int(start_epoch), "val_nll_final": stored,
                           "params": dict(params_0)},
        "gate": {"compute_nll_rebuilt_split": nll, "stored_val_loss": stored,
                 "abs_diff": abs(nll - stored), "tolerance": GATE_TOL},
        "epochs_run": epochs_run, "max_epochs": int(args.max_epochs),
        "stopped_early": stopped_early, "patience": int(args.patience),
        "best_epoch_overall": int(start_epoch + best),
        "best_epoch_in_continuation": best,
        "val_nll_before": stored, "val_nll_after": float(costs[best]),
        "val_nll_curve": [float(v) for v in costs],
        "train_loss_curve": [float(v) for v in train_losses],
        "by_shot_valid_nll_before": shot_valid_nll_before,
        "by_shot_valid_nll_after": shot_valid_nll_after,
        "lr": float(args.lr), "batch_size": int(args.batch_size), "seed": int(args.seed),
        "optimizer": "Adam (moments restarted: the checkpoint carries no optimiser state)",
        "elbo": True, "shuffle_random_state": "epoch index continuing from the checkpoint",
        "split": {
            "rule": args.split,
            "checkpoint_rule": ("SurvivalModel.fit default: "
                                "data.sample(frac=0.85, random_state=seed), "
                                "row level, not by shot"),
            "rows": {"train": int(idx_train.size), "valid": int(idx_valid.size)},
            "shots_on_both_sides": int(leak),
            "by_shot_rows": {k: int(shot_split[k].size) for k in ("train", "valid", "test")},
            "by_shot_shots": shot_split["shots"], "seed": int(args.seed),
        },
        "arrays": {k: str(DATA_DIR / f"{v}.pkl") for k, v in names.items()},
        "wall_s": wall_s, "hostname": socket.gethostname(), "slurm_job_id": job_id,
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch": torch.__version__, "fork": str(args.fork),
    }
    (out_dir / "training.json").write_text(json.dumps(training, indent=2) + "\n")

    provenance = {
        "artifact": weights.name,
        "sha256": {weights.name: sha256_of(weights),
                   NORMALIZATIONS.name: sha256_of(out_dir / NORMALIZATIONS.name)},
        "produced_by": "scripts/labeler/retrain_tearing_dsm.py",
        "git_sha": getenv("LABELER_GIT_SHA"),
        "slurm_job_id": job_id,
        "jobstats": None,   # filled in from `jobstats <jobid>` after the job ends
        "upstream_data": {
            k: {"path": str(DATA_DIR / f"{v}.pkl"), "sha256": sha256_of(DATA_DIR / f"{v}.pkl")}
            for k, v in names.items()
        },
        "upstream_checkpoint": {"path": str(args.checkpoint),
                                "sha256": sha256_of(args.checkpoint)},
        "normalizations_copied_from": {"path": str(NORMALIZATIONS),
                                       "sha256": sha256_of(NORMALIZATIONS)},
        "split": training["split"], "gate": training["gate"],
        "val_nll_before": stored, "val_nll_after": float(costs[best]),
        "epochs_run": epochs_run, "wall_s": wall_s,
    }
    (out_dir / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n")

    # `use("Agg")` has to run between the two imports, so these stay here
    # and in this order: a headless node has no display to fall back to.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(len(val_all)), val_all, color="#4a3aa7", lw=1.9,
            label="validation NLL (elbo=False)")
    ax.axvline(start_epoch - 0.5, color="#898781", lw=1.0, ls="--")
    ax.plot([start_epoch + best], [costs[best]], "o", color="#eb6834", ms=7,
            label=f"best epoch {start_epoch + best}: {costs[best]:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation NLL")
    ax.set_title(f"shipped 0-{start_epoch - 1} then continued at lr {args.lr:g}",
                 loc="left")
    ax.legend(frameon=False)
    ax.grid(color="#e1e0d9", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curve.png", dpi=150)

    print(f"wrote {weights} ({wall_s:.0f}s); val NLL {stored:.6f} -> {costs[best]:.6f}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
