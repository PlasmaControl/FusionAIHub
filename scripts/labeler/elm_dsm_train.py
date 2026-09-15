"""Train labeler's own ELM time-to-event DSM, with and without BES.

Why a new fit rather than the shipped weights: the upstream ELM model takes 124
inputs and 64 of them are BES, which the FAITH corpus fills on 2 of 24 sampled
shots. A model that needs BES cannot be served at corpus scale. So labeler
fits the same architecture on the same rows twice - `all124`, the upstream input
set, and `no_bes`, the 60 columns of the split pickle that are not BES (slots
0-11 and 76-123 of `new_diagnostic_order`) - and the ablation decides whether the servable
model is good enough to adopt (phase3-design section 6).

What is reused verbatim and what is not:

* the split is upstream's own, read from
  `/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl`
  (`train_final_x_normalized` / `_e` / `_t` and the `test_*` counterparts). It is
  **not** re-split: its by-shot split is the one upstream validated on, and the
  swapped-train/test bug lived in `train_elm_model.py`, not in this pickle. The
  script asserts the row counts agree and records the sha256;
* the `t + 1` millisecond offset is upstream's (`new_train_elm_model.py` adds 1
  to both `t` and `t_test` before fitting). It is kept so the fit matches, and
  it is written into `training.json` rather than hidden: a horizon `h` must be
  queried as `S(h + 1)`;
* the hyperparameters are `hiro_scripts/model.cfg`: `k=3, layers=[128],
  LogNormal, lr=1e-3, batch=1024, dropout=0.2, iters=1000`, seed 0;
* the fork is `hiro_scripts/auton-survival`, not `survival_tm_2`'s. The two
  differ in exactly three places (`diff` at the time of writing): hiro's inserts
  `nn.Dropout(0.2)` after every embedding ReLU6, threads a `dropout` argument
  through `DSMBase`, and calls `model.train()` before the loop. The dropout is
  the reason - `survival_tm_2`'s fork would silently drop it;
* the training loop is `utilities.train_dsm` ported here with **three**
  changes, each forced by something job 2924059 measured. (1) A wall-clock
  deadline, so a run that never triggers the fork's `train_patience = 5` ends
  with a checkpoint instead of dying at the SLURM wall. (2) The validation pass
  runs in `eval()` mode; the fork leaves the module in `train()` mode all the
  way through, so dropout corrupts the number the early stop reads. Done the
  fork's way, the validation NLL rose monotonically from epoch 0 on both column
  sets and the "best" epoch was the untrained one. Training itself stays in
  `train()` mode, so dropout regularises the gradient exactly as upstream. (3)
  The loop stops at the first non-finite loss and the best state is the argmin
  over the finite epochs: `all124` diverged to NaN at epoch 16 of job
  2924059_0, and `np.argmin` returns the first NaN's index, so the fork's rule
  would have shipped the diverged weights. Everything else - `pretrain_dsm`,
  Adam at `lr`, per-epoch `shuffle(..., random_state=i)`,
  `conditional_loss(elbo=True)` on minibatches, full-set validation at
  `elbo=False`, every epoch's `state_dict` kept - is the fork's code and the
  fork's order.

Two upstream properties that the numbers must be read with:

* upstream passes the **test** split as `val_data`, so the per-epoch curve and
  the early-stop decision both see the test rows. That is what the shipped model
  did and it is what makes "before" and "after" comparable, so it is reproduced;
  the test NLL reported here is therefore an early-stopping-selected number, not
  a held-out one. Both column sets are selected the same way, so the ablation
  comparison between them is fair even though the absolute NLL is optimistic;
* every reported NLL is an eval-mode number: the per-epoch curve, and the
  final `test_nll_eval_mode` / `train_nll_eval_mode` re-measured through
  `compute_nll` after the best state is reloaded. Eval mode is the only mode a
  checkpoint is ever read in.

Saves `[[survival_model, train_losses, val_losses, params]]` - the shape
`new_train_elm_model.py` writes and the shape
`labeler.models.runners.dsm_pickle.load_dsm` reads - plus `training.json`, a
normalisation dict beside the weights, `loss_curve.png` and `PROVENANCE.json`.
With `--ablation` it reads both sets' `training.json` and writes `ablation.json`.

Runs in the Phase 3 uv venv (`scripts/labeler/make_phase3_env.sh`), with
`PYTHONPATH=<repo>/src` so `labeler.alarm.ipcw_auc`,
`labeler.models.elm_inputs` and `labeler.models.runners.dsm_pickle` are
importable. It imports nothing else from labeler.
"""
from __future__ import annotations

import argparse
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

from labeler.env import getenv

SPLIT_PKL = Path("/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl")
FORK = Path("/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/auton-survival")
CFG = Path("/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/model.cfg")
HORIZONS_MS = (5.0, 10.0, 20.0, 50.0)
T_OFFSET_MS = 1.0          # upstream's `t + 1`, kept and documented
DECISION_TOL = 0.02        # written before the run; see `--ablation`


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, path)


def load_split(path: Path) -> dict:
    """The upstream split, as-is. Never re-split, never reordered."""
    with open(path, "rb") as fh:
        data = pickle.load(fh)
    out = {}
    for side in ("train", "test"):
        x = np.asarray(data[f"{side}_final_x_normalized"], dtype=np.float64)
        e = np.asarray(data[f"{side}_final_e"], dtype=np.float64).ravel()
        t = np.asarray(data[f"{side}_final_t"], dtype=np.float64).ravel()
        if not (len(x) == len(e) == len(t)):
            raise SystemExit(f"{side}: x/e/t disagree on row count")
        out[side] = {"x": x, "e": e, "t": t}
    out["keys"] = sorted(k for k in data if not k.startswith(("train_", "test_")))
    out["order_in_pickle"] = data.get("current_diagnostic_order")
    return out


def auroc(score: np.ndarray, positive: np.ndarray) -> float | None:
    """Rank AUROC with ties at half, or None if a class is empty."""
    score, positive = np.asarray(score, float), np.asarray(positive, bool)
    good = np.isfinite(score)
    score, positive = score[good], positive[good]
    if not positive.any() or positive.all():
        return None
    controls = np.sort(score[~positive])
    lower = np.searchsorted(controls, score[positive], side="left")
    upper = np.searchsorted(controls, score[positive], side="right")
    return float(np.mean((lower + upper) / 2) / controls.size)


def evaluate(graph, x, t_ms, e, ipcw_auc) -> dict:
    """Test-set metrics at each horizon from `1 - S(h + 1)`.

    `t_ms` is the raw upstream time to the next ELM, *without* the `+1` the fit
    used. The model therefore answers about real time `h` at `S(h + 1)`. Truth
    for the AUROC is `e == 1 and t <= h`; rows censored at or before `h` are
    neither cases nor controls because it is unknown whether an ELM happened,
    and the IPCW AUC uses them only to estimate the censoring distribution.
    """
    from labeler.models.runners import dsm_pickle

    horizons = [h + T_OFFSET_MS for h in HORIZONS_MS]
    surv = dsm_pickle.survival(graph, x, horizons)
    event = np.asarray(e, float) == 1.0
    out = {}
    for j, h in enumerate(HORIZONS_MS):
        risk = 1.0 - surv[:, j]
        case = event & (t_ms <= h)
        control = t_ms > h
        keep = case | control
        out[f"h{int(h)}ms"] = {
            "horizon_ms": h,
            "queried_at_ms": horizons[j],
            "auroc": auroc(risk[keep], case[keep]),
            "ipcw_auc": ipcw_auc(t_ms, event, risk, h),
            "n_cases": int(case.sum()),
            "n_controls": int(control.sum()),
            "n_censored_within_h": int((~event & (t_ms <= h)).sum()),
            "mean_risk": float(np.mean(risk)),
        }
    return out


def train(args) -> int:
    sys.path.insert(0, str(args.fork))
    import torch
    from auton_survival.estimators import SurvivalModel
    from auton_survival.models.dsm import DeepSurvivalMachines
    from auton_survival.models.dsm.losses import conditional_loss
    from auton_survival.models.dsm.utilities import (
        _reshape_tensor_with_nans,
        get_optimizer,
        pretrain_dsm,
        train_patience,
    )
    from sklearn.utils import shuffle

    from labeler.alarm import ipcw_auc
    from labeler.models import elm_inputs
    from labeler.models.runners import dsm_pickle

    t0 = time.time()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "8")))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(args.split_pkl)
    n_cols = split["train"]["x"].shape[1]
    print(f"split {args.split_pkl}\n  extra keys {split['keys']}\n"
          f"  train {split['train']['x'].shape} test {split['test']['x'].shape}\n"
          f"  events train {split['train']['e'].mean():.4%} "
          f"test {split['test']['e'].mean():.4%}", flush=True)
    if n_cols != len(elm_inputs.SPLIT_COLUMN_ORDER):
        raise SystemExit(f"pickle is {n_cols} columns wide, expected 124")
    order = split["order_in_pickle"]
    order_note = ("not stored in the split pickle; it is new_diagnostic_order from "
                  "data_processing.ipynb cell 43, MEASURED from the pickle's own "
                  "raw/normalized column pair to 1.2e-12 on all 124 columns")
    if order is not None:
        order = [str(name) for name in order]
        if tuple(order) != elm_inputs.SPLIT_COLUMN_ORDER:
            raise SystemExit("the pickle's column order is not new_diagnostic_order")
        order_note = "stored in the split pickle and equal to new_diagnostic_order"
    print(f"column order: {order_note}", flush=True)

    cols = elm_inputs.column_indices(args.column_set)
    names = list(elm_inputs.COLUMN_SETS[args.column_set])
    print(f"column set {args.column_set!r}: {len(cols)} columns, "
          f"slots {cols[0]}-{cols[-1]}", flush=True)

    x_train = split["train"]["x"][:, cols]
    x_test = split["test"]["x"][:, cols]
    t_train_raw, e_train = split["train"]["t"], split["train"]["e"]
    t_test_raw, e_test = split["test"]["t"], split["test"]["e"]

    # ---- normalisation constants of the columns as they arrive, for the card ----
    normalisation = {
        "columns": names,
        "column_set": args.column_set,
        "slots_in_current_diagnostic_order": list(cols),
        "mean": [float(v) for v in x_train.mean(axis=0)],
        "std": [float(v) for v in x_train.std(axis=0)],
        "note": ("measured on the training rows of train_test_split_model10.pkl, "
                 "which are already normalised upstream; these are the residual "
                 "mean/std, kept so a serving adapter can check its own inputs "
                 "land in the same place"),
    }
    with open(out_dir / f"elm_normalizations_{args.column_set}.pkl", "wb") as fh:
        pickle.dump(normalisation, fh)

    # ---- tensors, as `_preprocess_training_data` builds them, with `t + 1` ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    idx = np.arange(len(x_train))
    np.random.shuffle(idx)
    xt = torch.from_numpy(x_train[idx]).double()
    tt = torch.from_numpy(t_train_raw[idx] + T_OFFSET_MS).double()
    et = torch.from_numpy(e_train[idx]).double()
    xv = torch.from_numpy(x_test).double()
    tv = torch.from_numpy(t_test_raw + T_OFFSET_MS).double()
    ev = torch.from_numpy(e_test).double()
    tt_, et_ = _reshape_tensor_with_nans(tt), _reshape_tensor_with_nans(et)
    tv_, ev_ = _reshape_tensor_with_nans(tv), _reshape_tensor_with_nans(ev)

    # ---- the fork's model, built the way SurvivalModel/_fit_dsm builds it ----
    dsm = DeepSurvivalMachines(k=args.k, layers=[args.hidden], distribution="LogNormal",
                               temp=1.0, random_seed=args.seed, dropout=args.dropout)
    model = dsm._gen_torch_model(xt.shape[-1], "Adam", risks=1)

    print("pretraining the underlying distributions...", flush=True)
    premodel = pretrain_dsm(model, tt_, et_, tv_, ev_, n_iter=10000, lr=1e-2, thres=1e-4)
    for r in range(model.risks):
        model.shape[str(r + 1)].data.fill_(float(premodel.shape[str(r + 1)]))
        model.scale[str(r + 1)].data.fill_(float(premodel.scale[str(r + 1)]))
    model.double()
    model.train()
    optimizer = get_optimizer(model, args.lr)

    deadline = t0 + args.max_hours * 3600.0
    nbatches = int(xt.shape[0] / args.batch_size) + 1
    dics, costs, train_losses = [], [], []
    patience, oldcost = 0, float("inf")
    stopped = "iters"
    for i in range(args.iters):
        xt, tt, et = shuffle(xt, tt, et, random_state=i)
        train_loss = 0.0
        for j in range(nbatches):
            xb = xt[j * args.batch_size:(j + 1) * args.batch_size]
            tb = tt[j * args.batch_size:(j + 1) * args.batch_size]
            eb = et[j * args.batch_size:(j + 1) * args.batch_size]
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

        # Deviation from hiro's fork, measured not guessed: the fork holds the
        # module in train() mode for the whole loop, so dropout corrupts the
        # validation pass too. Job 2924059 did it the fork's way and the
        # validation NLL rose monotonically from epoch 0 on BOTH column sets
        # (all124 1.274 -> 1.94 by epoch 15, then NaN) - the early stop was
        # selecting the untrained model. Dropout noise grows as units
        # specialise, so that curve measures dropout, not fit. The validation
        # pass therefore runs in eval() mode; training stays in train() mode,
        # so dropout still regularises the gradient exactly as upstream.
        model.eval()
        with torch.no_grad():
            valid_loss = 0
            for r in range(model.risks):
                valid_loss += conditional_loss(model, xv, tv_, ev_, elbo=False,
                                               risk=str(r + 1))
        model.train()
        costs.append(float(valid_loss.detach().cpu().numpy()))
        dics.append(deepcopy(model.state_dict()))
        print(f"epoch {i} train {train_loss:.4f} valid {costs[-1]:.6f} "
              f"patience {patience} [{time.time() - t0:.0f}s]", flush=True)

        if not (np.isfinite(train_loss) and np.isfinite(costs[-1])):
            # all124 diverged to NaN at epoch 16 of job 2924059_0 and then ran
            # on producing nothing: `np.argmin` returns the first NaN's index,
            # so the "best" state would have been the diverged one. Stop at the
            # first non-finite epoch, keep the best finite state, and say so.
            stopped = "diverged"
            print(f"non-finite loss at epoch {i}: stopping", flush=True)
            break

        if costs[-1] >= oldcost:
            if patience == train_patience:
                stopped = "patience"
                break
            patience += 1
        else:
            patience = 0
        oldcost = costs[-1]
        if time.time() > deadline:
            stopped = "deadline"
            print(f"deadline of {args.max_hours} h reached after epoch {i}", flush=True)
            break

    finite = [c if np.isfinite(c) else np.inf for c in costs]
    if not np.isfinite(min(finite)):
        raise SystemExit("every epoch's validation loss was non-finite")
    best = int(np.argmin(finite))
    model.load_state_dict(dics[best])
    del dics
    dsm.torch_model = model.eval()
    dsm.fitted = True
    print(f"best epoch {best} (per-epoch validation NLL {costs[best]:.6f}); "
          f"stopped on {stopped} after {len(costs)} epochs", flush=True)

    survival_model = SurvivalModel(model="dsm", k=args.k, layers=[args.hidden],
                                   distribution="LogNormal",
                                   learning_rate=args.lr, batch_size=args.batch_size,
                                   iters=args.iters, dropout=args.dropout,
                                   random_seed=args.seed)
    survival_model._model = dsm
    survival_model.fitted = True

    params = {"k": args.k, "layers": [args.hidden], "distribution": "LogNormal",
              "learning_rate": args.lr, "batch_size": args.batch_size,
              "iters": args.iters, "dropout": args.dropout, "seed": args.seed,
              "column_set": args.column_set, "n_columns": len(cols),
              "t_offset_ms": T_OFFSET_MS}
    weights = out_dir / f"elm_dsm_{args.column_set}.pkl"
    tmp = weights.with_suffix(".pkl.tmp")
    with open(tmp, "wb") as fh:
        pickle.dump([[survival_model, [float(v) for v in train_losses],
                      [float(v) for v in costs], params]], fh)
    os.replace(tmp, weights)

    # ---- final numbers, in eval mode, and through labeler's own reader ----
    nll_test = float(dsm.compute_nll(x_test, t_test_raw + T_OFFSET_MS, e_test))
    nll_train = float(dsm.compute_nll(x_train, t_train_raw + T_OFFSET_MS, e_train))
    graph = dsm_pickle.load_dsm(weights)
    metrics = evaluate(graph, x_test, t_test_raw, e_test, ipcw_auc)
    for key, row in metrics.items():
        print(f"{key}: AUROC {row['auroc']} IPCW {row['ipcw_auc']} "
              f"cases {row['n_cases']} controls {row['n_controls']}", flush=True)
    print(f"eval-mode NLL: train {nll_train:.6f} test {nll_test:.6f}", flush=True)

    wall_s = time.time() - t0
    training = {
        "slug": "d3d_elm_time_to_event_dsm",
        "column_set": args.column_set,
        "columns": names,
        "n_columns": len(cols),
        "column_order": order_note,
        "t_offset_ms": T_OFFSET_MS,
        "t_offset_note": (
            "upstream new_train_elm_model.py fits on `t + 1` ms, so this fit does "
            "too; a horizon h must be queried as S(h + 1). Truth for the metrics "
            "below uses the raw t, not the offset one."
        ),
        "epochs_run": len(costs),
        "iters": args.iters,
        "stopped_on": stopped,
        "fork_patience": int(train_patience),
        "max_hours": args.max_hours,
        "best_epoch": best,
        "best_epoch_val_nll": float(costs[best]),
        "train_nll_eval_mode": nll_train,
        "test_nll_eval_mode": nll_test,
        "val_nll_curve": [float(v) for v in costs],
        "train_loss_curve": [float(v) for v in train_losses],
        "val_curve_note": (
            "the per-epoch validation pass runs in eval() mode (dropout off), "
            "unlike hiro's fork which leaves the module in train() mode; see the "
            "script docstring for the measurement that forced the change. "
            "test_nll_eval_mode is the same rows re-measured through compute_nll "
            "after the best state is reloaded."
        ),
        "split_note": (
            "upstream's own split, read as-is from train_test_split_model10.pkl and "
            "never re-split. Upstream passes the test split as val_data, so the "
            "early stop selects on the test rows: test_nll_eval_mode is an "
            "early-stopping-selected number, not held out. Both column sets are "
            "selected identically, so the ablation between them is fair."
        ),
        "rows": {"train": len(x_train), "test": len(x_test)},
        "event_rate": {"train": float(e_train.mean()), "test": float(e_test.mean())},
        "metrics_test": metrics,
        "lr": args.lr, "batch_size": args.batch_size, "seed": args.seed,
        "k": args.k, "layers": [args.hidden], "dropout": args.dropout,
        "distribution": "LogNormal", "temp": 1.0, "elbo": True,
        "wall_s": wall_s,
        "hostname": socket.gethostname(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "cpus": os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "torch": torch.__version__,
        "fork": str(args.fork),
        "config": str(CFG),
        "jobstats": None,
    }
    write_json(out_dir / f"training_{args.column_set}.json", training)

    provenance = {
        "artifact": weights.name,
        "produced_by": "scripts/labeler/elm_dsm_train.py",
        "git_sha": getenv("LABELER_GIT_SHA"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "jobstats": None,
        "inputs": {str(args.split_pkl): sha256_of(args.split_pkl)},
        "outputs": {
            weights.name: sha256_of(weights),
            f"elm_normalizations_{args.column_set}.pkl":
                sha256_of(out_dir / f"elm_normalizations_{args.column_set}.pkl"),
        },
        "column_set": args.column_set,
        "test_nll_eval_mode": nll_test,
        "epochs_run": len(costs), "stopped_on": stopped, "wall_s": wall_s,
        "hostname": socket.gethostname(),
    }
    write_json(out_dir / f"PROVENANCE_{args.column_set}.json", provenance)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(range(len(costs)), costs, color="#4a3aa7", lw=1.9,
            label="test-split NLL per epoch (eval mode)")
    ax.plot([best], [costs[best]], "o", color="#eb6834", ms=7,
            label=f"best epoch {best}: {costs[best]:.4f}")
    ax.axhline(nll_test, color="#2f8f5b", lw=1.2, ls="--",
               label=f"eval-mode test NLL {nll_test:.4f}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("NLL")
    ax.set_title(f"ELM DSM, {args.column_set} ({len(cols)} inputs), stopped on {stopped}",
                 loc="left")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="#e1e0d9", lw=0.6)
    fig.tight_layout()
    fig.savefig(out_dir / f"loss_curve_{args.column_set}.png", dpi=150)

    print(f"wrote {weights} in {wall_s:.0f}s", flush=True)
    return 0


def ablate(args) -> int:
    """The decision rule, written before the run, applied to both training.json."""
    out_dir = Path(args.out_dir)
    sets = {}
    for name in ("all124", "no_bes"):
        path = out_dir / f"training_{name}.json"
        if not path.exists():
            raise SystemExit(f"missing {path}: train that column set first")
        sets[name] = json.loads(path.read_text())

    def at20(payload):
        return payload["metrics_test"]["h20ms"]

    nll = {k: v["test_nll_eval_mode"] for k, v in sets.items()}
    auc20 = {k: at20(v)["auroc"] for k, v in sets.items()}
    d_nll = nll["no_bes"] - nll["all124"]
    d_auc = auc20["all124"] - auc20["no_bes"]
    adopt = bool(d_nll <= DECISION_TOL and d_auc <= DECISION_TOL)
    payload = {
        "rule": (
            "no_bes is adopted if its test NLL is within 0.02 of all124's and its "
            "20 ms AUROC is within 0.02 of all124's; otherwise report and stop, and "
            "a BES fetch becomes a separate decision. Written before the run."
        ),
        "tolerance": DECISION_TOL,
        "test_nll_eval_mode": nll,
        "auroc_20ms": auc20,
        "ipcw_auc_20ms": {k: at20(v)["ipcw_auc"] for k, v in sets.items()},
        "delta_nll_no_bes_minus_all124": d_nll,
        "delta_auroc20_all124_minus_no_bes": d_auc,
        "nll_within_tolerance": bool(d_nll <= DECISION_TOL),
        "auroc20_within_tolerance": bool(d_auc <= DECISION_TOL),
        "verdict": "adopt no_bes" if adopt else "keep BES: no_bes is not close enough",
        "per_horizon": {k: v["metrics_test"] for k, v in sets.items()},
        "epochs_run": {k: v["epochs_run"] for k, v in sets.items()},
        "stopped_on": {k: v["stopped_on"] for k, v in sets.items()},
        "wall_s": {k: v["wall_s"] for k, v in sets.items()},
        "score_note": (
            "score is 1 - S(h + 1) from labeler's own dsm_pickle.survival; the "
            "+1 ms is upstream's fit-time offset. AUROC cases are e == 1 and t <= h "
            "against controls t > h, rows censored inside h excluded; ipcw_auc is "
            "labeler.alarm.ipcw_auc on the same score with the same horizon."
        ),
    }
    write_json(out_dir / "ablation.json", payload)
    print(json.dumps(payload, indent=2)[:2000], flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", required=True, type=Path)
    ap.add_argument("--column-set", choices=("all124", "no_bes"))
    ap.add_argument("--ablation", action="store_true",
                    help="read both training.json files and write ablation.json")
    ap.add_argument("--split-pkl", type=Path, default=SPLIT_PKL)
    ap.add_argument("--fork", type=Path, default=FORK)
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--max-hours", type=float, default=4.5)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch-size", type=int, default=1024)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.ablation:
        return ablate(args)
    if args.column_set is None:
        ap.error("--column-set is required unless --ablation is given")
    return train(args)


if __name__ == "__main__":
    raise SystemExit(main())
