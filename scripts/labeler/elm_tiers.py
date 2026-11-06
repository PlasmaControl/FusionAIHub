#!/usr/bin/env python
"""ELM label tiers: silver teachers -> refined bronze -> a student for the all-shot database
(the ELM twin of ``ae_tiers.py``). The model is ``elm-ours`` (``labeler.elm``), retrained
from scratch on the FDP inputs of ``elm_fdp_label.py`` (its checkpoints are not on this
cluster); the label is ELMy-period occupancy per 1 ms, as elm-ours'.

- **silver** (train): the 450 silver shots plus the other owner-reviewed shots. Labels are
  the owner's review (``review/labels.csv``) where one exists, else the elm-ours extension
  (``extend_elm_ours/silver_gold.csv``; its present spans are taken as crowd spans, i.e.
  occupancy, as 97 % of elm-ours' training positives were). Silver shots with neither join
  bronze.
- **gold** (scored only): the 50 gold shots; only the owner-reviewed ones (7 on 2026-11-03)
  have labels to score.
- **bronze**: the AE tiers' bronze list (paper population minus silver and gold), labeled by
  the teacher ensemble and refined automatically (``refine``).
- Never trained on: gold, the cohort's test split and every ``protect: holdout`` list (the
  ELM challenge key); Shousha's review and Smith's windows are not on this cluster.

Subcommands, all resumable:
``lists``    (login node) labels/{silver,gold}.csv, lists/{silver,gold,bronze}.csv, excluded.csv.
``folds``    (login node) folds.json: the silver shots with inputs, dealt into 5 folds.
``train``    (GPU job) ``--role teacher|student --fold k``: one fold model in the
             ``labeler.elm.serve`` layout under ``models/<role>/`` (so ``elm_fdp_label.py infer
             --models`` serves it), and its out-of-fold silver predictions in ``oof/<role>/``.
             The student trains on silver plus refined bronze, batches half each, dropout 0.2.
``predict``  (GPU job) ``--role R --list FILE``: the fold ensemble's mean and sd of the event
             probability per ms, ``preds/<role>/<shot>.npz``.
``refine``   (CPU job) teacher predictions on bronze -> ``labels/bronze.csv``: a 11 ms median
             filter; present at or above ``hi`` and absent at or below ``lo``, both set on the
             teachers' out-of-fold silver frames (precision of present >= 0.95, of absent >= 0.98);
             frames where the teachers disagree (sd > 0.2) or within 10 ms of a present/absent
             boundary are left unlabeled.
``evaluate`` (CPU job) 1 ms frame precision, recall and F1 of each role on the owner-reviewed
             silver shots (out of fold), all silver (out of fold) and the reviewed gold shots.
             The gate: the student's precision and recall >= 0.75 on both owner-reviewed sets.

Inputs: ``$ELM_FDP_ROOT/inputs/<shot>.npz`` (``elm_fdp_label.py extract``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
LABELER_ROOT = Path(os.environ.get("LABELER_ROOT", REPO / ".cache/labeler"))
ROOT = Path(os.environ.get("ELM_TIERS_ROOT", LABELER_ROOT / "elm_tiers"))
INPUTS = Path(os.environ.get("ELM_FDP_ROOT", LABELER_ROOT / "elm_fdp_1ms")) / "inputs"
EVENTS = Path(os.environ.get("LABELER_LABEL_TABLES", "/scratch/gpfs/EKOLEMEN/D3D/events"))
ELM = EVENTS / "edge_localized_mode"
REVIEW = ELM / "review/labels.csv"
EXTENSION = ELM / "extend_elm_ours/silver_gold.csv"
SILVER_LIST = EVENTS / "silver/silver_shots.csv"
GOLD_LIST = EVENTS / "gold/gold_shots.csv"
BRONZE_LIST = LABELER_ROOT / "ae_tiers/lists/bronze.csv"
CANDIDATES = LABELER_ROOT / "ae_fdp_1ms/candidates.csv"

FOLDS, SEED, INNER_VAL = 5, 20261103, 30
ROLES = {"teacher": {"dropout": 0.1, "epochs": 25}, "student": {"dropout": 0.2, "epochs": 40}}
PREC_PRESENT, PREC_ABSENT = 0.95, 0.98
SD_MAX, MEDIAN_MS, EDGE_MS = 0.2, 11, 10
GATE = 0.75
COLUMNS = ["shot", "category", "t_start", "t_end", "confidence", "attrs"]
CROWD = json.dumps({"iscrowd": 1})


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def shots_of(path: Path) -> list[int]:
    return sorted({int(s) for s in pd.read_csv(path, usecols=["shot"]).shot})


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False)
    os.replace(tmp, path)


# ------------------------------------------------------------------ lists (login node)


def cmd_lists(args) -> None:
    from labeler.config import Paths
    from labeler.elm import train

    paths = Paths.from_env()
    cohort = pd.read_csv(paths.catalog / "cohort.csv").set_index("shot").split
    review = pd.read_csv(REVIEW)
    ext = pd.read_csv(EXTENSION)
    ext = ext.assign(attrs=np.where(ext.category == 1, CROWD, ""))[COLUMNS]
    gold = set(shots_of(GOLD_LIST))
    silver = set(shots_of(SILVER_LIST))
    reviewed, extended = set(review.shot), set(ext.shot)
    end_ms = dict(pd.read_csv(CANDIDATES, usecols=["shot", "end_ms"]).itertuples(index=False))

    excluded = []
    for s in sorted(silver | reviewed | gold | set(shots_of(BRONZE_LIST))):
        if train.shotlists.held_out(paths, s):
            excluded.append((s, "holdout list"))
        elif cohort.get(s) == "test" and s not in gold:
            excluded.append((s, "cohort test split"))
    out_of_training = {s for s, _ in excluded} | gold

    train_shots = sorted((silver | reviewed) - out_of_training)
    rows = []
    for s in train_shots:
        if s in reviewed:
            rows.append(review[review.shot == s].assign(source="owner review"))
        elif s in extended:
            rows.append(ext[ext.shot == s].assign(source="elm-ours extension"))
    silver_labels = pd.concat(rows, ignore_index=True)
    unlabeled = sorted(set(train_shots) - set(silver_labels.shot))
    gold_labels = review[review.shot.isin(gold)].assign(source="owner review")
    bronze = sorted((set(shots_of(BRONZE_LIST)) | set(unlabeled)) - out_of_training)

    write_csv(silver_labels, ROOT / "labels/silver.csv")
    write_csv(gold_labels, ROOT / "labels/gold.csv")
    for name, shots in (("silver", sorted(silver_labels.shot.unique())), ("gold", sorted(gold)),
                        ("bronze", bronze)):
        write_csv(pd.DataFrame({"shot": shots, "end_ms": [end_ms.get(s, np.nan) for s in shots]}),
                  ROOT / "lists" / f"{name}.csv")
    write_csv(pd.DataFrame(excluded, columns=["shot", "reason"]), ROOT / "excluded.csv")
    print(f"silver {silver_labels.shot.nunique()} shots ({silver_labels[silver_labels.source == 'owner review'].shot.nunique()} "
          f"owner-reviewed), {len(unlabeled)} unlabeled silver shots moved to bronze; gold {len(gold)} "
          f"({gold_labels.shot.nunique()} reviewed); bronze {len(bronze)}; excluded {len(excluded)}")


def cmd_folds(args) -> None:
    from labeler.elm import labels

    table = labels.review_table(ROOT / "labels/silver.csv")
    have = [s for s in sorted(table.shot.unique()) if (INPUTS / f"{s}.npz").exists()]
    rng = np.random.default_rng(SEED)
    strata: dict[tuple[bool, bool], list[int]] = {}
    for s in have:  # as labeler.elm.train.deal_folds, from the spans alone
        kinds = set(table.kind[table.shot == s])
        strata.setdefault(("crowd" in kinds, "non_crowd" in kinds), []).append(int(s))
    folds: list[list[int]] = [[] for _ in range(FOLDS)]
    turn = 0
    for key in sorted(strata):
        shots = sorted(strata[key])
        rng.shuffle(shots)
        for s in shots:
            folds[turn % FOLDS].append(s)
            turn += 1
    missing = sorted(set(table.shot) - set(have))
    (ROOT / "folds.json").write_text(json.dumps({"folds": [sorted(f) for f in folds], "seed": SEED,
                                                 "no_input": [int(s) for s in missing], "at": now()}))
    print(f"{len(have)} silver shots with inputs in {FOLDS} folds {[len(f) for f in folds]}; "
          f"{len(missing)} without inputs")


# ------------------------------------------------------------------ shots


def channels(shot: int) -> np.ndarray:
    from labeler.elm import inputs

    with np.load(INPUTS / f"{shot}.npz") as z:
        return inputs.channels_from_cells(z["fmax"].astype(np.float64), z["fcount"],
                                          z["dmean"].astype(np.float64), z["dcount"])


def _shot_data(job):
    shot, spans = job
    from labeler.elm import inputs, labels, train

    x = channels(shot)
    n_ms = x.shape[1] // inputs.CELLS_PER_MS
    spans = spans.reset_index(drop=True)
    c0, c1 = inputs.valid_intervals(x[inputs.VALID])
    cov0, cov1 = labels.merge_intervals(c0, c1)
    return train.ShotData(shot, x, labels.dense(spans, n_ms), n_ms, spans, cov0, cov1,
                          labels.scored_bins(spans, cov0, cov1))


def load(table_path: Path, shots, workers: int) -> dict:
    """``labeler.elm.train.ShotData`` of ``shots`` from the FDP inputs and a label table."""
    from labeler.elm import labels

    table = labels.review_table(table_path)
    jobs = [(int(s), table[table.shot == s]) for s in shots if (INPUTS / f"{s}.npz").exists()]
    with ProcessPoolExecutor(workers) as pool:
        return {d.shot: d for d in pool.map(_shot_data, jobs, chunksize=8)}


# ------------------------------------------------------------------ train (GPU job)


def cmd_train(args) -> None:
    import torch

    from labeler.elm import net, train

    role, k = args.role, args.fold
    folds = json.loads((ROOT / "folds.json").read_text())["folds"]
    silver = sorted(s for f in folds for s in f)
    test = folds[k]
    rest = [s for s in silver if s not in set(test)]
    tr, val = train.split_inner(rest, INNER_VAL, SEED + k)
    data = load(ROOT / "labels/silver.csv", silver, args.workers)
    bronze: list[int] = []
    cfg = train.Config(dropout=ROLES[role]["dropout"], epochs=ROLES[role]["epochs"], seed=SEED + 100 * k)
    if role == "student":
        bronze_data = load(ROOT / "labels/bronze.csv", shots_of(ROOT / "labels/bronze.csv"), args.workers)
        bronze_data = {s: d for s, d in bronze_data.items() if (d.dense.state >= 0).any()}
        data.update(bronze_data)
        bronze = sorted(bronze_data)
        original = train.batch_of

        def mixed(rng, data_, shots, cfg_, device):  # half silver, half bronze (Noisy Student)
            half = cfg_.batch // 2
            a = original(rng, data_, tr, replace(cfg_, batch=half), device)
            b = original(rng, data_, bronze, replace(cfg_, batch=cfg_.batch - half), device)
            return {key: torch.cat([a[key], b[key]]) for key in a}

        train.batch_of = mixed
    device = torch.device(args.device)
    print(f"{role} fold {k}: {len(tr)} silver train, {len(bronze)} bronze, {len(val)} inner-val, "
          f"{len(test)} out of fold; {asdict(cfg)}", flush=True)
    state, history, best = train.train_fold(data, tr, val, cfg, device)
    out = ROOT / "models" / role
    fold_dir = out / f"fold{k}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, fold_dir / "model.pt")
    run = {"name": f"elm_tiers_{role}", "config": asdict(cfg) | {"seed": SEED}, "folds": folds,
           "bronze": len(bronze), "inner_val": INNER_VAL, "at": now()}
    (out / "run.json").write_text(json.dumps(run, indent=1))  # the same for every fold but the time
    model = net.ElmUNet(dropout=cfg.dropout).to(device)
    model.load_state_dict(state)
    oof = ROOT / "oof" / role
    oof.mkdir(parents=True, exist_ok=True)
    for s in test:
        if s in data:
            np.savez_compressed(oof / f"{s}.npz", p=train.predict(model, data[s].x, device)[0].astype(np.float16),
                                threshold=best["val_threshold"])
    (fold_dir / "fold.json").write_text(json.dumps(
        {"fold": k, "train": tr, "inner_val": val, "test": test, "bronze": len(bronze), "best": best,
         "threshold": best["val_threshold"], "history": history}, indent=1))
    print(f"{role} fold {k} done: threshold {best['val_threshold']:.3f}, val AUPRC {best['val_auprc']:.3f}",
          flush=True)


# ------------------------------------------------------------------ predict (GPU job)


def cmd_predict(args) -> None:
    import torch
    from torch.utils.data import DataLoader, Dataset

    from labeler.elm import inputs, net, serve

    out = ROOT / "preds" / args.role
    out.mkdir(parents=True, exist_ok=True)
    todo = [s for s in shots_of(Path(args.list))
            if (INPUTS / f"{s}.npz").exists() and not (out / f"{s}.npz").exists()]
    todo = todo[args.part::args.parts]
    print(f"predict {args.role}: {len(todo)} shots", flush=True)

    class Shots(Dataset):
        def __len__(self):
            return len(todo)

        def __getitem__(self, i):
            x = channels(todo[i])
            buf = np.zeros((x.shape[0], net.pad_to(x.shape[1])), np.float32)
            buf[:, :x.shape[1]] = x
            return todo[i], torch.from_numpy(buf), x.shape[1] // inputs.CELLS_PER_MS

    device = torch.device(args.device)
    models, thresholds = serve.load_models(ROOT / "models" / args.role, device)
    loader = DataLoader(Shots(), batch_size=None, num_workers=args.workers,
                        prefetch_factor=4 if args.workers else None, pin_memory=device.type == "cuda")
    print("LOOP_START", datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f"), flush=True)  # noqa: DTZ005
    with torch.inference_mode():
        for i, (shot, x, n_ms) in enumerate(loader):
            xb = x.to(device, non_blocking=True)[None]
            p = torch.stack([torch.sigmoid(m(xb))[0, 0, :n_ms] for m in models])
            mean, sd = p.mean(0).cpu().numpy(), p.std(0).cpu().numpy()
            tmp = out / f".{shot}.{os.getpid()}.npz"
            np.savez_compressed(tmp, p=mean.astype(np.float16), sd=sd.astype(np.float16),
                                threshold=float(np.mean(thresholds)))
            os.replace(tmp, out / f"{shot}.npz")
            if i % 500 == 0:
                print(i, shot, flush=True)
    print("LOOP_END", datetime.now().strftime("%Y/%m/%d %H:%M:%S.%f"), flush=True)  # noqa: DTZ005


# ------------------------------------------------------------------ refine (CPU job)


def truth_and_scores(role: str, table_path: Path, shots, pred_dir: Path):
    """Pooled 1 ms (truth, score) over labeled frames with a full input ms; and per shot."""
    from labeler.elm import inputs, labels

    table = labels.review_table(table_path)
    truth, score = [], []
    for s in shots:
        f = pred_dir / f"{s}.npz"
        if not f.exists():
            continue
        p = np.load(f)["p"].astype(np.float32)
        x = channels(s)
        valid = x[inputs.VALID, : p.size * inputs.CELLS_PER_MS].reshape(-1, inputs.CELLS_PER_MS).all(axis=1)
        state = labels.dense(table[table.shot == s].reset_index(drop=True), p.size).state
        keep = (state >= 0) & valid[: p.size]
        truth.append(state[keep] > 0)
        score.append(p[keep])
    if not truth:
        return np.zeros(0, bool), np.zeros(0, np.float32)
    return np.concatenate(truth), np.concatenate(score)


def cutoffs(truth: np.ndarray, score: np.ndarray) -> tuple[float, float]:
    """``hi``: the lowest score whose calls above are >= PREC_PRESENT present; ``lo``: the highest
    score whose frames below are >= PREC_ABSENT absent."""
    grid = np.linspace(0.02, 0.98, 49)
    hi = next((t for t in grid if (score >= t).any() and truth[score >= t].mean() >= PREC_PRESENT), 0.9)
    lo = next((t for t in grid[::-1] if (score <= t).any() and (~truth[score <= t]).mean() >= PREC_ABSENT), 0.1)
    return float(hi), float(min(lo, hi))


def runs_of(mask: np.ndarray) -> list[tuple[int, int]]:
    d = np.diff(np.r_[0, mask.astype(np.int8), 0])
    return list(zip(np.flatnonzero(d == 1).tolist(), np.flatnonzero(d == -1).tolist()))


def _refine_one(job):
    shot, end_ms, hi, lo = job
    from scipy.ndimage import binary_dilation, median_filter

    from labeler.elm import inputs

    z = np.load(ROOT / "preds/teacher" / f"{shot}.npz")
    p, sd = z["p"].astype(np.float32), z["sd"].astype(np.float32)
    x = channels(shot)
    valid = x[inputs.VALID, : p.size * inputs.CELLS_PER_MS].reshape(-1, inputs.CELLS_PER_MS).all(axis=1)
    off = int(-inputs.GRID0_MS)
    k = np.arange(p.size) - off  # ms of the shot's clock
    inside = valid & (k >= 0) & (k < (end_ms if np.isfinite(end_ms) else p.size))
    ps = median_filter(p, size=MEDIAN_MS, mode="nearest")
    present, absent = (ps >= hi) & inside, (ps <= lo) & inside
    sure = sd <= SD_MAX
    edges = np.flatnonzero(np.diff(present.astype(np.int8)) != 0)
    near = np.zeros(p.size, bool)
    near[edges] = True
    near = binary_dilation(near, iterations=EDGE_MS)
    present &= sure & ~near
    absent &= sure & ~near
    rows = []
    for cat, mask in ((1, present), (0, absent)):
        for a, b in runs_of(mask):
            rows.append((shot, cat, float(k[a]), float(k[b - 1] + 1), np.nan, CROWD if cat == 1 else ""))
    n = int(inside.sum())
    return rows, {"shot": shot, "inside_ms": n, "present_ms": int(present.sum()), "absent_ms": int(absent.sum())}


def cmd_refine(args) -> None:
    folds = json.loads((ROOT / "folds.json").read_text())["folds"]
    silver = sorted(s for f in folds for s in f)
    truth, score = truth_and_scores("teacher", ROOT / "labels/silver.csv", silver, ROOT / "oof/teacher")
    hi, lo = cutoffs(truth, score)
    bronze = pd.read_csv(ROOT / "lists/bronze.csv")
    jobs = [(int(r.shot), float(r.end_ms), hi, lo) for r in bronze.itertuples(index=False)
            if (ROOT / "preds/teacher" / f"{int(r.shot)}.npz").exists()]
    rows, stats = [], []
    with ProcessPoolExecutor(args.workers) as pool:
        for r, st in pool.map(_refine_one, jobs, chunksize=16):
            rows += r
            stats.append(st)
    write_csv(pd.DataFrame(rows, columns=COLUMNS), ROOT / "labels/bronze.csv")
    st = pd.DataFrame(stats)
    summary = {"at": now(), "hi": hi, "lo": lo, "silver_oof_frames": int(truth.size),
               "shots": len(st), "labeled_share": float((st.present_ms + st.absent_ms).sum() / max(st.inside_ms.sum(), 1)),
               "present_share": float(st.present_ms.sum() / max(st.inside_ms.sum(), 1)),
               "shots_with_present": int((st.present_ms > 0).sum()),
               "rule": f"median {MEDIAN_MS} ms; present >= hi (precision {PREC_PRESENT}), absent <= lo "
                       f"(absent precision {PREC_ABSENT}) on teacher out-of-fold silver; sd <= {SD_MAX}; "
                       f"{EDGE_MS} ms either side of a boundary unlabeled"}
    (ROOT / "labels/bronze.summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps(summary, indent=1))


# ------------------------------------------------------------------ evaluate (CPU job)


def prf(truth: np.ndarray, call: np.ndarray) -> dict:
    tp, fp, fn = int((truth & call).sum()), int((~truth & call).sum()), int((truth & ~call).sum())
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / max(p + r, 1e-9), 4),
            "frames": int(truth.size), "present_frames": int(truth.sum())}


def cmd_evaluate(args) -> None:
    folds = json.loads((ROOT / "folds.json").read_text())["folds"]
    silver = sorted(s for f in folds for s in f)
    table = pd.read_csv(ROOT / "labels/silver.csv")
    owner = sorted(set(table.shot[table.source == "owner review"]) & set(silver))
    gold = shots_of(ROOT / "labels/gold.csv")
    result = {"at": now(), "gate": f"student precision and recall >= {GATE} on silver_owner_oof and gold_owner",
              "sets": {"silver_owner_oof": len(owner), "silver_all_oof": len(silver), "gold_owner": len(gold)},
              "scores": {}}
    for role in ROLES:
        mdir = ROOT / "models" / role
        if not (mdir / "run.json").exists():
            continue
        thr = float(np.mean([json.loads(f.read_text())["threshold"] for f in sorted(mdir.glob("fold*/fold.json"))]))
        res = {"threshold": round(thr, 4)}
        for name, tab, shots, pdir in (("silver_owner_oof", ROOT / "labels/silver.csv", owner, ROOT / "oof" / role),
                                       ("silver_all_oof", ROOT / "labels/silver.csv", silver, ROOT / "oof" / role),
                                       ("gold_owner", ROOT / "labels/gold.csv", gold, ROOT / "preds" / role)):
            truth, score = truth_and_scores(role, tab, shots, pdir)
            if truth.size:
                res[name] = prf(truth, score >= thr)
        result["scores"][role] = res
    st = result["scores"].get("student", {})
    result["gate_passed"] = bool(st) and all(
        st.get(n, {}).get("precision", 0) >= GATE and st.get(n, {}).get("recall", 0) >= GATE
        for n in ("silver_owner_oof", "gold_owner"))
    (ROOT / "evaluation.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(result, indent=1))


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("lists")
    sub.add_parser("folds")
    t = sub.add_parser("train")
    t.add_argument("--role", choices=list(ROLES), required=True)
    t.add_argument("--fold", type=int, required=True)
    t.add_argument("--workers", type=int, default=8)
    t.add_argument("--device", default="cuda")
    q = sub.add_parser("predict")
    q.add_argument("--role", choices=list(ROLES), required=True)
    q.add_argument("--list", required=True)
    q.add_argument("--workers", type=int, default=6)
    q.add_argument("--parts", type=int, default=1)
    q.add_argument("--part", type=int, default=0)
    q.add_argument("--device", default="cuda")
    for name in ("refine", "evaluate"):
        c = sub.add_parser(name)
        c.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    args = p.parse_args(argv)
    {"lists": cmd_lists, "folds": cmd_folds, "train": cmd_train, "predict": cmd_predict,
     "refine": cmd_refine, "evaluate": cmd_evaluate}[args.cmd](args)


if __name__ == "__main__":
    main()
