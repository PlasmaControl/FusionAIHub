#!/usr/bin/env python
"""Retrain elm-ours on the native-detection comparator's own 37-shot folds.

The native 124-input DSM refit trains on 24-28 shots per fold with 2-8 inner-
validation shots (`outputs/labeler/elm/dsm/native_detection.json`, `folds`), while
the headline `elm-ours` trains on 81-82 shots with 14. This script fits the same
network, recipe and seeds as the headline run on exactly those train, inner-validation
and test shots, so the comparator panel can be read at equal training data. It writes
predictions and fold records under `$LABELER_ROOT/round4/elm/cv/<run>/` in the layout
of `labeler.elm.train`, and `elm_native_detect.py --rescore` scores them as
`elm-ours-native-folds`. No cohort test shot is read.

    python scripts/labeler/elm_native_ours.py --device cuda
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from labeler.config import Paths, git_sha
from labeler.elm import net, train

REPO = Path(__file__).resolve().parents[2]
NATIVE = REPO / "outputs/labeler/elm/dsm/native_detection.json"
RUN = "native37"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default=RUN)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--epochs", type=int, default=train.Config.epochs)
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    folds = json.loads(NATIVE.read_text())["folds"]
    shots = sorted(
        {s for f in folds for key in ("train", "inner_val", "test") for s in f[key]}
    )
    data = train.load(paths, shots)
    missing = sorted(set(shots) - set(data))
    if missing:
        raise ValueError(f"native-fold shots without reviewed data: {missing}")
    cfg = train.Config(epochs=args.epochs)
    device = torch.device(args.device)
    out = paths.root / "round4" / "elm" / "cv" / args.run
    (out / "pred").mkdir(parents=True, exist_ok=True)
    record = {
        "config": asdict(cfg),
        "git": git_sha(),
        "source": str(NATIVE.relative_to(REPO)),
        "purpose": "elm-ours retrained on the native-detection comparator's shot "
        "folds: same recipe and seeds as the headline run, fewer training shots",
        "shots": len(data),
        "parameters": net.n_parameters(net.ElmUNet(dropout=cfg.dropout)),
        "folds": [f["test"] for f in folds],
        "fold_records": [],
    }
    for f in folds:
        k = f["fold"]
        tr, va, te = f["train"], f["inner_val"], f["test"]
        assert not (set(tr) & set(va) or set(tr) & set(te) or set(va) & set(te))
        print(
            f"fold {k}: {len(tr)} train, {len(va)} inner-val, {len(te)} test",
            flush=True,
        )
        fold_cfg = train.Config(**{**asdict(cfg), "seed": cfg.seed + 100 * k})
        state, history, best = train.train_fold(data, tr, va, fold_cfg, device)
        fold_dir = out / f"fold{k}"
        fold_dir.mkdir(exist_ok=True)
        torch.save(state, fold_dir / "model.pt")
        model = net.ElmUNet(dropout=cfg.dropout).to(device)
        model.load_state_dict(state)
        for s in te:
            p = train.predict(model, data[s].x, device)
            np.savez_compressed(out / "pred" / f"{s}.npz", p=p.astype(np.float16))
        info = {
            "fold": k,
            "train": tr,
            "inner_val": va,
            "test": te,
            "best": best,
            "threshold": best["val_threshold"],
            "onset_threshold": None,
            "history": history,
        }
        (fold_dir / "fold.json").write_text(json.dumps(info, indent=1))
        record["fold_records"].append(
            {
                key: info[key]
                for key in ("fold", "train", "inner_val", "test", "best", "threshold")
            }
        )
        (out / "run.json").write_text(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
