"""Write `normalization.json` beside the ELM DSM weights, and prove the order.

The DSM in `elm_dsm_no_bes.pkl` was fitted on
`train_test_split_model10.pkl`'s `*_final_x_normalized`, so a serving adapter
has to apply the SAME per-column mean/std to its own raw inputs. Those
constants are not in the split pickle. They are in the `normalizations` dict of
`/projects/EKOLEMEN/wpqh_elm_hiro/data/testing_model.pkl`, keyed by upstream's
own column names, which is what this script reads.

`--verify-split` then proves, rather than assumes, that those constants are the
ones the split pickle carries AND that its columns sit in
`elm_inputs.SPLIT_COLUMN_ORDER`: the pickle holds both `*_final_x` (raw) and
`*_final_x_normalized`, so upstream's per-column transform is recoverable as

    s = std(raw) / std(norm)          m = mean(raw) - s * mean(norm)

and can be matched name by name. Measured 2026-09-06: all 124 columns of both
the train and the test side match `new_diagnostic_order` to a worst relative
error of 1.2e-12, and only 6 of 124 match `current_diagnostic_order`.

    python scripts/labeler/elm_write_normalization.py --verify-split

Runs in either env; it imports only numpy and `labeler.models.elm_inputs`.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
# After the `sys.path` line above, deliberately: run from a checkout.
from labeler.models import elm_inputs

NORMS_PKL = Path("/projects/EKOLEMEN/wpqh_elm_hiro/data/testing_model.pkl")
SPLIT_PKL = Path("/projects/EKOLEMEN/wpqh_elm_hiro/data/train_test_split_model10.pkl")
OUT_DIR = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/models/d3d_elm_time_to_event_dsm")


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_split(norms: dict[str, tuple[float, float]]) -> dict:
    """Match the split pickle's implied per-column transform to the names."""
    with open(SPLIT_PKL, "rb") as fh:
        split = pickle.load(fh)
    report = {}
    for side in ("train", "test"):
        raw = np.asarray(split[f"{side}_final_x"], dtype=np.float64)
        nrm = np.asarray(split[f"{side}_final_x_normalized"], dtype=np.float64)
        s = raw.std(axis=0) / nrm.std(axis=0)
        m = raw.mean(axis=0) - s * nrm.mean(axis=0)

        # `m` and `s` are bound as defaults rather than closed over: the
        # closure is called inside this iteration only, and binding says so
        # instead of leaving a reader (or a linter) to check that it is.
        def worst(order, m=m, s=s):
            out = 0.0
            for i, name in enumerate(order):
                mu, sd = norms[name]
                scale = max(abs(sd), 1e-30)
                out = max(out, abs(m[i] - mu) / scale, abs(s[i] - sd) / scale)
            return float(out)

        report[side] = {
            "rows": int(raw.shape[0]),
            "worst_relative_error_vs_split_column_order":
                worst(elm_inputs.SPLIT_COLUMN_ORDER),
            "worst_relative_error_vs_current_diagnostic_order":
                worst(elm_inputs.CURRENT_DIAGNOSTIC_ORDER),
        }
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--column-set", default="no_bes")
    ap.add_argument("--verify-split", action="store_true",
                    help="also read the 1.57 GB split pickle and prove the order")
    args = ap.parse_args()

    with open(NORMS_PKL, "rb") as fh:
        norms = {str(k): (float(v[0]), float(v[1]))
                 for k, v in pickle.load(fh)["normalizations"].items()}
    names = list(elm_inputs.COLUMN_SETS[args.column_set])
    payload = {
        "column_set": args.column_set,
        "columns": names,
        "slots_in_split_column_order":
            list(elm_inputs.column_indices(args.column_set)),
        "mean": [norms[n][0] for n in names],
        "std": [norms[n][1] for n in names],
        "source": str(NORMS_PKL),
        "source_sha256": sha256_of(NORMS_PKL),
        "note": ("upstream's own per-column normalisation, the transform that "
                 "produced train_test_split_model10.pkl's *_final_x_normalized "
                 "columns the DSM was fitted on. A serving adapter applies "
                 "(raw - mean) / std to its own inputs, in this column order."),
    }
    if args.verify_split:
        payload["verification"] = verify_split(norms)
        payload["verification"]["split_pkl"] = str(SPLIT_PKL)
    out = Path(args.out_dir) / "normalization.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(tmp, out)
    print(f"wrote {out} ({len(names)} columns)")
    if args.verify_split:
        print(json.dumps(payload["verification"], indent=2))


if __name__ == "__main__":
    main()
