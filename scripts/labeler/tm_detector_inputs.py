#!/usr/bin/env python
"""Model-ready inputs and published outputs of the tearing models, per shot.

For each shot of the cohort (or, with ``--set population``, the population) that has a
feature file in the labeler root named by ``LABELER_ROOT`` (the private root
``$TM_ROOT/lroot`` the cohort's features were built in; the main root for the
population), builds `tm-onsetcnn`'s and `tm-dsm`'s
inputs exactly as `labeler.run` does and writes, per model and shot,
``$TM_ROOT/detector_inputs/<slug>/<shot>.npz``:

* ``d3d_tearing_onset_cnn1d``: ``t_s``, ``valid``, ``scalars`` (T, 11), ``profiles``
  (T, rho, 5), and the published ensemble output ``tm_prob`` / ``tm_lo`` / ``tm_hi``;
* ``d3d_tearing_time_to_event_dsm``: ``t_s``, ``valid``, ``x``
  (T, 38, the model's standardised and PCA-reduced input) and ``risk`` (T, 3: the risk
  of an onset within 250 ms, 500 ms, 1 s).

The retraining and scoring scripts read these files and need no feature store. Re-run
as more feature files land (a file under two minutes old is left for the next pass).
Run in the pixi env (it imports `labeler.run`)::

    PYTHONPATH=$PWD/src LABELER_ROOT=$TM_ROOT/lroot pixi run --frozen --no-install \\
        -e labelmaker python scripts/labeler/tm_detector_inputs.py
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler import run
from labeler.config import Paths
from labeler.models import registry
from labeler.models.runners import dsm_pickle

TM_ROOT = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/tm")
OUT = Path(os.environ.get("TM_ROOT", TM_ROOT)) / "detector_inputs"
CATALOG = REPO / "data/events/catalog"
CNN = "d3d_tearing_onset_cnn1d"
DSM = "d3d_tearing_time_to_event_dsm"
#: How old a feature file must be (s) before it is read, so one still being written
#: by the feature stage is not.
SETTLE_S = 120.0
NORMS: dict = {}


def predictor(slug, paths):
    adapter = registry.load_adapter(slug)
    return adapter, adapter.load(paths.models / slug)


def one_shot(shot, paths, models):
    features = paths.features_file(shot)
    if not features.is_file() or time.time() - features.stat().st_mtime < SETTLE_S:
        return "no-features"
    status = []
    for slug, (adapter, predict) in models.items():
        target = OUT / slug / f"{shot}.npz"
        if target.is_file():
            status.append("have")
            continue
        built = run._build_inputs(features, adapter)
        members = predict(built)
        arrays = {
            "t_s": np.asarray(built.t, dtype=np.float64),
            "valid": np.asarray(built.valid, dtype=bool),
        }
        if slug == CNN:
            decoded = adapter.output_spec.decode(members)["tm_prob"]
            arrays.update(
                scalars=built.scalars.astype(np.float32),
                profiles=built.profiles.astype(np.float32),
                tm_prob=decoded.mean,
                tm_lo=decoded.lo,
                tm_hi=decoded.hi,
            )
        else:
            spec = importlib.import_module(f"labeler.models.{slug}.spec")
            if slug not in NORMS:
                with open(paths.models / slug / adapter.artifacts[1], "rb") as fh:
                    NORMS[slug] = dsm_pickle.RestrictedUnpickler(fh).load()
            arrays.update(x=spec.preprocess(built, NORMS[slug]).astype(np.float32))
            arrays.update(risk=np.asarray(members[0][:, :3], dtype=np.float64))
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **arrays)
        tmp.rename(target)
        status.append("ok")
    return "/".join(status)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--shots", type=int, nargs="+")
    ap.add_argument(
        "--set",
        dest="source",
        choices=("cohort", "population"),
        default="cohort",
        help="the catalog table the shots come from (the labeler root must hold them)",
    )
    args = ap.parse_args(argv)
    paths = Paths.from_env()
    models = {slug: predictor(slug, paths) for slug in (CNN, DSM)}
    shots = args.shots or [
        int(s) for s in pd.read_csv(CATALOG / f"{args.source}.csv").shot
    ]
    tally: dict[str, int] = {}
    for shot in shots:
        try:
            status = one_shot(shot, paths, models)
        except Exception as exc:  # noqa: BLE001 - one shot's failure is its own
            status = f"error {type(exc).__name__}"
            (OUT / f"{shot}.error.json").parent.mkdir(parents=True, exist_ok=True)
            (OUT / f"{shot}.error.json").write_text(
                json.dumps(
                    {"shot": shot, "error": f"{type(exc).__name__}: {exc}"[:400]}
                )
            )
        tally[status] = tally.get(status, 0) + 1
    print(json.dumps(tally))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
