"""Make tests/labeler/data/tearing_dsm_golden.npz - ONCE, with the fork.

Run in a throwaway environment that has torch, scikit-learn, scikit-survival
and the auton-survival fork on sys.path; labeler's own environment has none
of them, by design. The file holds z-scored inputs drawn N(0, 1) (the model's
inputs are z-scored, so this is its domain) and the survival probabilities the
fork's own `predict_survival` returns for them.

    python make_tearing_dsm_golden.py /projects/EKOLEMEN/survival_tm_2/models/rt_fixed_rot.pkl
"""
import hashlib
import pickle
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/auton-survival")

src = Path(sys.argv[1])
horizons = [250.0, 500.0, 1000.0]
rng = np.random.default_rng(20260905)
x = rng.normal(size=(256, 38))
with src.open("rb") as handle:
    models = pickle.load(handle)
sm = models[0][0]
# The DeepSurvivalMachines level, as `get_survival_from_shot.py` reaches it
# through SurvivalModel; the estimator wrapper's DataFrame path trips on a
# pandas API this venv no longer has, and adds nothing but the conversion.
s = sm._model.predict_survival(x, horizons)                  # (256, 3)
out = Path(__file__).with_name("tearing_dsm_golden.npz")
np.savez_compressed(
    out, x=x, survival=np.asarray(s, dtype=np.float64), horizons_ms=np.array(horizons),
    meta=np.array([f"source={src}", f"sha256={hashlib.sha256(src.read_bytes()).hexdigest()}",
                   f"torch={torch.__version__}", "fork=/projects/EKOLEMEN/wpqh_elm_hiro/hiro_scripts/auton-survival"]),
)
print("wrote", out, s.shape,
      f"S(1000) mean {s[:, 2].mean():.4f} min {s[:, 2].min():.4f} "
      f"max {s[:, 2].max():.4f}")
