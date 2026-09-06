"""Per-row arrays for the pool-level figures, over the 500-shot pool.

For every aligned shot: the archived truth, the tearing CNN scored twice (on
its own archived inputs and on labelmaker's reconstruction) and the survival
model's three risks read back from the published labels, all on the same
matched rows.
"""
import sys

import numpy as np

from labelmaker import catalog, validate
from labelmaker.config import Paths
from labelmaker.labels.store import read_label
from labelmaker.models import registry
from labelmaker.models.d3d_tearing_onset_cnn1d.spec import ARTIFACTS, OUTPUT_SPEC, SLUG
from labelmaker.models.runners import keras_h5

out = sys.argv[1]
p = Paths.from_env()
spec = registry.load_adapter(SLUG).input_spec
graphs = keras_h5.load_ensemble(p.models / SLUG / n for n in ARTIFACTS)
DSM = "d3d_tearing_time_to_event_dsm"
RISKS = ("tm_risk_250ms", "tm_risk_500ms", "tm_risk_1s")
HORIZONS = (0.25, 0.5, 1.0)

# Half the pool is in the survival model's own training set (214 of 500
# shots), so every pooled number needs the flag beside it. `training_shots` is
# empty for a model that does not record its own, which reports every row as
# held out - true of the tearing CNN, whose training store is the archive
# itself, a different problem its card states.
TRAINING = registry.load_adapter(DSM).training_shots

cols = {k: [] for k in (
    "shot", "t", "valid", "truth_tm", "truth_bn", "p_arch", "p_recon", "b_arch",
    "b_recon", "dsm_valid", "time_to_onset", "has_onset", "in_training",
)}
for r in RISKS:
    cols[r] = []
per_shot = []
skipped = {}
for shot in catalog.read_shot_file(p.root / "shots_500.txt"):
    m = validate._matched_shot(shot, spec, p, validate.TM_ARCHIVE)
    if m.skip_reason:
        skipped[shot] = m.skip_reason
        continue
    idx = np.asarray(m.info["index"], dtype=int)
    order = np.argsort(m.built.t[idx], kind="mergesort")
    idx = idx[order]
    y = np.asarray(m.got["y"], dtype=np.float64)[order]
    n = idx.size
    t = np.asarray(m.built.t, dtype=np.float64)[idx]
    tm = y[:, 1] > 0.5
    onset = float(t[np.argmax(tm)]) if tm.any() else np.nan

    def cnn(scalars, profiles):
        dec = OUTPUT_SPEC.decode(keras_h5.predict_members(graphs, [scalars, profiles]))
        return dec["tm_prob"].mean, dec["betan"].mean

    p_a, b_a = cnn(np.asarray(m.got["x0"], dtype=np.float64)[order],
                   np.asarray(m.got["x1"], dtype=np.float64)[order])
    p_r, b_r = cnn(m.built.scalars[idx], m.built.profiles[idx])

    dsm_valid = np.zeros(n, bool)
    risks = {r: np.full(n, np.nan) for r in RISKS}
    try:
        dsm_valid = read_label(p.labels_file(shot), DSM, f"{RISKS[0]}_valid").y[0][idx].astype(bool)
        for r in RISKS:
            risks[r] = read_label(p.labels_file(shot), DSM, r).y[0][idx]
    except (KeyError, OSError):
        pass

    cnn_valid = np.asarray(m.built.valid, dtype=bool)[idx]
    cols["shot"].append(np.full(n, shot)); cols["t"].append(t)
    cols["valid"].append(cnn_valid); cols["dsm_valid"].append(dsm_valid)
    cols["truth_tm"].append(tm); cols["truth_bn"].append(y[:, 0])
    cols["p_arch"].append(p_a); cols["p_recon"].append(p_r)
    cols["b_arch"].append(b_a); cols["b_recon"].append(b_r)
    cols["time_to_onset"].append(onset - t if np.isfinite(onset) else np.full(n, np.nan))
    cols["has_onset"].append(np.full(n, np.isfinite(onset)))
    cols["in_training"].append(np.full(n, shot in TRAINING))
    for r in RISKS:
        cols[r].append(risks[r])
    full_valid = read_label(p.labels_file(shot), SLUG, "tm_prob_valid").y[0].astype(bool)
    try:
        dsm_full = read_label(p.labels_file(shot), DSM, f"{RISKS[0]}_valid").y[0].astype(bool)
    except (KeyError, OSError):
        dsm_full = np.zeros_like(full_valid)
    per_shot.append([shot, n, int(cnn_valid.sum()), int(tm.sum()),
                     float(full_valid.mean()), float(dsm_full.mean()),
                     onset, float(np.nanmax(risks["tm_risk_1s"]) if n else np.nan),
                     float(shot in TRAINING)])

arrays = {k: np.concatenate(v) for k, v in cols.items()}
arrays["per_shot"] = np.asarray(per_shot, dtype=np.float64)
arrays["horizons"] = np.asarray(HORIZONS)
np.savez_compressed(out, **arrays)
n = arrays["shot"].size
print(f"in-training shots {int(arrays['per_shot'][:, 8].sum())} of {len(per_shot)}")
print(f"shots {len(per_shot)} skipped {len(skipped)} rows {n} "
      f"cnn-valid {int(arrays['valid'].sum())} dsm-valid {int(arrays['dsm_valid'].sum())} "
      f"positives {int(arrays['truth_tm'].sum())} shots-with-onset "
      f"{int(np.isfinite(arrays['per_shot'][:, 6]).sum())}")
