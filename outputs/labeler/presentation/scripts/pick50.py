"""Shots with archived ground truth AND an actual onset, for the shot-level set."""
import json, numpy as np
from labelmaker import validate, catalog
from labelmaker.config import Paths
from labelmaker.labels.store import read_label

p = Paths.from_env()
shots = catalog.read_shot_file(p.root / "shots_500.txt")
rows = []
for shot in shots:
    tr = validate.archived_truth(shot, p)
    if not tr["available"]:
        continue
    cov = {}
    for slug, name in (("d3d_tearing_onset_cnn1d", "tm_prob"),
                       ("d3d_tearing_time_to_event_dsm", "tm_risk_1s")):
        try:
            v = read_label(p.labels_file(shot), slug, f"{name}_valid").y[0].astype(bool)
            cov[slug] = float(v.mean())
        except (KeyError, OSError):
            cov[slug] = 0.0
    rows.append({"shot": shot, "onset_s": tr["onset_s"], "n_truth_rows": tr["n_rows"],
                 "n_positive": int(tr["tm_label"].sum()),
                 "cnn_valid": cov["d3d_tearing_onset_cnn1d"],
                 "dsm_valid": cov["d3d_tearing_time_to_event_dsm"]})
with_onset = [r for r in rows if r["onset_s"] is not None]
print("aligned", len(rows), "with an onset", len(with_onset))
rng = np.random.default_rng(20260905)
pick = sorted(rng.choice(len(with_onset), size=min(50, len(with_onset)), replace=False))
sel = [with_onset[i] for i in pick]
catalog.write_shot_file(p.root / "shots_truth50.txt", [r["shot"] for r in sel])
json.dump({"selected": sel, "all_aligned": rows}, open(p.root / "shots_truth50.json", "w"), indent=1)
print("picked", len(sel), "->", p.root / "shots_truth50.txt")
print("cnn valid median %.2f, dsm valid median %.2f, onset median %.2f s, positives median %d" % (
    np.median([r["cnn_valid"] for r in sel]), np.median([r["dsm_valid"] for r in sel]),
    np.median([r["onset_s"] for r in sel]), np.median([r["n_positive"] for r in sel])))
