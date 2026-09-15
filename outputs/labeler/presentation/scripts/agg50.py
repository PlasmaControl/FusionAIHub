"""Aggregate the per-shot analyses: how did each model do across the 50 truth shots?"""
import json, glob, numpy as np
rows = []
for f in sorted(glob.glob("/scratch/gpfs/nc1514/FusionAIHub/outputs/labelmaker/analysis/*/*_analysis.json")):
    d = json.load(open(f))
    if not d["truth"]["available"] or d["truth"]["onset_s"] is None:
        continue
    r = {"shot": d["shot"], "onset_s": d["truth"]["onset_s"]}
    for lab, v in d["labels"].items():
        t = v.get("truth", {})
        if t.get("scored"):
            r[lab] = {k: t.get(k) for k in ("auroc", "f1", "precision", "recall", "lead_time_s", "n", "rmse")}
    rows.append(r)
print("shots with truth and an onset:", len(rows))
for lab in ("d3d_tearing_onset_cnn1d/tm_prob", "d3d_tearing_time_to_event_dsm/tm_risk_1s",
            "d3d_tearing_onset_cnn1d/betan"):
    vals = [r[lab] for r in rows if lab in r]
    def med(k):
        a = np.array([v[k] for v in vals if v.get(k) is not None], float)
        return (np.median(a), a.size) if a.size else (float("nan"), 0)
    print(f"\n{lab}  ({len(vals)} shots)")
    for k in ("auroc", "f1", "precision", "recall", "lead_time_s", "rmse"):
        m, n = med(k)
        if n: print(f"   median {k:12s} {m:+.3f}   (on {n} shots)")
    lead = np.array([v["lead_time_s"] for v in vals if v.get("lead_time_s") is not None], float)
    if lead.size: print(f"   fires before the archived onset on {int((lead > 0).sum())}/{lead.size} shots")
json.dump(rows, open("/scratch/gpfs/nc1514/FusionAIHub/outputs/labelmaker/analysis/per_shot_scores.json", "w"), indent=1)
