"""Rank sweep arms by DISTANCE TO THE ORACLE PROFILE, with the hard gates applied.

Why not a single scalar. `spec_nrmse` ranks the blur best and is therefore a FLOOR only; the
5-reference validation (logs/specport_metricval_*.out) rejected `ms_ssim` and
`mode_track_f1` as ranking keys because both put a blurred codec above a structured one, and
showed `spectral_contrast_ratio` scoring the static envelope above every codec. So no metric
here is trusted on its own. What IS trusted is the rank-192 out-of-sample PCA ORACLE: a
measured, achievable-in-principle reconstruction at exactly the codec's dimension budget.

    ORACLE PROFILE (240 held-out mhr windows, per-freq-z, full 0-250 kHz)
        ms_ssim 0.4285   hf_ratio 0.2149   spec_nrmse 0.8528   contrast 0.6898

An arm is scored by how close its 4-vector sits to that profile, each axis normalised by that
axis's range across the five validation references (a fixed, data-derived scale, so no axis
dominates merely because of its units). The per-axis deltas are printed alongside the total
so the reasoning is visible rather than hidden behind the scalar.

HARD GATES, applied before ranking:
    spec_nrmse < 1.0000            must beat the per-window constant-mean anchor
    n_distinct_codes >= 64         codebook alive (27 of 28 arms in this project's history
                                   ended at 1 code, so this is not a formality)
    patch_lattice_ratio <= 20.0    not drifting toward ms2_s1's 61.02 checkerboard

Usage: _specport_rank_arms.py <audit.json> [more.json ...]
"""
import json
import sys

ORACLE = {"ms_ssim": 0.4285, "hf_ratio": 0.2149, "spec_nrmse": 0.8528,
          "spectral_contrast_ratio": 0.6898}
# range of each axis across the 5 validation references (GT_self / oracle192 / ms2_s1 /
# lr1e4d6 / tmean) -- a fixed data-derived normaliser, quoted so it can be checked.
SCALE = {"ms_ssim": 1.0000 - 0.2604, "hf_ratio": 1.0000 - 0.0145,
         "spec_nrmse": 1.0952 - 0.0000, "spectral_contrast_ratio": 1.0000 - 0.4324}
GATES = {"spec_nrmse": ("<", 1.0000), "n_distinct_codes": (">=", 64),
         "patch_lattice_ratio": ("<=", 20.0)}

# The five validation references, ALWAYS printed alongside the arms so the ranking is
# self-contained. Measured on 240 held-out mhr windows (logs/specport_metricval_5411728.out).
# Note `tmean` sits CLOSER to the oracle than the current best codec `lr1e4d6` does
# (0.1523 vs 0.2076): the static envelope's distance is therefore the bar an arm has to beat
# to be worth shipping at all, not merely "closest arm wins".
REFERENCES = [
    {"arm": "[ref]oracle192", "ms_ssim": 0.4285, "hf_ratio": 0.2149, "spec_nrmse": 0.8528,
     "spectral_contrast_ratio": 0.6898, "n_distinct_codes": 1000, "patch_lattice_ratio": 1.14},
    {"arm": "[ref]tmean", "ms_ssim": 0.2830, "hf_ratio": 0.0162, "spec_nrmse": 0.8789,
     "spectral_contrast_ratio": 0.7959, "n_distinct_codes": 1000, "patch_lattice_ratio": 1.14},
    {"arm": "[ref]lr1e4d6", "ms_ssim": 0.3310, "hf_ratio": 0.0145, "spec_nrmse": 0.8985,
     "spectral_contrast_ratio": 0.4324, "n_distinct_codes": 1000, "patch_lattice_ratio": 3.88},
    {"arm": "[ref]ms2_s1", "ms_ssim": 0.2604, "hf_ratio": 0.8265, "spec_nrmse": 1.0952,
     "spectral_contrast_ratio": 0.7534, "n_distinct_codes": 1000, "patch_lattice_ratio": 61.02},
    {"arm": "[ref]prod", "ms_ssim": 0.2374, "hf_ratio": 0.1585, "spec_nrmse": 1.1320,
     "spectral_contrast_ratio": 0.4106, "n_distinct_codes": 50, "patch_lattice_ratio": 49.42},
]
rows = list(REFERENCES)
for f in sys.argv[1:]:
    rows.extend(json.load(open(f)))

print("ORACLE PROFILE (the ranking target): " +
      "  ".join(f"{k}={v}" for k, v in ORACLE.items()))
print("axis normaliser (range over the 5 validation references): " +
      "  ".join(f"{k}={v:.4f}" for k, v in SCALE.items()))
print()
hdr = (f"{'arm':<14}{'ms_ssim':>9}{'hf_ratio':>10}{'nrmse':>8}{'contrast':>10}"
       f"{'codes':>7}{'lattice':>9}   {'d_oracle':>9}  gates")
print(hdr)
print("-" * len(hdr))
scored = []
for r in rows:
    miss = [f"{k}{op}{lim}" for k, (op, lim) in GATES.items()
            if k in r and not ((r[k] < lim) if op == "<" else
                               (r[k] >= lim) if op == ">=" else (r[k] <= lim))]
    d = sum(abs(r.get(k, float("nan")) - v) / SCALE[k] for k, v in ORACLE.items()) / len(ORACLE)
    scored.append((d, r, miss))
    print(f"{r['arm']:<14}{r.get('ms_ssim', float('nan')):>9.4f}"
          f"{r.get('hf_ratio', float('nan')):>10.4f}{r.get('spec_nrmse', float('nan')):>8.4f}"
          f"{r.get('spectral_contrast_ratio', float('nan')):>10.4f}"
          f"{r.get('n_distinct_codes', -1):>7}{r.get('patch_lattice_ratio', float('nan')):>9.2f}"
          f"   {d:>9.4f}  {'PASS' if not miss else 'FAIL:' + ','.join(miss)}")
print("-" * len(hdr))
print()
print("PER-AXIS distance to the oracle (normalised units; 0 = on the oracle):")
for d, r, miss in sorted(scored):
    parts = "  ".join(
        f"{k.replace('spectral_contrast_ratio','contrast').replace('mode_','')}"
        f"={abs(r.get(k, float('nan')) - v) / SCALE[k]:+.3f}" for k, v in ORACLE.items())
    print(f"  {r['arm']:<14} total={d:.4f}   {parts}"
          f"   {'' if not miss else '[GATED OUT]'}")
arms = [(d, r) for d, r, m in sorted(scored) if not m and not r["arm"].startswith("[ref]")]
tm = next((d for d, r, _m in scored if r["arm"] == "[ref]tmean"), None)
print()
if arms:
    print(f"CLOSEST TO THE ORACLE among gate-passing ARMS: {arms[0][1]['arm']} "
          f"(d={arms[0][0]:.4f})")
    if tm is not None:
        beat = [r["arm"] for d, r in arms if d < tm]
        which = ", ".join(beat) if beat else (
            "NONE -- no arm is worth shipping over the static envelope on this criterion")
        print(f"THE BAR: the static time-mean envelope sits at d={tm:.4f}. "
              f"Arms that beat it: {which}")
else:
    print("NO ARM PASSES THE HARD GATES.")
