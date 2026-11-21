#!/bin/bash
# Re-render every presentation figure and video from the demo data files (CPU only, no rollouts).
# The data files are written by `demo_controllability_fig.py extract`; see that script.
#   bash analysis/render_demo_figures.sh [figure dir]
set -uo pipefail
D=${1:-/scratch/gpfs/EKOLEMEN/ps9551/ignite/demo_figures}
P=${PYTHON:-.pixi/envs/stellarai/bin/python}
F=analysis/demo_controllability_fig.py
V=analysis/demo_rollout_video.py
C204011="cer_ti=24,2,1;cer_rot=19,18,20"
run() { "$P" "$@" 2>&1 | grep -E "Traceback|Error" || true; }

# 201240: best validation, NBI-off / ECH-off scenarios (guidance 2)
run $F plot --kind validation --data $D/demo_201240_data.npz --out $D/201240_validation.png
run $F plot --kind scenarios --data $D/demo_201240_data.npz --out $D/201240_scenarios.png
run $F plot --kind combined --data $D/demo_201240_data.npz --out $D/201240_combined.png
run $F plot --kind validation --data $D/demo_201240_both_data.npz --width 19 --out $D/201240_validation_ece_mhr.png
# 204011: torque scenario and ECE/MHR modes (guidance 3; g1 = unguided)
run $F plot --kind validation --data $D/demo_204011_data.npz --channels "$C204011" --out $D/204011_validation.png
run $F plot --kind scenarios --data $D/demo_204011_data.npz --channels "$C204011" --out $D/204011_scenarios.png
run $F plot --kind combined --data $D/demo_204011_data.npz --channels "$C204011" --out $D/204011_combined.png
run $F plot --kind validation --data $D/demo_204011_g3_both_data.npz --channels "$C204011" --width 19 --out $D/204011_g3_validation_ece_mhr.png
run $F plot --kind validation --data $D/demo_204011_g1_both_data.npz --channels "$C204011" --width 19 --out $D/204011_validation_ece_mhr.png
# 200729: validation with CO2 / BES, cameras
run $F plot --kind validation --data $D/demo_200729_g2_spec_data.npz --spectro_ylim "co2=100,bes=100" --width 23 --out $D/200729_g2_validation_co2_bes_ece.png
"$P" - "$D" <<'EOF'
import sys, numpy as np
D = sys.argv[1]
d = dict(np.load(f"{D}/demo_200729_g2_spec_data.npz"))
d["spectro"] = np.array("co2,bes")
for k in [k for k in d if k.endswith("__ece")]:
    d.pop(k)
np.savez(f"{D}/demo_200729_g2_co2bes_data.npz", **d)
EOF
run $F plot --kind validation --data $D/demo_200729_g2_co2bes_data.npz --spectro_ylim "co2=100,bes=100" --width 19 --out $D/200729_g2_validation_co2_bes.png
for g in 1 2 3; do
  run $F plot --kind validation --data $D/demo_200729_g${g}_data.npz --width 19 --out $D/200729_g${g}_validation_ece_mhr.png
  for c in lower upper; do run $F plot --kind camera --camera tangtv_$c --data $D/demo_200729_g${g}_data.npz --out $D/200729_g${g}_camera_$c.png; done
done
for g in 2 3; do run $F plot --kind scenarios --data $D/demo_200729_g${g}_data.npz --out $D/200729_g${g}_scenarios.png; done
# other camera / validation shots
run $F plot --kind camera --data $D/demo_204989_camdata.npz --out $D/204989_camera.png
run $F plot --kind camera --camera tangtv_lower --data $D/demo_203213_val_data.npz --out $D/203213_camera_lower.png
run $F plot --kind validation --data $D/demo_203213_val_data.npz --width 19 --out $D/203213_validation_ece_mhr.png
run $F plot --kind validation --data $D/demo_204989_both_data.npz --width 19 --out $D/204989_validation_ece_mhr.png
# videos
run $V --data $D/demo_201240_both_data.npz --spectro ece --camera none --out $D/201240_rollout.mp4
run $V --data $D/demo_200729_g2_spec_data.npz --spectro co2 --ylim 100 --camera tangtv_lower --out $D/200729_g2_rollout_lower_co2.mp4
for g in 2 3; do for c in lower upper; do
  run $V --data $D/demo_200729_g${g}_data.npz --spectro ece --camera tangtv_$c --out $D/200729_g${g}_rollout_$c.mp4
done; done
echo "[render] done"
