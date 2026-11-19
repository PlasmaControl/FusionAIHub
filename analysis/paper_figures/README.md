# Paper figures (Results, Figs 1–5)

Model: `runs/soup_lu18_h16_30/dynamics_step30000.pt` (1:1 weight average of
`ss75g_live_unroll@18000` and `ss75g_spec3_h16@30000`), Gumbel reveal noise 3, seeds 1–8
(since 2026-11-13; seeds 5–8 run with the history cache by `claude_scratch/paper8/paper8.sh`).
Figs 1–5 render from the 8-seed set: `PAPER_FIG_ENS=.../demo_figures_live_unroll/ens8_soup`
`PAPER_FIG_DATA=/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data8`; without them the
script reads the 4-seed `ens4_soup` / `data`. Fig 5's TS and CER of 204011 and its upper camera
come from `_infer_204011_{ts,cer,tvupper}_bs` (hidden including their history:
`--infer_blind --infer_bootstrap`, cameras `--forbid_absent`; `claude_scratch/blind_infer/fig5_hidden.sh`).
Traces are the decoded seed mean; spectrograms show the medoid seed.

| Figure | File | Shots |
|---|---|---|
| 1 Replay | `figures/fig1_replay_204011.{pdf,png}` | 204011 |
| 2 Actuators | `figures/fig2_actuators.{pdf,png}` | 204011 torque → 0 / ½, 201240 ECH → 0 / ½ |
| 3 Detachment replay | `figures/fig3_detachment_replay.{pdf,png}` | 201081, 201085 |
| 4 Alternative gas paths | `figures/fig4_detachment_paths.{pdf,png}` | 201081 under the actuators of 201079 / 201083 / 201084 |
| 5 Missing diagnostics | `figures/fig5_missing_diagnostics.{pdf,png}` | 204011, 200729 (inferred curves pending) |

## Profile coordinate

Every profile (Figs 1, 2, 6 and `--extra`) is plotted against psi_N, from the EFIT01 slice
nearest each time (`make_figures.chord_psin`, `geqdsk_<shot>.npz` in `PAPER_FIG_DATA`,
fetched by `fetch_geqdsk.py`): TS core chords at their (R, Z), CER tangential chords where
their view crosses the beam (`cer_r`, `cer_z`). Changed-actuator predictions are placed on
the measured equilibrium. A shot without a `geqdsk_<shot>.npz` raises an error.

## Pipeline

Data locations:
- Rollout extracts (input): `/scratch/gpfs/EKOLEMEN/ps9551/ignite/demo_figures_live_unroll/ens4_soup/`
- Intermediate data (3 GB, not in git): `/scratch/gpfs/EKOLEMEN/ps9551/ignite/paper_figures/data/`
  (override with `PAPER_FIG_DATA`)
- Thomson scattering chord positions: `figures/ts_rz.zip` (`ts_rz/ts_rz_<shot>.csv`:
  system, index, R, Z, phi, in the data files' channel order). TS chords are labelled by
  R and Z. Shots with fewer core chords than the
  dataset's 44 are zero-padded at the end (204011 has 42); `make_figures.ts_z` refuses
  those padded slots.
- CER chord radii: `figures/cer_geometry/*_CER_Chord_Major_Radii_(cm).txt` (C. Chrystal's
  tables per shot range). Only the tangential chords T01-T48 are used (codec index = chord - 1),
  at the radius of the left source they view (30LT or 330LT). CER profiles (`cer_profile_chords`) draw T_i and |v_tor| on the same chords: every outboard
  (R >= 1.6 m) tangential chord the shot recorded for both TEMP and ROT (`cer_recorded`, read once
  from the raw file into `PAPER_FIG_DATA/cer_recorded.json`) whose measurement decodes to finite
  values. 204011 stored rotation only on T01-T08, T25-T28, T33-T36 and T41-T48 (T01 and T03
  decode to NaN), which leaves 18 pairs from psi_N 0 to 1.17.

1. **Rollouts and extracts** (GPU, about 25 min): `sbatch analysis/paper_figures/paper_rollouts.sh`.
   It rolls out 201081 under its measured actuators and the three donor shots, and
   re-extracts 204011 (replay, ramps), 201081 and 201085 into `ens4_soup/`. The other
   archives (204011/201240 ramps, 200729) are from the 2026-11-03 demo rollouts in
   `runs/soup_lu18_h16_30/_demo_*`.
2. **Spectrogram channel ranking** (4 GPUs, about 6 min): `sbatch analysis/paper_figures/rank_spectro.sh`
   writes `rank_<shot>.{json,npz}`. Scores per channel: mode strength, match, replay, and
   response per scenario (see the `rank_spectro.py` docstring). The best channel is chosen
   within each diagnostic, because the scores are not comparable across diagnostics.
3. **Measured gas and filterscopes** (login node):
   `pixi run --frozen -e stellarai python analysis/paper_figures/detach_gt.py 201079 201080 201081 201082 201083 201084 201085`
4. **Slow-TS chord ranking** (printout only; its choices are hardcoded in `CHORDS` in
   `make_figures.py`): `python analysis/paper_figures/rank_chords.py`
5. **Render** (login node, about 20 min for all five):

       PYTHONPATH=/scratch/gpfs/EKOLEMEN/ps9551/ignite/pyvendor \
           pixi run --frozen -e stellarai python analysis/paper_figures/make_figures.py [--only=1,2] [--gt_twice]

   `--only` re-renders just those figures. `--gt_twice` puts the measurement in every
   prediction slot, as a layout check.

## Style

6.5 in wide (the `\textwidth` of `main.tex`), 9 pt Nimbus Roman (Times, as the paper) with STIX math, no titles, channel names in the grey label box.
Palette A is checked colour-blind safe by `cvd_check.py`:

| Role | Colour |
|---|---|
| Measured | sky blue `#56b4e9`, thick, drawn underneath |
| Prediction | orange `#e07b1a`, thin, on top |
| Changed actuator and its prediction | purple `#7b3294` |

`notes_detachment_viz.md` surveys how to show detachment over time. It is groundwork for a
DZ-vs-time alternative to Fig 3, which needs X-point and strike-point pixel positions in
the lower TangTV view.
