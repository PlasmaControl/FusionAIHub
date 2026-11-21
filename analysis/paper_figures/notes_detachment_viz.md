# Detachment-level figure: literature and design notes

Target: show the detachment level over time for DIII-D 201081 / 201085 (gas-puff detachment,
~4 MW NBI, no ECH): measured, predicted under the measured gas, and predicted under other
gas trajectories. Inputs: `tangtv_lower` frames (120x360, each frame z-scored by the codec), TS
n_e/T_e, CER, gas, NBI. No Langmuir probes, no divertor Thomson (DTS), no bolometer in the model.

## 0. Key finding: these exact shots were run with an image-based front controller

**Chen, Byun, Jalalvand, Kim, Rothstein, Scotti, Allen, Eldon, Erickson, Kolemen (2025/26).**
"Regulation Compliant AI for Fusion: Real-Time Image Analysis-Based Control of Divertor
Detachment in Tokamaks". arXiv:2507.02897 (21 June 2025); journal DOI listed on arXiv as
10.1088/1741-4326/ae3972 (Nucl. Fusion). https://arxiv.org/abs/2507.02897
- Shots 200963–200978 (L-mode, e.g. 200977) and 201069–201085 (H-mode, 4 MW NBI, D2 gas only)
  were run with feedback on a front position computed from the lower-divertor C-III (465 nm)
  tangential camera (rt-TangTV, 720x480, 30 Hz).
- Control variable: **DZ = 1 − (Z_X − Z_E)/(Z_X − Z_S)**. DZ = 0: emission front at the strike
  point. DZ = 1: front at the X-point. Z_E is the RMS-weighted elevation of the emission of
  all pixels outboard of the X-point column, calibrated to metres. Training labels came from
  ~3,500 tomographically inverted frames; real time uses a linear map (learned activation map).
- **Their camera frames were normalised to zero mean / unit std before regression**, the same
  normalisation our codec applies. So a DZ-type metric is defined on exactly the information we
  keep.
- 201081: DZ ramped up then down; divertor-Thomson T_e cliff (about 20 eV to 3 eV) at
  ~2.65 s and ~4.45 s; the bifurcation sits near **DZ ≈ 0.5**. IRTV heat flux falls to about
  0.55 MW/m² at high DZ. Tracking error 2% after lag correction. Gas acts with about 200 ms delay.
- 201085: DZ held at 1.0 while the X-point and strike point moved (1.52 to 1.42 m). DZ first
  dropped to 0.45 because the front moved radially, not because the plasma reattached. They
  added a radial correction (R_edge = 1.35 m, α = 2.1).
- Consequence for us: use DZ and its name. The PCS probably recorded the real-time DZ and its
  target (check PTDATA/PCS pointnames for 201081/201085). That gives a measured reference curve
  to validate our offline image metric against.

## 1. Sources

DIII-D
1. Eldon, Kolemen, Barton, Briesemeister, Humphreys, Leonard, Maingi, Makowski, McLean, Moser,
   Stangeby (2017). "Controlling marginally detached divertor plasmas". Nucl. Fusion 57, 066039.
   doi:10.1088/1741-4326/aa6b16. Control variable is DTS T_e near the target (ELM-filtered with
   filterscope D-alpha, RC-smoothed), actuated by divertor gas (GAS C). Detachment threshold is
   defined as T_e = 5 eV. Requests: 2 eV (detached), 5 eV (dithers across the "T_e cliff"),
   12 eV (attached). T_e falls into two clusters: 1–3 eV and 8–20 eV. Pedestal n_e is the same
   (within error) for the marginally detached and dithering cases, so upstream density alone
   does not decide the state. Figures: T_e measured vs request vs time with gas below (Fig. 5);
   T_e vs height above target (Fig. 8); T_e vs L_parallel (Fig. 1, from McLean).
2. Kolemen, Allen, Bray, Fenstermacher, Humphreys, Hyatt, Lasnier, Leonard, Makowski, McLean,
   Maingi, Nazikian, Petrie, Soukhanovskii, Unterberg (2015). "Heat flux management via advanced
   magnetic divertor configurations and divertor detachment". J. Nucl. Mater. 463, 1186–1190.
   doi:10.1016/j.jnucmat.2014.11.099. Defines the **detachment front** as where T_e drops
   below a few eV, located from DTS T_e plus impurity line ratios. The near-strike-point valve
   holds the front "at a pre-set distance from the divertor target", and a far valve holds core
   density. Result: front held "at the mid-distance between the strike point and the X-point"
   (153816 vs 153814). Figures: 2-D projected DTS T_e map, divertor radiation profile.
3. Boyer, Scotti, Gajaraj (2024). "Neural networks for estimation of divertor conditions in
   DIII-D using C III imaging". Nucl. Fusion 64 (no. 10, Sept 2024; article number not checked).
   doi:10.1088/1741-4326/ad724d. "The poloidal distance from the target to the C III emission
   front is used to represent the level of detachment". CNNs on TangTV C-III frames: detachment
   class F1 up to 0.99, front-location RMSE about 2 cm, X-point/strike-point RMSE about 1 cm.
4. Victor, Scotti (2024). CNN detachment detection from upper/lower filtered divertor cameras
   (C III 465 nm), Rev. Sci. Instrum. (HTPD 2024). https://www.osti.gov/biblio/2439941. When the
   target T_e drops to about 2 eV ("detachment cliff"), C-III emission at the strike point drops
   sharply. Accuracy 100% upper, 96% lower divertor.
5. Fenstermacher, Meyer, Wood, Nilson, Ellis, Brooks (1997). "A tangentially viewing visible TV
   system for the DIII-D divertor". Rev. Sci. Instrum. 68, 974. doi:10.1063/1.1147729.
   Tangential view at about X-point height, about 1 cm resolution, integrates about 90° toroidally.
6. Fenstermacher et al. (1999). "Physics of the detached radiative divertor regime in DIII-D".
   Plasma Phys. Control. Fusion 41, A345. doi:10.1088/0741-3335/41/3A/028. Inverted 2-D CII/CIII
   emission. C-III is a good proxy for the main radiator (C-IV). With gas, emission appears first
   near the X-point and then moves upward as density rises (EPS 1999 companion paper).
7. Chen et al. (2024) APS-DPP abstract PP12.33, "Real-Time Machine-Learning Enabled Emission
   Front Control at DIII-D". https://meetings-archive.aps.org/dpp/2024/pp12/33/ (precursor of 0).
8. Jaervinen et al. (2018). "E×B flux driven detachment bifurcation in the DIII-D tokamak".
   Phys. Rev. Lett. 121, 075001. Physics of the T_e cliff (why DZ jumps around 0.5).
9. McLean et al. (2015). J. Nucl. Mater. 463, 533. T_e(L_parallel) profiles through the attached,
   partially detached and fully detached states (reproduced as Eldon 2017 Fig. 1).
10. Leonard (2018). "Plasma detachment in divertor tokamaks". Plasma Phys. Control. Fusion 60,
    044001. doi:10.1088/1361-6587/aaa7a9. Review; standard rollover and front language.

TCV
11. Ravensbergen, van Berkel, Perek, Galperti, Duval, Février, ..., de Baar (2021). "Real-time
    feedback control of the impurity emission front in tokamak divertor plasmas". Nat. Commun.
    12, 1105. doi:10.1038/s41467-021-21268-3. Leg detected in raw filtered (C-III) MANTIS
    images, no tomography. The front is the point along the leg where emission falls to 50%,
    mapped to poloidal distance from the target. Shown as front position vs time, with the
    reference and the gas valve trace. This is the closest Nature-family precedent for our panel (b).
12. Harrison et al. (2017). "Detachment evolution on the TCV tokamak". Nucl. Mater. Energy 12,
    1071. doi:10.1016/j.nme.2016.10.020. Inverted C-III and D-alpha. Before the ion-current
    rollover, the outer-leg C-III and D-alpha emission recede slowly from the strike point
    toward the X-point.
13. Theiler et al. (2017). "Results from recent detachment experiments in alternative divertor
    configurations on TCV". Nucl. Fusion 57, 072008. doi:10.1088/1741-4326/aa5fb7. C-III front
    position (poloidal distance from target) vs line-averaged density for different
    geometries. This is the "phase portrait" format (front vs n̄_e).
14. TCV X-point-target detachment dynamics (2026), arXiv:2606.23432. CIII front taken at 50% of
    peak emission along the leg from MANTIS inversions, given as L_pol from the target.

MAST-U
15. Wijkamp, Allcock, Février, ..., Verhaegh et al. (2023). "Characterisation of detachment in
    the MAST-U Super-X divertor using multi-wavelength camera imaging". Nucl. Fusion 63, 056003.
    doi:10.1088/1741-4326/acc191. 2-D emissivity inversions of several lines. The Fulcher-band
    front is the proxy for the ionisation front, plotted vs upstream density. D-alpha and
    D-gamma behave differently from impurity lines in deep detachment (recombination near the
    target).
16. Kool, Verhaegh, Derks, Wijkamp, et al. (2025). "Demonstration of Super-X divertor exhaust
    control for transient heat load management in compact fusion reactors". Nature Energy 10,
    1116. doi:10.1038/s41560-025-01824-7 (arXiv:2407.07784). Real-time Fulcher-front control by
    divertor gas, shown as front position vs time against its reference, with the gas command
    below. Nature-family precedent for panel (b).

AUG, JET, KSTAR, EAST
17. Kallenbach et al. (2010). "Divertor power load feedback with nitrogen seeding in ASDEX
    Upgrade". Plasma Phys. Control. Fusion 52, 055002. doi:10.1088/0741-3335/52/5/055002.
    "T_div" comes from the target thermoelectric (shunt) current, scaled so that 500 A
    corresponds to 10 eV. It is a qualitative detachment indicator: about 0 eV or below means
    detached.
18. Lingertat, Loarte et al. (1998), "Plasma detachment in JET Mark I divertor experiments",
    Nucl. Fusion 38, 331 (Loarte et al. 1998). **Degree of detachment** DoD = Γ_target,expected
    / Γ_target,measured, where Γ_expected ∝ n_e,up² comes from the high-recycling (two-point)
    scaling. DoD > 1 means detached. Needs target probes, so we cannot compute it.
19. Lipschultz, Parra, Hutchinson (2016). "Sensitivity of detachment extent to magnetic
    configuration and external parameters". Nucl. Fusion 56, 056007.
    doi:10.1088/0029-5515/56/5/056007. DLS model: front position along the leg as a function of
    a control parameter (n_up, f_z, P_SOL). Gives the "detachment window" and the theory behind a
    front-vs-density portrait.
20. Eldon et al. (2022). "Enhancement of detachment control with simplified real-time modelling
    on the KSTAR tokamak". Plasma Phys. Control. Fusion 64, 075002. Defines the **attachment
    fraction** A_frac = I_sat / I_sat,attached-model (Langmuir probes). Gupta et al. (2025),
    arXiv:2505.07978: KSTAR W divertor, A_frac and the DivControlNN surrogate.
21. EAST: control on target T_e or I_sat from Langmuir probes (as summarised in Gupta 2025).
    Wang L. et al. (2021), Nat. Commun. 12, 1365, doi:10.1038/s41467-021-21645-y (mostly DIII-D
    high-βp with detachment) is a Nature Commun. example of detachment traces vs time.

## 2. Techniques: who uses what, and can we compute it?

| Indicator | Used by | Signal | From z-scored frames? |
|---|---|---|---|
| T_e at target vs time (T_e cliff, 5 eV) | Eldon 2017, Kolemen 2015 (DIII-D DTS) | divertor Thomson | No. Annotation only (measured shot). |
| Degree of detachment (DoD), rollover of Γ_t vs n_up | Loarte/Lingertat 1998 (JET), Leonard 2018 | target probes | No |
| A_frac | KSTAR (Eldon 2022, Gupta 2025), EAST | Langmuir I_sat | No |
| T_div shunt current | AUG (Kallenbach 2010) | target currents | No |
| Radiation / X-point radiator position | AUG (Bernert), DIII-D bolometry | bolometer, AXUV | No (not in model) |
| **C-III front: poloidal distance target→front, or DZ ∈ [0,1]** | DIII-D (Chen 2025, Boyer 2024), TCV (Ravensbergen 2021, Harrison 2017, Theiler 2017) | filtered tangential camera, inverted or raw | **Yes**. Shape only; see §4. |
| Fulcher / D-alpha front | MAST-U (Wijkamp 2023, Kool 2025) | MWI camera | Yes, if our channel is the right filter (check) |
| Emission profile along the leg vs time (kymograph) | TCV, MAST-U papers (front plots), DIII-D 2-D DTS projections | camera | **Yes** |
| Detached / attached class probability | Boyer 2024, Victor & Scotti 2024 | C-III frames, CNN | Possibly. Their CNN would need raw-intensity frames; not shape only. |
| X-point vs target emission ratio | Fenstermacher 1999 (qualitative: emission moves to the X-point) | camera | Yes. A ratio of shifted intensities is affine-sensitive; prefer a min-max-normalised profile. |

## 3. Recommended figure designs (two-column, Nature style)

### (A) Hero panel: front position vs time, measured, predicted and counterfactual
- Layout, full width (~180 mm x 90 mm), 3 stacked rows with a shared time axis (s):
  1. D2 gas rate (Torr·L/s or the units in the data) for the measured trajectory (blue) and
     each alternative trajectory, each alternative in its own third colour.
  2. **DZ(t)** (0 = strike point, 1 = X-point; label "emission front height, DZ"). Measured
     image-derived DZ in blue. If available, the PCS real-time DZ as a thin grey line and its
     target as dashed grey. Prediction under the measured gas in orange (median over seeds,
     10–90% band). Each counterfactual prediction in the colour of its gas trace. Light band
     at DZ ≈ 0.4–0.6 labelled "T_e cliff" (Chen 2025; Eldon 2017 bifurcation). Vertical tick
     marks for the rollout start and for each gas change (+200 ms penetration lag).
  3. Optionally a thin row: core or pedestal n_e from TS (blue/orange/third colours) to show the
     front moves while the core stays nearly fixed (the Kolemen 2015 / Eldon 2017 argument).
- Why it works: the same quantity the DIII-D controller regulated in these shots, and the same
  format as Ravensbergen 2021 and Kool 2025. A reader sees causality: more gas, front rises
  after a delay; less gas, it falls back (reattachment).
- Optional: 3 small camera thumbnails (measured | predicted) above the panel at key times,
  with the leg polyline and front marker drawn on them.

### (B) Front kymograph: emission along the outer leg vs time
- Panels side by side (measured | predicted median | one counterfactual), x = time, y =
  normalised leg coordinate s (0 target to 1 X-point), colour = min-max-normalised emission along
  the leg. Overlay the DZ (or 50% front) curve as a line. Sequential colormap (e.g. `magma` /
  `cividis`); no truth/prediction colours inside the images, use the coloured frame/label strip.
- Why it works: shows the whole profile, not just one number. The front lift-off and the
  retreat of emission from the target are visible, and so are failures (smearing, a missing
  front). Comparable to the TCV/MAST-U front-vs-time plots and the DIII-D 2-D DTS projections.
- Pitfall: the ensemble median of kymographs blurs fronts. Show the medoid seed (as for the
  spectrograms) and say so in the caption.

### (C) Detachment phase portrait: DZ vs density
- x = line-averaged or pedestal-top n_e (TS; model predicts it), or the time-integrated gas
  input. y = DZ. One curve per trajectory (measured blue, predicted orange, counterfactuals in
  their third colours), with arrows for time and dots every 0.5 s.
- Why it works: the standard "front vs n̄_e" format (Theiler 2017; Wijkamp 2023; DLS model,
  Lipschultz 2016). It shows hysteresis or the bifurcation, a jump in DZ at almost constant
  density (Eldon 2017: same pedestal n_e attached vs detached). If the model reproduces a jump
  rather than a smooth curve, that is a physics claim.
- Pitfall: TS pedestal n_e is noisy and ELM-sensitive. Average over ±2 frames. With only a few
  trajectories the portrait may look sparse; best as a small inset or a 1/3-width panel next to (A).

Suggested figure: (A) as the main row, with (B) for 201081 (measured | predicted) and (C) as a
third column or inset. 201085 is better left out of the DZ panel, or treated with the radial
correction, because its strike-point sweep moves DZ without any change in detachment
(Chen 2025).

## 4. Processing recipe for our standardised frames

Principle: the codec frame is x_z = (x − μ)/σ per frame, an affine map of the intensity. Every
metric invariant under x → a·x + b (a > 0) is unaffected. This includes argmax position,
positions where the profile crosses a fraction of its range ((I − min)/(max − min)), and
centroids of a min-max-normalised profile. Ratios and sums of raw intensity are not invariant.

1. **Channel and filter.** `tangtv_lower` = TANGTV ch0 (LODIV_240RM1 PAR intensified) + ch2
   (LODIV_240RM1 PERP standard). Check per shot which one carries the C-III 465 nm filter in
   201081/201085 (MDSplus TANGTV filter metadata, or compare with Chen et al. Figs). D-alpha
   behaves differently in deep detachment (recombination brightens the target), so mixing
   filters would invert the interpretation.
2. **Three versions of each measured frame:** raw (from tigerdata `<shot>.h5`, z-scored the
   same way), codec reconstruction of the measured codes, and the model prediction. Raw vs
   codec gives the codec ceiling, so blur from the codec is not blamed on the dynamics model.
3. **Geometry in pixel space.** Constant for 201081. Get the X-point and outer strike point
   pixels by projecting EFIT (`rxpt1/zxpt1`, outer strike point) through the TangTV calibration
   (Scotti's tomography geometry), or by hand on the time-averaged frame. Trace the outer-leg
   polyline from target to X-point; parametrise s ∈ [0,1].
4. **Leg profile.** For each frame, sample a band of ±w px (w ≈ 3–5 at 120x360) normal to the
   polyline, take the band mean (or max) to get I(s), n_s ≈ 40 points. Smooth lightly in s.
5. **Normalise:** Ĩ(s) = (I − min_s I)/(max_s I − min_s I). This is affine invariant.
6. **Front metrics (report one, keep the others as checks):**
   - s_50: the largest s on the target side where Ĩ first falls below 0.5, scanning from the
     target (TCV 50% convention). Front = s_50.
   - DZ_img: the Chen-style weighted elevation. Z_E = sqrt(Σ_i (i·r_i)² / Σ_i r_i²) with row
     sums r_i of the clipped image max(x_z − q20(x_z), 0) over columns outboard of the X-point,
     then DZ = 1 − (Z_X − Z_E)/(Z_X − Z_S). The shift by the 20th percentile replaces their
     background subtraction. Check sensitivity to the percentile (10–30).
   - Centroid of Ĩ(s) (smooth, robust, but biased toward 0.5 when the profile is flat).
7. **Calibrate against the measured reference.** Compare the image metric of the measured raw
   frames against (i) the PCS DZ if recorded, and (ii) DTS T_e at the target (cliff at
   ~2.65 s and ~4.45 s in 201081). Fit nothing more than an affine map if needed. Report the
   correlation in the methods.
8. **Time and ensemble.** 50 ms frames (the camera is 30 Hz). ELMs: take the median over 2–3
   frames, or mask frames with a high filterscope ELM envelope (Eldon 2017 removes ELM
   slices). Predictions: compute the metric per seed, then median and 10–90% band. Do not
   compute it on expected (mean) frames, which smear the front.
9. **Flag "no front" frames.** Low-contrast frames (max of z-scored leg profile < ~2–3, or
   max − min small) are almost pure noise after standardisation. Their front is undefined;
   draw a gap, not a value.

## 5. Pitfalls

- **Line integration:** the tangential view integrates about 90° toroidally (Fenstermacher
  1997). The image row is not the poloidal front height. Chen and Boyer handled this with
  regression from inverted labels. For a paper figure, call it an "image-plane front height",
  or reuse their DZ definition with the PCS DZ as validation.
- **Reflections:** the carbon tiles reflect less than metal walls, but bright strike-point
  reflections and the inner leg can still fall to the right of the X-point column. Crop to
  the outer-leg ROI.
- **Standardisation:** absolute brightness is lost, so "the target goes dark" (the
  detachment-cliff signature, Victor & Scotti 2024) looks like "the X-point gets brighter".
  Only relative position is meaningful. State this in the caption. The X-point / target
  ratio needs Ĩ, not x_z.
- **Strike-point or X-point motion** (201085 sweep, 1.52 to 1.42 m) changes DZ_img without
  any change in detachment. Use the leg-following coordinate, or Chen's radial correction, or
  leave 201085 out of the DZ panel.
- **Codec resolution:** 120x360 and the patch decoder blur the leg. Check that the front of
  the codec reconstruction tracks raw (step 2) before claiming model skill.
- **Counterfactual gas** must be physically possible (valve range, no negative flow, ramps
  rather than steps). Keep the measured NBI. Mark the ~200 ms gas penetration lag.
- **Language:** say "emission front" or "C-III front" and "detachment level DZ (0 = attached at
  target, 1 = front at X-point)". Avoid "degree of detachment" (Loarte's DoD needs target
  fluxes) and T_e-at-target claims, since we predict neither.
