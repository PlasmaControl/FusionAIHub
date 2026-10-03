# Detachment

## Description
Divertor detachment is the state in which the plasma at the divertor target has cooled to a few eV (T_e,target <~ 5 eV) so that volumetric losses, including radiation, charge exchange, recombination, dissipate most of the parallel heat and particle flux before it reaches the plate. The signature is a roll-over: as upstream density rises, the target ion saturation current and heat flux first grow and then FALL, the pressure along the field line is no longer conserved (p_target << p_upstream), and the radiation front moves from the target toward the X-point. Partial (outer strike point only) and full detachment are distinguished; a MARFE is the extreme case where the front moves onto the confined plasma edge.

First studied systematically in the 1990s (JET Mark I, DIII-D, ASDEX Upgrade) as
the route to tolerable divertor heat loads.

Typically found via divertor Langmuir probes (j_sat roll-over), divertor Thomson
scattering (T_e), bolometry (radiation front) and, visible divertor cameras.

## Data Provenance
### Dataset 1

**Dataset File(s)**: `raw/emission_structure_*.sav`, `raw/emission_structure_*_raw.sav` (41 shots; packed by `scripts/labeler/detach_inversions.py`)

**Author**: Nathaniel Chen (plasma_tv)

**Description**: TangTV tomographic inversions: the C-III (465 nm) emissivity on the lower-divertor plane per camera frame, from which the front height ZE and `DZ = 1 - (ZX - ZE)/(ZX - ZS)` are read (Chen 2026). They exist for 28 lower-shelf shots (2018-2019 and 2021) and 13 floor-strike shots (195952 to 195963, 206879 to 206894; outside the indicator's gate). The `_raw.sav` files hold the camera video (`VID`), which equals the corpus `tangtv` channel 2.

**Publications**: Chen et al., Nucl. Fusion 66, 036014 (2026).

### Dataset 2

**Dataset File(s)**: none; read from the corpus and fetched with `scripts/labeler/detach_fetch.py` into `$LABELER_ROOT/round4/detach/cache`

**Author**:

**Description**: The corpus `langmuir`, `filterscopes`, `pinj`, `tangtv` and `bolo` groups, and from DIII-D (login node, `fdp run`): `\BOLOM::PRAD_*`, the EFIT01 `aeqdsk` scalars, the CO2 line density, `ECHPWR`, `\NB::PINJ`, and the divertor Thomson Te points `TSSDIVTE00-05` (the independent check, read by no indicator). The EFIT01 flux maps for the appendix figure are fetched by `detach_fetch_efit.py`.

**Publications**:

### Dataset 3

**Dataset File(s)**:

**Author**: Cheolsik Byun (contact)

**Description**: Cheolsik Byun has worked on detachment algorithms; no table has arrived, so there is no hand-made reference label. The `detach_vote` label is unverified until `detach-ui` reviews it.

**Publications**:

## Models
**stable**: none

**latest**: detach_vote | 2026_10_03 (unverified)

**all**:
- detach_vote | 2026_10_03 (label model over the Afrac, Prad,div and TangTV votes, 50 ms bins, 259 shots; `extend_detach_vote/`; 28 % attached, 9 % detached, 5 % MARFE, 58 % uncertain)
- detach_rule | 2026_10_03 (the fallback: the valid votes agree, else uncertain; 45 / 12 / 1 / 43 %; `$LABELER_ROOT/round4/detach/labels_rule.csv`)
- detach-ours | 2026_10_03 | accuracy: 0.75 | kappa: 0.50 | macro F1: 0.64 (1-D CNN on 0-D signals, scored against `detach_vote` by shot; majority accuracy 0.66; 26 test shots: accuracy 0.84, kappa 0.65)
- detach-victor | 2026_10_03 | accuracy: 0.73 | kappa: 0.56 | macro F1: 0.68 (CNN on one raw TangTV frame after Victor and Scotti 2024, three states, scored against `detach_vote` by shot; majority accuracy 0.55; below the majority predictor where TangTV did not vote)

Scores are against the `detach_vote` label, not against truth. Single indicators against the label with their own vote withheld: Afrac kappa 0.20, Prad,div 0.16, TangTV 0.35 (`docs/labeler/results/detachment_benchmark.json`).

## Inputs
**detach_vote**, **detach_rule**:
- `langmuir` (raw swept probes; Jsat decoded, Afrac)
- `filterscopes` (divertor D-alpha, the ELM mask)
- `pinj`, `ECHPWR`, `\BOLOM::PRAD_DIVL` (Prad,div over the heating power)
- `tangtv` channel 2 (raw lower-divertor frames; the front height is regressed on them where there is no inversion) and the 41 inversions
- EFIT01 `aeqdsk` strike points and X-point (the lower-shelf gate)
- `Ip`, CO2 line density (bin start, Afrac normalisation)

**detach-ours**:
- `Ip`, heating power, line density, divertor D-alpha, ELM share, EFIT scalars (`betan`, `wmhd`, `q95`, `kappa`, `bcentr`, X-point and strike-point positions)

**detach-victor**:
- `tangtv` channel 2 (30 x 90 block means)

## Method
Divertor detachment is a state of the lower divertor, so the label is a redundant,
non-binary one: attached (1), detached (2), MARFE (3) or uncertain (4), per 50 ms bin
(0 = not assessed), the coding `detach-ui` uses. Three indicators vote:

- **Afrac** (Eldon 2021, 2022): the divertor Jsat over the value an attached divertor
  would carry at the same density and power, `1 / DOD`. Attached at or above 0.75,
  detached at or below 0.5. Uncalibrated, decoded from raw probes, self-referenced.
- **Prad,div** (Eldon 2019): the lower-divertor radiated power over the heating power.
  Attached at or below 0.35, detached at or above 0.5. A radiation measure; never votes
  MARFE.
- **TangTV front height** (Chen 2026): `DZ`, attached below 0.35, detached 0.5 to 1.0,
  MARFE above 1.0. Valid only on the lower shelf (EFIT gate); never emits a state on the
  floor-strike shots or any other geometry.

Each indicator also reports a validity mask with the reason. The votes are combined by a
Snorkel-style label model (Ratner et al. 2017: accuracies from agreement, exact
enumeration, accuracies learned where all three are valid, class balance attached 1/2,
detached 1/4, MARFE 1/4, posterior at least 0.7) and by a transparent rule (agreeing
votes, else uncertain). A bin is labelled where at least two indicators are valid. Result:
the three agree little (Afrac with Prad,div kappa 0.01), the model trusts TangTV (implied
accuracy 0.96) and leaves most bins where only Afrac or Prad,div speak uncertain.
Details, thresholds with sources and the failure analysis: `docs/labeler/detachment.md`.

## Alias
detachment, detach, detached, divertor detachment

## Future Implementations
- Expert review in `detach-ui` is the next step; until then the label is unverified.
- A floor-strike model: TangTV is invalid on the floor shots (195952 to 195963, 206879 to 206894) and there is no replacement indicator there.
- IRTV heat flux (the corpus group is a stub) and the bolometer geometry (not in the corpus or the `\BOLOM` tree) would add a fourth voter and a bolometer image.
- A calibrated Afrac needs the probe positions and a Jsat calibration, which the corpus lacks.

## Reference
- S. I. Krasheninnikov and A. S. Kukushkin, "Physics of ultimate detachment of a
  tokamak divertor plasma", J. Plasma Phys. 83, 155830501 (2017).
- A. Loarte et al., "Plasma detachment in JET Mark I divertor experiments",
  Nucl. Fusion 38, 331 (1998).
- A. Ratner et al., "Snorkel: rapid training data creation with weak supervision",
  PVLDB 11(3), 2017.
- N. Chen et al., "Regulation compliant AI for fusion: explainable image-based feedback
  control of divertor detachment in DIII-D tokamak", Nucl. Fusion 66, 036014 (2026).
- D. Eldon et al., Nucl. Mater. Energy 18, 285 (2019); 27, 100963 (2021); Plasma Phys.
  Control. Fusion 64, 075002 (2022).
- B. S. Victor and F. Scotti, "Identifying divertor detachment using a machine learning
  model trained on divertor camera images from DIII-D", Rev. Sci. Instrum. 95, 083503
  (2024).

## Contact
- **Cheolsik Byun**: csbyun [at] princeton [dot] edu
- **Nathaniel Chen**: nathaniel [at] princeton [dot] edu
