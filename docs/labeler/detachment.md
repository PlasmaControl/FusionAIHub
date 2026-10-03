# Detachment label

Status: **built 2026-10-03 on branch `r4-detach`** (round four, stream `detach`).
The label is a redundant, non-binary state of the lower divertor, voted by three
independent indicators and combined by a Snorkel-style label model with a
transparent fallback rule. No expert has reviewed it yet: the review page
(`detach-ui`) reads the indicator traces this stream writes
(`$LABELER_ROOT/round4/detach/indicators/<shot>.csv`) and is where the labels
get verified.

## Result in brief

**What exists.** A state for every 50 ms bin of 259 DIII-D discharges, the ones where at
least two of the three indicators were valid: 24,786 assessed bins, 2,413 intervals in
`data/events/detachment/extend_detach_vote/detach_shots.csv`. The label model is the primary
label; the transparent rule is reported beside it.

50032 bins of 470 shots with bins; 259 eligible shots (train 182, val 23, test 26, outside 28).

| indicator | valid bins | shots | attached | detached | marfe |
| --- | --- | --- | --- | --- | --- |
| Afrac | 22305 | 219 | 6905 | 10291 | 0 |
| Prad,div | 44059 | 461 | 31540 | 3760 | 0 |
| TangTV | 5942 | 101 | 2192 | 1848 | 1271 |

**What the indicators say.** They agree far less than three views of one state should:

| pair | both valid | both vote | agreement | kappa |
| --- | --- | --- | --- | --- |
| Afrac / Prad,div | 21823 | 13902 | 0.41 | 0.01 |
| Afrac / TangTV | 2648 | 1565 | 0.51 | 0.20 |
| Prad,div / TangTV | 5607 | 2405 | 0.41 | 0.09 |

Afrac and Prad,div agree no better than chance (kappa 0.01 on 13,902 bins where both
vote; -0.09 on the TangTV shots alone). TangTV is the only indicator that relates to
either of them, weakly (kappa 0.20 and 0.09), and the one the independent
divertor-Thomson check (below) supports best. The label model, which learns accuracy
from agreement, therefore gives TangTV an implied accuracy of 0.96 (the cap of the fit)
and the other two 0.66 and 0.67. Where TangTV votes it decides; where only Afrac and
Prad,div vote, a lone vote (posterior 0.66 to 0.67) falls below the 0.7 threshold, and the
bin is `uncertain` unless the two agree.

**The label.** Of the assessed bins the label model calls 28 % attached, 9 % detached, 5 %
MARFE and 58 % `uncertain`; the rule calls 45 %, 12 %, 1 % and 43 %. The two agree on
68 % of bins (kappa 0.48), and where both are certain they disagree on 33 bins of 8,316
(the model does not contradict the rule; it labels fewer bins). Lowering the posterior
threshold to 0.6 would label the lone-vote bins and cut `uncertain` to 35 % (the
threshold table below).

**Independent evidence is weak.** Divertor-Thomson Te, which no indicator reads, orders
the label states as expected (median of the per-bin peak: attached 25 eV, detached 15 eV,
MARFE 11 eV) and separates detached from attached bins with AUROC 0.60 [0.52, 0.68]:
better than chance, far from a confirmation. The label is **unverified**: nobody has
reviewed it, and `detach-ours` and `detach-victor` below are scored against it, not
against truth.

**What it cannot say.** TangTV is invalid on the floor-strike shots (195952 to 195963,
206879 to 206894) and on every shot with neither an inversion nor a corpus camera frame;
there is no IRTV row (the corpus stub is empty); Afrac exists only on the 2018-2019 shots
with a probe record in the corpus (219 shots); shot 201081, the worked example of
Chen 2026, has no beam power and gets no label.

## The label

One state per 50 ms bin of a discharge, coded as `detach-ui` codes it:

| code | state | meaning |
| --- | --- | --- |
| 0 | absent | fewer than two indicators were valid: nothing can be said |
| 1 | attached | the strike point carries the full heat and particle flux |
| 2 | detached | the radiating front has left the plate (partial or full) |
| 3 | marfe | the front has moved above the X-point onto the confined plasma |
| 4 | uncertain | indicators disagree, or all valid ones sit in a transition band |

`marfe` is the extreme end of detachment: a bin that is MARFE is also detached in
the physical sense, but the label keeps the stage apart because only the camera
can see it. A bin is never labelled where fewer than two indicators were valid.
Unknown is a missing value, never an attached or a detached.

The label table is
`data/events/detachment/extend_detach_vote/detach_shots.csv` (intervals of
consecutive bins in one state; `.meta.json` beside it records the sources, the git
sha and the record files). The per-bin grids are `detach_shots/<shot>.npz` (git
ignored, regenerable with `detach_label.py`). The full per-bin outputs, with the
three values, votes, posteriors and both labelers' states, are under
`$LABELER_ROOT/round4/detach/` (`labels_bins.csv.gz`, `labels_rule.csv`).

## The indicators

Every threshold is in `src/labeler/events/detachment/thresholds.py`, with its
source; none was fitted to a score. Each indicator votes attached, detached or
(TangTV only) marfe, abstains inside a transition band, and is *invalid*, which is
not the same as abstaining, where its data or geometry cannot support a vote.

**Afrac** (Eldon et al. 2022, 2021). `Afrac = Jsat / (C <ne>^2 q_par^(-3/7))`, the
divertor ion saturation current over the value a two-point model gives for an
attached divertor at the same density and power; it equals `1 / DOD`. Attached at or
above 0.75, detached at or below 0.5 (the reference-shot windows DOD 1, 2 and 4 of
Eldon 2021 read Afrac 1, 0.5 and 0.25). The corpus has the raw swept probes only:
Jsat is decoded from the swept current/voltage pairs (`langmuir.py`), has no
calibration and no probe positions, so the indicator takes the *peak over the live
probes* of the inter-ELM median ion current and references it to the shot's own
attached level (its 0.90 quantile). Limits: a shot that is detached throughout is
mis-called attached in its top tail; density dips spike it (`1/ne^2`); between two
probes the peak under-reads; it never votes marfe. In-ELM samples (a D-alpha
detector) are dropped, and a bin more than 80 % in ELMs is invalid.

**Prad,div** (Eldon et al. 2019; Chen 2026). The lower-divertor radiated power
`\BOLOM::PRAD_DIVL` (calibrated, post-shot) over the heating power (beams, ohmic,
ECH): `f_div`. Attached at or below 0.35, detached at or above 0.50, the two end
states of the worked example in Chen 2026 (shot 201081: 1.6 to 2.2 MW of 4.4 MW)
rounded outward. It is a radiation measure rather than a detachment measure (it rises
before the temperature cliff and stays up after reattachment), so it is the weakest
voter and abstains in a wide band; it never votes marfe.

**TangTV front height** (Chen 2026). `DZ = 1 - (ZX - ZE) / (ZX - ZS)`, with ZX the
X-point height, ZS the outer strike-point height and ZE the weighted height of the C-III
(465 nm) emission at and outboard of the X-point radius, from the tomographic
inversion of the lower-divertor tangential camera. Attached below 0.35, detached from
0.5 to 1.0 (the temperature cliff is at 0.5), marfe above 1.0; DZ below -0.25 is
unphysical and invalid. **It is valid only on the geometry it was built for:** outer
strike point on the lower shelf (Z = -1.25 m, R above 1.37 m), lower single null.
On the floor (Z = -1.363 m, R below 1.37 m: shots 195952 to 195963, 206879 to 206894)
or any other geometry the indicator is invalid with reason `strike_on_floor`,
`not_lower_null` or `strike_not_on_shelf`, whatever the frames show, and *no state
is ever emitted from it there* (`tangtv.shelf_gate`, tested).

Where a shot has no inversion on disk, ZE is regressed from the raw camera frame
(see "TangTV from raw frames"); the bin records `tangtv_source`.

## Validity masks and what is common

* **Bins** are 50 ms starting at the first sample with plasma current at least
  0.3 MA; indicators are aggregated per bin by median (Afrac, DZ) or mean (power).
* **ELMs**: a 10 kHz divertor D-alpha sample is in an ELM when it exceeds a 50 ms
  running-median baseline by 4 robust sigmas and by 50 % of the baseline (a quiet
  detached plasma has none). Samples in ELMs are dropped from Afrac and Prad,div;
  bins more than 80 % in ELMs are invalid for both.
* **Ramps**: Afrac is invalid where `|dIp/dt|` exceeds 1 MA/s (Eldon 2022).
* **Power**: Prad,div is invalid below 0.5 MW input power.
* **Geometry**: TangTV only inside the shelf gate, from EFIT01 strike points and
  X-point; EFIT sentinel values (-0.89, -9.99, 0) are read as missing.

## The label model and the rule

**Label model.** Data programming (Ratner et al. 2017): each indicator is a labelling
function; a log-linear generative model with the true state as a latent variable
learns each indicator's accuracy from how often they agree, with no labelled data
(`label_model.py`). Three states and at most four votes per function make a few dozen
vote configurations, so the likelihood and its gradient are computed exactly by
enumeration. Three things differ from a stock Snorkel setup, each for a reason
(and each tested): (1) the likelihood is *conditional on which indicators are valid*
on each bin, so a pattern in which TangTV is mostly absent does not teach the model
that it abstains by choice; (2) every function has its own allowed vote set (Afrac
and Prad never vote marfe), and a "detached" vote from either is counted accurate
when the state is detached *or* marfe; (3) accuracy weights are capped at 4 (an
implied accuracy of 0.96 to 0.98), because an indicator that never disagrees on a
finite sample would otherwise run to infinity. Where TangTV did not vote, the
marfe mass is pooled into detached (nothing can resolve it). The structure (no
correlation, or one or all pairs of correlated indicators) is chosen by
shot-grouped 5-fold cross-validated log-likelihood with the one-standard-error
rule. A bin is labelled with the most probable state when its posterior is at least
0.7, else `uncertain`; a one-bin run between two bins of one state is merged into
them. The model is fitted on shots outside the cohort's test split only.

**Rule.** The fallback anyone can apply by hand: collect the non-abstaining votes of
the valid indicators; if they all agree that is the state; if they disagree, or all
sit in a transition band, `uncertain`; if none was valid, no label. Both labels are
reported with the agreement matrix between them.

### What the label model learned

Fit: 233 shots, 22222 bins (test split excluded); accuracies learned on 2557 bins of 40 shots where all three indicators are valid.

| indicator | implied accuracy (anchored fit, used) | weight | accuracy if fitted on all bins |
| --- | --- | --- | --- |
| Afrac | 0.66 | 0.65 | 0.50 |
| Prad,div | 0.67 | 0.71 | 0.98 |
| TangTV | 0.96 | 4.00 | 0.51 |

Fitting on all bins instead gives the third column: it puts the whole disagreement of the
two-voter bins on Afrac (0.50) and calls Prad,div near-perfect (0.98), though the two-voter
bins fix only the *product* of the two accuracies and the state is nearly constant there.
That is why `fit_anchored` exists (point 4). The comparison is recorded in
`records/label_model.json`.

A fourth departure from a stock Snorkel setup, for the same reason: (4) the accuracy and
correlation weights are learned on the bins where all three indicators are valid
(2,557 bins of 40 shots), where three voters identify them; the class balance is then
fixed at attached 1/2, detached 1/4, MARFE 1/4 (nothing in the data identifies it across
shots of different regimes; a uniform prior over the three states would count the pooled
"detached or MARFE" of the bins where only Afrac and Prad,div speak twice as likely as
attached before any vote, so a lone detached vote would label a bin and a lone attached
vote would not), and only the propensities are fitted on all bins. The prior was set
after the first full run showed that asymmetry (lone detached votes labelled bins, lone
attached votes did not), with no score in view. The accuracy is the price of this: it
rests on 40 shots, those of the owner's inversions, and cannot be checked against truth.

**Structure.** The independent structure (no correlation) was chosen by the one-standard-error
rule; the held-out log-likelihood does not separate the candidates:

| structure | held-out log-lik / bin |
| --- | --- |
| independent | -2.347 |
| prad_tangtv | -2.348 |
| afrac_tangtv | -2.344 |
| afrac_prad | -2.349 |
| all_pairs | -2.351 |

**States and the rule.**

| labeler | attached | detached | marfe | uncertain |
| --- | --- | --- | --- | --- |
| label_model | 6832 | 2285 | 1172 | 14497 |
| rule | 11070 | 2897 | 274 | 10545 |

Rule (rows) against label model (columns), 24786 bins: agreement 0.68, kappa 0.48.

| rule \ label model | attached | detached | marfe | uncertain |
| --- | --- | --- | --- | --- |
| attached | 6529 | 7 | 2 | 4532 |
| detached | 11 | 1484 | 9 | 1393 |
| marfe | 0 | 4 | 270 | 0 |
| uncertain | 292 | 790 | 891 | 8572 |

Where the rule says `attached` and the model `uncertain` (4,532 bins), it is a lone
Afrac or Prad,div vote (posterior 0.66 or 0.67; two agreeing votes reach 0.8); where the
rule says `detached` and the model `uncertain` (1,393), the same for a lone detached vote. Where both
labels are certain they differ on 33 of 8,316 bins. Where the rule is `uncertain`
(conflicting votes) the model resolves 1,973 bins; 98 % of them carry a TangTV vote and the
model follows it in 98 % of those.

**Posterior threshold.** Share of the 24,786 assessed bins that stay `uncertain` as the
threshold moves (the label uses 0.7, chosen before any score was seen):

| threshold | attached | detached | MARFE | uncertain |
| --- | --- | --- | --- | --- |
| 0.5 | 18,837 | 3,974 | 1,192 | 783 |
| 0.6 | 11,260 | 3,733 | 1,192 | 8,601 |
| 0.7 | 6,848 | 2,327 | 1,192 | 14,419 |
| 0.8 | 2,050 | 1,756 | 1,192 | 19,788 |
| 0.9 | 2,050 | 1,723 | 1,184 | 19,829 |

(The 0.7 row is before the one-bin merge, which gives the label counts above.) The step
between 0.6 and 0.7 is the lone Afrac or Prad,div vote (posterior 0.66 to 0.67): at 0.6 a
single weak indicator labels a bin, at 0.7 it does not. The choice of 0.7 is the
conservative one: a label that needs two agreeing indicators, or the camera.

## TangTV from raw frames

The owner's inversion exists for 41 shots; the corpus holds raw TangTV frames for
most shots from 190000 to 204999 (channel 2, the lower-divertor perpendicular view; the
frames are the `VID` of the plasma_tv `.sav` files, correlation 1.00 on the 15 shots
checked, resampled to 50 fps). The inversion is a linear operator on the frame, so the
quantity the indicator needs, the height ZE of the C-III front, can be regressed on the
frame (Chen 2026 trains its weighted-emission models the same way). This is an
extension of the brief, built so that more than a few dozen shots can carry a TangTV
vote at all; it is validated here, not assumed.

*Model.* ZE (from the inversion, `outer_leg_ze`) on the black-level-subtracted,
square-rooted, 6 x 6 block-averaged frame (40 x 120 features) and the X-point radius,
by ridge regression in the dual form (`detach_tv_surrogate.py`, numpy only; no
framework). It is trained only on the shelf-geometry frames of the inversion shots
(28 shots, 4050 frames; the floor shots are outside the gate and are not used) and
applied only inside the same geometry gate: on any other geometry the indicator stays
invalid, whatever the regression says. The regularisation (alpha = 100, from
1, 10, 100, 1000) was the best of four by ZE error in the CV below.

*Validation.* Leave-one-shot-out, the held-out shot never in the fit, with 1000-replicate
shot-bootstrap intervals (`docs/labeler/results/detachment_tangtv_surrogate.json`).
On the `.sav` frames the model was trained on and on the corpus frames at the inverted
times (the path the deployed model takes):

| | frames | ZE error (cm) | agreement of the TangTV vote | kappa | attached / detached / marfe recall |
| --- | --- | --- | --- | --- | --- |
| `.sav` frames | 4050 (28 shots) | 1.37 [1.02, 1.79] | 0.92 [0.86, 0.97] | 0.88 [0.78, 0.95] | 0.97 / 0.89 / 0.91 |
| corpus frames | 1969 (15 shots) | 1.83 [0.81, 3.10] | 0.86 [0.70, 0.99] | 0.79 [0.55, 0.98] | 0.75 / 0.86 / 0.96 |

(The vote is the indicator's own, from DZ with the thresholds above, on the frames where
both the inversion and the regression vote.) The regression is therefore good enough
to vote with, with the corpus path the weaker; its bins are marked `tangtv_source =
surrogate` in the per-bin files, and the benchmark scores the indicator with and without
them (`detachment_benchmark.json`). The training shots were selected by what the owner
has already inverted, which is the early-detachment campaigns of 2018-2019 and 2021:
it is not known how well the regression generalises to a campaign with a different
camera exposure or filter, and the CV cannot say.

## Bin width

**50 ms.** The brief allows 10 to 50 ms. Inside the range the label is insensitive to the
width and the noise floor moves with it, so the coarsest allowed bin is the right one. The
three grids were built for the 81 shots that carry a TangTV vote, with the one fitted model
(nothing refitted per width):

| bin | bins | assessed | uncertain / assessed | flicker / s | agreement with 50 ms | kappa with 50 ms |
| --- | --- | --- | --- | --- | --- | --- |
| 20ms | 21637 | 0.76 | 0.22 | 1.23 | 0.974 | 0.959 |
| 50ms | 8694 | 0.78 | 0.22 | 0.66 | - | - |
| 100ms | 4364 | 0.78 | 0.23 | 0.44 | 0.967 | 0.949 |

* Labels at 20 ms and 100 ms agree with the 50 ms labels at kappa 0.96 and 0.95 where both
  are certain: the label is not an artefact of the grid.
* The share of assessed bins that are uncertain does not change (0.22 at every width): the
  uncertainty is the indicators' disagreement, not time resolution.
* State flicker (transitions per second between certain states) falls from 1.23 per second
  at 20 ms to 0.66 at 50 ms and 0.44 at 100 ms: a finer grid mostly resolves noise
  flickers of the indicators, not new events.
* The camera frames are 20 ms apart, so a 50 ms bin holds two or three frames (a 20 ms
  bin one); the ELM mask uses a 50 ms running median.

The sensitivity is `docs/labeler/results/detachment_bin_sensitivity.json`
(`detach_sensitivity.py`).

## Benchmark

Each indicator is scored on its own, with the thresholds published for it (Afrac per
Eldon 2021/2022, Prad,div per Eldon 2019, the TangTV front height per Chen 2026), against
the combined label, on the bins where it is valid and the reference is a certain state;
bins where the indicator abstains are not scored (`vote_rate` in the JSON). Two references
(`docs/labeler/results/detachment_benchmark.json`, `detach_benchmark.py`; 1,000-replicate
shot-bootstrap 95 % intervals in brackets):

* **combined**: the label model's state, which includes the indicator scored. Circular;
  an upper bound, and meaningless for the indicator the model trusts most.
* **leave-one-out**: the same fitted model with that indicator's vote withheld; certain only
  where the other two are both valid. This is the honest agreement, and it is small:
  the indicators are not three noisy copies of one truth.

Single indicator against the label with its own vote withheld, on all 259 shots (the
model was fitted on the 233 that are not in the cohort's test split):

| indicator | bins | agreement | kappa | F1 attached | F1 detached | F1 marfe | AUROC att / not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Afrac | 1564 | 0.51 [0.40, 0.62] | 0.20 [0.06, 0.33] | 0.61 [0.44, 0.73] | 0.53 [0.37, 0.65] | n/a | 0.63 [0.55, 0.71] |
| Prad,div | 1077 | 0.52 [0.38, 0.65] | 0.16 [-0.09, 0.37] | 0.51 [0.22, 0.70] | 0.61 [0.44, 0.74] | n/a | 0.62 [0.51, 0.74] |
| TangTV | 363 | 0.64 [0.48, 0.76] | 0.35 [-0.00, 0.55] | 0.66 [0.21, 0.83] | 0.72 [0.52, 0.84] | 0.00 [0.00, 0.00] | 0.81 [0.66, 0.89] |

Test shots (the model was not fitted on them; they carry 4 to 80 such bins, so these rows
are anecdotes, not scores):

| indicator | bins | agreement | kappa | F1 attached | F1 detached | F1 marfe | AUROC att / not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Afrac | 80 | 0.07 [0.07, 0.07] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.14 [0.14, 0.14] | n/a | - |
| Prad,div | 19 | 0.58 [0.58, 0.58] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.73 [0.73, 0.73] | n/a | - |
| TangTV | 4 | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | 0.00 [0.00, 0.00] | - | - |

Against the combined label (circular), all shots:

| indicator | bins | agreement | kappa | F1 attached | F1 detached | F1 marfe | AUROC att / not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Afrac | 6604 | 0.88 [0.84, 0.92] | 0.64 [0.55, 0.72] | 0.94 [0.92, 0.96] | 0.72 [0.64, 0.80] | n/a | 0.84 [0.80, 0.89] |
| Prad,div | 7608 | 0.81 [0.76, 0.86] | 0.53 [0.43, 0.62] | 0.92 [0.89, 0.95] | 0.62 [0.51, 0.71] | n/a | 0.87 [0.83, 0.90] |
| TangTV | 4951 | 0.99 [0.98, 0.99] | 0.98 [0.97, 0.99] | 1.00 [1.00, 1.00] | 0.98 [0.97, 0.99] | 0.97 [0.96, 0.98] | 1.00 [1.00, 1.00] |

The same, test shots:

| indicator | bins | agreement | kappa | F1 attached | F1 detached | F1 marfe | AUROC att / not |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Afrac | 644 | 0.88 [0.67, 1.00] | 0.56 [0.13, 1.00] | 0.93 [0.78, 1.00] | 0.62 [0.17, 1.00] | n/a | 0.75 [0.56, 1.00] |
| Prad,div | 728 | 0.90 [0.77, 0.99] | 0.74 [0.48, 0.96] | 0.97 [0.93, 1.00] | 0.78 [0.36, 0.99] | n/a | 0.96 [0.89, 1.00] |
| TangTV | 288 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] |

How to read it. Afrac and Prad,div have F1 for marfe n/a because they never vote it.
TangTV against `combined` is 0.99 only because it decides every bin where it votes (its
implied accuracy is 0.96, the cap of the fit); that is not a performance claim. Against
what the other two say it reaches kappa 0.35 [-0.00, 0.55] on 363 bins, and its
front height from the regression (`surrogate`) against the inversion (`inversion`) is:

| source | reference | bins | agreement | kappa |
| --- | --- | --- | --- | --- |
| inversion | loo | 146 | 0.71 [0.48, 0.90] | 0.10 [-0.01, 0.38] |
| inversion | combined | 1167 | 0.99 [0.98, 1.00] | 0.98 [0.96, 0.99] |
| surrogate | loo | 217 | 0.60 [0.35, 0.75] | 0.29 [-0.20, 0.51] |
| surrogate | combined | 3784 | 0.99 [0.98, 0.99] | 0.98 [0.97, 0.99] |

(the two sources do not differ beyond the intervals; the leave-one-out references are
few, 146 and 217 bins.)

**Independent check.** Divertor Thomson Te (the six real-time points `TSSDIVTE00-05`;
the peak of them per bin; an exact 0 eV is a failed fit and is read as missing) was read
by no indicator:

| label state | bins with Te | Te quartiles | share < 5 eV |
| --- | --- | --- | --- |
| attached | 4830 | 9.7 / 24.9 / 216.5 | 0.20 |
| detached | 1966 | 3.3 / 14.9 / 42.8 | 0.29 |
| marfe | 1043 | 1.9 / 11.0 / 29.4 | 0.40 |
| uncertain | 9509 | 2.9 / 23.3 / 229.3 | 0.30 |

AUROC of -Te for detached against attached bins: 0.60 [0.52, 0.68] (6796 bins, 226 shots).

Each indicator's value against a cold plate (Te < 5 eV; AUROC, no label involved):

| indicator | AUROC | bins | shots |
| --- | --- | --- | --- |
| Afrac | 0.58 [0.55, 0.60] | 14932 | 198 |
| Prad,div | 0.43 [0.37, 0.50] | 17346 | 234 |
| TangTV | 0.66 [0.56, 0.76] | 4861 | 76 |

The ordering of the medians is the expected one (attached 25 eV, detached 15 eV, MARFE
11 eV), but the distributions overlap almost entirely and the real-time points are sparse
and not at the strike point: this is a weak check, not a validation. By indicator, the
camera front is the best predictor of a cold plate (AUROC 0.66 [0.56, 0.76]), Afrac next
(0.58) and Prad,div does no better than chance (0.43 [0.37, 0.50]): the radiated fraction
is high in attached and detached plasmas alike.

### Where each indicator fails

All numbers from `failure_analysis` in the JSON.

* **Prad,div: radiation is not detachment.** On the 2,049 bins where TangTV says attached
  and Prad,div is valid, Prad,div votes detached on 11 % and its median `f_div` is 0.42, inside
  its own abstention band (0.35 to 0.50): the radiated fraction rises before the
  temperature cliff and stays up after it. The attached-versus-not agreement of its vote
  with the label rises with heating power (0.73 below 2 MW, 0.84 at 2 to 5 MW, 0.95 above
  5 MW), so it is least reliable in the low-power shots.
* **Afrac: an uncalibrated, self-referenced detector.** In the 38 shots that the label calls
  detached or MARFE for most of their length, Afrac still casts 28 % of its votes for
  attached (its reference is the shot's own 0.90 quantile, so a shot that is detached
  throughout has no attached level to compare with). Its worst shots against the label
  are 192154 (agreement 0.20), 192231 (0.36), 192152 (0.36), 194959 (0.44) and 192230
  (0.49). Its agreement with the label rises less with heating power than Prad,div's (0.85,
  0.90, 0.93) and does not fall with the share of the bin in ELMs (0.89, 0.91, 0.99 for
  0 to 20 %, 20 to 50 % and above 50 %; the last has 482 bins).
* **TangTV: valid on a minority of shots.** Of the 50,032 bins of the 470 shots with data it
  is invalid on 20,530 for lack of both an inversion and a frame regression output (the
  shot has no usable corpus camera frames), 13,754 on the
  floor strike geometry (`strike_on_floor`: shots 195952 to 195963 and 206879 to 206894,
  where the camera view and the C-III front do not mean DZ), 5,661 for missing EFIT
  geometry, 2,462 for missing frames, 1,103 off the shelf, 337 for EFIT gaps and 243 for
  an unphysical DZ. On the floor shots it never votes, whatever the frames show.
  Where it does vote it is the only indicator that can say MARFE.

### The learned baselines

`detach-ours` (a small 1-D CNN on a 2 s window of plasma current, heating power, line
density, divertor D-alpha, the ELM share and the EFIT scalars; `detach_ours.py`) and
`detach-victor` (the CNN of Victor and Scotti 2024 on one raw TangTV frame, read here
with three outputs and trained on this label, not their hand-labelled binary one;
`detach_victor.py`). Both are trained on the bins where the label is a certain state and
scored by shot (5-fold cross-validation over the non-test shots, then one score on the
test shots), against the majority-class predictor:

| model | scored on | windows or frames / shots | accuracy | kappa | macro F1 | majority accuracy |
| --- | --- | --- | --- | --- | --- | --- |
| detach-ours | CV, by shot | 10289 / 257 | 0.75 [0.70, 0.80] | 0.50 [0.42, 0.58] | 0.64 [0.57, 0.69] | 0.66 |
| detach-ours | test split | - / 26 | 0.84 [0.70, 0.96] | 0.65 [0.42, 0.83] | 0.72 [0.53, 0.87] | 0.69 |
| detach-victor | CV, by shot | 5877 / 107 | 0.73 [0.66, 0.79] | 0.56 [0.46, 0.65] | 0.68 [0.61, 0.75] | 0.55 |
| detach-victor | test split | - / 8 | 0.76 [0.52, 0.94] | 0.63 [0.17, 0.89] | 0.73 [0.46, 0.96] | 0.50 |
| detach-victor | CV, TangTV not voting | 1422 / 70 | 0.79 [0.68, 0.86] | 0.28 [0.13, 0.42] | 0.41 [0.35, 0.49] | 0.90 |

`detach-ours` beats the majority predictor in cross-validation (accuracy 0.75 against 0.66,
kappa 0.50 [0.42, 0.58]) and on the 26 test shots (0.84 against 0.69, kappa 0.65 [0.42,
0.83]) with F1 for detached and MARFE near 0.5; its inputs include the EFIT X-point and
strike-point heights that the TangTV indicator itself uses, and the heating power and
line density that normalise Afrac and Prad,div, so it is not wholly independent of the
indicators. `detach-victor` also beats the majority predictor (0.73 against 0.55, kappa 0.56 [0.46, 0.65];
the test split has 8 shots, kappa 0.63 [0.17, 0.89]), but the frames and the TangTV vote
share information, and on the bins where TangTV did not vote, where the label is Afrac and
Prad,div alone, it is below the majority predictor (accuracy 0.79 against 0.90, F1 for
detached 0.27): the frames do not know what those two indicators say. Neither model is a
claim about detachment; both say how much of this label can be read from cheaper
inputs. The 0.96 F1 reported by Victor and Scotti on their floor geometry comes from a split by
time slice, and is for another label (binary, hand-made), geometry and split, so it is not
comparable with these numbers.

## Appendix figure

`fig_detachment_views.pdf` (vector) and `fig_detachment_views.png` (150 dpi), written by
`scripts/labeler/detach_figure.py` to `$LABELER_ROOT/round4/detach/figure/`. Shot **189057**,
chosen because it has the owner's own inversion (no regression) and passes through
attached, detached and MARFE in the label. Three columns, one per state, at the bin of
that state the label model is most confident of among the bins with a camera frame (2075,
2625 and 3725 ms; had the shot no MARFE bin, the third column would be its deepest
detached bin):

1. the raw TangTV frame nearest the time (corpus channel 2, lower-divertor view);
2. the TangTV inversion nearest the time (the C-III emissivity on the lower-divertor
   plane), with the EFIT01 flux surfaces (the separatrix, psi_N = 1, thick; psi_N = 0.98,
   1.02, 1.05 and 1.1 thin), the X-point (x), the outer strike point (o) and the front
   height ZE (dashed);
3. the **bolometer chord profile** (change of the raw chord voltage over its pre-shot level)
   of the lower (L) and upper (U) arrays;
4. spanning the columns, the timeline: strips for the label and the votes of the three
   indicators, then Afrac, Prad,div and ZE with the strike-point and X-point heights, the
   three chosen times marked in the state colours;
5. spanning the columns, the state fractions of the 259 labelled shots, ordered by state
   mix, with 189057 marked.

Deviations from the brief, each on purpose: **(a)** the bolometer row is a chord profile, not
an image or an inversion, and carries no flux-surface overlay: the corpus holds 48 raw chord
voltages and no chord geometry, and the `\BOLOM` tree holds none either (a probe of its 515
nodes, `$LABELER_ROOT/round4/detach/bolom_nodes.txt`), so a chord cannot be placed in R, Z;
**(b)** there is no IRTV row: the corpus group is a stub on every inspected shot; **(c)**
the raw frame carries no EFIT overlay: there is no camera calibration to project onto it. The
EFIT overlay on the inversion uses the EFIT01 flux map `psirz` of the nearest time
(`detach_fetch_efit.py` fetches it for the figure shots). Afrac peaks off the top of its
panel around 2000 to 2500 ms.

## Inputs and provenance

From the FAITH corpus (`/scratch/gpfs/EKOLEMEN/foundation_model/<shot>_processed.h5`,
read only):

| indicator | corpus group | what is used |
| --- | --- | --- |
| Afrac | `langmuir` | the raw swept-probe current and voltage pairs; Jsat is decoded here |
| Afrac, Prad,div | `filterscopes` | divertor D-alpha at 10 kHz, for the ELM mask |
| Prad,div | `pinj` | neutral-beam power; the corpus group is a stub on some shots, where the fetched `\NB::PINJ` replaces it |
| TangTV | `tangtv` (channel 2) | the raw lower-divertor camera frames, for the regression of the front height |
| figure only | `bolo` | the 48 raw bolometer chord voltages |

Fetched from DIII-D (`scripts/labeler/detach_fetch.py`, login node, `fdp run`, three
workers, one request a second; everything lands in `$LABELER_ROOT/round4/detach/cache`):
`\BOLOM::PRAD_{DIVL,DIVU,TOT,CORE}`; the EFIT01 `aeqdsk` scalars (X-point,
the four strike points, stored energy, plasma current, field, shape, `betan`, `q95`,
ohmic power); the CO2 chords `DENV2UF`, `DENV3UF`, `DENR0UF` (10 ms block means) for the
line density; `ECHPWR`; the beam total `\NB::PINJ` where the corpus `pinj` is a stub;
and the divertor Thomson Te points `TSSDIVTE00-05`, which no indicator reads (they are
the independent check). For the appendix figure also the EFIT01 flux map (`PSIRZ`, `SSIMAG`, `SSIBRY`,
the `R`, `Z` grid and the boundary `RBBBS`, `ZBBBS`; `detach_fetch_efit.py`).

From the owner's plasma_tv work (`/scratch/gpfs/nc1514/plasma_tv`, and the catalog's
`data/events/detachment/raw/`): 41 TangTV inversions (`emission_structure_*.sav`, the
C-III emissivity on the lower-divertor plane, Chen 2026), packed by
`detach_inversions.py`; these are the only places the inversion exists, and the only
training data of the frame regression.

Provenance lines: the label producer is `detach_vote` (git sha in the `.meta.json`),
the fallback is `detach_rule`. The cohort split is `data/events/catalog/cohort.csv`.

## Reproduce

Every number in this page is written by a script in `scripts/labeler/` into a JSON in
`docs/labeler/results/` (the JSON is named beside each result). Order (`pixi run -e
labelmaker` for everything except the two GPU steps; `LABELER_ROOT` set):

```bash
# 1. survey the corpus, fetch (login node, fdp), pack the owner's inversions
python scripts/labeler/detach_survey.py
pixi run --frozen -e labelmaker fdp run python scripts/labeler/detach_fetch.py \
    --shots-file $LABELER_ROOT/round4/detach/shots_fetch.txt --workers 3 --pace 1
python scripts/labeler/detach_inversions.py

# 2. the TangTV front-height regression (CV, then the deployed predictions)
python scripts/labeler/detach_tv_surrogate.py --shots-file $SHOTS --no-predict
python scripts/labeler/detach_tv_surrogate.py --shots-file $SHOTS --predict-only

# 3. bins, the label, the benchmark
python scripts/labeler/detach_bins.py --shots-file $SHOTS --redo
python scripts/labeler/detach_label.py
python scripts/labeler/detach_benchmark.py

# 4. bin-width sensitivity (three grids, one fitted model)
for w in 20 50 100; do python scripts/labeler/detach_bins.py --shots-file $SHOTS_TV \
    --redo --width-ms $w --out-dir $LABELER_ROOT/round4/detach/bins_w$w; done
python scripts/labeler/detach_sensitivity.py --shots-file $SHOTS_TV

# 5. the two learned baselines (GPU: the phase3 environment, one device)
python scripts/labeler/detach_ours.py prep
python scripts/labeler/detach_victor.py prep
CUDA_VISIBLE_DEVICES=1 $LABELER_ROOT/envs/phase3/bin/python scripts/labeler/detach_ours.py train
CUDA_VISIBLE_DEVICES=1 $LABELER_ROOT/envs/phase3/bin/python scripts/labeler/detach_victor.py train

# 6. the appendix figure (flux maps fetched first, login node)
pixi run --frozen -e labelmaker fdp run python scripts/labeler/detach_fetch_efit.py \
    --shots 189057 --pace 1
python scripts/labeler/detach_figure.py --shot 189057
```

Tests: `tests/labeler/test_detachment_{core,indicators,label_model,surrogate}.py`.

## Limitations

* **The label is unverified.** No expert has reviewed it; `detach-ui` is where that
  happens. The only independent check (divertor Thomson Te) is weak, so the label is an
  agreement of three imperfect indicators and nothing more.
* **The indicators barely agree** (Afrac and Prad,div at kappa 0.01), so the label model's
  accuracies are learned from a small anchor (2,557 bins of 40 shots, the owner's inversion
  shots) and the 0.96 for TangTV is the cap of the fit. A different anchor set would give
  different weights; there is no truth to calibrate them against.
* **Uncertain is 58 % of the assessed bins.** That is the model's honest answer where only
  Afrac and/or Prad,div speak, and a threshold of 0.6 instead of 0.7 would label most of
  them (35 % uncertain) on the strength of one weak indicator.
* **Afrac is not the published Afrac.** The corpus has raw swept probes only (decoded here,
  uncalibrated, no probe positions), and the indicator is self-referenced to the shot's own
  0.90 quantile; it exists on 219 of the 470 shots (the 2018-2019 shots with a probe
  record), never on the 2026 shots.
* **Prad,div is a radiation measure.** It abstains in a wide band and does no better than
  chance against the cold-plate check; `f_div` needs the heating power, so a shot without
  beam power (201081, the worked example of Chen 2026) has no Prad,div and, with no probes
  either, no label.
* **TangTV is valid only on the lower shelf.** The floor-strike shots (195952 to 195963,
  206879 to 206894) have no TangTV vote; a separate floor model would be needed and is not
  built. On most shots it is a regression from the raw frame, trained on 28 shots of the
  2018-2019 and 2021 campaigns, validated by leave-one-shot-out only; how well it
  generalises to other campaigns is not known.
* **No IRTV, no bolometer image.** The brief's fourth row and the geometry of the bolometer
  are not in the corpus or in the `\BOLOM` tree.
* **The Te check is sparse**: real-time points, the peak per bin, an exact 0 eV (30 % of
  bins) read as a failed fit.
* **Small test split.** The cohort's test split carries 4 to 80 bins per leave-one-out row
  and 8 shots with a camera frame; its intervals are wide, and nothing was tuned on it.
* **The baselines are scored against this label**, not against truth (a model that
  reproduces the label reproduces its errors); `detach-ours` shares inputs with Afrac and
  Prad,div and with the TangTV geometry gate, and `detach-victor` shares frames with the TangTV
  vote.
* **The threshold values are from the literature, not fitted** (`thresholds.py`); no
  tuning was done on the owner's inversion shots or on the test split.

## References

- Ratner, A., Bach, S. H., Ehrenberg, H., Fries, J., Wu, S., Re, C., "Snorkel:
  rapid training data creation with weak supervision", PVLDB 11(3), 2017,
  doi:10.14778/3157794.3157797.
- Chen, N., Byun, C., Jalalvand, A., Kim, S., Rothstein, A., Scotti, F., Allen, S.,
  Eldon, D., Erickson, K., Kolemen, E., "Regulation compliant AI for fusion:
  explainable image-based feedback control of divertor detachment in DIII-D
  tokamak", Nucl. Fusion 66, 036014 (2026), doi:10.1088/1741-4326/ae3972.
- Eldon, D. et al., "Controlling marginally detached divertor plasmas", Nucl.
  Fusion 57, 066039 (2017), doi:10.1088/1741-4326/aa6b16.
- Eldon, D. et al., "Advances in radiated power control at DIII-D", Nucl. Mater.
  Energy 18, 285 (2019), doi:10.1016/j.nme.2019.01.010.
- Eldon, D. et al., "An analysis of controlled detachment by seeding various
  impurity species in high performance scenarios on DIII-D and EAST", Nucl. Mater.
  Energy 27, 100963 (2021), doi:10.1016/j.nme.2021.100963.
- Eldon, D. et al., "Enhancement of detachment control with simplified real-time
  modelling on the KSTAR tokamak", Plasma Phys. Control. Fusion 64, 075002 (2022),
  doi:10.1088/1361-6587/ac6ff9.
- Leonard, A. W., "Plasma detachment in divertor tokamaks", Plasma Phys. Control.
  Fusion 60, 044001 (2018) (OSTI 1432037).
- Victor, B. S., Scotti, F., "Identifying divertor detachment using a machine
  learning model trained on divertor camera images from DIII-D", Rev. Sci. Instrum.
  95, 083503 (2024), doi:10.1063/5.0218724.
