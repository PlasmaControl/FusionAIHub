Source: outputs/labeler/rwm/evaluation.json. Brackets report 95% shot-bootstrap intervals (1,000 resamples), conditional on fixed fitted predictions; exploratory, unadjusted for multiple comparisons. Individual metrics and detection-minus-reference use percentile intervals; between-model differences use basic paired intervals. Within-shot means/medians weight each two-class Hanson shot equally; one-class shots are omitted from those summaries, with counts in JSON. Phase-controlled AUROC compares only positive-negative pairs within campaign x 200 ms elapsed-time bin, weighted by pair count; phase shot resampling is stratified by campaign. High-beta means beta_N >= 0.8 times the shot's whole-window beta_N p95; above-proxy means beta_N/li > 4. Negative slices are assumed negative.

### Five-split AUROC ranges — rwm-brf (seeds 0–4)

| campaign / scope | primary AUROC range | high-beta conditional AUROC range | above-proxy conditional AUROC range |
|---|---|---|---|
| pooled | 0.760–0.806 | 0.602–0.696 | 0.546–0.657 |
| 2014 | 0.663–0.735 | 0.311–0.527 | 0.277–0.508 |
| 2018 | 0.819–0.848 | 0.698–0.747 | 0.647–0.706 |

Ranges are min/max point estimates across seeds 0–4, not confidence intervals.

### Five-split paired AUROC — forest minus elapsed time

| fold seed | primary AUROC difference | high-beta conditional AUROC difference | above-proxy conditional AUROC difference |
|---|---|---|---|
| 0 | 0.001 [-0.049, 0.060] | -0.040 [-0.120, 0.061] | -0.051 [-0.157, 0.092] |
| 1 | 0.028 [-0.023, 0.084] | 0.010 [-0.064, 0.101] | 0.007 [-0.091, 0.129] |
| 2 | 0.014 [-0.032, 0.065] | -0.011 [-0.087, 0.072] | -0.024 [-0.115, 0.087] |
| 3 | 0.047 [-0.000, 0.098] | 0.054 [-0.012, 0.129] | 0.061 [-0.025, 0.169] |
| 4 | 0.034 [-0.017, 0.088] | 0.022 [-0.054, 0.105] | 0.017 [-0.076, 0.128] |
| point range | 0.001–0.047 | -0.040–0.054 | -0.051–0.061 |

Forest minus elapsed time; 95% basic paired shot-bootstrap intervals. Intervals condition on fixed fitted predictions; elapsed-time ranks are fixed across splits. These three pooled strata retain discharge-phase information. The primary seed-3 lower bound is -0.0001, borderline near zero; a bootstrap-bound sign change alone would not establish robust superiority.

### Campaign paired AUROC — five splits and run-record holdout

| campaign | evaluation | primary AUROC difference | high-beta conditional AUROC difference | above-proxy conditional AUROC difference |
|---|---|---|---|---|
| 2014 | seed 0 | -0.047 [-0.126, 0.050] | -0.255 [-0.422, -0.067] | -0.262 [-0.451, -0.045] |
| 2014 | seed 1 | 0.006 [-0.066, 0.091] | -0.113 [-0.231, 0.059] | -0.105 [-0.246, 0.062] |
| 2014 | seed 2 | -0.011 [-0.078, 0.068] | -0.148 [-0.247, -0.002] | -0.155 [-0.267, -0.004] |
| 2014 | seed 3 | 0.025 [-0.039, 0.111] | -0.039 [-0.148, 0.125] | -0.031 [-0.169, 0.159] |
| 2014 | seed 4 | 0.012 [-0.061, 0.110] | -0.097 [-0.223, 0.111] | -0.106 [-0.245, 0.089] |
| 2014 | run-record holdout | 0.019 [-0.052, 0.096] | -0.126 [-0.245, 0.005] | -0.134 [-0.293, 0.011] |
| 2018 | seed 0 | -0.025 [-0.057, 0.033] | -0.033 [-0.078, 0.048] | -0.044 [-0.109, 0.065] |
| 2018 | seed 1 | -0.012 [-0.039, 0.031] | -0.009 [-0.049, 0.059] | -0.023 [-0.081, 0.061] |
| 2018 | seed 2 | -0.020 [-0.050, 0.036] | -0.024 [-0.068, 0.066] | -0.042 [-0.106, 0.075] |
| 2018 | seed 3 | 0.004 [-0.020, 0.046] | 0.016 [-0.016, 0.081] | 0.015 [-0.040, 0.109] |
| 2018 | seed 4 | -0.012 [-0.041, 0.045] | -0.005 [-0.052, 0.080] | -0.014 [-0.084, 0.101] |
| 2018 | run-record holdout | -0.049 [-0.073, -0.013] | -0.070 [-0.117, -0.010] | -0.057 [-0.109, 0.011] |

Forest minus elapsed time; 95% basic paired shot-bootstrap intervals condition on fixed fitted predictions. High-beta: beta_N >= 0.8 times the shot's beta_N p95; above-proxy: beta_N/li > 4. Campaign 2014 forest is below chance on the reference split (0.311 [0.22, 0.41]); the scalar rules are near chance. Forest five-split range: 0.311–0.527; below elapsed time on every split (point estimates; CI excludes zero on 2 of 5 seeds). Included 2018 run-record holdout CIs exclude zero: primary -0.049 [-0.073, -0.013]; high-beta -0.070 [-0.117, -0.010].

### Piccione-style primary scores — all models (reference split, seed 0)

| model | AUROC (95% CI) | AUPRC (95% CI) | F1 (95% CI) | slice TPR (95% CI) | slice FPR (95% CI) |
|---|---|---|---|---|---|
| rwm-brf | 0.760 [0.706, 0.809] | 0.163 [0.123, 0.220] | 0.275 [0.227, 0.337] | 0.756 [0.646, 0.869] | 0.342 [0.261, 0.421] |
| Elapsed time | 0.759 [0.712, 0.815] | 0.228 [0.212, 0.307] | 0.267 [0.217, 0.331] | 0.692 [0.525, 0.830] | 0.318 [0.244, 0.368] |
| βN | 0.707 [0.639, 0.772] | 0.166 [0.128, 0.246] | 0.246 [0.194, 0.312] | 0.750 [0.619, 0.865] | 0.397 [0.323, 0.454] |
| βN/li | 0.723 [0.664, 0.784] | 0.159 [0.127, 0.232] | 0.261 [0.212, 0.328] | 0.735 [0.602, 0.847] | 0.356 [0.269, 0.427] |
| RWM screen | 0.500 [0.500, 0.500] | 0.084 [0.070, 0.102] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Broader Hanson-negative sensitivity — same models and predictions

| model | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| rwm-brf | 0.740 [0.668, 0.799] | 0.065 [0.043, 0.103] |
| Elapsed time | 0.390 [0.333, 0.440] | 0.025 [0.019, 0.032] |
| βN | 0.715 [0.641, 0.776] | 0.064 [0.043, 0.099] |
| βN/li | 0.752 [0.688, 0.815] | 0.074 [0.052, 0.115] |
| RWM screen | 0.498 [0.495, 0.500] | 0.034 [0.026, 0.041] |

### Within-shot AUROC — primary and broad Hanson masks (reference split, seed 0)

| model | mask | shots with both classes | median | mean |
|---|---|---|---|---|
| rwm-brf | primary | 30 | 0.830 | 0.784 |
| rwm-brf | broad | 30 | 0.760 | 0.719 |
| Elapsed time | primary | 30 | 1.000 | 0.931 |
| Elapsed time | broad | 30 | 0.419 | 0.417 |
| βN | primary | 30 | 0.884 | 0.814 |
| βN | broad | 30 | 0.821 | 0.739 |
| βN/li | primary | 30 | 0.900 | 0.855 |
| βN/li | broad | 30 | 0.866 | 0.786 |
| RWM screen | primary | 30 | 0.500 | 0.500 |
| RWM screen | broad | 30 | 0.500 | 0.499 |

### Paired within-shot mean AUROC differences (95% basic shot CIs)

| forest minus scalar | primary | broad | two-class shots per mask |
|---|---|---|---|
| rwm-brf - Elapsed time | -0.147 [-0.200, -0.084] | 0.302 [0.217, 0.384] | 30/30 |
| rwm-brf - βN | -0.031 [-0.109, 0.045] | -0.020 [-0.095, 0.058] | 30/30 |
| rwm-brf - βN/li | -0.071 [-0.133, -0.010] | -0.067 [-0.137, 0.002] | 30/30 |

### Phase-controlled primary AUROC — within campaign and 200 ms time bins

| model / rule | AUROC (95% percentile shot CI) |
|---|---|
| rwm-brf | 0.544 [0.453, 0.629] |
| Elapsed time | 0.582 [0.552, 0.650] |
| βN | 0.541 [0.439, 0.643] |
| βN/li | 0.595 [0.494, 0.696] |
| RWM screen | 0.500 [0.500, 0.500] |

### Paired phase-controlled AUROC — forest minus scalar

| forest minus scalar | AUROC difference (95% basic paired shot CI) |
|---|---|
| rwm-brf - Elapsed time | -0.037 [-0.125, 0.091] |
| rwm-brf - βN | 0.004 [-0.112, 0.134] |
| rwm-brf - βN/li | -0.051 [-0.167, 0.072] |

### Comparison pool — reference forest alarms by run title

| run title | unlabelled shots | shots with an alarm | alarm incidence |
|---|---|---|---|
| control of divertor radiation with impurity seeding in high betap scenario | 15 | 0 | 0.000 |
| explore access to beta_n~5 using high-li approach | 28 | 1 | 0.036 |
| explore access to bn~5 using high qmin approach - day 1 | 2 | 1 | 0.500 |
| explore access to bn~5 using high qmin approach - day 2 | 19 | 3 | 0.158 |
| extend fully non-inductive high beta-p scenario to 1ma, q95=5 | 34 | 9 | 0.265 |
| rwm control development for high βp scenario | 3 | 1 | 0.333 |
| testing kinetic rwm stabilization theory at marginal stability | 31 | 4 | 0.129 |

### Rotation sensitivity — one reference-split no-rotation CV

| mask | original AUROC | no-rotation AUROC | paired change |
|---|---|---|---|
| primary | 0.760 [0.706, 0.809] | 0.752 [0.700, 0.803] | -0.008 [-0.021, 0.005] |
| broad | 0.740 [0.668, 0.799] | 0.754 [0.686, 0.810] | 0.014 [-0.001, 0.028] |

Source: outputs/labeler/rwm/rotation_ablation.json. Change is no rotation minus original forest; identical outer shots, inner splits and seeds. Input-dependence sensitivity does not measure upstream timing bias.

### Broad-mask run-record holdout — forest minus elapsed time

| AUROC difference (95% basic paired CI) |
|---|
| 0.362 [0.304, 0.428] |

### High-beta conditional scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) | positive slices | assumed-negative slices | prevalence |
|---|---|---|---|---|---|
| rwm-brf | 0.602 [0.491, 0.688] | 0.168 [0.123, 0.228] | 365 | 2351 | 0.134 |
| Elapsed time | 0.642 [0.557, 0.732] | 0.255 [0.241, 0.343] | 365 | 2351 | 0.134 |
| βN | 0.563 [0.468, 0.651] | 0.176 [0.132, 0.273] | 365 | 2351 | 0.134 |
| βN/li | 0.593 [0.509, 0.677] | 0.165 [0.132, 0.249] | 365 | 2351 | 0.134 |
| RWM screen | 0.500 [0.500, 0.500] | 0.134 [0.108, 0.171] | 365 | 2351 | 0.134 |

### Above no-wall-proxy conditional scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) | positive slices | assumed-negative slices | prevalence |
|---|---|---|---|---|---|
| rwm-brf | 0.546 [0.416, 0.642] | 0.176 [0.130, 0.240] | 365 | 1913 | 0.160 |
| Elapsed time | 0.597 [0.524, 0.676] | 0.268 [0.254, 0.355] | 365 | 1913 | 0.160 |
| βN | 0.509 [0.386, 0.617] | 0.183 [0.139, 0.279] | 365 | 1913 | 0.160 |
| βN/li | 0.512 [0.428, 0.591] | 0.168 [0.132, 0.252] | 365 | 1913 | 0.160 |
| RWM screen | 0.500 [0.500, 0.500] | 0.160 [0.125, 0.215] | 365 | 1913 | 0.160 |

### Leave-one-run-record-out — rwm-brf (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| pooled four-record holdout | primary | 33 | 480 | 5255 | 0.084 | 0.779 [0.731, 0.823] | 0.181 [0.149, 0.248] |
| pooled four-record holdout | high-beta conditional | 33 | 365 | 2351 | 0.134 | 0.621 [0.536, 0.700] | 0.176 [0.149, 0.239] |
| pooled four-record holdout | above-proxy conditional | 33 | 365 | 1913 | 0.160 | 0.579 [0.475, 0.677] | 0.186 [0.154, 0.258] |

### Leave-one-run-record-out — each held-out run record

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (point estimate) | AUPRC (point estimate) |
|---|---|---|---|---|---|---|---|
| 20140421 | primary | 11 | 180 | 1273 | 0.124 | 0.759 | 0.240 |
| 20140421 | high-beta conditional | 11 | 119 | 395 | 0.232 | 0.532 | 0.238 |
| 20140421 | above-proxy conditional | 11 | 113 | 334 | 0.253 | 0.457 | 0.224 |
| 20140620A | primary | 9 | 80 | 1007 | 0.074 | 0.742 | 0.152 |
| 20140620A | high-beta conditional | 9 | 49 | 359 | 0.120 | 0.546 | 0.164 |
| 20140620A | above-proxy conditional | 9 | 63 | 371 | 0.145 | 0.482 | 0.164 |
| 20180314 | primary | 6 | 130 | 1414 | 0.084 | 0.813 | 0.297 |
| 20180314 | high-beta conditional | 6 | 110 | 819 | 0.118 | 0.705 | 0.288 |
| 20180314 | above-proxy conditional | 6 | 104 | 395 | 0.208 | 0.601 | 0.323 |
| 20180314A | primary | 7 | 90 | 1561 | 0.055 | 0.757 | 0.133 |
| 20180314A | high-beta conditional | 7 | 87 | 778 | 0.101 | 0.600 | 0.143 |
| 20180314A | above-proxy conditional | 7 | 85 | 813 | 0.095 | 0.639 | 0.147 |

### Leave-one-run-record-out — F1 and alarms (95% shot CIs)

| held-out group | primary F1 (95% shot CI) | onsets warned | onset detection (95% shot CI) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (95% shot CI) | median warning, ms (95% shot CI) |
|---|---|---|---|---|---|---|---|
| pooled four-record holdout | 0.285 [0.233, 0.344] | 8/48 | 0.167 [0.049, 0.309] | 4/30 | 9/33 | 0.273 [0.152, 0.424] | 214 [140, 343] |

### Leave-one-run-record-out — F1 and alarms by held-out run record

| held-out group | primary F1 (point estimate) | onsets warned | onset detection (point estimate) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (point estimate) | median warning, ms (point estimate) |
|---|---|---|---|---|---|---|---|
| 20140421 | 0.364 | 0/18 | 0.000 | 0/11 | 0/11 | 0.000 | - |
| 20140620A | 0.249 | 4/8 | 0.500 | 2/7 | 5/9 | 0.556 | 214 |
| 20180314 | 0.329 | 1/13 | 0.077 | 0/6 | 0/6 | 0.000 | 140 |
| 20180314A | 0.153 | 3/9 | 0.333 | 2/6 | 4/7 | 0.571 | 343 |

### Alarm counts — all models (n=1 targets)

| model | onsets warned | Hanson shots: unexplained alarm | unlabelled shots: any alarm |
|---|---|---|---|
| rwm-brf | 9/48 | 8/33 | 19/132 |
| Elapsed time | 3/48 | 4/33 | 128/132 |
| βN | 4/48 | 1/33 | 58/132 |
| βN/li | 1/48 | 3/33 | 30/132 |
| RWM screen | 0/48 | 0/33 | 36/132 |

### Alarm rates — all models (95% shot CIs)

| model | onset detection | Hanson unexplained incidence | alarm incidence on unlabelled shots |
|---|---|---|---|
| rwm-brf | 0.188 [0.049, 0.333] | 0.242 [0.091, 0.394] | 0.144 [0.083, 0.212] |
| Elapsed time | 0.062 [0.000, 0.137] | 0.121 [0.030, 0.242] | 0.970 [0.939, 1.000] |
| βN | 0.083 [0.019, 0.163] | 0.030 [0.000, 0.121] | 0.439 [0.356, 0.515] |
| βN/li | 0.021 [0.000, 0.064] | 0.091 [0.000, 0.212] | 0.227 [0.167, 0.295] |
| RWM screen | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.273 [0.197, 0.348] |

### Warning times — all models (detected onsets only)

| model | median warning, ms (95% CI) | uniform-alarm reference | detection minus reference |
|---|---|---|---|
| rwm-brf | 356 [286, 389] | 0.190 [0.080, 0.302] | -0.002 [-0.058, 0.052] |
| Elapsed time | 145 [36, 235] | 0.046 [0.023, 0.069] | 0.016 [-0.039, 0.083] |
| βN | 34 [16, 385] | 0.025 [0.006, 0.050] | 0.058 [0.014, 0.115] |
| βN/li | 36 [36, 36] | 0.030 [0.006, 0.065] | -0.009 [-0.052, 0.030] |
| RWM screen | - | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Forest alarm sensitivity — five splits and run-record holdout

| evaluation | first alarm Detected/Early/Missed | any alarm Detected/Early/Missed (sensitivity) | onsets warned | onset detection (95% CI) | detection minus random reference (95% percentile CI) | median warning, ms (95% CI) |
|---|---|---|---|---|---|---|
| seed 0 | 2/6/22 | 6/2/22 | 9/48 | 0.188 [0.049, 0.333] | -0.002 [-0.058, 0.052] | 356 [286, 389] |
| seed 1 | 5/5/20 | 6/4/20 | 8/48 | 0.167 [0.046, 0.302] | 0.031 [-0.042, 0.118] | 179 [75, 343] |
| seed 2 | 4/5/21 | 7/2/21 | 10/48 | 0.208 [0.085, 0.340] | 0.038 [-0.012, 0.094] | 302 [56, 340] |
| seed 3 | 8/5/17 | 11/2/17 | 13/48 | 0.271 [0.158, 0.392] | 0.049 [-0.054, 0.142] | 239 [114, 347] |
| seed 4 | 4/1/25 | 5/1/24 | 6/48 | 0.125 [0.025, 0.245] | 0.057 [-0.021, 0.158] | 135 [38, 341] |
| run-record holdout | 3/4/23 | 6/1/23 | 8/48 | 0.167 [0.049, 0.309] | 0.014 [-0.025, 0.053] | 214 [140, 343] |
| five-split point range | — | — | — | 0.125–0.271 | -0.002–0.057 | 135–356 |

No improvement over the approximate rate-matched random-alarm reference was established: all five detection-difference intervals include zero. This does not establish equivalence. Warning medians condition on detected onsets; intervals condition on fixed fitted predictions. In the reference split, the beta_N rule warns 4/48 onsets and has detection minus reference 0.058 [0.014, 0.115]; its low detection coverage limits that result.

### n=1 onset physics — actual onset, by campaign

| campaign | shot | n=1 onset, ms | beta_N | li | beta_N/li | elapsed time, ms | proxy category | pre-onset high-beta slices |
|---|---|---|---|---|---|---|---|---|
| 2014 | 156785 | 856.0 | 1.28 | 0.52 | 2.46 | 749 | below 4 | 0 |
| 2014 | 156786 | 1892.0 | 2.59 | 0.61 | 4.23 | 1782 | at or above 4 | 10 |
| 2014 | 156786 | 2318.5 | 2.58 | 0.72 | 3.61 | 2208 | below 4 | 10 |
| 2014 | 156787 | 1146.5 | 2.63 | 0.63 | 4.20 | 1040 | at or above 4 | 10 |
| 2014 | 156787 | 1616.9 | 2.62 | 0.65 | 4.06 | 1510 | at or above 4 | 10 |
| 2014 | 156790 | 1920.0 | 3.21 | 0.69 | 4.64 | 1812 | at or above 4 | 10 |
| 2014 | 156791 | 1429.5 | 2.90 | 0.68 | 4.29 | 1322 | at or above 4 | 10 |
| 2014 | 156792 | 1498.5 | 3.02 | 0.63 | 4.78 | 1390 | at or above 4 | 10 |
| 2014 | 156792 | 2802.0 | 3.51 | 0.75 | 4.65 | 2693 | at or above 4 | 10 |
| 2014 | 156793 | 1945.5 | 2.92 | 0.59 | 4.97 | 1836 | at or above 4 | 2 |
| 2014 | 156793 | 2725.0 | 3.60 | 0.62 | 5.76 | 2616 | at or above 4 | 10 |
| 2014 | 156794 | 1127.0 | 2.80 | 0.70 | 4.02 | 870 | at or above 4 | 0 |
| 2014 | 156794 | 1306.0 | 3.01 | 0.62 | 4.86 | 1049 | at or above 4 | 3 |
| 2014 | 156795 | 1656.0 | 3.14 | 0.58 | 5.43 | 1419 | at or above 4 | 10 |
| 2014 | 156795 | 2283.3 | 2.98 | 0.62 | 4.84 | 2046 | at or above 4 | 10 |
| 2014 | 156796 | 2730.0 | 2.30 | 0.66 | 3.51 | 2360 | below 4 | 0 |
| 2014 | 156797 | 1598.0 | - | - | - | 1236 | missing EFIT | 0 |
| 2014 | 156797 | 1998.0 | 2.76 | 0.60 | 4.60 | 1636 | at or above 4 | 4 |
| 2014 | 158012 | 1406.0 | 2.59 | 0.62 | 4.18 | 1190 | at or above 4 | 0 |
| 2014 | 158013 | 1450.0 | 3.05 | 0.64 | 4.81 | 1232 | at or above 4 | 10 |
| 2014 | 158013 | 3061.1 | 3.35 | 0.59 | 5.69 | 2843 | at or above 4 | 10 |
| 2014 | 158014 | 1927.0 | 2.78 | 0.59 | 4.73 | 1691 | at or above 4 | 10 |
| 2014 | 158018 | 2016.0 | 3.03 | 0.57 | 5.34 | 1777 | at or above 4 | 9 |
| 2014 | 158019 | 1608.0 | 2.91 | 0.58 | 4.99 | 1373 | at or above 4 | 0 |
| 2014 | 158021 | 1084.0 | 2.24 | 0.62 | 3.63 | 845 | below 4 | 0 |
| 2014 | 158022 | 2996.4 | 3.61 | 0.71 | 5.05 | 2758 | at or above 4 | 10 |
| 2018 | 176067 | 2638.0 | 2.72 | 0.59 | 4.60 | 2177 | at or above 4 | 10 |
| 2018 | 176068 | 2418.5 | 2.63 | 0.62 | 4.25 | 1964 | at or above 4 | 10 |
| 2018 | 176068 | 2692.0 | 2.70 | 0.58 | 4.67 | 2238 | at or above 4 | 10 |
| 2018 | 176068 | 2970.5 | 2.63 | 0.57 | 4.64 | 2516 | at or above 4 | 10 |
| 2018 | 176068 | 3293.0 | 2.74 | 0.57 | 4.79 | 2839 | at or above 4 | 10 |
| 2018 | 176069 | 2269.7 | 2.66 | 0.63 | 4.23 | 1815 | at or above 4 | 10 |
| 2018 | 176069 | 3235.0 | - | - | - | 2781 | missing EFIT | 0 |
| 2018 | 176069 | 3789.5 | 2.87 | 0.56 | 5.16 | 3335 | at or above 4 | 10 |
| 2018 | 176070 | 2818.4 | - | - | - | 2330 | missing EFIT | 6 |
| 2018 | 176070 | 3013.5 | - | - | - | 2525 | missing EFIT | 8 |
| 2018 | 176071 | 3071.0 | 2.45 | 0.57 | 4.28 | 2588 | at or above 4 | 6 |
| 2018 | 176074 | 2239.0 | 2.64 | 0.60 | 4.41 | 1790 | at or above 4 | 10 |
| 2018 | 176074 | 3461.0 | 2.85 | 0.58 | 4.95 | 3012 | at or above 4 | 10 |
| 2018 | 176077 | 1653.5 | 2.82 | 0.61 | 4.63 | 1206 | at or above 4 | 10 |
| 2018 | 176078 | 3435.0 | 3.05 | 0.58 | 5.25 | 2985 | at or above 4 | 10 |
| 2018 | 176078 | 4149.0 | 3.06 | 0.64 | 4.80 | 3699 | at or above 4 | 10 |
| 2018 | 176085 | 3560.0 | 3.33 | 0.61 | 5.45 | 3116 | at or above 4 | 10 |
| 2018 | 176087 | 3138.0 | 3.13 | 0.50 | 6.24 | 2680 | at or above 4 | 10 |
| 2018 | 176088 | 2121.0 | 3.19 | 0.62 | 5.12 | 1664 | at or above 4 | 10 |
| 2018 | 176088 | 2328.0 | 3.19 | 0.62 | 5.18 | 1871 | at or above 4 | 10 |
| 2018 | 176088 | 3616.0 | 3.74 | 0.61 | 6.12 | 3159 | at or above 4 | 10 |
| 2018 | 176089 | 2843.0 | 2.45 | 0.63 | 3.86 | 2359 | below 4 | 7 |

Offline EFIT inputs held at the actual listed n=1 onset (last sample age <= 50 ms): 5 snapshots are below beta_N/li = 4 and 4 have missing EFIT inputs; missing inputs remain explicit rather than being classified above or below the proxy. Elapsed time is measured from the first |Ip| >= 0.5 MA crossing. The high-beta slice count uses the forecast-positive pre-onset window.

### n=1 onset-window physics — first pre-onset window slice, by campaign

| campaign | shot | n=1 onset, ms | window sample, ms | beta_N | li | beta_N/li | elapsed time, ms | proxy category | pre-onset high-beta slices |
|---|---|---|---|---|---|---|---|---|---|
| 2014 | 156785 | 856.0 | 840.0 | 1.28 | 0.52 | 2.46 | 733 | below 4 | 0 |
| 2014 | 156786 | 1892.0 | 1880.0 | 2.59 | 0.60 | 4.30 | 1770 | at or above 4 | 10 |
| 2014 | 156786 | 2318.5 | 2300.0 | 2.81 | 0.71 | 3.96 | 2190 | below 4 | 10 |
| 2014 | 156787 | 1146.5 | 1130.0 | 2.59 | 0.66 | 3.95 | 1023 | below 4 | 10 |
| 2014 | 156787 | 1616.9 | 1600.0 | 2.62 | 0.65 | 4.06 | 1493 | at or above 4 | 10 |
| 2014 | 156790 | 1920.0 | 1900.0 | 3.33 | 0.69 | 4.82 | 1792 | at or above 4 | 10 |
| 2014 | 156791 | 1429.5 | 1410.0 | 2.75 | 0.68 | 4.05 | 1302 | at or above 4 | 10 |
| 2014 | 156792 | 1498.5 | 1480.0 | 3.02 | 0.63 | 4.78 | 1371 | at or above 4 | 10 |
| 2014 | 156792 | 2802.0 | 2790.0 | 3.55 | 0.76 | 4.68 | 2681 | at or above 4 | 10 |
| 2014 | 156793 | 1945.5 | 1930.0 | 2.93 | 0.59 | 4.95 | 1821 | at or above 4 | 2 |
| 2014 | 156793 | 2725.0 | 2710.0 | 3.64 | 0.63 | 5.78 | 2601 | at or above 4 | 10 |
| 2014 | 156794 | 1127.0 | 1110.0 | 2.75 | 0.72 | 3.83 | 853 | below 4 | 0 |
| 2014 | 156794 | 1306.0 | 1290.0 | 2.99 | 0.63 | 4.76 | 1033 | at or above 4 | 3 |
| 2014 | 156795 | 1656.0 | 1640.0 | 3.11 | 0.59 | 5.25 | 1403 | at or above 4 | 10 |
| 2014 | 156795 | 2283.3 | 2270.0 | 3.15 | 0.64 | 4.95 | 2033 | at or above 4 | 10 |
| 2014 | 156796 | 2730.0 | 2710.0 | 2.31 | 0.66 | 3.49 | 2340 | below 4 | 0 |
| 2014 | 156797 | 1598.0 | 1580.0 | - | - | - | 1218 | missing EFIT | 0 |
| 2014 | 156797 | 1998.0 | 1980.0 | 2.76 | 0.60 | 4.60 | 1618 | at or above 4 | 4 |
| 2014 | 158012 | 1406.0 | 1390.0 | 2.60 | 0.62 | 4.16 | 1174 | at or above 4 | 0 |
| 2014 | 158013 | 1450.0 | 1430.0 | 2.88 | 0.63 | 4.61 | 1212 | at or above 4 | 10 |
| 2014 | 158013 | 3061.1 | 3050.0 | 3.35 | 0.59 | 5.69 | 2832 | at or above 4 | 10 |
| 2014 | 158014 | 1927.0 | 1910.0 | 2.77 | 0.61 | 4.55 | 1674 | at or above 4 | 10 |
| 2014 | 158018 | 2016.0 | 2000.0 | 2.98 | 0.57 | 5.22 | 1761 | at or above 4 | 9 |
| 2014 | 158019 | 1608.0 | 1590.0 | 2.83 | 0.60 | 4.71 | 1355 | at or above 4 | 0 |
| 2014 | 158021 | 1084.0 | 1070.0 | 2.24 | 0.62 | 3.63 | 831 | below 4 | 0 |
| 2014 | 158022 | 2996.4 | 2980.0 | 3.61 | 0.72 | 5.05 | 2742 | at or above 4 | 10 |
| 2018 | 176067 | 2638.0 | 2620.0 | 2.72 | 0.59 | 4.60 | 2159 | at or above 4 | 10 |
| 2018 | 176068 | 2418.5 | 2400.0 | 2.63 | 0.62 | 4.25 | 1946 | at or above 4 | 10 |
| 2018 | 176068 | 2692.0 | 2680.0 | 2.70 | 0.58 | 4.67 | 2226 | at or above 4 | 10 |
| 2018 | 176068 | 2970.5 | 2960.0 | 2.63 | 0.57 | 4.64 | 2506 | at or above 4 | 10 |
| 2018 | 176068 | 3293.0 | 3280.0 | 2.75 | 0.57 | 4.81 | 2826 | at or above 4 | 10 |
| 2018 | 176069 | 2269.7 | 2250.0 | 2.65 | 0.65 | 4.09 | 1796 | at or above 4 | 10 |
| 2018 | 176069 | 3235.0 | 3220.0 | - | - | - | 2766 | missing EFIT | 0 |
| 2018 | 176069 | 3789.5 | 3770.0 | 2.80 | 0.57 | 4.92 | 3316 | at or above 4 | 10 |
| 2018 | 176070 | 2818.4 | 2800.0 | - | - | - | 2312 | missing EFIT | 6 |
| 2018 | 176070 | 3013.5 | 3000.0 | 2.25 | 0.55 | 4.10 | 2512 | at or above 4 | 8 |
| 2018 | 176071 | 3071.0 | 3060.0 | 2.45 | 0.57 | 4.28 | 2577 | at or above 4 | 6 |
| 2018 | 176074 | 2239.0 | 2220.0 | 2.64 | 0.60 | 4.41 | 1771 | at or above 4 | 10 |
| 2018 | 176074 | 3461.0 | 3450.0 | 2.92 | 0.58 | 5.02 | 3001 | at or above 4 | 10 |
| 2018 | 176077 | 1653.5 | 1640.0 | 2.74 | 0.64 | 4.28 | 1193 | at or above 4 | 10 |
| 2018 | 176078 | 3435.0 | 3420.0 | 3.05 | 0.58 | 5.25 | 2970 | at or above 4 | 10 |
| 2018 | 176078 | 4149.0 | 4130.0 | 3.07 | 0.64 | 4.79 | 3680 | at or above 4 | 10 |
| 2018 | 176085 | 3560.0 | 3540.0 | 3.36 | 0.61 | 5.56 | 3096 | at or above 4 | 10 |
| 2018 | 176087 | 3138.0 | 3120.0 | 3.13 | 0.50 | 6.24 | 2662 | at or above 4 | 10 |
| 2018 | 176088 | 2121.0 | 2110.0 | 3.21 | 0.62 | 5.13 | 1653 | at or above 4 | 10 |
| 2018 | 176088 | 2328.0 | 2310.0 | 3.32 | 0.58 | 5.72 | 1853 | at or above 4 | 10 |
| 2018 | 176088 | 3616.0 | 3600.0 | 3.74 | 0.61 | 6.12 | 3143 | at or above 4 | 10 |
| 2018 | 176089 | 2843.0 | 2830.0 | 2.62 | 0.62 | 4.24 | 2346 | at or above 4 | 7 |

First slice in the [onset - 20 ms, onset) window: 6 snapshots are below beta_N/li = 4 and 3 have missing EFIT inputs; missing inputs remain explicit rather than being classified above or below the proxy. Elapsed time is measured from the first |Ip| >= 0.5 MA crossing. The high-beta slice count uses the forecast-positive pre-onset window.

### n=1 onset and window coverage summary — by campaign

| campaign | n=1 onsets | actual onset below beta_N/li = 4 | actual onset missing EFIT | window sample below beta_N/li = 4 | window sample missing EFIT | no high-beta pre-onset slices |
|---|---|---|---|---|---|---|
| 2014 | 26 | 4/26 (15.4%) | 1/26 (3.8%) | 6/26 (23.1%) | 1/26 (3.8%) | 7/26 (26.9%) |
| 2018 | 22 | 1/22 (4.5%) | 3/22 (13.6%) | 0/22 (0.0%) | 2/22 (9.1%) | 1/22 (4.5%) |

### Piccione-style per-shot categories — primary alarm definition

| model | Detected Hanson shots | Missed Hanson shots | Early Hanson shots | Hanson shots without n=1 targets (excluded) | FP on comparison shots (alarm incidence) |
|---|---|---|---|---|---|
| rwm-brf | 2/30 | 22/30 | 6/30 | 3 | 19/132 |
| Elapsed time | 3/30 | 25/30 | 2/30 | 3 | 128/132 |
| βN | 3/30 | 26/30 | 1/30 | 3 | 58/132 |
| βN/li | 1/30 | 26/30 | 3/30 | 3 | 30/132 |
| RWM screen | 0/30 | 30/30 | 0/30 | 3 | 36/132 |

Detected, Early and Missed are mutually exclusive on Hanson shots with an n=1 target. The first considered alarm decides the primary shot category. Early means unexplained and more than 400 ms before a future target. The FP column reports unlabelled-shot alarm incidence, not a verified stable-shot false-positive rate.

### Any-alarm per-shot categories — labelled sensitivity

| model | Detected Hanson shots | Missed Hanson shots | Early Hanson shots | Hanson shots without n=1 targets (excluded) | FP on comparison shots (alarm incidence) |
|---|---|---|---|---|---|
| rwm-brf | 6/30 | 22/30 | 2/30 | 3 | 19/132 |
| Elapsed time | 3/30 | 25/30 | 2/30 | 3 | 128/132 |
| βN | 4/30 | 26/30 | 0/30 | 3 | 58/132 |
| βN/li | 1/30 | 26/30 | 3/30 | 3 | 30/132 |
| RWM screen | 0/30 | 30/30 | 0/30 | 3 | 36/132 |

Detected, Early and Missed are mutually exclusive on Hanson shots with an n=1 target. Sensitivity: any warning wins, otherwise any Early alarm, then Missed. Early means unexplained and more than 400 ms before a future target. The FP column reports unlabelled-shot alarm incidence, not a verified stable-shot false-positive rate.

### Alarm definition sensitivity — rwm-brf

| alarm definition | onsets warned | onset detection (95% CI) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (95% CI) | unlabelled shots with an alarm | unlabelled alarm incidence (95% CI) |
|---|---|---|---|---|---|---|---|
| primary: end 100 ms after last n=1/n=2 onset | 9/48 | 0.188 [0.049, 0.333] | 6/30 | 8/33 | 0.242 [0.091, 0.394] | 19/132 | 0.144 [0.083, 0.212] |
| full-trace sensitivity | 1/48 | 0.021 [0.000, 0.064] | 0/30 | 4/33 | 0.121 [0.030, 0.242] | 1/132 | 0.008 [0.000, 0.023] |

The primary alarm window ends 100 ms after the last n=1 or n=2 explanation onset on Hanson shots; comparison traces retain their full span. This tolerance extends beyond the primary slice mask, which ends at the last n=1 target onset. Both alarm definitions are tuned within the inner folds; unlabelled comparisons never tune alarms.

### Legacy NSTX — separately sourced published reference

| published model | slice AUROC | slice TPR | slice FPR | detected unstable shots | false-positive stable shots |
|---|---|---|---|---|---|
| NSTX RUS forest | 0.918 | 92.4% | 21.4% | 10/11 | 2/17 |

Piccione et al. (2022), doi:10.1088/1741-4326/ac44af. Different machine (NSTX), expert-reviewed stable shots, different inputs and validation; these published test results are not comparable to the DIII-D benchmark. F1 and confidence intervals are not available in the source digest. Source in evaluation.json: legacy.source = /scratch/gpfs/nc1514/FusionAIHub/.tmp/label_papers/Piccione_2022_Nucl._Fusion_62_036002.md.

### Paired differences — rwm-brf versus rules, primary

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - Elapsed time | 0.001 [-0.049, 0.060] | -0.065 [-0.100, 0.016] |
| rwm-brf - βN | 0.053 [-0.038, 0.140] | -0.003 [-0.064, 0.069] |
| rwm-brf - βN/li | 0.037 [-0.038, 0.115] | 0.004 [-0.047, 0.071] |

### Paired differences — rwm-brf versus rules, broad

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - Elapsed time | 0.350 [0.287, 0.414] | 0.040 [0.006, 0.058] |
| rwm-brf - βN | 0.025 [-0.068, 0.117] | 0.001 [-0.037, 0.036] |
| rwm-brf - βN/li | -0.013 [-0.092, 0.072] | -0.009 [-0.045, 0.029] |

### Paired differences — rwm-brf versus rules, high-beta conditional

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - Elapsed time | -0.040 [-0.120, 0.061] | -0.087 [-0.128, 0.011] |
| rwm-brf - βN | 0.039 [-0.101, 0.187] | -0.008 [-0.077, 0.087] |
| rwm-brf - βN/li | 0.009 [-0.092, 0.123] | 0.003 [-0.049, 0.081] |

### Paired differences — rwm-brf versus rules, above-proxy conditional

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - Elapsed time | -0.051 [-0.157, 0.092] | -0.091 [-0.138, 0.002] |
| rwm-brf - βN | 0.037 [-0.123, 0.192] | -0.007 [-0.074, 0.081] |
| rwm-brf - βN/li | 0.034 [-0.092, 0.167] | 0.008 [-0.046, 0.085] |

### Paired alarm differences — rwm-brf versus rules (95% basic CIs)

| first model minus rule | detection difference | Hanson incidence difference | unlabelled incidence difference |
|---|---|---|---|
| rwm-brf - Elapsed time | 0.125 [-0.021, 0.270] | 0.121 [-0.091, 0.333] | -0.826 [-0.894, -0.765] |
| rwm-brf - βN | 0.104 [-0.053, 0.245] | 0.212 [0.061, 0.333] | -0.295 [-0.394, -0.197] |
| rwm-brf - βN/li | 0.167 [0.015, 0.310] | 0.152 [0.000, 0.303] | -0.083 [-0.182, 0.015] |

### Split sensitivity — rwm-brf (fixed hyperparameters)

| model | fold seed | primary AUROC | high-beta conditional AUROC | detection rate | unlabelled alarm incidence |
|---|---|---|---|---|---|
| rwm-brf | 0 | 0.760 | 0.602 | 0.188 | 0.144 |
| rwm-brf | 1 | 0.787 | 0.651 | 0.167 | 0.280 |
| rwm-brf | 2 | 0.773 | 0.631 | 0.208 | 0.174 |
| rwm-brf | 3 | 0.806 | 0.696 | 0.271 | 0.242 |
| rwm-brf | 4 | 0.793 | 0.664 | 0.125 | 0.091 |

### Campaign sensitivity — rwm-brf, reference split, seed 0 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.663 [0.596, 0.745] | 0.130 [0.099, 0.190] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.311 [0.220, 0.408] | 0.125 [0.091, 0.180] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.277 [0.172, 0.382] | 0.132 [0.095, 0.200] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.819 [0.744, 0.879] | 0.236 [0.159, 0.336] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.698 [0.569, 0.801] | 0.229 [0.155, 0.332] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.647 [0.526, 0.763] | 0.244 [0.168, 0.362] |

### Campaign sensitivity — rwm-brf, seed 1 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.715 [0.646, 0.788] | 0.165 [0.118, 0.245] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.454 [0.337, 0.578] | 0.173 [0.111, 0.279] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.434 [0.332, 0.530] | 0.179 [0.120, 0.287] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.832 [0.763, 0.893] | 0.221 [0.163, 0.343] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.723 [0.599, 0.830] | 0.218 [0.154, 0.335] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.667 [0.536, 0.792] | 0.227 [0.163, 0.376] |

### Campaign sensitivity — rwm-brf, seed 2 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.698 [0.631, 0.772] | 0.147 [0.116, 0.209] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.419 [0.335, 0.527] | 0.149 [0.110, 0.220] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.384 [0.288, 0.481] | 0.155 [0.116, 0.237] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.824 [0.755, 0.881] | 0.211 [0.151, 0.312] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.707 [0.576, 0.805] | 0.206 [0.140, 0.312] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.649 [0.518, 0.764] | 0.216 [0.147, 0.336] |

### Campaign sensitivity — rwm-brf, seed 3 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.735 [0.673, 0.801] | 0.190 [0.137, 0.256] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.527 [0.395, 0.656] | 0.219 [0.144, 0.292] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.508 [0.383, 0.607] | 0.231 [0.149, 0.321] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.848 [0.787, 0.902] | 0.313 [0.231, 0.437] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.747 [0.626, 0.846] | 0.318 [0.233, 0.444] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.706 [0.577, 0.821] | 0.335 [0.252, 0.468] |

### Campaign sensitivity — rwm-brf, seed 4 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.722 [0.649, 0.794] | 0.161 [0.123, 0.232] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.470 [0.338, 0.616] | 0.168 [0.118, 0.249] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.433 [0.300, 0.558] | 0.170 [0.121, 0.256] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.832 [0.761, 0.890] | 0.240 [0.167, 0.344] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.727 [0.599, 0.829] | 0.241 [0.163, 0.349] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.677 [0.543, 0.790] | 0.257 [0.179, 0.382] |

### Leave-one-run-record-out — by campaign (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.729 [0.672, 0.814] | 0.173 [0.139, 0.292] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.441 [0.369, 0.582] | 0.167 [0.135, 0.278] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.404 [0.329, 0.536] | 0.172 [0.140, 0.304] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.796 [0.744, 0.845] | 0.206 [0.150, 0.301] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.661 [0.576, 0.746] | 0.198 [0.146, 0.295] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.633 [0.545, 0.729] | 0.213 [0.159, 0.319] |
