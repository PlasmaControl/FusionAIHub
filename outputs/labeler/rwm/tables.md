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

Forest minus elapsed time; 95% basic paired shot-bootstrap intervals. Elapsed-time ranks are fixed across splits. These three pooled strata retain discharge-phase information.

### Piccione-style primary scores — all models (split 0)

| model | AUROC (95% CI) | AUPRC (95% CI) | F1 (95% CI) |
|---|---|---|---|
| rwm-brf | 0.760 [0.706, 0.809] | 0.163 [0.123, 0.220] | 0.275 [0.227, 0.337] |
| rule-time-since-flattop | 0.759 [0.712, 0.815] | 0.228 [0.212, 0.307] | 0.267 [0.217, 0.331] |
| rule-betan | 0.707 [0.639, 0.772] | 0.166 [0.128, 0.246] | 0.246 [0.194, 0.312] |
| rule-betan-over-li | 0.723 [0.664, 0.784] | 0.159 [0.127, 0.232] | 0.261 [0.212, 0.328] |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.084 [0.070, 0.102] | 0.000 [0.000, 0.000] |

### Broader Hanson-negative sensitivity — same models and predictions

| model | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| rwm-brf | 0.740 [0.668, 0.799] | 0.065 [0.043, 0.103] |
| rule-time-since-flattop | 0.390 [0.333, 0.440] | 0.025 [0.019, 0.032] |
| rule-betan | 0.715 [0.641, 0.776] | 0.064 [0.043, 0.099] |
| rule-betan-over-li | 0.752 [0.688, 0.815] | 0.074 [0.052, 0.115] |
| rule-rwm-candidates | 0.498 [0.495, 0.500] | 0.034 [0.026, 0.041] |

### High-beta conditional scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) | positive slices | assumed-negative slices | prevalence |
|---|---|---|---|---|---|
| rwm-brf | 0.602 [0.491, 0.688] | 0.168 [0.123, 0.228] | 365 | 2351 | 0.134 |
| rule-time-since-flattop | 0.642 [0.557, 0.732] | 0.255 [0.241, 0.343] | 365 | 2351 | 0.134 |
| rule-betan | 0.563 [0.468, 0.651] | 0.176 [0.132, 0.273] | 365 | 2351 | 0.134 |
| rule-betan-over-li | 0.593 [0.509, 0.677] | 0.165 [0.132, 0.249] | 365 | 2351 | 0.134 |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.134 [0.108, 0.171] | 365 | 2351 | 0.134 |

### Above no-wall-proxy conditional scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) | positive slices | assumed-negative slices | prevalence |
|---|---|---|---|---|---|
| rwm-brf | 0.546 [0.416, 0.642] | 0.176 [0.130, 0.240] | 365 | 1913 | 0.160 |
| rule-time-since-flattop | 0.597 [0.524, 0.676] | 0.268 [0.254, 0.355] | 365 | 1913 | 0.160 |
| rule-betan | 0.509 [0.386, 0.617] | 0.183 [0.139, 0.279] | 365 | 1913 | 0.160 |
| rule-betan-over-li | 0.512 [0.428, 0.591] | 0.168 [0.132, 0.252] | 365 | 1913 | 0.160 |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.160 [0.125, 0.215] | 365 | 1913 | 0.160 |

### Campaign sensitivity — rwm-brf, split 0 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.663 [0.596, 0.745] | 0.130 [0.099, 0.190] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.311 [0.220, 0.408] | 0.125 [0.091, 0.180] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.277 [0.172, 0.382] | 0.132 [0.095, 0.200] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.819 [0.744, 0.879] | 0.236 [0.159, 0.336] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.698 [0.569, 0.801] | 0.229 [0.155, 0.332] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.647 [0.526, 0.763] | 0.244 [0.168, 0.362] |

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
| pooled four-record holdout | 0.285 [0.233, 0.344] | 8/48 | 0.167 [0.049, 0.309] | 1/30 | 9/33 | 0.273 [0.152, 0.424] | 214 [140, 343] |

### Leave-one-run-record-out — F1 and alarms by held-out run record

| held-out group | primary F1 (point estimate) | onsets warned | onset detection (point estimate) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (point estimate) | median warning, ms (point estimate) |
|---|---|---|---|---|---|---|---|
| 20140421 | 0.364 | 0/18 | 0.000 | 0/11 | 0/11 | 0.000 | - |
| 20140620A | 0.249 | 4/8 | 0.500 | 0/7 | 5/9 | 0.556 | 214 |
| 20180314 | 0.329 | 1/13 | 0.077 | 0/6 | 0/6 | 0.000 | 140 |
| 20180314A | 0.153 | 3/9 | 0.333 | 1/6 | 4/7 | 0.571 | 343 |

### Alarm counts — all models (n=1 targets)

| model | onsets warned | Hanson shots: unexplained alarm | unlabelled shots: any alarm |
|---|---|---|---|
| rwm-brf | 9/48 | 8/33 | 19/132 |
| rule-time-since-flattop | 3/48 | 4/33 | 128/132 |
| rule-betan | 4/48 | 1/33 | 58/132 |
| rule-betan-over-li | 1/48 | 3/33 | 30/132 |
| rule-rwm-candidates | 0/48 | 0/33 | 36/132 |

### Alarm rates — all models (95% shot CIs)

| model | onset detection | Hanson unexplained incidence | alarm incidence on unlabelled shots |
|---|---|---|---|
| rwm-brf | 0.188 [0.049, 0.333] | 0.242 [0.091, 0.394] | 0.144 [0.083, 0.212] |
| rule-time-since-flattop | 0.062 [0.000, 0.137] | 0.121 [0.030, 0.242] | 0.970 [0.939, 1.000] |
| rule-betan | 0.083 [0.019, 0.163] | 0.030 [0.000, 0.121] | 0.439 [0.356, 0.515] |
| rule-betan-over-li | 0.021 [0.000, 0.064] | 0.091 [0.000, 0.212] | 0.227 [0.167, 0.295] |
| rule-rwm-candidates | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.273 [0.197, 0.348] |

### Warning times — all models (detected onsets only)

| model | median warning, ms (95% CI) | uniform-alarm reference | detection minus reference |
|---|---|---|---|
| rwm-brf | 356 [286, 389] | 0.190 [0.080, 0.302] | -0.002 [-0.058, 0.052] |
| rule-time-since-flattop | 145 [36, 235] | 0.046 [0.023, 0.069] | 0.016 [-0.039, 0.083] |
| rule-betan | 34 [16, 385] | 0.025 [0.006, 0.050] | 0.058 [0.014, 0.115] |
| rule-betan-over-li | 36 [36, 36] | 0.030 [0.006, 0.065] | -0.009 [-0.052, 0.030] |
| rule-rwm-candidates | - | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |

### Piccione-style per-shot categories — primary alarm definition

| model | Detected Hanson shots | Missed Hanson shots | Early Hanson shots | Hanson shots without n=1 targets (excluded) | FP on comparison shots (alarm incidence) |
|---|---|---|---|---|---|
| rwm-brf | 6/30 | 22/30 | 2/30 | 3 | 19/132 |
| rule-time-since-flattop | 3/30 | 25/30 | 2/30 | 3 | 128/132 |
| rule-betan | 4/30 | 26/30 | 0/30 | 3 | 58/132 |
| rule-betan-over-li | 1/30 | 26/30 | 3/30 | 3 | 30/132 |
| rule-rwm-candidates | 0/30 | 30/30 | 0/30 | 3 | 36/132 |

Detected, Early and Missed are mutually exclusive on Hanson shots with an n=1 target: any Detected alarm takes precedence over Early, then Missed. Early means more than 400 ms before a listed onset. The FP column reports unlabelled-shot alarm incidence, not a verified stable-shot false-positive rate.

### Alarm definition sensitivity — rwm-brf

| alarm definition | onsets warned | onset detection (95% CI) | Early Hanson shots | Hanson shots with an unexplained alarm | Hanson unexplained incidence (95% CI) | unlabelled shots with an alarm | unlabelled alarm incidence (95% CI) |
|---|---|---|---|---|---|---|---|
| primary: end 100 ms after last n=1/n=2 onset | 9/48 | 0.188 [0.049, 0.333] | 2/30 | 8/33 | 0.242 [0.091, 0.394] | 19/132 | 0.144 [0.083, 0.212] |
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
| rwm-brf - rule-time-since-flattop | 0.001 [-0.049, 0.060] | -0.065 [-0.100, 0.016] |
| rwm-brf - rule-betan | 0.053 [-0.038, 0.140] | -0.003 [-0.064, 0.069] |
| rwm-brf - rule-betan-over-li | 0.037 [-0.038, 0.115] | 0.004 [-0.047, 0.071] |

### Paired differences — rwm-brf versus rules, high-beta conditional

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.040 [-0.120, 0.061] | -0.087 [-0.128, 0.011] |
| rwm-brf - rule-betan | 0.039 [-0.101, 0.187] | -0.008 [-0.077, 0.087] |
| rwm-brf - rule-betan-over-li | 0.009 [-0.092, 0.123] | 0.003 [-0.049, 0.081] |

### Paired differences — rwm-brf versus rules, above-proxy conditional

| first model minus rule | AUROC difference (95% basic CI) | AUPRC difference (95% basic CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.051 [-0.157, 0.092] | -0.091 [-0.138, 0.002] |
| rwm-brf - rule-betan | 0.037 [-0.123, 0.192] | -0.007 [-0.074, 0.081] |
| rwm-brf - rule-betan-over-li | 0.034 [-0.092, 0.167] | 0.008 [-0.046, 0.085] |

### Paired alarm differences — rwm-brf versus rules (95% basic CIs)

| first model minus rule | detection difference | Hanson incidence difference | unlabelled incidence difference |
|---|---|---|---|
| rwm-brf - rule-time-since-flattop | 0.125 [-0.021, 0.270] | 0.121 [-0.091, 0.333] | -0.826 [-0.894, -0.765] |
| rwm-brf - rule-betan | 0.104 [-0.053, 0.245] | 0.212 [0.061, 0.333] | -0.295 [-0.394, -0.197] |
| rwm-brf - rule-betan-over-li | 0.167 [0.015, 0.310] | 0.152 [0.000, 0.303] | -0.083 [-0.182, 0.015] |

### Split sensitivity — rwm-brf (fixed hyperparameters)

| model | fold seed | primary AUROC | high-beta conditional AUROC | detection rate | unlabelled alarm incidence |
|---|---|---|---|---|---|
| rwm-brf | 0 | 0.760 | 0.602 | 0.188 | 0.144 |
| rwm-brf | 1 | 0.787 | 0.651 | 0.167 | 0.280 |
| rwm-brf | 2 | 0.773 | 0.631 | 0.208 | 0.174 |
| rwm-brf | 3 | 0.806 | 0.696 | 0.271 | 0.242 |
| rwm-brf | 4 | 0.793 | 0.664 | 0.125 | 0.091 |

### Campaign sensitivity — rwm-brf, split 1 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.715 [0.646, 0.788] | 0.165 [0.118, 0.245] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.454 [0.337, 0.578] | 0.173 [0.111, 0.279] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.434 [0.332, 0.530] | 0.179 [0.120, 0.287] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.832 [0.763, 0.893] | 0.221 [0.163, 0.343] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.723 [0.599, 0.830] | 0.218 [0.154, 0.335] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.667 [0.536, 0.792] | 0.227 [0.163, 0.376] |

### Campaign sensitivity — rwm-brf, split 2 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.698 [0.631, 0.772] | 0.147 [0.116, 0.209] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.419 [0.335, 0.527] | 0.149 [0.110, 0.220] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.384 [0.288, 0.481] | 0.155 [0.116, 0.237] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.824 [0.755, 0.881] | 0.211 [0.151, 0.312] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.707 [0.576, 0.805] | 0.206 [0.140, 0.312] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.649 [0.518, 0.764] | 0.216 [0.147, 0.336] |

### Campaign sensitivity — rwm-brf, split 3 (95% shot CIs)

| group | slice mask | Hanson shots | positive slices | assumed-negative slices | prevalence | AUROC (95% shot CI) | AUPRC (95% shot CI) |
|---|---|---|---|---|---|---|---|
| 2014 | primary | 20 | 260 | 2280 | 0.102 | 0.735 [0.673, 0.801] | 0.190 [0.137, 0.256] |
| 2014 | high-beta conditional | 20 | 168 | 754 | 0.182 | 0.527 [0.395, 0.656] | 0.219 [0.144, 0.292] |
| 2014 | above-proxy conditional | 20 | 176 | 705 | 0.200 | 0.508 [0.383, 0.607] | 0.231 [0.149, 0.321] |
| 2018 | primary | 13 | 220 | 2975 | 0.069 | 0.848 [0.787, 0.902] | 0.313 [0.231, 0.437] |
| 2018 | high-beta conditional | 13 | 197 | 1597 | 0.110 | 0.747 [0.626, 0.846] | 0.318 [0.233, 0.444] |
| 2018 | above-proxy conditional | 13 | 189 | 1208 | 0.135 | 0.706 [0.577, 0.821] | 0.335 [0.252, 0.468] |

### Campaign sensitivity — rwm-brf, split 4 (95% shot CIs)

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
