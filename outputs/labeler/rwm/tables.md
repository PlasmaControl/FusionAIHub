### Piccione-style primary scores — all models

| model | AUROC (95% CI) | AUPRC (95% CI) |
|---|---|---|
| rwm-brf | 0.760 [0.706, 0.809] | 0.163 [0.123, 0.220] |
| rule-time-since-flattop | 0.759 [0.712, 0.815] | 0.228 [0.212, 0.307] |
| rule-betan | 0.707 [0.639, 0.772] | 0.166 [0.128, 0.246] |
| rule-betan-over-li | 0.723 [0.664, 0.784] | 0.159 [0.127, 0.232] |
| rule-rwm-candidates | 0.500 [0.500, 0.500] | 0.084 [0.070, 0.102] |

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

### Alarm counts — all models (n=1 targets)

| model | onsets warned | Hanson shots: unexplained alarm | unlabelled shots: any alarm |
|---|---|---|---|
| rwm-brf | 1/48 | 4/33 | 1/132 |
| rule-time-since-flattop | 3/48 | 25/33 | 130/132 |
| rule-betan | 2/48 | 9/33 | 52/132 |
| rule-betan-over-li | 2/48 | 4/33 | 25/132 |
| rule-rwm-candidates | 0/48 | 7/33 | 36/132 |

### Alarm rates — all models (95% shot CIs)

| model | onset detection | Hanson unexplained incidence | alarm incidence on unlabelled shots |
|---|---|---|---|
| rwm-brf | 0.021 [0.000, 0.064] | 0.121 [0.030, 0.242] | 0.008 [0.000, 0.023] |
| rule-time-since-flattop | 0.062 [0.000, 0.130] | 0.758 [0.606, 0.879] | 0.985 [0.962, 1.000] |
| rule-betan | 0.042 [0.000, 0.109] | 0.273 [0.151, 0.424] | 0.394 [0.318, 0.470] |
| rule-betan-over-li | 0.042 [0.000, 0.103] | 0.121 [0.030, 0.242] | 0.189 [0.121, 0.258] |
| rule-rwm-candidates | 0.000 [0.000, 0.000] | 0.212 [0.091, 0.364] | 0.273 [0.197, 0.348] |

### Warning times — all models (detected onsets only)

| model | median warning, ms (95% CI) | uniform-alarm reference | detection minus reference |
|---|---|---|---|
| rwm-brf | 90 [90, 90] | 0.017 [0.003, 0.038] | 0.003 [-0.031, 0.047] |
| rule-time-since-flattop | 130 [36, 145] | 0.089 [0.075, 0.102] | -0.026 [-0.088, 0.046] |
| rule-betan | 201 [16, 385] | 0.033 [0.012, 0.057] | 0.009 [-0.035, 0.069] |
| rule-betan-over-li | 122 [56, 188] | 0.025 [0.006, 0.049] | 0.017 [-0.028, 0.069] |
| rule-rwm-candidates | - | 0.021 [0.006, 0.039] | -0.021 [-0.039, -0.006] |

### Paired differences — rwm-brf versus rules, primary

| first model minus rule | AUROC difference (95% CI) | AUPRC difference (95% CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | 0.001 [-0.057, 0.052] | -0.065 [-0.145, -0.029] |
| rwm-brf - rule-betan | 0.053 [-0.034, 0.144] | -0.003 [-0.074, 0.058] |
| rwm-brf - rule-betan-over-li | 0.037 [-0.041, 0.111] | 0.004 [-0.063, 0.056] |
| rwm-brf - rule-rwm-candidates | 0.260 [0.206, 0.309] | 0.079 [0.043, 0.129] |

### Paired differences — rwm-brf versus rules, high-beta

| first model minus rule | AUROC difference (95% CI) | AUPRC difference (95% CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.040 [-0.140, 0.041] | -0.087 [-0.185, -0.046] |
| rwm-brf - rule-betan | 0.039 [-0.108, 0.180] | -0.008 [-0.103, 0.061] |
| rwm-brf - rule-betan-over-li | 0.009 [-0.105, 0.111] | 0.003 [-0.076, 0.055] |
| rwm-brf - rule-rwm-candidates | 0.102 [-0.009, 0.188] | 0.033 [-0.010, 0.087] |

### Paired differences — rwm-brf versus rules, above-proxy

| first model minus rule | AUROC difference (95% CI) | AUPRC difference (95% CI) |
|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.051 [-0.195, 0.055] | -0.091 [-0.185, -0.045] |
| rwm-brf - rule-betan | 0.037 [-0.119, 0.197] | -0.007 [-0.095, 0.060] |
| rwm-brf - rule-betan-over-li | 0.034 [-0.099, 0.160] | 0.008 [-0.068, 0.063] |
| rwm-brf - rule-rwm-candidates | 0.046 [-0.084, 0.142] | 0.016 [-0.030, 0.071] |

### Paired alarm differences — rwm-brf versus rules

| first model minus rule | detection difference | Hanson incidence difference | unlabelled incidence difference |
|---|---|---|---|
| rwm-brf - rule-time-since-flattop | -0.042 [-0.102, 0.000] | -0.636 [-0.818, -0.455] | -0.977 [-1.000, -0.947] |
| rwm-brf - rule-betan | -0.021 [-0.093, 0.044] | -0.152 [-0.303, 0.000] | -0.386 [-0.462, -0.303] |
| rwm-brf - rule-betan-over-li | -0.021 [-0.091, 0.045] | 0.000 [-0.121, 0.121] | -0.182 [-0.250, -0.114] |
| rwm-brf - rule-rwm-candidates | 0.021 [0.000, 0.064] | -0.091 [-0.242, 0.061] | -0.265 [-0.341, -0.189] |

### Split sensitivity — rwm-brf (fixed hyperparameters)

| model | fold seed | primary AUROC | high-beta AUROC | detection rate | unlabelled alarm incidence |
|---|---|---|---|---|---|
| rwm-brf | 0 | 0.760 | 0.602 | 0.021 | 0.008 |
| rwm-brf | 1 | 0.787 | 0.651 | 0.042 | 0.182 |
| rwm-brf | 2 | 0.773 | 0.631 | 0.000 | 0.015 |
| rwm-brf | 3 | 0.806 | 0.696 | 0.042 | 0.030 |
| rwm-brf | 4 | 0.793 | 0.664 | 0.000 | 0.000 |
