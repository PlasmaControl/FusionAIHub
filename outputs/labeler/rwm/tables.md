<!-- selection balance (shots.json) -->
| campaign | set | shots | mean flat-top beta_N p95 | mean beta_N/l_i p95 |
|---|---|---|---|---|
| 2014 | Hanson shots | 20 | 3.25 | 5.07 |
| 2014 | chosen comparison | 80 | 3.31 | 4.43 |
| 2014 | pool, not chosen | 22 | 4.08 | 4.65 |
| 2018 | Hanson shots | 13 | 3.05 | 4.98 |
| 2018 | chosen comparison | 52 | 3.13 | 5.11 |
| 2018 | pool, not chosen | 20 | 3.36 | 5.63 |

<!-- growth (growth.json) -->
| quantity (48 n=1 onsets) | first quartile | median | third quartile |
|---|---|---|---|
| largest 20 ms growth rate of N1RMS, -150 to +30 ms from the onset (per s) | 91 | 112 | 174 |
| e-folding time at that rate (ms) | 5.8 | 9.0 | 11.0 |
| time of the maximum growth relative to the onset (ms) | -103.5 | -39.0 | 0.0 |
| beta_N 100 ms before the onset | 2.55 | 2.74 | 3.01 |
| beta_N at the onset | 2.55 | 2.68 | 2.85 |
| beta_N 40 ms after the onset | 1.09 | 1.93 | 2.36 |

<!-- slice scores, Hanson shots (evaluation.json) -->
| model | AUROC | AUPRC | F1 | TPR | FPR |
|---|---|---|---|---|---|
| rwm-brf | 0.798 [0.740, 0.845] | 0.084 [0.061, 0.116] | 0.149 [0.112, 0.193] | 0.854 [0.768, 0.932] | 0.333 [0.282, 0.388] |
| rwm-nnpu | 0.794 [0.743, 0.845] | 0.083 [0.060, 0.134] | 0.136 [0.104, 0.176] | 0.767 [0.655, 0.874] | 0.329 [0.268, 0.394] |
| rule-betan-over-li | 0.752 [0.688, 0.815] | 0.074 [0.052, 0.115] | 0.141 [0.103, 0.183] | 0.735 [0.602, 0.847] | 0.302 [0.245, 0.363] |
| rule-rwm-candidates | 0.498 [0.495, 0.500] | 0.034 [0.026, 0.041] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.004 [0.001, 0.009] |

<!-- slice scores with the comparison shots as negatives -->
| model | AUROC | AUPRC | FPR |
|---|---|---|---|
| rwm-brf | 0.899 [0.866, 0.928] | 0.047 [0.032, 0.069] | 0.186 [0.161, 0.213] |
| rwm-nnpu | 0.900 [0.867, 0.933] | 0.052 [0.035, 0.091] | 0.172 [0.146, 0.199] |
| rule-betan-over-li | 0.784 [0.718, 0.843] | 0.017 [0.011, 0.027] | 0.265 [0.240, 0.291] |
| rule-rwm-candidates | 0.495 [0.493, 0.498] | 0.006 [0.005, 0.007] | 0.009 [0.005, 0.015] |

<!-- per-shot alarm scores -->
| model | onsets warned | detection rate | detection by chance | median warning (ms) | Hanson shots with a false alarm | rate | per shot | comparison shots with an alarm | rate | per shot |
|---|---|---|---|---|---|---|---|---|---|---|
| rwm-brf | 17 of 48 | 0.354 [0.217, 0.500] | 0.328 | 330 [276, 367] | 21 of 33 | 0.636 [0.485, 0.788] | 3.03 [1.91, 4.30] | 21 of 132 | 0.159 [0.098, 0.227] | 0.83 [0.36, 1.36] |
| rwm-nnpu | 20 of 48 | 0.417 [0.296, 0.553] | 0.349 | 203 [143, 297] | 22 of 33 | 0.667 [0.514, 0.818] | 3.27 [2.09, 4.46] | 15 of 132 | 0.114 [0.061, 0.174] | 0.59 [0.28, 0.98] |
| rule-betan-over-li | 1 of 48 | 0.021 [0.000, 0.070] | 0.029 | 188 [188, 188] | 5 of 33 | 0.152 [0.030, 0.273] | 0.24 [0.06, 0.45] | 17 of 132 | 0.129 [0.076, 0.182] | 0.30 [0.15, 0.48] |
| rule-rwm-candidates | 0 of 48 | 0.000 [0.000, 0.000] | 0.021 | - | 7 of 33 | 0.212 [0.091, 0.364] | 0.42 [0.12, 0.79] | 36 of 132 | 0.273 [0.197, 0.348] | 0.80 [0.50, 1.14] |

<!-- legacy against Tokamak-SI -->
| setting | slice AUROC | slice TPR | slice FPR | unstable shots detected | false alarms |
|---|---|---|---|---|---|
| Legacy: Piccione et al. 2022, NSTX, 28 test shots | 0.918 | 0.924 | 0.214 | 10 of 11 unstable shots | 2 of 17 stable shots |
| Tokamak-SI: rwm-brf, DIII-D, shot-grouped CV | 0.798 [0.740, 0.845] | 0.854 [0.768, 0.932] | 0.333 [0.282, 0.388] | 12 of 30 shots with an n=1 onset (17 of 48 onsets) | 21 of 33 Hanson shots; 21 of 132 unlabelled comparison shots (upper bound) |

<!-- rwm-brf feature and training ablations -->
| configuration | positive slices | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|---|
| rwm-brf | 480 | 0.798 [0.740, 0.845] | 0.084 [0.061, 0.116] | 0.354 [0.217, 0.500] | 0.636 [0.485, 0.788] | 0.159 [0.098, 0.227] |
| rwm-brf-hanson-only | 480 | 0.805 [0.756, 0.852] | 0.089 [0.066, 0.124] | 0.208 [0.095, 0.341] | 0.394 [0.242, 0.546] | 0.205 [0.144, 0.273] |
| rwm-brf-equilibrium-only | 480 | 0.767 [0.710, 0.818] | 0.068 [0.049, 0.096] | 0.271 [0.140, 0.417] | 0.697 [0.545, 0.848] | 0.136 [0.076, 0.197] |
| rwm-brf-magnetics-only | 480 | 0.688 [0.639, 0.740] | 0.059 [0.043, 0.086] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] | 0.000 [0.000, 0.000] |
| rwm-brf-no-rotation | 480 | 0.799 [0.743, 0.847] | 0.085 [0.061, 0.119] | 0.375 [0.208, 0.559] | 0.727 [0.576, 0.879] | 0.242 [0.174, 0.318] |
| rwm-brf-with-locked-mode | 480 | 0.799 [0.744, 0.845] | 0.085 [0.063, 0.120] | 0.438 [0.286, 0.583] | 0.788 [0.636, 0.909] | 0.205 [0.136, 0.273] |

<!-- rwm-brf target variants -->
| configuration | positive slices | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|---|
| rwm-brf-horizon-50 | 240 | 0.792 [0.738, 0.840] | 0.042 [0.031, 0.060] | 0.354 [0.229, 0.489] | 0.636 [0.455, 0.788] | 0.167 [0.106, 0.227] |
| rwm-brf | 480 | 0.798 [0.740, 0.845] | 0.084 [0.061, 0.116] | 0.354 [0.217, 0.500] | 0.636 [0.485, 0.788] | 0.159 [0.098, 0.227] |
| rwm-brf-horizon-200 | 958 | 0.806 [0.750, 0.854] | 0.170 [0.124, 0.241] | 0.396 [0.265, 0.535] | 0.758 [0.606, 0.909] | 0.197 [0.129, 0.265] |
| rwm-brf-all-modes | 540 | 0.800 [0.748, 0.844] | 0.088 [0.066, 0.126] | 0.333 [0.216, 0.471] | 0.636 [0.485, 0.788] | 0.189 [0.129, 0.258] |

<!-- rwm-nnpu variants -->
| configuration | positive slices | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|---|
| rwm-nnpu-prior-x0.5 | 480 | 0.791 [0.738, 0.842] | 0.083 [0.059, 0.140] | 0.333 [0.217, 0.463] | 0.636 [0.455, 0.788] | 0.098 [0.053, 0.152] |
| rwm-nnpu | 480 | 0.794 [0.743, 0.845] | 0.083 [0.060, 0.134] | 0.417 [0.296, 0.553] | 0.667 [0.514, 0.818] | 0.114 [0.061, 0.174] |
| rwm-nnpu-prior-x2 | 480 | 0.792 [0.738, 0.844] | 0.083 [0.060, 0.130] | 0.333 [0.234, 0.442] | 0.636 [0.484, 0.788] | 0.106 [0.061, 0.159] |

<!-- rwm-brf split-seed sensitivity -->
| fold seed | AUROC | AUPRC | detection rate | comparison false-alarm shot rate |
|---|---|---|---|---|
| 0 | 0.798 | 0.084 | 0.354 | 0.159 |
| 1 | 0.807 | 0.092 | 0.438 | 0.227 |
| 2 | 0.790 | 0.088 | 0.333 | 0.318 |
| 3 | 0.800 | 0.082 | 0.229 | 0.258 |
| 4 | 0.796 | 0.083 | 0.417 | 0.227 |

<!-- paired differences on shared shot resamples -->
| first - second | AUROC | AUPRC | detection rate | Hanson false-alarm shot rate | comparison false-alarm shot rate |
|---|---|---|---|---|---|
| rwm-brf - rule-betan-over-li | 0.046 [-0.018, 0.114] | 0.010 [-0.016, 0.032] | 0.333 [0.184, 0.491] | 0.485 [0.303, 0.667] | 0.030 [-0.045, 0.106] |
| rwm-nnpu - rule-betan-over-li | 0.041 [-0.027, 0.108] | 0.009 [-0.018, 0.045] | 0.396 [0.263, 0.546] | 0.515 [0.333, 0.697] | -0.015 [-0.091, 0.068] |
| rwm-brf - rwm-nnpu | 0.004 [-0.023, 0.028] | 0.001 [-0.027, 0.011] | -0.062 [-0.227, 0.109] | -0.030 [-0.152, 0.091] | 0.045 [-0.008, 0.098] |
| rwm-brf - rwm-brf-hanson-only | -0.007 [-0.040, 0.022] | -0.004 [-0.029, 0.018] | 0.146 [-0.045, 0.333] | 0.242 [0.030, 0.455] | -0.045 [-0.129, 0.038] |
| rwm-brf - rwm-brf-equilibrium-only | 0.031 [0.008, 0.056] | 0.016 [0.004, 0.031] | 0.083 [-0.085, 0.264] | -0.061 [-0.182, 0.061] | 0.023 [-0.030, 0.076] |
| rwm-brf - rwm-brf-no-rotation | -0.002 [-0.009, 0.006] | -0.000 [-0.010, 0.007] | -0.021 [-0.170, 0.135] | -0.091 [-0.212, 0.000] | -0.083 [-0.136, -0.038] |

<!-- rwm-brf by campaign -->
| campaign | Hanson / comparison shots | positive slices | AUROC | AUPRC | onsets warned | Hanson shots with a false alarm | comparison shots with an alarm |
|---|---|---|---|---|---|---|---|
| 2014 | 20 / 80 | 260 | 0.797 | 0.089 | 10 of 26 | 14 of 20 | 14 of 80 |
| 2018 | 13 / 52 | 220 | 0.800 | 0.080 | 7 of 22 | 7 of 13 | 7 of 52 |

<!-- rwm-brf scored time and alarm rate -->
| shots | count | mean scored span (s) | shortest | longest | total (s) | alarms | alarms per scored second |
|---|---|---|---|---|---|---|---|
| hanson | 33 | 4.51 | 1.35 | 5.89 | 148.9 | 136 | 0.91 |
| comparison | 132 | 5.14 | 1.56 | 6.87 | 678.7 | 109 | 0.16 |

<!-- rwm-nnpu scored time and alarm rate -->
| shots | count | mean scored span (s) | shortest | longest | total (s) | alarms | alarms per scored second |
|---|---|---|---|---|---|---|---|
| hanson | 33 | 4.51 | 1.35 | 5.89 | 148.9 | 168 | 1.13 |
| comparison | 132 | 5.14 | 1.56 | 6.87 | 678.7 | 78 | 0.11 |

<!-- single-feature AUROC -->
| feature | AUROC | unstable when | fraction of slices with a value |
|---|---|---|---|
| betan | 0.715 | higher | 0.987 |
| li | 0.777 | lower | 0.987 |
| q95 | 0.542 | higher | 0.987 |
| qmin | 0.656 | higher | 0.987 |
| wmhd_mj | 0.704 | higher | 0.987 |
| betan_over_li | 0.752 | higher | 0.987 |
| betan_minus_4li | 0.754 | higher | 0.987 |
| ip_ma | 0.550 | higher | 1.000 |
| n1rms_g | 0.553 | higher | 1.000 |
| n1rms_max_g | 0.530 | higher | 1.000 |
| n1rms_growth_per_s | 0.512 | lower | 1.000 |
| n2rms_g | 0.565 | higher | 1.000 |
| rot_core_khz | 0.680 | higher | 0.854 |
| rot_mid_khz | 0.631 | higher | 0.854 |
| lock_v | 0.599 | lower | 1.000 |

<!-- rwm-brf per Hanson shot -->
| shot | campaign | n=1 onsets | warning per onset (ms; - missed) | false |
|---|---|---|---|---|
| 156785 | 2014 | 1 | - | 5 |
| 156786 | 2014 | 2 | 392, - | 0 |
| 156787 | 2014 | 2 | -, 237 | 0 |
| 156790 | 2014 | 1 | 300 | 2 |
| 156791 | 2014 | 1 | - | 0 |
| 156792 | 2014 | 2 | 348, - | 2 |
| 156793 | 2014 | 2 | 166, 395 | 5 |
| 156794 | 2014 | 2 | -, - | 14 |
| 156795 | 2014 | 2 | 276, 363 | 1 |
| 156796 | 2014 | 1 | - | 2 |
| 156797 | 2014 | 2 | -, - | 0 |
| 158012 | 2014 | 1 | - | 14 |
| 158013 | 2014 | 2 | 340, - | 5 |
| 158014 | 2014 | 1 | 367 | 4 |
| 158015 | 2014 | 0 | - | 0 |
| 158018 | 2014 | 1 | - | 3 |
| 158019 | 2014 | 1 | - | 4 |
| 158021 | 2014 | 1 | - | 0 |
| 158022 | 2014 | 1 | - | 5 |
| 158023 | 2014 | 0 | - | 7 |
| 176067 | 2018 | 1 | - | 0 |
| 176068 | 2018 | 4 | -, 112, 390, 383 | 1 |
| 176069 | 2018 | 3 | 50, -, 330 | 3 |
| 176070 | 2018 | 2 | -, - | 4 |
| 176071 | 2018 | 1 | 281 | 1 |
| 176074 | 2018 | 2 | -, - | 9 |
| 176077 | 2018 | 1 | - | 6 |
| 176078 | 2018 | 2 | 195, - | 3 |
| 176085 | 2018 | 1 | - | 0 |
| 176087 | 2018 | 1 | - | 0 |
| 176088 | 2018 | 3 | -, -, - | 0 |
| 176089 | 2018 | 1 | - | 0 |
| 176092 | 2018 | 0 | - | 0 |

<!-- rwm-nnpu per Hanson shot -->
| shot | campaign | n=1 onsets | warning per onset (ms; - missed) | false |
|---|---|---|---|---|
| 156785 | 2014 | 1 | - | 7 |
| 156786 | 2014 | 2 | -, 228 | 1 |
| 156787 | 2014 | 2 | -, 297 | 0 |
| 156790 | 2014 | 1 | 370 | 2 |
| 156791 | 2014 | 1 | - | 0 |
| 156792 | 2014 | 2 | 88, - | 0 |
| 156793 | 2014 | 2 | -, 265 | 5 |
| 156794 | 2014 | 2 | -, 56 | 8 |
| 156795 | 2014 | 2 | 166, 143 | 1 |
| 156796 | 2014 | 1 | 380 | 3 |
| 156797 | 2014 | 2 | -, 38 | 0 |
| 158012 | 2014 | 1 | 296 | 9 |
| 158013 | 2014 | 2 | 170, - | 11 |
| 158014 | 2014 | 1 | 367 | 2 |
| 158015 | 2014 | 0 | - | 7 |
| 158018 | 2014 | 1 | - | 1 |
| 158019 | 2014 | 1 | 378 | 9 |
| 158021 | 2014 | 1 | - | 0 |
| 158022 | 2014 | 1 | - | 8 |
| 158023 | 2014 | 0 | - | 3 |
| 176067 | 2018 | 1 | 178 | 7 |
| 176068 | 2018 | 4 | -, -, -, 133 | 5 |
| 176069 | 2018 | 3 | -, -, - | 0 |
| 176070 | 2018 | 2 | -, - | 3 |
| 176071 | 2018 | 1 | 341 | 1 |
| 176074 | 2018 | 2 | -, 101 | 6 |
| 176077 | 2018 | 1 | - | 8 |
| 176078 | 2018 | 2 | 105, 259 | 1 |
| 176085 | 2018 | 1 | - | 0 |
| 176087 | 2018 | 1 | - | 0 |
| 176088 | 2018 | 3 | -, -, - | 0 |
| 176089 | 2018 | 1 | - | 0 |
| 176092 | 2018 | 0 | - | 0 |

