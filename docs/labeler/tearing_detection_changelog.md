# Tearing-mode label rule: changelog

The methods document states only the current rule. This appendix keeps what each round changed and the development-cohort numbers it moved. Round 4 figures come from `benchmark/sources/rule_diagnostics_fix4.json`, round 3 from `rule_diagnostics_fix3.json`.

## Round 4

| Change | Before | After |
|---|---:|---:|
| Lock confirmation: a step at each candidate time (median 20 to 120 ms after, 5 above the median 200 to 20 ms before); confirmed cohort locks | 15 | 12 |
| False-confirmation rate, previous test against the step test, at lags of 300 / 1000 / 2000 ms and lags drawn from the interval durations (realized median 430 ms) | 0.7% / 2.0% / 5.7% / 2.2% | 0.2% / 0.2% / 0.1% / 0.4% |
| Harmonic veto recalibrated on the bins that fit n = 2: level (n = 2 seeds removed of 86) | 0.72 (10) | 0.57 (9) |
| Inside-the-onset-window count, Seo / survival (the old flag compared the reference onset with the interval end; the count is not widened) | 11 of 12 / 16 of 16 | 2 of 12 / 8 of 16 |
| Duplicate rows in `tm_interval.csv` | 13 | 0 |
| Uncertain share of observable time, catalog window / flat-top (pooled) | 34.9% / 40.3% | 34.0% / 39.2% |

## Round 3

| Change | Before | After |
|---|---:|---:|
| Weak tracks released at the weak floor, one weak screen over ramp-up and flat-top: uncertain share of observable flat-top time (pooled) | 49.6% | 40.3% |
| Lock confirmation by a rise over the pre-onset median: false confirmations at a 300 ms lag (absolute level against relative rise) | 10.4% | 1.2% |
| Intervals ending in a confirmed lock | 18 | 15 |
| n = 2 frequency cap scaled with n (30 to 60 kHz): n = 2 seeds it removes | 10 | 9 |
| Harmonic veto calibrated on the bins that fit n = 1: level (seeds removed) | 0.57 (9) | 0.72 (10) |

Round 3's harmonic level was calibrated on the bins that fit n = 1, which are not the harmonic bins (see the harmonic veto in the methods document); round 4 replaced it. The round 4 false-confirmation rows re-measure the previous test and the step test on the same draws (absent stretches of at least 320 ms, 20 draws per shot and lag, seed 0); round 3's figures used stretches of at least 700 ms, so the two rounds' numbers for the previous test differ.
