# ELM benchmark: elm-elmo

ELM-O is a rule-based diagnostic-vote detector reimplemented from O'Shea et al., *Automatic identification of edge localized modes in the DIII-D tokamak*, APL Machine Learning 1, 026102 (2023), doi:10.1063/5.0134001. The public repository has no licence; its code was not copied.

It reads DENV2F/3F, FS02–FS04 and all 64 BES channels, interpolating diagnostics by chunked DCT upsampling at the BES rate. The lower-range 32 BES channels are doubled before averaging. BES exceeds 1 V; absolute first differences of each density/filterscope signal exceed its 0.997 quantile. Density and filterscope masks are widened by 100 microseconds on each side. A call requires either density chord, two of three filterscopes and BES. Gaps at most 100 microseconds are bridged; onset matching uses span starts, while the FS03 peak remains a separate span descriptor. Parameters are fixed across shots.

## Selected-window benchmark

Smith's 2,316 hand-labelled ELM windows on 211 shots have no review shot or run-day overlap. All windows have diagnostic-record coverage. One repeated acquisition axis uses an audited first-monotone-record derivative. All methods use the same repaired source. Smith was historically used for ELM-O; overlap with the paper's tuning events is unresolved. This does not establish independent ELM-O tuning.

One-to-one onset matching at ±2 and ±5 ms gives precision 0.958 [0.945, 0.970], recall 0.987 [0.971, 0.996], F1 0.972 [0.962, 0.981] (TP 2,286, FP 100, FN 30). Extra fragments count as unmatched predictions. Median signed timing error is 0.075 ms; full timing quantiles and shot-bootstrap intervals are recorded.

Region-overlap scoring gives precision 0.997 [0.995, 0.999], recall 0.986 [0.970, 0.995] (TP 2,283, FP 6, FN 33). This differs from onset matching. The paper reports 0.995/0.976 on 972 tuning ELMs; the prior local 0.997/0.980 used the old source record and overlap counting. Historical checks and results are preserved in the stream report and external documentation archive.

## Reviewed occupancy

BES coverage restricts this comparison to 73 review shots and 6,527 common 50 ms bins: AUROC 0.914 [0.869, 0.948], AUPRC 0.833 [0.755, 0.892], F1 0.841 [0.786, 0.885]. Hard calls use the fixed published setting; ranking curves use the recorded eta sweep. These occupancy targets are largely crowd spans and the clock seeded review; disagreements cannot adjudicate physical false alarms.

The DSM and legacy onset table were built on WPQH phases with breakthrough-ELM targets; Finding 1 and low DSM AUROCs partly reflect definition and domain shift (192721: 1 legacy bin versus 17 non-crowd review spans).

Canonical sources are `outputs/labeler/elm/smith/evaluation.json` (`methods.elm-elmo`, `reimplemented_elmo_overlap`, `protocol`) and `dsm/evaluation.json:sets.bes73.methods.elm-elmo`. Reproduce with `elm_smith_evaluate.py evaluate`, `elm_dsm_evaluate.py --rescore` and `elm_protocol.py` through the mandated labelmaker wrapper. Large signals and predictions stay under `$LABELER_ROOT`; current code/source hashes and valid/undefined bootstrap draws remain in the canonical records.
