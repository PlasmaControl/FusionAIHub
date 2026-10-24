# Raising the challenge score: published methods, priced on the merit function

Status: **survey and a development-data re-pricing, 2026-10-24. Plan items 1 and 2 are
implemented and scored (`elm_inst2.md`, round 2): `elm-inst2-sized` 1,635.7, against
2,206.0 for `elm-inst` typed 2.** It
follows `elm_train_methods.md` (2026-10-09), which covers training a detector for ELM
trains. This one asks which published methods for event, spike and anomaly detection move
the challenge merit function, one component at a time.

## Where the points go

`elm-inst` typed as small ELM (type 2) scores 2,206.0, the best so far. This was scored
with the challenge merit function at commit b71f991 and is not an official result.

| Component | Points | What drives it |
|---|---|---|
| Timing | 620 | Mostly the length of missed events: no detection gives 698, perfect timing 6.5 |
| Classification | 371 | About 74 found ELMs that the key calls large, typed small (5.0 each) |
| False negatives and false positives | 1,215 | 257 missed events; the 5 false detections cost 15 |

This breakdown comes from the key. It is fine for deciding which component to work on,
but thresholds and types must be chosen on development data.

## What one detection is worth

The values come from the merit function's published matrix (`miscounting_penalties`,
detection row against key column). They are multiplied by 10, as for an event that counts
at all three uncertainty widths.

| Type reported | Large ELM found | Small ELM found | False detection |
|---|---|---|---|
| 5 (all our submissions until 2026-10-24) | +5.0 | +0.5 | -9.5 |
| 2, small ELM | +7.0 | +4.0 | -3.0 |
| 1, large ELM | +12.0 | +1.0 | -10.0 |

The timing term adds about ±1.1 for each detection, its length in ms. Below the
break-even precision, adding detections loses points:

- **Type 5:** 0.63 for large ELMs and 0.87 for small ones. Under type 5 the merit
  function is saying "do not report small ELMs".
- **Type 2:** 0.34 for large ELMs and 0.45 for small ones.

Every threshold so far was chosen under type-5 prices. That is why elm-inst2 was served
at 0.90, and why its extra train ELMs did not pay.

## Development data, re-priced at type 2

These are the out-of-fold counts already in `outputs/labeler/elm/inst2/evaluation.json`
(`tradeoff_pooled_out_of_fold`), priced with the type-2 row instead of type 5. The two
rules are the ones defined in `rule_gains`. No key data is used.

| Rule | elm-inst2: best threshold, points saved | elm-inst: best threshold, points saved |
|---|---|---|
| Type 5, conservative (the rule that was served) | 0.90, 7,982 | 0.65, 10,201 |
| Type 2, conservative | **0.50, 63,241** | 0.30, 57,422 |
| Type 2, as trained | **0.40, 108,841** | 0.20, 102,541 |

Under type-2 prices, elm-inst2 beats elm-inst on development data under both rules. The
best threshold also moves near elm-inst2-alt's 0.53. The type-2 curve for elm-inst2 is
flat from 0.45 to 0.55 (within 1,300 of its best).

## What the literature offers, by component

### Detecting single spikes, for recall (the 1,215 and most of the 620)

- **Seismic phase pickers.** SeisBench compared six deep pickers on eight datasets
  ([Münchmeyer et al. 2022](https://export.arxiv.org/abs/2110.13671)). EQTransformer,
  PhaseNet and GPD were best; EQTransformer won on all three tasks in-domain. PhaseNet
  and EQTransformer predict a Gaussian onset heatmap, as elm-inst's CenterNet head does.
- **Musical onset detection.** A CNN followed by peak picking
  ([Schlüter & Böck 2014](https://research.jku.at/en/publications/improved-musical-onset-detection-with-convolutional-neural-networ/))
  was best in MIREX. It is the same design.
- **CenterNet itself.** "Objects as Points"
  ([Zhou et al. 2019](https://arxiv.org/pdf/1904.07850)) loses fewer than 0.1 % of COCO
  objects to centres that collide. Crowd ELMs are a median 6 ms apart, against our 3 ms
  separation, so collisions are not our failure either.

**Verdict:** the architecture is already what the single-spike fields use. The recall
gap is the labels and training (synthetic small-ELM trains and filtered self-training,
`elm_train_methods.md` items 2 and 3), plus a threshold priced at type 2.

### Anomaly detection (asked about; not a fit)

- **TSB-AD** ([Liu & Paparrizos, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/c3f3c690b7a99fba16d0efd35cb83b2c-Paper-Datasets_and_Benchmarks_Track.pdf)):
  40 methods on 1,070 series. Simple statistical methods often beat neural networks;
  foundation models do well on point anomalies.
- **Wu & Keogh** ([2020](https://arxiv.org/pdf/2009.13807)): the popular anomaly
  benchmarks are flawed, and the reported progress may be illusory.

Anomaly detection asks "is this unusual?". In an ELMy H-mode an ELM is the usual thing:
hundreds of labelled, repeated events. A supervised picker with thousands of labels is the
stronger tool. The time-series foundation models
([MOMENT, ICML 2024](https://arxiv.org/html/2402.03885v3)) report anomaly results only
in low-supervision settings.

### Typing each event (the 371)

- **DOSED** ([Chambon et al. 2018](https://arxiv.org/pdf/1812.04079)) and
  **ActionFormer** ([Zhang et al., ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf))
  give every event a class score as well as its location. DOSED covers sleep spindles,
  K-complexes and arousals; ActionFormer classifies every moment and regresses the
  boundaries. For us: a two-class large/small head at each heatmap peak.
- **Bayes decision under the cost matrix** ([Elkan 2001](https://mlanthology.org/ijcai/2001/elkan2001ijcai-foundations)),
  the same idea as minimum-Bayes-risk decoding in speech. Report the type with the lowest
  expected cost under the model's probabilities. From the matrix, with probabilities
  p0 (no ELM), pL (large) and pS (small):
  - report large (type 1) instead of small when 0.5 pL > 0.3 pS + 0.7 p0, that is
    P(large | ELM) > 0.375 for a certain ELM;
  - report at all, as type 2, when 0.3 p0 < 0.7 pL + 0.4 pS.
- **Temperature scaling** ([Guo et al., ICML 2017](https://proceedings.mlr.press/v70/guo17a.html)):
  those rules need calibrated probabilities, which modern networks are not.
- **Fusion.** ELM type classification exists only per window, not per event
  (HGTS-Former, [arXiv 2508.02411](https://arxiv.org/pdf/2508.02411): Type-I/Type-III
  on EAST).

Labels: no development data marks large against small. A proxy is Smith's isolated ELMs
as large (median D-alpha rise 0.94 decades, quartiles 0.79-1.09) and crowd ELMs as small
(0.34, quartiles 0.21-0.57, `elm_train_methods.md`). The quartiles do not overlap.
Whether the key's "large" means the same thing is the risk.

### Event extent (the matched part of the 620)

- **Boundary distributions instead of one offset:** TriDet
  ([Shi et al., CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Shi_TriDet_Temporal_Action_Detection_With_Relative_Boundary_Modeling_CVPR_2023_paper.html))
  and TBT-Former ([arXiv 2512.01298](https://arxiv.org/html/2512.01298v1)).
- **Sound event bounding boxes** ([Ebbers et al., Interspeech 2024](https://www.arxiv.org/pdf/2406.04212)):
  turn frame posteriors into onset, offset and confidence by change detection, then
  threshold the box confidence. Boundaries no longer move with the threshold. Raised the
  DCASE 2023 state of the art from 0.644 to 0.686 PSDS1.

This is low priority. We serve a fixed 1.1 ms length, and timing on matched events is
small next to the missed events' time, which falls as recall rises. We also have no end
labels except the clock's.

## Plan (ranked by points per effort)

1. **Price decisions at type 2** (no retraining).
   - `spikes.merit_gain` takes the matrix row; `rule_gains` and the fold thresholds use it.
   - Rerun `evaluate` (CPU, seconds).
   - The development choice is elm-inst2 at about 0.50, typed 2.
   - Scoring it needs the owner to lift `round4/elm/inst/inst2/key.lock`.
2. **A large/small head and Bayes typing.**
   - A per-peak two-class head, or a small calibrated model on peak features (D-alpha
     rise in decades, density and W_mhd drop, gaps to neighbours), trained on the proxy
     labels above.
   - Temperature-scaled on a held-out fold, typed by the rule above.
   - Ceiling: the 371 classification points.
   - Risk: the key's size criterion. Typing a found ELM large instead of small gains 5.0
     when the key calls it large and loses 3.0 when it calls it small. It pays when more
     than 37.5 % of the ELMs typed large are large by the key's criterion.
3. **Recall in trains at the new operating point.** Synthetic small-ELM superposition and
   two rounds of filtered self-training (`elm_train_methods.md` items 2 and 3). A small ELM
   found now earns about +5 and a false one costs about -4, so this is the largest pool.
4. **Event extent** (TriDet-style end head, or SEBB change detection), only after 3.

## Not recommended

- **Unsupervised anomaly detectors and time-series foundation models as the detector**,
  for the reasons above.
- **DETR-style set prediction:** it removes peak collisions, which are not our failure.
- **Hedging with types the detector does not believe:** "small other" (4), or "hidden
  pellet" (7, which costs 0 as a false detection). The matrix rewards both for uncertain
  detections. That would game the metric, not detect ELMs.

## Caveats

- The key has now been looked at several times: elm-inst, elm-inst-alt, the two elm-inst2
  decodings, two retypings, and the key fed back as a submission. A further score is a
  development number on public data, not a held-out result.
- The type-2 re-pricing uses the same counts and the same two rules as `evaluate`. It
  inherits their assumptions: Smith's ELMs and his ELMs outside crowds count as large,
  and every alarm in reviewed absent time counts as false.
- The sources were opened through search summaries; the numbers quoted are the papers'
  own.
