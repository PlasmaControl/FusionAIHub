# Stage 5 scientific review: historical comparison and paper context

Reviewer: ML scientist / AI conference reviewer. Date: 2026-10-01.

**Stage readiness: 8/10. Proceed to artifact and manuscript handoff.** The
scientifically valid outcome is that a fair historical comparison has inadequate
common L support. It is useful as a coverage and threshold audit, not evidence
that the new model is better. Honest negative outcomes meet a readiness gate;
the gate must not force a superiority claim.

**Current full-paper ICML-style assessment: approximately 5/10 as-is** on the
project's internal 10-point scale. This is separate from stage readiness and
does not mean the confinement work should be discarded. The extension improves
reproducibility and claim discipline but leaves major independent-validation,
aligned-cohort, and release gaps open.

## Evidence inspected and verified

I inspected `src/labeler/confinement/comparison.py`, its tests, the historical
frame probability/observability code, `legacy_comparison/comparison.json`,
`matched_predictions.csv`, the existing freeze/evaluation, measured report,
manuscript appendix and generated confinement table. Artifacts are under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/confinement/v1`.

All comparison input hashes match, including old/new model records, thresholds,
evaluation, predictions and historical feature files. Matched-output and
comparison-code hashes match. Matched bins are unique, exactly 50 ms long,
and retain the new frozen labels/probabilities. No old training or validation
shot enters the comparison. Historical probability pooling requires all five
observed 10 ms frames, with missing/partial bins excluded; its cached bin
probabilities are checked for consistency rather than silently averaged.

Fresh focused validation:
`OMP_NUM_THREADS=4 pixi run --frozen -e shot-design-cpu python -m pytest tests/labeler/test_confinement_comparison.py -q -W error`
returned **2 passed**. Independent whole-shot resampling reproduces 1,427 usable
two-class draws of 2,000. The primary freeze and H2 failure remain unchanged.
This inspection did not refit either model or retune either threshold.

## What the intersection establishes

The comparison correctly follows this contraction:

| Domain | Shots | Bins |
| --- | ---: | ---: |
| New diagnostic-eligible test | 28 | 1,403 |
| Excluding old training and validation | 8 | 355 |
| Historical feature files available | 4 | — |
| Complete observable common bins | 3 | 122 |

The final shots are 185915 (30 H bins), 187036 (76 H bins), and 189602
(15 H bins, one L bin). All three were in the previously exposed historical
test set. They were not old fitting/validation shots, but that fact alone does
not make them a fresh confirmatory test.

The new frozen model classifies all 122 bins correctly. The old frozen threshold
classifies all as H. On the only L bin, [1800,1850) ms of 189602, new H-score
0.936067 is below its threshold 0.963534; old H-score 0.446895 is above its
threshold 0.33. Both models have AUROC 1 on the matched cohort. The observed
decision difference is therefore a threshold outcome on one negative, not
demonstrated ranking superiority.

The new model's perfect bootstrap intervals resample this single observed
negative. They do not quantify performance on new L shots. Roughly 29% of
requested draws contain no L and are omitted; the recorded intervals condition
on the other 1,427 draws. Even a nominal paired interval away from zero cannot
repair this lack of independent negative support or prior test exposure.

The saved audit usefully identifies where a comparison fails: historical
training overlap, absent feature files, missing whole windows, and the scarcity
of independently observable L phases. Different model inputs and temporal
context add another reason to avoid a causal architecture or modality claim.
Current prose handles these limits correctly and does not overwrite the older
headline numbers or replace the new primary definition.

## Overall paper strengths and remaining limits

The confinement extension makes several concrete contributions to the paper:

- It demonstrates that donor count is not independent reader count, through
  exactly matching workbook/CSV intervals and consistent BES overlaps.
- It exercises exact coverage, uncertainty and conflict handling on real data,
  rather than only stating a schema. Test-only reservations and physical-data
  exclusions are reproducible and auditable.
- It provides a standard feature-based ML baseline with realistic missingness,
  a separate four-class experiment, and all failed bars retained. H1 passes;
  H2 still fails. Point scores are reported beside shot-level uncertainty.
- It produces a cross-source QH/onset candidate queue while showing that
  revised aligned ELM coverage is too sparse to adjudicate it. This explains
  why aligned independent review is needed without fabricating a completed
  cross-phenomenon result.

These strengthen the application and resource case but do not by themselves
make the full manuscript a strong ICML submission. The largest remaining gates
in `dev/label_paper/ICML_SCORE_REQUIREMENTS.md` still apply:

1. **Independent reference quality:** single-expert AE relabels and assisted
   ELM spans are not multi-reader gold. Agreement before adjudication and
   sufficiently diverse positive/negative shots are still needed.
2. **Aligned cohort evidence:** the planned cohort is not yet an independently
   validated, jointly assessed resource across the phenomena. A real result
   requiring aligned labels, with adequate joint denominators, remains open.
3. **External usability:** a release outsiders can download and score, including
   usable inputs or a public-input companion, is not established here.
4. **Confirmatory testing:** an internally recorded freeze makes this extension
   reproducible, but is not a public prospective confirmation against new gold.
5. **General ML evidence:** the feature-based GBM improves baseline coverage;
   it does not replace broader standard architectures, a third supervision
   study, or a demonstrated evaluation lesson beyond these selected fusion
   references. The current spectral test support is zero.

The abstract currently says the resource provides “a ground truth,” and the
introduction speaks of “multiple expert reviews” and a created uniform resource.
Those broad statements should be narrowed to the completed provenance tiers and
current review evidence, consistent with the limitations. This is a concrete
claim-hygiene fix, but wording alone cannot close the scientific gates above.
State completed observations, suggestions/imports, and planned independent gold
separately. Keep the added confinement result exploratory in the main pointer
and appendix.

The 5/10 overall assessment is an honest current-evidence judgment, not a
prediction of acceptance or a requirement to manufacture 8/10 evidence during
handoff. The appropriate next research investment is independent aligned review
and external reproducibility, rather than more retuning of the opened test or
an attempt to rescue the one-negative historical comparison.
