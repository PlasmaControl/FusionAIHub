# Stage 2 scientific review: confinement–ELM consistency

Reviewer: ML scientist / AI conference reviewer. Date: 2026-10-01.

**Rating: 8/10. Proceed.** The outcome is scientifically ready as an exploratory
coverage and legacy-consistency audit. It does not establish an association
between revised ELM labels and confinement regimes, independently validate QH
physics, or identify which source is wrong. No blocking issue remains within
that scope after the reporting corrections below.

## Inspection and independent verification

I inspected `src/labeler/confinement/elm.py`, `test_confinement_elm.py`, the
Confinement–ELM section of `confinement_analysis.md`, all study CSVs and
`study.json` under
`/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/confinement/v1/elm_study`, and the refreshed
`figures/elm_consistency.png`. I also read the original onset archive through
`read_elm` and independently recomputed raw sample intersections with the
unambiguous exact confinement intervals.

Fresh validation:
`pixi run --frozen -e labelmaker pytest tests/labeler/test_confinement_elm.py -q -W error`
returned **4 passed**. The saved confinement hash and all seven study CSV hashes
match the manifest. The reviewed ELM snapshot is pinned separately from the live
review table.

- The revised ELM table has 110 shot identities and exactly one shared identity
  with confinement: 192751. Its joint assessed coverage is QH [2000,2300) ms,
  entirely ELM-absent. Guards of 50/100 ms leave 200/100 ms, also absent. No
  other regime has revised joint assessed support.
- Original onset traces share 104 identities with confinement. All observed
  adjacent trace timestamps differ by at least 1 ms, preventing overlapping
  1 ms sample support. There are no adjacent positive 1 ms samples in the
  merged traces; the reported positive-sample counts are not inflated by
  consecutive labeled samples of the same pulse.
- Direct raw intersections reproduce L: 18 shots, 5.043948 seconds, zero positive
  samples; QH: 69 shots, 71.216071 seconds, 60 positive samples on 15 shots;
  WP: 65 shots, 91.520799 seconds, 43 positive samples on 20 shots. Standard H
  has no onset-assessed overlap.
- The QH candidate queue has 60 unique `(shot,onset_ms)` entries. Every entry
  lies inside its half-open QH phase, and every recorded boundary distance
  matches the exact phase bounds. The 100 ms exclusion retains 51 entries on
  11 shots, over 56.557 seconds of supplied exposure. These numbers agree with
  the saved guarded CSV.

## Scientific strengths

The audit explicitly separates dense reviewed ELMing occupancy from legacy
positive onset samples. This avoids the central error of treating an onset-free
millisecond as a non-ELMing phase, or missing review as absence. Contradictory
review states become uncertain and are excluded from the assessed 0/1
denominator. Unresolved confinement subtypes are excluded from regime-specific
claims. Guards operate on consolidated physical regime phases rather than
provenance changes, and do not bridge coverage gaps.

Shot-bootstrap intervals appropriately reflect within-shot dependence for these
descriptive rates. The zero-event L outcome is presented without a misleading
all-zero resampling interval. The figure now explicitly marks unassessed reviewed
regimes, distinguishes the legacy measurement, and has legible annotations.

The strongest result is an evidence-backed coverage limitation plus an actionable
legacy candidate queue. The revised labels cannot yet adjudicate the legacy
QH/onset candidates because their aligned coverage is almost absent. This is
useful evidence for the paper's argument that shared time-axis review matters.

## Correction resolved during review

The initial prose quoted outdated bootstrap intervals. The final text now agrees
with the saved artifact: QH rate 0.842506 samples/s, CI [0.263798,1.532259]; WP
rate 0.469839 samples/s, CI [0.283247,0.677295]. The refreshed figure uses these
saved intervals and resolves the original annotation/legend overlap.

## Claim boundaries and next steps

1. **Call these unresolved candidates.** Interior overlaps rule out an explanation
   based solely on the first/last 100 ms of annotated phases. They do not rule out
   false onset detections, wrong confinement labels, phase semantics, timing
   errors, or real transient behavior. Neither donor label should be corrected
   automatically from the nominal definition of the other. No diagnostic
   adjudication has been demonstrated by a machine-generated candidate queue.
2. **Keep source-conditioned denominators explicit.** These rates describe the
   supplied, jointly covered legacy sample windows. They are not whole-QH-phase
   or population event rates. Shots contribute to multiple regimes, and there
   is no standard-H support. The evidence supports neither an H-versus-QH test
   nor a claim that L has a true zero rate. Separate descriptive per-regime
   intervals are not a paired association test.
3. **Do not present sensitivity choices as prospective registration.** The saved
   0/50/100 ms analyses are transparent exploratory sensitivity checks. A public
   timestamp before an independent gold read is needed for a confirmatory
   claim; hard-coded guards alone establish no such chronology.
4. **Independently review aligned evidence next.** Review QH candidates using
   diagnostics without automatic answers derived from the existing labels.
   Include onset-negative QH intervals and L/standard-H comparison phases,
   report selection rules and joint observable coverage, and retain uncertain
   or not-observable outcomes. If candidates are enriched, do not use that
   enriched set to estimate population association without a valid sampling
   design. Assisted ELM review remains dependent evidence until separately
   validated.

A defensible paper statement is: “Original onset annotations intersect nominal
QH intervals on 15 shots; 51 of 60 positive onset samples on 11 shots survive a
100 ms boundary exclusion. Revised ELM review overlaps only one confinement
shot, so these remain candidates for an aligned independent review.”

Nonblocking reproducibility improvement: derive reported intervals directly from
`study.json` and give each regime a stable bootstrap random stream. The latter
avoids changing QH/WP Monte Carlo endpoints when an unrelated zero-event regime
is skipped; it does not change the substantive counts or current conclusions.
