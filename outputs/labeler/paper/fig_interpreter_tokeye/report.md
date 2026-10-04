# Figure 1 — final state (fix round 11)

Status: **final**. The owner capped review rounds at 10, so round 11 closes this stream
without a further review. Primary: **DIII-D 201978, 1500–3300 ms**; data, windows,
detector outputs and the sawtooth source pin (**ad0ca40f**) are unchanged since round 9.
Last reviews: Sol rounds 8–10 accepted (8/8/8/8); Opus round 11 scored 8/8/7/7, with
one Important item (harmonic wording), fixed here. Earlier rounds are in git; see
`docs/labeler/fig1_fix_history.md`.

Commit range for round 11: **3941c5b6..34ad44f2** (branch r4-fig1), plus this report.

| Commit | What |
|---|---|
| `2b4cd292` | AE display band ≥60 kHz and scale break at 60 kHz; harmonic gate; AE chip; dashed NTM component outline; regime greys; tests |
| `335cbdf5`, `7f6ab5b2` | Room for the 30–60 kHz strip tick labels (≥1 pt clearance on every shot) |
| `34937d65` | 50–60 kHz drawn from the wide pass (the zoom-pass decimation filter rolls off above ~50 kHz) |
| `34ad44f2` | Six renders (from clean `34937d65`), sidecars, `audit.json`, README, history index |

## Primary caption (97 words, `201978.caption.tex`)

> DIII-D shot 201978. Top: raw Mirnov spectrogram (three frequency scales: 0–30 kHz
> stretched, 30–60 and 60–250 kHz compressed; bands normalised separately), D-alpha, NBI
> power. Middle: TokEye coherent-mode mask after small-object removal; below 30 kHz
> coloured by toroidal mode number n (Mirnov array). Pink: mask pixels ≥60 kHz while the
> CO2 AE detector (80–250 kHz input band; trained on TokEye-mask-derived targets, so not
> independent of TokEye) is positive (25 ms bins). Orange outlines: n=1/2 pixels while
> the NTM detector (held-out F1 0.46, below our 0.7 bar) is positive. Highlights mark
> time/band coincidence only. Bottom: label tracks with sources.

## Round 11 changes (review `fig1-opus11`)

- **Harmonics (Important).** "Not separate islands" deleted. The appendix gives the
  measured support: n=2 within 5% of 2×f(n=1) in 326 of 700 ms where both are measured
  (46%); n=3 within 5% of 3×f(n=1) in 232 of 571 ms (41%). It adds that a frequency ratio
  cannot separate harmonics of one island from phase-locked coupled modes (m needs EFIT q
  or the poloidal array). The caption clause now requires a passing fraction ≥ 0.6, so it
  is absent from every caption.
- **AE band (owner decision).** The owner asked for "AE only above 60 kHz; sawtooth and
  NTM below 60 kHz" and earlier said the 80 kHz floor was only a labelling view. AE pink is
  now mask pixels ≥60 kHz during detector-positive time, and both spectrogram groups break
  at 60 kHz. The detector's own input band (80–250 kHz) is stated; 60–80 kHz pink is time
  coincidence only. AE-tagged pixels on 201978 rise from 319 to 365 (`201978.json`).
- **Minors.** "AE" chip at 2100 ms / 200 kHz replaces the diagonal pointer; a thin dashed
  outline shows the rest of a tagged NTM component cut by the detector-positive clip;
  distinct L and WPQH greys; one-line sawtooth source; more room under the NBI axis; per-
  shot boilerplate dropped from alternates; 191782 flagged as in the NTM detector's
  training set; 186636's AE frame-detector membership recorded (in neither its train nor
  test split); README records why 201978 was chosen (clear AE, L–H transition, ELMs and a
  late n=1 mode in one window) and that only its ELM row is expert-reviewed.

## Verification

`scripts/labeler/fig1_audit.py --rebuild-primary --baseline-ref 3941c5b6` passes with a
byte-identical primary rebuild. 85 covering tests pass; ruff check/format clean; all six
captions compile. Primary and alternate PNGs, strip crops, the dashed key and the AE chip
were inspected.

## Deviations and open limits

- AE band: 60 kHz display floor (owner) vs the 80–250 kHz detector input band; history in
  rounds 4–10 used 80 kHz.
- Caption cap raised from 95 to 120 words (primary 97, longest alternate 117).
- "Held-out F1 0.46" is the NTM detector's score on its own 761 held-out shots, below the
  0.7 bar; NTM shown as suggestions.
- AE highlights are not independent of TokEye (detector targets derived from its mask).
- Only the ELM row of the primary is expert-reviewed; the physics sawtooth labels are
  unvalidated; the sawtooth row is uncertain or blank (ECE cut-off) on this shot.
- The 50–60 kHz strip comes from the wide pass and looks darker than the zoom pass below
  50 kHz at quiet times; this is a pass change, not physics.
- Three frequency scales kept; a two-scale layout was not rendered for comparison.
- Re-pin the sawtooth source at integration if that stream moves.
