# Equilibrium and RWM review

The poloidal-beta, minimum-safety-factor and resistive-wall-mode editors use
the existing shot queue, draft tables and save/history flow. Drafts are rules
for review, with blank confidence. Only saved reviews become human labels.

## Poloidal beta

`betap_rule` reads the dimensionless EFIT01 `betap` scalar. It marks
`beta_p > 1` present for finite runs of at least 500 ms inside the longest
Ip flat-top, `|Ip| > 0.9 max(|Ip|)`. Short excursions are uncertain; measured
values at or below 1 are absent. Transitions between sampled classes remain
uncertain, including isolated high samples. Gaps and time outside the gate are not
observable. The row shows beta_p, the threshold at 1, and Ip where available.

The threshold at 1 is a chosen review convention, not a validated boundary
for the high-beta_p advanced-tokamak scenario. Published high-beta_p scenarios
typically use larger values and require additional information about q, current
drive and confinement; see
[DIII-D high-beta_p confinement experiments](https://www.osti.gov/servlets/purl/1374574).
The rule's threshold and duration are stored in each draft's `.meta.json`,
and `spans.check_rule` requires a new version after either changes.

## Minimum safety factor

`qmin_bands` uses the inventory's boundaries and 500 ms minimum duration,
on the same Ip gate. The bands are exclusive:

| ID | Label | EFIT01 q_min |
|---|---|---|
| 1 | low | q <= 0.95 |
| 2 | hybrid | 0.95 < q <= 1.5 |
| 3 | elevated | 1.5 < q <= 2 |
| 4 | high | q > 2 |
| 5 | uncertain | finite runs shorter than 500 ms or between classified runs |
| 6 | not_observable | gaps, missing inputs or outside the flat-top |

The drafts carry all six. The review page offers 1-5 only: it opens a draft's
not-observable (6) time as a gap, and time a reviewer leaves unmarked is
likewise not observable.

These are q_min bands, not a complete diagnosis of a hybrid or steady-state
scenario. Existing `qmin_rule` event exports keep their original three-band
definition. The editor adds the low band without changing those exports.
EFIT01 uses magnetic constraints; the q profile's reliability is limited
without additional constraints such as MSE. The optional q-profile row uses
normalized psi, not rho. A missing profile or Ip row does not suppress q_min.

## Resistive wall mode

The RWM roster includes the 450 non-blind cohort shots and the curated database's
33 reference shots. A database shot outside the cohort gets a view from 0 to
500 ms past its last onset. This is a display window, not observation coverage.
Original point times are drawn as peaks in a separate onset row and ticks on
the Source track. An exact onset list gives every time, toroidal mode number
(`NTOR`) and original `MODE_TYPE`, with source attribution; clicking a time
zooms to it, even when neighboring annotations share a pixel. No RWM end
time, interval or negative label is inferred. The drafts record the exact
point times and source filename in `per_shot.onsets`.

`rwm_candidates` is a screening rule for review. Inside the Ip gate, it looks
for an n=1 magnetic RMS excursion above the flat-top median plus six median
absolute deviations, together with beta_N > 4 l_i, lasting at least 10 ms.
These thresholds are an explicit conservative screening convention and have
not been fitted or validated on the curated onset database. A candidate is
uncertain (2), and other measured time is unassessed (4). Missing screen
inputs are not observable (3). The method never automatically labels a shot
present or absent. A reviewer can mark present, uncertain or unassessed; the
page does not offer not observable (3), and unmarked time is a gap.

The rows show N1RMS, N2RMS, beta_N, l_i, Ip and curated onsets, independently
where available. N1RMS includes rotating tearing modes and applied-field
response. A calibrated RWM detector needs appropriately corrected saddle-loop
signals, mode phase/rotation and an equilibrium-specific no-wall limit:
[Garofalo et al., direct RWM observation](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.82.3811),
[RWM dynamics and feedback measurements](https://www.osti.gov/etdeweb/servlets/purl/20261323).
Beta_N/l_i alone is insufficient to identify an RWM.

N1RMS/N2RMS are postprocessed, noncausal finite-frequency amplitudes. They may
miss a nearly stationary RWM, and the screen's 10 ms sampled excursion does
not measure physical growth or persistence. Offline review retains their
reported times rather than applying an assumed universal offset:
[Tang et al., magnetic RMS signal processing](https://onlinelibrary.wiley.com/doi/10.1002/ctpp.202200095),
[Garofalo et al., slowly rotating RWM observations](https://digital.library.unt.edu/ark:/67531/metadc686778/).

The binary `--gold` and `review.agreement` scores explicitly reject the
multiclass q_min editor; hybrid/elevated IDs must not be interpreted as binary
uncertain/not-observable states. RWM's unassessed state is treated as uncertain
for binary comparison, and the screen itself supplies no confirmed positives.

## Preparing an editor

Replace `poloidal_beta` below with `minimum_safety_factor` or
`resistive_wall_mode`. Prefetch runs on a login node under the fdp wrapper;
drafts read local stores only. Fetches cache under `$LABELER_RAW_CACHE` and
leave the production corpus read-only. Missing inputs are reported, and the
subsequent draft retains their unobservability.
Archive-capable legacy raw-cache signals without a recorded resolver are also
unavailable until refreshed, because their timestamps cannot be aligned safely.

```bash
pixi run --frozen -e labelmaker fdp run python -m labeler.events.equilibrium \
    --event poloidal_beta --workers 8
pixi run --frozen -e labelmaker python -m labeler.events.spans \
    --event poloidal_beta --workers 8
pixi run --frozen -e labelmaker python -m labeler.events.review.cohort_rosters \
    --event poloidal_beta --point
pixi run --frozen -e labelmaker fdp run python -m labeler.events.review.build \
    --event poloidal_beta --workers 8
pixi run --frozen -e labelmaker labeler-verify
```

`equilibrium --limit N` prepares a pilot; `--shots` restricts it to specified
roster shots. `LABELER_NO_FETCH=1` prevents fetching. A draft remains unchanged
on a rerun unless `--force` is passed; changed rules require `--version v2` or
later. Every draft sidecar records the category mapping, inputs, rule and
equilibrium provenance.

To open a newer draft, pass its version to both commands, for example
`spans --event poloidal_beta --version v2` and
`review.cohort_rosters --event poloidal_beta --version v2 --point`.
The Ip gate is computed from its full record before clipping the draft to its
window. Timestamp dropouts split finite runs and appear blank in the rows.
Archive samples are placed at their physical boxcar centers by removing the
documented 25 ms stamp lag; their 50 ms averaging remains. Per-input source,
locator and clock-correction metadata follow the signals into the cache and
draft sidecars. Source stores remain untouched.

If FDP reports expired credentials, run
`pixi run --frozen -e labelmaker fdp login`, then repeat preparation. A draft
already written with unavailable inputs is kept by default. Write a new draft
version and select it with `cohort_rosters --version v2 --point`; rebuild rows
with `review.build --force` to display newly fetched signals. Existing saved
human reviews and their history remain available.
