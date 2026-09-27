# Event Catalog

The shared tables of the DIII-D event catalog (v1): 500 shots from 2021-2025,
each assessed for six phenomena - Alfvén eigenmodes, neoclassical tearing
modes, H-mode, ELMs, sawteeth and disruptions. A phenomenon's own labels stay in
its category directory (`../<category>/review/`); what belongs to the catalog
as a whole is here.

This directory is not an event category. `events.yaml` lists it under
`non_category_dirs`, so category scans (`labeler.events.databases.category_dirs`)
skip it.

## Tables

| File | Written by | Contents |
| --- | --- | --- |
| `cohort.csv` | `python -m labeler.events.catalog.cohort` | The drawn shots: group, cell, weight, split, review-queue rank and assessed window |
| `cohort_manifest.yaml` | the same command | The seed, rules, allocation, N and n per cell, the rejection counts and the checksums of every input |
| `population.csv` | the same command | Every population shot with its group, cell, window, flat-top, legacy sets and `in_cohort`, from which `verify_cohort` draws the cohort again |
| `runaway.csv` | `python -m labeler.events.catalog.runaway` | Rule 5's scan (D2e): one row per shot passing rules 1-4, with its window's Thomson core Te statistic (`te_p90_ev`), its Thomson and profile sample counts (`n_thomson`, `n_profile`), its beam power (`pinj_kw`), its neutron rates (`neutron_rate_mean`, `;`-joined channels) and `runaway` |
| `runaway.meta.json` | the same command | The statistic and its 60 eV threshold, the counts (4,885 shots, 13 marked, 77 with no usable Thomson profile) and the sha256 of the pool, the Ip log and the corpus it read; the cohort draws only with the pool and Ip log it names |
| `papers.csv` | `python -m labeler.literature.osti links` | One row per shot-paper link the context rule verified (below) |
| `papers.meta.json` | the same command | What the build read and wrote, with sha256s; its commit, rules and coverage: the records by fetch status, the truncated query, the hits in texts it could not read |
| `osti_probe.py` | kept as it ran, 2026-09-23 | The OSTI probe that wrote the hits `links` reads (`osti_phase1.jsonl`, `osti_phase2.jsonl`); its sha256 is `papers.meta.json`'s `probe_script` |
| `cards/<method>@<version>.json` | the scoring library, later | A method's scores against the blind reference |
| Release manifest | exporter, at release (later) | Release contents and checksums |

The release manifest lists the catalog tables, the cohort, the links, the method
cards and the code commit held in the release, with their checksums.

Each file is written under `$LABELER_ROOT` first (`catalog/` for the cohort,
`literature/osti/` for the links) and copied here only with the owner's
go-ahead.

The cohort is drawn once, from inputs frozen beforehand, and is never redrawn in
place: a new draw is a new freeze, whose manifest names the freeze it supersedes and
why (`supersedes`). The first freeze (commit 10282e8, drawn at d84612f, 2026-09-26)
was superseded on 2026-09-27, before any label was made, after the literature rule
dropped a false L link, the Ip windows were measured again with the restrike rule,
and rule 5 kept the runaway-electron plateaus out of the population. The same seed
drew both; the first freeze's files stay in git history.

## Columns

`cohort.csv` has one row per drawn shot and `population.csv` one per population
shot; `verify_cohort` draws the cohort again from `population.csv` and the manifest
alone. Columns marked "both" are in both files.

| Column | In | Meaning |
| --- | --- | --- |
| `shot`, `year` | both | The shot and its campaign year (2021-2025) |
| `group` | both | `L` if the shot has a verified OSTI link at the freeze (a `papers.csv` row), else `G` if a legacy label table names it, else `R` |
| `cell` | both | The group and the year (`G2021`): the stratum the draw samples |
| `in_cohort` | population | Whether the draw took the shot |
| `weight` | cohort | N_h / n_h: the cell's population shots over its cohort shots. A blind shot's scoring weight is this times its group's cohort shots over its blind shots, n_g / b_g; `labeler.scoring.stats.two_stage_weights` computes it, and it is not stored |
| `u` | cohort | The order key: the first 8 bytes of sha256(`<seed>:order:<shot>`), big-endian, over 2^64. The cell draw uses the same key with `draw` in place of `order` |
| `split` | cohort | `test`, the blind subset: each group's smallest u, 50 shots shared across L, G and R by their cohort counts; `val`, the next 50 by the same rule; `train`, the rest |
| `blind` | cohort | `True` exactly when `split` is `test` |
| `queue_rank` | cohort | 1-500, the review order: the blind shots by u, then the rest by u, so any prefix of the queue is a random subsample of every group |
| `window_start_ms`, `window_end_ms` | both | The assessed window in whole ms, half-open [start, end): the longest stretch of Ip at or beyond 50 kA in the plasma's direction, gaps of up to 10 ms bridged, ended at a restrike (`labeler.events.catalog.window`) |
| `flattop_s` | both | The Ip flat-top inside the window, in s (the recommender's `flattop_from_ip`); the population needs 1 s |
| `ip_peak_ma` | cohort | The largest current inside the window in the plasma's direction, in MA |
| `ip_ma`, `pulse_length_s`, `pbeam_max_mw`, `pech_max_mw` | cohort | The shot table's IP, PULSE-LENGTH, PBEAM-MAX and PECH-MAX, which the population's first rule reads, from the screen's `pool.csv`. That table is not released; the manifest holds its sha256. A blank is a missing value, not zero (3 cohort shots have no `pech_max_mw`) |
| `run_id` | cohort | The run the shot belongs to, from its text bundle |
| `legacy_sets` | both | The category directories of the legacy tables that name the shot, `;`-joined; blank for none |
| `n_links_verified` | both | The shot's rows in `papers.csv` |
| `span_<group>_s` | both | The corpus census's span for the diagnostic group (mhr, ece, filterscopes, co2, sxr, mirnov), in s: the length of its longest record, not the time it covers inside the window (mhr 4.194 and ece 6.193 on every population shot); 0 where the census has none. The population needs 2 s of mhr, ece and filterscopes |

## The literature links

`papers.csv` places a shot in group L, and each row is a link an automatic rule
verified: `verified_by = auto` means the context rule, not a person's reading.

- **The rule.** A number from 185601 to 204999 in a paper's text is a shot when
  "shot", "discharge", "#" or "DIII-D" stands within 60 characters of it, or when
  it lies inside a stated range with such a word in reach (spec §9, D4). It
  belongs to the machine named nearest to it, and to the paper's machine when
  none is in reach (D22). Round axis ticks, identifiers, article numbers,
  citations and postal codes are not shots (D23). A number split by one space
  right after "shot(s)", "discharge(s)" or "#" ("discharge #194 306", as pypdf
  reads AIP's typesetting) is joined, and so are the split numbers of the run or
  range it starts (D26). A paper dated before its shot's year cannot name it
  (D14).
- **Precision.** The step judges read all 404 links of the first build and found
  401 right: 190000 and 195000 were axis ticks of one figure, and 192798 a
  contract number. D23 removed those three and added 192276, the written end of
  "192275 to 192276": 402 links to 350 shots from 111 records. D26 added 27 and
  removed none: 429 links to 374 shots from 115 records. Each of the 27 names its
  shot, one only as a range's end ("between shots 140000 - 195000").
- **Recall is bounded.** A table or a figure legend with no keyword within 60
  characters gives `no_context`: record 3364421's Table 1 lists 196094, 196097 and
  196100 under a column headed "shot". So does a number split by a space with no
  keyword right before it ("an I-mode (189 381 at 2850 ms)", "prediction in 199
  598 and 199 599"). 63 population shots have a `no_context` hit and no verified
  link: 14 of them are plainly named (the step judges read every one), and for 24
  no hit's number is in the extracted text at all (figure text pypdf cannot
  decode, images).
- **Texts not read.** Of the 439 records the fetch took (D14), OSTI has no PDF
  for 108, 2 are not PDFs, and one PDF (2538688) is a scan with no text. 36
  population shots have a hit in one of those (25 and 11 shots) and no verified
  link. Shot 189631's query matched 246 records and the probe kept 100; the
  other 146 are dated 1996-2001, before its year
  (`literature/osti/q189631_pages1-3.jsonl` under `$LABELER_ROOT`, sha256
  `a2be9615bd181fc03bdcfda93a00edff55a9167637e002a1d9157fe3a57d5688`, written by
  `q189631.py` there).

`papers.meta.json` records, for each build, the records' fetch statuses, the
truncated query, the hits in unread texts and the `no_context` hits whose number
is not in the text, beside the sha256 of every input and output.

## Known limits

- **Runaway-electron plateaus (rule 5, D2e).** A shot whose window has a median
  Thomson core Te p90 under 60 eV, with at least one profile sample (half the
  core channels valid), is a runaway plateau, not a flat-top; 13 shots are marked
  and left out of the population (194904-194906, 200809, 200811-200814, 200819,
  200821, 200823, 200824, 200827). The profile condition is window-wide: on a
  plateau most channels have no fit, and the profiles lie in the thermal phase
  before the disruption. So each mark is also checked against neutrons of at
  least 1e15 /s with beams under 1 MW (`runaway_corroboration` in the manifest):
  all 13 meet it, and so do two unmarked shots, 201447 and 203629, whose plasmas
  are thermal (2.5 and 1.8 keV) and stay in. The 77 shots with no usable Thomson
  profile have no rule-5 verdict and stay in.
- **203529, a missed restrike (D2b).** Its second hump reaches 0.57 of the peak,
  under the 0.6 the rule needs, so its window keeps about 600 ms of a second
  plasma. Its membership does not change. Recorded, not fixed: a threshold moved
  after one inspection would be tuned to the shot.
- **Post-quench tails.** Same-sign current above 50 kA after a quench stays in
  the window; D2b ends a window only at a restrike. In the version-2 audit, 50
  fast-quench population shots extended more than 25 ms past t20, and 198958 by
  362 ms.
- **Flat-tops near 1 s.** 191046's flat-top (1.0005 s) falls on either side of
  rule 4's 1 s depending on the sampling phase; at the Ip log's sampling it passes.
- **The literature links.** No person has checked a link (`verified_by = auto`).
  Two L shots rest on one weak mention each: 187328 (the end of a training-data
  range) and 189998 (a software input example). D26's join can make one shot of
  two numbers of three digits each ("Shot 200 400" → 200400, a population shot;
  "shot 195 196" → 195196); neither is among the links. Recall is bounded
  (above).
