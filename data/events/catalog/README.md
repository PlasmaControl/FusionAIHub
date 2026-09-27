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
| `papers.csv` | `python -m labeler.literature.osti links` | One row per shot-paper link the context rule verified (below) |
| `papers.meta.json` | the same command | What the build read and wrote, with sha256s; its commit, rules and coverage: the records by fetch status, the truncated query, the hits in texts it could not read |
| `osti_probe.py` | kept as it ran, 2026-09-23 | The OSTI probe that wrote the hits `links` reads (`osti_phase1.jsonl`, `osti_phase2.jsonl`); its sha256 is `papers.meta.json`'s `probe_script` |
| `cards/<method>@<version>.json` | the scoring library, later | A method's scores against the blind reference |
| Release manifest | exporter, at release (later) | Release contents and checksums |

The release manifest lists the catalog tables, the cohort, the links, the method
cards and the code commit held in the release, with their checksums.

Each file is written under `$LABELER_ROOT` first (`catalog/` for the cohort,
`literature/osti/` for the links) and copied here only with the owner's
go-ahead. The cohort is drawn once, from inputs frozen beforehand, and is never
redrawn in place.

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
