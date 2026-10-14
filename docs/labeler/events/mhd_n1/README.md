# MHD mode, n = 1

> The label data is `data/events/mhd_n1/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/labeler/raw_archive/`.

## Description
Any coherent MHD mode with toroidal mode number n = 1, read from the toroidal
Mirnov array: an n = 1 mode track (`mhd_modes` v5; v1 used the n = 1 amplitude
above a threshold). The poloidal number m is labelled afterwards, as a
subcategory inside the n = 1 spans:

- 1/1 — internal kink: sawtooth crashes and their precursors, fishbones
- 2/1 — tearing mode (NTM); the one that slows, locks and disrupts
- 3/1 — tearing mode at q = 3 (high q_min), or external kink near q95 = 3

Subcategories overlap each other freely; each one is labelled only where the
n = 1 label is present. The list lives in `configs/labeler/events.yaml` (`taxonomy`).

Replaces the review page's `sawtooth_oscillation`, `fishbone` and
`neoclassical_tearing_mode` events (folded, their data untouched).

## Data Provenance
### Dataset 1

**Dataset File(s)**: none yet; `shots.csv` is the sawtooth roster (450 cohort
shots + 10 former sawtooth gold shots), all unverified.

## Models
**stable**: none

**latest**: mhd_modes v5. Since 2026-10-14 09:40 UTC the review page opens on this
suggestion table (`review/source.json`; v4 from 08:36 UTC, v3 from 06:10 UTC, v2 before).
The tables earlier versions wrote stay on disk.

**all**:
- mhd_modes | v5 | 2026_10_14 | mhd_n1: present 1331.3 s on 543 shots (425.9 s of it only
  from crash v3 and the union's merged gaps); absent 1672.1 s; not observable 368.8 s.
  Statuses of the modelled n = 1 tracks: m/n 7302, uncertain 3214, not resonant 2485, no
  rotation 1070, q_min above 518.
- mhd_modes | v5 | 2026_10_14 | 1/1 (m1): present 774.2 s on 434 shots (536.4 s of it only
  from crash v3); uncertain 117.5 s on 286; absent 1947.0 s; not observable 533.6 s. 4987
  1/1 tracks are m/n and 3894 are uncertain.
- mhd_modes | v5 | 2026_10_14 | 2/1 (m2): present 155.8 s on 213 shots; uncertain 124.0 s
  on 406; absent 2553.7 s; not observable 538.8 s. Tracks: 1268 m/n, 2787 uncertain.
- mhd_modes | v5 | 2026_10_14 | 3/1 (m3): present 85.3 s on 95 shots; uncertain 133.7 s on
  385; absent 2600.5 s; not observable 552.7 s. Tracks: 827 m/n, 2545 uncertain.
- mhd_modes | v4 | 2026_10_14 | mhd_n1: present 1330.7 s on 543 shots (425.3 s of it only
  from crash v3); absent 1672.8 s; not observable 368.8 s. Statuses of the modelled n = 1
  tracks: m/n 7142, uncertain 3128, not resonant 2422, no rotation 986, q_min above 516.
- mhd_modes | v4 | 2026_10_14 | 1/1 (m1): present 773.0 s on 436 shots (539.7 s of it only
  from crash v3); uncertain 113.6 s on 279; absent 1936.3 s; not observable 549.3 s. 4882
  1/1 tracks are m/n and 3748 are uncertain.
- mhd_modes | v4 | 2026_10_14 | 2/1 (m2): present 151.6 s on 209 shots; uncertain 120.2 s
  on 396; absent 2541.0 s; not observable 559.4 s. Tracks: 1239 m/n, 2696 uncertain.
- mhd_modes | v4 | 2026_10_14 | 3/1 (m3): present 83.3 s on 91 shots; uncertain 128.2 s on
  375; absent 2587.4 s; not observable 573.3 s. Tracks: 802 m/n, 2427 uncertain.
- mhd_modes | v3 | 2026_10_14 | mhd_n1: present 905.4 s on 539 shots; absent 2092.6 s; not
  observable 374.3 s. Statuses of the modelled n = 1 tracks: m/n 6070, uncertain 2804,
  not resonant 1898, no rotation 1333, q_min above 501.
- mhd_modes | v3 | 2026_10_14 | 1/1 (m1): present 322.5 s on 364 shots (128.7 s of it only
  from crash v2); uncertain 112.0 s; absent 2509.6 s; not observable 428.1 s. 3974 1/1
  tracks are m/n and 3609 are uncertain.
- mhd_modes | v3 | 2026_10_14 | 2/1 (m2): present 145.0 s on 184 shots; uncertain 122.1 s;
  absent 2481.1 s; not observable 624.1 s. Tracks: 1139 m/n, 2717 uncertain.
- mhd_modes | v3 | 2026_10_14 | 3/1 (m3): present 79.5 s on 81 shots; uncertain 124.8 s;
  absent 2530.0 s; not observable 637.8 s. Tracks: 758 m/n, 2335 uncertain.
- mhd_modes | v2 | 2026_10_14 | mhd_n1: present 838.8 s on 505 shots; absent 1977.9 s; not
  observable 555.5 s. Statuses of the modelled n = 1 tracks: m/n 5629, uncertain 2568,
  not resonant 1600, q_min above 473, no rotation 406.
- mhd_modes | v2 | 2026_10_14 | 1/1 (m1): present 307.2 s on 341 shots; uncertain 91.5 s;
  absent 2429.1 s; not observable 544.4 s. 3737 1/1 tracks are m/n and 2859 are uncertain.
- mhd_modes | v2 | 2026_10_14 | 2/1 (m2): present 135.8 s on 169 shots; uncertain 88.4 s;
  absent 2313.7 s; not observable 834.3 s. Tracks: 1064 m/n, 2000 uncertain.
- mhd_modes | v2 | 2026_10_14 | 3/1 (m3): present 68.3 s on 70 shots; uncertain 93.1 s;
  absent 2365.5 s; not observable 845.2 s. Tracks: 636 m/n, 1715 uncertain.
- n_amplitude | v1 | 2026_10_11 | n = 1 amplitude above a threshold (the earlier source).
- crash | v1 and q_rule | v1 | 2026_10_11 | the earlier sources for m1, and for m2 and m3.

The method (`src/labeler/mhd/`, ported from `dev/mhd`) finds Mirnov peaks every 4 ms, fits
the toroidal n over six midplane probes and links the peaks into tracks. Rules 1-4 remove
fixed-frequency lines and harmonics and keep the m/n whose q surface lies between q_min - 0.2
and q95 + 0.2. A label model (frozen, `src/labeler/mhd/freeze.json`) combines the
magnetics, ECE and CO2 views, and a rotation gate checks that f/n matches the plasma rotation
(ZIPFIT) at that q surface. A track gets the status m/n, uncertain, q_min above, not resonant
or no rotation. Each shot runs once over its plasma window (the catalog window, else the Ip
window); the result is cached under `$LABELER_ROOT/mhd/labeler/v5/` (a copy of the v4 cache,
which stays as it was, with the shots whose key changed run again). A Mirnov record that
is NaN over part of the window is used where it is finite, and the rest is not observable
(v3; v2 dropped such a shot).

The equilibrium (v4) is one file a shot: EFIT02 (`$LABELER_ROOT/mhd/efit02/`) when it holds
at least two time slices, else the fallback file (`$LABELER_ROOT/mhd/efit_fallback/`,
EFIT02ER else EFIT01). EFIT02ER is MSE-constrained and is trusted like EFIT02. EFIT01 is
magnetics-only and goes through the same frozen law, but its q_min reads high, so a
q-min-above track is uncertain for its candidate class, never absent. The file read (path,
mtime, size) is part of the cache key, and its tree is in the cache and in each shot's
meta (`per_shot.tree`). v3 read EFIT02 only. v5 has the last fallback files: 21 more
shots read EFIT02ER (8) or EFIT01 (13).

Rows (v5):
- mhd_n1 is present wherever an n = 1 track is, except fixed-frequency lines. This includes
  harmonics and modes with no resonant class. Each frame is padded by 2 ms, gaps of up to
  20 ms are merged and spans shorter than 20 ms are dropped. The confidence is the largest
  median n-fit score in the span.
- mhd_n1 is also present over the present spans of crash v3
  (`$LABELER_ROOT/suggestions/crash/v3/`): a q-gated ECE sawtooth crash is 1/1 internal-kink
  evidence, so n = 1. This holds where no n = 1 track is, and over Mirnov gaps and shots with
  no Mirnov record, since the crash is seen on ECE. That time has no confidence.
- The span rule is applied again to the union (v5): absent gaps of up to 20 ms between
  present spans become present (no confidence), and a span shorter than 20 ms that holds no
  track time is dropped. Gaps over Mirnov gaps or no record stay not observable.
- A child (1/1, 2/1, 3/1) is present over tracks with status m/n and that label, with
  P(resonant) as the confidence. It is uncertain over uncertain, no-rotation and q-min-above
  tracks whose 90% credible set holds it; on an EFIT01 shot a q-min-above track is also
  uncertain for its candidate class. It is not observable on a shot with no equilibrium tree
  at all, where no slice lies within 30 ms and over tracks the model could not run on.
- The crash is 1/1 evidence only. 2/1 and 3/1 are set against the n = 1 tracks alone, so
  over mhd_n1 time from crash alone they are absent where the Mirnov saw no n = 1 track and
  not observable where it saw nothing, as without the crash.
- A child is set only inside mhd_n1's present time; outside it the child takes mhd_n1's
  state (absent or not observable).
- 1/1 is the resonant 1/1 tracks united with crash v3's present spans, inside mhd_n1's
  present time. Only the crash runs under 20 ms that the union's span rule drops fall
  outside (54 ms in v5; v3 dropped all crash time with no n = 1 track, decision M3). The
  crash adds present time with no confidence; a resonant 1/1 track under it keeps its
  P(resonant) (v4 blanked it).
- Time with no q = 1 surface (`labeler.sawtooth.qgate`) is absent for 1/1, except where 1/1
  is not observable.

Run (v5): 614 shots, the `mhd_n*` rosters plus the silver and gold lists (`mhd_roster`);
no blind shot is among them (the writer drops any).
- 608 shots were written. 6 have no window (no catalog window and no Ip record).
- The Mirnov was read on 544 shots (2 of them have no n = 1-3 track). 64 have no corpus
  record; those shots are not observable for all eight events, bar crash time for mhd_n1
  and 1/1. A record that starts late or ends early (34 shots) is used where it is finite,
  as in v3.
- Equilibrium tree of the written shots: EFIT02 495, EFIT02ER 15, EFIT01 95, none 3
  (193075, 194401 and 203784, whose fallback fetch failed). The 3 hold a one-slice EFIT02
  and no fallback file, so their children are not observable inside the parent's present
  time, except 1/1's crash time (present from the crash table). 21 shots gained a tree over v4 (EFIT02ER 8, EFIT01 13); v4 had 104 over v3.
- Rotation (ZIPFIT) on 546 shots: EFIT02 485 of 495, EFIT02ER 15 of 15, EFIT01 43 of 95,
  none 3 of 3.
- The pipeline ran again only on the 21 shots whose cache key changed (the equilibrium
  file); the other 587 came from the cache. 16 workers, 87 s wall clock (32 s a shot,
  median). v4 took 18.1 min for all 614. The eight tables from the cache:
  `python -m labeler.events.candidates --method mhd_modes --event all --cached --version v5`
  (a cached shot whose key no longer holds is skipped as stale, never used).
- Crash v3 (`$LABELER_ROOT/suggestions/crash/v3/mhd_n1.m1_suggest_crash_v3.csv`, sha256
  42775bc0, q-gated on EFIT02, else EFIT02ER, else EFIT01) has present spans on 383 of
  these shots.

Checks (2026-10-14):
- v4 against v5: same 608 shots; no blind shot; 199597's rows are the same for all eight
  events. Children are never present outside their parent's present time (0 ms, in v4 too).
  - mhd_n1 changes on 55 shots, all from the union's span rule: 0.706 s of absent gaps
    became present, and 54 ms of crash-only runs under 20 ms were dropped on 4 shots
    (186025, 190633, 200012, 202197). Present goes 1330.7 -> 1331.3 s. It now has no
    present run under 20 ms and no absent gap of 20 ms or less (v4 had 6 and 66).
  - 1/1 changes on 198 shots. On 14 of the 21 new-tree shots (13 EFIT01, and EFIT02ER on
    202181) the children inside mhd_n1's present time are set for the first time (v4:
    not observable). On 13 others the categories change:
    35 ms present -> absent (the dropped runs) and 105 ms absent -> not observable (merged
    gaps where no equilibrium slice lies within 30 ms). On 171 shots only the confidence
    changes: 90.7 s of resonant 1/1 present under crash time get their P(resonant) back.
  - 2/1 and 3/1 change on the same 14 new-tree shots only (they are set against the tracks
    alone, so the union's span rule does not reach them). Present 2/1 151.6 -> 155.8 s, 3/1
    83.3 -> 85.3 s; not observable falls by 20.7 and 20.6 s.
  - The other 7 new-tree shots (EFIT02ER: 199166, 199171, 199351-199354, 200977) do not
    change: their EFIT02ER covers only 0.9-2.1 s (3-17 slices), no slice lies within 30 ms
    of their n = 1-3 present time, so the children stay not observable there.
  - Crash v3 is still the crash source: the new files change no crash v3 row, so no crash
    v4 was built. The q = 1 gate read every shot (`q_gate.errors` is empty).
- v3 against v4: mhd_n1 without crash v3 is v3's rows on all 608 shots. Crash v3 adds 425.3 s of present time,
  5.5 s of it on 8 shots over Mirnov gaps or shots with no Mirnov record; 361 shots change
  and 543 have mhd_n1 present (539 in v3). Every crash v3 present ms inside a written window
  (630.4 s) is present in both mhd_n1 and 1/1; none is clipped (v3 dropped 427.1 s of crash
  v2).
- 1/1: crash v3 adds 539.7 s of present time over the 1/1 tracks: 461.7 s that would be
  absent, 50.3 s uncertain and 27.7 s not observable. 168.9 s of 1/1 not-observable time
  lies in no-q=1 spans, which v3 would have marked absent. 1/1 is absent on 544 shots (607
  in v3, which marked no-q=1 time absent on shots with no Mirnov too).
- 2/1 and 3/1 are cell for cell the same with or without crash v3 (all 608 shots). They
  change on 125 and 127 shots, from the trees and rotation added since v3.
- By tree, v4: on EFIT01 1/1 is present 140.3 s on 63 shots (mostly crash time), 2/1 2.3 s
  on 5 and 3/1 0 s (uncertain 14.5 s); on EFIT02ER 1/1 14.4 s on 6 shots and 2/1 0.3 s on 4.
  The 24 shots that lost their one-slice EFIT02 had 7.5 s of 1/1 present in v3 and no other
  child present; in v4 they have 23.6 s of 1/1 present from crash v3.
- The EFIT01 q_min-above rule changed no cell in this run. EFIT01 shots have 6 q_min-above
  tracks, 0.4% of their 1590 modelled tracks (2.3% on EFIT02); 558 of the 1590 are no
  rotation, since 52 of the 82 EFIT01 shots have no rotation.
- The 2/1 spans on the case shots are the same as v3: 199597 2.790-6.038 s, 199603
  3.574-6.094 s, 199607 1.526-1.558 s (P(resonant) 0.904) and 5.058-5.606 s.
- The freeze is unchanged (sha256 3c71b767ebd2636b); `python -m labeler.mhd.freeze
  --check-only` leaves `freeze.json` as it is.
- v2 against v3: on the 529 other shots whose inputs did not change (same cache key) the
  tracks, statuses and rows are the same, except 6 ms on 186636: its Mirnov record ends at
  7.279 s, 6 ms before its window, and v3 marks those 6 ms not observable. On the 34 shots
  v3 finds 2-267 tracks a shot. mhd_n1 is present 66.6 s on 34 of them, 1/1 17.2 s on 24,
  2/1 9.2 s on 15 and 3/1 11.2 s on 11. The crash v2 time dropped outside mhd_n1's present
  time on them went from 33.3 s to 24.9 s.
- Freeze: fit on 208 dev-corpus shots, 11292 tracks, no gold shot among them (shot-list
  sha256 in `freeze.json`). Its statuses match dev `resonance_corpus.csv` on 0.9847 of the
  tracks. The dev file was made at feature cut 0.98, and a refit at 0.98 matches all 11292.
  The freeze uses 0.95, the cut of dev's case-shot run, and matches `cases_res.csv` on
  599 of 601 tracks.
- Port check: run on the same window as dev, the pipeline reproduces dev's tracks on six
  case shots (190384, 190866, 191599, 199597, 199603, 199607): 640 tracks, every one the
  same. Statuses match on 499 of 501 tracks.
- Regression test (`tests/labeler/test_mhd_modes.py`): five 199607 case tracks, one per
  status, in `tests/labeler/data/mhd_frozen_golden.npz`. The frozen model must give their
  stored posterior, distances, 90% set and status within 1e-9, and dev's status, label and
  class. The port's refit of dev's case model gives dev's posterior on all 601 case tracks
  within 1.8e-15.
- 1/1 present time only from crash v2 is the time it adds over the 1/1 tracks: 128.7 s
  in v3, 123.7 s on the v2 cache with the same q gate. The 198.7 s this README gave for
  v2 was crash v2's whole time inside mhd_n1's present time, 1/1 track time included.
- Over the labeler window the 4 ms frame grid shifts, so some tracks split differently.
  Against `cases_res.csv` on 199597, 199603 and 199607, the m/n time agrees with IoU 0.911,
  and with IoU 0.93-0.99 for 2/1. On 199597 the 2/1 starts at 2.790 s (dev 2.793 s). On
  199603 it starts at 3.574 s (dev 3.589 s). On 199607 it runs 5.058-5.606 s (dev
  5.057-5.485 s), plus one 32 ms 2/1 span at 1.526 s (P(resonant) 0.904).
