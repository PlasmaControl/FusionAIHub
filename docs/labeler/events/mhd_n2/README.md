# MHD mode, n = 2

> The label data is `data/events/mhd_n2/`: the `raw/`, `format/`, `review/` and `extend_*/` paths
> below are relative to it. It is not in git. `scripts/labeler/hf_publish.py pull`
> restores the label tables (`format/`, `review/`, `extend_*/`), but not `raw/`: the
> third-party originals are not in the Hugging Face release; they are on the Stellar
> checkout's disk and archived in `/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler/raw_archive/`.

## Description
Any coherent MHD mode with toroidal mode number n = 2, read from the toroidal
Mirnov array: an n = 2 mode track (`mhd_modes` v5; v1 used the n = 2 amplitude
above a threshold). The poloidal number m is labelled afterwards, as a
subcategory inside the n = 2 spans:

- 3/2 — tearing mode (NTM)

Each subcategory is labelled only where the n = 2 label is present. The list
lives in `configs/labeler/events.yaml` (`taxonomy`).

## Data Provenance
### Dataset 1

**Dataset File(s)**: none yet; `shots.csv` is the sawtooth roster, all unverified.

## Models
**stable**: none

**latest**: mhd_modes v5. Since 2026-10-14 09:40 UTC the review page opens on this
suggestion table (`review/source.json`; v4 from 08:36 UTC, v3 from 06:10 UTC, v2 before).
The tables earlier versions wrote stay on disk.

**all**:
- mhd_modes | v5 | 2026_10_14 | mhd_n2: present 886.6 s on 446 shots; absent 2111.3 s; not
  observable 374.3 s (the same rows as v4). Statuses of the modelled n = 2 tracks: m/n
  6687, not resonant 2067, uncertain 1423, no rotation 592, q_min above 120.
- mhd_modes | v5 | 2026_10_14 | 3/2 (m3): present 428.9 s on 272 shots; uncertain 68.8 s on
  246; absent 2377.6 s; not observable 497.0 s. Tracks: 5805 m/n, 1515 uncertain.
- mhd_modes | v4 | 2026_10_14 | mhd_n2: present 886.6 s on 446 shots; absent 2111.3 s; not
  observable 374.3 s (the same rows as v3). Statuses of the modelled n = 2 tracks: m/n
  6609, not resonant 2048, uncertain 1415, no rotation 557, q_min above 120.
- mhd_modes | v4 | 2026_10_14 | 3/2 (m3): present 421.4 s on 267 shots; uncertain 65.7 s on
  241; absent 2368.7 s; not observable 516.5 s. Tracks: 5731 m/n, 1478 uncertain.
- mhd_modes | v3 | 2026_10_14 | mhd_n2: present 886.6 s on 446 shots; absent 2111.3 s; not
  observable 374.3 s. Statuses of the modelled n = 2 tracks: m/n 5978, not resonant 1818,
  uncertain 1228, no rotation 1136, q_min above 101.
- mhd_modes | v3 | 2026_10_14 | 3/2 (m3): present 383.2 s on 236 shots; uncertain 74.7 s;
  absent 2359.5 s; not observable 554.8 s. Tracks: 5247 m/n, 1554 uncertain.
- mhd_modes | v2 | 2026_10_14 | mhd_n2: present 818.2 s on 413 shots; absent 1998.5 s; not
  observable 555.5 s. Statuses of the modelled n = 2 tracks: m/n 5451, not resonant 1604,
  uncertain 1093, no rotation 412, q_min above 90.
- mhd_modes | v2 | 2026_10_14 | 3/2 (m3): present 353.6 s on 220 shots; uncertain 46.9 s;
  absent 2199.4 s; not observable 772.4 s. Tracks: 4816 m/n, 1114 uncertain.
- n_amplitude | v1 | 2026_10_11 | n = 2 amplitude above a threshold (the earlier source).
- q_rule | v1 | 2026_10_11 | the earlier source for m3.

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

Rows (v4, the same in v5):
- The parent is present wherever a track with that n is, except fixed-frequency lines. This
  includes harmonics (4/2, 6/3) and modes with no resonant class. Each frame is padded by
  2 ms, gaps of up to 20 ms are merged and spans shorter than 20 ms are dropped.
- The child is present over tracks with status m/n and its label (confidence P(resonant)).
  It is uncertain over uncertain, no-rotation and q-min-above tracks whose 90% credible set
  holds it; on an EFIT01 shot a q-min-above track is also uncertain for its candidate class
  (v4). It is not observable on a shot with no equilibrium tree at all, where no slice lies
  within 30 ms and over tracks the model could not run on.
- The child is set only inside the parent's present time.

Run (v5): 614 shots, the `mhd_n*` rosters plus the silver and gold lists (`mhd_roster`);
no blind shot is among them (the writer drops any).
- 608 shots were written. 6 have no window (no catalog window and no Ip record).
- The Mirnov was read on 544 shots (2 of them have no n = 1-3 track). 64 have no corpus
  record; those shots are not observable for all eight events. A record that
  starts late or ends early (34 shots) is used where it is finite, as in v3.
- Equilibrium tree of the written shots: EFIT02 495, EFIT02ER 15, EFIT01 95, none 3
  (193075, 194401 and 203784, whose fallback fetch failed). The 3 hold a one-slice EFIT02
  and no fallback file, so their children are not observable inside the parent's present
  time. 21 shots gained a tree over v4 (EFIT02ER 8, EFIT01 13); v4 had 104 over v3.
- Rotation (ZIPFIT) on 546 shots: EFIT02 485 of 495, EFIT02ER 15 of 15, EFIT01 43 of 95,
  none 3 of 3.
- The pipeline ran again only on the 21 shots whose cache key changed (the equilibrium
  file); the other 587 came from the cache. 16 workers, 87 s wall clock (32 s a shot,
  median). v4 took 18.1 min for all 614. The eight tables from the cache:
  `python -m labeler.events.candidates --method mhd_modes --event all --cached --version v5`
  (a cached shot whose key no longer holds is skipped as stale, never used).

Checks (2026-10-14):
- v4 against v5: same 608 shots; no blind shot; 199597's rows are the same. mhd_n2's rows
  are the same as v4 (the union's span rule is mhd_n1's only, and the Mirnov tracks did not
  change). 3/2 changes on 13 of the 21 new-tree shots (12 EFIT01, and EFIT02ER on 202181),
  whose children inside mhd_n2's present time were not observable in v4: present 421.4 ->
  428.9 s (267 -> 272 shots), uncertain 65.7 -> 68.8 s, not observable 516.5 -> 497.0 s.
  3/2 is never present outside mhd_n2's present time (0 ms, in v4 too). The other 8
  new-tree shots have no n = 2 present time.
- v3 against v4: mhd_n2's rows are the same as v3 (the Mirnov tracks did not change). 3/2
  changes on 62 shots, from the trees and rotation added since v3: present 383.2 -> 421.4 s
  (236 -> 267 shots), not observable 554.8 -> 516.5 s. On EFIT01 3/2 is present 9.6 s on 9
  shots, on EFIT02ER 9.1 s on 6. The 24 shots that lost their one-slice EFIT02 had no 3/2
  present in v3. The EFIT01 q_min-above rule changed no cell in this run (6 such tracks
  over all n on EFIT01 shots).
- The freeze is unchanged (sha256 3c71b767ebd2636b).
- v2 against v3: on the 529 other shots whose inputs did not change (same cache key) the
  tracks, statuses and rows are the same, except 6 ms on 186636: its Mirnov record ends at
  7.279 s, 6 ms before its window, and v3 marks those 6 ms not observable. On the 34 shots
  v3 finds 2-267 tracks a shot. mhd_n2 is present 68.4 s on 33 of them and 3/2 29.6 s on 16.
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
- Over the labeler window the 4 ms frame grid shifts, so some tracks split differently.
  Against `cases_res.csv` on 199597, 199603 and 199607, the m/n time agrees with IoU 0.911,
  and with IoU 0.93-0.99 for 2/1. On 199597 the 2/1 starts at 2.790 s (dev 2.793 s). On
  199603 it starts at 3.574 s (dev 3.589 s). On 199607 it runs 5.058-5.606 s (dev
  5.057-5.485 s), plus one 32 ms 2/1 span at 1.526 s (P(resonant) 0.904).
