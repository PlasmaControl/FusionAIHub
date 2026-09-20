# Frontier Shot Designer database build — September 2026

Data root `SHOT_DESIGN_DATA_ROOT=/lustre/orion/fus187/proj-shared/nchen/shot_design`; corpus
`/lustre/orion/fus187/proj-shared/foundation_model`; text
`/lustre/orion/fus187/proj-shared/foundation_model_text/shotsummary/processed/per_shot_txt`
(22,950 bundles); no `sql/` logbook by decision (mpid None, the per-mpid cap is skipped).

## F1 — census and shot list

| step | job / command | result |
|---|---|---|
| census | `sbatch scripts/slurm_frontier/shot_design_census.sh` → job 5514929, batch, 2 m 47 s on frontier08821 | 8,753 shot files (all openable), 286,635 rows → `db/corpus_coverage.parquet` |
| select | `corpus select --n 5000 --name recommender_frontier_v1 --seed 20260919 --frame-codes .../ignite_prod_v4/frame_codes` (login node, 12 s) | 3,031 eligible of 8,753 → **3,031 selected** (every eligible shot) |

The Frontier corpus directory holds 8,753 shot files, not the ~16.9k the spec assumed. The
§5.7 eligibility rules (Ip, flat-top, heating, ≥ 200 chars of shot text, shot type) leave
3,031 shots, so the list is the whole eligible pool: `corpus select` switches to the no-cap
budget (`Quotas.all_eligible`) when `--n` is at least the eligible count (commit f8e23bb), and
the list is committed as `configs/shot_design/shot_lists/recommender_frontier_v1.yaml`
(commit 2b76037). Theme counts: detachment_divertor 657, startup_checkout 247, transport 231,
hybrid_high_beta 228, current_drive 221, rmp_elm 195, control 170, qh_mode 140, neg_tri 139,
tearing_mhd 113, pedestal 110, lh_threshold_isotope 62, disruption_runaway 61, fast_ion_ae 34.
Years: 2025 1,014 · 2024 740 · 2023 59 (+ 2022 and earlier for the rest). 3,030 of 3,031 have
v4 frame codes already.

## F2 — labels, build, blurbs, encode

| step | command | result |
|---|---|---|
| logs missing | `logs missing --list recommender_frontier_v1` | 0 of 3,031 shots lack a `shot_<N>.txt` bundle |
| labels join | `labels join --list recommender_frontier_v1` (login node, 2 m 17 s) | 3,031 shots joined; **0 labelled, 0 with events** (the Frontier `LABELER_ROOT` `labels/`, `events/`, `models/` are empty — the labeler outputs were never produced on Frontier); text_claims 120,878 rows (pos 110,799 / neg 3,395 / uncertain 6,684) |
| build | `SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_build.sh` → job 5515013 (extended, 6 h, 48 workers) | pending |
| blurbs | `bash scripts/shot_design/blurb_frontier.sh` (WORKERS=8, Gemini Flash low via agy) | after E2/E3 land |
| encode | `SHOT_LIST=recommender_frontier_v1 sbatch scripts/slurm_frontier/shot_design_encode.sh` | after build |

## F2 Step 3 — build: Frontier has no plasma current (2026-09-20 02:00)

| job | partition | result |
|---|---|---|
| 5515013 | extended | cancelled (queue est. 10:20) |
| 5515059 | batch/debug, `--reader legacy` | FAILED 12 s: "3031 of 3031 shots have no Ip signal on disk" |
| 5515085 | batch/debug, `--reader corpus --no-encode` | FAILED 12 s: same message |
| 5515086 | extended, corpus reader | pending; to be cancelled and resubmitted after F2b |

Root cause, verified on the login node with `CorpusSignalReader`:
- the FAITH corpus file has 29 groups (diagnostics + actuators) and no `ip`; `additional_data`,
  Peter's `data_overlay`, `test_dataset` and the v4 `frame_codes` actuators (88) carry none either;
- the corpus reader's `ip` spec has no `corpus:` address — it resolves only through labeler's
  feature store `$LABELER_ROOT/features/<shot>_features.h5`, which does not exist on Frontier
  (`LABELER_ROOT` holds empty `events/`, `labels/`, `models/`); `signal_status` → `pending`;
- labeler's `ip` feature sources are `archive` (d3d_fusion_data) and `fdp` (MDSplus): Stellar-only.

Decision (SDD ruling, ledger 02:08): Task F2b builds segments from the shot table's `PULSE-LENGTH`
using the proxy `select.py` validated (`PROXY_RAMP_S` 1.27 s; 99.1 % agreement on 319 shots),
ramp-up 1.0 s / ramp-down 0.27 s, marked in every record (`coverage_reasons["ip"]`,
`end_reason == no_ip_signal`, no `ip` in `raw_sources`). Path to real Ip later: run
`labeler.run features` for the 3,031 shots on Stellar (fdp) and place the `<shot>_features.h5`
files under `LABELER_ROOT/features`, then rebuild (the build is atomic).

## fdp from Frontier (2026-09-20 02:22)

Works. `.pixi/envs/fdp` (frozen, linux-64) + owner token at `~/.fdp/token` (mode 600). Transport is
Pelican/OSDF over 443 (`osg-htc.org`), no MDSplus port needed. Test: PTDATA `ip` for 190736 →
480,256 samples, max |Ip| 1.66 MA, 4.1 s. So the labeler `features` stage can produce
`<shot>_features.h5` on the login node and the corpus reader then reads real Ip.

Features run (login node, 12 workers, 15 features/shot through fdp): started 02:25, log
`$LABELER_ROOT/runs/features_recommender_frontier_v1_20260920_0225.log`. Pilot rate ~35 s/shot/worker.
190736 with real Ip: flat_top 1359–5121 ms, ip_mean 1.61 MA (the proxy would have said 1000–PULSE-LENGTH−270 ms).

Result (04:56): 2 h 31 m for 3,031 shots → 2826 ok, 202 partial, 3 skipped; all 3,031
`<shot>_features.h5` carry an `ip` group (`resolver: fdp`, `decimated_to_s: 0.001`, units A, `complete: 1`).
The PULSE-LENGTH proxy (F2b) is therefore a fallback that this list never exercises.

## F2 Step 4 — build with real Ip (2026-09-20 12:28)

Job 5516457, debug QOS, `SHOT_LIST=recommender_frontier_v1`, wrapper default `--reader corpus --no-encode`.
(The 04:57 submission never landed: the session was interrupted at the sbatch call.)

Job 5516457 FAILED after 3 min (13:58): `_check_publish` refused the first build because
`shot_design labels join` (01:18, 0 labels) had already written `db/manifest.json` with only a
`labels` block — no `shot_source`/`n_shots` → "invalid manifest fields". Fix in the final fix
round (I3: a db dir without `shots.parquet` is not a database to protect). Resubmitted with
`BUILD_ARGS="--reader corpus --no-encode --force"` as job 5517542 (18:12). Encode of the one
uncached shot (191544, job 5516504) completed in 72 s → `$ROOT/frame_codes/191544.pt`; the other
3,030 list shots are served from `ignite_prod_v4/frame_codes`.

Job 5517542 (`--force`) FAILED at 18:11: `[Errno 2] .../foundation_model_text/sql/logs.jsonl` —
`text.build_logs_subset` streams the logbook unconditionally and Frontier has no `sql/`. Fixed in
the final fix round (I4: absent logbook → warn once, return 0). Next build after that commit.

Final fix round landed (b5d4e59…45931ae + 1e95d16): I3 (join-only manifest no longer blocks the
first build), I4 (absent `sql/logs.jsonl` → one warning, no logbook records). Smoke build of
190736/190000/191544 against a scratch root at 1e95d16: 3 shots / 12 segments in 21 s, real-Ip
segments (`end_reason fast_current_quench` / `target_signal_unusable`), logbook tier empty as
expected. Full build resubmitted as job 5517600 (18:48, debug QOS, `--force`).
