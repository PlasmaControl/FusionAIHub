---
title: stellarai (Princeton)
---

The second NVIDIA cluster, set up in Oct 2026 with its own copy of the
FAITH data under `/scratch/gpfs/EKOLEMEN/D3D`. Unlike Frontier, this copy was
not transferred from Stellar: it was fetched from DIII-D over GA's
Pelican/OSDF route with `fdp`, then processed here with the unchanged
`scripts/data_preparation/prepare_data.py`. See [Stellar](./stellar.md) and
[Frontier](./frontier.md) for the other two clusters, and
[Adding a cluster](./adding-a-cluster.md) for the general pattern.

## Hardware and scheduler

| Item | Value |
|---|---|
| Login / vis node | `stellarai-vis2`: 96 CPU, 1 TB, RTX PRO 6000 Blackwell (sm_120) |
| GPU partition `gpu` | 20 nodes, 8x B200 each (sm_100), 96 CPU, ~980 GB |
| CPU partition `cpu` (default) | 95 nodes, 96 CPU, 370 GB, 7-day limit |
| Account | `nstx-u` (QOS `test`/`short`/`medium`/`long`, `gpu-*`) |
| Filesystem | `/scratch/gpfs` is its own 50 TB GPFS (`stellarai.storage`), **not** Stellar's; `/projects/EKOLEMEN` is shared with Stellar |
| Network | the login node reaches `atlas.gat.com:8000` and OSDF (HTTPS); `chiron.gat.com` and `d3drdb.gat.com:8001` are unreachable; **compute nodes have no outbound network** (no DNS) |
| pixi | not a module here; installed in `$HOME/.pixi/bin` (`curl -fsSL https://pixi.sh/install.sh \| bash`) |

The default environment's torch (2.6, `cu124`) has no `sm_100`/`sm_120`
kernels, so it can run data processing but not GPU training on these
GPUs. Training here needs a CUDA 12.8+ torch feature, not yet added.

## Data layout

```
/scratch/gpfs/EKOLEMEN/D3D/
  fdp_stage/<shot>/       native DIII-D files from FDP's per-shot view (~3.4 GB/shot):
                            mdsplus/shots/<tree>_<shot>.*, ptdata/<shot>.<EXT>, meta/
  d3d_time_series_data/   <shot>.fetch.json per-node record; the raw <shot>.h5 (~16 GB) is
                            deleted once processed (regenerate from fdp_stage in ~40 s)
  signals/                FAITH corpus, <shot>_processed.h5 (~3 GB)   (= SHOT_DESIGN_CORPUS;
                            Stellar's foundation_model/)
  foundation_model_meta/  preprocessing_stats.pt (copied from /projects/EKOLEMEN/foundation_model)
  text/                   logbook text (= shot_design's text_root; Stellar's foundation_model_text/):
                            sql/, shotsummary/processed/per_shot_txt/,
                            logfetching_data/ (raw d3dlogfetching output per run),
                            d3drdb_<stamp>/ (DIII-D logbook database snapshot from FDP)
  diiid-diagnostics-wiki/ clone of PlasmaControl/diiid-diagnostics-wiki
  shot_lists/             recommender_v1.txt (the 500-shot list)
  runs/{fetch,process,prepare,slurm}/
  events/                 label tables (= LABELER_LABEL_TABLES)
  corpus_cameras/         camera groups of the 447 label-set shots (= LABELER_CORPUS)
  tangtv_efit/            TangTV frames (X.npy) + per-shot EFIT01 (efit01/), built by dev/tangtv
/scratch/gpfs/EKOLEMEN/MAST/
  level1/, mast_ss/       FAIR-MAST level-1 Zarr shots and the CER act/ss build (manifests, logs beside them)
  fair-mast/              clone of ukaea/fair-mast with the local apptainer server data
  cxrs_probe/             MAST and C-Mod CXRS probe (2026-10-08)
/scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/   (gitignored)
  labeler/                LABELER_ROOT
  shot_design/            SHOT_DESIGN_DATA_ROOT
  stellarai_setup/        2026-10-18 setup records (bundle docs, logs, backups)
```

`d3d_time_series_data` and `signals` are Stellar's raw/processed split under
another name, so `prepare_data.py` needs no changes, only path overrides.

## Environment variables

The package commands and the sbatch launchers need nothing exported for the three data roots.
On `stellarai-*` and `stellar-ai-*` hosts the `stellarai` row of `labeler.env.CLUSTERS` applies, and the longest
hostname prefix wins, so neither is taken for Stellar:

```
LABELER_ROOT          = /scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/labeler
SHOT_DESIGN_DATA_ROOT = /scratch/gpfs/EKOLEMEN/nc1514/FusionAIHub/.cache/shot_design
LABELER_LABEL_TABLES  = /scratch/gpfs/EKOLEMEN/D3D/events
```

`SHOT_DESIGN_CORPUS` is not in the table: export `/scratch/gpfs/EKOLEMEN/D3D/signals`, since
the pixi default names Stellar's corpus. The hostname prefixes in the row (`stellarai`, `stellar-ai`) were
set from the login node and a guess at the compute nodes; a compute node's hostname and `SLURM_CLUSTER_NAME` here have not been
checked. In a job, print `python src/labeler/env.py --cluster`; if it does not say `stellarai`,
set `LABELER_CLUSTER=stellarai` for the job. The `scripts/labeler` launchers run `env.py` with
Stellar's main-checkout `labeler` environment unless `LABELER_PIXI_ENV` names this checkout's. See
[Environment variables](../reference/environment-variables.md#which-cluster).

## Signals: stage (login node), then extract and process (SLURM)

There is no MDSplus login on atlas from here and chiron is unreachable, so
`scripts/data_fetching_omega/` (mdsh5 + Globus) does not apply; fdp (GA's
Pelican/OSDF route) is the only source. Reading it node by node is slow: a
full shot took 26 min even with every optimisation below, because each MDSplus
read is many small remote reads. FDP also publishes each shot as a flat
"view" (`/fdp-d3d/views/shots/<shot // 100>/<shot>/v1/`) holding the shot's
native MDSplus tree files and PTDATA files. Downloading the needed files
whole, then reading them from local disk, takes about 1 min per shot.

```bash
pixi install --frozen -e fdp && pixi install --frozen       # once
# 1. login node, needs ~/.fdp/token: ~3.4 GB/shot, ~300 MB/s with 8 shots in flight
pixi run --frozen -e fdp fdp -D d3d run python scripts/data_fetching_fdp/stage_shots_fdp.py \
    --shot-file /scratch/gpfs/EKOLEMEN/D3D/shot_lists/recommender_v1.txt \
    --stage-dir /scratch/gpfs/EKOLEMEN/D3D/fdp_stage --shots-parallel 8
# 2. CPU nodes, no network: extract (fetch_raw_fdp.py --stage-dir) + prepare_data.py
sbatch scripts/slurm_stellarai/process_staged.sbatch       # from the repo root
```

Both steps skip finished work, so either can be rerun or resubmitted while
the other is still going. `fetch_raw_fdp.py` without `--stage-dir` still
fetches over the network under `fdp run`, at about 26 min per shot.

What staging keeps, out of a ~41 GB view: every `mdsplus/shots` tree and
characteristics file, every datafile except `rpi`, `slp` and `irtv` (large,
and unread or all `TreeNODATA`), and the PTDATA files holding the configured pointnames (looked up in
the shot's `meta/index.json`), plus `.ECF .SXR .BOL .NBD .NBI`, which MDSplus
expressions read. Extraction from a stage is checked against the network
route on 185955: all 567 returned nodes are identical (data and time base),
and all 457 misses match.

Measured on this cluster, and why the extractor looks the way it does:

- **Many MDSplus leaves are TDI wrappers around PTDATA.** Examples:
  `TECEF` is `PTDATA2(ecevsNN, 4)` times a calibration; SXR and bolometer
  leaves are `ptdata("…")`; the gas `FLOW` and NB `VOLTAGE_CAL` nodes are
  expressions over PTDATA nodes and `PTHEAD_*` headers. Over Pelican, TDI's
  PTDATA path costs 17–40 s per ECE channel. On local files, `libd3`'s path
  does not work at all. The extractor therefore fetches every PTDATA call and
  header through the Python ptdata engine, recursively through referenced
  nodes, and passes them to TDI as `$1..$n` arguments (float32, as TDI's
  `ptdata2` returns). A call that fails permanently becomes `ABORT()`, so
  the expression's own `IF_ERROR` applies.
- **Subsystem trees.** `\D3D::TOP.<SUB>...` is read as `\<SUB>::TOP...`
  from the subsystem's own tree, with `d3d` as the fallback for a
  node-not-found.
- **Permanent misses are not retried.** These are `TreeNODATA`, `TreeNNF`,
  `TdiABORT` (deterministic, for example on some SXR leaves), and PTDATA
  points absent from the shot's index (`PtDataError … PTSERVER`). So is a
  1-D signal stored with a time base of a different length (ECH system 3's
  antenna angles on 195625, 195646, 195648), which `prepare_data.py` cannot
  use.
- **OSDF cache reachability fluctuates.** On 2026-10-14 the director's
  mghpcc, internet2 newy32aoa and cinc caches refused port 8443 for a while
  and then answered again, while internet2 chic dropped to half its connects.
  A transfer assigned only unreachable caches fails after 31 s, and one
  failed file aborts the rest of a multi-file `pelican object get`. So the
  stager works around it:
  - it fetches small files 8 per call and PTDATA files one per call
  - retries alternate between the director's choice and a pinned list
    (`-c uw,psu,chic,+`), with 6 attempts and growing backoff
  - failed shots are requeued for up to 3 rounds
  - a shot's `.part` directory resumes on any rerun
- **No video.** `tangtv` and IRTV nodes return nothing over fdp. Both groups
  come out as `(7, 1)` placeholders, the corpus convention for an absent
  modality.
- **Processing.** `prepare_data.py` takes about 70 s and 15 GB peak RSS per
  shot. The job runs 20 shots at a time on a 370 GB node and writes into a
  temporary directory, so a killed job leaves no truncated
  `<shot>_processed.h5` behind.

All 500 `recommender_v1` shots were staged and processed on 2026-10-14 (1.7 TB
staged, 1.5 TB processed). Staging is the bottleneck: about 1.5 shots/min from
the login node, against about 75 s for a 4-node job to extract and process 56
shots. Groups present in fewer than all 500 shots: langmuir 242, bes 260, mse
382, vib 404, cer 453–457, pinj/tinj 465, beam_voltage 468, Thomson 483–490,
gas/neutrons 499. Every missing langmuir and BES point is absent from that
shot's FDP index; the MDSplus groups' misses are `TreeNODATA`/`TreeNNF` in
the shot's own trees.

## Text

`text/` holds copies of
`/projects/EKOLEMEN/fusionaihub_logbooks/foundation_model_text/{sql,shotsummary}` and
`.../repo_data_logbooks/logfetching_data`, in the layout `shot_design` reads as
`text_root`: `sql/logs.jsonl` (53k lines), `sql/index.json` (53k shots) and
`shotsummary/processed/per_shot_txt/shot_<n>.txt` (23k shots). All 500
`recommender_v1` shots have a per-shot text file and an index entry.

### The d3drdb snapshot

FDP publishes periodic parquet snapshots of DIII-D's `d3drdb` SQL database
under `/fdp-d3d/metadata/d3drdb/`. The 2026-10-06 snapshot is 1.2 GB, 67
tables, and covers shots up to 209027. It includes `ENTRIES` (498k logbook
entries), `SUMMARIES`, `SHOTS`, `RUNS` and `preshot_summaries`; every shot of
`recommender_v1` has logbook entries and a summary row. To list and fetch:

```bash
pixi run --frozen -e fdp fdp -D d3d ls /fdp-d3d/metadata/d3drdb
pixi run --frozen -e fdp fdp -D d3d run pelican object get \
    pelican://osg-htc.org:443/fdp-d3d/metadata/d3drdb/<stamp>/<table>.parquet <dest>
```

Verify each file against the `sha256` values in the snapshot's
`manifest.json`. `shot_design` reads the derived `sql/` and `shotsummary/`
layout beside it, not these tables. The miniproposal PDFs come from
nomos.gat.com, which is unreachable from stellarai, and are not in FDP.
