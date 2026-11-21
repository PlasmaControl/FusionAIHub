---
title: nvidia (node13)
---

A single aarch64 workstation-class node, set up on 2026-11-21 for one
workstream: IGNITE retrained on MAST data. The `nvidia` branch carries this
setup. Nothing from DIII-D is stored here; the data is FAIR-MAST and the
TokaMark benchmark built on it. See [Stellar](./stellar.md),
[StellarAI](./stellarai.md) and [Frontier](./frontier.md) for the clusters, and
[Adding a cluster](./adding-a-cluster.md) for the general pattern. This machine
has no Slurm and no `labeler.env.CLUSTERS` row: the labeler and shot_design
workstreams do not run here.

## Hardware

| Item | Value |
|---|---|
| Host | `node13`, Ubuntu 26.04, aarch64 (64k pages) |
| CPU / RAM | 352 cores, 1.4 TB |
| GPU | 4x NVIDIA, compute capability 10.7, 286 GB each, driver 620.47 |
| CUDA | system toolkit 13.5 at `/usr/local/cuda-13.5` (`nvcc --list-gpu-arch` includes `compute_107`) |
| Storage | `/models`, 28 TB (`/dev/mapper/models-data`); `/` 1.8 TB NVMe |
| Network | internet through a USB 1 GbE adapter (`enx905a08c5ca3d`, private 172.25.x.x address); the ConnectX and InfiniBand ports are down |

## PyTorch on compute capability 10.7

The PyPI aarch64 wheels (measured with torch 2.15.0+cu132) ship kernels for
`sm_80 sm_90 sm_100 sm_110 sm_120`. Precompiled kernels run (sm_100 cubins are
binary compatible with 10.7), but anything torch compiles at run time through
NVRTC (the jiterator: `torch.special.*`, some fused elementwise ops) fails with

```
nvrtc: error: invalid value for --gpu-architecture (-arch)
```

because the NVRTC bundled in the wheel (13.2) does not know sm_107. Preloading
the system's NVRTC 13.5 fixes it (checked 2026-11-21 with `torch.special.i0e`):

```bash
export LD_PRELOAD=/usr/local/cuda-13.5/lib64/libnvrtc.so.13:/usr/local/cuda-13.5/lib64/libnvrtc-builtins.so.13.5
```

`LD_LIBRARY_PATH` alone does not work: torch loads the wheel's
`nvidia/cu13/lib/libnvrtc.so.13` by its own path.

## Environments

The pixi workspace lists `linux-64`, `win-64` and `osx-arm64`, so none of the
repository's pixi environments installs here yet. Adding `linux-aarch64` means
re-solving the lock, which [Pixi environments](../reference/pixi-environments.md)
explains is deferred for every other cluster; it is the open item for this
branch. Until then:

| Tool | Where | For |
|---|---|---|
| pixi, uv, s5cmd, aria2c | `~/.pixi/bin` (`pixi global install`) | tooling |
| `hf` (huggingface_hub 2.4, hf_xet) | uv tool, `~/.local/bin` | Hugging Face downloads; `hf auth login` once |
| `mineru` 4.2.3 (`mineru[torch]`) | uv tool, `~/.local/bin` | PDF to markdown for the paper library |
| TokaMind + TokaMark | uv venv `/models/tokamind/.venv` (Python 3.12, torch 2.15+cu132) | reproducing TokaMind's TokaMark scores |

## Storage layout

```
/models/
  mast/
    fairmast/{meta,level2,level1}/   FAIR-MAST metadata parquet and per-shot Zarr
    tokamark/hf_batches/             TokaMark v1 tars from Hugging Face (~565 GB)
    tokamark/v1/                     unpacked TokaMark <shot>.zarr
  intermediate/                      derived data (frame codes, statistics, caches)
  tokamind/{repo,tokamark,.venv}/    TokaMind and TokaMark checkouts, their env, weights under repo/runs/
  papers/<topic>/{pdf,md}/           the literature library and its MinerU extractions
```

`/models/README.md` on the machine holds the same map with the access recipes.

## Getting MAST data

FAIR-MAST is a public S3 bucket, `s3://mast` at `https://s3.echo.stfc.ac.uk`,
read anonymously:

- Metadata: `fairmast-meta/{l1,l2}_shots.parquet`, `{l1,l2}_sources.parquet`,
  `{l1,l2}_signals/signals_<shot>.parquet` (pandas, no credentials). Level 1 has
  15,969 shots (campaigns M5–M9), level 2 has 11,573. The old `mastapp.site`
  JSON API answers 404.
- Data: one Zarr per shot, `level{1,2}/shots/<shot>.zarr`, one group per source
  (level 2 groups follow IMAS names: `equilibrium`, `magnetics`, `pf_active`,
  `thomson_scattering`, `soft_x_rays`, ...). Level 2 is also published as one
  netCDF file per shot under `level2/netcdf/`.
- TokaMark v1 is on the same bucket (`tokamark/v1/<shot>.zarr`) and on Hugging
  Face (`UKAEA-IBM-STFC/tokamark-v1`, 58 tars, ~565 GB, unpacked by its
  `unpack_all.py`).

Measured from this machine on 2026-11-21: the STFC bucket gives 0.15 MB/s on
one stream and 0.4–0.5 MB/s however many streams are used (s5cmd with 128
workers, aria2c with 16 connections). Level 2 shot 30420 is 85 MB in 1,381 Zarr
objects as Zarr and 397 MB as netCDF, which puts a full level-2 pull at weeks.
Hugging Face gave 21.5 MB/s on one file and 30–40 MB/s across 32 files once
authenticated (unauthenticated it fell to a few MB/s). TokaMark therefore comes
from Hugging Face.

Globus: Globus Connect Personal has no aarch64 build (its 3.3.2 tarball ships
only `gt_amd64`). Globus Connect Server 5.4 has arm64 packages for Ubuntu 26.04,
but it needs inbound 443 and 50000–51000, which the private address does not
give.
