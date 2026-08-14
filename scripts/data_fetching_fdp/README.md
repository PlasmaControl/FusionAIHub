# FDP Batch Data Fetcher

Fetches DIII-D shot data through the **Fusion Data Platform** (FDP) into
per-shot HDF5 files, running on Stellar. This replaces the omega pipeline in
`../data_fetching_omega`, which ran `mdsh5`'s `read_mds` against
`atlas.gat.com` on a GA cluster and shipped results back over Globus.

Output layout is unchanged, so files from either pipeline are interchangeable
downstream.

## Environment

Everything runs in the pixi project at **`/scratch/gpfs/ps9551/fdp-demo`**,
which provides the supported FDP stack: `fdp` 0.4.2, `toksearch` 2.8.1,
`toksearch_d3d` 0.10.1, `ptdata` 2.0.15, `mdsplus-xrdcl`, `libfdpio2`,
`libjson_index_plugin`, `pelican`, and `h5py`. See `FDP.pdf` in that directory
for the tutorial these scripts build on.

Reads are served from the **Princeton Pelican cache**,
`sherpa.princeton.edu:8443`, which is on campus (1 ms TCP, 12 ms TLS) against
178/251 ms to GA's origin. Selecting it needs three variables, which
`module load d3d-cache/1.0` sets:

```
PELICAN_CLIENT_PREFERREDCACHES=https://sherpa.princeton.edu:8443
PELICAN_NEAREST_CACHE=https://sherpa.princeton.edu:8443
BEARER_TOKEN_FILE=$HOME/.fdp/token
```

That module lives under `/home/PPPL` and needs live `pppl` group membership. A
login session predating the grant will not have it, in which case `module load`
fails and the scripts set the three variables directly instead; the effect is
identical.

## Why it works the way it does

The FDP stack can read signals straight over the network: `libfdpio2` turns
MDSplus and PTDATA file operations into HTTP range requests. For a *scan* (a few
signals across many shots) that is the right tool. For a dense config it is not,
because each read pays a round trip and re-traverses the file structure.
Measured from Stellar against the warm Princeton cache:

| read | streamed | staged, then local |
| --- | --- | --- |
| `\D3D::TOP.ELECTRONS.TS...CORE:TEMP` | 49.6 s | **0.003 s** |
| `\D3D::TOP.ELECTRONS.TS...CORE:DENSITY` | 48.1 s | **0.004 s** |
| ECE `TECEF01` (PTDATA-backed TDI node) | 18.9 s | **1.34 s** |
| NB `NB21L:IFIX` | 1.30 s | **0.01 s** |

Streaming the full 1925-signal config measured 7.55 s/signal, falling to 2.88 s
as the cache warmed, projecting to 1.5-2 h per shot. So this pipeline **stages
the shot's objects once and then reads them locally**, which is the strategy
`ptdata.fsspec_provider.FsspecReader` documents for exactly this case:
*"download-and-cache ... for situations where streaming small-range reads is
impractical (e.g. cold caches, slow-start storage)."* Its own transfer goes via
`fsspec`, whose http backend buffers a whole object in memory (measured: 1.68 GB
resident for a 502 MB file, nothing written after 6 minutes), so the transfer
uses `pelican object get` instead: same objects, same cache, 13-80 MB/s.

**Nothing is staged to Stellar's disks.** Staging defaults to `/dev/shm`, which
is RAM, and is deleted when the shot finishes. A shot needs ~3-5 GB.

## Three passes, and why they are separate processes

`fetch_shot_fast.sh` runs `fetch_shot.py` three times:

| pass | environment | what it does |
| --- | --- | --- |
| 1. stage | full `fdp run` | reads the shot index, stages tree files and PTDATA objects, scans node records for the pointnames the TDI expressions read, stages those too |
| 2. MDSplus | `PTDATA_JSON_INDEX_DIR` **unset** | reads every non-PTDATA signal |
| 3. PTDATA | full `fdp run` | reads the config's pointnames from the staged objects, repairs the two PTDATA defects below, and retries what pass 2 deferred |

Pass 2 needs the federation index *absent*, or the PTDATA-backed TDI nodes read
over the network (ECE `TECEF01`: 18.9 s indexed vs 1.34 s staged). Pass 3 needs
it *present*. Toggling that variable inside one process leaves the ptdata plugin
in a broken state: it opens sockets to GA's origin, bypasses the Princeton
cache, and deadlocks in `futex`. Separate processes with clean environments
avoid the problem.

## The shot index replaces any local mapping

`PTDATA_JSON_INDEX_DIR` holds one JSON per shot (~500 kB) with two
dictionaries: `pointname_ext` (pointname to extension) and `ext_location`
(extension to object URL). For shot 200000 it lists **18007 pointnames across
102 objects**, and it is authoritative, so choosing which objects to stage is a
lookup rather than a search. Of the config's 213 pointnames, 69 exist for that
shot and span 4 objects.

The expressions matter more than the config here: the config asks for 213
pointnames, but its MDSplus nodes read **393 others** internally (ECE reads
`ecevs*`, GASFLOW timebases read `gaionoff*`, NB reads `BMC*` and header
pointnames like `NB21LEFT`). `fdp_stage.referenced_pointnames` finds them by
scanning records, which is cheap once the trees are local.

## Files

```
fetch_shot_fast.sh         # entry point: three passes for one shot
_without_index.sh          # drops the index from the environment for pass 2
fetch_shot.py              # one pass: stage and/or read and write
fdp_stage.py               # index, URL derivation, pelican transfers
mds_reader.py              # local reads + HDF5 writer (mdsh5-compatible)
fetch_shot.sh              # SLURM array worker, delegates to fetch_shot_fast.sh
submit_fetch_batches.sh    # batching driver with resume
config_fdp.yaml            # 1925 signals (same list as config_atlas.yaml)
```

## Usage

One shot:

```bash
./fetch_shot_fast.sh 200000 /path/to/output
```

A campaign: set `OUTPUT_DIR` at the top of `submit_fetch_batches.sh`, then

```bash
nohup ./submit_fetch_batches.sh > submit_fdp.log 2>&1 &
```

One array task per shot, `MAX_SUBMIT_LIMIT` concurrent, resuming from
`.completed_shots`. Note that `--mem` in `fetch_shot.sh` must cover the staging
area as well as the process, because staging is in RAM: keep
`concurrent tasks x 5 GB` inside a node's memory, or point `STAGE_PARENT` at a
filesystem.

## Output layout

Identical to the omega pipeline:

```
<shot>.h5
|-- <shot>/
    |-- D3D/                                  # tree key, upper-cased
    |   `-- \D3D::TOP....:SIGNAL/              # config path, verbatim
    |       |-- data                          # values
    |       |-- dim0 ... dimN                 # one per data dimension
    |       `-- units                         # scalar byte string
    `-- PTDATA/
        `-- MPI1A322D/                        # bare pointname, upper-cased
```

Contiguous and uncompressed. Nodes whose data is empty or 0-dimensional are not
written, because the old pipeline omitted them: neither a reference file nor a
production file contains a single one, while a config asks for 173 of them.
PTDATA is stored as `float32`, matching the old files.

## Validation

Shot 200000 fetched through this pipeline and compared dataset by dataset
against `/scratch/gpfs/ps9551/200000.h5` (omega output, same shot and config):

| | reference | this pipeline |
| --- | --- | --- |
| signals | 946 | **954** |
| in both | 943 | |
| byte-for-byte identical | | **853 / 943 (90.5 %)** |
| only in the other file | 3 | 11 |

Shot **115401** was fetched and diffed the same way after the fixes above:
**715/715 signals, zero reference-only, zero new-only, and zero real
differences** (645 byte-identical, 70 differing only in `units`, where the
reference stored blanks). Zero non-monotonic timebases and zero raw-volt signals.
760 s total. Compare case-normalised: that file carries duplicate `D3D` + `d3d`
groups.

The 3 reference-only signals are `PTDATA/IL330` (the config asks for `IL330F`
instead), `PTDATA/MPI79FA322D` (absent from the shot index, so absent for this
shot) and the TANGTV video (not in this config, and not in FDP at all). The 11
extra are 6 ICOIL toroidal-harmonic amplitudes and 5 PTDATA pointnames the old
pipeline missed.

The non-identical overlaps are benign: 69 where the reference stored blank units
and we store the real ones, 19 where we add `dim0`/`units` to signals that had
`data` alone, 7 PTDATA `dim0` dtypes, and 4 value differences (`\MHD::N1RMS` and
`N2RMS` at float32 rounding, plus the two pointnames noted below).

**AOT is recovered.** `\AOT::Q` is a pointer whose `\AOT::TOP:TREE` names
another tree (`AOT02`); staging tries every layout in `default_tree_path`, so
the target is found under `usershots`/`models` even though `codes/aot02` covers
only shot prefixes 11-16.

## Two PTDATA defects, and the one fallback that fixes both

`ptdata.PtDataReader` is what makes the staged path fast, but it gets two things
wrong that GA's TDI `PTDATA2()` gets right. Both are now detected per pointname
in `mds_reader.ptdata_defect` and repaired by re-reading through TDI:

| defect | detector | symptom |
| --- | --- | --- |
| `uncalibrated` | `units == b"vo"` | raw volts. On 115401, `IPTIPP` off 5e5x, `DSTDENP` off 1.7e16x |
| `timebase` | `dim0` not strictly increasing | multi-rate pointnames duplicate a sample at each rate change, then lag one and jump backwards |

The `timebase` detector cannot false-positive: a physical timebase always
increases, and a genuine multi-rate axis still passes. `data` is unaffected by
that defect, so only `dim0` is replaced — which keeps the *real* units the
staged reader returns and TDI reports blank. An uncalibrated read is wrong in
the values themselves, and its `b"vo"` units misdescribe the scaling, so there
the whole TDI payload replaces it.

Verified on shot 115401: TDI reproduces the reference file exactly (`dim0` and
`data` both max|d| = 0) for all six affected pointnames. The repair costs
2.5-5.4 s each and fires on ~8 of 61 pointnames, so ~20-40 s per shot.

**The fallback only works with the index present**, so it runs on pass 3 and is
skipped on the MDSplus pass. With `PTDATA_JSON_INDEX_DIR` unset, `PTDATA2()`
returns a degenerate shape `(1,)` even when the objects are staged; a TDI read
smaller than the staged one is discarded rather than written. A surviving
`defect:` entry in the run's `PTDATA fixups` line means something was written
unrepaired.

**Pass 2 cannot retry its own failures.** It runs with the index unset, and a
PTDATA-backed TDI expression node cannot be read at all in that environment --
staged objects do not help, because the index is what routes the read. Retrying
in place recovers nothing, whatever `RETRYABLE` says. Pass 2 therefore writes
its retryable failures to `--write-retry-manifest` and pass 3 picks them up with
`--retry-manifest`, reading them with the index present. On 115401 that recovers
the 12 NB beam `CURRENT`/`VOLTAGE`/`PERVEANCE`/`F0` nodes an earlier version of
this pipeline silently dropped.

`fdp_stage.split_unreadable_nodes` then removes the deferred work that provably
cannot succeed, by a local index lookup and no network. The 160 deferred signals
split three ways:

| category | count | ever recover |
| --- | --- | --- |
| literal pointname, `$SHOT`, absent from the shot index | 120 | **0** |
| no PTDATA call -- reads sibling nodes | 24 | 6 |
| shot argument is a node, not `$SHOT` | 16 | 6 |

Only the first is dropped, which took the retry from 1839 s to 666 s while
keeping all 12 recoverables. Note the argument is narrow: this is sound because
the retry resolves such a node through the very same index lookup, **not**
because the index is authoritative about absence. It is not -- see below. The
dropped names are always written to `<manifest>.skipped.json`, and
`--no-defer-filter` turns the filter off.

**The shot index is not authoritative about absence, so pass 3 keeps its retry.**
`MPI79FA322D` is absent from the index yet reads fine through `PtDataSignal` and
matches the reference byte for byte. Retrying all 152 index-absent pointnames
costs **5.6 s** total (median 2 ms: they fail on a local routing check, not a
round trip) and is exactly what takes the file from 714 to 715 signals. Never
disable it to save time.

Do **not** cap the manifest to save time. It is in config order, not ordered by
likelihood of success, so a cap is a random sample of a 12/160 hit rate and
quietly loses NB actuator channels, differently on each shot.

## Known issues

**Video is not in FDP.** There is no `tangtv` tree anywhere (`shots`, `codes`,
`usershots`, `models`); `tangtva` is an analysis tree that is empty for shot
200000, and `irtv` holds only geometry. So `config_chiron.yaml` cannot be
ported, and video still needs the omega/chiron path. Re-probed for shot 115401
(all four layouts, `tree` and `datafile`): every object misses.

**Older production files carry a duplicate tree group.** They hold both `D3D`
and `d3d` with identical signals, which is why `115401.h5` was 520 MB against
this pipeline's 362 MB. `rename_h5_groups.py` in the omega directory is what
that is for; compare case-normalised or the duplicate reads as missing data.

Deleting the lowercase copy is safe for this repo: every `tree:` value in
`modalities.yaml` is upper case (`D3D`, `IRTV`, `PTDATA`, `TANGTV`), and
`prepare_data.py` tries the name as given before falling back to lower case, so
it resolves `D3D` directly. Note the fallback is one-directional -- a config
asking for `d3d` would *not* find `D3D`. `115401.h5` has been replaced with the
deduplicated 715-signal file (2026-08-13).

**The token expires.** JWTs from `t.nationalresearchplatform.org` are
short-lived (the guide cites 28 days). `submit_fetch_batches.sh` checks that the
token exists and that `fdp ls` works before submitting, but cannot check the
remaining lifetime; a batch failing wholesale with 401/403 means it expired.
Note `fdp login` writes to `~/.fdp/cache/` and never updates `~/.fdp/token`.
