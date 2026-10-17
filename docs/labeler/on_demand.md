# Shots built on demand (API 12)

The review server opens any shot of the catalog (about 4,800, `catalog` in
docs/labeler/shotlists.md) on any event, though only a few hundred have a built review
store. Opening one with no store builds it (`events.ondemand`, called from
`ui.app.Builds`), fetching what the build lacks with fdp. Only labels are stored for
good: whatever else a shot needs is rebuilt, or fetched again, when it is opened again.

## Which shots are protected

A shot builds under `$LABELER_ROOT` as ever when it is **protected**: its store is
already there, or gold, silver, the event's roster (a subcategory's: its parent's) or a
list with `protect` (registry or sidecar) holds it. Every other shot builds into the
temporary cache. Any six-digit shot that is not a blind test shot opens on any event, as
"unverified" off the roster (`ondemand.openable`); the blind shots open nowhere.

## The temporary cache

`$LABELER_ROOT/cache/ondemand/<shot>/` (`Paths.ondemand`) holds, per shot, the review
store (`spectrograms/<event>/<shot>.h5`), the feature file (`features/`) and the raw
groups fetched for it (`raw/`).

- It is written through `Paths.overlay`, which `ondemand.overlay_paths` sets while a
  shot builds. What the data root already holds of the shot (its feature file, its raw
  record) is symlinked in; the first write copies it and renames over the link, so
  the original is never written.
- Only the review server reads the cache back. `create_app` sets `Paths.read_cache`, so
  that its `features_file` and `spectrogram_file` give the data root's file, else the
  cache's, and the server serves a shot it built. Everything else (`Paths.from_env()`:
  the features pipeline, the prebuild CLI, the paper, AE and equilibrium readers) has
  `read_cache` off and sees the data root alone, so no pipeline reads or writes a
  cached shot by accident. A protected shot builds with `read_cache` off too, so an
  older cached copy never stands in for the data root's file.
- The cap is `LABELER_CACHE_GB` gigabytes (default **50**; a value that is not a
  positive number is the default). After each unprotected build `evict` removes the
  least recently opened shots until the cache fits. A shot's open time is the mtime of
  `<shot>/.opened`, touched when it is built and each time it is served.
- Never evicted: the shot just built, a shot being built, a shot opened in the last
  `GRACE_S` (300) seconds. So the cap is soft by a few shots, and a reader holding an
  older shot is not cut off. A shot evicted while its page is open answers 404 on
  the next row read; reopening rebuilds it.
- A build that fails and leaves nothing in its directory removes it.
- Size counts regular files only, not the links to the data root.
- Deleting the folder is always safe.

## The fetch

When the build stops with a `MissingFeature` (a `KeyError` that names every feature the
panel found missing, so one fetch covers all), `ondemand.fetch_features` fetches them:

1. a feature with a corpus route (`pinj_total`: the eight beam nodes summed from the
   raw `pinj` group) is read from the corpus, else from the raw cache, else the raw
   group is fetched into the cache by `raw._fetch` (MDS);
2. the rest (`ip`, `betan`, ...) come from fdp in one batched call;
3. what was resolved is merged into the shot's feature file (`write_features`, under
   the file's writer lock, as `equilibrium.signal` does, through a temporary copy named
   after the process and thread) and the build runs once more.

The archive is not used (its rows are 25 ms late). Local data is read first, so a shot
whose data is on disk opens with fetching off. There is no bulk prefetch.

**One fdp call at a time.** Every in-process toksearch call (`resolve_fdp._fetch_ptdata`
and `_fetch_mds`, which `raw`, `equilibrium.signal` and this module all go through)
holds one lock, `resolve_fdp.FETCH_LOCK`. A call waits for it up to 120 s
(`resolve_fdp.LOCK_TIMEOUT_S`) and then fails with `FdpBusy`, which is neither
retried nor followed by the next signal's wait. The build comes back as
`fetch_off` ("fdp is busy, try again"). The workspace's own fetch worker is a child
process and does not take this lock.

**No miss is recorded.** Only what was fetched is written; the feature file gets an
empty `missing`. A failure here (fetching off, auth, a timeout) belongs to the server,
not to the shot, and a recorded miss would make the features pipeline's `is_complete`
skip the feature for good. A feature that comes back with fewer than two samples is a
failure here too (`write_features` would store it as a permanent miss).

**A refused token stops the fetching.** An fdp answer that matches `fetch_worker.AUTH`
(401, 403, unauthorized, an expired token) is kept, with its message, as the cause
(the 502 `detail` has it). It latches `ondemand.auth_refused()`: no feature fetch asks
fdp, and no raw group is fetched (`raw.FETCH_GATE`), until the token's digest changes
(`fdp login` wrote another). The build then answers `fetch_off` ("fdp refused the
request, log in again") at once. A build that fails with an upstream error while the
server cannot fetch at all (not under `fdp run`, `LABELER_NO_FETCH`, an expired token)
says why in the same way. `FetchReport.refused` carries the reason.

Checked 2026-10-17 on the login node under `fdp run`: shot 194216 (catalog, no feature
file) built locked mode in 6.5 s, `ip` and `betan` from fdp, `pinj_total` from the raw
record, all under `cache/ondemand/194216/`, nothing in the data root.

## What the page gets

`GET /api/shot` while a build runs: 202 with

```
{"building": true, "progress": ..., "stage": "queued" | "fetching" | "building",
 "fetching": ["pinj_total"], "note": "Fetching pinj_total",
 "elapsed_s": 4.2, "queue_position": 0, "prebuilt": false}
```

`fetching` is always a list (empty unless fetching). `note` is at most 40 characters,
for the activity bar. `stage` is `fetching` while a feature or raw group is being
fetched. The older `reading` and `writing` stages are `building` now.

On failure, 502 with `{"error", "detail", "code"}`:

| code | when | example `error` |
|---|---|---|
| `missing_signal` | a feature the shot lacks and the fetch could not get | No beam power (pinj_total) for 170659 |
| `no_data` | the shot has no such data | No co2 data for shot 170659 |
| `fetch_off` | the fetch was refused | No beam power (pinj_total) for 170659: fdp token expired |
| `build_failed` | anything else | Could not build locked mode for 170659 |

`error` has no path and no exception name; `detail` is the full text, with the fetch's
report. The next open of a failed shot retries.

`GET /api/shots/search`: see docs/labeler/shotlists.md. `GET /api/version` reports
`api: 12`; an older server has no search route.

## Known limits

- The cap is soft (see above).
- A shot's features read after its build (the workspace's signal list) come from the
  data root's feature file when that exists, which lacks what was fetched into the
  cache; the review rows already carry it.
- Page-triggered fetches (the workspace rows) use their own cache, `signals/`, and their
  own child process (see docs/labeler/workspace.md); the lock above is not theirs.
- A build holds its thread (the pool has two) while it waits for the fdp lock: up to
  120 s for each call that finds it held (a fetch that hangs can occupy both threads
  for that long, then the builds fail with "fdp is busy").
- The cache is not read-safe against eviction: a shot evicted mid-read is a 404 on the
  next row read.
- The next shot is not warmed ahead of the walk.
