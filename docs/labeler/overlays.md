# TokEye overlays and build status

Status: **added 2026-10-16 (API 11; not committed).** TokEye's U-Net now draws on every
spectrogram row of every review store, not only on the AE CO2 rows. The page shows each
row's overlay state, and a missing overlay can be asked for from the page. Code:
`src/labeler/events/review/overlays.py` (files, states, queue, producer) and
`src/labeler/events/ui/overlay_routes.py` (routes). `app.py` calls them from `/api/shot`,
`/api/tokeye` and the gate.

## Files

Under `$LABELER_ROOT/overlays/tokeye/<model sha8>/<store event>/` (sha8 of the pinned
checkpoint `events/unet.py`, today `4afc3948`):

- `<shot>.npz`: `meta` (JSON: each layer's key, row, grid, y0, dy, n_y, level) and one
  `np.packbits(lit, axis=1)` array per layer, `(n_y, ceil(n / 8))`, on that row's store
  grid at the pyramid level the producer read. The AE store has one layer, `union`.
- `<shot>.json`: the state record: state, reason, the store's size and mtime when it was
  read, the model's sha256, each row's state and layer key, the SLURM job, and times.

## Which rows

An image row on a kHz axis without a modes legend is a spectrogram. Traces, mode-number
maps, ECE profiles and bolometer chords are `not_applicable`.

- **Rows of any event:** the U-Net runs on the row's own uint8 store spectrogram at the
  coarsest level with columns of at most 1 ms. The row is clipped to its own 1st and 99th
  percentiles and z-scored over the columns that hold data. It is padded to a multiple
  of 16 bins and cut into 512-column tiles that overlap by 64 (averaged when stitched, as
  in `events/masks.py`). A pixel is lit where either channel (coherent or transient)
  reaches 0.2.
- **AE** (`alfven_eigenmode` and its children) keeps `ae.seg.whole.build`: four CO2
  chords through `labeler.ae.gpu` (resample to 500 kHz, STFT, clip, standardise), the
  U-Net in 2,304-frame tiles that keep 2,048, then `clean_mask` (coherent and not
  transient), `persist_open` (20 frames), the notch (bins lit over 80 % of the record)
  and the 2-of-4 union on the page's 257 bins and the store's level-8 grid. It covers the
  whole record. There is no UCI-v1 refusal, no 0-6 s crop and no probability file.
- **Which AE layer the page shows** (`/api/tokeye` and the rows' state):
  - A shot with a `tokeye-full` layer keeps showing it until the AE gate has passed for
    this model (`gate.json` says passed). A failed, missing or unreadable gate keeps it.
  - After a pass, the page shows this layer when it is done and current, and
    `tokeye-full` otherwise.
  - A shot without a `tokeye-full` layer shows this layer as soon as it is done, whether
    the gate passed, failed or has not run. This is the owner's decision (2026-10-16
    02:18, review C N1): such a shot has no older layer to fall back on, so a failed gate
    still shows the new one.
- **The AE gate:** `python -m labeler.events.review.overlays compare` scores the new
  layer against `tokeye-full` on every shot that has one (180 today).
  - The IoU is taken over 0-6 s in the page's 0-250 kHz and 80-250 kHz bins.
  - The bar is that of `ae/full.py`: a median of at least 0.75 and at most 10 % of
    shots below 0.5, in both bands.
  - It passes only when every one of those shots is compared. A shot that is not
    (new layer not done, unreadable, or on another grid) is listed by number with its
    reason and fails the gate.
  - It writes `overlays/tokeye/<sha8>/alfven_eigenmode/gate.json`: passed, the numbers,
    the excluded shots, the shots compared and the time. With `--shots`, it compares
    that subset for a look and writes no gate. It exits 0 only on a pass.
- **What differs from `tokeye-full`.** Three choices here differ from the run that made
  `tokeye-full` (`scripts/labeler/ae_masks_full.py`), and the gate measures all three:
  - That run gave each chord one float32 pass over the whole record (`--tile-ms 0`).
    Here the tiles are cut so that passes batch.
  - CUDA uses fp16 autocast here (`--no-amp` turns it off).
  - That run cut the record at 6 s. Here the clip percentiles, the standardisation and
    the notch's 80 % are taken over the whole record.

## States

| State | When | Reason |
|---|---|---|
| done | a layer from this model, of the store as it is now (AE: or the tokeye-full layer, see above) | "TokEye from model 4afc3948", "TokEye from the tokeye-full layer (0-6 s)" |
| stale | the store was rebuilt after TokEye ran, or only an older model's layer exists | |
| queued | a request waits for a job, or its job has not reached it | "waiting for a GPU job", "GPU job J is pending", "GPU job J is running; this shot is waiting in it" |
| computing | its job is running | "GPU job J is running" |
| failed | the job failed on the shot, or ended without a result | the error, or "see runs/slurm/J.out" |
| not_computed | nothing yet | "No TokEye yet" |
| not_applicable | not a spectrogram | "not a spectrogram" |

A job that has left squeue while its shots are still queued or computing turns those
shots `failed`. squeue answers are reused for 15 s.

## Producer

The producer runs only in SLURM GPU jobs, one per list (`scripts/labeler/overlays.sbatch`;
its header holds the pilot, full-run and gate commands):

```bash
sbatch scripts/labeler/overlays.sbatch --event mhd_n1 edge_localized_mode --shots-file F
sbatch scripts/labeler/overlays.sbatch --event alfven_eigenmode --list silver
```

Seven DataLoader workers read the stores, and the AE CO2, while the GPU does the math.
Each forward pass batches tiles from many rows and shots, about 32 tiles of 512 by 512
(`--batch`). A shot already done on the same store is skipped (`--force` reruns it). A
shot without a store is skipped and counted. Nothing is fetched. `status` and `request`
print or queue the states from the command line.

## Queue and drain

- `POST /api/overlays/request` writes `overlays/queue/<store>__<shot>.req`. It does so
  only when no row is done or already waiting, and only for a shot whose store exists.
- The server's drain thread (`overlays.drain`) moves up to 8 waiting requests, oldest
  first, into `overlays/queue/batches/<stamp>/requests.txt` (validated `<store> <shot>`
  lines). The rest wait for the next job.
- It then submits `sbatch --job-name=overlays-queue --time=00:05:00 --cpus-per-task=4
  --mem=8G --export=ALL,REPO=<the server's repo> overlays.sbatch --requests <that file>`.
  This is the GPU rule's tiny-model branch: one batched job of at most 5 minutes.
- It submits only when none of this user's `overlays-queue` jobs is pending or running,
  and at most once a minute. The command line carries no text from a request.
- The job writes `finished.json` next to the batch when it ends. `queue_position` counts
  the open batches' queued shots first, then the waiting requests, oldest first.

## Routes

- `GET /api/overlays?event&shot` returns `{event, shot, rows: {<row>: <state>}}`.
- `POST /api/overlays/request {event, shot}` returns 202 `{state, queue_position}`. The
  shot must be an int. It answers 404 when the shot has no store yet.
- `GET /api/shot` (200): every `rows[i]` gains `overlay: {state, reason, elapsed_s?,
  queue_position?, updated?, layer?}`.
- `GET /api/shot` (202): `{building, progress}` plus `stage` (queued, reading, fetching,
  building or writing), `elapsed_s`, `queue_position` (0 while it runs) and `prebuilt`
  (a registered shot list holds the shot).
- `GET /api/tokeye?event&shot`: AE and its children get one layer, as before. Any other
  event gets `{rows: {<row>: layer}}` for its done rows. A layer is `{shot, layer, row?,
  grid: {t0_ms, dt_ms, n}, y0_khz, dy_khz, y0, dy, n_y, bits}`, and `bits` is base64 of
  the packed rows. The route answers 404 when there is nothing to serve.
- **Caching:**
  - `/` and `/index.html` serve the page with `?v=<sha8 of the file>` on each local
    script and stylesheet URL.
  - A static file asked for at its current `?v=` is cached for a year (`private,
    max-age=31536000, immutable`).
  - The page, `/api/*` and everything else stay `no-store`.
  - `/api/events` is computed once until the taxonomy, a roster or a saves table changes.
