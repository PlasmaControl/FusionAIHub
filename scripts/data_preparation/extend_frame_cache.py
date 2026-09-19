#!/usr/bin/env python3
"""Add or UPDATE one modality in an existing frame-code cache, without re-encoding the rest.

``train_dynamics.precompute_frame_codes`` SKIPS any shot whose ``<shot>.pt`` already exists, so
re-running it with more codecs is a no-op on a populated cache. This tool is the missing write
path: it opens each shot file, encodes only the named modality, and rewrites the file in place.

THE TRAP THIS HANDLES. ``n_frames`` in a cache file is ``min`` over ALL modalities of their
single-shot dataset length. A modality added later can be SHORTER, which shrinks that minimum — so
inserting a key without truncating every other array leaves ragged lengths that the trainer either
rejects or silently mis-indexes. Every array is therefore truncated to the new minimum, and the
number of shots where that happened is reported.

Writes are atomic (``<shot>.pt.tmp.<pid>`` then ``replace``), so an interrupted run leaves the cache
consistent rather than half-written, and concurrent shards cannot clobber each other's temporaries.

    # add mirnov once its codec is worth having
    python scripts/data_preparation/extend_frame_cache.py \
        --cache_dir /lustre/.../ignite_prod_v3/frame_codes \
        --modality mirnov --family spectro \
        --codec /lustre/.../ignite_codecs_mirnov_x/codec_best.pt

    # replace the stale fallback co2 codes with a real codec
    python scripts/data_preparation/extend_frame_cache.py ... --modality co2 --family spectro \
        --codec .../co2/codec_best.pt --update

    --dry_run   report what would change, write nothing
    --shard I/N run only shots I mod N (for a multi-rank SLURM sweep)
"""
import argparse, json, os, sys, time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import torch
from tokamak_foundation_model.ignite import train_dynamics as td
from tokamak_foundation_model.ignite import spike as _spike


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True)
    p.add_argument("--modality", required=True)
    p.add_argument("--family", required=True,
                   choices=["spectro", "video", "slowts", "fastts"])
    p.add_argument("--codec", required=True, help="checkpoint for this modality")
    p.add_argument("--data_dir", default=None)
    p.add_argument("--t0_start", type=float, default=None,
                   help="default: read from the cache manifest, else 0.0")
    p.add_argument("--update", action="store_true",
                   help="overwrite the modality if already present (default: skip such shots)")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--shard", default="0/1", help="I/N — process shots where idx %% N == I")
    a = p.parse_args()

    cache = Path(a.cache_dir)
    shots = sorted(f.stem for f in cache.glob("*.pt"))
    if not shots:
        raise SystemExit(f"no shot files in {cache}")
    i, n = (int(x) for x in a.shard.split("/"))
    mine = [s for k, s in enumerate(shots) if k % n == i]

    # t0_start MUST match the original build or frames misalign across modalities.
    t0 = a.t0_start
    man = cache / "_codec_manifest.json"
    if t0 is None:
        t0 = float(json.load(open(man)).get("t0_start", 0.0)) if man.exists() else 0.0
    data_dir = a.data_dir or _spike.DEFAULT_DATA_DIR

    codec, cfg = td._load_codec(a.family, Path(a.codec))
    vocab = int(codec.quantizer.fsq.codebook_size)
    print(f"[extend] {a.modality} ({a.family}) vocab={vocab} n_tok={getattr(cfg,'n_tok','?')} "
          f"t0_start={t0} shard {i}/{n}: {len(mine)} of {len(shots)} shots"
          f"{' DRY RUN' if a.dry_run else ''}", flush=True)

    n_done = n_skip = n_trunc = n_fail = 0
    t_start = time.time()
    for k, shot in enumerate(mine):
        fp = cache / f"{shot}.pt"
        try:
            d = torch.load(fp, map_location="cpu", weights_only=False)
            if a.modality in d["codes"] and not a.update:
                n_skip += 1
                continue

            ds = td._single_shot_dataset(a.modality, a.family, cfg, shot, data_dir, t0_start=t0)
            new_n = len(ds)
            if not new_n:
                n_skip += 1
                continue

            # nmin can only SHRINK: it is a min over modalities.
            old_n = int(d["n_frames"])
            nmin = min(old_n, new_n)
            if nmin < 8:
                n_skip += 1
                continue

            frames = []
            with torch.no_grad():
                for t in range(nmin):
                    item = ds[t]
                    x = (item[0] if isinstance(item, tuple) else item).unsqueeze(0)
                    frames.append(td.encode_flat(codec, x)[0].to(torch.int32))
            new_codes = torch.stack(frames, 0)                       # (nmin, n_tok)

            if a.dry_run:
                n_done += 1
                if nmin < old_n:
                    n_trunc += 1
                continue

            if nmin < old_n:                                          # truncate EVERYTHING
                for name in list(d["codes"]):
                    d["codes"][name] = d["codes"][name][:nmin]
                if "actuators" in d and d["actuators"] is not None:
                    d["actuators"] = d["actuators"][:nmin]
                d["n_frames"] = nmin
                n_trunc += 1

            d["codes"][a.modality] = new_codes
            d.setdefault("vocabs", {})[a.modality] = vocab

            tmp = cache / f"{shot}.pt.tmp.{os.getpid()}"
            torch.save(d, tmp)
            tmp.replace(fp)
            n_done += 1
        except Exception as e:
            n_fail += 1
            print(f"[extend] skip {shot}: {type(e).__name__}: {e}", flush=True)
        if (k + 1) % 200 == 0:
            r = (time.time() - t_start) / (k + 1)
            print(f"[extend] {k+1}/{len(mine)}  done={n_done} skip={n_skip} trunc={n_trunc} "
                  f"fail={n_fail}  {r:.2f}s/shot  eta {r*(len(mine)-k-1)/60:.0f}m", flush=True)

    print(f"[extend] DONE {a.modality}: written={n_done} skipped={n_skip} "
          f"truncated={n_trunc} failed={n_fail} in {(time.time()-t_start)/60:.1f}m", flush=True)
    if n_trunc:
        print(f"[extend] NOTE {n_trunc} shots had n_frames REDUCED — the new modality was "
              f"shorter, and all other modalities were truncated to match.", flush=True)
    if not a.dry_run and n_done:
        print("[extend] REMINDER: _codec_manifest.json is NOT updated by this tool — record the "
              "codec you used, or a later reader will trust a stale 'resolved' entry.", flush=True)


if __name__ == "__main__":
    main()
