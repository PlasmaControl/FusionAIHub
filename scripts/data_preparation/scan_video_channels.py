"""Per-channel tangtv liveness scan → per-divertor valid shot lists.

The 7 tangtv channels are frequently PARTIALLY populated: a channel is either
entirely real or entirely NaN (camera off) for a shot. The video-presence filter
only checks "any channel present", which over-counts for the split divertor model
(a shot can have a lower channel but zero upper channels). This scan records, per
shot, which of the 7 channels are LIVE (sampled middle frame is finite), then
writes per-divertor valid shot lists so the split video codecs / production train
on shots that actually have data for each divertor.

Channel map (config_chiron.yaml): ch0-2 = LODIV (lower), ch3-6 = UPDIV (upper).
Parallel over shots. Run via scripts/slurm_frontier/scan_video_channels.sbatch.
"""
import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import h5py
import numpy as np
import torch


def _scan_one(path):
    shot = int(os.path.basename(path).split("_")[0])
    live = [0] * 7
    try:
        with h5py.File(path, "r") as f:
            yd = f.get("tangtv/ydata")
            if yd is None or yd.ndim != 4 or yd.shape[0] < 7 or yd.shape[1] < 1:
                return shot, live
            mid = yd.shape[1] // 2
            for c in range(7):
                fr = np.asarray(yd[c, mid])          # one frame (H, W)
                if np.isfinite(fr).mean() > 0.5:
                    live[c] = 1
    except Exception:
        return shot, live
    return shot, live


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/lustre/orion/fus187/proj-shared/foundation_model")
    ap.add_argument("--out", default="/lustre/orion/fus187/proj-shared/foundation_model_meta")
    ap.add_argument("--workers", type=int, default=56)
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data_dir, "*_processed.h5")))
    print(f"[vidchan] scanning {len(files)} shots for tangtv channel liveness, "
          f"workers={args.workers}", flush=True)
    results = {}
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for shot, live in ex.map(_scan_one, files, chunksize=16):
            results[shot] = live
            done += 1
            if done % 1000 == 0:
                print(f"[vidchan] {done}/{len(files)}", flush=True)

    LOWER, UPPER = [0, 1, 2], [3, 4, 5, 6]
    LOWER_CORE, UPPER_CORE = [0, 2], [4, 6]   # channels actually live in practice
    per_ch = [sum(r[c] for r in results.values()) for c in range(7)]
    names = ["ch0 LODIV PAR-int", "ch1 LODIV PAR-std", "ch2 LODIV PERP",
             "ch3 UPDIV225 PERP", "ch4 UPDIV0 PERP", "ch5 UPDIV225 PAR", "ch6 UPDIV0 PAR"]
    print(f"\n[vidchan] per-channel live counts (of {len(results)} shots):")
    for c in range(7):
        print(f"   {names[c]:22s}: {per_ch[c]}")

    def valid(chs):
        return sorted(s for s, r in results.items() if any(r[c] for c in chs))

    sets = {
        "lower_any": valid(LOWER), "upper_any": valid(UPPER),
        "lower_core": valid(LOWER_CORE), "upper_core": valid(UPPER_CORE),
        "both_any": sorted(set(valid(LOWER)) & set(valid(UPPER))),
        "both_core": sorted(set(valid(LOWER_CORE)) & set(valid(UPPER_CORE))),
    }
    print("\n[vidchan] valid-shot counts:")
    for k, v in sets.items():
        print(f"   {k:12s}: {len(v)}")
        with open(os.path.join(args.out, f"shots_video_{k}.txt"), "w") as fh:
            fh.write("\n".join(map(str, v)) + "\n")
    torch.save(results, os.path.join(args.out, "video_channel_liveness.pt"))
    print(f"\n[vidchan] wrote per-divertor lists + video_channel_liveness.pt to {args.out}", flush=True)


if __name__ == "__main__":
    main()
