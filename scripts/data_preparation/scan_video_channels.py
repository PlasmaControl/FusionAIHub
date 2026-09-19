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
    ap.add_argument("--family", choices=("video", "spectro"), default="video",
                    help="video = the 7 tangtv cameras (default, unchanged); "
                         "spectro = the 5 STFT modalities (see main_spectro).")
    ap.add_argument("--limit", type=int, default=0,
                    help="scan only the first N shots (smoke test); 0 = all.")
    ap.add_argument("--n_probe", type=int, default=0, help="override N_PROBE (spectro).")
    ap.add_argument("--l_probe", type=int, default=0, help="override L_PROBE (spectro).")
    args = ap.parse_args()
    if args.family == "spectro":
        return main_spectro(args)

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


# ===================================================================================== #
# SPECTRO channel liveness (2026-09-03) — the SAME audit for ece / bes / co2 / mhr /
# mirnov, driven by `--family spectro`.
#
# Motivation: the spectro codec dataset (ignite.train_codec.CodecPairDataset) has NO
# per-channel validity mask at all. `_build_pair` discards `_load_signal_raw`'s nan_mask
# and guards only on (a) valid_len, (b) global finiteness, (c) a GLOBAL std over ALL
# channels. So a 40-channel ece window in which 12 channels are a zero slab passes every
# guard, and those dead channels are reconstructed, discriminated and entropy-counted as
# if they were real plasma data — exactly the defect measured on the video path.
#
# Per (shot, modality, channel) this records:
#   live       — 1 if the channel carries real (finite, non-constant) samples SOMEWHERE.
#   live_frac  — fraction of the SAMPLED time positions at which it is real.
# Per (shot, modality) it also records the record span [t0, t1] and the sample count, so
# "how much of the window map does this modality actually cover" can be answered without
# reopening 8753 files.
# ===================================================================================== #
SPECTRO_SIGNALS = {                    # name -> channels_to_use (data_loader SIGNAL_CONFIGS)
    "ece": (0, 40), "bes": (48, 64), "co2": (0, 4), "mhr": (2, 8), "mirnov": (0, 29),
}
# Probe geometry. The cost is dominated by Lustre SEEK latency (~13 ms per strided row
# read on these contiguous (C, N) datasets, measured), not by bytes, so few positions with
# a long block is far cheaper than many short ones: 4 x 8192 costs ~5 s/shot where 12 x 4096
# cost ~32 s/shot. Overridable from the CLI (--n_probe / --l_probe).
N_PROBE = 4           # time positions sampled per channel, uniform over the record
L_PROBE = 8192        # raw samples per probe (>= 8 x the 1024-pt STFT window)


def _scan_one_spectro(path):
    global N_PROBE, L_PROBE
    shot = int(os.path.basename(path).split("_")[0])
    rec = {}
    try:
        f = h5py.File(path, "r")
    except Exception:
        return shot, rec
    with f:
        for name, (c0, c1) in SPECTRO_SIGNALS.items():
            nch = c1 - c0
            entry = {"n": 0, "t0": float("nan"), "t1": float("nan"),
                     "live": [0] * nch, "live_frac": [0.0] * nch}
            g = f.get(name)
            if g is None or "ydata" not in g or "xdata" not in g:
                rec[name] = entry
                continue
            yd, xd = g["ydata"], g["xdata"]
            n = int(xd.shape[0])
            entry["n"] = n
            if n < 2 or yd.ndim != 2 or yd.shape[0] < c1:
                rec[name] = entry
                continue
            entry["t0"], entry["t1"] = float(xd[0]), float(xd[-1])
            starts = np.linspace(0, max(0, n - L_PROBE), N_PROBE).astype(np.int64)
            hits = np.zeros(nch, dtype=np.int64)
            for s in starts:
                blk = np.asarray(yd[c0:c1, s:s + L_PROBE], dtype=np.float64)
                if blk.size == 0:
                    continue
                fin = np.isfinite(blk)
                # A channel is REAL at this position iff most samples are finite AND the
                # block is not a constant plate (zero-fill / railed channel -> std 0).
                ok = fin.mean(axis=1) > 0.5
                std = np.where(fin, blk, 0.0).std(axis=1)
                hits += (ok & (std > 0.0)).astype(np.int64)
            entry["live_frac"] = (hits / max(1, len(starts))).tolist()
            entry["live"] = (hits > 0).astype(np.int64).tolist()
            rec[name] = entry
    return shot, rec


def main_spectro(args):
    global N_PROBE, L_PROBE
    if getattr(args, "n_probe", 0):
        N_PROBE = int(args.n_probe)
    if getattr(args, "l_probe", 0):
        L_PROBE = int(args.l_probe)
    files = sorted(glob.glob(os.path.join(args.data_dir, "*_processed.h5")))
    if args.limit:
        files = files[: args.limit]
    print(f"[specchan] scanning {len(files)} shots for spectro channel liveness, "
          f"workers={args.workers}", flush=True)
    results, done = {}, 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for shot, rec in ex.map(_scan_one_spectro, files, chunksize=8):
            results[shot] = rec
            done += 1
            if done % 500 == 0:
                print(f"[specchan] {done}/{len(files)}", flush=True)

    n_shots = len(results)
    print(f"\n[specchan] === per-modality summary over {n_shots} shots ===")
    for name, (c0, c1) in SPECTRO_SIGNALS.items():
        nch = c1 - c0
        empty = sum(1 for r in results.values() if r.get(name, {}).get("n", 0) < 2)
        anylive = sum(1 for r in results.values() if sum(r.get(name, {}).get("live", [])) > 0)
        alllive = sum(1 for r in results.values()
                      if sum(r.get(name, {}).get("live", [])) == nch)
        tot, tot_live, n_live_shots, span = np.zeros(nch), np.zeros(nch), 0, []
        for r in results.values():
            e = r.get(name)
            if e is None:
                continue
            lf = np.asarray(e["live_frac"], dtype=np.float64)
            if lf.size != nch:
                lf = np.zeros(nch)
            tot += lf
            if lf.sum() > 0:
                tot_live += lf
                n_live_shots += 1
                if np.isfinite(e["t0"]) and np.isfinite(e["t1"]):
                    span.append(e["t1"] - e["t0"])
        frac_all = tot.sum() / max(1, n_shots * nch)
        frac_live = tot_live.sum() / max(1, n_live_shots * nch)
        med_span = float(np.median(span)) if span else float("nan")
        print(f"  {name:7s} C={nch:3d}  empty/absent {empty:5d} ({100*empty/n_shots:5.1f}%)"
              f"  any-live {anylive:5d} ({100*anylive/n_shots:5.1f}%)"
              f"  all-{nch}-live {alllive:5d} ({100*alllive/n_shots:5.1f}%)")
        print(f"          REAL channel-slot fraction: ALL shots {frac_all:.4f}"
              f" | live shots only {frac_live:.4f} | median record span {med_span:.2f} s")
        print(f"          per-channel real fraction (all shots): "
              + " ".join(f"{v:.2f}" for v in (tot / max(1, n_shots))))
    torch.save(results, os.path.join(args.out, "spectro_channel_liveness.pt"))
    print(f"\n[specchan] wrote spectro_channel_liveness.pt to {args.out}", flush=True)


if __name__ == "__main__":
    main()
