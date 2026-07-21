"""Rank shots by spectrogram MODE activity, per modality.

For each shot and each spectro modality (ece/co2/bes/mhr), compute the
maximum-over-channels low-frequency (0-60 kHz) mode activity, using the SAME
STFT (n_fft=1024, hop=256) + log-standardize + mode detector (`_hard` =
_spec_mode_arg thresholded) that training/rendering use. Padding windows are
naturally ~0 mode activity (flat spectrogram -> no structure over background),
so they rank low without an explicit filter.

Output: a per-modality ranked table (shot, mode_density, best_channel) + a
combined pickle. Used to pick mode-bearing shots for codec training/eval so
co2/bes/mhr are represented, not only ECE (the original 5-shot set was
ECE-selected).

Parallel over shots via ProcessPoolExecutor (one node, many cores). Run via
scripts/slurm_frontier/scan_spectro_modes.sbatch.
"""
import argparse
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "training"))

LOWF = 123  # ~60 kHz (250 kHz / 512 bins = 0.488 kHz/bin)

# per-worker globals (filled by _init)
_STATS = None
_MODS = None
_KMAP = None
_TARGET = "modes"
_ELM = None  # (prom_k, refractory_ms) for the elm target


def _init(stats_path, mods, target="modes", elm=None):
    global _STATS, _MODS, _KMAP, _TARGET, _ELM
    torch.set_num_threads(1)  # avoid BLAS oversubscription across pool workers
    _STATS = torch.load(stats_path, weights_only=False)
    _MODS = mods
    _TARGET = target
    _ELM = elm
    if target == "modes":
        from train_e2e_stage1 import _SPEC_STRUCT_K
        _KMAP = {m: _SPEC_STRUCT_K.get(m, 2.0) for m in mods}


def _scan_one(args):
    shot, data_dir, n_windows = args
    from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset
    from train_e2e_stage1 import _spec_mode_arg, _SPEC_STRUCT_GAMMA, _SPEC_STRUCT_CUT

    def _hard(x, k):
        return (_spec_mode_arg(x, k).clamp(0.0, 1.0) ** _SPEC_STRUCT_GAMMA
                > _SPEC_STRUCT_CUT).float()

    path = os.path.join(data_dir, f"{shot}_processed.h5")
    out = {m: (0.0, -1, 0) for m in _MODS}  # (mode_density, best_ch, n_windows)
    try:
        ds = TokamakMultiFileDataset(
            hdf5_paths=[path], chunk_duration_s=0.05, prediction_mode=True,
            prediction_horizon_s=0.05, step_size_s=0.01, warmup_s=1.0,
            n_fft=1024, hop_length=256, preprocessing_stats=_STATS,
            input_signals=list(_MODS), target_signals=list(_MODS),
            lengths_cache_path=None,
        )
        n = len(ds)
        if n == 0:
            return shot, out
        idxs = range(0, n, max(1, n // n_windows)) if n_windows > 0 else range(n)
        per_mod = {m: [] for m in _MODS}
        for i in idxs:
            s = ds[i]
            for m in _MODS:
                a = s["inputs"].get(m)
                if a is None:
                    continue
                per_mod[m].append(torch.nan_to_num(torch.as_tensor(a).float()))
        for m in _MODS:
            if not per_mod[m]:
                continue
            X = torch.stack(per_mod[m])                    # (W, C, F, T)
            F_ = X.shape[2]
            lf = min(LOWF, F_)
            h = _hard(X, _KMAP[m])[:, :, :lf, :]           # (W, C, lf, T)
            act = h.sum(dim=(0, 2, 3))                      # (C,)
            denom = h.shape[0] * lf * h.shape[3]
            ch = int(act.argmax())
            out[m] = (float(act[ch]) / denom, ch, X.shape[0])
    except Exception as e:
        return shot, {"__error__": str(e)}
    return shot, out


def _scan_one_elm(args):
    """Filterscope ELM-activity score = MAX-over-channels (p99 - p50) in
    standardized units: the ABSOLUTE elevation of the top ~1% of samples.

    Real ELM trains elevate ~1% of samples by a real amount (score ~1-3 on the
    Dalpha channel). All confounders collapse: flat channels ~0.04, continuous
    OSCILLATION ~0.06 (bounded, low absolute amplitude), single-spike/disruption
    ~<0.4 (only 0.1% of samples elevated, so p99 stays at baseline). Crucially this
    is scale-DEPENDENT, unlike kurtosis, which selected FLAT channels (tiny bumps
    on a near-constant baseline -> huge sigma-relative deviations -> huge kurtosis).
    Channel = argmax(p99-p50) = the ELM channel. We also record channel std."""
    shot, data_dir, _ = args
    from tokamak_foundation_model.data.multi_file_dataset import TokamakMultiFileDataset

    path = os.path.join(data_dir, f"{shot}_processed.h5")
    empty = {"filterscopes": (0.0, -1, 0, 0.0)}  # (p99-p50, best_ch, n_win, std)
    try:
        ds = TokamakMultiFileDataset(
            hdf5_paths=[path], chunk_duration_s=0.05, prediction_mode=True,
            prediction_horizon_s=0.05, step_size_s=0.05, warmup_s=1.0,
            preprocessing_stats=_STATS, input_signals=["filterscopes"],
            target_signals=["filterscopes"], lengths_cache_path=None)
        n = len(ds)
        if n == 0:
            return shot, empty
        wins = []
        for i in range(n):
            v = ds[i]["inputs"].get("filterscopes")
            if v is None:
                continue
            wins.append(torch.nan_to_num(torch.as_tensor(v).float()))  # (C, WIN)
        if not wins:
            return shot, empty
        G = torch.stack(wins).permute(1, 0, 2).reshape(wins[0].shape[0], -1).numpy()  # (C, T)
        p50 = np.percentile(G, 50, axis=1)          # (C,)
        elev = np.percentile(G, 99, axis=1) - p50    # (C,) absolute top-1% elevation
        best_ch = int(np.argmax(elev))
        return shot, {"filterscopes": (float(elev[best_ch]), best_ch, len(wins),
                                       float(G[best_ch].std()))}
    except Exception as e:
        return shot, {"__error__": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/lustre/orion/fus187/proj-shared/foundation_model")
    ap.add_argument("--stats", default="/lustre/orion/fus187/proj-shared/foundation_model_meta/preprocessing_stats.pt")
    ap.add_argument("--out", default="eval_runs/spectro_mode_scan")
    ap.add_argument("--modalities", nargs="+", default=["ece", "co2", "bes", "mhr"])
    ap.add_argument("--n_windows", type=int, default=40)
    ap.add_argument("--max_shots", type=int, default=0, help="0 = all")
    ap.add_argument("--workers", type=int, default=56)
    ap.add_argument("--target", choices=["modes", "elm"], default="modes",
                    help="modes = spectro low-freq mode density (default); "
                         "elm = filterscope ELM peak-rate (isolated-peak detection)")
    ap.add_argument("--prom_k", type=float, default=5.0,
                    help="[elm] peak prominence threshold in per-channel MAD units")
    ap.add_argument("--refractory_ms", type=float, default=1.0,
                    help="[elm] minimum spacing between ELM peaks (ms)")
    args = ap.parse_args()

    if args.target == "elm":
        args.modalities = ["filterscopes"]
    os.makedirs(args.out, exist_ok=True)
    shots = sorted(
        int(os.path.basename(p).split("_")[0])
        for p in glob.glob(os.path.join(args.data_dir, "*_processed.h5"))
    )
    if args.max_shots:
        shots = shots[: args.max_shots]
    print(f"[scan] target={args.target} {len(shots)} shots, modalities={args.modalities}, "
          f"workers={args.workers}", flush=True)

    worker = _scan_one_elm if args.target == "elm" else _scan_one
    elm = (args.prom_k, args.refractory_ms) if args.target == "elm" else None
    tasks = [(s, args.data_dir, args.n_windows) for s in shots]
    results = {}
    done = 0
    with ProcessPoolExecutor(
        max_workers=args.workers, initializer=_init,
        initargs=(args.stats, tuple(args.modalities), args.target, elm),
    ) as ex:
        for shot, res in ex.map(worker, tasks, chunksize=4):
            results[shot] = res
            done += 1
            if done % 500 == 0:
                print(f"[scan] {done}/{len(shots)}", flush=True)

    fname = "elm_scan.pt" if args.target == "elm" else "mode_scan.pt"
    torch.save(results, os.path.join(args.out, fname))
    errs = sum(1 for r in results.values() if "__error__" in r)
    present = sum(1 for r in results.values()
                  if "__error__" not in r and any(v[2] > 0 for v in r.values()))
    print(f"[scan] done. {len(results)} shots, {present} with data, {errs} errors.", flush=True)

    if args.target == "elm":
        rows = [(s, r["filterscopes"][0], r["filterscopes"][1], r["filterscopes"][2],
                 r["filterscopes"][3])
                for s, r in results.items() if "filterscopes" in r and "__error__" not in r]
        rows.sort(key=lambda x: -x[1])
        p = os.path.join(args.out, "rank_filterscopes_elm.txt")
        with open(p, "w") as fh:
            fh.write("# shot  p99_minus_p50  best_ch  n_windows  std  "
                     "(score = max-channel absolute top-1% elevation, standardized units)\n")
            for s, elev, ch, nw, sd in rows:
                fh.write(f"{s}  {elev:.4f}  {ch}  {nw}  {sd:.4f}\n")
        print(f"\n[elm] top 15 ELM shots (by p99-p50):")
        for s, elev, ch, nw, sd in rows[:15]:
            print(f"   {s}  p99-p50={elev:.3f}  ch={ch}  nwin={nw}  std={sd:.3f}")
        print(f"   -> {p}")
        return

    # per-modality ranked text tables
    for m in args.modalities:
        rows = [(s, r[m][0], r[m][1], r[m][2])
                for s, r in results.items() if m in r and "__error__" not in r]
        rows.sort(key=lambda x: -x[1])
        p = os.path.join(args.out, f"rank_{m}.txt")
        with open(p, "w") as fh:
            fh.write(f"# shot  mode_density  best_ch  n_windows   (modality={m}, 0-60kHz)\n")
            for s, dens, ch, nw in rows:
                fh.write(f"{s}  {dens:.4f}  {ch}  {nw}\n")
        top = rows[:10]
        print(f"\n[{m}] top 10 mode-shots:")
        for s, dens, ch, nw in top:
            print(f"   {s}  density={dens:.4f}  ch={ch}  nwin={nw}")
        print(f"   -> {p}")


if __name__ == "__main__":
    main()
