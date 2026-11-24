"""Do the Mirnov modes survive the tokenizer and the rollout? TokEye mask agreement on val windows.

    scripts/mast/py.sh scripts/mast/tokeye_mode_fidelity.py --keep DIR --out DIR \
        [--ckpt RUN/dynamics_stepN.pt] [--cache-dir FRAMES] [--n-shots 12] [--windows 3]

For the val shots whose Mirnov spectrogram TokEye scores as mode-rich (``tokeye_keep.py``
output in ``--keep``), each window's truth, codec ceiling (the shot's own codes decoded) and,
with ``--ckpt``, IGNITE rollout (``k0`` true frames, real actuators) and a persistence baseline
(the last seed frame's codes held) are segmented by TokEye one by one, and the coherent masks
compared with `port.specquality.mode_agreement`. Windows start where the predicted frames are
TokEye mode-active (`port.curate.active_starts`). Only probes classed ``ok`` count.

Writes ``<out>/mode_fidelity.json`` (one row per window x probe, plus medians) and, for the
first windows, figures with the masks drawn over the spectrograms.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

DEFAULT_FRAMES = Path("/models/intermediate/mast/stage2r/frame_codes_v1b_logspec_cam_live_da")
DEFAULT_TOKEYE = Path("/models/tokeye_dl/big_tf_unet/big_tf_unet_251210.pt")


def _pick_windows(fr: pd.DataFrame, shot: str, sig: str, n_frames: int, win: int, k0: int,
                  n_win: int, mode_coh: float) -> list[int]:
    from tokamak_foundation_model.ignite.mast.port import curate

    d = fr[(fr.shot == shot) & (fr.sig == sig)]
    act = np.zeros(n_frames, dtype=bool)
    idx = d.frame.to_numpy()
    ok = idx < n_frames
    act[idx[ok]] = d["mode"].to_numpy()[ok] >= mode_coh
    starts = curate.active_starts(act, n_frames, win, k0, min_active=max(1, (win - k0) // 2))
    if not starts:
        return []
    pick = np.unique(np.linspace(0, len(starts) - 1, n_win).round().astype(int))
    return [int(starts[i]) for i in pick]


def _peak_db(power: np.ndarray) -> float:
    """Peak contrast: median over columns and 32-bin bands of (p95 - median) of 10 log10 power."""
    p = np.asarray(power, dtype=np.float64)[1:225]
    lp = 10.0 * np.log10(np.maximum(np.where(np.isfinite(p), p, np.nan), 1e-30))
    b = lp.reshape(7, 32, -1)
    v = np.nanpercentile(b, 95, axis=1) - np.nanmedian(b, axis=1)
    return float(np.nanmedian(v)) if np.isfinite(v).any() else float("nan")


def _figure(path, shot, sig, probe, start, k0, cols_t, panels, masks, scores):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axs = plt.subplots(1, n, figsize=(3.6 * n, 3.0), squeeze=False)
    ref = np.log10(np.maximum(panels[0][1], 1e-30))[:155]
    fin = np.isfinite(ref)
    lo, hi = np.percentile(ref[fin], [2, 99.5]) if fin.any() else (0, 1)
    ext = [cols_t[0] * 1e3, cols_t[-1] * 1e3, 0, 151]
    for a, (title, p), m in zip(axs[0], panels, masks):
        a.imshow(np.log10(np.maximum(p, 1e-30))[:155], aspect="auto", origin="lower", extent=ext,
                 cmap="magma", vmin=lo, vmax=hi)
        if m is not None:
            a.contour(np.linspace(ext[0], ext[1], m.shape[1]), np.linspace(0, 500 / 512 * 255, m.shape[0]),
                      m.astype(float), levels=[0.5], colors="c", linewidths=0.6)
            a.set_ylim(0, 151)
        a.set_title(f"{title}\n{scores.get(title, '')}", fontsize=8)
        a.set_xlabel("ms", fontsize=7)
        a.tick_params(labelsize=6)
    axs[0][0].set_ylabel(f"{sig.upper()} probe {probe}\nkHz", fontsize=7)
    fig.suptitle(f"shot {shot}, frames {start}+[{k0}, ...): TokEye coherent mask (cyan)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=80)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--keep", type=Path, required=True, help="tokeye_keep.py output directory")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--ckpt", type=Path, default=None)
    ap.add_argument("--tokeye", type=Path, default=DEFAULT_TOKEYE)
    ap.add_argument("--n-shots", type=int, default=12)
    ap.add_argument("--windows", type=int, default=3)
    ap.add_argument("--figures", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--temperature", type=float, default=1.0)
    # dev-peter's decode options (eval_dynamics --gate / --consensus_*), off by default.
    ap.add_argument("--gate", type=float, default=0.0)
    ap.add_argument("--consensus-n", type=int, default=1)
    ap.add_argument("--consensus-q", type=float, default=0.5)
    ap.add_argument("--consensus-mode", choices=("majority", "mean"), default="majority")
    # MaskGIT unmasking steps per frame; 0 keeps the checkpoint's (maskgit_decode_steps, 10 here).
    ap.add_argument("--decode-steps", type=int, default=0)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    from labeler.events.unet import load_unet
    from tokamak_foundation_model.ignite.mast import paths, registry
    from tokamak_foundation_model.ignite.mast.port import dalpha, framecache, layout, logspec
    from tokamak_foundation_model.ignite.mast.port import rollout as port_rollout
    from tokamak_foundation_model.ignite.mast.port import specquality as q
    from tokamak_foundation_model.ignite.mast.port.layout import ev

    out = paths.prepare_write_root(args.out)
    dev = torch.device(args.device)
    unet = load_unet(str(args.tokeye)).to(dev).eval()
    manifest = layout.load_manifest(args.cache_dir)
    dt_frame = float(manifest.get("dt_frame", framecache.DT_FRAME))
    cache_root = Path(manifest["cache_v1_root"])
    stats_sha = str(manifest.get("stats_sha256") or "") or None
    codecs, stats, _ = framecache.build_mast_codecs(
        registry.all_signals(), cache_root, Path(manifest["fit_dir"]), Path(manifest["assignment_path"]),
        logspec_fit=logspec.resolve_manifest_logspec_fit(manifest),
        dalpha_fit=dalpha.resolve_manifest_dalpha_fit(manifest),
    )
    codecs = {n: codecs[n] for n in q.SIGNALS}
    model = cfg = step = None
    if args.ckpt is not None:
        model, cfg, step = port_rollout.load_mast_model(args.ckpt, args.cache_dir, dev)
        if args.decode_steps > 0:
            # generate_frame reads model.cfg; set both in case they are different objects.
            cfg.maskgit_decode_steps = model.cfg.maskgit_decode_steps = int(args.decode_steps)
        win, k0 = int(cfg.max_frames), int(cfg.k0_seed)
    else:
        layout.patch_ignite(args.cache_dir)
        win, k0 = 32, 12

    sg = pd.read_parquet(args.keep / "signals.parquet")
    pr = pd.read_parquet(args.keep / "probes.parquet")
    fr = pd.read_parquet(args.keep / "frames.parquet")
    val = sg[(sg.split == "val") & sg.keep & (sg.sig == "omv")].sort_values("mode_mean", ascending=False)
    shots = list(val.shot.head(args.n_shots))
    print(f"[fidelity] {len(shots)} val shots, window {win} (k0 {k0}), step {step}", flush=True)

    rows, n_fig = [], 0
    for shot in shots:
        entry = ev.load_shot_cache(args.cache_dir, shot)
        n_frames, t0 = int(entry["n_frames"]), float(entry["t0"])
        starts = _pick_windows(fr, shot, "omv", n_frames, win, k0, args.windows, q.MODE_COH)
        for start in starts:
            F = min(win, n_frames - start)
            view_t0 = t0 + start * dt_frame
            gt = {n: entry["codes"][n][start:start + F] for n in q.SIGNALS if n in entry["codes"]}
            preds = {"ceiling": gt}
            if model is not None:
                g = torch.Generator(device=dev).manual_seed(args.seed)
                ro = ev.rollout_shot(model, cfg, entry, k0, args.temperature, g, dev, actuator_mode="real",
                                     cache_dir=args.cache_dir, shot=shot, start=start,
                                     decode={"gate": args.gate, "consensus_n": args.consensus_n,
                                             "consensus_q": args.consensus_q,
                                             "consensus_mode": args.consensus_mode})
                preds["rollout"] = {n: ro.pred_codes[n] for n in gt}
                pers = {}
                for n, c in gt.items():
                    c = c.clone()
                    c[k0:] = c[k0 - 1]
                    pers[n] = c
                preds["persistence"] = pers
            dec = {}
            for kind, pc in preds.items():
                view = SimpleNamespace(gt_codes=gt, pred_codes=pc, k0=k0, n_frames=F)
                d, _ = port_rollout.decode_and_score_rollout(
                    view, cache_root / "shots" / f"{shot}.h5", codecs, stats, t0=view_t0,
                    dt_frame=dt_frame, present=entry.get("present"), stats_sha256=stats_sha)
                dec[kind] = d
            for name in gt:
                s = q.SHORT[name]
                d0 = dec["ceiling"][name]
                if not d0["present"]:
                    continue
                truth = d0["truth"]
                fi = np.floor((d0["times_s"] - view_t0) / dt_frame + 1e-9).astype(int)
                sel = fi >= k0
                classes = pr[(pr.shot == shot) & (pr.sig == s)].sort_values("probe").cls.tolist()
                for p in range(truth.shape[0]):
                    if p >= len(classes) or classes[p] != "ok":
                        continue
                    row = dict(shot=shot, sig=s, probe=p, start=start, n_cols=int(sel.sum()),
                               truth_peak_db=_peak_db(truth[p][:, sel]))
                    panels, masks, scores = [("truth", truth[p][:, sel])], [], {}
                    for kind in preds:
                        other = (dec[kind][name]["ceiling"] if kind == "ceiling" else dec[kind][name]["rollout"])
                        m = q.mode_agreement(unet, truth[p][:, sel], other[p][:, sel], device=str(dev))
                        for k, v in m.items():
                            row[f"{kind}_{k}"] = v
                        row[f"{kind}_peak_db"] = _peak_db(other[p][:, sel])
                        panels.append((kind, other[p][:, sel]))
                        scores[kind] = f"F1 {m['f1']:.2f}  recall {m['recall']:.2f}"
                    rows.append(row)
                    if n_fig < args.figures and s == "omv":
                        vt = q.valid_columns(truth[p][:, sel])
                        for _, pp in panels:
                            v = q.valid_columns(pp)
                            masks.append(q.segment(unet, q.tokeye_input(pp, v), str(dev))[0] >= q.LIT)
                        scores["truth"] = f"lit {float(masks[0][:, vt].mean()):.3f}"
                        t_cols = d0["times_s"][sel]
                        _figure(out / f"fidelity_{shot}_{s}{p}_{start}.png", shot, s, p, start, k0,
                                t_cols, panels, masks, scores)
                        n_fig += 1
        print(f"[fidelity] {shot}: {len(starts)} windows, rows {len(rows)}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(out / "mode_fidelity.parquet")
    summ = {"ckpt": str(args.ckpt) if args.ckpt else None, "step": step, "k0": k0, "window": win,
            "n_shots": len(shots), "n_rows": int(len(df)), "tol_px": q.AGREE_TOL,
            "temperature": args.temperature, "gate": args.gate, "consensus_n": args.consensus_n,
            "consensus_q": args.consensus_q, "consensus_mode": args.consensus_mode,
            "decode_steps": int(cfg.maskgit_decode_steps) if cfg is not None else None}
    for s in sorted(df.sig.unique()) if len(df) else []:
        d = df[df.sig == s]
        summ[s] = {"n": int(len(d))}
        for kind in ("ceiling", "rollout", "persistence"):
            if f"{kind}_f1" in d:
                summ[s][kind] = {k: float(np.nanmedian(d[f"{kind}_{k}"])) for k in
                                 ("f1", "recall", "precision", "other_lit", "peak_db")}
        summ[s]["truth_lit"] = float(np.nanmedian(d["ceiling_truth_lit"]))
        summ[s]["truth_peak_db"] = float(np.nanmedian(d["truth_peak_db"]))
    (out / "mode_fidelity.json").write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
