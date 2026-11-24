"""Turn TokEye spectrogram scores into keep decisions (``port.specquality`` verdicts).

    scripts/mast/py.sh scripts/mast/tokeye_keep.py --scores DIR [--frames CACHE] [--out DIR]

Reads ``<scores>/shots/<shot>.npz`` (from ``tokeye_spec_quality.py``) and writes, in ``--out``
(default: the scores directory):

  probes.parquet          one row per (shot, signal, probe): split, scores, class
  signals.parquet         one row per (shot, signal): probe classes, keep, mode-active frames
  frames.parquet          one row per (shot, signal, frame) of a kept signal: mode activity
  presence_tokeye.json    the frame cache's ``_presence.json`` with each dropped spectrogram
                          signal set absent; pass it as ``--presence_path`` with
                          ``--mask_absent`` so those (shot, signal) pairs leave the loss
  summary.json            counts by split, signal and class
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_FRAMES = Path("/models/intermediate/mast/stage2r/frame_codes_v1b_logspec_cam_live_da")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scores", type=Path, required=True)
    ap.add_argument("--frames", type=Path, default=DEFAULT_FRAMES)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    from tokamak_foundation_model.ignite.mast import paths
    from tokamak_foundation_model.ignite.mast.port import framecache, specquality as q

    out = paths.prepare_write_root(args.out or args.scores)
    splits = json.loads((args.frames / "splits.json").read_text())
    split_of = {str(s): k for k in ("train", "val", "test") for s in splits[k]}
    long_of = {v: k for k, v in q.SHORT.items()}

    probes, signals, frames = [], [], []
    for f in sorted((args.scores / "shots").glob("*.npz"), key=lambda p: int(p.stem)):
        r = q.load_scores(f)
        shot = f.stem
        for s in (str(x) for x in r["signals_present"]):
            classes = []
            for p in range(r[f"{s}_coh"].shape[0]):
                val = r[f"{s}_valid"][p]
                valid = float(np.nan_to_num(val, nan=0.0).mean())
                coh = r[f"{s}_coh"][p]
                fin = np.isfinite(coh)
                b_ac1, b_coh5 = r[f"{s}_band_ac1"][p], r[f"{s}_band_coh5"][p]
                ac1_max = float(np.nanmax(b_ac1)) if np.isfinite(b_ac1).any() else float("nan")
                coh5_max = float(np.nanmax(b_coh5)) if np.isfinite(b_coh5).any() else float("nan")
                c0 = q.probe_class(valid, ac1_max, coh5_max)
                c = q.probe_class(valid, ac1_max, coh5_max, coh, s)
                classes.append(c)
                probes.append(dict(
                    shot=shot, split=split_of.get(shot, "?"), sig=s, probe=p, cls=c, valid=valid,
                    rescued=bool(c0 == "noise" and c == "ok"),
                    ac1=ac1_max, coh5=coh5_max, ac1_all=float(r[f"{s}_ac1"][p]),
                    coh5_all=float(r[f"{s}_coh5"][p]), best_band=int(np.nanargmax(b_ac1)) if np.isfinite(b_ac1).any() else -1,
                    contrast_db=float(r[f"{s}_contrast_db"][p]), pickup_frac=float(r[f"{s}_pickup_frac"][p]),
                    n_pickup_rows=int(r[f"{s}_n_pickup_rows"][p]),
                    coh=float(coh[fin].mean()) if fin.any() else np.nan,
                    tra=float(np.nanmean(r[f"{s}_tra"][p])) if fin.any() else np.nan,
                ))
            keep = q.signal_verdict(classes)
            mf = q.mode_frames(r, s, classes)
            fin = np.isfinite(mf)
            signals.append(dict(
                shot=shot, split=split_of.get(shot, "?"), sig=s, keep=keep, classes=",".join(classes),
                n_ok=classes.count("ok"), n_noise=classes.count("noise"), n_absent=classes.count("absent"),
                n_frames=int(r["n_frames"]), mode_mean=float(mf[fin].mean()) if fin.any() else np.nan,
                mode_frac=float((mf[fin] >= q.MODE_COH).mean()) if fin.any() else np.nan,
            ))
            if keep:
                for i in np.flatnonzero(fin):
                    frames.append((shot, s, int(i), float(mf[i])))

    pr = pd.DataFrame(probes)
    sg = pd.DataFrame(signals)
    fr = pd.DataFrame(frames, columns=["shot", "sig", "frame", "mode"])
    pr.to_parquet(out / "probes.parquet")
    sg.to_parquet(out / "signals.parquet")
    fr.to_parquet(out / "frames.parquet")

    base = json.loads((args.frames / "_presence.json").read_text())
    pres = {shot: dict(m) for shot, m in base["presence"].items()}
    dropped = Counter()
    for row in sg.itertuples():
        name = long_of[row.sig]
        if row.shot in pres and pres[row.shot].get(name, False) and not row.keep:
            pres[row.shot][name] = False
            dropped[row.sig] += 1
    meta = json.loads((args.scores / "meta.json").read_text())
    doc = {
        "cache_dir": base["cache_dir"],
        "presence": pres,
        "tokeye": {
            "scores": str(args.scores), "ckpt_sha256": meta["ckpt_sha256"], "score_params": meta["params"],
            "verdict_params": {k: getattr(q, k) for k in ("ABSENT_VALID", "NOISE_AC1", "NOISE_COH5", "MODE_COH",
                                                          "RESCUE_P90", "RESCUE_RATIO", "RESCUE_SIGNALS")},
            "dropped": dict(dropped),
        },
    }
    framecache.save_json_atomic(doc, out / "presence_tokeye.json", indent=None)

    summ = {"n_shots": int(sg.shot.nunique()), "dropped_from_presence": dict(dropped)}
    for s in sorted(sg.sig.unique()):
        d = sg[sg.sig == s]
        p = pr[pr.sig == s]
        summ[s] = {
            "signals": int(len(d)), "keep": int(d.keep.sum()),
            "probe_classes": {k: int(v) for k, v in p.cls.value_counts().items()},
            "rescued_by_tokeye": int(p.rescued.sum()),
            "keep_by_split": {k: [int(v.keep.sum()), int(len(v))] for k, v in d.groupby("split")},
            "train_frames_kept": int(d[(d.split == "train") & d.keep].n_frames.sum()),
            "train_frames_all": int(d[d.split == "train"].n_frames.sum()),
            "kept_mode_frame_frac": float((fr[fr.sig == s]["mode"] >= q.MODE_COH).mean()) if len(fr) else None,
        }
    framecache.save_json_atomic(summ, out / "summary.json")
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
