#!/usr/bin/env python
"""Count chord rejection/clipping and frozen all119 inference sensitivity."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import inputs, methods, net, prepare, score, train

REPO = Path(__file__).resolve().parents[2]
NAMES = ("FS02", "FS03", "FS04", "DENV2F", "DENV3F")


def main():
    torch.set_num_threads(4)
    paths = Paths.from_env()
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4/elm/cv/cv2")
    rows, affected, signal_hashes = {}, {}, {}
    parts = {"screen-enabled": [], "screen-disabled": []}
    models = {}
    out = paths.root / "round4/elm/rejection_sensitivity"
    out.mkdir(exist_ok=True)
    for s, d in data.items():
        source = prepare.signals_dir(paths) / f"{s}.npz"
        signal_hashes[str(s)] = sha256_of(source)
        with np.load(source) as z:
            tf, fs, ti, ne = (
                z[k] for k in ("t_fs_ms", "filterscopes", "t_int_ms", "interferometer")
            )
            n = inputs.grid_length(tf, ti)
            fmax, fc = inputs.block_reduce(tf, fs, n, "max")
            dm, dc = inputs.block_reduce(ti, ne, n, "mean")
            med = np.median(np.abs(dm[:, dc > 0]), axis=1)
            bad = med > inputs.BAD_DENSITY
            density = dm / inputs.DENSITY_UNIT
            density[:, dc == 0] = np.median(density[:, dc > 0], axis=1, keepdims=True)
            hp = inputs.HP_GAIN * (density - inputs.running_mean(density, inputs.HP_S))
            raw_x = (
                inputs.channels(tf, fs, ti, ne, screen_density=False)
                if bad.any()
                else None
            )
        valid = (fc > 0) & (dc > 0)
        indices = ((d.bins.t0 + 50) * 10).astype(int)[:, None] + np.arange(500)
        shot_rows = {}
        for j, name in enumerate(NAMES):
            screened = j >= 3 and bool(bad[j - 3])
            if j < 3:
                level_clip = fmax[j] < inputs.FS_FLOOR
                hp_clip = np.zeros(n, bool)
                threshold = "log floor: native x < 1e12; no upper clip"
            else:
                k = j - 3
                level_clip = (density[k] < -3) | (density[k] > 12)
                hp_clip = (hp[k] < -10) | (hp[k] > 10)
                threshold = (
                    "native x / 1e14 outside [-3,12]; 10*high-pass outside [-10,10]"
                )
            preclip = valid & (level_clip | hp_clip)
            clipped = preclip & (not screened)
            shot_rows[name] = {
                "screened": screened,
                "median_abs_native": float(med[j - 3]) if j >= 3 else None,
                "screen_affected_bins": len(d.bins.t0) if screened else 0,
                "clipped_cells": int(clipped.sum()),
                "clipping_affected_bins": int(clipped[indices].any(axis=1).sum()),
                "pre_screen_clipping_affected_bins": int(
                    preclip[indices].any(axis=1).sum()
                ),
                "clipping_threshold": threshold,
            }
        rows[str(s)] = shot_rows
        p = oof.trace(s)[0]
        cover = methods.cover_frame(d.cov0, d.cov1)
        parts["screen-enabled"].append(
            methods.trace_part(d.spans, s, d.bins, cover, p, oof.threshold[s])
        )
        if raw_x is not None:
            k = oof.fold_of[s]
            if k not in models:
                model = net.ElmUNet(dropout=oof.record["config"]["dropout"])
                model.load_state_dict(
                    torch.load(
                        oof.dir / f"fold{k}/model.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                models[k] = model
            model = models[k]
            reproduced = (
                train.predict(model, d.x, "cpu")[0].astype(np.float16).astype(float)
            )
            delta = float(np.max(np.abs(reproduced - p)))
            if delta > 0.001:
                raise ValueError(
                    f"Frozen screened predictions do not reproduce: {s} {delta}"
                )
            # Saved primary traces are float16; use the same storage precision.
            p = train.predict(model, raw_x, "cpu")[0].astype(np.float16).astype(float)
            affected[str(s)] = {
                "chords": [NAMES[j + 3] for j in range(2) if bad[j]],
                "screened_reproduction_max_abs": delta,
            }
            np.savez_compressed(out / f"{s}.npz", p=p)
        parts["screen-disabled"].append(
            methods.trace_part(d.spans, s, d.bins, cover, p, oof.threshold[s])
        )
        print(s, "screened", bool(bad.any()), flush=True)
    totals = {}
    for name in NAMES:
        values = [r[name] for r in rows.values()]
        totals[name] = {
            "screen_shots": sum(v["screened"] for v in values),
            "screen_bins": sum(v["screen_affected_bins"] for v in values),
            "clipping_shots": sum(v["clipping_affected_bins"] > 0 for v in values),
            "clipping_bins": sum(v["clipping_affected_bins"] for v in values),
            "clipped_cells": sum(v["clipped_cells"] for v in values),
            "pre_screen_clipping_bins": sum(
                v["pre_screen_clipping_affected_bins"] for v in values
            ),
            "threshold": values[0]["clipping_threshold"],
        }
    summary = methods.summarise_methods(
        parts, score.draws(len(data)), "screen-disabled"
    )
    record = {
        "git": git_sha(),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "shots": sorted(data),
        "n_shots": len(data),
        "bins": sum(len(d.bins.t0) for d in data.values()),
        "screen_threshold": "density only: median abs native 0.1ms cell mean > 1e16",
        "count_definition": "Bins count if any valid input cell is screened or "
        "clipped. Clipping caps/floors samples; no bin is discarded. "
        "Clipping columns exclude chords already zeroed by the screen; "
        "pre-screen clipping counts also retained.",
        "sensitivity": "Original screen zero-fills both density features (not a fitted "
        "mean). Disabled screen uses raw chords with original scaling, "
        "clipping, baselines, coverage, frozen weights and thresholds. "
        "No retraining or new threshold selection.",
        "totals": totals,
        "per_shot": rows,
        "affected_shots": affected,
        "signal_sha256": signal_hashes,
        "checkpoints_sha256": {
            p.name + "/" + p.parent.name: sha256_of(p)
            for p in oof.dir.glob("fold*/model.pt")
        },
        **summary,
    }
    before, after = (record["methods"][k]["point"] for k in parts)
    record["metric_change_disabled_minus_enabled"] = {
        k: after[k] - before[k]
        for k in (
            "auroc",
            "auprc",
            "f1",
            "false_alarm_bin_rate",
            "absent_span_alarm_rate",
        )
    }
    target = REPO / "outputs/labeler/elm/ours/rejection_sensitivity.json"
    target.write_text(json.dumps(record, indent=1) + "\n")
    print("totals", totals, flush=True)
    print("change", record["metric_change_disabled_minus_enabled"], flush=True)


if __name__ == "__main__":
    main()
