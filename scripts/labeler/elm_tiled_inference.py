#!/usr/bin/env python
"""Score `elm-ours` with 4,096 ms tiled inference, frozen weights and thresholds.

The U-Net's group normalisation takes its statistics over the whole input sequence.
Training used 4,096 ms crops; the reported predictions run each shot whole (zero-
padded to the network's length multiple), which is also the serving protocol. This
script runs the same frozen fold models over non-overlapping 4,096 ms tiles
(the last one zero-padded, as a training crop that overruns a shot is) and scores
them on the same bins with the same thresholds, so the only change is the context
each normalisation sees. No weight, threshold or bin is refitted or reselected.

Output: `outputs/labeler/elm/ours/tiled_inference.json`; tiled traces under
`$LABELER_ROOT/round4/elm/tiled_inference/`.
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch

from labeler.config import Paths, git_sha, sha256_of
from labeler.elm import compare, inputs, methods, net, score, train

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/labeler/elm/ours/tiled_inference.json"
TILE_MS = 4096
WHOLE = "elm-ours (whole shot)"
TILED = f"elm-ours ({TILE_MS:,} ms tiles)"
METRICS = ("auroc", "auprc", "f1", "precision", "recall", "false_alarm_bin_rate")


@torch.no_grad()
def tiled_trace(model, x: np.ndarray, tile_ms: int = TILE_MS) -> np.ndarray:
    """Event probability per ms from independent `tile_ms` tiles of one shot."""
    model.eval()
    cells = tile_ms * inputs.CELLS_PER_MS
    n_ms = x.shape[1] // inputs.CELLS_PER_MS
    out = np.zeros(-(-n_ms // tile_ms) * tile_ms, dtype=np.float32)
    for i, a in enumerate(range(0, x.shape[1], cells)):
        buf = np.zeros((x.shape[0], cells), dtype=np.float32)
        piece = x[:, a : a + cells]
        buf[:, : piece.shape[1]] = piece
        p = torch.sigmoid(model(torch.from_numpy(buf)[None]))[0, 0]
        out[i * tile_ms : (i + 1) * tile_ms] = p.numpy()
    return out[:n_ms]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", default="cv2")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    torch.set_num_threads(8)
    paths = Paths.from_env()
    data = train.load(paths)
    oof = methods.Oof(paths.root / "round4/elm/cv" / args.run)
    store = paths.root / "round4/elm/tiled_inference"
    store.mkdir(parents=True, exist_ok=True)
    models = {}
    tiled = {}
    for shot in sorted(data):
        k = oof.fold_of[shot]
        if k not in models:
            model = net.ElmUNet(dropout=oof.record["config"]["dropout"])
            model.load_state_dict(
                torch.load(
                    oof.dir / f"fold{k}/model.pt", map_location="cpu", weights_only=True
                )
            )
            models[k] = model
        # saved whole-shot traces are float16; store the tiled ones alike
        p = tiled_trace(models[k], data[shot].x).astype(np.float16)
        np.savez_compressed(store / f"{shot}.npz", p=p)
        tiled[shot] = p.astype(np.float32)
        print(shot, "fold", k, flush=True)
    sets = compare.load_sets(paths, data)
    record = {
        "git": git_sha(full=True),
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "script_sha256": sha256_of(Path(__file__)),
        "run": args.run,
        "tile_ms": TILE_MS,
        "protocol": "Frozen fold checkpoints and fold thresholds; independent "
        "non-overlapping tiles, the last zero-padded; the primary bins of each "
        "set. Only the context seen by group normalisation differs from the "
        "reported whole-shot inference. Nothing is refitted or reselected.",
        "serving_protocol": "Whole-shot inference: each fold model scores a shot "
        "in one pass over its zero-padded sequence; a new shot is scored by the "
        "five fold models and their mean (the saved out-of-fold trace is the "
        "single model that never saw the shot).",
        "fold_thresholds": {
            str(r["fold"]): r["threshold"] for r in oof.record["fold_records"]
        },
        "serving_threshold_rule": "mean of the five fold thresholds; not evaluated, "
        "because every reviewed shot lies in some fold's training set and the "
        "blind-test shots may not be used",
        "cohort_test_shots_used": 0,
        "sets": {},
    }
    record["serving_threshold"] = float(
        np.mean(list(record["fold_thresholds"].values()))
    )
    for tag, sdef in sets.items():
        parts = {WHOLE: [], TILED: []}
        for shot in sdef.shots:
            d, bins, cover = data[shot], sdef.bins[shot], sdef.cover[shot]
            thr = oof.threshold[shot]
            parts[WHOLE].append(
                methods.trace_part(d.spans, shot, bins, cover, oof.trace(shot)[0], thr)
            )
            parts[TILED].append(
                methods.trace_part(d.spans, shot, bins, cover, tiled[shot], thr)
            )
        summary = methods.summarise_methods(parts, score.draws(len(sdef.shots)), WHOLE)
        record["sets"][tag] = {
            "n_shots": len(sdef.shots),
            "bins": int(sum(len(p.truth) for p in parts[WHOLE])),
            "methods": {
                name: {
                    "point": {m: summary["methods"][name]["point"][m] for m in METRICS},
                    "ci95": {m: summary["methods"][name]["ci95"][m] for m in METRICS},
                }
                for name in (WHOLE, TILED)
            },
            "paired_whole_minus_tiled": {
                m: summary["paired"][f"{WHOLE} - {TILED}: {m}"]
                for m in ("auroc", "auprc", "f1")
            },
        }
    args.out.write_text(json.dumps(record, indent=1) + "\n")
    for tag, body in record["sets"].items():
        print(tag, {n: m["point"] for n, m in body["methods"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
