"""Re-draw an eval figure from its archived codes, with no rollout.

``eval_dynamics`` archives ``codes_<shot>_step<N>.npz`` beside its metrics precisely so a
render iteration need not cost a fresh 80-frame rollout -- but nothing ever read them back,
so every figure tweak meant another GPU allocation and another wait in the queue. This is
that reader.

It decodes the archived ground-truth and predicted codes through the same frozen codecs and
calls the same ``render_figure``, so the output is identical to what the original run would
have drawn, pinned to exactly the checkpoint that produced those codes.

Two things it deliberately does NOT do: re-run the model (the codes are the model's output,
already fixed), and re-derive the measured ground truth from the H5 (that path needs the raw
data dir; the decoded GT is the codec round-trip, which is the comparison that isolates the
dynamics model's error from the codec's).

    python scripts/evaluation/ignite_rerender.py \
        --run data/outputs/ignite_cases/elm_190735_real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))

DEFAULT_CODEC_TMPL = (
    "/lustre/orion/fus187/proj-shared/models/ignite_codecs_current/{m}/codec_best.pt"
)


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="an eval_dynamics --out_dir holding codes_*.npz")
    ap.add_argument("--out", default=None, help="where to write (default: alongside the codes)")
    ap.add_argument("--codec-tmpl", default=DEFAULT_CODEC_TMPL)
    ap.add_argument("--t-origin", type=float, default=0.0,
                   help="window origin in shot time; must match the cache's t0_start")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def main() -> int:
    from tokamak_foundation_model.ignite import eval_dynamics as E
    from tokamak_foundation_model.ignite.train_dynamics import load_frozen_codecs

    args = parse_args()
    run = Path(args.run)
    hits = sorted(run.glob("codes_*_step*.npz"))
    if not hits:
        raise SystemExit(f"no codes_*.npz in {run} — that run predates the archive fix, so it "
                         f"can only be re-rendered by re-running the rollout.")
    z = np.load(hits[-1], allow_pickle=True)
    K0, F, step = int(z["K0"]), int(z["F"]), int(z["step"])
    shot = str(z["shot"])
    print(f"[rerender] {hits[-1].name}: shot={shot} step={step} K0={K0} F={F}")

    names = [k[4:] for k in z.files if k.startswith("gt__")]
    codecs = load_frozen_codecs([n for n in E.EVAL_MODALITIES if n in names],
                                repo=Path.cwd(), tmpl=args.codec_tmpl)
    if not codecs:
        raise SystemExit("no codecs resolved — pass an absolute --codec-tmpl")
    dev = torch.device(args.device)
    for _n, (c, _cfg, _f) in codecs.items():
        c.to(dev)
    print(f"[rerender] {len(codecs)} codecs on {dev}: {sorted(codecs)}")

    gt_codes = {n: torch.from_numpy(z[f"gt__{n}"]) for n in codecs}
    pred_codes = {n: torch.from_numpy(z[f"pred__{n}"]) for n in codecs}
    decoded = E.decode_all(codecs, gt_codes, pred_codes, K0, F, dev)

    out = Path(args.out) if args.out else run
    out.mkdir(parents=True, exist_ok=True)
    png, pdf = E.render_figure(decoded, shot, step, 1.0, K0, F, out, t_origin=args.t_origin)
    print(f"[rerender] {png}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
