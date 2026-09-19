"""Encode named shots into an IGNITE frame-code cache.

Peter's production cache (``ignite_production/frame_codes``) is not group-writable, so
shots that live outside it -- e.g. the 199595-199609 block, which sits in
``proj-shared/additional_data`` rather than the main corpus -- get encoded into an
EXTENDED cache: a directory of symlinks to every production entry plus real files for
the new shots. ``eval_dynamics --cache_dir`` then resolves both the reference shot and
any ``donor:<shot>`` from one place.

Two things here are load-bearing:

* ``t0_start`` MUST match the value the production cache was built with (it is recorded
  in ``_codec_manifest.json``; 0.0 for this cache). It sets the window origin, so a wrong
  value silently produces codes describing a DIFFERENT slice of the shot -- the frames
  would look plausible and be misaligned with every other shot in the cache. Read from
  the manifest, never assumed.
* The frame LAYOUT (which modalities, how many tokens each, and the FSQ vocab sizes) must
  match the existing entries exactly, because the dynamics model consumes a flat token
  frame whose slices are positional. This verifies the new entry against a reference
  entry and refuses to leave a mismatched file in place.

Usage::

    python scripts/evaluation/ignite_encode_shots.py \
        --shots 199598,199596 \
        --data-dir /lustre/orion/fus187/proj-shared/additional_data \
        --cache-dir data/outputs/ignite_cache_ext
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve()
sys.path.insert(0, str(_HERE.parents[2] / "src"))

DEFAULT_CODEC_TMPL = (
    "/lustre/orion/fus187/proj-shared/models/ignite_codecs_current/{m}/codec_best.pt"
)
DEFAULT_CACHE = "data/outputs/ignite_cache_ext"
PROD_CACHE = "/lustre/orion/fus187/proj-shared/models/ignite_production/frame_codes"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", required=True, help="comma-separated shot numbers")
    ap.add_argument("--data-dir", required=True, help="dir holding <shot>_processed.h5")
    ap.add_argument("--cache-dir", default=DEFAULT_CACHE)
    ap.add_argument("--codec-tmpl", default=DEFAULT_CODEC_TMPL)
    ap.add_argument("--reference-shot", default="200729",
                   help="existing cache entry whose layout the new entries must match")
    ap.add_argument("--t0-start", type=float, default=None,
                   help="override; default reads _codec_manifest.json in --cache-dir")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--timeout", type=int, default=1800,
                   help="per-shot encode budget; a hang becomes a logged skip")
    return ap.parse_args()


def layout_of(entry: dict) -> dict:
    """{modality: (n_tok, vocab)} — the positional contract the dynamics model depends on."""
    return {m: (int(v.shape[1]), int(entry["vocabs"][m])) for m, v in entry["codes"].items()}


def main() -> int:
    from tokamak_foundation_model.ignite.train_dynamics import (
        load_frozen_codecs, precompute_frame_codes,
    )

    args = parse_args()
    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    shots = [s.strip() for s in args.shots.split(",") if s.strip()]

    # t0_start from the manifest — the origin the rest of the cache was built with.
    t0 = args.t0_start
    if t0 is None:
        man = cache / "_codec_manifest.json"
        if not man.exists():
            raise SystemExit(f"no _codec_manifest.json in {cache}; pass --t0-start explicitly")
        t0 = float(json.loads(man.read_text()).get("t0_start", 0.0))
    print(f"[encode] t0_start={t0} (from manifest unless overridden)")

    ref_path = cache / f"{args.reference_shot}.pt"
    if not ref_path.exists():
        raise SystemExit(f"reference entry {ref_path} missing — cannot verify layout")
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_layout = layout_of(ref)
    names = sorted(ref_layout)
    print(f"[encode] target layout from shot {args.reference_shot}: {len(names)} modalities")

    codecs = load_frozen_codecs(names, repo=Path.cwd(), tmpl=args.codec_tmpl)
    missing = [n for n in names if n not in codecs]
    if missing:
        raise SystemExit(
            f"codecs did not resolve for {missing}. FROZEN_CODEC_CKPTS holds repo-relative "
            f"paths that exist only in Peter's tree — pass an absolute --codec-tmpl."
        )
    dev = torch.device(args.device)
    for _n, (c, _cfg, _fam) in codecs.items():
        c.to(dev)
    print(f"[encode] {len(codecs)} codecs loaded on {dev}")

    todo = [s for s in shots if not (cache / f"{s}.pt").exists()]
    for s in shots:
        if s not in todo:
            print(f"[encode] {s}: already in cache, skipping")
    if not todo:
        print("[encode] nothing to do")
        return 0
    for s in todo:
        h5 = Path(args.data_dir) / f"{s}_processed.h5"
        if not h5.exists():
            raise SystemExit(f"{h5} does not exist")

    n = precompute_frame_codes(todo, codecs, cache, args.data_dir,
                              t0_start=t0, shot_timeout_s=args.timeout)
    print(f"[encode] precompute cached {n}/{len(todo)}")

    # Verify each new entry against the reference layout; a mismatch is worse than a
    # missing shot because the rollout would consume misaligned token slices silently.
    rc = 0
    for s in todo:
        p = cache / f"{s}.pt"
        if not p.exists():
            print(f"[FAIL] {s}: not written (see skip reason above)")
            rc = 1
            continue
        e = torch.load(p, map_location="cpu", weights_only=False)
        lay = layout_of(e)
        if lay != ref_layout:
            diff = {k: (ref_layout.get(k), lay.get(k))
                    for k in set(ref_layout) | set(lay) if ref_layout.get(k) != lay.get(k)}
            print(f"[FAIL] {s}: layout mismatch vs {args.reference_shot}: {diff}")
            p.rename(p.with_suffix(".pt.badlayout"))
            rc = 1
            continue
        dead = [m for m, v in e["codes"].items() if torch.unique(v).numel() <= 1]
        print(f"[ok]   {s}: n_frames={e['n_frames']} act={tuple(e['actuators'].shape)} "
              f"live={len(e['codes']) - len(dead)}/{len(e['codes'])}"
              + (f" absent={dead}" if dead else ""))
        if e["n_frames"] < ref["n_frames"]:
            print(f"       NOTE only {e['n_frames']} frames vs {ref['n_frames']} in the "
                  f"reference — donor mode needs >= F frames of the shot it splices into.")
    return rc


if __name__ == "__main__":
    sys.exit(main())
