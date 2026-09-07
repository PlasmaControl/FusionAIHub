#!/usr/bin/env python
"""G-ENC: re-encode shots from the corpus and compare with the caches shipped in the bundle.

    python scripts/ideate/g_enc.py --shots 190090 202537 204346 --device cuda

The bundle ships ten production `frame_codes/<shot>.pt` files. This gate rebuilds three of them
from `<shot>_processed.h5` with `ideate.design.seed.encode_frame_codes` and asserts:

* every non-actuator modality's codes are BIT-IDENTICAL (`torch.equal`), and
* the 88 actuator channels agree within float16 tolerance (max |dz| <= 2e-3) on >= 82 of 88,
  the failures listed by name.

That is a strong claim and it is the point: it says our encoder is the production encoder, not
something that merely resembles it, which is what makes a Phase-5 rollout on a NEW shot mean
anything at all. It earned its keep immediately -- see the trailing-pad bug at the end of this
docstring, worth ~0.04 z on twelve channels and undetectable any other way.

WHAT REPRODUCES AND WHAT DOES NOT (all ten shipped shots, 2026-09-07, V100S / CUDA 12.4 /
torch 2.6.0, against production's MI250X / ROCm 7.1 / torch 2.10):

    bit-identical on 10/10 shots   bes, ts_core_density, ts_core_temp, ts_tangential_density,
                                   ts_tangential_temp, cer_ti, cer_rot, mse, filterscopes
    bit-identical on  8/10         mhr (the two misses agree on >= 99.98 % of tokens)
    bit-identical on  4/10         co2 (misses >= 99.20 %)
    bit-identical on  3/10         ece (misses >= 99.991 %)
    bit-identical on  8/10, 6/10   tangtv_lower, tangtv_upper
    actuators, 88/88 bit-identical in float16 on 8/10; 78/88 and 77/88 within 2e-3 z on
    190735 and 190736, the residual on rmp[11] and i_coil[0:6]

WHAT WAS MEASURED ABOUT THE RESIDUAL -- AND WHAT WAS NOT. No production INPUT was ever compared
against anything. The corpus production encoded from lives on Frontier and is not reachable from
here; no hash and no value comparison of an input file is part of this evidence. Everything
below is a statement about our own arithmetic and about the SHAPE of the disagreement, and it is
deliberately weaker than the first draft of this docstring, which asserted a cause it had not
measured.

*Our arithmetic is stable to 1e-5 for the spectro codecs, and not stable at all for video.*
ece / mhr / co2 give IDENTICAL codes on cuda and on cpu, identical codes in float32 and in
float64, and not one flipped token when the codec input is perturbed by 1e-6 or 1e-5 relative.
The video codecs give none of that: float64 instead of float32 moves 0.39 % of tangtv_lower's
tokens on one GPU, cuda versus cpu moves 0.42 % / 0.88 %, and batch sizes 1 / 2 / 4 / 32 give
four different code sets. |f64 - f32| on the pre-FSQ features reaches 0.061 where the features
themselves are ~3.06: a deep 3-D conv stack into a 64000-code quantiser puts tokens on bin
boundaries, and a code that flips when the arithmetic is made MORE accurate cannot be
bit-reproduced across GPU vendors by any means available here.

*The spectro disagreement is a scatter of ISOLATED SINGLE TOKENS -- the bin-boundary signature.*
Diffing our re-encoded 204346 against the shipped file, token by token:

    ece      4 mismatched tokens of 45888, in  4 frames of 239   (one per frame: 17/115/139/221)
    mhr      9 mismatched tokens of 45888, in  9 frames of 239   (one per frame)
    co2    367 mismatched tokens of 45888, in 88 frames of 239   (median 2 per affected frame,
                                                                  75 of the 88 at <= 4)
    190090 tangtv_lower 145 tokens / 63 frames, tangtv_upper 161 / 45 -- the same scatter

A differently fetched signal perturbs contiguous regions or whole frames; four isolated tokens
spread over eleven seconds of shot do not look like that. And the perturbation sweep locates the
scale rather than excluding it: 1e-4 relative moves <= 0.03 % of tokens, i.e. ~14 of 45888 --
the same order as the 4 and 9 actually observed. So the margin sits at ~1e-4, comfortably inside
what an MI250X/ROCm FFT and conv stack differs from a V100S/CUDA one. That is CONSISTENT WITH a
cross-vendor numerics difference. It is not a demonstration of one, and no claim stronger than
that is supported by anything measured here.

*An input difference IS demonstrated -- for 190735 and 190736, and only for them.* On those two
shots the ACTUATOR block disagrees by 1.8-2.4 z on rmp[11]. That block is pure NumPy arithmetic
on raw HDF5 values: no codec, no GPU, no quantiser, nothing a vendor difference can reach. Their
tangtv_upper agrees on 0.04 % / 0.10 % of tokens as well. Those two corpus files are not what
production read. The inference does NOT transfer to 204346: its actuator block is 88/88
bit-identical to the shipped cache, which is evidence AGAINST that file being a different fetch,
and nothing measured here justifies distrusting 204346's data.

One real bug did surface here and is fixed in `design.actuators`: `CorpusReader.read` strips the
corpus's trailing all-NaN pad sample, production averaged it in as a zero, and the shortened
divisor moved all twelve `rmp` channels of 190090 by ~0.04 z. Small, systematic, and invisible
without this comparison -- which is the argument for the gate.

`--no-video` restricts the run to the twelve non-video modalities. The default keeps the
criterion as written, so a shot that does not reproduce fails visibly.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from ideate import config
from ideate.design import actuators as act
from ideate.design import seed
from ideate.shotdb import ignite
from ideate.shotdb.corpus import CorpusReader

DEFAULT_SHOTS = (190090, 202537, 204346)
#: float16 holds ~3 decimal digits, so two z traces that round to the same float16 differ by at
#: most this in the units the model reads. The threshold is absolute because the criterion is
#: about the model's input, not about relative precision.
ACT_TOL = 2e-3
ACT_MIN_PASS = 82  # of 88; the known residuals are i_coil[0:6]-shaped and shot-specific


def compare(got: dict, ref: dict) -> dict:
    """One shot's verdict: per-modality equality plus the actuator channel tally.

    Raises `ValueError` when the two caches do not cover the same number of frames. Comparing
    `min(got, ref)` frames would let a SHORT encode compare its own prefix: a cache holding 4 of
    239 frames agrees with the shipped file on all four and would be declared bit-identical.
    """
    import torch

    if int(got["n_frames"]) != int(ref["n_frames"]):
        raise ValueError(
            f"frame count mismatch: encoded {int(got['n_frames'])} frames, "
            f"the shipped cache has {int(ref['n_frames'])} -- refusing to compare a prefix"
        )
    frames = int(ref["n_frames"])
    modalities = {}
    for name, want in ref["codes"].items():
        have = got["codes"].get(name)
        if have is None:
            # Not encoded on purpose (`--no-video`). A modality that was ASKED for and produced
            # nothing never reaches here -- `encode_frame_codes` raises instead.
            modalities[name] = {"equal": None, "agreement": None, "note": "not encoded"}
            continue
        a, b = have[:frames], want[:frames]
        # The SHAPE of a disagreement is the evidence, not just its size: isolated single tokens
        # scattered one per frame and a contiguous block of frames both show up as ">= 99 % of
        # tokens agree", and they mean different things. Recorded per modality so the next reader
        # does not have to re-derive it from the saved caches (the reviewer of I8 did).
        diff = a.ne(b)
        per_frame = diff.sum(dim=tuple(range(1, diff.dim())))
        modalities[name] = {
            "equal": bool(torch.equal(a, b)),
            "agreement": float((a == b).float().mean()),
            "n_mismatched_tokens": int(diff.sum()),
            "n_frames_affected": int((per_frame > 0).sum()),
            "max_per_frame": int(per_frame.max()) if per_frame.numel() else 0,
            "note": "",
        }

    a16 = got["actuators"][:frames].float().numpy()
    b16 = ref["actuators"][:frames].float().numpy()
    delta = np.abs(a16 - b16).max(axis=0)
    names = act.CHANNEL_NAMES
    within = delta <= ACT_TOL
    return {
        "n_frames": frames,
        "modalities": modalities,
        "actuators": {
            "within_tol": int(within.sum()),
            "bit_identical": int(np.count_nonzero(np.all(a16 == b16, axis=0))),
            "max_delta_z": float(delta.max()),
            "failing": {names[i]: float(delta[i]) for i in np.flatnonzero(~within)},
        },
    }


def verdict(result: dict, video: tuple[str, ...]) -> tuple[bool, list[str]]:
    """`(passed, reasons)`; video modalities are only judged when they were encoded."""
    reasons = []
    for name, m in result["modalities"].items():
        if m["equal"] is False:
            reasons.append(
                f"{name} not bit-identical ({m['agreement']:.4%} of tokens agree; "
                f"{m['n_mismatched_tokens']} tokens in {m['n_frames_affected']} frames, "
                f"max {m['max_per_frame']} per frame)"
            )
    n_ok = result["actuators"]["within_tol"]
    if n_ok < ACT_MIN_PASS:
        reasons.append(f"actuators within {ACT_TOL:g} z on {n_ok}/88 channels (< {ACT_MIN_PASS})")
    if video:
        reasons.append(f"(video modalities included: {', '.join(video)})")
    return not [r for r in reasons if not r.startswith("(")], reasons


def encode_one(shot: int, args, codecs, paths, out_dir: Path, ref) -> tuple[dict, float]:
    import torch

    started = time.perf_counter()
    path = seed.encode_frame_codes(
        shot,
        reader=CorpusReader(paths.foundation_model_processed_dir),
        device=args.device,
        include_video=not args.no_video,
        out_dir=out_dir,
        n_frames=int(ref["n_frames"]),
        codecs=codecs,
        paths=paths,
        workers=args.workers,
    )
    elapsed = time.perf_counter() - started
    return torch.load(path, weights_only=False, map_location="cpu"), elapsed


def print_table(shot: int, result: dict, elapsed: float, ok: bool, reasons: list[str]) -> None:
    print(f"\n=== {shot}   {result['n_frames']} frames   {elapsed:.1f} s   "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"{'modality':24s} {'bit-identical':>13s} {'token agreement':>16s}")
    for name, m in result["modalities"].items():
        mark = "-" if m["equal"] is None else ("yes" if m["equal"] else "NO")
        agree = "" if m["agreement"] is None else f"{m['agreement']:.4%}"
        print(f"{name:24s} {mark:>13s} {agree:>16s}  {m['note']}".rstrip())
    a = result["actuators"]
    print(
        f"{'actuators (88 ch)':24s} {a['bit_identical']:>10d}/88 "
        f"{a['within_tol']:>10d}/88 within {ACT_TOL:g} z   max |dz| {a['max_delta_z']:.4g}"
    )
    for name, d in sorted(a["failing"].items(), key=lambda kv: -kv[1]):
        print(f"    over tolerance: {name:18s} max |dz| = {d:.4g}")
    for r in reasons:
        print(f"    {r}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shots", type=int, nargs="+", default=list(DEFAULT_SHOTS))
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda when available)")
    ap.add_argument("--workers", type=int, default=None, help="CPU dataloader workers per codec")
    ap.add_argument(
        "--no-video",
        action="store_true",
        help="skip tangtv_lower/tangtv_upper, whose codes are not reproducible across float "
        "precisions or GPU vendors (see the module docstring)",
    )
    ap.add_argument("--out-dir", type=Path, default=None, help="where the re-encoded caches go")
    ap.add_argument("--json", type=Path, default=None, help="gate report (default: gates/g_enc.json)")
    args = ap.parse_args(argv)

    import torch

    paths = config.load_paths()
    bundle = ignite.bundle_dir(paths)
    shipped = bundle / "frame_codes"
    if not shipped.is_dir():
        print(f"g_enc: the bundle at {bundle} ships no frame_codes/ to compare against", file=sys.stderr)
        return 1
    gates = Path(paths.data_root) / "gates"
    out_dir = args.out_dir or gates / "g_enc"
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    names = seed.wanted_modalities(include_video=not args.no_video)
    video = () if args.no_video else seed.VIDEO_MODALITIES
    codecs = ignite.load_codecs(bundle, names=list(names), device=args.device)

    report = {
        "gate": "G-ENC",
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "bundle": str(bundle),
        "revision": ignite.model_cfg()["revision"],
        "corpus": str(paths.foundation_model_processed_dir),
        "include_video": not args.no_video,
        "actuator_tolerance_z": ACT_TOL,
        "actuator_min_pass": ACT_MIN_PASS,
        "shots": {},
    }
    failures = []
    for shot in args.shots:
        ref_path = shipped / f"{shot}.pt"
        if not ref_path.is_file():
            print(f"g_enc: no shipped cache for {shot} at {ref_path}", file=sys.stderr)
            failures.append(shot)
            continue
        ref = torch.load(ref_path, weights_only=False, map_location="cpu")
        got, elapsed = encode_one(shot, args, codecs, paths, out_dir, ref)
        try:
            result = compare(got, ref)
        except ValueError as exc:
            print(f"\n=== {shot}   FAIL   {exc}", file=sys.stderr)
            report["shots"][str(shot)] = {"passed": False, "reasons": [str(exc)]}
            failures.append(shot)
            continue
        ok, reasons = verdict(result, video)
        result["passed"], result["reasons"], result["elapsed_s"] = ok, reasons, round(elapsed, 1)
        report["shots"][str(shot)] = result
        print_table(shot, result, elapsed, ok, reasons)
        if not ok:
            failures.append(shot)

    report["passed"] = not failures
    report["failed_shots"] = failures
    dest = args.json or gates / "g_enc.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nG-ENC {'PASS' if not failures else 'FAIL'} -> {dest}")
    return 0 if not failures else 1


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
