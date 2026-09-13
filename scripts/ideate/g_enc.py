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

*The video codecs are wildly unstable and the spectro codecs are stable ONLY ON SOME SHOTS.*
Video first: float64 instead of float32 moves 0.39 % of tangtv_lower's tokens on one GPU, cuda
versus cpu moves 0.42 % / 0.88 %, and batch sizes 1 / 2 / 4 / 32 give four different code sets.
|f64 - f32| on the pre-FSQ features reaches 0.061 where the features themselves are ~3.06: a
deep 3-D conv stack into a 64000-code quantiser puts tokens on bin boundaries, and a code that
flips when the arithmetic is made MORE accurate cannot be bit-reproduced across GPU vendors by
any means available here.

The spectro codecs looked stable -- on 190090 / 202537 / 204346, ece / mhr / co2 give identical
codes on cuda and on cpu, identical codes in float32 and float64, and not one flipped token when
the codec input is perturbed by 1e-6 or 1e-5 relative (1e-4 moves <= 0.03 %). That invariance is
SHOT-SPECIFIC, and this was measured afterwards on two other shots (2026-09-07, same login
node):

    185786   cuda == cpu(4 threads) == cpu(8 threads)      14/14 modalities bit-identical
    185955   cuda vs cpu(4 threads)   bes 1 token, mhr 4 tokens   (one per affected frame)
             cuda vs cpu(8 threads)   bes 4 tokens, mhr 4 tokens
             cpu(4)  vs cpu(8)        bes 3 tokens  -- SAME machine, SAME device, SAME input;
                                                       only OMP_NUM_THREADS changed

Changing a BLAS thread count reorders a reduction at the 1e-7 level and flips spectro tokens.
So the spectro codes are marginal in exactly the way the video codes are, just more rarely --
and the eliminative argument that once concluded "the spectro residual cannot be our arithmetic,
therefore it is the input" does not survive its own test being run on a second pair of shots.
(The 88 actuator channels are bit-identical in every one of those pairings; they are NumPy
arithmetic with no codec, which is why they are the half of this gate that can be trusted to
reproduce.)

*The spectro disagreement is a scatter of ISOLATED SINGLE TOKENS -- the bin-boundary signature.*
Diffing our re-encoded 204346 against the shipped file, token by token:

    ece      4 mismatched tokens of 45888, in  4 frames of 239   (one per frame: 17/115/139/221)
    mhr      9 mismatched tokens of 45888, in  9 frames of 239   (one per frame)
    co2    367 mismatched tokens of 45888, in 88 frames of 239   (median 2 per affected frame,
                                                                  75 of the 88 at <= 4)
    190090 tangtv_lower 145 tokens / 63 frames, tangtv_upper 161 / 45 -- the same scatter

A differently fetched signal perturbs contiguous regions or whole frames; four isolated tokens
spread over eleven seconds of shot do not look like that. It is the same shape as the 185955
thread-count flips above, which are unambiguously arithmetic. And the perturbation sweep locates
the scale rather than excluding it: 1e-4 relative moves <= 0.03 % of tokens, i.e. ~14 of 45888 --
the same order as the 4 and 9 actually observed. So the margin sits at ~1e-4, comfortably inside
what an MI250X/ROCm FFT and conv stack differs from a V100S/CUDA one. That is CONSISTENT WITH a
cross-vendor numerics difference. It is not a demonstration of one, and no claim stronger than
that is supported by anything measured here.

*The 190735 / 190736 residual is OUTPUT disagreement that no vendor difference explains -- and
that is as far as it goes.* On those two shots the ACTUATOR block disagrees by 1.8-2.4 z on
rmp[11]. That block is pure NumPy arithmetic on raw HDF5 values: no codec, no GPU, no quantiser,
nothing a cross-vendor FFT can reach, so whatever moved it is upstream of this module's
arithmetic. Their tangtv_upper agrees on only 0.04 % / 0.10 % of tokens as well. A different
corpus file is the readiest explanation and it is the one this docstring used to assert; the
assertion is withdrawn. It is output disagreement; input difference not established
without matched input hashes and the preprocessing provenance of both sides, and neither exists
-- production's corpus is on Frontier and no input file was ever hashed or compared (see the
paragraph above). What the residual DOES rule out is this module's own codec arithmetic. The
observation does not transfer to 204346 either way: its actuator block is 88/88 bit-identical to
the shipped cache, and nothing measured here justifies distrusting 204346's data.

One real bug did surface here and is fixed in `design.actuators`: `CorpusReader.read` strips the
corpus's trailing all-NaN pad sample, production averaged it in as a zero, and the shortened
divisor moved all twelve `rmp` channels of 190090 by ~0.04 z. Small, systematic, and invisible
without this comparison -- which is the argument for the gate.

`--no-video` restricts the run to the twelve non-video modalities. The default keeps the
criterion as written, so a shot that does not reproduce fails visibly.

WHAT COUNTS AS THE GATE, AND WHAT DOES NOT. The G-ENC gate is the three shots 190090 / 202537 /
204346 with all fourteen modalities and the 88 actuator channels. Anything narrower -- one shot,
`--no-video`, `--allow-partial` -- is a DIAGNOSTIC: it is useful, it is cheap, and it can PASS
while the gate fails, which is exactly how a one-shot CPU smoke run gets quoted as if it were
the gate. Such a run marks its report `"diagnostic": true` and says so on stdout. The gate's
standing verdict is `gates/g_enc.json`, which is FAILED (190090, 204346) and stays failed until
those two shots reproduce or a separately justified acceptance criterion replaces the current
one; a diagnostic run must never overwrite it (use `--json` to send it somewhere else).

Every REQUESTED modality must be present on both sides and bit-identical. A modality that was
asked for and produced nothing is a FAILURE, not an abstention -- see `compare`/`verdict`.
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


def is_diagnostic(shots, *, no_video: bool, allow_partial: bool) -> bool:
    """Whether a run is narrower than the gate - and so may PASS while the gate fails.

    The G-ENC gate is `DEFAULT_SHOTS`, all fourteen modalities, no codec skipped. A run over a
    different shot set, without the video codecs, or allowed to proceed with a codec missing is
    a DIAGNOSTIC: useful, cheap, and not a verdict on the gate. One rule, here, so the report's
    `"diagnostic"` field, the stdout notice and the tests cannot drift on what counts.
    """
    return bool(
        no_video or allow_partial or sorted(int(s) for s in shots) != sorted(DEFAULT_SHOTS)
    )
#: float16 holds ~3 decimal digits, so two z traces that round to the same float16 differ by at
#: most this in the units the model reads. The threshold is absolute because the criterion is
#: about the model's input, not about relative precision.
ACT_TOL = 2e-3
ACT_MIN_PASS = 82  # of 88; the known residuals are i_coil[0:6]-shaped and shot-specific


#: A full DIII-D shot is 239 frames. The gate is about full shots: two caches that agree with
#: each other over the first 8 frames agree about nothing this gate asks. `main` passes this to
#: `compare`; a diagnostic call may pass `expect_frames=None` and get only internal consistency.
FULL_SHOT_FRAMES = 239
N_ACTUATOR_CHANNELS = 88
#: The four keys `ignite_infer.validate_shot` requires, and the dtypes it requires them in.
CACHE_KEYS = ("codes", "actuators", "n_frames", "vocabs")


def validate_cache(payload: dict, side: str, expect_frames: int | None = None) -> int:
    """`payload`'s frame count, after checking it IS a frame-code cache. Raises `ValueError`.

    `compare` used to trust everything except the frame count, so a cache with int64 codes, a
    `n_frames` that disagreed with its own tensors, tokens past the end of their codebook or an
    88-channel block that had become 87 compared perfectly well against the shipped file and was
    declared bit-identical. None of those is a cache the dynamics checkpoint would load. The
    gate's claim is "our encoder is the production encoder", and a structural check is the half
    of that claim `torch.equal` cannot make.
    """
    import torch

    if set(payload) != set(CACHE_KEYS):
        raise ValueError(
            f"{side}: a frame-code cache has exactly {CACHE_KEYS}, this one has "
            f"{tuple(sorted(payload))}"
        )
    frames = int(payload["n_frames"])
    if frames < 1:
        raise ValueError(f"{side}: n_frames is {frames}")
    if expect_frames is not None and frames != expect_frames:
        raise ValueError(
            f"{side}: {frames} frames, and this gate compares full shots of {expect_frames} "
            f"-- a shorter run is a diagnostic, not the gate"
        )
    act = payload["actuators"]
    if act.dtype != torch.float16:
        raise ValueError(f"{side}: actuators are {act.dtype}, the shipped layout is float16")
    if tuple(act.shape) != (frames, N_ACTUATOR_CHANNELS):
        raise ValueError(
            f"{side}: actuators are {tuple(act.shape)}, expected "
            f"({frames}, {N_ACTUATOR_CHANNELS}) -- n_frames and the tensors disagree"
        )
    vocabs = payload["vocabs"]
    for name, codes in payload["codes"].items():
        if codes.dtype != torch.int32:
            raise ValueError(f"{side}: {name} codes are {codes.dtype}, the shipped layout is int32")
        if codes.dim() != 2:
            raise ValueError(f"{side}: {name} codes have shape {tuple(codes.shape)}, expected (F, n_tok)")
        if int(codes.shape[0]) != frames:
            raise ValueError(
                f"{side}: {name} has {int(codes.shape[0])} frames but n_frames says {frames}"
            )
        vocab = vocabs.get(name)
        if vocab is None:
            raise ValueError(f"{side}: {name} has codes but no vocab size")
        lo, hi = int(codes.min()), int(codes.max())
        if lo < 0 or hi >= int(vocab):
            raise ValueError(
                f"{side}: {name} tokens run {lo}..{hi}, outside its vocab of {int(vocab)} "
                f"-- the model would read them as another codebook's entries"
            )
    extra = set(vocabs) - set(payload["codes"])
    if extra:
        raise ValueError(f"{side}: vocabs name modalities with no codes: {', '.join(sorted(extra))}")
    return frames


def compare(
    got: dict,
    ref: dict,
    requested: tuple[str, ...] | None = None,
    expect_frames: int | None = None,
) -> dict:
    """One shot's verdict: per-modality equality plus the actuator channel tally.

    `requested` is the modality set the run ASKED for. Every one of them must be present on both
    sides and bit-identical; a modality of the shipped cache that was not requested (`--no-video`)
    is recorded as `requested: False` and judged by nobody. Defaults to the shipped cache's own
    modality list, which is the full gate.

    Raises `ValueError` when either side is not a well-formed frame-code cache, or when the two
    do not cover the same number of frames. Comparing `min(got, ref)` frames would let a SHORT
    encode compare its own prefix: a cache holding 4 of 239 frames agrees with the shipped file
    on all four and would be declared bit-identical.
    """
    import torch

    frames = validate_cache(ref, "the shipped cache", expect_frames)
    if validate_cache(got, "the re-encoded cache", expect_frames) != frames:
        raise ValueError(
            f"frame count mismatch: encoded {int(got['n_frames'])} frames, "
            f"the shipped cache has {frames} -- refusing to compare a prefix"
        )
    asked = set(tuple(ref["codes"]) if requested is None else requested)
    unknown = asked - set(ref["codes"]) - set(got["codes"])
    if unknown:
        raise ValueError(f"requested modalities nobody has: {', '.join(sorted(unknown))}")

    modalities = {}
    for name in dict.fromkeys((*ref["codes"], *got["codes"])):
        want, have = ref["codes"].get(name), got["codes"].get(name)
        if name not in asked:
            modalities[name] = {
                "requested": False, "equal": None, "agreement": None,
                "note": "not requested",
            }
            continue
        if have is None or want is None:
            # THE DEFECT this gate shipped with: a requested modality that produced nothing used
            # to land here with `equal=None`, and `verdict` rejected only `equal is False`, so
            # removing `ece` from an otherwise identical cache PASSED. It is a failure.
            modalities[name] = {
                "requested": True, "equal": None, "agreement": None,
                "note": "not encoded" if have is None else "not in the shipped cache",
            }
            continue
        if tuple(have.shape) != tuple(want.shape):
            raise ValueError(
                f"{name}: encoded shape {tuple(have.shape)} against the shipped "
                f"{tuple(want.shape)}"
            )
        if int(got["vocabs"][name]) != int(ref["vocabs"][name]):
            raise ValueError(
                f"{name}: encoded vocab {int(got['vocabs'][name])} against the shipped "
                f"{int(ref['vocabs'][name])} -- these are two different codebooks"
            )
        a, b = have[:frames], want[:frames]
        # The SHAPE of a disagreement is the evidence, not just its size: isolated single tokens
        # scattered one per frame and a contiguous block of frames both show up as ">= 99 % of
        # tokens agree", and they mean different things. Recorded per modality so the next reader
        # does not have to re-derive it from the saved caches (the reviewer of I8 did).
        diff = a.ne(b)
        per_frame = diff.sum(dim=tuple(range(1, diff.dim())))
        modalities[name] = {
            "requested": True,
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
        "requested": sorted(asked),
        "modalities": modalities,
        "actuators": {
            "within_tol": int(within.sum()),
            "bit_identical": int(np.count_nonzero(np.all(a16 == b16, axis=0))),
            "max_delta_z": float(delta.max()),
            "failing": {names[i]: float(delta[i]) for i in np.flatnonzero(~within)},
        },
    }


def verdict(result: dict, video: tuple[str, ...]) -> tuple[bool, list[str]]:
    """`(passed, reasons)`. Every REQUESTED modality must be bit-identical -- including one that
    was not encoded at all, which is the failure this gate used to pass."""
    reasons = []
    for name, m in result["modalities"].items():
        if not m.get("requested", True):
            continue
        if m["equal"] is None:
            reasons.append(f"{name} was requested and {m['note'] or 'is missing'}")
        elif m["equal"] is False:
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
        allow_partial=args.allow_partial,
    )
    elapsed = time.perf_counter() - started
    return torch.load(path, weights_only=False, map_location="cpu"), elapsed


def print_table(shot: int, result: dict, elapsed: float, ok: bool, reasons: list[str]) -> None:
    print(f"\n=== {shot}   {result['n_frames']} frames   {elapsed:.1f} s   "
          f"{'PASS' if ok else 'FAIL'}")
    print(f"{'modality':24s} {'bit-identical':>13s} {'token agreement':>16s}")
    for name, m in result["modalities"].items():
        if not m.get("requested", True):
            mark = "-"
        elif m["equal"] is None:
            mark = "MISSING"
        else:
            mark = "yes" if m["equal"] else "NO"
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
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="proceed when the bundle has no codec for a requested modality (DIAGNOSTIC ONLY: "
        "the modality is then recorded as not encoded and the shot fails)",
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
        "requested_modalities": list(names),
        "full_shot_frames": FULL_SHOT_FRAMES,
        # A run that asked for fewer than all fourteen modalities, or fewer than the three gate
        # shots, is a DIAGNOSTIC. It can pass and say nothing about the gate; the header says so
        # and the report has to as well, because the report is what gets quoted.
        "diagnostic": is_diagnostic(
            args.shots, no_video=args.no_video, allow_partial=args.allow_partial
        ),
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
            result = compare(got, ref, requested=names, expect_frames=FULL_SHOT_FRAMES)
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
    if report["diagnostic"]:
        print(
            "  DIAGNOSTIC RUN, NOT THE GATE: the G-ENC gate is the three shots "
            f"{', '.join(str(s) for s in DEFAULT_SHOTS)} with all fourteen modalities. "
            "This run's verdict does not replace the gate's."
        )
    return 0 if not failures else 1


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
