#!/usr/bin/env python
"""G-ENC: compare fresh IGNITE v4 encodes against `model.frame_codes_cache`.

    python scripts/shot_design/g_enc.py --device cuda --out runs/genc_v4.json

Five reference shots cover all 15 modalities: exact match for slow/fast time series,
at least 99% token agreement for spectro/video. Requested modalities that produce
no codes fail. Actuator differences are reported per channel at float16 tolerance.
The v4 bundle ships no frame-code samples; `--cache-dir` selects the training cache.
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

from shot_design import config
from shot_design.design import actuators as act
from shot_design.design import seed
from shot_design.shotdb import ignite
from shot_design.shotdb.corpus import CorpusReader

DEFAULT_V4_SHOTS = (190000, 190090, 204346, 190735, 190736)


#: float16 holds ~3 decimal digits, so two z traces that round to the same float16 differ by at
#: most this in the units the model reads. The threshold is absolute because the criterion is
#: about the model's input, not about relative precision.
ACT_TOL = 2e-3


#: Full-shot v4 caches begin at 1.0 s and hold 219 frames.
FULL_SHOT_FRAMES = 219
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
        raise ValueError(
            f"{side}: actuators are {act.dtype}, the shipped layout is float16"
        )
    if tuple(act.shape) != (frames, N_ACTUATOR_CHANNELS):
        raise ValueError(
            f"{side}: actuators are {tuple(act.shape)}, expected "
            f"({frames}, {N_ACTUATOR_CHANNELS}) -- n_frames and the tensors disagree"
        )
    vocabs = payload["vocabs"]
    for name, codes in payload["codes"].items():
        if codes.dtype != torch.int32:
            raise ValueError(
                f"{side}: {name} codes are {codes.dtype}, the shipped layout is int32"
            )
        if codes.dim() != 2:
            raise ValueError(
                f"{side}: {name} codes have shape {tuple(codes.shape)}, expected (F, n_tok)"
            )
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
        raise ValueError(
            f"{side}: vocabs name modalities with no codes: {', '.join(sorted(extra))}"
        )
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
    encode compare its own prefix: a cache holding 4 of 219 frames agrees with the shipped file
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
        raise ValueError(
            f"requested modalities nobody has: {', '.join(sorted(unknown))}"
        )

    modalities = {}
    for name in dict.fromkeys((*ref["codes"], *got["codes"])):
        want, have = ref["codes"].get(name), got["codes"].get(name)
        if name not in asked:
            modalities[name] = {
                "requested": False,
                "equal": None,
                "agreement": None,
                "note": "not requested",
            }
            continue
        if have is None or want is None:
            # THE DEFECT this gate shipped with: a requested modality that produced nothing used
            # to land here with `equal=None`, so
            # removing `ece` from an otherwise identical cache PASSED. It is a failure.
            modalities[name] = {
                "requested": True,
                "equal": None,
                "agreement": None,
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


# Spectro/video quantiser boundaries tolerate small numerical differences.
FRACTION_THRESHOLD_LOOSE = 0.99  # spectro, video
FRACTION_THRESHOLD_STRICT = 1.0  # slowts, fastts


def fraction_verdict(result: dict, families: dict[str, str]) -> tuple[bool, list[str]]:
    """Per-family token agreement; a missing requested modality always fails."""
    reasons = []
    for name, m in result["modalities"].items():
        if not m.get("requested", True):
            continue
        if m["equal"] is None:
            reasons.append(f"{name} was requested and {m['note'] or 'is missing'}")
            continue
        family = families.get(name, "?")
        loose = family in ("spectro", "video")
        threshold = FRACTION_THRESHOLD_LOOSE if loose else FRACTION_THRESHOLD_STRICT
        agreement = m["agreement"]
        if agreement < threshold:
            n_tok, n_frm = m["n_mismatched_tokens"], m["n_frames_affected"]
            reasons.append(
                f"{name} ({family}) agreement {agreement:.4%} below {threshold:.0%} "
                f"({n_tok} tokens in {n_frm} frames, "
                f"max {m['max_per_frame']} per frame)"
            )
    return not reasons, reasons


def encode_one(
    shot: int,
    codecs,
    paths,
    out_dir: Path,
    ref: dict,
    *,
    device: str | None,
    workers: int | None = None,
    include_video: bool = True,
    allow_partial: bool = False,
) -> tuple[dict, float]:
    import torch

    started = time.perf_counter()
    path = seed.encode_frame_codes(
        shot,
        reader=CorpusReader(paths.foundation_model_processed_dir),
        device=device,
        include_video=include_video,
        out_dir=out_dir,
        n_frames=int(ref["n_frames"]),
        codecs=codecs,
        paths=paths,
        workers=workers,
        allow_partial=allow_partial,
        # The gate RE-ENCODES. Without this it would be handed production's own frame codes for
        # any shot in `model.frame_codes_cache` and compare them against the shipped cache they
        # were copied from, which passes whatever our encoder does.
        use_cache=False,
    )
    elapsed = time.perf_counter() - started
    return torch.load(path, weights_only=False, map_location="cpu"), elapsed


def print_table(
    shot: int, result: dict, elapsed: float, ok: bool, reasons: list[str]
) -> None:
    print(
        f"\n=== {shot}   {result['n_frames']} frames   {elapsed:.1f} s   "
        f"{'PASS' if ok else 'FAIL'}"
    )
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
    """Fresh-encode each shot and compare it with the dynamics training cache."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="reference frame-code cache (default: model.frame_codes_cache in "
        "ignite_modalities.yaml)",
    )
    _shots_help = (
        f"default: {' '.join(str(s) for s in DEFAULT_V4_SHOTS)} "
        "(5 shots present in the production cache)"
    )
    ap.add_argument("--shots", type=int, nargs="+", default=None, help=_shots_help)
    ap.add_argument(
        "--device", default=None, help="cuda | cpu (default: cuda when available)"
    )
    ap.add_argument(
        "--workers", type=int, default=None, help="CPU dataloader workers per codec"
    )
    ap.add_argument(
        "--no-video",
        action="store_true",
        help="skip tangtv_lower/tangtv_upper (diagnostic only)",
    )
    ap.add_argument(
        "--allow-partial",
        action="store_true",
        help="proceed when the bundle has no codec for a requested modality (DIAGNOSTIC ONLY: "
        "the modality is then recorded as not encoded and the shot fails)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="gate report (default: gates/g_enc_v4.json)",
    )
    args = ap.parse_args(argv)

    import tempfile

    import torch

    paths = config.load_paths()
    cache_dir = args.cache_dir or ignite.model_cfg().get("frame_codes_cache")
    if not cache_dir:
        print(
            "g_enc: no --cache-dir given and model.frame_codes_cache is not set in "
            "ignite_modalities.yaml",
            file=sys.stderr,
        )
        return 1
    cache_dir = Path(cache_dir)
    shots = args.shots or list(DEFAULT_V4_SHOTS)
    args.device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    bundle = ignite.bundle_dir(paths)
    families = ignite.model_cfg()["families"]
    names = seed.wanted_modalities(include_video=not args.no_video)
    codecs = ignite.load_codecs(bundle, names=list(names), device=args.device)

    report = {
        "gate": "G-ENC-v4",
        "device": args.device,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
        "bundle": str(bundle),
        "generation": ignite.model_cfg()["generation"],
        "cache_dir": str(cache_dir),
        "corpus": str(paths.foundation_model_processed_dir),
        "include_video": not args.no_video,
        "requested_modalities": list(names),
        "fraction_threshold_loose": FRACTION_THRESHOLD_LOOSE,
        "fraction_threshold_strict": FRACTION_THRESHOLD_STRICT,
        "shots": {},
    }
    failures = []
    with tempfile.TemporaryDirectory(prefix="g_enc_v4_") as tmp:
        for shot in shots:
            ref_path = cache_dir / f"{shot}.pt"
            if not ref_path.is_file():
                print(f"g_enc: no cache for {shot} at {ref_path}", file=sys.stderr)
                report["shots"][str(shot)] = {
                    "passed": False,
                    "reasons": [f"no cache at {ref_path}"],
                }
                failures.append(shot)
                continue
            ref = torch.load(ref_path, weights_only=False, map_location="cpu")
            # STEP 0C: a full shot's frame count is the CACHE's own `n_frames` -- the
            # pinned generation's own property, not this script's constant -- with
            # `FULL_SHOT_FRAMES` only as a fallback for a cache that somehow lacks the
            # key (`validate_cache` inside `compare` then raises on it, structurally,
            # rather than silently comparing a prefix).
            have_frames = ref.get("n_frames")
            expect_frames = int(have_frames) if have_frames else FULL_SHOT_FRAMES
            got, elapsed = encode_one(
                shot,
                codecs,
                paths,
                Path(tmp),
                ref,
                device=args.device,
                workers=args.workers,
                include_video=not args.no_video,
                allow_partial=args.allow_partial,
            )
            try:
                result = compare(got, ref, requested=names, expect_frames=expect_frames)
            except ValueError as exc:
                print(f"\n=== {shot}   FAIL   {exc}", file=sys.stderr)
                report["shots"][str(shot)] = {"passed": False, "reasons": [str(exc)]}
                failures.append(shot)
                continue
            ok, reasons = fraction_verdict(result, families)
            result["passed"] = ok
            result["reasons"] = reasons
            result["elapsed_s"] = round(elapsed, 1)
            report["shots"][str(shot)] = result
            print_table(shot, result, elapsed, ok, reasons)
            if not ok:
                failures.append(shot)

    report["passed"] = not failures
    report["failed_shots"] = failures
    dest = args.out or Path(paths.data_root) / "gates" / "g_enc_v4.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"\nG-ENC {'PASS' if not failures else 'FAIL'} -> {dest}")
    return 0 if not failures else 1


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())
