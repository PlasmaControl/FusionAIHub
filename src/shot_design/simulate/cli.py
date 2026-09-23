"""shot_design simulate: an ensemble of IGNITE rollouts for one design.

Three arms from the design's own seed frames (`core.run_ensemble`): ``real`` (the reference
shot's measured actuators), ``proposed`` (the design's) on the same random numbers, and
``null`` (the real actuators on fresh ones), then decoded (`decode`), scored and written
(`report`). The UI submits this through sbatch and polls ``status.json``, so that file is
written first and last on every path, a raised exception included.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import traceback
from datetime import UTC, datetime
from pathlib import Path

import torch

from tokamak_foundation_model.ignite.sampling import SamplerConfig

from .. import config
from ..design import actuators as act
from ..design import program, program_reference
from ..shotdb import ignite as shotdb_ignite
from . import core, decode, report

__all__ = ["run"]


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_status(out_dir: Path, status: dict) -> None:
    """Atomic write: the poller must never observe a half-written file."""
    fd, temp = tempfile.mkstemp(prefix=".status-", dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status, f, indent=2)
            f.write("\n")
        os.replace(temp, out_dir / "status.json")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _window(prog) -> tuple[int, int]:
    """The design seed's frames ``[context, end)``, counted from the frame origin.

    `program.export_ignite` cuts the seed from the reference cache over the same slice:
    ``SEED_FRAMES`` before the display start, to the display end. A window that would
    need clamping never gets here; `export_ignite` has already refused it.
    """
    t0 = program_reference.frame_origin_s()
    start = round((prog.start_s - t0) / act.FRAME_S)
    return start - program.SEED_FRAMES, round((prog.end_s - t0) / act.FRAME_S)


def run(args) -> int:
    paths = config.load_paths()
    out_dir = (
        Path(args.out)
        if args.out
        else paths.data_root / "outputs" / args.ident / "simulation"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    status = {
        "state": "running",
        "started": _now(),
        "finished": None,
        "error": None,
        "report": None,
    }
    _write_status(out_dir, status)

    try:
        prog = program.load_program(args.ident, paths)
        seed_path = program.export_ignite(prog, paths)
        design_seed = torch.load(seed_path, map_location="cpu", weights_only=True)
        ref = program_reference.reference(prog.reference_shot, paths)
        if ref.cache is None:
            raise ValueError(
                f"Reference shot {prog.reference_shot} has no prepared IGNITE "
                "cache; prepare it before simulating"
            )
        context, end = _window(prog)

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, step = core.load_dynamics(paths, device)
        # cfg.max_frames is k0_seed + n_predict: read the trained horizon before both change
        room = min(design_seed["n_frames"], cfg.max_frames)
        n_predict = args.n_predict if args.n_predict is not None else room - args.k0
        if n_predict < 1 or args.k0 + n_predict > room:
            raise ValueError(
                f"--k0 {args.k0} + --n-predict {n_predict} does not fit the design seed's "
                f"{design_seed['n_frames']} frames and the checkpoint's {cfg.max_frames}"
            )
        cfg.k0_seed, cfg.n_predict = args.k0, n_predict
        cfg.maskgit_decode_steps = args.decode_steps

        real_act, prop_act = core.actuator_arms(
            {"actuators": ref.cache["actuators"][context:end]},
            design_seed,
            args.k0,
            n_predict,
        )
        arms = {
            "real": (real_act, args.seed),
            "proposed": (prop_act, args.seed),
            "null": (real_act, args.seed + args.members),
        }
        ens = core.run_ensemble(
            model,
            cfg,
            design_seed["codes"],
            arms,
            args.members,
            sampler=SamplerConfig(temperature=args.temperature),
            batch=args.batch,
            bf16=args.bf16,
        )

        names = [n.strip() for n in (args.decode or "").split(",") if n.strip()]
        codecs = shotdb_ignite.load_codecs(
            shotdb_ignite.bundle_dir(paths),
            names or [m.name for m in cfg.modalities],
            device,
        )
        feats = decode.decode_ensemble(codecs, ens)
        origin = program_reference.frame_origin_s()
        meta = {
            "design_id": prog.id,
            "bundle_manifest_sha256": (
                shotdb_ignite.bundle_identity(paths).get("manifest_sha256") or ""
            ),
            "codec_generation": shotdb_ignite.model_cfg()["generation"],
            "dynamics_step": step,
            "window_s": [prog.start_s, prog.end_s],
            "frame_origin_s": origin,
            "frame_s": act.FRAME_S,
            # rollout frame i is at t0_s + (i - k0) * frame_s
            "t0_s": round(origin + (context + args.k0) * act.FRAME_S, 6),
            "k0": args.k0,
            "n_predict": n_predict,
            "members": args.members,
            "decode_steps": args.decode_steps,
            "temperature": args.temperature,
            "precision": "bf16" if args.bf16 else "fp32",
        }
        report.write(
            out_dir,
            ens,
            feats,
            {name: family for name, (_, _, family) in codecs.items()},
            meta,
            actuators={"real": real_act, "proposed": prop_act},
        )

        status.update(state="complete", report="report.md", finished=_now())
        _write_status(out_dir, status)
        return 0
    except Exception as exc:  # noqa: BLE001 -- any failure must still fail status.json
        # status.json keeps the short message for the poller; the traceback goes to
        # stderr, the Slurm log an operator reads
        print(f"shot_design simulate: {exc}", file=sys.stderr)
        print(traceback.format_exc(), file=sys.stderr)
        status.update(state="failed", error=str(exc), finished=_now())
        _write_status(out_dir, status)
        return 1
