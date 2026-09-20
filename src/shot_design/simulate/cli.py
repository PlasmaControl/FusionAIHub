"""shot_design simulate: run one design's paired real/proposed IGNITE rollout.

Wires D1 (``core``: load the pinned dynamics checkpoint, build the real/proposed
actuator arms, roll both out from the design's own seed frames) and D2
(``decode``: put predicted tokens back into a comparable per-modality series;
``report``: write ``simulation.h5`` + panels + ``report.md``) behind one CLI
entry point. D4 submits this command through ``sbatch`` and polls the
``status.json`` this module writes, so writing that file first and last -- in
every code path, including a raised exception -- is the point of this module,
not incidental logging.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import torch

from .. import config
from ..design import actuators as act
from ..design import program, program_reference
from ..shotdb import ignite as shotdb_ignite
from . import core, decode, report

# The brief's own default list. Kept here (not duplicated in shot_design/cli.py's
# argparse setup) so there is exactly one place this default can drift.
DEFAULT_DECODE = ("filterscopes", "mhr", "mirnov", "ts_core_density", "ts_core_temp")


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _write_status(out_dir: Path, status: dict) -> None:
    """Atomic write: D4's poller must never observe a half-written file."""
    fd, temp = tempfile.mkstemp(prefix=".status-", dir=out_dir)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(status, f, indent=2)
            f.write("\n")
        os.replace(temp, out_dir / "status.json")
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _windowed_real_actuators(ref_cache: dict, prog, design_seed: dict) -> dict:
    """``ref_cache["actuators"]`` sliced to the design seed's own window.

    `program.export_ignite` cuts the seed from ``ref.cache`` at
    ``[context:display_end]`` (`program.py`'s `_evaluate`, ~L191-388):
    ``context = display_start - SEED_FRAMES`` and ``display_end`` come from
    ``program.start_s``/``program.end_s`` rounded to frames, UNCLAMPED, because
    a window that needed clamping is exactly the case `_evaluate` reports as a
    validation error -- one `export_ignite` would already have raised before
    this function is ever reached.
    """
    context = round(prog.start_s / act.FRAME_S) - program.SEED_FRAMES
    display_end = round(prog.end_s / act.FRAME_S)
    return {"actuators": ref_cache["actuators"][context:display_end]}


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

        total = args.k0 + args.n_predict
        if total > design_seed["n_frames"]:
            raise ValueError(
                f"--k0 {args.k0} + --n-predict {args.n_predict} = {total} exceeds "
                f"the design seed's {design_seed['n_frames']} frames"
            )

        ref = program_reference.reference(prog.reference_shot, paths)
        if ref.cache is None:
            raise ValueError(
                f"Reference shot {prog.reference_shot} has no prepared IGNITE "
                "cache; prepare it before simulating"
            )
        reference_cache = _windowed_real_actuators(ref.cache, prog, design_seed)

        device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model, cfg, step = core.load_dynamics(paths, device)
        cfg.k0_seed, cfg.n_predict = args.k0, args.n_predict

        real_act, prop_act = core.actuator_arms(
            reference_cache, design_seed, args.k0, args.n_predict
        )
        arms = core.run_paired(
            model, cfg, design_seed["codes"], real_act, prop_act, seed=args.seed
        )

        names = [n.strip() for n in (args.decode or "").split(",") if n.strip()]
        if not names:
            names = list(DEFAULT_DECODE)
        codecs = shotdb_ignite.load_codecs(
            shotdb_ignite.bundle_dir(paths), names, device
        )
        decoded = decode.decode_modalities(codecs, arms, names)

        mcfg = shotdb_ignite.model_cfg()
        meta = {
            "design_id": prog.id,
            "dynamics_sha256": (
                shotdb_ignite.bundle_identity(paths).get("manifest_sha256") or ""
            ),
            "codec_generation": mcfg.get("generation", "v2"),
            "window_s": [prog.start_s, prog.end_s],
            "dynamics_step": step,
        }
        report.write(
            out_dir,
            arms,
            decoded,
            meta,
            actuators={"real": real_act, "proposed": prop_act},
        )

        status.update(state="complete", report="report.md", finished=_now())
        _write_status(out_dir, status)
        return 0
    except Exception as exc:  # noqa: BLE001 -- any failure must still fail status.json
        print(f"shot_design simulate: {exc}", file=sys.stderr)
        status.update(state="failed", error=str(exc), finished=_now())
        _write_status(out_dir, status)
        return 1
