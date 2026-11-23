"""Train dev-peter's IGNITE ``VideoCodec`` on MAST ``rgb`` frames under the port patch.

Works single-process (CPU or 1 GPU) and under ``torchrun --nproc_per_node N``:
every rank imports this script, applies :func:`port.camera.patched_video_codec`
before calling ``train_codec.train_video_codec``, and passes top-level picklable
:class:`port.camera.MastCameraDataset` instances into ``DataLoader`` workers.

Usage::

    CUDA_VISIBLE_DEVICES=0 scripts/mast/py.sh scripts/mast/port_train_camera_codec.py \
        --frames-dir /models/intermediate/mast/camera/frames_rgb_v1 \
        --fit /models/intermediate/mast/camera/camera_fit_v1 \
        --cache-dir /models/intermediate/mast/stage2r/frame_codes_v1b_logspec \
        --out /models/intermediate/mast/camera/codec_rgb_v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch

from tokamak_foundation_model.ignite.mast import paths
from tokamak_foundation_model.ignite.mast.port import camera
from tokamak_foundation_model.ignite.mast.port.camera import tc


class _ExplicitDeviceDDP:
    """Single-process ``_DDPState``-compatible wrapper pinning an explicit device."""

    def __init__(self, device: torch.device) -> None:
        self.rank = 0
        self.local_rank = 0
        self.world_size = 1
        self.device_index = device.index or 0
        self.distributed = False
        self._device = device

    @property
    def is_main(self) -> bool:
        return True

    @property
    def device(self) -> torch.device:
        return self._device

    def wrap(self, module: torch.nn.Module) -> torch.nn.Module:
        return module

    def barrier(self) -> None:
        return None

    def shutdown(self) -> None:
        return None


def _make_ddp(device_spec: str):
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if ws > 1:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        return tc._DDPState(backend=backend)
    if device_spec == "auto":
        return tc._DDPState()
    return _ExplicitDeviceDDP(torch.device(device_spec))


def train_camera_codec(
    *,
    frames_dir: Path,
    fit_dir: Path,
    cache_dir: Path,
    out_dir: Path,
    steps: int = 20000,
    batch_size: int = 8,
    eval_every: int = 500,
    eval_batches: int = 8,
    eval_batch_size: int = 4,
    eval_frames: int = 6,
    num_workers: int = 4,
    lr: float = 1e-3,
    seed: int = 0,
    device_spec: str = "auto",
    max_train_shots: int = 0,
    height: int = camera.CAMERA_HEIGHT,
    width: int = camera.CAMERA_WIDTH,
    patch_h: int = 20,
    patch_w: int = 20,
    d_model: int = 256,
    enc_depth: int = 6,
    dec_depth: int = 6,
    heads: int = 8,
) -> dict[str, Any]:
    """Run ``train_codec.train_video_codec`` on MAST ``rgb`` frames."""
    frames_path = Path(frames_dir)
    fit_path = Path(fit_dir)
    cdir = Path(cache_dir)

    ddp = _make_ddp(device_spec)
    try:
        out_path = Path(out_dir)
        if ddp.is_main:
            out_path = paths.prepare_write_root(out_path)
        ddp.barrier()

        cam_fit = camera.load_camera_fit(fit_path)
        splits = json.loads((cdir / "splits.json").read_text())
        train_all = sorted((str(s) for s in splits["train"]), key=lambda s: int(s))
        val_all = sorted((str(s) for s in splits["val"]), key=lambda s: int(s))

        plasma_ranges, dt_frame = camera.load_plasma_ranges(cdir, train_all + val_all)
        plasma_exps = camera.compute_plasma_exposures(
            frames_path, plasma_ranges, dt_frame, cam_fit.time_convention
        )

        train_shots = [s for s in train_all if s in plasma_exps]
        val_shots = [s for s in val_all if s in plasma_exps]
        if max_train_shots > 0:
            train_shots = train_shots[: int(max_train_shots)]
        if not train_shots:
            raise ValueError("no training shots with plasma camera exposures available")
        if not val_shots:
            raise ValueError("no validation shots with plasma camera exposures available")

        cfg = camera.default_video_codec_config(
            height=height,
            width=width,
            patch_h=patch_h,
            patch_w=patch_w,
            d_model=d_model,
            enc_depth=enc_depth,
            dec_depth=dec_depth,
            heads=heads,
        )

        if ddp.is_main:
            n_tr_exp = sum(len(plasma_exps[s]) for s in train_shots)
            n_va_exp = sum(len(plasma_exps[s]) for s in val_shots)
            print(
                f"[port_train_camera_codec] device={ddp.device} world_size={ddp.world_size} "
                f"train_shots={len(train_shots)} ({n_tr_exp} exposures) "
                f"val_shots={len(val_shots)} ({n_va_exp} exposures) "
                f"cfg=({cfg.channels}x{cfg.frames}x{cfg.height}x{cfg.width}, "
                f"patch={cfg.patch_h}x{cfg.patch_w}, n_tok={cfg.n_tok}, "
                f"d={cfg.d_model}, depth={cfg.enc_depth}/{cfg.dec_depth})",
                flush=True,
            )

        with camera.patched_video_codec(
            frames_dir=frames_path,
            s_floor=cam_fit.s_floor,
            plasma_exposures=plasma_exps,
        ):
            return tc.train_video_codec(
                cfg,
                camera.CAMERA_MODALITY,
                train_shots,
                val_shots,
                steps=int(steps),
                eval_every=int(eval_every),
                batch_size=int(batch_size),
                num_workers=int(num_workers),
                data_dir=frames_path,
                lr=float(lr),
                eval_batches=int(eval_batches),
                eval_batch_size=int(eval_batch_size),
                eval_frames=int(eval_frames),
                out_dir=out_path,
                seed=int(seed),
                ddp=ddp,
            )
    finally:
        ddp.shutdown()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser("Train IGNITE VideoCodec on MAST rgb camera frames")
    p.add_argument("--frames-dir", default=str(camera.DEFAULT_FRAMES_DIR))
    p.add_argument("--fit", default=str(camera.DEFAULT_CAMERA_FIT_DIR))
    p.add_argument("--cache-dir", default=str(camera.DEFAULT_SPLITS_PATH.parent))
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--eval-every", type=int, default=500)
    p.add_argument("--eval-batches", type=int, default=8)
    p.add_argument("--eval-batch-size", type=int, default=4)
    p.add_argument("--eval-frames", type=int, default=6)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    p.add_argument("--max-train-shots", type=int, default=0)
    p.add_argument("--height", type=int, default=camera.CAMERA_HEIGHT)
    p.add_argument("--width", type=int, default=camera.CAMERA_WIDTH)
    p.add_argument("--patch-h", type=int, default=20)
    p.add_argument("--patch-w", type=int, default=20)
    p.add_argument("--d-model", type=int, default=256)
    p.add_argument("--enc-depth", type=int, default=6)
    p.add_argument("--dec-depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    return p


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)
    train_camera_codec(
        frames_dir=Path(args.frames_dir),
        fit_dir=Path(args.fit),
        cache_dir=Path(args.cache_dir),
        out_dir=Path(args.out),
        steps=args.steps,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_batches=args.eval_batches,
        eval_batch_size=args.eval_batch_size,
        eval_frames=args.eval_frames,
        num_workers=args.num_workers,
        lr=args.lr,
        seed=args.seed,
        device_spec=args.device,
        max_train_shots=args.max_train_shots,
        height=args.height,
        width=args.width,
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        d_model=args.d_model,
        enc_depth=args.enc_depth,
        dec_depth=args.dec_depth,
        heads=args.heads,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
