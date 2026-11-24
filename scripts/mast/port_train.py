"""Train dev-peter's IGNITE Phase-B dynamics model on a MAST frame-code cache.

Patches ``train_dynamics`` from outside via :func:`port.layout.patch_ignite`
(layout from ``_codec_manifest.json``, splits from ``splits.json``, exact presence
from ``_presence.json``) and forwards every CLI option unchanged to
``train_dynamics.main``.
"""

from __future__ import annotations

import argparse
import sys

from tokamak_foundation_model.ignite.mast.port import layout
from tokamak_foundation_model.ignite.mast.port.layout import td

_TOOL_FLAGS = {"--precompute", "--build_presence", "--patch_actuators", "-h", "--help"}


def run(argv: list[str] | None = None):
    """Apply MAST layout/split patches and invoke ``train_dynamics.main``."""
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if any(a == "--live_end_json" or a.startswith("--live_end_json=") for a in raw_argv):
        raise ValueError(
            "--live_end_json is not supported by port_train.py: the MAST cache uses "
            "per-shot t0 and plasma-only 5 ms frame ranges, and the upstream "
            "frame_s/t0_start live_end path is not ported."
        )

    # Port-only options, consumed here and not forwarded (off unless given): repeat TokEye
    # mode-active training windows (port.curate; frames.parquet from tokeye_keep.py).
    own = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    own.add_argument("--tokeye_frames", default=None)
    own.add_argument("--tokeye_factor", type=int, default=0)
    own.add_argument("--tokeye_min_active", type=int, default=4)
    mine, raw_argv = own.parse_known_args(raw_argv)

    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--cache_dir", default=None)
    known, _rest = peek.parse_known_args(raw_argv)

    is_tool_or_help = any(flag in raw_argv for flag in _TOOL_FLAGS)
    if not is_tool_or_help and "--mask_absent" not in raw_argv:
        # Never train absent MAST modalities without a presence mask.
        raw_argv = raw_argv + ["--mask_absent"]

    layout.patch_ignite(known.cache_dir)
    if mine.tokeye_frames and mine.tokeye_factor > 0:
        from tokamak_foundation_model.ignite.mast.port import curate

        curate.patch_mode_oversample(
            mine.tokeye_frames, mine.tokeye_factor, curate.train_shots_from_splits(known.cache_dir),
            min_active=mine.tokeye_min_active,
        )
    return td.main(raw_argv)


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
