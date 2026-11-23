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
    peek = argparse.ArgumentParser(add_help=False)
    peek.add_argument("--cache_dir", default=None)
    known, _rest = peek.parse_known_args(raw_argv)

    is_tool_or_help = any(flag in raw_argv for flag in _TOOL_FLAGS)
    if not is_tool_or_help and "--mask_absent" not in raw_argv:
        # Never train absent MAST modalities without a presence mask.
        raw_argv = raw_argv + ["--mask_absent"]

    layout.patch_ignite(known.cache_dir)
    return td.main(raw_argv)


def main(argv: list[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
