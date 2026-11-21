"""Command line for the shot designer: ``<tool> [JSON arguments | @file.json]``.

Examples::

    python -m tokamak_foundation_model.ignite.designer list
    python -m tokamak_foundation_model.ignite.designer machine '{"shot": 204011}'
    python -m tokamak_foundation_model.ignite.designer describe_shot '{"shot": 204011}'
    python -m tokamak_foundation_model.ignite.designer search_precedents \\
        '{"conditions": ["torque_Nm < 1", "pnbi_MW > 5"], "near": 204011}'
    python -m tokamak_foundation_model.ignite.designer beam_options '{"shot": 204011, "t": 3.5}'
    python -m tokamak_foundation_model.ignite.designer check_plan @plan_request.json

Output is JSON on stdout.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

from . import tools


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if argv[0] == "list":
        for name, t in tools.TOOLS.items():
            print(f"{name}\n    {t['description']}\n    args: "
                  f"{json.dumps(t['schema'].get('properties', {}))}")
        return 0
    name = argv[0]
    raw = argv[1] if len(argv) > 1 else "{}"
    if raw.startswith("@"):
        raw = Path(raw[1:]).read_text()
    warnings.simplefilter("ignore")
    try:
        print(tools.to_json(tools.call(name, json.loads(raw))))
    except Exception as e:  # noqa: BLE001 - report to the agent, not a traceback
        print(tools.to_json({"error": f"{type(e).__name__}: {e}"}))
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
