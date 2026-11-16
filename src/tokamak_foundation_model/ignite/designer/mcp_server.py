"""The shot designer as a Model Context Protocol server over stdio.

Any MCP client (Claude Code, Claude Desktop, other agents) can connect; the repository's
``.mcp.json`` registers it for Claude Code. Messages are JSON-RPC 2.0, one per line.
Implemented without the ``mcp`` package (not in the pixi environments): ``initialize``,
``ping``, ``tools/list`` and ``tools/call``.

    pixi run --frozen -e stellarai python -m tokamak_foundation_model.ignite.designer.mcp_server
"""

from __future__ import annotations

import json
import sys
import traceback
import warnings

from . import tools

PROTOCOL = "2024-11-05"
INSTRUCTIONS = (
    "IGNITE shot designer for DIII-D. Call `machine` first and read its knowledge. Translate "
    "the user's goal into physical targets from your own knowledge and any source you can "
    "look up (cite it), check what DIII-D ran in the database, find precedents, construct actuator "
    "trajectories with plan ops, validate with check_plan, then export_plan. Never call "
    "simulate with confirm=true unless the user approved the GPU rollout."
)


def _reply(msg_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": msg_id}
    if error is not None:
        out["error"] = error
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def handle(msg: dict) -> None:
    method, msg_id = msg.get("method"), msg.get("id")
    if msg_id is None:  # notification (e.g. notifications/initialized)
        return
    if method == "initialize":
        _reply(msg_id, {"protocolVersion": msg.get("params", {}).get("protocolVersion", PROTOCOL),
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "ignite-shot-designer", "version": "0.1"},
                        "instructions": INSTRUCTIONS})
    elif method == "ping":
        _reply(msg_id, {})
    elif method == "tools/list":
        _reply(msg_id, {"tools": [{"name": n, "description": t["description"],
                                   "inputSchema": t["schema"]} for n, t in tools.TOOLS.items()]})
    elif method == "tools/call":
        params = msg.get("params", {})
        try:
            result = tools.call(params.get("name"), params.get("arguments") or {})
            _reply(msg_id, {"content": [{"type": "text", "text": tools.to_json(result)}],
                            "isError": False})
        except Exception as e:  # noqa: BLE001 - the agent gets the error as tool output
            _reply(msg_id, {"content": [{"type": "text", "text": f"{type(e).__name__}: {e}"}],
                            "isError": True})
    else:
        _reply(msg_id, error={"code": -32601, "message": f"method not found: {method}"})


def main() -> None:
    warnings.simplefilter("ignore")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            _reply(None, error={"code": -32700, "message": "parse error"})
            continue
        try:
            handle(msg)
        except Exception:  # noqa: BLE001 - keep serving
            sys.stderr.write(traceback.format_exc())


if __name__ == "__main__":
    main()
