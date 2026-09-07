"""`build_server()`: the three tools of `tools.py` and the `ideate://manifest` resource.

`TOOLS` is the registry -- one list, appended to as later tasks add tools -- so there is one
place that says what this server offers and `test_mcp` can assert against it. Registration is
`MCPServer.add_tool`, which builds each tool's JSON schema from the function's signature and its
description from the docstring; nothing here restates either, because a restatement is a second
definition that can drift from the function it describes.
"""

from __future__ import annotations

import json

from mcp.server import MCPServer

from .. import __version__, config
from .tools import describe_shot, get_events, search_shots

#: Every tool this server offers, in the order an assistant sees them. Later tasks append.
TOOLS = [search_shots, describe_shot, get_events]

INSTRUCTIONS = """\
DIII-D shot retrieval over a locally built database of tokamak discharges.

`search_shots` finds shots resembling a description, a reference shot or a set of conditions;
`describe_shot` returns everything the database holds about one shot; `get_events` returns the
time-resolved events for a shot.

Every reply carries `caveats`, and they change what the reply means -- read them before
answering. Two of them matter most: an empty result with "no channel had anything to search on"
means the query was empty, not that no such shot exists; and `get_events` keeps `forecasts`
apart from `events` because a forecast is a model's estimate of what was about to happen and an
event is a claim about what a diagnostic showed. Never report one as the other, and quote the
operator logbook only from `record.human.log_entries`, verbatim.
"""


def build_server() -> MCPServer:
    """The configured server. Transport is the caller's: `__main__` runs it on stdio."""
    server = MCPServer(
        name="ideate",
        title="ideate: DIII-D shot retrieval",
        version=__version__,
        instructions=INSTRUCTIONS,
    )
    for fn in TOOLS:
        server.add_tool(fn)

    @server.resource(
        "ideate://manifest",
        name="manifest",
        title="ideate database manifest",
        description="What the built database contains: shot counts, reader, PCA, labels join.",
        mime_type="application/json",
    )
    def manifest() -> dict:
        """The database's `manifest.json`, or the same missing-database error the tools return.

        Read this to find out which shots the tools can see at all before searching for one.
        """
        paths = config.load_paths()
        path = paths.db_dir / "manifest.json"
        if not path.exists():
            return {
                "error": f"no database at {paths.db_dir} -- run `ideate build --list poc_v1` first"
            }
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return {"error": f"could not read {path}: {type(exc).__name__}: {exc}"}

    return server
