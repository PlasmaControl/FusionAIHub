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
from .tools import describe_shot, get_events, never_raises, search_shots

#: Every tool this server offers, in the order an assistant sees them. Later tasks append.
#:
#: `never_raises` is applied HERE rather than at each definition, so that the promise the module
#: docstring makes -- every failure reaches the model as `{"error", "caveats"}`, never as the
#: framework's bare "Error executing tool <name>" -- holds for a tool whose author forgot an
#: `except`, which is the only kind of tool it has ever failed for. `functools.wraps` keeps the
#: signature and the docstring, so the schema below is still the plain function's.
TOOLS = [never_raises(fn) for fn in (search_shots, describe_shot, get_events)]

INSTRUCTIONS = """\
DIII-D shot retrieval over a locally built database of tokamak discharges.

`search_shots` finds shots resembling a description, a reference shot or a set of conditions;
`describe_shot` returns everything the database holds about one shot; `get_events` returns the
time-resolved events for a shot.

Every reply the tools THEMSELVES produce carries `caveats`, and they change what the reply means
-- read them before answering. That promise covers application-level results, including failures
these tools anticipate and ones they do not (a database caught mid-publish, a table written to
some other schema): those come back as `{"error": ..., "caveats": [...]}`. It does NOT cover a
call whose arguments do not match the tool's schema -- a string where a shot number belongs. Such
a call is rejected by the MCP framework before any of this code runs, so it surfaces as a
protocol/validation error with no `caveats` field. Fix the argument and call again.

Two caveats matter most. An empty search result with "no channel had anything to search on"
means the query was empty, not that no such shot exists.

And `get_events` returns THREE separate lists plus a `status`, because they are three different
kinds of claim:

  events         what a DIAGNOSTIC showed, per `source` and `confidence`.
  text_mentions  a lexicon hit in the operator logbook: somebody wrote the word. Not evidence
                 that the phenomenon occurred.
  forecasts      a MODEL's estimate of what was about to happen, from a risk curve and a
                 threshold. Never report one as an observation.

`status` says what an EMPTY `events` means, and the four are not interchangeable:
`unindexed` (the shot is not in the database at all), `unprocessed` (no detector is recorded as
having run over it -- absence is not evidence), `uncovered` (detectors ran but not over the
window you asked about; the caveat names the span that was covered), `observed` (somebody looked
and saw nothing, which is a real finding).

Quote the operator logbook only from `record.human.log_entries`, verbatim.
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
