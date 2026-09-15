"""The MCP server: `ideate`'s retrieval as tools an assistant can call (plan B7).

`tools.py` is the whole contract -- three plain functions whose signatures and docstrings BECOME
the tool schemas, and whose return values are JSON-serialisable dicts. `server.py` registers them
on an `MCPServer` and adds the `ideate://manifest` resource; `__main__.py` runs stdio, which is
what `pixi run -e ideate-cpu ideate-mcp` and the `.mcp.json` at the repo root invoke.

Nothing here is a second implementation of retrieval: `search_shots` and `describe_shot` call
`ideate.retrieval.rank` and `ideate.retrieval.describe` over the same `ShotDB` the CLI opens, so
an assistant and `ideate query` cannot disagree about a shot.
"""
