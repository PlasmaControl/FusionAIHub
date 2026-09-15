"""`python -m shot_design.mcp`: the server on stdio. What `.mcp.json` and `pixi run shot_design-mcp` start.

stdio is the transport, so **stdout belongs to the protocol**. Anything this process prints there
corrupts the stream and the client sees a parse error rather than the message. Nothing in
`tools.py` or `server.py` prints; if something added under here ever needs to say something, it
says it on stderr, which the client forwards to its own log.
"""

from .server import build_server

if __name__ == "__main__":
    build_server().run("stdio")
