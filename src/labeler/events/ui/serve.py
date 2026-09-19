"""Local launcher. Binds the loopback; remote users connect over SSH."""

from __future__ import annotations

import os
from urllib.parse import urlencode

DEFAULT_PORT = 8811

#: The wrapper every live fdp fetch has to run under, spelled the way
#: `docs/LABELER.md` spells it. `fdp run` execs a program, so it takes the
#: module form rather than the `labeler-verify` pixi task, which is not an
#: executable on PATH.
FDP_COMMAND = "pixi run -e labelmaker fdp run python -m labeler.events.ui"

#: What `fdp run` actually leaves in the environment (measured on stellar,
#: 2026-09-18): the PTDATA client library and the MDSplus tree path. Either
#: one missing means a shot outside the corpus cannot be fetched. Checked at
#: startup rather than at the first fetch, because the first fetch is minutes
#: into a review of a shot the corpus does not have, and failing there wastes
#: all of it.
FDP_MARKERS = ("PTDATA_LIBRARY", "default_tree_path")


def _run(app, host: str, port: int) -> int:
    import uvicorn

    # `access_log=False` is load-bearing, not tidiness: the token rides in the
    # query string, so every fresh link open would otherwise write the live
    # credential to stdout and into anything capturing this terminal.
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    return 0


def _under_fdp() -> bool:
    return all(os.environ.get(marker) for marker in FDP_MARKERS)


def main(
    host: str | None = None,
    port: int | None = None,
    token: str | None = None,
) -> int:
    from .app import create_app

    host = "127.0.0.1" if host is None else host
    port = DEFAULT_PORT if port is None else int(port)
    if host != "127.0.0.1":
        raise ValueError(
            "the verify server binds only to 127.0.0.1; use an SSH forward"
        )
    app = create_app(token=token)
    query = urlencode({"token": app.state.token})
    print(f"verify: http://127.0.0.1:{port}/?{query}", flush=True)
    print(f"ssh -L {port}:localhost:{port} stellar", flush=True)
    if not _under_fdp():
        print(
            "note: this process is not under the fdp wrapper. A shot outside "
            "the corpus will fail to fetch (PTSERVER / TREE-E-FOPENR). "
            f"Restart as:\n  {FDP_COMMAND}",
            flush=True,
        )
    return _run(app, host, port)
