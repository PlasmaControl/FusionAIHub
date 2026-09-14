"""Local launcher, ported from shot-recommender-system's shotrec/ui/serve.py.

Access uses shotrec's token link and cookie; remote users connect with SSH forwarding.
"""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urlencode

from .. import config


def main(
    host: str | None = None, port: int | None = None,
    db_dir: Path | None = None, token: str | None = None,
) -> int:
    import uvicorn

    from .app import create_app

    server = config.load_yaml("ui.yaml")["server"]
    host = server["host"] if host is None else host
    port = server["port"] if port is None else port
    if host != "127.0.0.1":
        raise ValueError("ideate serve binds only to 127.0.0.1; use an SSH forward")
    app = create_app(db_dir=db_dir, token=token)
    query = urlencode({"token": app.state.token})
    print(f"IDEATE: http://127.0.0.1:{port}/?{query}", flush=True)
    print(f"ssh -L {port}:localhost:{port} stellar", flush=True)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
    return 0
