"""Review-page routes for 2-D slices at a time (plan D.B): the EFIT equilibrium and the
TangTV inversion that the dock draws beside the rows.

`install` adds them to the app that `app.create_app` builds, before the static mount.
Any event may ask, since the data is per shot: `event` is validated, not used. A bad
parameter answers 422, a shot without that slice on disk (or none within 40 ms) 404.
"""

from __future__ import annotations

import json
from typing import Annotated

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import Response

from ...config import Paths
from ..review import slices


def install(app: FastAPI, paths: Paths) -> None:
    """Add this module's routes to `app`."""
    from .app import require_event  # app imports this module

    def known(event: str) -> None:
        try:
            require_event(event, paths)
        except HTTPException as error:
            raise HTTPException(422, error.detail) from None

    @app.get("/api/slices")
    def slice_kinds(event: str, shot: Annotated[int, Query(ge=1)]):
        known(event)
        return {"kinds": slices.available(shot, paths)}

    @app.get("/api/slice")
    def slice_view(
        event: str, shot: Annotated[int, Query(ge=1)], kind: str, t_ms: int,
    ):
        known(event)
        if kind not in slices.KINDS:
            raise HTTPException(
                422, f"unknown kind {kind[:32]!r}: expected {' or '.join(slices.KINDS)}"
            )
        try:
            if kind == "equilibrium":
                return slices.equilibrium(shot, t_ms, paths)
            data, record = slices.inversion(shot, t_ms, paths)
        except slices.SliceMissing as error:
            raise HTTPException(404, str(error)) from None
        return Response(
            data, media_type="image/png", headers={"X-Slice": json.dumps(record)}
        )
