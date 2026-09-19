"""Design routes inherit the application's existing token middleware."""

from fastapi import APIRouter, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.routing import APIRoute

from ..design import program as service


class _FiniteValidationRoute(APIRoute):
    """Don't echo NaN/Infinity input into FastAPI's JSON validation response."""

    def get_route_handler(self):
        handler = super().get_route_handler()

        async def validated(request: Request):
            try:
                return await handler(request)
            except RequestValidationError as exc:
                detail = [
                    {key: error[key] for key in ("type", "loc", "msg")}
                    for error in exc.errors()
                ]
                return JSONResponse({"detail": detail}, status_code=422)

        return validated


router = APIRouter(prefix="/api/design", route_class=_FiniteValidationRoute)


def _call(fn, *args):
    try:
        return fn(*args)
    except FileNotFoundError as exc:
        raise HTTPException(404, "Unknown design revision") from exc
    except service.ProgramValidationError as exc:
        raise HTTPException(422, {"errors": exc.errors, "can_export": False}) from exc
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc


@router.post("/preview")
def preview(body: service.DesignProgram, request: Request):
    return _call(service.preview, body, request.app.state.paths)


@router.post("/prepare")
def prepare(body: service.DesignProgram, request: Request):
    return _call(service.prepare_ignite, body, request.app.state.paths)


@router.post("/merge")
def merge(body: service.DesignProgram, request: Request):
    return _call(service.average_references, body, request.app.state.paths)


@router.post("")
def save(body: service.DesignProgram, request: Request):
    return _call(service.save, body, request.app.state.paths)


@router.get("")
def list_programs(request: Request):
    return service.list_programs(request.app.state.paths)


@router.get("/{ident}")
def load(ident: str, request: Request):
    return _call(service.load, ident, request.app.state.paths)


@router.get("/{ident}/json")
def download_json(ident: str, request: Request):
    program = _call(service.load_program, ident, request.app.state.paths)
    return Response(
        program.model_dump_json(indent=2),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="design-{ident}.json"'},
    )


@router.get("/{ident}/ignite")
def download_ignite(ident: str, request: Request):
    paths = request.app.state.paths
    program = _call(service.load_program, ident, paths)
    path = _call(service.export_ignite, program, paths)
    return FileResponse(
        path, filename=f"design-{ident}.pt", media_type="application/octet-stream"
    )
