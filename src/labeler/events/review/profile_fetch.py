"""Fetching profile readers, and the fetch worker's entry point that runs them.

`fetch_worker.run` knows `ptdata`, `mds` and `efit01_q`. This module's `run` adds the
profiles of `SPECS` (recipes of configs/labeler/signals.yaml) and tidies how a point
that is simply not there is reported; `serve` is the child's loop with it, and
`Signals` starts its `Fetcher` with that target.

- `geqdsk`: an EFIT01 or EFIT02 G-EQDSK profile (pressure, p', FF', F, q) on the 65-point
  normalized-psi axis. The 2-D node's own `times` dimension is that radial axis, so its
  time comes from `GTIME` (the same layout `fetch_worker._efit01_q` reads).
- `zipfit`: a ZIPFIT fit (n_e, T_e, T_i from CER, toroidal rotation) against rho, whose
  record carries both its axes.

A payload is `{"t_ms" (n_t,), "x" (n_x,), "y" (n_t, n_x), "units"}`, the layout
`Signals.fetched_profile` stores and `profiles.read` sends. Values stay in the
record's units; the catalog source's `scale` takes them into the entry's.

PTDATA and MDSplus answer a point the shot does not have with errors that look like a
broken server (`getservbyname failed for task 'PTSERVER'`, `TreeFOPENR`). `run` asks
the archive for `ip` of the same shot right after one: when that works, the point is
missing for this shot (404, cached for a while, the next source is tried); when it does
not, the error stays upstream.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import fetch_worker as fw
from .fetch_worker import FetchError


@dataclass(frozen=True)
class Spec:
    """One fetched profile: where it lives, its radial axis, and the tag the page
    shows after its label."""

    kind: str
    tree: str
    leaf: str
    x_kind: str
    tag: str
    absolute: bool = False

    @property
    def node(self) -> str:
        if self.kind == "geqdsk":
            return rf"\{self.tree}::TOP.RESULTS.GEQDSK:{self.leaf}"
        return rf"\{self.tree}::TOP.PROFILES.{self.leaf}"


def _geqdsk(tree: str, leaf: str, absolute: bool = False) -> Spec:
    return Spec("geqdsk", tree, leaf, "psi_n", tree, absolute)


def _zipfit(tree: str, leaf: str) -> Spec:
    return Spec("zipfit", tree, leaf, "rho", tree)


#: Recipe name -> what it reads. `efit01_q` stays `fetch_worker`'s.
SPECS: dict[str, Spec] = {
    **{f"{tree.lower()}_{name}": _geqdsk(tree, leaf, absolute=name == "q")
       for tree in ("EFIT01", "EFIT02")
       for name, leaf in (("pres", "PRES"), ("pprime", "PPRIME"),
                          ("ffprim", "FFPRIM"), ("fpol", "FPOL"), ("q", "QPSI"))
       if (tree, name) != ("EFIT01", "q")},
    **{f"{tree.lower()}_{name}": _zipfit(tree, leaf)
       for tree in ("ZIPFIT01", "ZIPFIT02")
       for name, leaf in (("ne", "EDENSFIT"), ("te", "ETEMPFIT"), ("ti", "ITEMPFIT"),
                          ("rot", "TROTFIT"))},
}
#: The tag a fetched recipe's profile carries (`efit01_q` included).
TAGS: dict[str, str] = {"efit01_q": "EFIT01", **{k: v.tag for k, v in SPECS.items()}}

#: Error text that means "this shot does not have it" when the archive still answers.
UNREACHABLE_MARKERS = (*fw.TRANSIENT_MARKERS, "FOPENR")
CONTROL_POINT = "ip"


def _profile(spec: Spec, shot: int) -> dict:
    """`{"t_ms", "x", "y", "units"}` for `spec`."""
    rec = fw._mds(spec.node, spec.tree, shot,
                  dims=("x", "t_ms") if spec.kind == "zipfit" else None)
    with np.errstate(invalid="ignore"):
        data = np.asarray(rec.get("data"), dtype=np.float64)
    if data.ndim != 2 or data.size == 0:
        raise FetchError("missing", f"{spec.tree} {spec.leaf} is not a profile for "
                                    f"this shot (data {data.shape})")
    t_ms, x = (_geqdsk_axes if spec.kind == "geqdsk" else _zipfit_axes)(
        spec, rec, data, shot)
    if data.shape == (x.size, t_ms.size) and data.shape != (t_ms.size, x.size):
        data = data.T
    if data.shape != (t_ms.size, x.size):
        raise FetchError("upstream", f"{spec.tree} {spec.leaf} {data.shape} does not "
                                     f"match its {t_ms.size} times and {x.size} points")
    if spec.absolute:
        data = np.abs(data)
    keep = np.isfinite(t_ms)
    order = np.argsort(t_ms[keep], kind="stable")
    t_ms, data = t_ms[keep][order], data[keep][order]
    if t_ms.size < 2 or not np.isfinite(data).any():
        raise FetchError("missing", f"{spec.tree} {spec.leaf} has no profile for "
                                    "this shot")
    units = rec.get("units") if isinstance(rec.get("units"), dict) else {}
    return {"t_ms": t_ms, "x": x.astype(np.float32), "y": data.astype(np.float32),
            "units": str(units.get("data", "")).strip()}


def _geqdsk_axes(spec: Spec, rec: dict, data: np.ndarray, shot: int):
    gtime = fw._mds(rf"\{spec.tree}::TOP.RESULTS.GEQDSK:GTIME", spec.tree, shot)
    with np.errstate(invalid="ignore"):
        t_ms = np.asarray(gtime["data"], dtype=np.float64).ravel()
    n_x = data.shape[1] if data.shape[0] == t_ms.size else data.shape[0]
    return t_ms, np.linspace(0.0, 1.0, n_x)


def _zipfit_axes(spec: Spec, rec: dict, data: np.ndarray, shot: int):
    del shot
    t_ms = x = None
    for _, values, unit in fw._dims(rec):
        if unit in fw.TIME_UNITS and t_ms is None:
            t_ms = values * fw.TIME_UNITS[unit]
        elif x is None and unit not in fw.TIME_UNITS:
            x = values
    if t_ms is None or x is None:
        raise FetchError("missing", f"{spec.tree} {spec.leaf} has no time and radius "
                                    "axes for this shot")
    return t_ms, x


def _unreachable(error: BaseException) -> bool:
    return any(marker in str(error) for marker in UNREACHABLE_MARKERS)


def _what(request: dict) -> str:
    via = request.get("via")
    if via == "ptdata":
        return f"PTDATA {request.get('name')}"
    if via == "mds":
        return str(request.get("node"))
    spec = SPECS.get(via)
    return spec.node if spec is not None else str(via)


def run(request: dict, shot: int) -> dict:
    """One fetch, in the child: `fetch_worker.run` plus the profiles of `SPECS`, with
    a point the shot lacks reported as `missing`."""
    spec = SPECS.get(request.get("via"))
    try:
        if spec is not None:
            return _profile(spec, shot)
        return fw.run(request, shot)
    except FetchError:
        raise
    except Exception as error:
        if not _unreachable(error):
            raise
        try:
            fw._ptdata(CONTROL_POINT, shot)
        except Exception:  # noqa: BLE001 - the archive is not answering: upstream
            raise error from None
        raise FetchError("missing", f"{_what(request)} has no data for shot "
                                    f"{shot}") from None


def serve(conn, token: str | None) -> None:
    """The fetch worker child's loop with `run` as its work."""
    fw.serve(conn, token, work=run)
