"""Poloidal beta directly from EFIT, with its draft threshold and plasma current."""

from __future__ import annotations

from ...config import Paths
from .. import equilibrium
from ._shared import optional


def panels(shot, *, t_range=None, paths=None):
    paths = Paths.from_env() if paths is None else paths
    kwargs = {"paths": paths, "t_range": t_range}
    return (
        optional("beta_p", shot, lambda: equilibrium.line_panel(
            shot, "betap", "beta_p (EFIT01 aeqdsk)", ylabel="β_p",
            hlines=[equilibrium.BETAP_THRESHOLD], **kwargs))
        + optional("ip", shot, lambda: equilibrium.line_panel(
            shot, "ip", "ip", ylabel="A", **kwargs))
    )
