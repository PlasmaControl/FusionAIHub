"""Task F2b: `build_record` gives a shot with no Ip trace real segments from
PULSE-LENGTH.

New for the Frontier port -- there is no Ip signal anywhere on that cluster, so this
path is the only way `find_segments`'s zero-segment answer does not reach every one of
its 3,031 shots.
"""

from __future__ import annotations

import numpy as np
import pytest

from shot_design.shotdb import build
from shot_design.shotdb.reader import Signal

from .conftest import text_bundle

PROXY_SHOT = 900050


class _NoIpOneActuatorReader:
    """A minimal SignalReader: no Ip trace anywhere, one constant beam actuator."""

    kind = "fake_no_ip"

    def __init__(self, paths):
        self.paths = paths
        self.reasons: dict[str, str] = {}

    def groups(self, shot):
        return ["p_inj"]

    def read_shot(self, shot, specs):
        t = np.arange(0.0, 5001.0, 10.0)
        beam = np.full_like(t, 2.0e6, dtype=np.float32)
        signals: dict[str, Signal | None] = {}
        coverage: dict[str, str] = {}
        for spec in specs:
            if spec.name == "pnbi_15L":
                signals[spec.name] = Signal(
                    t, beam, "W", "corpus", "p_inj", "pinjf_15l"
                )
                coverage[spec.name] = "present"
            elif spec.installed:
                signals[spec.name] = None
                coverage[spec.name] = "unavailable"
            else:
                signals[spec.name] = None
                coverage[spec.name] = "not_installed"
        return signals, coverage


@pytest.fixture
def no_ip_bundle(paths):
    bundle = text_bundle(PROXY_SHOT, row={"PULSE-LENGTH": "5.0"})
    (paths.per_shot_txt_dir / f"shot_{PROXY_SHOT}.txt").write_text(bundle)
    return PROXY_SHOT


def test_build_record_uses_pulse_length_proxy_when_ip_is_missing(paths, no_ip_bundle):
    rec, _ = build.build_record(
        PROXY_SHOT, paths, build.load_build_cfg(), reader=_NoIpOneActuatorReader(paths)
    )
    flat = rec.segment("flat_top")
    assert flat is not None
    assert flat.t0_ms == pytest.approx(1000.0)
    assert flat.t1_ms == pytest.approx(4730.0)
    reason = rec.coverage_reasons.get("ip")
    assert reason is not None and reason.startswith("segments from PULSE-LENGTH proxy")
    assert rec.outcome.end_reason == "no_ip_signal"
    assert flat.raw["pnbi_15L_mean"] == pytest.approx(2.0e6, abs=1.0)


def test_build_record_without_a_bundle_still_has_no_segments(paths):
    """No bundle at all (not just no PULSE-LENGTH in it) is the same empty answer as
    before this task -- the proxy needs the shot table row to exist, not merely no Ip
    trace."""
    rec, shapes = build.build_record(
        999999, paths, build.load_build_cfg(), reader=_NoIpOneActuatorReader(paths)
    )
    assert rec.segments == [] and shapes == {}
    assert rec.coverage_reasons.get("ip") is None
