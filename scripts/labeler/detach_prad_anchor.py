#!/usr/bin/env python
"""Measure the Prad,div anchor: shot 201081's P_in and its attached/detached Prad,div,L.

Chen 2026 (NF 66 036014, worked example 201081) reports Prad,div,L 1.6 MW attached
and 2.2 MW detached, divertor Thomson Te cliffs at about 2650 ms (attached to
detached) and about 4450 ms (back to attached). This script measures, from the
cached bolometer, beam, EFIT ohmic and ECH records and the same 250 ms ELM-masked
averaging the labels use:

* P_in = beams + EFIT POH + ECH, the denominator of f_div. ONE number is used: the
  median of the 250 ms-averaged P_in over the 50 ms bins of the attached window
  (`measured_anchor.p_in_mw`, 4.274 MW on 32 bins). No other P_in total is
  recorded; the beam, ohmic and ECH components are flat-top medians given only to
  say what P_in is made of;
* Prad,div,L medians in windows defined by the published Te-cliff times, never by
  TangTV: attached before the first cliff and after the second, detached between.

The record (`docs/labeler/results/detachment_prad_anchor.json`) carries the
measured values beside the published ones; `thresholds.py` takes its anchor
constants from the measured values and a test checks they agree. Run through
`pixi run --frozen -e labelmaker python scripts/labeler/detach_prad_anchor.py`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from detach_json import dumps

from labeler.events.detachment import core, prad, signals

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "docs/labeler/results/detachment_prad_anchor.json"

SHOT = 201081
#: Chen 2026 digest (`.tmp/label_papers/Chen_2026_Nucl._Fusion_66_036014.md`): the
#: divertor Thomson Te cliffs and the two Prad,div,L operating values.
CLIFF_DETACH_MS = 2650.0
CLIFF_REATTACH_MS = 4450.0
PUBLISHED_ATTACHED_MW = 1.6
PUBLISHED_DETACHED_MW = 2.2
PUBLISHED_NBI_MW = 4.0
#: A bin enters a phase only when its whole 250 ms radiation window lies this far
#: from a cliff (the cliff time is read off a plot, and Prad,div,L lags Te).
GUARD_MS = 100.0
#: The first bin with the beams at flat-top power; the last before the ramp-down.
FLAT_TOP_START_MS = 1500.0
FLAT_TOP_END_MS = 5700.0
#: The slow post-cliff relaxation of Prad,div,L: re-attachment windows start later.
REATTACH_SETTLE_MS = 500.0


def median_mw(values, mask):
    return float(np.nanmedian(values[mask])) if mask.any() else None


def main() -> int:
    cache = signals.load_cache(SHOT)
    t_p, p_heat, _ = signals.heating_power(SHOT, cache)
    elm_t, elm_flag = signals.elm_mask(SHOT, cache)
    edges = core.bin_edges(500.0, 6500.0, core.BIN_MS)
    centres = core.bin_centres(edges)
    half = 0.5 * 250.0
    rt, ry = cache["prad_divl"]
    ind = prad.prad_indicator(edges, rt, ry, t_p, p_heat, elm_t, elm_flag)
    p_in = signals.window_mean(t_p, p_heat, centres, 250.0)
    prad_mw = ind.value * p_in / 1e6
    p_in_mw = p_in / 1e6
    beam = signals.beam_power(SHOT, cache, centres, 250.0) / 1e6
    poh = signals.window_mean(
        *(np.asarray(v, float) for v in cache["poh"]), centres, 250.0
    )
    ech = (p_in - beam * 1e6 - poh) / 1e6
    ok = ind.valid
    lo, hi = centres - half, centres + half
    in_top = ok & (centres >= FLAT_TOP_START_MS) & (centres <= FLAT_TOP_END_MS)
    pre = in_top & (hi <= CLIFF_DETACH_MS - GUARD_MS)
    det = (
        in_top
        & (lo >= CLIFF_DETACH_MS + GUARD_MS)
        & (hi <= CLIFF_REATTACH_MS - GUARD_MS)
    )
    post = in_top & (lo >= CLIFF_REATTACH_MS + REATTACH_SETTLE_MS)
    att = pre | post
    phases = {
        "attached_before_first_cliff": pre,
        "detached_between_cliffs": det,
        "attached_after_second_cliff": post,
        "attached_both": att,
    }
    out = {
        "shot": SHOT,
        "source": (
            "Chen 2026 NF 66 036014 worked example: Te cliffs ~2650 and ~4450 ms, "
            "Prad,div,L 1.6 MW attached and 2.2 MW detached, 4 MW NBI"
        ),
        "published": {
            "cliff_detach_ms": CLIFF_DETACH_MS,
            "cliff_reattach_ms": CLIFF_REATTACH_MS,
            "prad_divl_attached_mw": PUBLISHED_ATTACHED_MW,
            "prad_divl_detached_mw": PUBLISHED_DETACHED_MW,
            "nbi_mw": PUBLISHED_NBI_MW,
        },
        "window_rule": {
            "averaging_ms": 250.0,
            "elm_masked": True,
            "guard_ms": GUARD_MS,
            "flat_top_ms": [FLAT_TOP_START_MS, FLAT_TOP_END_MS],
            "reattach_settle_ms": REATTACH_SETTLE_MS,
            "windows_from": "published Te-cliff times, not TangTV",
        },
        "p_in_mw": {
            "anchor_value_used": "measured_anchor.p_in_mw below: the median of the "
            "250 ms-averaged P_in over the 32 attached-window bins, the one P_in "
            "number the cutoffs and the documents use",
            "components_flat_top_median_mw_not_p_in": {
                "beams": median_mw(beam, in_top),
                "poh": median_mw(poh / 1e6, in_top),
                "ech": median_mw(ech, in_top),
            },
            "beam_source": "PTDATA BMSPINJ (corpus pinj is a stub on this shot)",
        },
        "phases": {},
    }
    for name, mask in phases.items():
        out["phases"][name] = {
            "n_bins": int(mask.sum()),
            "centres_ms": [float(centres[mask].min()), float(centres[mask].max())]
            if mask.any()
            else None,
            "prad_divl_mw_median": median_mw(prad_mw, mask),
            "prad_divl_mw_range": [
                float(np.nanmin(prad_mw[mask])),
                float(np.nanmax(prad_mw[mask])),
            ]
            if mask.any()
            else None,
            "f_div_median": median_mw(ind.value, mask),
            "p_in_mw_median": median_mw(p_in_mw, mask),
        }
    out["measured_anchor"] = {
        "p_in_mw": round(out["phases"]["attached_both"]["p_in_mw_median"], 3),
        "attached_mw": round(out["phases"]["attached_both"]["prad_divl_mw_median"], 3),
        "detached_mw": round(
            out["phases"]["detached_between_cliffs"]["prad_divl_mw_median"], 3
        ),
    }
    OUT.write_text(dumps(out, indent=1) + "\n")
    print(dumps(out["measured_anchor"]), dumps(out["p_in_mw"]))
    for name, ph in out["phases"].items():
        print(name, ph["n_bins"], ph["prad_divl_mw_median"], ph["f_div_median"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
