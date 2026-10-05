"""Audit phase-search false rejection under an independent refractory null.

This conditions on qualified edge count and span. It tests the statistical
group-search calibration, not ECE noise detection or physical sawtooth validity.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from sawtooth_physics import OUTPUT, REPO, WORK, save_json

from labeler.sawtooth.physics import (
    DEFAULT_RULE,
    core_relaxation_phases,
    periodicity_null,
)


def binomial_record(rejected, draws):
    """Monte Carlo rate, standard error, and 95% Wilson binomial interval."""
    rate = rejected / draws
    z = 1.959963984540054
    denominator = 1 + z * z / draws
    center = (rate + z * z / (2 * draws)) / denominator
    half = z * np.sqrt(rate * (1 - rate) / draws + z * z / (4 * draws**2))
    half /= denominator
    return {
        "rejected": rejected,
        "draws": draws,
        "rate": rate,
        "standard_error": float(np.sqrt(rate * (1 - rate) / draws)),
        "ci95": [float(max(0, center - half)), float(min(1, center + half))],
        "interval_method": "Wilson binomial interval over independent generated draws",
    }


def audit(*, draws=1000, seed=23, edge_count=20, span_s=1.0):
    rule = DEFAULT_RULE
    holdoff = rule.frame_ms / 2000
    free_span = span_s - (edge_count - 1) * holdoff
    if draws < 1 or edge_count < rule.relaxation_minimum_edges or free_span <= 0:
        raise ValueError("audit needs positive draws and supported count/span")
    rng = np.random.default_rng(seed)
    raw_rejections, search_rejections, periodic_acceptances = 0, 0, 0
    support = [(-0.1 * span_s, 1.1 * span_s)]
    periodic_rng = np.random.default_rng(seed + 1)
    jitter_sigma = 0.0015
    for draw in range(draws):
        # An independent direct generator: fixed endpoints/count/span with
        # shifted uniform order statistics, exactly preserving the holdoff.
        edges = np.r_[
            0.0, np.sort(rng.uniform(0, free_span, edge_count - 2)), free_span
        ]
        edges += np.arange(edge_count) * holdoff
        raw_rejections += periodicity_null(edges, rule)["p_value"] <= (
            rule.periodicity_null_alpha
        )
        search_rejections += bool(core_relaxation_phases(edges, support, rule))
        periodic = np.linspace(0, span_s, edge_count)
        periodic[1:-1] += periodic_rng.normal(0, jitter_sigma, edge_count - 2)
        periodic_acceptances += bool(core_relaxation_phases(periodic, support, rule))
        if (draw + 1) % 100 == 0:
            print(f"phase null audit: {draw + 1}/{draws}", flush=True)
    rapid = np.arange(edge_count) * 0.006
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "seed": seed,
        "periodic_seed": seed + 1,
        "draws": draws,
        "qualified_edge_count": edge_count,
        "edge_span_s": span_s,
        "refractory_holdoff_ms": holdoff * 1000,
        "null_generator": "fixed count/endpoints/span; shifted uniform order statistics",
        "periodic_contrast": {
            **binomial_record(periodic_acceptances, draws),
            "rate_meaning": "acceptance, not rejection; synthetic positive contrast",
            "period_ms": span_s / (edge_count - 1) * 1000,
            "interior_timing_jitter_sigma_ms": jitter_sigma * 1000,
        },
        "raw_whole_run_null": binomial_record(raw_rejections, draws),
        "full_group_search_null": binomial_record(search_rejections, draws),
        "subfloor_6ms_regular_edges_accepted": bool(
            core_relaxation_phases(rapid, [(-0.1, 1.0)], rule)
        ),
        "rule": asdict(rule),
        "rule_json": str(
            Path(__file__).resolve().parents[2] / "src/labeler/sawtooth/freeze.json"
        ),
        "references": [
            {
                "path": str(REPO / "src/labeler/sawtooth/physics.py"),
                "functions": [
                    "core_relaxation_phases",
                    "periodicity_null",
                    "_relaxation_groups",
                    "_selected_phase_null",
                ],
            },
            {
                "path": str(REPO / "tests/labeler/test_sawtooth_phase_null.py"),
                "purpose": "independent regression for grouping-induced false rejection",
            },
        ],
        "interpretation": (
            "Full-search null replicates the same physical-gap grouping and uses "
            "minimum CV across every eligible group, accounting for selection "
            "and multiplicity within an observable run. Monte Carlo rate "
            "intervals describe synthetic draws, not physical label accuracy. "
            "Separate observable runs and shots do not share a global familywise "
            "error guarantee. POSR qualification is conditioned on, not simulated."
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--output", type=Path, default=OUTPUT / "phase_null_audit.json")
    args = parser.parse_args()
    result = audit(draws=args.draws, seed=args.seed)
    save_json(args.work / "phase_null_audit.json", result)
    save_json(args.output, result)
    print(
        {
            key: result[key]
            for key in (
                "raw_whole_run_null",
                "full_group_search_null",
                "periodic_contrast",
            )
        },
        flush=True,
    )


if __name__ == "__main__":
    main()
