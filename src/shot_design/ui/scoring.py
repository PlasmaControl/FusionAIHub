"""Read-only scoring explanation from the same configuration as retrieval."""

from ..retrieval import channels, phenomena, rank

# These describe the registered implementations; numbers come from their loaders.
COMPARISONS = {
    "scalar_knn": "Segment scalar embedding of the reference shot; requested scalar targets when no reference is given",
    "text_knn": "MiniLM embedding of mini-proposal and logbook text",
    "bm25": "Exact words in mini-proposal and logbook text",
    "ignite_knn": "IGNITE codec embeddings; needs a reference shot and an encoded database",
    "phenomenon": "Resolved phenomenon evidence, ordered by evidence class then score",
}


def scoring_info(db_summary: dict, phenomenon_config: tuple) -> dict:
    cfg = rank.load_cfg()
    # Search's phenomenon channel uses this loaded snapshot, including its cache.
    weights, saturation_n, _floor = phenomenon_config
    tiers = {name: getattr(phenomena, name) for name in
             ("OBSERVED", "LABELLED", "FORECAST", "DATABASE", "TEXTUAL")}
    return {
        "method": "Weighted reciprocal rank fusion",
        "formula": "score = Σ_c w_c / (k0 + rank_c)",
        **{key: cfg[key] for key in
           ("k0", "dedup_threshold", "run_diversity_decay", "outcome_penalty")},
        "channels": [{
            "name": name,
            "weight": float(cfg["weights"].get(name, 1.0)),
            "weight_source": "configured" if name in cfg["weights"] else "default",
            "compares": COMPARISONS.get(name) or
            (channel.__doc__ or "Registered retrieval channel").strip().splitlines()[0],
        } for name, channel in channels.CHANNELS.items()],
        # Illustrative scale in the brief, not a calibrated probability or bound.
        "score_range": [0.01, 0.04],
        "hard_filters": ["constraints", "segment", "require labels", "avoid labels"],
        "phenomenon": {
            "formula": "score = label × max_p + event × (1 − exp(−n_events / saturation_n)) + text × tanh(hits / 2) + database",
            "weights": weights,
            "saturation_n": saturation_n,
            "class_order": sorted(tiers, key=tiers.get, reverse=True),
            "text_only_ceiling": phenomena.lexicon_mod.TEXT_ONLY_CEILING,
            "terms": "max_p is the strongest qualifying model label. n_events is the sum of matching event-rule weights from phenomena.yaml. hits counts operator mentions. The database term applies when a curated list names the shot. Forecasts set the evidence class; they add no event term.",
        },
        "db": {key: db_summary[key] for key in ("n_shots", "shot_range", "git_sha")}
        | {"built": db_summary["built_at"]},
    }
