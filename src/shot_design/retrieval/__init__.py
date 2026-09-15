"""Multi-channel retrieval: channels -> RRF fusion -> rerank -> explained results.

    from shot_design.retrieval import search
    found = search(query_state, shot_db)
    found.items      # the explained top n
    found.rankings   # each channel's own candidates, before fusion

Two files. `channels.py` holds the channels and the `CHANNELS` registry -- adding one is a
function plus a line in that dict. `rank.py` holds fusion, reranking, explanation and `search`,
which is the whole path in a dozen lines.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from .channels import CHANNELS, hard_filter, nan_excluded
from .rank import (
    SearchResult,
    display,
    display_pair,
    explain,
    explanation_scales,
    load_cfg,
    query_values,
    rerank,
    rrf_fuse,
    search,
    search_report,
)

# suggest (the function) is intentionally not re-exported here: since Python 3.7, submodule
# attribute access (`import shot_design.retrieval.suggest as sg`) binds through this package's
# namespace, so a `suggest` name here would shadow the `shot_design.retrieval.suggest` submodule
# itself. Import the function from the submodule directly: `from .suggest import suggest`.
from .suggest import ActuatorSuggestion, Suggestion

__all__ = [
    "CHANNELS",
    "ActuatorSuggestion",
    "SearchResult",
    "Suggestion",
    "display",
    "display_pair",
    "explain",
    "explanation_scales",
    "hard_filter",
    "load_cfg",
    "nan_excluded",
    "query_values",
    "rerank",
    "rrf_fuse",
    "search",
    "search_report",
]
