"""Learned tokenizers of the IGNITE-on-MAST stage 1 (plan 1c): recoders behind a fixed codec.

A learned tokenizer sits behind the top rung of a 1b codec. The 1b codec (the "front end") turns a
signal into front values ``(F, n_tok, n_val)``; this package regroups them into streams
``(F, S, D)``, standardizes each stream (`store.Norm`) and trains a recoder over the streams.
Units: "front values" are what the front codec's ``encode_domain`` returns; "example units" are
front values after the per-stream standardization.

`code_sha256` marks the scores of this plan. It covers the modules directly in this directory and
nothing else; 1b's own hash (`tokenizers.code_sha256`) covers the modules one level up and does not
descend into this sub-package (ruling B4), so a change here never marks a 1b score stale.
"""

from __future__ import annotations

from pathlib import Path

from .. import sha256_of_py_files


def code_sha256() -> str:
    """Hex sha256 of this sub-package's own modules, in sorted name order (see the module docstring)."""
    return sha256_of_py_files(Path(__file__).parent)
