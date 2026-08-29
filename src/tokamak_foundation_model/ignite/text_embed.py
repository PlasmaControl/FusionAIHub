"""Per-shot DIII-D text bundles -> Qwen3 embeddings: splitting rules + consolidated-H5 loader.

Splits a per-shot text bundle (see ``embed_shot_text.py``) into the strictly
pre-experiment slice ("input", everything an operator would know before the shot ran)
and the whole bundle ("total"), and loads the consolidated H5 of precomputed embeddings
that a later task wires in as dynamics-model conditioning.

Keep this module free of transformers/GPU dependencies: it must import cleanly on a
CPU-only, transformers-less environment (the embedding CLI imports transformers lazily).
"""

from __future__ import annotations

import numpy as np
import torch

# Anchor strings marking section boundaries in a per-shot text bundle. Each is located as
# the first line equal to or starting with the anchor string.
ANCHOR_GENERAL = "## General session context"
ANCHOR_SESSION_SUMMARIES = "SESSION-WIDE SUMMARIES"
ANCHOR_PLANNED = "## Planned context (mini-proposal)"
ANCHOR_SHOT_SPECIFIC = "## Shot-specific context (from summary.html)"


def _find_anchor_line(lines: list[str], anchor: str) -> int | None:
    """Index of the first line starting with `anchor`, or None if absent."""
    for i, line in enumerate(lines):
        if line.startswith(anchor):
            return i
    return None


def split_bundle(text: str, counters: dict | None = None) -> tuple[str, str]:
    """Split a per-shot text bundle into (input_text, total_text).

    total_text is the whole bundle unchanged. input_text is the strictly pre-experiment
    content: the header, the pre-experiment slice of the general section (everything known
    before the shot ran), and the whole planned/mini-proposal section. Session summaries
    (operator/diagnostics logs written during/after the run day) and shot-specific results
    are always excluded. See the module docstring and the anchor constants above.

    ``counters`` (optional): when given, incremented in place (``counters[key] += 1``, key
    created at 0 first) for every ANCHOR-MISSING fallback branch actually taken on this call —
    a missing anchor silently changes the causality-critical INPUT slice, so callers processing
    many bundles can tally how often each fallback fired. No behavior change to the returned
    slices either way.
    """
    def _bump(key: str) -> None:
        if counters is not None:
            counters[key] = counters.get(key, 0) + 1

    total_text = text
    lines = text.splitlines(keepends=True)

    idx_general = _find_anchor_line(lines, ANCHOR_GENERAL)
    if idx_general is None:
        # header can't be delimited safely without the general-section anchor
        _bump("missing_general_anchor")
        return "", total_text

    idx_summaries = _find_anchor_line(lines, ANCHOR_SESSION_SUMMARIES)
    if idx_summaries is None:
        _bump("missing_summaries_anchor")
    idx_planned = _find_anchor_line(lines, ANCHOR_PLANNED)
    idx_shot_specific = _find_anchor_line(lines, ANCHOR_SHOT_SPECIFIC)
    if idx_shot_specific is None:
        _bump("missing_shot_anchor")

    header = "".join(lines[:idx_general])

    # general slice ends at the first PRESENT anchor after it (summaries, planned, or
    # shot-specific, whichever comes first) -- else EOF. Using only "summaries else planned"
    # here would let the general slice silently swallow shot-specific content (and run to
    # EOF) whenever BOTH summaries and planned are missing.
    later_anchors = [i for i in (idx_summaries, idx_planned, idx_shot_specific) if i is not None]
    general_end = min(later_anchors) if later_anchors else None
    general_slice = "".join(lines[idx_general:general_end])

    if idx_planned is None:
        _bump("missing_planned_anchor")
        return header + general_slice, total_text

    planned_end = idx_shot_specific if idx_shot_specific is not None else len(lines)
    planned_slice = "".join(lines[idx_planned:planned_end])

    input_text = header + general_slice + planned_slice
    return input_text, total_text


def load_text_embeddings(h5_path, dim: int, key: str = "input") -> dict[str, torch.Tensor]:
    """Load precomputed per-shot text embeddings from the consolidated H5.

    Validates `dim` against the file's `embed_dim` attr and `key`, bulk-reads `/shots` and
    `/{key}`, slices each row to the first `dim` components (Matryoshka prefix), and
    L2-renormalizes (a prefix slice of a unit vector is not itself unit-norm).
    """
    import h5py

    if key not in ("input", "total"):
        raise ValueError(f"key must be 'input' or 'total', got {key!r}")

    with h5py.File(h5_path, "r") as f:
        embed_dim = int(f.attrs["embed_dim"])
        if not (0 < dim <= embed_dim):
            raise ValueError(f"dim={dim} must be in (0, {embed_dim}]")
        shots = f["shots"][:]
        vecs = f[key][:, :dim]

    out: dict[str, torch.Tensor] = {}
    for shot, row in zip(shots, vecs):
        v = torch.from_numpy(row.astype(np.float32))
        v = v / v.norm().clamp_min(1e-8)
        out[str(int(shot))] = v
    return out


def lookup(embeds: dict, shot, dim: int) -> tuple[torch.Tensor, bool]:
    """Look up a shot's embedding: (vector, True) on hit, (zeros(dim), False) on miss."""
    vec = embeds.get(str(int(shot)))
    if vec is None:
        return torch.zeros(dim), False
    return vec, True
