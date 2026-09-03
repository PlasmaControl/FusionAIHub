"""IGNITE — a fresh Genie-style discrete world model for the tokamak.

Design: ``docs/IGNITE_DESIGN.md``.

HARD RULE: this package reuses NO existing FAITH model code (no MultiWindowBackbone,
SharedBackbone, output_heads, E2EFoundationModel, or the Stage-1 trainer's model/loss
logic). It may reuse only (a) the data-loading pipeline and (b) external validated
libraries (``vector-quantize-pytorch``, ``x-transformers``). All model-side code here is
written from scratch. See ``docs/IGNITE_DESIGN.md`` §7.
"""
