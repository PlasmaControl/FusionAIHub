"""IGNITE: a Genie-style discrete world model of the tokamak.

Design: ``docs/IGNITE_DESIGN.md``.

This package reuses no FAITH model code (no MultiWindowBackbone, SharedBackbone,
output_heads, E2EFoundationModel, or the Stage-1 trainer's model and loss logic). It may
reuse only the data-loading pipeline and external libraries (``vector-quantize-pytorch``,
``x-transformers``); all model code here is new. See ``docs/IGNITE_DESIGN.md`` §7.
"""
