"""labelmaker - run the group's trained models over the FAITH shot corpus.

Stages are `features` -> `infer` -> `validate`, with a per-shot HDF5 file
between each, driven by `python -m labelmaker.run`. See
docs/superpowers/specs/2026-09-03-labelmaker-design.md.
"""

__version__ = "0.1.0"
