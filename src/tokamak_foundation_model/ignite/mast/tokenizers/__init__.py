"""Fixed tokenizers of the IGNITE-on-MAST stage 1: signals to tokens of 5 ms frames.

`code_sha256` marks the scores of this plan: it covers the modules directly in this package, not
sub-packages (plan 1c's ``tokenizers/learned/`` carries its own hash), so learned-tokenizer code
does not mark these scores stale.
"""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_of_py_files(directory: Path) -> str:
    """Hex sha256 over the bytes of every ``*.py`` file directly in `directory`, in sorted name order.

    Sub-directories are not entered.
    """
    h = hashlib.sha256()
    for f in sorted(Path(directory).glob("*.py"), key=lambda p: p.name):
        h.update(f.read_bytes())
    return h.hexdigest()


def code_sha256() -> str:
    """Hex sha256 of this package's own modules (see the module docstring)."""
    return sha256_of_py_files(Path(__file__).parent)
