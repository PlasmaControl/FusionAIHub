"""Source-preserving confinement reconciliation and missing-diagnostic models."""

import os
from pathlib import Path


def run_directory() -> Path:
    """Keep experiment outputs in this checkout; shared stores remain inputs."""
    default = Path(__file__).resolve().parents[3] / "runs/labeler/confinement/v1"
    return Path(os.environ.get("CONFINEMENT_RUN_DIR", str(default)))
