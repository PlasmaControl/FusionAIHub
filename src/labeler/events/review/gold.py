"""The gold set: 50 DIII-D shots with hand labels or a curated source, a few shots
from each of a few run days, so a model trained elsewhere gives up few run days to
be evaluated on them. `silver.py` holds the 450-shot list reviewed for every event."""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

import pandas as pd

from .silver import blind_shots

log = logging.getLogger(__name__)

GOLD_FILE = "gold/gold_shots.csv"
COLUMNS = ("shot", "run_day", "year", "classes", "sources", "notes")
FIRST_SHOT = 170_000


@lru_cache(maxsize=4)
def _read_gold(path: str, mtime_ns: int, inode: int, size: int, blind: frozenset) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"run_day": str}, keep_default_na=False)
    if tuple(frame.columns) != COLUMNS:
        raise ValueError(f"{path}: columns must be {', '.join(COLUMNS)}")
    if frame.shot.duplicated().any():
        raise ValueError(f"{path}: duplicate shots")
    early = frame.shot[frame.shot < FIRST_SHOT].tolist()
    if early:
        raise ValueError(f"{path}: shots below {FIRST_SHOT}: {early}")
    if (~frame.run_day.str.fullmatch(r"\d{8}")).any():
        raise ValueError(f"{path}: run_day must be YYYYMMDD")
    refused = sorted(set(frame.shot) & blind)
    if refused:
        raise ValueError(f"{path}: blind test shots may not be gold: {refused}")
    return frame


def read_gold(label_tables) -> pd.DataFrame:
    """The validated gold list, cached on its file version."""
    path = Path(label_tables) / GOLD_FILE
    stat = path.stat()
    return _read_gold(str(path), stat.st_mtime_ns, stat.st_ino, stat.st_size,
                      frozenset(blind_shots(label_tables, missing_ok=True)))


def is_gold(label_tables, shot: int) -> bool:
    try:
        return bool((read_gold(label_tables).shot == int(shot)).any())
    except FileNotFoundError:
        return False
    except (ValueError, OSError) as error:
        log.warning("gold list unreadable: %s", error)
        return False
