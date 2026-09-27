"""The paper's figures and tables, read from what the round-two runs wrote.

The manuscript (`dev/label_paper`: "Tokamak-SI: Automatic Shot Interpreter with
a Catalog of Fusion Reactor Data") shows the interpreter on one phenomenon, AE,
first. A figure that will show every catalog phenomenon draws all six now: AE
from the round-two runs, the other five as empty panels marked "coming", so the
layout and sizes are settled before their numbers exist.

- `scores`: the AE methods' frame scores (the six-phenomenon grid), their
  false-positive rates on MHD frames, the segmentation's scores, and tables;
- `coverage`: reviewed, positive and suggested shots per phenomenon;
- `shots`: one discharge as the interpreter shows it, and AE examples;
- `build`: every product into `$LABELER_ROOT/paper/`, and the copy into the
  manuscript's `figures/`.

Nothing here trains, fetches or scores. A figure reads the JSON, CSV and review
stores the runs left, and `build` records which.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

from ..config import Paths, atomic_path
from ..events.catalog.states import PHENOMENA

AE = "alfven_eigenmode"
ORDER = tuple(PHENOMENA)  # AE first, then the five still to come
COLUMN_IN = 3.25  # ICML \columnwidth, inches
PAGE_IN = 6.75  # ICML \textwidth
FONT_PT = 7
COMING = "coming"
STYLE = {
    "font.size": FONT_PT,
    "axes.titlesize": FONT_PT,
    "axes.labelsize": FONT_PT,
    "xtick.labelsize": FONT_PT - 1,
    "ytick.labelsize": FONT_PT - 1,
    "legend.fontsize": FONT_PT - 1,
    "pdf.fonttype": 42,
    "axes.spines.top": False,
    "axes.spines.right": False,
}


def title(category: str) -> str:
    """The phenomenon's short name, as the catalog gives it."""
    return PHENOMENA[category].name


def paper_dir(paths: Paths) -> Path:
    return paths.root / "paper"


@contextmanager
def style() -> Iterator[None]:
    """The paper's text sizes; draw and save inside it."""
    with matplotlib.rc_context(STYLE):
        yield


def placeholder(ax, heading: str) -> None:
    """An empty panel for a phenomenon whose results are still to come."""
    ax.set_title(heading)
    ax.text(
        0.5,
        0.5,
        COMING,
        transform=ax.transAxes,
        ha="center",
        va="center",
        color="#888888",
        style="italic",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linestyle((0, (2, 2)))
        spine.set_color("#bbbbbb")


def save(fig, stem: Path) -> list[Path]:
    """`stem.pdf` for the manuscript and `stem.png` to look at."""
    out = []
    for suffix, extra in (
        (".pdf", {"metadata": {"CreationDate": None}}),
        (".png", {"dpi": 300}),
    ):
        path = Path(stem).with_suffix(suffix)
        with atomic_path(path) as tmp:
            fig.savefig(tmp, format=suffix[1:], **extra)
        out.append(path)
    return out
