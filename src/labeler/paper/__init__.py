"""The paper's figures and tables, read from what the round-two runs wrote.

The manuscript (`dev/label_paper`: "Tokamak-SI: Automatic Shot Interpreter with
a Catalog of Fusion Reactor Data") shows the interpreter on one phenomenon, AE,
first. A figure that will show every paper phenomenon (`ORDER`: the catalog's,
less disruption for now) draws all five now. The score figures draw AE from the
round-two runs and the other four as empty panels marked "coming", so the layout
and sizes are settled before their numbers exist. The coverage figure and table
count all five: AE from the round-two runs, the other four from the owner's
review, their frame models' splits and applications, and the legacy tables'
labelled shots; only a phenomenon with none of its inputs yet is marked
"coming".

- `scores`: the AE methods' frame scores (one panel per phenomenon), their
  false-positive rates on MHD frames, the segmentation's scores, and tables;
- `coverage`: reviewed, positive and suggested shots per phenomenon;
- `shots`: one discharge as the interpreter shows it, and AE examples;
- `build`: every product into `$LABELER_ROOT/paper/`, and the copy into the
  manuscript's `figures/`;
- `snapshot`: each file the build reads, read once and pinned by its sha256;
- `staging`: the directory the build draws into, and the swap that puts it in
  the old output's place.

Nothing here trains or fetches. The score figures and tables read the JSON the
evaluations wrote; to rank and draw the example and interpreter shots, `build`
runs the chosen model over its test shots' review stores, against the labels it
was scored on. `build` records every input it read.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import matplotlib

from ..config import Paths, atomic_path
from ..events.catalog.states import PHENOMENA

AE = "alfven_eigenmode"
#: The paper's phenomena, AE first, then the four still to come. Disruption is
#: left out for now (the owner, 2026-09-28: "ignore/remove disruptions for now").
LEFT_OUT = ("disruption",)
ORDER = tuple(p for p in PHENOMENA if p not in LEFT_OUT)
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


def placeholder(ax, heading: str, mark: str = COMING) -> None:
    """An empty panel marked `mark`: results still to come, or a run not run."""
    ax.set_title(heading)
    ax.text(
        0.5,
        0.5,
        mark,
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
    """`stem.pdf` for the manuscript and `stem.png` to look at, both with their
    images rastered at 300 dpi (a PDF's default, 100, blurs a spectrogram)."""
    out = []
    for suffix, extra in (
        (".pdf", {"metadata": {"CreationDate": None}, "dpi": 300}),
        (".png", {"dpi": 300}),
    ):
        path = Path(stem).with_suffix(suffix)
        with atomic_path(path) as tmp:
            fig.savefig(tmp, format=suffix[1:], **extra)
        out.append(path)
    return out
