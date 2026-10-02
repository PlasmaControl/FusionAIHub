"""The paper's figures and tables, read from what the round-two runs wrote.

The manuscript (`dev/label_paper`: "Tokamak-SI: Automatic Shot Interpreter with
a Catalog of Fusion Reactor Data") shows the interpreter on one phenomenon, AE,
first. A figure that will show every paper phenomenon (`ORDER`: the catalog's,
less disruption for now) draws all five now. The score figure draws each
phenomenon's selected model: its test F1, and its ROC and precision-recall
curve over the same test shots.
The coverage figure and table count all five: AE from the round-two runs, the
other four from their frame models' splits (the original labels with the
owner's reviews over them, F2), the owner's review and the applications; only a
phenomenon with none of its inputs yet is marked "coming".

- `scores`: the selected models' test F1, ROC and PR (`fig_scores`), and the AE
  methods' and the segmentation's score tables;
- `roc`: each selected model's ROC and PR over the test shots its F1 was scored on,
  recorded as `roc.json` beside its evaluation;
- `coverage`: labelled, reviewed, positive and suggested shots per phenomenon;
- `shots`: AE examples from the test shots, and the pieces of a shot figure
  the interpreter figure shares (the spectrogram, the mask, the state bars,
  the legend);
- `frame_examples`: the frame models' examples (`fig_examples_ntm`,
  `fig_examples_hmode`, `fig_examples_elm`, `fig_examples_sawtooth`): three test
  shots each, the best, median and worst F1, drawn as the frames gallery draws
  a test shot (`labeler.frames.gallery`);
- `roster`: the interpreter figure, `fig_interpreter`: one roster shot's
  suggestions from the models, every phenomenon on it;
- `label_figure`: the manuscript's `fig_interpreter` now: one cohort shot's
  review-page rows with the catalog's labels (expert reviews, imported tables
  and the automated labeler's generated ones, each with its tier); its own
  CLI, not `build`;
- `build`: every product into `$LABELER_ROOT/paper/`, and the copy into the
  manuscript's `figures/`;
- `snapshot`: each file the build reads, read once and pinned by its sha256;
- `staging`: the directory the build draws into, and the swap that puts it in
  the old output's place.

Nothing here trains or fetches. The score figure and tables read the JSON the
evaluations and `roc` wrote; to rank and draw the example shots, `build` runs the chosen
model over its test shots' review stores, against the labels it was scored on,
and to draw the interpreter figure, over its roster shot's corpus CO2. `build`
records every input it read.
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
#: torch's threads: the build and the roster CLI run one shot at a time, on the
#: login node.
LOGIN_THREADS = 2
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
