r"""fig_ae_annotation: why the dense AE labels and the Heidbrink annotation differ.

    PYTHONPATH=src pixi run --frozen -e labelmaker \
        python scripts/labeler/paper/fig_ae_annotation.py \
        [--out dev/label_paper/figures/fig_ae_annotation.pdf] [--png PATH] \
        [--shots A B]

Two of the 19 held-out AE shots (the ones the benchmark scores,
`outputs/labeler/ae/baselines/evaluation.json`), one column each. Per column the
CO2 R0 spectrogram, 80-250 kHz (the AE review page's band), read from the shot's
AE review store, and two tracks on its time axis over 0-2 s:

- the Heidbrink annotation, as the benchmark scores it: the annotation's samples
  on the uniform grid of the shot's 7,820 columns, a 10 ms frame present when
  half its columns are (the dataset's `annotated`, `labeler.ae.dataset`). A frame
  it does not mark is blank: the annotation never says absent;
- the dense labels: the expert's review (`review/labels.csv`), each 10 ms frame
  present or absent (`labeler.scoring.frames`).

Each shot's gap is the dense labels' present share of 0-2 s less the
annotation's. The two shots (`PICK_RULE`): the largest gap of the 19, and the
largest gap among the shots whose dense labels also call a good part of the shot
absent (present share under the 19 shots' mean), so the second column shows the
dense labels stopping where the activity does. `--shots` names two instead.
The rule and each shot's shares are printed.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.figure import Figure
from matplotlib.patches import Patch, Rectangle

from labeler.ae.xpower.data import targets
from labeler.config import Paths
from labeler.events.catalog.states import ABSENT, PRESENT
from labeler.events.review import labels as review_labels
from labeler.events.review import rows as store_rows
from labeler.paper import PAGE_IN, style
from labeler.paper.roster import read_row
from labeler.paper.shots import show_image
from labeler.scoring.frames import FRAME_MS

REPO = Path(__file__).resolve().parents[3]
EVALUATION = REPO / "outputs" / "labeler" / "ae" / "baselines" / "evaluation.json"
DEFAULT_OUT = REPO / "dev" / "label_paper" / "figures" / "fig_ae_annotation.pdf"
RECORD_MS = 2000
N_FRAMES = RECORD_MS // FRAME_MS
SELD_FRAMES = 7820  # the dataset's columns over 0-2 s
CO2_ROW = "R0"
BAND_KHZ = (80.0, 250.0)
PICK_RULE = (
    "left: the largest gap (dense less annotation present share, 0-2 s) of the "
    "19 held-out shots; right: the largest gap among shots whose dense present "
    "share is under the 19 shots' mean"
)

PRESENT_COLOUR = "#d62728"  # the catalog figures' present
ABSENT_COLOUR = "#e2e2e2"  # and their absent
LIGHTER = 0.5  # the annotation's tint: this share of the colour, the rest white
INK = "#333333"
EDGE = "#999999"
BAR = (0.12, 0.76)


def tint(colour: str, share: float = LIGHTER) -> tuple[float, ...]:
    return tuple(share * c + (1 - share) for c in to_rgb(colour))


def frame_mean(values: np.ndarray) -> np.ndarray:
    """The mean of `values` (SELD_FRAMES columns over 0-2 s) in each 10 ms frame."""
    centre = (np.arange(values.size) + 0.5) * RECORD_MS / values.size
    k = np.minimum((centre // FRAME_MS).astype(int), N_FRAMES - 1)
    return np.bincount(k, weights=values, minlength=N_FRAMES) / np.maximum(
        np.bincount(k, minlength=N_FRAMES), 1
    )


def runs(flags: np.ndarray) -> list[tuple[float, float]]:
    """`(start ms, length ms)` of each run of true frames."""
    edge = np.diff(np.r_[0, flags.astype(int), 0])
    starts, stops = np.where(edge == 1)[0], np.where(edge == -1)[0]
    return [(a * FRAME_MS, (b - a) * FRAME_MS) for a, b in zip(starts, stops)]


def annotation_frames(paths: Paths, shot: int) -> np.ndarray:
    """The Heidbrink annotation's present frames, as the benchmark scores them."""
    path = paths.root / "ae" / "dataset" / f"{shot}_valid.npz"
    with np.load(path) as z:
        annotated = z["annotated"].astype(float)
    if annotated.size != SELD_FRAMES:
        raise ValueError(f"{path}: {annotated.size} columns, not {SELD_FRAMES}")
    return frame_mean(annotated) >= 0.5


def dense_states(table: dict, shot: int) -> np.ndarray:
    """The expert's state of each 10 ms frame of 0-2 s."""
    return targets(table[shot], 0, N_FRAMES)


def gaps(paths: Paths, shots: list[int]) -> dict[int, dict[str, float]]:
    """Each shot's present shares of 0-2 s: annotation, dense, and their gap."""
    table = review_labels.read_labels(
        paths.label_tables / "alfven_eigenmode/review/labels.csv"
    )
    out = {}
    for shot in shots:
        states = dense_states(table, shot)
        if not np.isin(states, (ABSENT, PRESENT)).all():
            raise ValueError(f"shot {shot}: dense labels leave frames unassessed")
        ann = annotation_frames(paths, shot).mean()
        dense = (states == PRESENT).mean()
        out[shot] = {
            "annotation": float(ann),
            "dense": float(dense),
            "gap": float(dense - ann),
        }
    return out


def pick(shares: dict[int, dict[str, float]]) -> tuple[int, int]:
    """`PICK_RULE`'s two shots; ties go to the lower shot number."""
    mean_dense = float(np.mean([v["dense"] for v in shares.values()]))

    def best(shots):
        return min(shots, key=lambda s: (-round(shares[s]["gap"], 6), s))

    first = best(shares)
    rest = [s for s in shares if s != first and shares[s]["dense"] < mean_dense]
    return first, best(rest)


def spectrogram(ax, paths: Paths, shot: int, ylabel: bool) -> None:
    store = paths.spectrogram_file("alfven_eigenmode", shot)
    row = next(r for r in store_rows.meta(store)["rows"] if r["name"] == CO2_ROW)
    read = read_row(store, row, 0, RECORD_MS)
    if read is None:
        raise ValueError(
            f"shot {shot}: the review store does not reach 0-{RECORD_MS} ms"
        )
    show_image(
        ax, read.values, read.extent, BAND_KHZ[1], "CO$_2$ R0\nkHz" if ylabel else ""
    )
    ax.set_ylim(*BAND_KHZ)
    ax.set_xlim(0, RECORD_MS)
    if not ylabel:
        ax.set_yticks([])


def track(ax, spans, colour, name: str | None, *, backdrop=None) -> None:
    ax.set_xlim(0, RECORD_MS)
    ax.set_xticks(np.arange(0, RECORD_MS + 1, 500))
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    if backdrop is not None:
        ax.broken_barh(backdrop, BAR, facecolor=ABSENT_COLOUR, lw=0)
    if spans:
        ax.broken_barh(spans, BAR, facecolor=colour, lw=0)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    if name:
        ax.set_ylabel(name, rotation=0, ha="right", va="center")


def draw(out: Path, png: Path | None, shots: tuple[int, int] | None) -> None:
    paths = Paths.from_env()
    fair = json.loads(EVALUATION.read_text())["shots"]["fair"]
    shares = gaps(paths, fair)
    print(f"rule: {PICK_RULE}")
    for shot in sorted(shares, key=lambda s: -shares[s]["gap"]):
        v = shares[shot]
        print(
            f"  {shot}: annotation {v['annotation']:.1%}, dense {v['dense']:.1%}, "
            f"gap {v['gap']:.1%}"
        )
    chosen = shots if shots else pick(shares)
    table = review_labels.read_labels(
        paths.label_tables / "alfven_eigenmode/review/labels.csv"
    )
    print(f"shots: {chosen}")
    with style():
        fig = Figure(figsize=(PAGE_IN, 2.75))
        grid = fig.add_gridspec(
            3,
            2,
            height_ratios=[4.2, 0.9, 0.9],
            left=0.115,
            right=0.98,
            top=0.9,
            bottom=0.2,
            hspace=0.1,
            wspace=0.1,
        )
        for col, shot in enumerate(chosen):
            ax = [fig.add_subplot(grid[i, col]) for i in range(3)]
            first = col == 0
            spectrogram(ax[0], paths, shot, first)
            ann = annotation_frames(paths, shot)
            dense = dense_states(table, shot) == PRESENT
            track(
                ax[1],
                runs(ann),
                tint(PRESENT_COLOUR),
                "annotation" if first else None,
            )
            track(
                ax[2],
                runs(dense),
                PRESENT_COLOUR,
                "dense\nlabels" if first else None,
                backdrop=runs(~dense),
            )
            ax[1].add_patch(
                Rectangle(
                    (0, BAR[0]),
                    RECORD_MS,
                    BAR[1],
                    facecolor="none",
                    edgecolor=EDGE,
                    lw=0.4,
                    ls=(0, (2, 1.5)),
                )
            )
            ax[0].set_xticks(np.arange(0, RECORD_MS + 1, 500))
            for a in ax[:2]:
                a.tick_params(labelbottom=False)
            ax[1].tick_params(bottom=False)
            ax[2].set_xlabel("time (ms)")
            v = shares[shot]
            ax[0].set_title(
                f"shot {shot}: present {v['annotation']:.0%} (annotation), "
                f"{v['dense']:.0%} (dense labels)",
                loc="left",
            )
            for start, length in runs(ann):  # the annotation's windows, on the image
                for edge in (start, start + length):
                    ax[0].axvline(edge, color="white", lw=0.5, ls=(0, (3, 2)))
        fig.legend(
            handles=[
                Patch(color=PRESENT_COLOUR, label="present (dense labels)"),
                Patch(color=tint(PRESENT_COLOUR), label="present (annotation)"),
                Patch(color=ABSENT_COLOUR, label="absent (dense labels)"),
                Patch(
                    facecolor="none",
                    edgecolor=EDGE,
                    ls=(0, (2, 1.5)),
                    label="not marked: not absent",
                ),
            ],
            loc="lower center",
            ncol=4,
            frameon=False,
            fontsize=6,
            handlelength=1.6,
            columnspacing=1.6,
            bbox_to_anchor=(0.55, 0.0),
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, format="pdf", metadata={"CreationDate": None}, dpi=300)
        if png is not None:
            png.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(png, dpi=200)
    print(f"wrote {out}", file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--png", type=Path, help="also save a PNG preview here")
    parser.add_argument(
        "--shots", type=int, nargs=2, help="two shots instead of the rule's"
    )
    args = parser.parse_args(argv)
    draw(args.out, args.png, tuple(args.shots) if args.shots else None)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
