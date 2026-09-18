"""Loading and plotting saved event-label grids.

Reading a category's ORIGINAL annotation format is case-by-case work - six
formats, six readers, one consumer each - so it lives in each category's
`example.ipynb` rather than here. What stays is what is generic over the saved
grid: opening one, and drawing it.
"""

from pathlib import Path

import numpy as np

from labeler.config import Paths

from .interval_tables import SAMPLE_MS, read_label_grid


def load_shot(event: str, shot: int, *, source="format/shots", root=None) -> dict:
    """Open one formatted or extended NPZ, independently of the working directory.

    By default the data root is LABELER_LABEL_TABLES or the checkout's
    data/events directory. Only the selected NPZ is read.
    """
    root = Paths.from_env().label_tables if root is None else Path(root)
    path = root / event / source / f"{int(shot)}.npz"
    grid = read_label_grid(path)
    return {
        **grid,
        "shot": int(shot),
        "event": event,
        "source": source,
        "data_root": str(root),
    }


def plot_shot(grid: dict):
    """Plot saved integer labels, keeping unknown cells white and zero gray.

    ELM onset counts, when present, get a second panel. Returns the figure
    so notebook users can save it with figure.savefig(...).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap

    ids = sorted(int(k) for k in grid["categories"])
    palette = ["#dddddd", "#2166ac", "#e69f00", "#009e73", "#8856a7"]
    colors = ListedColormap([palette[i % len(palette)] for i in range(len(ids))])
    colors.set_bad("white")
    # Map category IDs to colors even when the IDs are not contiguous.
    values = np.full(grid["label"].shape, np.nan)
    for index, category in enumerate(ids):
        values[grid["label"] == category] = index
    norm = BoundaryNorm(np.arange(len(ids) + 1) - 0.5, colors.N)
    counts = grid.get("event_count")
    fig, axes = plt.subplots(
        2 if counts is not None else 1,
        1,
        figsize=(10, 6 if counts is not None else 3.5),
        sharex=True,
        squeeze=False,
        layout="constrained",
    )
    if counts is not None:
        axes[0, 0].bar(
            grid["time_ms"], counts, width=SAMPLE_MS, align="edge", color="#2166ac"
        )
        axes[0, 0].set(
            ylabel="Labelled onset count", ylim=(0, max(1, counts.max()) + 1)
        )
    ax = axes[-1, 0]
    image = ax.pcolormesh(
        np.r_[grid["time_ms"], grid["time_ms"][-1] + SAMPLE_MS],
        grid["rho_edges"],
        values.T,
        shading="flat",
        cmap=colors,
        norm=norm,
    )
    ax.set(xlabel="Time (ms)", ylabel="Normalized rho", ylim=(0, 1))
    axes[0, 0].set_title(f"Shot {grid['shot']} — {grid['source']}")
    bar = fig.colorbar(image, ax=axes.ravel().tolist(), ticks=range(len(ids)))
    bar.ax.set_yticklabels([f"{i}: {grid['categories'][str(i)]}" for i in ids])
    if not np.isfinite(values).any():
        ax.text(
            0.5,
            0.5,
            "No known labels for this shot",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
    plt.show()
    return fig
