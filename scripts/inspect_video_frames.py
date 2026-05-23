"""Save sample tangtv frames as PNGs for visual inspection.

Step 0 follow-up. The Step-0 inspection script measured a NaN fraction
of ~65% in mid-shot frames, which we initially interpreted as a spatial
off-sensor region. Subsequent debugging revealed the 7 "channels" are
optical filters and most of the NaN budget is fully-NaN off-channels,
not an off-FOV mask. This script renders the *active* channels of two
representative shots so the user can confirm whether any spatial
off-sensor region exists *within* an active channel.

Output: ``inspect_video_frames/{shot}_ch{C}_t{frame}.png`` at the raw
240x720 resolution, plus a ``summary.txt`` listing per-channel stats.

Read-only on the data; only writes to ``inspect_video_frames/``.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


DATA_DIR = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
OUT_DIR = Path("inspect_video_frames")
OUT_DIR.mkdir(exist_ok=True)


# Two shots representative of typical tangtv data:
# - 191599: filters 4 and 6 active (from earlier debugging)
# - 204510: filters 0, 2, 4, 6 active
SHOTS = [
    ("191599_processed.h5", [4, 6]),
    ("204510_processed.h5", [0, 2, 4, 6]),
]


def render_frame(arr: np.ndarray, out_path: Path, title: str) -> dict:
    """Save *arr* as a labelled PNG. Returns per-frame stats."""
    finite = arr[np.isfinite(arr)]
    stats = {
        "shape": arr.shape,
        "nan_frac": float(np.isnan(arr).mean()),
        "min": float(finite.min()) if finite.size else float("nan"),
        "max": float(finite.max()) if finite.size else float("nan"),
        "mean": float(finite.mean()) if finite.size else float("nan"),
        "p01": float(np.percentile(finite, 1)) if finite.size else float("nan"),
        "p99": float(np.percentile(finite, 99)) if finite.size else float("nan"),
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    # Stretch to 1st–99th percentile so faint structure is visible without
    # being washed out by bright spikes; NaN renders as black via cmap.bad.
    cmap = plt.get_cmap("inferno").copy()
    cmap.set_bad(color="cyan")  # cyan = NaN, very visible against inferno
    masked = np.ma.array(arr, mask=np.isnan(arr))
    im = ax.imshow(masked, cmap=cmap, vmin=stats["p01"], vmax=stats["p99"],
                   aspect="auto")
    fig.colorbar(im, ax=ax)
    ax.set_title(
        f"{title}\n"
        f"shape={arr.shape}  "
        f"nan_frac={stats['nan_frac']:.3f}  "
        f"min={stats['min']:.1f}  max={stats['max']:.1f}  "
        f"mean={stats['mean']:.1f}\n"
        f"(p01..p99 stretch; cyan = NaN)"
    )
    ax.set_xlabel("W")
    ax.set_ylabel("H")
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return stats


def main() -> None:
    log_lines = []
    for shot_name, active_channels in SHOTS:
        shot_path = DATA_DIR / shot_name
        if not shot_path.exists():
            log_lines.append(f"SKIP {shot_name}: not found")
            continue
        log_lines.append(f"\n=== {shot_name} ===")
        with h5py.File(shot_path, "r") as f:
            yd = f["tangtv"]["ydata"]
            n_frames = yd.shape[1]
            mid = n_frames // 2
            # Pick three frames: 25%, 50%, 75% of the way through
            picks = [n_frames // 4, mid, (3 * n_frames) // 4]
            for c in active_channels:
                for t_idx in picks:
                    arr = np.asarray(yd[c, t_idx, :, :], dtype=np.float32)
                    out = OUT_DIR / (
                        f"{shot_path.stem}_ch{c}_t{t_idx}.png"
                    )
                    title = (
                        f"{shot_path.stem}  channel {c}  frame {t_idx} "
                        f"of {n_frames}"
                    )
                    stats = render_frame(arr, out, title)
                    log_lines.append(
                        f"  ch{c} t{t_idx}: nan={stats['nan_frac']:.3f}  "
                        f"range=[{stats['min']:.1f}, {stats['max']:.1f}]  "
                        f"mean={stats['mean']:.1f}  -> {out.name}"
                    )

    summary = OUT_DIR / "summary.txt"
    summary.write_text("\n".join(log_lines))
    print("\n".join(log_lines))
    print(f"\nWrote PNGs and summary.txt to {OUT_DIR.resolve()}")


if __name__ == "__main__":
    main()