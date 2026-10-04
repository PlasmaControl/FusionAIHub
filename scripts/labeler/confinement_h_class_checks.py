r"""confinement_h_class_checks: what the corpus shots' I-coil currents say about our H.

    PYTHONPATH=src pixi run --frozen --no-install -e labelmaker \
        python scripts/labeler/confinement_h_class_checks.py [--out PATH.json]

The paper's H-mode (Gill et al. 2024) is "standard ELMy H-mode": it leaves out
ELM-suppressed, ELM-free and negative-triangularity H-modes. Whether our H class (the
experts' intervals, merged from Gill's tables) contains such plasmas is not verified.
This script reads the one cheap indicator on disk, and writes it down as an
observation, not a test:

* the 12 I-coil currents (``rmp`` group of the corpus HDF5: IU/IL 30, 90, 150, 210,
  270, 330) of the 117 corpus shots, in amperes. The peak ``max |I|`` over the 12
  channels and over each curated interval is taken per interval, and per class the
  shots whose intervals reach 1 kA are counted. Resonant magnetic perturbations that
  suppress ELMs take a few kA, but the same coils are driven at about 1.5 kA for
  non-resonant fields in QH-mode work, so a current says "an applied field", not
  "ELM-suppressed". The 1 kA is a reading convention, fixed before the count and not
  tuned.

Not checked, because the quantity is not on disk and fetching was out of scope: the EFIT
triangularity (the negative-triangularity H-mode), H98y2 and the ELM state.

The record: ``outputs/labeler/confinement/bes/h_class_checks.json``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import h5py
import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from labeler.confinement import bes_windows as bw

LABELER = Path(
    os.environ.get("LABELER_ROOT", "/scratch/gpfs/EKOLEMEN/nc1514/labelmaker")
)
CORPUS = Path("/scratch/gpfs/EKOLEMEN/foundation_model")
BES_CORPUS = LABELER / "round4/conf/bes500k"
DEFAULT_OUT = REPO / "outputs/labeler/confinement/bes/h_class_checks.json"
#: A current the reading counts as "a field is applied" (A).
APPLIED_A = 1000.0
COILS = ("IU30", "IU90", "IU150", "IU210", "IU270", "IU330") + (
    "IL30",
    "IL90",
    "IL150",
    "IL210",
    "IL270",
    "IL330",
)


def interval_peaks(shot: int, intervals) -> list[dict]:
    """Peak |I| (A) over the 12 coils and over each interval of a shot; ``None`` where
    the interval holds no finite sample."""
    with h5py.File(CORPUS / f"{shot}_processed.h5", "r") as f:
        t_ms = f["rmp"]["xdata"][:] * 1000.0
        current = np.abs(f["rmp"]["ydata"][:])
    out = []
    for iv in intervals.itertuples():
        inside = (t_ms >= iv.t_start) & (t_ms <= iv.t_end)
        block = current[:, inside]
        peak = float(np.nanmax(block)) if np.isfinite(block).any() else None
        out.append(
            {
                "shot": int(shot),
                "interval": int(iv.interval),
                "class": bw.CLASSES[int(iv.label)],
                "t_start_ms": float(iv.t_start),
                "t_end_ms": float(iv.t_end),
                "peak_a": peak,
            }
        )
    return out


def summarise(rows: list[dict]) -> dict:
    """Per class: intervals and shots read, those at or above ``APPLIED_A``."""
    summary = {}
    for cls in bw.CLASSES:
        mine = [r for r in rows if r["class"] == cls and r["peak_a"] is not None]
        shots = sorted({r["shot"] for r in mine})
        peak = {s: max(r["peak_a"] for r in mine if r["shot"] == s) for s in shots}
        on = [s for s in shots if peak[s] >= APPLIED_A]
        summary[cls] = {
            "intervals": len(mine),
            "intervals_at_or_above": sum(r["peak_a"] >= APPLIED_A for r in mine),
            "shots": len(shots),
            "shots_at_or_above": len(on),
            "shot_list_at_or_above": on,
            "shot_peak_a_median": float(np.median(list(peak.values())))
            if peak
            else None,
            "shot_peak_a_max": float(max(peak.values())) if peak else None,
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    corpus_shots = sorted(int(f.stem) for f in BES_CORPUS.glob("*.npz"))
    intervals = bw.curated_intervals()
    rows, unread = [], []
    for shot in corpus_shots:
        mine = intervals[intervals.shot == shot]
        if mine.empty:
            continue
        try:
            rows += interval_peaks(shot, mine)
        except (KeyError, OSError):
            unread.append(shot)
    record = {
        "created": datetime.now(UTC).isoformat(timespec="seconds"),
        "what": "peak |I| over the 12 I-coil currents (rmp group, A) in each curated "
        "interval of the corpus shots; an observation, not a test of the H class",
        "applied_a": APPLIED_A,
        "corpus_shots": len(corpus_shots),
        "shots_without_the_group": unread,
        "classes": summarise(rows),
        "intervals": rows,
        "not_checked": "EFIT triangularity, H98y2 and the ELM state are not on disk",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1))
    for cls, e in record["classes"].items():
        print(
            f"{cls:3s} {e['shots']:3d} shots, {e['shots_at_or_above']:3d} reach "
            f"{APPLIED_A:.0f} A ({e['intervals_at_or_above']} of {e['intervals']} "
            f"intervals); shot peak median {e['shot_peak_a_median']}, max "
            f"{e['shot_peak_a_max']}"
        )
    print(f"{len(unread)} shots without the group; wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
