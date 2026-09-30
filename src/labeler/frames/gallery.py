"""One picture a shot of a method's test and roster shots (round three, Part B;
spec §3.5).

    python -m labeler.frames.gallery --method M [--limit N]

Each picture is a JPEG, `gallery_dir(paths, spec)/{test,roster}/<shot>.jpg`:
- the rows the model reads, as it reads them: its features (`features`), one
  block of channels a role, scaled 0-1;
- the target per bin where the shot has one (a test shot's, or a roster shot's
  in the split): present, absent, uncertain or unknown, the states `prepare`
  wrote into its features (the original's with the owner's label over them,
  `targets.merged`, F2); else why not (not in the split, or its features
  dropped, with `prepare`'s reason);
- P per frame, the threshold dotted (`train.load`'s: threshold.json's where
  one was re-chosen, T1), and the frames said present shaded.
A test shot is drawn from its prepared features, a roster shot from its review
store (`apply.label`). `--limit` draws the first N of each set. The pictures
are suggestions, never reviewed labels, and their titles say so. Every file is
`frames.VERSION`'s (F1).
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Patch

from ..config import Paths, atomic_path
from ..events import spans
from ..scoring.frames import FRAME_MS
from . import SPECS, VERSION, EventSpec, features_dir, gallery_dir, model_dir, prepare
from . import apply as frames_apply
from . import train as frames_train
from .features import _width
from .targets import ABSENT, PRESENT_T, UNCERTAIN_T, UNKNOWN

log = logging.getLogger(__name__)

TARGET_COLOURS = {
    PRESENT_T: "#d62728",
    ABSENT: "#dddddd",
    UNCERTAIN_T: "#ffb000",
    UNKNOWN: "#ffffff",
}
TARGET_NAMES = {
    PRESENT_T: "present",
    ABSENT: "absent",
    UNCERTAIN_T: "uncertain",
    UNKNOWN: "unknown",
}


def _runs(flags) -> list[tuple[int, int]]:
    """`(start, stop)` of each run of True."""
    edges = np.flatnonzero(np.diff(np.r_[0, np.asarray(flags, np.int8), 0]))
    return list(zip(edges[::2], edges[1::2], strict=True))


def draw(
    path,
    *,
    title: str,
    spec: EventSpec,
    x,
    first: int,
    prob,
    threshold,
    target=None,
    no_target: str = "not in the split",
) -> None:
    """One shot's picture: `x` `(C, subs * n)` features and `prob` per frame from
    frame `first`; `target` `(bin starts ms, states)`, or None and `no_target`,
    why there is none."""
    n = len(prob)
    t0, t1 = first * FRAME_MS, (first + n) * FRAME_MS
    fig = Figure(figsize=(12, 6), dpi=100, layout="constrained")
    rows, strip, model = fig.subplots(
        3, 1, sharex=True, gridspec_kw={"height_ratios": [4, 0.5, 1.5]}
    )
    x = np.asarray(x, dtype=np.float32)
    rows.imshow(
        x,
        aspect="auto",
        extent=(t0, t1, x.shape[0], 0),
        cmap="viridis",
        vmin=0,
        vmax=1,
        interpolation="nearest",
    )
    widths = [_width(role) for role in spec.roles]
    edges = np.cumsum([0, *widths])
    rows.set_yticks((edges[:-1] + edges[1:]) / 2, [role.name for role in spec.roles])
    for edge in edges[1:-1]:
        rows.axhline(edge, color="white", lw=0.6)
    rows.set_ylabel("features")
    if target is not None:
        starts, states = target
        for state, colour in TARGET_COLOURS.items():
            for a, b in _runs(np.asarray(states) == state):
                strip.axvspan(
                    starts[a], starts[b - 1] + spec.bin_ms, color=colour, lw=0
                )
        strip.set_ylabel("target", rotation=0, ha="right", va="center")
    else:
        strip.text(
            0.5,
            0.5,
            f"no target: {no_target}",
            transform=strip.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )
    strip.set_yticks([])
    frame_edges = (first + np.arange(n + 1)) * FRAME_MS
    said = np.nan_to_num(np.asarray(prob, dtype=np.float64), nan=-1.0) >= threshold
    for a, b in _runs(said):
        model.axvspan(frame_edges[a], frame_edges[b], color="#d62728", alpha=0.25, lw=0)
    model.plot(frame_edges[:-1] + FRAME_MS / 2, prob, color="black", lw=0.8)
    model.axhline(threshold, color="black", lw=0.6, ls=":")
    model.set_ylim(0, 1)
    model.set_ylabel("P")
    model.set_xlabel("time (ms)")
    model.set_xlim(t0, t1)
    handles = [
        Patch(color=colour, label=TARGET_NAMES[state])
        for state, colour in TARGET_COLOURS.items()
    ]
    fig.legend(handles=handles, loc="outside lower center", fontsize=8, ncols=4)
    fig.suptitle(title)
    with atomic_path(path) as tmp:
        fig.savefig(tmp, format="jpeg", pil_kwargs={"quality": 85})


def _title(spec: EventSpec, shot: int, which: str, threshold: float) -> str:
    return (
        f"{shot}  {spec.method} {VERSION} suggestions ({which}), not reviewed; "
        f"threshold {threshold:g}"
    )


def _target(paths: Paths, method: str, shot: int, split: dict, gone: dict):
    """`(target, why)`: the shot's `(bin starts, states)` from its features, or
    None and why not; `split` and `gone` are `prepare.split_shots` and
    `prepare.dropped`."""
    path = features_dir(paths, method, VERSION) / f"{shot}.npz"
    if path.is_file():
        with np.load(path) as z:
            return (z["bins"], z["states"]), ""
    if shot not in split:
        return None, "not in the split"
    if shot in gone:
        return None, f"features dropped: {gone[shot]}"
    return None, "features not prepared"


def draw_test(paths: Paths, method: str, model, threshold: float, shots) -> list:
    """The test shots' pictures, from their prepared features."""
    spec, out, drawn = SPECS[method], gallery_dir(paths, SPECS[method], VERSION), []
    for shot in shots:
        with np.load(features_dir(paths, method, VERSION) / f"{shot}.npz") as z:
            x, observed, first = z["x"], z["observed"], int(z["first"])
            target = (z["bins"], z["states"])
        prob = frames_apply.frame_probs(model, spec, x, observed, first)
        draw(
            out / "test" / f"{shot}.jpg",
            title=_title(spec, shot, "test", threshold),
            spec=spec,
            x=x,
            first=first,
            prob=prob,
            threshold=threshold,
            target=target,
        )
        drawn.append(shot)
    return drawn


def draw_roster(paths: Paths, method: str, model, threshold: float, shots) -> tuple:
    """The roster shots' pictures, from their review stores; the shots drawn and
    those that could not be, with why."""
    spec, out = SPECS[method], gallery_dir(paths, SPECS[method], VERSION)
    found = frames_apply.windows(paths)
    split, gone = prepare.split_shots(paths, method), prepare.dropped(paths, method)
    drawn, left = [], {}
    for shot in shots:
        if shot not in found:
            left[str(shot)] = "no window"
            continue
        try:
            got = frames_apply.label(paths, spec, model, shot, found[shot], roster=True)
        except frames_apply.Skipped as reason:
            left[str(shot)] = str(reason)
            continue
        except (*spans.INPUT_MISSING, ValueError) as error:
            left[str(shot)] = f"{type(error).__name__}: {error}"
            continue
        target, why = _target(paths, method, shot, split, gone)
        draw(
            out / "roster" / f"{shot}.jpg",
            title=_title(spec, shot, "roster", threshold),
            spec=spec,
            x=got.x,
            first=got.first,
            prob=got.prob,
            threshold=threshold,
            target=target,
            no_target=why,
        )
        drawn.append(shot)
    return drawn, left


def gallery(paths: Paths, method: str, *, limit: int = 0) -> dict:
    """Draw the method's test and roster pictures (module docstring)."""
    spec = SPECS[method]
    model, blob = frames_train.load(model_dir(paths, method, VERSION) / "model.pt")
    threshold = float(blob["threshold"])
    split = prepare.split_shots(paths, method)
    test = [
        s
        for s in sorted(split)
        if split[s] == "test"
        and (features_dir(paths, method, VERSION) / f"{s}.npz").is_file()
    ]
    roster = frames_apply.set_shots(paths, spec, "roster")
    if limit:
        test, roster = test[:limit], roster[:limit]
    drawn_roster, left = draw_roster(paths, method, model, threshold, roster)
    return {
        "method": method,
        "test": draw_test(paths, method, model, threshold, test),
        "roster": drawn_roster,
        "left_out": left,
        "folder": str(gallery_dir(paths, spec, VERSION)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--method", required=True, choices=list(SPECS))
    p.add_argument("--limit", type=int, default=0, help="the first N of each set")
    args = p.parse_args(argv)
    if args.limit < 0:
        p.error("--limit must be nonnegative")
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    try:
        record = gallery(Paths.from_env(), args.method, limit=args.limit)
    except (OSError, ValueError) as error:
        p.error(str(error))
    counts = {k: len(record[k]) for k in ("test", "roster", "left_out")}
    print(json.dumps({"method": args.method, **counts, "folder": record["folder"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
