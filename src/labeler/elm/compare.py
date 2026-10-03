"""The reviewed shots, their bins, and every method's part on a common set of bins.

`elm-ours`, `elm-dsm`, `elm-elmo` and the ELM clock are scored on the review's 50 ms
bins; this module builds the pieces the evaluation and the reference-swap scripts
share: the two sets of shots with their analysed time, the detected spans of the two
methods that make spans (ELM-O, the clock), the `elm-dsm` scores (`DsmScores`) and
`shot_parts`, one shot's `score.ShotScore` for every method.

**Sets.** `bes73` is the 73 reviewed shots ELM-O runs on (it needs BES); analysed time
is ELM-O's chunks (`review_coverage.csv`). `all119` is every reviewed shot; analysed
time is where the fetched D-alpha and density records have samples. A set's bins are
the benchmark's own (`labeler.elm.labels.scored_bins`).

**Common bins.** The DSM serves rows only where the ECE record covers the 50 ms they
summarise and only up to 5975 ms, so a comparison that includes it keeps the bins whose
forecast row (the one before the bin) and detection row (the one summarising it) both
exist, for every method, and cuts every method's analysed time to the time those rows
summarise (`dsm.usable_cover`), which is what the span counts use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from ..config import Paths
from . import dsm, inputs, labels, methods, prepare, score
from .train import ShotData

ELMO_DIR = Path("benchmarks/elm/elmo")
CLOCK_CSV = Path(
    "suggestions/elm_clock/v1/edge_localized_mode_suggest_elm_clock_v1.csv"
)
#: the forecast row of a bin is the one ending where the bin starts
FORECAST_LAG_ROWS = -int(methods.WINDOW_MS / methods.ROW_MS)
NAME = {
    "ours": "elm-ours",
    "dsm": "elm-dsm",
    "detect": "elm-dsm-detect",
    "init": "elm-dsm-detect-init",
    "elmo": "elm-elmo",
    "clock": "elm-clock",
}


#: the two detection retrains of the DSM (from scratch; from the published embedding)
VARIANTS = (NAME["detect"], NAME["init"])


@dataclass
class SetDef:
    """A set of reviewed shots with each shot's bins and analysed time."""

    name: str
    shots: list[int]
    bins: dict[int, labels.Bins]
    cover: dict[int, pd.DataFrame]
    has_elmo: bool = False


def own_coverage(paths: Paths, data: dict[int, ShotData]) -> dict[int, pd.DataFrame]:
    out = {}
    for s in data:
        c0, c1 = inputs.valid_intervals(
            np.load(prepare.inputs_dir(paths) / f"{s}.npy")[inputs.VALID]
        )
        out[s] = methods.cover_frame(*labels.merge_intervals(c0, c1))
    return out


def load_sets(paths: Paths, data: dict[int, ShotData]) -> dict[str, SetDef]:
    """`bes73` and `all119`."""
    all_shots = sorted(data)
    cover = pd.read_csv(paths.root / ELMO_DIR / "review_coverage.csv")
    shots_bes = sorted(int(s) for s in cover.shot.unique())
    elmo_cover = {
        s: methods.cover_frame(
            *labels.merge_intervals(
                *(g.sort_values("t_start_ms")[c] for c in ("t_start_ms", "t_end_ms"))
            )
        )
        for s, g in cover.groupby("shot")
    }
    elmo_bins = {
        s: labels.scored_bins(
            data[s].spans,
            elmo_cover[s].t_start_ms.to_numpy(float),
            elmo_cover[s].t_end_ms.to_numpy(float),
        )
        for s in shots_bes
    }
    return {
        "bes73": SetDef(
            "bes73", shots_bes, elmo_bins, {s: elmo_cover[s] for s in shots_bes}, True
        ),
        "all119": SetDef(
            "all119",
            all_shots,
            {s: data[s].bins for s in all_shots},
            own_coverage(paths, data),
        ),
    }


def load_detected(paths: Paths):
    """Detected spans of ELM-O (the published setting, 73 shots) and the ELM clock."""
    found = pd.read_csv(paths.root / ELMO_DIR / "review_elms.csv")
    found = found[found.variant == "paper"]
    elmo = {
        int(s): g[["t_start_ms", "t_end_ms"]].reset_index(drop=True)
        for s, g in found.groupby("shot")
    }
    clock = pd.read_csv(paths.root / CLOCK_CSV)
    clock = clock[clock.category == 1].rename(
        columns={"t_start": "t_start_ms", "t_end": "t_end_ms"}
    )
    clock = {
        int(s): g[["t_start_ms", "t_end_ms"]].reset_index(drop=True)
        for s, g in clock.groupby("shot")
    }
    return elmo, clock


@dataclass
class DsmScores:
    """`elm-dsm` on the reviewed shots: rows, the published risk, detector scores."""

    rows: dict[int, dsm.Rows]
    risk: dict[int, np.ndarray]  # (240, 4): published risk at 5, 10, 20, 50 ms
    scores: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    threshold: dict[str, dict[int, float]] = field(default_factory=dict)

    def save(self, directory: Path) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            directory / "published_risk.npz",
            **{f"s{s}": v.astype(np.float32) for s, v in self.risk.items()},
        )
        for name, by_shot in self.scores.items():
            np.savez_compressed(
                directory / f"scores_{name}.npz",
                **{f"s{s}": v.astype(np.float32) for s, v in by_shot.items()},
            )
        (directory / "thresholds.json").write_text(
            json.dumps(
                {
                    n: {str(s): t for s, t in v.items()}
                    for n, v in self.threshold.items()
                }
            )
        )

    @classmethod
    def load(cls, directory: Path, rows: dict[int, dsm.Rows], variants=()):
        with np.load(directory / "published_risk.npz") as z:
            risk = {int(k[1:]): z[k] for k in z.files}
        scores = {}
        for name in variants:
            with np.load(directory / f"scores_{name}.npz") as z:
                scores[name] = {int(k[1:]): z[k] for k in z.files}
        thr = json.loads((directory / "thresholds.json").read_text())
        return cls(
            rows,
            risk,
            scores,
            {n: {int(s): t for s, t in v.items()} for n, v in thr.items()},
        )


def shot_parts(
    shot: int,
    spans: pd.DataFrame,
    bins0: labels.Bins,
    cover0: pd.DataFrame,
    oof: methods.Oof,
    dscores: DsmScores,
    elmo_spans: dict | None,
    clock_spans: dict,
):
    """Every method's `ShotScore` on the shot's common bins, and those bins."""
    r = dscores.rows[shot]
    bins = dsm.bins_with_rows(bins0, r.usable, lags=(FORECAST_LAG_ROWS, 0))
    cover = dsm.usable_cover(r.usable, cover0)
    out = {
        NAME["ours"]: methods.trace_part(
            spans, shot, bins, cover, oof.trace(shot)[0], oof.threshold[shot]
        ),
        NAME["dsm"]: methods.row_part(
            spans,
            shot,
            bins,
            cover,
            dsm.ROW_T_MS,
            dscores.risk[shot][:, -1],
            dscores.threshold[NAME["dsm"]][shot],
            lag_rows=FORECAST_LAG_ROWS,
            ahead=True,
        ),
    }
    for key in ("detect", "init"):
        if NAME[key] in dscores.scores:
            out[NAME[key]] = methods.row_part(
                spans,
                shot,
                bins,
                cover,
                dsm.ROW_T_MS,
                dscores.scores[NAME[key]][shot],
                dscores.threshold[NAME[key]][shot],
            )
    if elmo_spans is not None:
        e = elmo_spans.get(shot, methods.span_frame([], []))
        out[NAME["elmo"]] = methods.span_part(spans, shot, bins, cover, e)
    c = clock_spans.get(shot, methods.span_frame([], []))
    out[NAME["clock"]] = methods.span_part(spans, shot, bins, cover, c)
    return out, bins


def common_parts(
    sdef: SetDef,
    data: dict[int, ShotData],
    oof: methods.Oof,
    dscores: DsmScores,
    elmo_spans: dict,
    clock_spans: dict,
):
    """`(parts by method name, bins by shot)` over the set's shots, common bins."""
    parts: dict[str, list[score.ShotScore]] = {}
    bins_of = {}
    for shot in sdef.shots:
        one, bins = shot_parts(
            shot,
            data[shot].spans,
            sdef.bins[shot],
            sdef.cover[shot],
            oof,
            dscores,
            elmo_spans if sdef.has_elmo else None,
            clock_spans,
        )
        for name, part in one.items():
            parts.setdefault(name, []).append(part)
        bins_of[shot] = bins
    return parts, bins_of
