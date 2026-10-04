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
from . import dsm, inputs, labels, methods, prepare, score, swap
from .train import ShotData

ELMO_DIR = Path("benchmarks/elm/elmo")
CLOCK_CSV = Path(
    "suggestions/elm_clock/v1/edge_localized_mode_suggest_elm_clock_v1.csv"
)
#: the forecast row of a bin is the one ending where the bin starts
FORECAST_LAG_ROWS = -int(methods.WINDOW_MS / methods.ROW_MS)
# Float32 second timestamps can shift nominal bin edges by about 1e-5 ms.
# This tolerance is 1000 times smaller than the diagnostic's 0.1 ms grid.
COVER_BOUNDARY_TOL_MS = 1e-4
NAME = {
    "ours": "elm-ours",
    "dsm": "elm-dsm",
    "detect": "elm-dsm-detect",
    "exposed": "elm-dsm-detect-exposed",
    "init": "elm-dsm-detect-init",
    "elmo": "elm-elmo",
    "clock": "elm-clock",
}
DISPLAY_NAME = {
    NAME["ours"]: "elm-ours",
    NAME["dsm"]: "elm-dsm-survival",
    NAME["detect"]: "elm-dsm-detect (60-input 1×128)",
    NAME["exposed"]: "elm-dsm-detect (source statistics)",
    NAME["init"]: "elm-dsm-detect (source weights and statistics)",
    NAME["elmo"]: "elm-elmo",
    NAME["clock"]: "elm-clock",
}


#: Confirmatory fold-isolated detection and historical exposed supplemental fits.
VARIANTS = (NAME["detect"], NAME["exposed"], NAME["init"])


@dataclass
class SetDef:
    """A set of reviewed shots with each shot's bins and analysed time."""

    name: str
    shots: list[int]
    bins: dict[int, labels.Bins]
    cover: dict[int, pd.DataFrame]
    has_elmo: bool = False
    elmo_cover: dict[int, pd.DataFrame] = field(default_factory=dict)


def own_coverage(paths: Paths, data: dict[int, ShotData]) -> dict[int, pd.DataFrame]:
    out = {}
    for s in data:
        c0, c1 = inputs.valid_intervals(
            np.load(prepare.inputs_dir(paths) / f"{s}.npy")[inputs.VALID]
        )
        out[s] = methods.cover_frame(
            *labels.merge_intervals(c0, c1, tol=COVER_BOUNDARY_TOL_MS)
        )
    return out


def load_sets(paths: Paths, data: dict[int, ShotData]) -> dict[str, SetDef]:
    """`bes73` and `all119`."""
    all_shots = sorted(data)
    input_cover = own_coverage(paths, data)
    cover = pd.read_csv(paths.root / ELMO_DIR / "review_coverage.csv")
    shots_bes = sorted(int(s) for s in cover.shot.unique())
    raw_elmo_cover = {
        s: methods.cover_frame(
            *labels.merge_intervals(
                *(g.sort_values("t_start_ms")[c] for c in ("t_start_ms", "t_end_ms")),
                tol=COVER_BOUNDARY_TOL_MS,
            )
        )
        for s, g in cover.groupby("shot")
    }
    elmo_cover = {
        s: methods.cover_frame(
            *methods.intersect(
                c.t_start_ms,
                c.t_end_ms,
                input_cover[s].t_start_ms,
                input_cover[s].t_end_ms,
            )
        )
        for s, c in raw_elmo_cover.items()
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
            "bes73",
            shots_bes,
            elmo_bins,
            {s: elmo_cover[s] for s in shots_bes},
            True,
            raw_elmo_cover,
        ),
        "all119": SetDef(
            "all119",
            all_shots,
            {s: data[s].bins for s in all_shots},
            input_cover,
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
    """Limited-input DSM refit: rows, survival risk, detection-objective scores."""

    rows: dict[int, dsm.Rows]
    risk: dict[int, np.ndarray]  # (240, 4): published risk at 5, 10, 20, 50 ms
    scores: dict[str, dict[int, np.ndarray]] = field(default_factory=dict)
    threshold: dict[str, dict[int, float]] = field(default_factory=dict)
    detection_rows: dict[int, dsm.Rows] = field(default_factory=dict)

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
        (directory / "row_cache_manifest.json").write_text(
            json.dumps(
                {
                    "display_names": DISPLAY_NAME,
                    "rows_sha256": {
                        str(s): dsm.rows_digest(r)
                        for s, r in self.rows.items()
                        if r is not None
                    },
                },
                indent=1,
            )
        )

    @classmethod
    def load(
        cls,
        directory: Path,
        rows: dict[int, dsm.Rows],
        variants=(),
        detection_rows: dict[int, dsm.Rows] | None = None,
    ):
        manifest = directory / "row_cache_manifest.json"
        actual = {str(s): dsm.rows_digest(r) for s, r in rows.items() if r is not None}
        saved = (
            json.loads(manifest.read_text())["rows_sha256"] if manifest.exists() else {}
        )
        if actual != saved:
            raise ValueError(
                "DSM saved scores were fitted with different or unrecorded rows; "
                "rerun elm_dsm_evaluate.py without --rescore"
            )
        with np.load(directory / "published_risk.npz") as z:
            risk = {int(k[1:]): z[k] for k in z.files}
        scores = {}
        for name in variants:
            with np.load(directory / f"scores_{name}.npz") as z:
                scores[name] = {int(k[1:]): z[k] for k in z.files}
        if NAME["detect"] in scores:
            if detection_rows is None:
                detection_rows = {
                    s: dsm.load_rows(s, directory / "repaired_raw_rows" / f"{s}.npz")
                    for s in scores[NAME["detect"]]
                }
            if any(detection_rows.get(s) is None for s in scores[NAME["detect"]]):
                raise ValueError(
                    "saved repaired detection rows are required to audit coverage"
                )
        thr = json.loads((directory / "thresholds.json").read_text())
        return cls(
            rows,
            risk,
            scores,
            {n: {int(s): t for s, t in v.items()} for n, v in thr.items()},
            detection_rows or {},
        )


def covered_bin_mask(bins: labels.Bins, cover: pd.DataFrame) -> np.ndarray:
    """Require all 50 ms of a bin to lie in analysed time; gaps stay unknown."""
    lo, hi = labels.merge_intervals(
        cover.t_start_ms, cover.t_end_ms, tol=COVER_BOUNDARY_TOL_MS
    )
    if not len(lo):
        return np.zeros(len(bins.t0), bool)
    index = np.searchsorted(lo, bins.t0 + COVER_BOUNDARY_TOL_MS, side="right") - 1
    return (index >= 0) & (
        hi[np.maximum(index, 0)] + COVER_BOUNDARY_TOL_MS >= bins.t0 + labels.BIN_MS
    )


def bin_support(shot, bins, cover, trace, dscores, *, include_detection=True):
    """Per-method support before scoring, including repaired detection inputs.

    A saved number on an unusable row is unsupported. A missing analysed sample
    cannot be inferred to mean that a span detector called the bin negative.
    """
    first = (bins.t0 - inputs.GRID0_MS).astype(int)
    width = int(labels.BIN_MS)
    trace_ok = (first >= 0) & (first + width <= len(trace))
    bad = np.r_[0, np.cumsum(~np.isfinite(trace))]
    trace_ok[trace_ok] &= bad[first[trace_ok] + width] - bad[first[trace_ok]] == 0
    masks = {
        "analysed_time": covered_bin_mask(bins, cover),
        NAME["ours"]: trace_ok,
    }

    def row_mask(values, usable, lag):
        index = dsm.row_index(bins, lag)
        ok = (index >= 0) & (index < len(values))
        ok[ok] &= usable[index[ok]] & np.isfinite(values[index[ok]])
        return ok

    rows = dscores.rows[shot]
    masks[NAME["dsm"]] = row_mask(
        dscores.risk[shot][:, -1], rows.usable, FORECAST_LAG_ROWS
    )
    if include_detection:
        for name, by_shot in dscores.scores.items():
            if name == NAME["detect"] and shot not in dscores.detection_rows:
                masks[name] = np.zeros(len(bins.t0), bool)
                continue
            usable = (
                dscores.detection_rows[shot].usable
                if name == NAME["detect"]
                else rows.usable
            )
            masks[name] = row_mask(by_shot[shot], usable, 0)
    return masks


def shot_parts(
    shot: int,
    spans: pd.DataFrame,
    bins0: labels.Bins,
    cover0: pd.DataFrame,
    oof: methods.Oof,
    dscores: DsmScores,
    elmo_spans: dict | None,
    clock_spans: dict,
    elmo_cover: pd.DataFrame | None = None,
):
    """Every method's `ShotScore` on the shot's common bins, and those bins."""
    r = dscores.rows[shot]
    trace = oof.trace(shot)[0]
    masks = bin_support(shot, bins0, cover0, trace, dscores)
    if elmo_cover is not None:
        masks[NAME["elmo"]] = covered_bin_mask(bins0, elmo_cover)
    keep = np.logical_and.reduce(list(masks.values()))
    bins = methods.restrict_bins(bins0, keep)
    cover = dsm.usable_cover(r.usable, cover0)
    if NAME["detect"] in dscores.scores and shot in dscores.detection_rows:
        cover = dsm.usable_cover(dscores.detection_rows[shot].usable, cover)
    out = {
        NAME["ours"]: methods.trace_part(
            spans,
            shot,
            bins,
            cover,
            np.nan_to_num(trace, nan=0.0, posinf=0.0, neginf=0.0),
            oof.threshold[shot],
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
    for key in ("detect", "exposed", "init"):
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
    elmo_sweep: pd.DataFrame | None = None,
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
            sdef.elmo_cover.get(shot),
        )
        if sdef.has_elmo and elmo_sweep is not None:
            one[NAME["elmo"]].score = swap.sweep_bin_scores(
                elmo_sweep[elmo_sweep.shot == shot], bins
            )
        for name, part in one.items():
            parts.setdefault(name, []).append(part)
        bins_of[shot] = bins
    return parts, bins_of


def load_elmo_sweep(paths: Paths) -> pd.DataFrame:
    """Saved nested eta detections, for identical-bin rank metrics."""
    return pd.read_csv(paths.root / ELMO_DIR / "review_sweep.csv.gz")
