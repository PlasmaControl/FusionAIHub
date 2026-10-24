"""The detachment review source: the round-4 vote, divertor Thomson Te, and their merge.

Suggestion tables (`suggestions.table_path`), each on the producer's 50 ms bins
(`core.BIN_MS`; bin starts are whole multiples of 50 ms, so every edge sits on the
page's 10 ms grid). Categories are the review page's: 1 attached, 2 detached, 3 marfe
(here only ever a candidate), 4 uncertain; 0 is time no rule assessed, which the page
leaves unmarked. Confidence is blank: neither rule has one. Te fills only this review
source; the curated label set (`extend_detach_vote`) uses no Te.

* `detach_vote` v1: the round-4 vote rows
  (`<events>/detachment/extend_detach_vote/detach_shots.csv`, written by
  `scripts/labeler/detach_label.py`), each row's tier mapped by `vote_state`: the two
  state tiers (`certain`, TangTV + Afrac agreement, and `tangtv_only`) keep their
  attached or detached state, `candidate_marfe` is 3 and every other tier
  (`conflict`, `tangtv_only_lmode`, `insufficient_support`, `no_vote`,
  `lower_shelf_window`, `geometry_unknown`, `elm_unknown`) is 4.
* `detach_te` v1: the processed divertor Thomson Te (`\\ELECTRONS::TSTE_DIV`) of each
  bin, by the round-4 check's own selection (`scripts/labeler/detach_te_check.py`):
  the chords 1.5 to 5 cm above the shelf (Z = -1.25 m), each sample kept only while
  its chord lies in the near SOL, 1.000 < psiN <= 1.05 (the strike sits on the
  chords), and only inside 0.1-50 eV; a bin's Te is the median of its kept samples.
  Te <= 5 eV is detached, >= 10 eV attached, between them uncertain (Eldon 2017's
  bands, fixed before any scoring). psiN is read from EFIT02 where a slice lies within
  40 ms (`signals.flux_at_positions`), else from EFIT01. Every bin of the record.
* `detach_te` v2: v1's rule on the bins that lie wholly inside the shot's labeler
  plasma window (`plasma_windows`: `frames.targets.window_for`, the catalog's rule-4
  window, else the Ip window, as the other sources take it) and rest on at least
  `MIN_SAMPLES` kept samples (chord x laser time, the table's `n`).
* `detach_vote` v2: v1 wherever the vote assessed the bin (any of its categories),
  `detach_te` v1 on the bins it did not assess.
* `detach_vote` v3: the vote first, but `detach_te` v2 on the bins the vote did not
  assess and on its category-4 bins whose tier is an abstention (`ABSTAIN_TIERS`:
  `no_vote`, `insufficient_support`); `conflict`, `tangtv_only_lmode`,
  `lower_shelf_window` and the rest stay as voted. The vote is read from the written
  `detach_vote` v1 table (`vote_from_table`: its categories, with the tiers of the
  extend table it was made from, once that table's sha256 matches v1's record).

`agreement` compares the vote with Te on the bins both assessed. Run as

    pixi run --frozen -e labeler python -m labeler.events.detachment.source vote
    pixi run --frozen -e labeler python -m labeler.events.detachment.source te
    pixi run --frozen -e labeler python -m labeler.events.detachment.source merge

each with `--pointer` to move `review/source.json` to the table it writes
(`labels.write_pointer`) and `--dry-run` to print the coverage without writing. `te`
and `merge` write the current rules (`detach_te` v2, `detach_vote` v3); the earlier
versions are `te_bins(..., windows=None, min_samples=1)` and `merge(vote, te)`. A
table that exists is never rewritten: a changed rule or input is a new version.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ...config import Paths, events_ref, git_sha, sha256_of
from .. import suggestions
from ..review import labels
from . import core, signals
from . import thresholds as th

EVENT = "detachment"
BIN_MS = core.BIN_MS
GRID_MS = 10
VOTE, TE = "detach_vote", "detach_te"
CATEGORIES = {
    "0": "unmarked (not assessed)",
    "1": "attached",
    "2": "detached",
    "3": "marfe (candidate)",
    "4": "uncertain",
}
#: The vote tiers that carry a state; the rest are uncertain or a MARFE candidate.
STATE_TIERS = ("certain", "tangtv_only")
MARFE_TIER = "candidate_marfe"
#: The uncertain tiers where the vote abstained (too little support, no vote); Te
#: fills them in `detach_vote` v3. Other uncertain tiers are findings and stay.
ABSTAIN_TIERS = ("no_vote", "insufficient_support")
#: `detach_te` v2: a bin's Te needs this many kept samples (chord x laser time).
MIN_SAMPLES = 2

#: The round-4 Te check's chord selection and bands (`detach_te_check.py`).
CHORD_ABOVE_SHELF_M = (0.015, 0.05)
PSI_N_SOL = (1.0, 1.05)
TE_VALID_EV = (0.1, 50.0)
TE_DETACHED_MAX_EV = 5.0
TE_ATTACHED_MIN_EV = 10.0
EFIT_GAP_MS = 40.0


def event_dir(paths: Paths) -> Path:
    """The detachment folder of the label tables (`<events>/detachment`)."""
    return paths.label_tables / EVENT


def vote_table(paths: Paths) -> Path:
    """The round-4 vote rows `scripts/labeler/detach_label.py` writes."""
    return event_dir(paths) / "extend_detach_vote" / "detach_shots.csv"


def data_root() -> Path:
    """Where the round-5 fetch parks divertor Thomson and EFIT01 maps."""
    return Paths.from_env().root / "round5" / "detach"


# --- detach_vote v1 -----------------------------------------------------------------


def vote_state(category: int, tier: str) -> int:
    """A vote row's page category from its tier (module docstring)."""
    if tier in STATE_TIERS:
        if category not in (core.ATTACHED, core.DETACHED):
            raise ValueError(f"tier {tier!r} with category {category}")
        return int(category)
    return core.MARFE if tier == MARFE_TIER else core.UNCERTAIN


def vote_bins(path=None) -> pd.DataFrame:
    """One row per 50 ms bin the vote assessed: shot, start, category, tier.

    `path` defaults to `vote_table` of the environment's label tables."""
    path = vote_table(Paths.from_env()) if path is None else path
    table = pd.read_csv(path)
    out = []
    for row in table.itertuples():
        tier = json.loads(row.attrs)["tier"]
        length = row.t_end - row.t_start
        if length <= 0 or length % BIN_MS:
            raise ValueError(
                f"{path}: shot {row.shot} row {row.t_start}-{row.t_end} ms is not "
                f"whole {BIN_MS:g} ms bins"
            )
        starts = np.arange(row.t_start, row.t_end - 1e-6, BIN_MS)
        state = vote_state(int(row.category), tier)
        out += [(int(row.shot), float(s), state, tier) for s in starts]
    bins = pd.DataFrame(out, columns=["shot", "start", "category", "tier"])
    if bins.duplicated(["shot", "start"]).any():
        raise ValueError(f"{path}: overlapping vote rows")
    if (bins.start % GRID_MS).any():
        raise ValueError(f"{path}: a bin edge off the {GRID_MS} ms grid")
    return bins


def bin_rows(bins: pd.DataFrame) -> list[list]:
    """Table rows from bins: each shot's window runs from its first assessed bin to
    its last, runs of one category merge, and the bins between are 0 (unmarked)."""
    rows = []
    for shot, group in bins.groupby("shot", sort=True):
        window = (group.start.min(), group.start.max() + BIN_MS)
        spans = [(s, s + BIN_MS, c) for s, c in zip(group.start, group.category)]
        rows += suggestions.span_rows(int(shot), window, spans)
    return rows


def coverage(bins: pd.DataFrame) -> dict:
    """Bins and shots per category, and in total."""
    out = {"bins": len(bins), "shots": int(bins.shot.nunique())}
    for code, name in CATEGORIES.items():
        if code == "0":
            continue
        have = bins[bins.category == int(code)]
        out[name] = {"bins": len(have), "shots": int(have.shot.nunique())}
    return out


# --- detach_te v1 -------------------------------------------------------------------


def dts_record(shot: int, roots) -> tuple[dict | None, str]:
    """The parked divertor Thomson record of `shot` from the first root holding one,
    and "ok"; else None and why: `no_record` (nothing parked), `no_dts_node` (parked
    with the node TreeNODATA) or `dts_error` (parked with another error)."""
    for root in roots:
        path = Path(root) / f"{int(shot)}.npz"
        if not path.is_file():
            continue
        with np.load(path) as f:
            if "te" in f.files and "t_ms" in f.files:
                return {k: f[k] for k in f.files if k != "status"}, "ok"
            status = json.loads(str(f["status"])) if "status" in f.files else {}
        missing = [str(status.get(k, "")) for k in ("te", "t_ms") if k not in f.files]
        nodata = all("TreeNODATA" in s for s in missing)
        return None, "no_dts_node" if nodata else "dts_error"
    return None, "no_record"


def shelf_selection(root: Path | None = None) -> dict[int, str]:
    """Per shot the round-5 fetch selection (`detach_fetch_te.py --select`):
    `selected`, `no_geometry` or `off_shelf`; {} where it was never run."""
    root = data_root() if root is None else Path(root)
    chosen, reasons = root / "shelf_shots.txt", root / "shelf_shots.json"
    if not (chosen.is_file() and reasons.is_file()):
        return {}
    picked = {int(s) for s in chosen.read_text().split()}
    out = {}
    for shot, found in json.loads(reasons.read_text())["reasons"].items():
        out[int(shot)] = (
            "selected"
            if int(shot) in picked
            else "no_geometry"
            if found == "no_geometry"
            else "off_shelf"
        )
    return out


def plasma_windows(paths: Paths, shots) -> tuple[dict, dict]:
    """Each shot's labeler plasma window `(lo, hi)` in ms (`frames.targets.window_for`
    with the catalog read once, as `elm_jump` and `rwm_onsets` take it), and where it
    came from ("catalog" or "ip"); a shot with none is left out of both."""
    from ...frames.targets import catalog_windows, window_for  # it imports spans
    from ..verify import NoDataError

    catalog = catalog_windows(paths)
    got, origin = {}, {}
    for shot in shots:
        try:
            got[int(shot)], origin[int(shot)] = window_for(
                int(shot), paths, None, catalog=catalog
            )
        except (ValueError, NoDataError):
            continue
    return got, origin


def load_maps(shot: int) -> list[tuple[str, dict]]:
    """The shot's EFIT flux maps, EFIT02 first: the round-5 EFIT02 park
    (`$LABELER_ROOT/mhd/efit02`), the round-4 detachment maps, the round-5 EFIT01."""
    root = Paths.from_env().root
    found = []
    path = root / "mhd" / "efit02" / f"{int(shot)}.npz"
    if path.is_file():
        with np.load(path) as f:
            if "psirz" in f.files and len(f["times"]) >= 2:
                found.append(
                    (
                        "EFIT02",
                        {
                            "gtime_ms": f["times"].astype(float),
                            **{
                                k: f[k] for k in ("psirz", "r", "z", "ssimag", "ssibry")
                            },
                        },
                    )
                )
    maps = signals.load_flux_map(shot)
    if maps is not None:
        found.append((str(maps.get("source", "EFIT01")), maps))
    path = data_root() / "efit01" / f"{int(shot)}.npz"
    if path.is_file():
        with np.load(path) as f:
            if len(f["gtime_ms"]) >= 2:
                found.append(("EFIT01", {k: f[k] for k in f.files}))
    return found


def chord_flux(maps: list[tuple[str, dict]], t_ms, positions):
    """psiN of each chord at each time from the first map with a slice within
    `EFIT_GAP_MS`, and the index of that map per time (-1 where none has one)."""
    t_ms = np.asarray(t_ms, float)
    psi = np.full((len(positions), len(t_ms)), np.nan)
    used = np.full(len(t_ms), -1)
    for k, (_, one) in enumerate(maps):
        todo = used < 0
        if not todo.any():
            break
        got = signals.flux_at_positions(one, t_ms[todo], positions, EFIT_GAP_MS)
        have = np.isfinite(got).any(axis=0)
        index = np.flatnonzero(todo)[have]
        psi[:, index], used[index] = got[:, have], k
    return psi, used


def bin_te(dts: dict, maps, starts) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per bin of `starts`: the median gated chord Te (eV, NaN without one), the
    number of samples behind it and the index in `maps` of its flux source (-1)."""
    starts = np.asarray(starts, float)
    nan = np.full(len(starts), np.nan)
    none = np.zeros(len(starts), int), np.full(len(starts), -1)
    above = np.asarray(dts["z"], float) - th.SHELF_Z
    chords = np.flatnonzero(
        (above >= CHORD_ABOVE_SHELF_M[0]) & (above <= CHORD_ABOVE_SHELF_M[1])
    )
    if not len(chords) or not len(starts):
        return nan, *none
    t = np.asarray(dts["t_ms"], float)
    r, z = np.asarray(dts["r"], float), np.asarray(dts["z"], float)
    psi, used = chord_flux(maps, t, np.stack([r[chords], z[chords]], axis=1))
    te = np.asarray(dts["te"], float)[chords]
    ok = (
        np.isfinite(te)
        & (te >= TE_VALID_EV[0])
        & (te <= TE_VALID_EV[1])
        & (psi > PSI_N_SOL[0])
        & (psi <= PSI_N_SOL[1])
    )
    te = np.where(ok, te, np.nan)
    index = np.floor((t - starts[0]) / BIN_MS).astype(int)
    keep = (index >= 0) & (index < len(starts))
    value, count, source = nan.copy(), *none
    for b in np.unique(index[keep]):
        inside = keep & (index == b)
        samples = te[:, inside]
        if np.isfinite(samples).any():
            value[b] = np.nanmedian(samples)
            count[b] = int(np.isfinite(samples).sum())
            source[b] = int(used[inside][np.isfinite(samples).any(axis=0)].min())
    return value, count, source


def te_state(te) -> np.ndarray:
    """2 at or below 5 eV, 1 at or above 10 eV, 4 between, 0 without a Te."""
    te = np.asarray(te, float)
    state = np.where(te <= TE_DETACHED_MAX_EV, core.DETACHED, core.UNCERTAIN)
    state = np.where(te >= TE_ATTACHED_MIN_EV, core.ATTACHED, state)
    return np.where(np.isfinite(te), state, core.ABSENT).astype(int)


def te_bins(
    shots, dts_roots, *, windows=None, min_samples: int = 1, selection=None
) -> tuple[pd.DataFrame, dict, dict]:
    """Te bins of `shots` (shot, start, category, te_ev, n, efit); per shot not in
    them, why; and how many bins with a Te each gate dropped.

    `windows` (shot -> `(lo, hi)` ms; None: no gate) keeps the bins wholly inside the
    shot's window, and a shot it lacks is `no_window`. A bin needs `min_samples` kept
    samples. A shot with nothing parked is `not_fetched` when `selection`
    (`shelf_selection`) chose it, else `not_selected:<reason>` (`no_record` without a
    selection); a parked record is `no_dts_node` (TreeNODATA), `dts_error` or, with no
    finite time, `no_dts_time`. Then `no_efit`, `strike_off_chords` (no kept sample),
    `outside_window` and `too_few_samples` (no bin passes that gate)."""
    frames, skipped = [], {}
    dropped = Counter(bins_with_te=0, outside_window=0, too_few_samples=0)
    for shot in sorted(int(s) for s in shots):
        dts, why = dts_record(shot, dts_roots)
        if dts is None:
            if why == "no_record" and selection is not None:
                chosen = selection.get(shot, "unknown")
                why = (
                    "not_fetched" if chosen == "selected" else f"not_selected:{chosen}"
                )
            skipped[str(shot)] = why
            continue
        t = np.asarray(dts["t_ms"], float)
        t = t[np.isfinite(t)]
        if not len(t):
            skipped[str(shot)] = "no_dts_time"
            continue
        if windows is not None and shot not in windows:
            skipped[str(shot)] = "no_window"
            continue
        maps = load_maps(shot)
        if not maps:
            skipped[str(shot)] = "no_efit"
            continue
        lo = BIN_MS * np.floor(max(t.min(), 0.0) / BIN_MS)
        starts = np.arange(lo, t.max() + BIN_MS, BIN_MS)
        value, count, source = bin_te(dts, maps, starts)
        have = np.isfinite(value)
        dropped["bins_with_te"] += int(have.sum())
        if not have.any():
            skipped[str(shot)] = "strike_off_chords"
            continue
        inside = np.ones(len(starts), bool)
        if windows is not None:
            w_lo, w_hi = windows[shot]
            inside = (starts >= w_lo) & (starts + BIN_MS <= w_hi)
        enough = count >= min_samples
        dropped["outside_window"] += int((have & ~inside).sum())
        dropped["too_few_samples"] += int((have & inside & ~enough).sum())
        keep = have & inside & enough
        if not keep.any():
            reason = (
                "outside_window" if not (have & inside).any() else "too_few_samples"
            )
            skipped[str(shot)] = reason
            continue
        names = np.array([name for name, _ in maps])
        frames.append(
            pd.DataFrame(
                {
                    "shot": shot,
                    "start": starts[keep],
                    "category": te_state(value[keep]),
                    "te_ev": value[keep],
                    "n": count[keep],
                    "efit": names[source[keep]],
                }
            )
        )
    columns = ["shot", "start", "category", "te_ev", "n", "efit"]
    bins = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=columns)
    )
    return bins, skipped, dict(dropped)


# --- detach_vote v2, v3 and the agreement -------------------------------------------


def merge(vote: pd.DataFrame, te: pd.DataFrame, fill_tiers=()) -> pd.DataFrame:
    """The vote's bins, except those whose tier is in `fill_tiers` where Te has a
    bin, and Te's on every bin the vote left; `from` says which, `tier` is the vote's
    tier where it assessed the bin ("" elsewhere)."""
    key = ["shot", "start"]
    tiers = vote["tier"] if "tier" in vote else pd.Series("", index=vote.index)
    te_keys = pd.MultiIndex.from_frame(te[key])
    replaced = tiers.isin(tuple(fill_tiers)).to_numpy() & pd.MultiIndex.from_frame(
        vote[key]
    ).isin(te_keys)
    kept = vote[~replaced]
    fill = te[~te_keys.isin(pd.MultiIndex.from_frame(kept[key]))]
    tier_of = dict(zip(zip(vote.shot, vote.start), tiers))
    return pd.concat(
        [
            kept[[*key, "category"]].assign(**{"from": VOTE, "tier": tiers[~replaced]}),
            fill[[*key, "category"]].assign(
                **{
                    "from": TE,
                    "tier": [tier_of.get(k, "") for k in zip(fill.shot, fill.start)],
                }
            ),
        ],
        ignore_index=True,
    ).sort_values(key, ignore_index=True)


def filled(merged: pd.DataFrame) -> dict:
    """Per vote tier, the bins Te filled in place of the vote, by Te's category."""
    names = {int(k): v for k, v in CATEGORIES.items()}
    rows = merged[(merged["from"] == TE) & (merged["tier"] != "")]
    return {
        tier: {names[c]: int(n) for c, n in group.category.value_counts().items()}
        for tier, group in rows.groupby("tier")
    }


def kappa(a, b) -> float | None:
    """Cohen's kappa of two raters' labels on the same items (None if undefined)."""
    a, b = np.asarray(a), np.asarray(b)
    if not len(a):
        return None
    classes = np.union1d(a, b)
    observed = np.mean(a == b)
    expected = sum(np.mean(a == c) * np.mean(b == c) for c in classes)
    return None if expected >= 1 else float((observed - expected) / (1 - expected))


def agreement(vote: pd.DataFrame, te: pd.DataFrame) -> dict:
    """The vote against Te on the bins both assessed: the full category table, and
    agreement and kappa where both name a state (attached or detached), pooled, by
    vote tier and per shot."""
    both = vote.merge(te, on=["shot", "start"], suffixes=("_vote", "_te"))
    names = {int(k): v for k, v in CATEGORIES.items()}
    table = {
        names[v]: {names[t]: int(n) for t, n in rows.category_te.value_counts().items()}
        for v, rows in both.groupby("category_vote")
    }
    states = both[
        both.category_vote.isin((core.ATTACHED, core.DETACHED))
        & both.category_te.isin((core.ATTACHED, core.DETACHED))
    ]

    def summary(rows) -> dict:
        return {
            "bins": len(rows),
            "shots": int(rows.shot.nunique()),
            "agree": int((rows.category_vote == rows.category_te).sum()),
            "agreement": (
                float((rows.category_vote == rows.category_te).mean())
                if len(rows)
                else None
            ),
            "kappa": kappa(rows.category_vote, rows.category_te),
        }

    uncertain = both[both.category_vote == core.UNCERTAIN]
    return {
        "population": "bins the vote assessed (any category) that carry a Te",
        "bins": len(both),
        "shots": int(both.shot.nunique()),
        "vote_by_te": table,
        "states": {
            **summary(states),
            "by_tier": {tier: summary(rows) for tier, rows in states.groupby("tier")},
            "per_shot": {
                str(int(shot)): summary(rows) for shot, rows in states.groupby("shot")
            },
        },
        "vote_uncertain_with_te": {
            "bins": len(uncertain),
            "bins_with_te_state": int(
                uncertain.category_te.isin((core.ATTACHED, core.DETACHED)).sum()
            ),
            "by_tier": {
                tier: {
                    names[c]: int(n) for c, n in rows.category_te.value_counts().items()
                }
                for tier, rows in uncertain.groupby("tier")
            },
        },
    }


# --- writing ------------------------------------------------------------------------


def silver_gold_shots(paths: Paths) -> list[int]:
    from ..review import gold, silver

    shots = set(silver.read_silver(paths.label_tables).shot.astype(int))
    shots |= set(gold.read_gold(paths.label_tables).shot.astype(int))
    return sorted(shots)


def blind(paths: Paths) -> set[int]:
    """The catalog cohort's blind shots, which no table may hold."""
    from ..review.silver import blind_shots

    return blind_shots(paths.label_tables)


def write(
    paths: Paths,
    method: str,
    version: str,
    bins: pd.DataFrame,
    meta: dict,
    *,
    pointer: bool,
) -> dict:
    """Write one table (never over an existing one, never with a blind shot) and,
    with `pointer`, point the review page at it."""
    path = suggestions.table_path(paths, EVENT, method, version)
    if path.exists():
        raise FileExistsError(
            f"{path} exists: a changed rule or input is a new version"
        )
    leaked = sorted(set(bins.shot.astype(int)) & blind(paths))
    if leaked:
        raise ValueError(f"{len(leaked)} blind cohort shots in the {method} bins")
    meta = {
        "event": EVENT,
        "method": method,
        "version": version,
        "git_sha": git_sha(),
        "made_by": "labeler.events.detachment.source",
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "categories": CATEGORIES,
        "bin_ms": BIN_MS,
        "grid_ms": GRID_MS,
        "confidence": "blank: the rules have none",
        **meta,
        "coverage": coverage(bins),
    }
    frame = suggestions.write_table(path, bin_rows(bins), meta)
    out = {"table": str(path), "rows": len(frame), "coverage": meta["coverage"]}
    if pointer:
        out["pointer"] = labels.write_pointer(
            event_dir(paths), path, method=method, version=version
        )
    return out


def vote_meta(bins: pd.DataFrame, path=None) -> dict:
    path = vote_table(Paths.from_env()) if path is None else Path(path)
    return {
        "rule": {
            "input": "round-4 vote rows: the geometry-gated TangTV state with Afrac "
            "as the second vote (README of <events>/detachment)",
            "states": f"tiers {', '.join(STATE_TIERS)} keep their state (1 or 2)",
            "marfe": f"tier {MARFE_TIER} is 3, a candidate, never a MARFE state",
            "uncertain": "every other tier is 4",
            "unmarked": "time the vote did not assess is 0",
        },
        "input": {
            "table": events_ref(path),
            "sha256": sha256_of(path),
        },
        "bins_by_tier": {k: int(v) for k, v in Counter(bins.tier).most_common()},
    }


def te_meta(
    bins: pd.DataFrame, shots, skipped: dict, dropped: dict, windows_from: dict
) -> dict:
    return {
        "rule": {
            "te": "processed divertor Thomson \\ELECTRONS::TSTE_DIV, the chords "
            f"{CHORD_ABOVE_SHELF_M[0] * 100:g}-{CHORD_ABOVE_SHELF_M[1] * 100:g} cm "
            f"above the shelf (Z = {th.SHELF_Z} m), a sample kept while its chord "
            f"lies at {PSI_N_SOL[0]:.3f} < psiN <= {PSI_N_SOL[1]:.2f} and Te is "
            f"{TE_VALID_EV[0]:g}-{TE_VALID_EV[1]:g} eV; a bin's Te is the median "
            "of its kept samples",
            "states": f"Te <= {TE_DETACHED_MAX_EV:g} eV detached (2), >= "
            f"{TE_ATTACHED_MIN_EV:g} eV attached (1), between uncertain (4)",
            "flux": f"EFIT02 where a slice lies within {EFIT_GAP_MS:g} ms, else EFIT01",
            "window": "only bins wholly inside the shot's labeler plasma window "
            "(frames.targets.window_for: the catalog's rule-4 window, else the Ip "
            "window)",
            "min_samples": f"only bins with at least {MIN_SAMPLES} kept samples "
            "(chord x laser time)",
            "unmarked": "bins with no kept sample, outside the window or under the "
            "sample minimum are 0",
            "source": "scripts/labeler/detach_te_check.py (round-4 check)",
        },
        "shots_requested": len(list(shots)),
        "skipped": dict(sorted(skipped.items(), key=lambda kv: int(kv[0]))),
        "skipped_counts": dict(Counter(skipped.values()).most_common()),
        "bins_dropped": dropped,
        "windows_from": dict(Counter(windows_from.values()).most_common()),
        "bins_by_efit": {k: int(v) for k, v in Counter(bins.efit).most_common()},
        "bins_by_samples": {
            str(k): int(v) for k, v in sorted(Counter(bins.n.astype(int)).items())
        },
        "te_ev_quantiles_10_50_90": (
            [float(v) for v in np.percentile(bins.te_ev, [10, 50, 90])]
            if len(bins)
            else None
        ),
    }


def table_bins(path) -> pd.DataFrame:
    """The assessed bins (category 1-4) of a written table: shot, start, category."""
    frame = pd.read_csv(path)
    out = []
    for row in frame[frame.category != 0].itertuples():
        for s in np.arange(row.t_start, row.t_end - 1e-6, BIN_MS):
            out.append((int(row.shot), float(s), int(row.category)))
    return pd.DataFrame(out, columns=["shot", "start", "category"])


def vote_from_table(
    paths: Paths, version: str = "v1", extend=None
) -> tuple[pd.DataFrame, dict]:
    """The vote's bins with their tiers, read from the written `detach_vote`
    `version` table: its categories, with the tiers of the extend table it was made
    from. Refused unless that table's sha256 is the one `version` recorded and its
    bins give exactly the written table's. `extend` defaults to `vote_table(paths)`."""
    extend = vote_table(paths) if extend is None else extend
    path = suggestions.table_path(paths, EVENT, VOTE, version)
    recorded = json.loads(path.with_suffix(".meta.json").read_text())["input"]
    if sha256_of(extend) != recorded["sha256"]:
        raise ValueError(
            f"{extend} changed since {VOTE} {version} was written from it "
            f"(sha256 {recorded['sha256'][:12]}...): write a new vote version first"
        )
    written = table_bins(path)
    bins = vote_bins(extend)
    key = ["shot", "start"]
    left = bins[[*key, "category"]].sort_values(key, ignore_index=True)
    right = written.sort_values(key, ignore_index=True)
    if len(left) != len(right) or not (left.to_numpy() == right.to_numpy()).all():
        raise ValueError(f"{path}: its bins differ from {extend}'s vote")
    return bins, {
        "table": str(path),
        "sha256": sha256_of(path),
        "tiers_from": recorded,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("step", choices=("vote", "te", "merge"))
    parser.add_argument("--version", default=None, help="default v1, v2 (te), v3")
    parser.add_argument("--te-version", default="v2", help="merge: the Te table")
    parser.add_argument("--vote-version", default="v1", help="merge: the vote table")
    parser.add_argument("--shots-file", type=Path, help="te: shots (silver + gold)")
    parser.add_argument("--pointer", action="store_true", help="move source.json")
    parser.add_argument("--dry-run", action="store_true", help="print, write nothing")
    parser.add_argument("--record", type=Path, help="merge: docs JSON to write")
    args = parser.parse_args(argv)
    paths = Paths.from_env()
    dts_roots = (
        data_root() / "dts",
        paths.root / "round4" / "detach" / "dts",
    )
    if args.step == "vote":
        bins = vote_bins(vote_table(paths))
        meta, version, method = vote_meta(bins), args.version or "v1", VOTE
    elif args.step == "te":
        shots = (
            [int(s) for s in args.shots_file.read_text().split()]
            if args.shots_file
            else silver_gold_shots(paths)
        )
        shots = sorted(set(shots) - blind(paths))
        windows, windows_from = plasma_windows(paths, shots)
        bins, skipped, dropped = te_bins(
            shots,
            dts_roots,
            windows=windows,
            min_samples=MIN_SAMPLES,
            selection=shelf_selection(),
        )
        meta = te_meta(bins, shots, skipped, dropped, windows_from)
        version, method = args.version or "v2", TE
    else:
        vote, vote_input = vote_from_table(paths, args.vote_version)
        te_path = suggestions.table_path(paths, EVENT, TE, args.te_version)
        te = table_bins(te_path)
        bins = merge(vote, te, ABSTAIN_TIERS)
        meta = {
            "rule": {
                "vote": f"{VOTE} {args.vote_version} first, in any category",
                "te": f"{TE} {args.te_version} on the bins the vote did not assess",
                "abstentions": f"{TE} {args.te_version} also replaces the vote's "
                f"category-4 bins of tier {' or '.join(ABSTAIN_TIERS)} where it has "
                "a bin; every other tier stays as voted",
            },
            "inputs": {
                VOTE: vote_input,
                TE: {"table": str(te_path), "sha256": sha256_of(te_path)},
            },
            "bins_from": {k: int(v) for k, v in Counter(bins["from"]).items()},
            "abstentions_filled": filled(bins),
            "agreement_vote_vs_te": agreement(vote, te),
        }
        version, method = args.version or "v3", VOTE
    if args.dry_run:
        keys = ("bins_by_tier", "skipped_counts", "bins_dropped", "windows_from")
        keys += ("bins_by_efit", "bins_by_samples", "bins_from", "abstentions_filled")
        shown = {k: v for k, v in meta.items() if k in keys}
        if "agreement_vote_vs_te" in meta:
            shown["agreement_vote_vs_te"] = summary_of(meta["agreement_vote_vs_te"])
        print(json.dumps({"coverage": coverage(bins), **shown}, indent=1))
        return 0
    out = write(paths, method, version, bins, meta, pointer=args.pointer)
    if args.record is not None and args.step == "merge":
        record = record_of(paths, out)
        args.record.write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps(out, indent=1))
    return 0


def summary_of(record: dict) -> dict:
    """An agreement record without its per-shot rows (shot numbers stay in the
    table metas under `$LABELER_ROOT`)."""
    states = {k: v for k, v in record["states"].items() if k != "per_shot"}
    return {**record, "states": states}


def versions(paths: Paths, method: str) -> list[str]:
    """The written versions of `method`, oldest first."""
    root = paths.root / "suggestions" / method
    found = [p.name for p in root.glob("v*") if p.name[1:].isdigit()]
    return sorted(
        (v for v in found if suggestions.table_path(paths, EVENT, method, v).is_file()),
        key=lambda v: int(v[1:]),
    )


def record_of(paths: Paths, merged: dict) -> dict:
    """Every written vote and Te table's rule and coverage, and the newest merge's
    agreement, from their metas, for the docs (no shot numbers)."""

    def meta(method, version):
        path = suggestions.table_path(paths, EVENT, method, version)
        return json.loads(path.with_suffix(".meta.json").read_text())

    vote_keys = ("rule", "coverage", "bins_by_tier", "bins_from", "abstentions_filled")
    te_keys = ("rule", "coverage", "shots_requested", "skipped_counts")
    te_keys += ("bins_dropped", "windows_from", "bins_by_efit", "bins_by_samples")
    te_keys += ("te_ev_quantiles_10_50_90",)
    newest = meta(VOTE, versions(paths, VOTE)[-1])
    out = {
        "schema": "detachment_review_source",
        "made_at": newest["made_at"],
        "git_sha": newest["git_sha"],
        "categories": CATEGORIES,
        "bin_ms": BIN_MS,
    }
    for method, keys in ((VOTE, vote_keys), (TE, te_keys)):
        for version in versions(paths, method):
            one = meta(method, version)
            out[f"{method}_{version}"] = {k: one[k] for k in keys if k in one}
    out["agreement_vote_vs_te"] = {
        "vote": newest["inputs"][VOTE]["table"].rsplit("/", 1)[-1],
        "te": newest["inputs"][TE]["table"].rsplit("/", 1)[-1],
        **summary_of(newest["agreement_vote_vs_te"]),
    }
    out["pointer"] = merged.get("pointer")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
