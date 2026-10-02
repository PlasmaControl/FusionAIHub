"""Reconcile donor labels on exact half-open intervals, before temporal binning.

Time is milliseconds throughout this workflow. Agreement between supplied
files is consistency evidence, not evidence of independent annotation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from itertools import combinations, permutations
from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import h5py
import numpy as np
import pandas as pd

from ..config import Paths

MODES = ("L", "H", "QH", "WP")
COLUMNS = ["shot", "t_start", "t_end", "regime", "source", "record"]
MERGED_COLUMNS = ["shot", "t_start", "t_end", "regimes", "sources", "status",
                  "label", "source_regimes"]


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _table(raw: pd.DataFrame, source: str, path: Path):
    required = {"Shot", "Confinement Start Time (ms)",
                "Confinement Stop Time (ms)", *MODES}
    if not required.issubset(raw.columns):
        raise ValueError(f"{path}: missing columns {required - set(raw.columns)}")
    records, test_only = [], set()
    unknown, blank = 0, 0
    for i, row in enumerate(raw.to_dict("records"), 2):
        shot_value = row.get("Shot")
        if pd.isna(shot_value) or str(shot_value).strip() == "":
            blank += 1
            continue
        shot_float = float(shot_value)
        if not np.isfinite(shot_float) or shot_float % 1 or shot_float < 0:
            raise ValueError(f"{path}:{i}: invalid shot")
        shot = int(shot_float)
        flag = row.get("Test only")
        # Preserve every nonempty test-only marker, including named cohorts.
        if pd.notna(flag) and str(flag).strip() not in ("", "0", "0.0"):
            test_only.add(shot)
        flags = [0 if pd.isna(row[m]) or str(row[m]).strip() == ""
                 else float(row[m]) for m in MODES]
        if not all(f in (0, 1) for f in flags) or sum(flags) > 1:
            raise ValueError(f"{path}:{i}: expected zero or one regime flags")
        if sum(flags) == 0:
            unknown += 1
            continue
        start, stop = [float(row[f"Confinement {edge} Time (ms)"])
                       for edge in ("Start", "Stop")]
        if not np.isfinite([start, stop]).all() or start < 0 or stop <= start:
            raise ValueError(f"{path}:{i}: invalid labelled interval")
        records.append((shot, start, stop, MODES[flags.index(1)], source,
                        f"{path.name}:{i}"))
    frame = pd.DataFrame(records, columns=COLUMNS)
    return frame, test_only, {"path": str(path), "sha256": digest(path),
                             "rows": len(raw), "labelled_rows": len(frame),
                             "unlabelled_rows": unknown, "blank_rows": blank,
                             "test_only_shots": sorted(test_only)}


def read_workbook(path: Path):
    """Read the supplied XLSX without installing an Excel engine dependency."""
    ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    with ZipFile(path) as archive:
        strings = []
        if "xl/sharedStrings.xml" in archive.namelist():
            strings = ["".join(node.itertext()) for node in ET.fromstring(
                archive.read("xl/sharedStrings.xml")).findall("m:si", ns)]
        sheet = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        parsed = []
        for row in sheet.findall(".//m:sheetData/m:row", ns):
            cells = {}
            for cell in row:
                col = re.sub(r"\d+", "", cell.attrib["r"])
                value = cell.find("m:v", ns)
                text = "" if value is None else value.text
                if cell.get("t") == "s":
                    text = strings[int(text)]
                elif cell.get("t") == "inlineStr":
                    text = "".join(cell.find("m:is", ns).itertext())
                cells[col] = text
            parsed.append(cells)
    header, *rows = parsed
    raw = pd.DataFrame([{name: row.get(col, "") for col, name in header.items()}
                        for row in rows])
    return _table(raw, "kevin_workbook", path)


def read_hdf(path: Path, modes=MODES):
    """Run-length encode one-hot samples; all-zero/NaN samples remain unknown.

    A sample covers one median cadence. Gaps larger than 1.5 cadences split
    intervals; neither a gap nor an empty file asserts an absent regime.
    """
    match = re.fullmatch(r"bes_signals_(\d+)at[\d.]+", path.stem)
    if match is None or set(modes) != set(MODES) or len(modes) != 4:
        raise ValueError(f"{path}: invalid filename or class ordering")
    shot = int(match[1])
    with h5py.File(path, "r") as f:
        t = np.asarray(f["time"], dtype=np.float64)
        y = np.asarray(f["labels"], dtype=np.float64)
    if t.ndim != 1 or y.shape != (len(t), 4):
        raise ValueError(f"{path}: labels must be (len(time), 4)")
    audit = {"path": str(path), "sha256": digest(path), "shot": shot,
             "samples": len(t), "mode_order": list(modes),
             "labels_sha256": hashlib.sha256(y.tobytes()).hexdigest(),
             "time_sha256": hashlib.sha256(t.tobytes()).hexdigest()}
    if not len(t):
        return pd.DataFrame(columns=COLUMNS), {**audit, "empty": True,
                                               "unknown_samples": 0}
    if not np.isfinite(t).all() or (t < 0).any() or (np.diff(t) <= 0).any():
        raise ValueError(f"{path}: time must be finite and strictly increasing")
    if len(t) < 2:
        raise ValueError(f"{path}: a singleton has no documented cadence")
    finite = np.isfinite(y).all(axis=1)
    if not np.isin(y[finite], (0, 1)).all() or (y[finite].sum(axis=1) > 1).any():
        raise ValueError(f"{path}: expected one-hot or all-zero labels")
    known = finite & (y.sum(axis=1) == 1)
    codes = np.where(known, np.nan_to_num(y).argmax(axis=1), -1)
    dt = float(np.median(np.diff(t)))
    gaps = np.diff(t) > 1.5 * dt
    edges = np.r_[0, np.flatnonzero((np.diff(codes) != 0) | gaps) + 1, len(t)]
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if codes[lo] < 0:
            continue
        end = min(t[hi], t[hi - 1] + dt) if hi < len(t) else t[-1] + dt
        rows.append((shot, round(float(t[lo]), 6), round(float(end), 6),
                     modes[int(codes[lo])], "kevin_bes", path.name))
    return pd.DataFrame(rows, columns=COLUMNS), {
        **audit, "empty": False, "unknown_samples": int((~known).sum()),
        "cadence_ms": dt, "gaps": int(gaps.sum()),
        "sample_counts": {m: int((codes == i).sum()) for i, m in enumerate(modes)},
    }


def merge_intervals(rows: pd.DataFrame) -> pd.DataFrame:
    """Partition the union into source-consistent spans; H/L conflict is -1."""
    result = []
    for shot, group in rows.groupby("shot", sort=True):
        events = defaultdict(list)
        for i, row in enumerate(group.itertuples(index=False)):
            events[round(row.t_start, 6)].append((i, row.regime, row.source))
            events[round(row.t_end, 6)].append((i, None, None))
        active = {}
        axis = sorted(events)
        for start, stop in zip(axis[:-1], axis[1:], strict=True):
            for i, regime, source in events[start]:
                if regime is None:
                    active.pop(i, None)
                else:
                    active[i] = (regime, source)
            if not active or stop <= start:
                continue
            regimes = sorted({r for r, _ in active.values()})
            sources = sorted({s for _, s in active.values()})
            binaries = {int(r != "L") for r in regimes}
            label = binaries.pop() if len(binaries) == 1 else -1
            status = ("binary_conflict" if label < 0 else "subtype_conflict"
                      if len(regimes) > 1 else "agreement" if len(sources) > 1
                      else "single_source")
            record = [int(shot), start, stop, "|".join(regimes),
                      "|".join(sources), status, label,
                      json.dumps({s: sorted({r for r, src in active.values() if src == s})
                                  for s in sources}, sort_keys=True)]
            if result and result[-1][0] == shot and result[-1][2] == start \
                    and result[-1][3:] == record[3:]:
                result[-1][2] = stop
            else:
                result.append(record)
    return pd.DataFrame(result, columns=MERGED_COLUMNS)


def bins_from_intervals(merged: pd.DataFrame, bin_ms=50.) -> pd.DataFrame:
    """A target requires complete coverage with one unambiguous binary class."""
    result = []
    for shot, group in merged.groupby("shot", sort=True):
        duration = np.zeros((int(np.ceil(group.t_end.max() / bin_ms)), 3))
        for row in group.itertuples(index=False):
            first = int(np.floor(row.t_start / bin_ms))
            last = int(np.ceil(row.t_end / bin_ms))
            bins = np.arange(first, last)
            overlap = np.maximum(0, np.minimum((bins + 1) * bin_ms, row.t_end)
                                 - np.maximum(bins * bin_ms, row.t_start))
            duration[bins, row.label if row.label >= 0 else 2] += overlap
        labels = np.full(len(duration), -1, dtype=int)
        labels[duration[:, 0] >= bin_ms - 1e-6] = 0
        labels[duration[:, 1] >= bin_ms - 1e-6] = 1
        # Excluded bins remain inspectable, but bins before any supplied label
        # are not counted in the annotation exclusion denominator.
        for k in np.flatnonzero(duration.sum(axis=1) > 0):
            result.append((int(shot), k * bin_ms, (k + 1) * bin_ms, int(labels[k]),
                           float(duration[k].sum()), float(duration[k, 2])))
    return pd.DataFrame(result, columns=["shot", "t_start", "t_end", "label",
                                         "covered_ms", "conflict_ms"])


def balanced_weights(labels, shots) -> np.ndarray:
    """Equal total mass per H/L class and equal shot mass within each class."""
    labels, shots = np.asarray(labels), np.asarray(shots)
    if len(labels) != len(shots) or set(np.unique(labels)) != {0, 1}:
        raise ValueError("Balancing needs both known binary classes")
    weights = np.zeros(len(labels), dtype=float)
    for label in (0, 1):
        members, counts = np.unique(shots[labels == label], return_counts=True)
        for shot, count in zip(members, counts, strict=True):
            mask = (labels == label) & (shots == shot)
            weights[mask] = 0.5 / (len(members) * count)
    return weights * len(labels)


def split_shots(targets: pd.DataFrame, test_only: set[int], seed=20261001):
    """Deterministic group split, stratified by a shot's represented H/L classes."""
    rng = np.random.default_rng(seed)
    groups = defaultdict(list)
    result = []
    for shot, rows in targets.loc[targets.label >= 0].groupby("shot", sort=True):
        if int(shot) in test_only:
            result.append((int(shot), "test", "raw_test_only"))
        else:
            groups[tuple(sorted(rows.label.unique()))].append(int(shot))
    for members in groups.values():
        members = rng.permutation(members)
        n = len(members)
        n_val = max(1, round(0.15 * n)) if n >= 3 else 0
        n_test = max(1, round(0.15 * n)) if n >= 5 else 0
        for i, shot in enumerate(members):
            split = "val" if i < n_val else "test" if i < n_val + n_test else "train"
            result.append((int(shot), split, "seeded_class_presence_strata"))
    return pd.DataFrame(result, columns=["shot", "split", "reason"]).sort_values("shot")


def comparison(rows: pd.DataFrame):
    """Duration confusion matrices and shot-level denominators for source pairs."""
    output, per_shot = {}, []
    for left, right in combinations(sorted(rows.source.unique()), 2):
        pair = rows.loc[rows.source.isin([left, right])]
        merged = merge_intervals(pair)
        both = merged.loc[merged.sources == "|".join(sorted([left, right]))]
        matrix = np.zeros((4, 4))
        for shot, spans in both.groupby("shot"):
            shot_matrix = np.zeros((4, 4))
            for span in spans.itertuples(index=False):
                mapping = json.loads(span.source_regimes)
                l, r = mapping[left], mapping[right]
                if len(l) == len(r) == 1:
                    shot_matrix[MODES.index(l[0]), MODES.index(r[0])] += span.t_end - span.t_start
            matrix += shot_matrix
            binary = shot_matrix[0, 1:].sum() + shot_matrix[1:, 0].sum()
            per_shot.append({"pair": f"{left} vs {right}", "shot": int(shot),
                             "overlap_ms": float(shot_matrix.sum()),
                             "regime_conflict_ms": float(shot_matrix.sum() - np.trace(shot_matrix)),
                             "binary_conflict_ms": float(binary)})
        total = float(matrix.sum())
        binary = float(matrix[0, 1:].sum() + matrix[1:, 0].sum())
        perm_scores = []
        if "kevin_bes" in (left, right):
            for perm in permutations(range(4)):
                score = sum(matrix[i, perm[i]] for i in range(4))
                perm_scores.append((float(score), [MODES[i] for i in perm]))
        output[f"{left} vs {right}"] = {
            "row_source": left, "column_source": right, "modes": list(MODES),
            "duration_confusion_ms": matrix.tolist(), "overlap_ms": total,
            "overlap_shots": int(both.shot.nunique()), "binary_conflict_ms": binary,
            "regime_agreement": float(np.trace(matrix) / total) if total else None,
            "binary_agreement": 1 - binary / total if total else None,
            "best_mode_permutations": sorted(perm_scores, reverse=True)[:3],
        }
    return output, pd.DataFrame(per_shot)


def build(raw_root: Path, out: Path, seed=20261001):
    raw_root, out = Path(raw_root).resolve(), Path(out).resolve()
    if out == raw_root or raw_root in out.parents:
        raise ValueError("Outputs must be outside the untouched raw directory")
    jalal_path = raw_root / "Jalal_28042024_confinement_regime_shotlist.csv"
    jalal, test_only, audit = _table(pd.read_csv(jalal_path), "jalal", jalal_path)
    sources, audits = [jalal], [audit]
    workbook = raw_root / "confinement_mode_database.xlsx"
    if workbook.exists():
        frame, _, audit = read_workbook(workbook)
        sources.append(frame)
        audits.append(audit)
    clips = sorted((raw_root / "confinement_labels_time").glob("*.hdf5"))
    seen = {p.name for p in clips}
    clips += [p for p in sorted((raw_root / "confinement_data").glob("*.hdf5"))
              if p.name not in seen]
    duplicate_clips = []
    for path in clips:
        try:
            frame, audit = read_hdf(path)
        except OSError as error:
            audits.append({"path": str(path), "sha256": digest(path),
                           "readable": False, "error": str(error)})
            continue
        sources.append(frame)
        audits.append(audit)
        other = raw_root / "confinement_data" / path.name
        if path != other and other.exists():
            try:
                _, other_audit = read_hdf(other)
            except OSError as error:
                duplicate_clips.append({"name": path.name, "path": str(other),
                                        "sha256": digest(other), "readable": False,
                                        "error": str(error)})
                continue
            equal = all(audit[key] == other_audit[key]
                        for key in ("labels_sha256", "time_sha256"))
            if not equal:
                raise ValueError(f"Duplicate clip {path.name} has inconsistent labels")
            duplicate_clips.append({"name": path.name, "identical_label_intervals": equal,
                                    "readable": True, "path": str(other),
                                    "sha256": other_audit["sha256"]})
    rows = pd.concat(sources, ignore_index=True)
    before = len(rows)
    rows = rows.drop_duplicates(["shot", "t_start", "t_end", "regime", "source"])
    merged = merge_intervals(rows)
    bins = bins_from_intervals(merged)
    split = split_shots(bins, test_only, seed)
    bins = bins.merge(split, on="shot", how="left", validate="many_to_one")
    train = (bins.split == "train") & (bins.label >= 0)
    bins["train_weight"] = 0.
    bins.loc[train, "train_weight"] = balanced_weights(bins.loc[train, "label"],
                                                       bins.loc[train, "shot"])
    comparisons, shot_comparison = comparison(rows)
    durations = merged.assign(duration_ms=merged.t_end - merged.t_start)
    summary = {
        "time_units": "ms", "seed": seed, "modes": list(MODES),
        "hdf_class_order_basis": "Inferred from workbook L,H,QH,WP header and interval agreement; HDF has no class-name metadata",
        "sources": audits, "duplicate_signal_clips": duplicate_clips,
        "duplicate_intervals_removed": before - len(rows),
        "source_dependence": "Donor attribution is not annotation independence. Workbook/Jalal matching and clip reuse are audited; no independent ground-truth claim.",
        "source_stats": {source: {"shots": int(group.shot.nunique()), "intervals": len(group)}
                         for source, group in rows.groupby("source")},
        "merged_shots": int(merged.shot.nunique()),
        "status_duration_ms": durations.groupby("status").duration_ms.sum().to_dict(),
        "comparisons": comparisons,
        "known_bins": int((bins.label >= 0).sum()), "excluded_bins": int((bins.label < 0).sum()),
        "test_only_shots": sorted(test_only),
        "split_stats": {name: {"shots": int(group.shot.nunique()),
                               "L_bins": int((group.label == 0).sum()),
                               "H_bins": int((group.label == 1).sum()),
                               "excluded_bins": int((group.label < 0).sum())}
                        for name, group in bins.groupby("split")},
        "balancing": "Training only: equal H/L effective mass; equal shot mass per class. Recompute after feature availability exclusion.",
        "bin_policy": "50 ms; full coverage and one binary class; H/L conflicts, transition bins and annotation gaps excluded",
    }
    out.mkdir(parents=True, exist_ok=True)
    for name, table in (("source_intervals", rows), ("merged_intervals", merged),
                        ("targets", bins), ("split", split), ("shot_comparison", shot_comparison),
                        ("conflicts", merged.loc[merged.status.str.contains("conflict")])):
        table.to_csv(out / f"{name}.csv", index=False)
    summary["output_hashes"] = {p.name: digest(p) for p in sorted(out.glob("*.csv"))}
    (out / "labels.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw", type=Path, default=Paths.from_env().label_tables / "confinement/raw")
    parser.add_argument("--out", type=Path, default=Paths.from_env().root / "confinement/v1")
    parser.add_argument("--seed", type=int, default=20261001)
    args = parser.parse_args(argv)
    result = build(args.raw, args.out, args.seed)
    print(json.dumps({k: result[k] for k in ("merged_shots", "source_stats", "status_duration_ms", "split_stats")}, indent=2))


if __name__ == "__main__":
    main()
