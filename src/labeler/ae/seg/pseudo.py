"""Pseudo-masks: TokEye's coherent mask inside the owner's AE frames, 80-250 kHz.

    python -m labeler.ae.seg.pseudo [--shots S ...]

Drawing an AE mode pixel by pixel is slow, and a mode can cover a large part of
the spectrogram. Two things already say where it is. TokEye's cleaned coherent
mask (`$LABELER_ROOT/ae/masks/<shot>_<split>_clean.npz`, 0-2 s, four chords)
lights every coherent line, AE or not. The owner's saved frames say when AE is
present. Where both agree, inside 80-250 kHz, the pixel is very likely the mode.

**Grid.** The review store's level 8: 2.048 ms columns, the page's 257 bins of
0.977 kHz to 250 kHz. A TokEye column falls in the store column that holds its
time. A page bin takes TokEye's two 0.488 kHz bins whose centres lie in it. A
pixel is lit where TokEye lights at least two of the four chords (the rule
`labeler.ae.xpower.data` uses for MHD frames), in any of its TokEye columns.

**Values.** 1 is AE, 0 is background, 255 is ignored by training and scoring:
- in a frame the owner calls absent, every 80-250 kHz pixel is 0, TokEye's
  lines included: those are the coherent modes that are not AE (the hard
  negatives, an MHD mode's harmonics among them);
- in a frame the owner calls present, a lit 80-250 kHz pixel is 1 and an unlit
  one 0, except a ring `RING_BINS` bins and `RING_COLS` columns wide around the
  lit pixels, which is ignored (a mode's faint edge);
- a present column with no lit pixel is ignored: TokEye missed the mode there,
  and 0 would teach the network that there is none;
- a lit region of fewer than `MIN_AREA` pixels (8-connected) is ignored;
- everything else is ignored: uncertain, not observable, outside the window,
  outside TokEye's 0-2 s, below 80 kHz.

The reviewer can reject a region on the review page (`regions`); training then
takes that region as background.

**Output.** `$LABELER_ROOT/segmentation/alfven_eigenmode/pseudo-v1/<shot>.npz`
(`mask`, `t0_ms`, `dt_ms`, `y0_khz`, `dy_khz`) and `index.csv` beside them.

**pseudo-v2** (SegNet v2's masks, `SEG_VERSIONS["v2"]`):

    python -m labeler.ae.seg.pseudo --version v2 [--shots S ...]

The rule above over the owner's whole window and 0-250 kHz, on the same grid,
from TokEye's whole-shot masks (`$LABELER_ROOT/ae/masks-full`) and ae_xpower
v3's label snapshot (`models/ae_xpower/v3/review/labels.csv`, refused unless its
sha256 is v3's), never the live labels. An absent column is 0 at every bin.
Below 80 kHz TokEye lights every MHD line, so in a present column three rules
(`mhdlines`) IGNORE pixels, never 0 or 1:
- **absent-run:** a region lit below 80 kHz with at least `run_ms` of its
  columns in frames the owner calls absent;
- **steady:** a region lit below 80 kHz whose column-centroid frequency stays
  below 60 kHz, within `drift_khz`, for at least `steady_ms`;
- **bright:** an unlit pixel, at any frequency, whose brightest cross-power row
  reaches `bright_u8`.
The first two see only the pixels below 80 kHz, so a region crossing 80 kHz is
cut there. Above 80 kHz a present column's lit pixels are AE, as in pseudo-v1:
the owner labelled on that view. The ring, `MIN_AREA` and a present column left
with no AE pixel are as above.

The rules' values are picked on SegNet v2's training shots only (the `train`
shots of `v2_split`), in two passes. First, the rule shots' lines give `run_ms`,
`drift_khz` and `steady_ms` (`mhdlines.pick`), and their masks under those rules
give `bright_u8` (`mhdlines.bright_level`). Then every shot's mask is built with
the finished rules. A value that falls back, no candidate being within the 5 %
budget, is a WARNING on stderr and is stated in rules.md. A rule shot that cannot
be read, a damaged file included, stops the command before it writes anything.

**Output.** `$LABELER_ROOT/segmentation/alfven_eigenmode/pseudo-v2/`: the masks
and `index.csv` as above; `rules.json` and `rules.md`, the values, every
candidate's cost and take, `bright_background_share` and, per 20 kHz band, what
the chosen rules take; and `meta.json`, written last. The command prints a JSON
summary: the rules, `run_ms_within`, `steady_within`, `bright_u8` and
`bright_background_share`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import zipfile
import zlib
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import numpy as np
from scipy import ndimage

from ...config import Paths, atomic_path, git_sha, sha256_of
from ...events.catalog.states import ABSENT, PRESENT
from ...events.review import labels
from ...scoring.frames import FRAME_MS, OUTSIDE
from ..xpower import event_dir, read_snapshot, snapshot_file, tokeye_masks
from ..xpower.data import (
    BAND_KHZ,
    MIN_CHORDS,
    N_VAL,  # v2_split reads pseudo.N_VAL at call time, so a test can set it
    band_slice,
    clean_path,
    make_split,
    seldnet_split,
    store_rows,
    targets,
    tokeye_clean,
    window_frames,
)
from . import EVENT, PSEUDO, SEG_VERSIONS, VERSION, pseudo_dir
from .mhdlines import (
    TOKEYE_VERSION,
    Rules,
    ShotLines,
    bright_background_share,
    bright_level,
    brightest,
    fallbacks,
    mhd_like,
    pick,
    write_rules,
)

LEVEL = 8
IGNORE = 255
RING_BINS = 2
RING_COLS = 1
MIN_AREA = 6
EIGHT = np.ones((3, 3), dtype=bool)
INDEX_COLUMNS = (
    "shot",
    "file",
    "ae_px",
    "background_px",
    "ignored_px",
    "regions",
    "ae_cols",
    "present_cols_unlit",
)
#: What reading one shot can raise in pseudo-v2, a damaged record's errors among
#: them (np.load of a truncated or corrupt .npz).
UNREADABLE = (KeyError, OSError, ValueError, EOFError, zipfile.BadZipFile, zlib.error)


@dataclass(frozen=True)
class PseudoMask:
    """One shot's mask: columns `t0_ms + k * dt_ms`, bins `y0_khz + j * dy_khz`."""

    shot: int
    t0_ms: float
    dt_ms: float
    y0_khz: float
    dy_khz: float
    mask: np.ndarray  # (n_y, n) uint8: 0, 1 or IGNORE
    #: Columns the owner calls present where TokEye lit nothing in the band.
    present_unlit: int = 0

    def save(self, path) -> None:
        with atomic_path(Path(path)) as tmp, open(tmp, "wb") as f:
            np.savez_compressed(
                f,
                shot=np.int64(self.shot),
                t0_ms=self.t0_ms,
                dt_ms=self.dt_ms,
                y0_khz=self.y0_khz,
                dy_khz=self.dy_khz,
                mask=self.mask,
                present_unlit=np.int64(self.present_unlit),
            )

    @classmethod
    def load(cls, path) -> PseudoMask:
        with np.load(path) as z:
            return cls(
                int(z["shot"]),
                float(z["t0_ms"]),
                float(z["dt_ms"]),
                float(z["y0_khz"]),
                float(z["dy_khz"]),
                z["mask"],
                int(z["present_unlit"]),
            )


def tokeye_rows(clean: np.ndarray, min_chords: int = MIN_CHORDS) -> np.ndarray:
    """TokEye's `(chords, 512, T)` mask on the page's 257 bins, `(257, T)`.

    TokEye bin b is centred on (b + 1) x 0.488 kHz (its DC bin is dropped) and
    page bin k on k x 0.977 kHz, so page bin k takes TokEye bins 2k - 2 and 2k - 1.
    """
    lit = np.asarray(clean).sum(axis=0) >= min_chords
    padded = np.zeros((514, lit.shape[1]), dtype=bool)
    padded[2:] = lit
    return padded.reshape(257, 2, -1).any(axis=1)


def pool_columns(lit: np.ndarray, t_ms, grid) -> np.ndarray:
    """`(n_y, T)` flags at times `t_ms` onto the grid's columns, OR within each."""
    t = np.asarray(t_ms, dtype=np.float64)
    cols = np.floor((t - grid.t0_ms) / grid.dt_ms).astype(np.int64)
    keep = np.flatnonzero((cols >= 0) & (cols < grid.n))
    out = np.zeros((lit.shape[0], grid.n), dtype=bool)
    if keep.size:
        which = cols[keep]
        starts = np.flatnonzero(np.r_[True, np.diff(which) > 0])
        out[:, which[starts]] = np.logical_or.reduceat(lit[:, keep], starts, axis=1)
    return out


def covered_columns(t_ms, grid) -> np.ndarray:
    """The grid columns whose whole span lies inside TokEye's record."""
    t = np.asarray(t_ms, dtype=np.float64)
    half = (t[1] - t[0]) / 2 if len(t) > 1 else 0.0
    starts = grid.t0_ms + np.arange(grid.n) * grid.dt_ms
    return (starts >= t[0] - half) & (starts + grid.dt_ms <= t[-1] + half)


def column_states(label: labels.Label, grid) -> np.ndarray:
    """The owner's state of the frame holding each column's centre; OUTSIDE if none."""
    first, n = window_frames(label.window)
    states = targets(label, first, n)
    centres = grid.t0_ms + (np.arange(grid.n) + 0.5) * grid.dt_ms
    k = np.floor(centres / FRAME_MS).astype(np.int64) - first
    out = np.full(grid.n, OUTSIDE, dtype=np.int64)
    inside = (k >= 0) & (k < n)
    out[inside] = states[k[inside]]
    return out


def build(
    shot: int, label: labels.Label, grid, n_y: int, y0: float, dy: float, tokeye
) -> PseudoMask:
    """The pseudo-mask on `grid`; `tokeye` is `tokeye_clean`'s `(t_ms, clean, ann)`."""
    t_ms, clean, _ = tokeye
    band = band_slice(y0, dy, n_y, BAND_KHZ)
    in_band = np.zeros(n_y, dtype=bool)
    in_band[band] = True
    lit = pool_columns(tokeye_rows(clean), t_ms, grid)[:n_y]
    state = column_states(label, grid)
    state[~covered_columns(t_ms, grid)] = OUTSIDE
    present, absent = state == PRESENT, state == ABSENT
    positive = lit & in_band[:, None] & present[None, :]
    regions, count = ndimage.label(positive, structure=EIGHT)
    sizes = np.bincount(regions.ravel(), minlength=count + 1)
    small = (sizes < MIN_AREA)[regions] & positive
    positive &= ~small
    ring = ndimage.binary_dilation(
        positive, structure=np.ones((2 * RING_BINS + 1, 2 * RING_COLS + 1), bool)
    )
    mask = np.full((n_y, grid.n), IGNORE, dtype=np.uint8)
    scored = in_band[:, None] & (absent | present)[None, :]
    mask[scored] = 0
    mask[ring & ~positive & present[None, :] & in_band[:, None]] = IGNORE
    mask[small] = IGNORE
    unlit = present & ~positive.any(axis=0)
    mask[:, unlit] = IGNORE
    mask[positive] = 1
    return PseudoMask(
        int(shot), grid.t0_ms, grid.dt_ms, float(y0), float(dy), mask, int(unlit.sum())
    )


def summary(pm: PseudoMask, file: str) -> dict:
    mask = pm.mask
    _, count = ndimage.label(mask == 1, structure=EIGHT)
    positive_cols = (mask == 1).any(axis=0)
    return {
        "shot": pm.shot,
        "file": file,
        "ae_px": int((mask == 1).sum()),
        "background_px": int((mask == 0).sum()),
        "ignored_px": int((mask == IGNORE).sum()),
        "regions": int(count),
        "ae_cols": int(positive_cols.sum()),
        "present_cols_unlit": pm.present_unlit,
    }


def make(
    paths: Paths, shot: int, label: labels.Label, *, tokeye_bytes: bytes | None = None
) -> PseudoMask:
    grid, values, y0, dy = store_rows(paths.spectrogram_file(EVENT, shot), LEVEL)
    tokeye = clean_path(tokeye_masks(paths), shot)
    if tokeye is None:
        raise FileNotFoundError(f"{shot} has no TokEye mask")
    data = tokeye.read_bytes() if tokeye_bytes is None else tokeye_bytes
    return build(
        shot, label, grid, values.shape[1], y0, dy, tokeye_clean(BytesIO(data))
    )


def v2_split(shots, masks_dir) -> dict[int, str]:
    """SegNet v2's split: make_split(shots, seldnet_split(masks_dir), n_val=N_VAL),
    with this module's `N_VAL` read at call time, so a test can set it."""
    return make_split(shots, seldnet_split(masks_dir), n_val=N_VAL)


def _masks_full_file(paths: Paths, shot: int) -> Path:
    """The shot's clean file in TokEye's whole-shot masks; FileNotFoundError if none."""
    masks = tokeye_masks(paths, TOKEYE_VERSION)
    tokeye = clean_path(masks, shot)
    if tokeye is None:
        raise FileNotFoundError(f"{shot} has no TokEye mask in {masks}")
    return tokeye


def shot_lines(
    paths: Paths, shot: int, label: labels.Label, *, tokeye_bytes: bytes | None = None
) -> ShotLines:
    """One shot as pseudo-v2 reads it: store_rows(store, LEVEL); tokeye_clean of
    the shot's masks-full clean file (`tokeye_bytes`, its bytes, when given;
    FileNotFoundError without the file); lit = pool_columns(tokeye_rows(clean),
    t_ms, grid)[:n_y]; state = column_states(label, grid), OUTSIDE where not
    covered_columns; bright = brightest(values)."""
    grid, values, y0, dy = store_rows(paths.spectrogram_file(EVENT, shot), LEVEL)
    tokeye = _masks_full_file(paths, shot)
    data = tokeye.read_bytes() if tokeye_bytes is None else tokeye_bytes
    t_ms, clean, _ = tokeye_clean(BytesIO(data))
    lit = pool_columns(tokeye_rows(clean), t_ms, grid)[: values.shape[1]]
    state = column_states(label, grid)
    state[~covered_columns(t_ms, grid)] = OUTSIDE
    return ShotLines(
        int(shot),
        grid.t0_ms,
        grid.dt_ms,
        float(y0),
        float(dy),
        lit,
        state,
        brightest(values),
    )


def build_v2(lines: ShotLines, rules: Rules) -> PseudoMask:
    """pseudo-v2's mask of one shot. In this order: IGNORE everywhere; 0 in every
    absent or present column, at every bin; positive = lit & ~mhd_like & present
    (mhd_like takes pixels below 80 kHz only, so above it lit present pixels are
    AE); its regions under MIN_AREA are IGNORED and dropped from it; the ring (a
    (2*RING_BINS+1) x (2*RING_COLS+1) dilation of positive) & ~positive & present
    is IGNORED; mhd_like & present is IGNORED; with bright_u8, present & ~lit &
    bright >= bright_u8 is IGNORED; a present column with no positive pixel is
    IGNORED (present_unlit counts them); positive is 1."""
    lit = lines.lit
    present = lines.state == PRESENT
    mhd = mhd_like(lines, rules) & present
    mask = np.full(lit.shape, IGNORE, dtype=np.uint8)
    mask[:, present | (lines.state == ABSENT)] = 0
    positive = lit & ~mhd & present
    regions, count = ndimage.label(positive, structure=EIGHT)
    sizes = np.bincount(regions.ravel(), minlength=count + 1)
    small = (sizes < MIN_AREA)[regions] & positive
    positive &= ~small
    mask[small] = IGNORE
    ring = ndimage.binary_dilation(
        positive, structure=np.ones((2 * RING_BINS + 1, 2 * RING_COLS + 1), bool)
    )
    mask[ring & ~positive & present] = IGNORE
    mask[mhd] = IGNORE
    if rules.bright_u8 is not None:
        mask[present & ~lit & (lines.bright >= rules.bright_u8)] = IGNORE
    unlit = present & ~positive.any(axis=0)
    mask[:, unlit] = IGNORE
    mask[positive] = 1
    return PseudoMask(
        int(lines.shot),
        float(lines.t0_ms),
        float(lines.dt_ms),
        float(lines.y0_khz),
        float(lines.dy_khz),
        mask,
        int(unlit.sum()),
    )


def make_v2(
    paths: Paths,
    shot: int,
    label: labels.Label,
    rules: Rules,
    *,
    tokeye_bytes: bytes | None = None,
) -> PseudoMask:
    """build_v2(shot_lines(...), rules)."""
    return build_v2(shot_lines(paths, shot, label, tokeye_bytes=tokeye_bytes), rules)


def _main_v2(p: argparse.ArgumentParser, args: argparse.Namespace) -> int:
    """`main --version v2` (the module's **pseudo-v2**). Every failure before the
    masks are written is a `p.error`, and writes nothing."""
    paths = Paths.from_env()
    spec = SEG_VERSIONS["v2"]
    masks = tokeye_masks(paths, TOKEYE_VERSION)
    try:
        data, saved = read_snapshot(paths, spec.labels)
    except (KeyError, OSError, ValueError) as error:
        p.error(f"label snapshot {spec.labels}: {type(error).__name__}: {error}")
    try:
        split = v2_split(sorted(saved), masks)
    except (KeyError, OSError, ValueError) as error:
        p.error(f"{masks}: {type(error).__name__}: {error}")
    # Each rule shot's masks-full bytes are read once: its lines, and their sha256.
    ruled = {}
    for shot in sorted(s for s, part in split.items() if part == "train"):
        try:
            tokeye = _masks_full_file(paths, shot).read_bytes()
            ruled[shot] = (
                shot_lines(paths, shot, saved[shot], tokeye_bytes=tokeye),
                hashlib.sha256(tokeye).hexdigest(),
            )
        except UNREADABLE as error:
            p.error(f"rule shot {shot}: {type(error).__name__}: {error}")
    lines = [line for line, _ in ruled.values()]
    try:
        rules, report = pick(lines)
    except ValueError as error:
        p.error(str(error))
    for warning in fallbacks(rules, report):
        print(f"WARNING: pseudo-v2 rules: {warning}", file=sys.stderr, flush=True)
    # The first pass: the rule shots' masks under these rules give bright_u8,
    # and the share of their present-frame background it IGNOREs.
    first = [build_v2(line, rules).mask for line in lines]
    rules = replace(rules, bright_u8=bright_level(first, lines))
    share = bright_background_share(first, lines, rules.bright_u8)
    report = {**report, "bright_background_share": share}
    del first
    out = pseudo_dir(paths, "v2")
    out.mkdir(parents=True, exist_ok=True)
    rows, failed, tokeye_hashes = [], [], {}
    for shot in args.shots or sorted(saved):
        try:
            if shot in ruled:  # from the lines, and bytes, the rules were read from
                line, digest = ruled[shot]
                pm = build_v2(line, rules)
            else:
                tokeye = _masks_full_file(paths, shot).read_bytes()
                pm = make_v2(paths, shot, saved[shot], rules, tokeye_bytes=tokeye)
                digest = hashlib.sha256(tokeye).hexdigest()
        except UNREADABLE as error:
            failed.append(shot)
            print(f"{shot}: {type(error).__name__}: {error}", flush=True)
            continue
        tokeye_hashes[str(shot)] = digest
        pm.save(out / f"{shot}.npz")
        rows.append(summary(pm, f"{shot}.npz"))
    with atomic_path(out / "index.csv") as tmp, open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    write_rules(out, rules, report)
    meta = {
        "pseudo": spec.pseudo,
        "shots": len(rows),
        "failed": failed,
        "tokeye": str(masks),
        "tokeye_sha256": tokeye_hashes,
        "labels": str(snapshot_file(paths, spec.labels)),
        "labels_sha256": hashlib.sha256(data).hexdigest(),
        "rules": asdict(rules),
        "rule_shots": report["rule_shots"],
        "split": {
            part: sum(v == part for v in split.values())
            for part in ("train", "val", "test")
        },
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out / "meta.json") as tmp:
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
    said = {
        "out": str(out),
        "shots": len(rows),
        "failed": failed,
        "rule_shots": len(lines),
        "rules": asdict(rules),
        "run_ms_within": report["run_ms_within"],
        "steady_within": report["steady_within"],
        "bright_u8": rules.bright_u8,
        "bright_background_share": share,
    }
    print(json.dumps(said))
    return 1 if failed else 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--shots", type=int, nargs="*", help="default: every saved shot")
    p.add_argument(
        "--version",
        choices=sorted(SEG_VERSIONS),
        default=VERSION,
        help="v1 (the default): pseudo-v1; v2: pseudo-v2 (from v3's snapshot)",
    )
    args = p.parse_args(argv)
    if args.version == "v2":
        return _main_v2(p, args)
    paths = Paths.from_env()
    directory = event_dir(paths)
    saved = labels.read_saved(directory)
    shots = args.shots or sorted(saved)
    out = pseudo_dir(paths)
    out.mkdir(parents=True, exist_ok=True)
    rows, failed, tokeye_hashes = [], [], {}
    for shot in shots:
        try:
            tokeye = clean_path(tokeye_masks(paths), shot)
            if tokeye is None:
                raise FileNotFoundError(f"{shot} has no TokEye mask")
            data = tokeye.read_bytes()
            pm = make(paths, shot, saved[shot], tokeye_bytes=data)
            tokeye_hashes[str(shot)] = hashlib.sha256(data).hexdigest()
        except (KeyError, OSError, ValueError) as error:
            failed.append(shot)
            print(f"{shot}: {type(error).__name__}: {error}", flush=True)
            continue
        pm.save(out / f"{shot}.npz")
        rows.append(summary(pm, f"{shot}.npz"))
    with atomic_path(out / "index.csv") as tmp, open(tmp, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    labels_file = labels.labels_path(directory)
    meta = {
        "pseudo": PSEUDO,
        "shots": len(rows),
        "failed": failed,
        "tokeye_sha256": tokeye_hashes,
        "labels": str(labels_file),
        "labels_sha256": sha256_of(labels_file) if labels_file.is_file() else None,
        "git_sha": git_sha(),
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    with atomic_path(out / "meta.json") as tmp:
        tmp.write_text(json.dumps(meta, indent=1) + "\n")
    print(f"wrote {len(rows)} pseudo-masks to {out}; {len(failed)} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
