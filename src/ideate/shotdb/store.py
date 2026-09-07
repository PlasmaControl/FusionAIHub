"""In-memory view of a built database: Parquet tables + embedding matrices, boolean masks for
hard filters, and brute-force cosine k-NN.

Brute force is not a placeholder. The database is a few thousand segment rows of 24-dimensional
unit vectors; the whole similarity computation is one `M @ q`, which is faster than any index's
lookup overhead at this size and has no build step, no approximation and no stale-index failure
mode. Revisit at ~10^6 rows, not before.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Range, ShotRecord


class ShotDB:
    """`segments` is the queryable view (segment rows joined to their shot's metadata);
    `segments_base` is exactly what segments.parquet holds, which is what `build.add` rewrites.
    Keeping them apart matters: joining the shot columns into the persisted frame would write
    them back on the next add, and the join after that would collide with itself."""

    def __init__(
        self,
        db_dir: Path,
        shots: pd.DataFrame,
        segments: pd.DataFrame,
        emb: dict[str, np.ndarray],
        pca: dict,
        manifest: dict,
        shapes: np.ndarray | None = None,
        windows: pd.DataFrame | None = None,
    ):
        self.db_dir = Path(db_dir)
        self.shots = shots
        self.segments_base = segments
        self.shapes = np.zeros((len(segments), 0), np.float32) if shapes is None else shapes
        self.emb = emb
        self.pca = pca
        self.manifest = manifest
        # windows.parquet: one row per 250 ms window of every encoded shot, aligned with
        # emb["ignite_win"]; only fragment queries (--ref-window) read it.
        self.windows = windows
        meta = shots[["shot", "run_id", "regime", "verdict", "operational"]].set_index("shot")
        # A left join preserves row order, which is what keeps self.segments aligned row-for-row
        # with emb["scalar"]; every lookup in this class relies on that.
        self.segments = segments.join(meta, on="shot", rsuffix="_shot")

    @classmethod
    def load(cls, db_dir: Path) -> ShotDB:
        db_dir = Path(db_dir)
        shots = pd.read_parquet(db_dir / "shots.parquet")
        segments = pd.read_parquet(db_dir / "segments.parquet")
        emb = {p.stem.removeprefix("emb_"): np.load(p) for p in sorted(db_dir.glob("emb_*.npy"))}
        pca = json.loads((db_dir / "pca.json").read_text())
        manifest = json.loads((db_dir / "manifest.json").read_text())
        shape_path = db_dir / "shapes.npy"
        shapes = np.load(shape_path) if shape_path.exists() else None
        win_path = db_dir / "windows.parquet"
        windows = pd.read_parquet(win_path) if win_path.exists() else None
        return cls(db_dir, shots, segments, emb, pca, manifest, shapes, windows)

    # ------------------------------------------------------------------ single-row access

    def get(self, shot: int) -> ShotRecord:
        """The full record, rehydrated from the JSON column. Parquet holds the flat view for
        querying and the record verbatim for reading; this is the second one."""
        return ShotRecord.model_validate_json(self.shots.loc[shot, "record_json"])

    def row(self, seg_id: str) -> pd.Series:
        return self.segments.loc[seg_id]

    def shape_matrix(self) -> np.ndarray:
        """The waveform-shape block of the feature matrix, one row per segment.

        Persisted alongside the tables (shapes.npy) rather than reconstructed: it is half the
        scalar feature vector's width and cannot be recovered from segments.parquet, so without
        it neither a refit nor an audit of emb["scalar"] is possible without re-reading every
        raw HDF5 file. At 140 float32 per segment it costs ~0.6 kB a row."""
        return self.shapes

    # ------------------------------------------------------------------------- hard filters

    def mask(
        self,
        segment: str,
        constraints: dict[str, Range] | None = None,
        require_labels: Iterable[str] = (),
        avoid_labels: Iterable[str] = (),
        exclude_shots: Iterable[int] = (),
        exclude_runs: Iterable[str] = (),
    ) -> np.ndarray:
        s = self.segments
        # copy=True: to_numpy() can hand back a read-only view of pandas' own block, and
        # every filter below narrows this array in place.
        m = (s["segment"] == segment).to_numpy(dtype=bool, copy=True)
        for col, rng in (constraints or {}).items():
            if col not in s.columns:  # unknown column: nothing can satisfy it, so say so loudly
                raise KeyError(f"constraint on unknown column {col!r}")
            v = s[col].to_numpy(dtype=float)
            # A NaN never satisfies a range. "Not recorded" is not "within tolerance": a shot
            # whose density was never measured must not come back as a density match.
            ok = np.isfinite(v)
            if rng.lo is not None:
                ok &= v >= rng.lo
            if rng.hi is not None:
                ok &= v <= rng.hi
            m &= ok
        req, avoid = set(require_labels), set(avoid_labels)
        if req or avoid:
            # The regime joins the operational set here so a caller can require "QH" or avoid
            # "L" with the same vocabulary it uses for "dud".
            labels = [
                set(ops) | {reg} for ops, reg in zip(s["operational"], s["regime"], strict=True)
            ]
            if req:
                m &= np.array([req <= lab for lab in labels], dtype=bool)
            if avoid:
                m &= np.array([not (avoid & lab) for lab in labels], dtype=bool)
        if exclude_shots:
            m &= ~s["shot"].isin(set(exclude_shots)).to_numpy()
        if exclude_runs:
            m &= ~s["run_id"].isin(set(exclude_runs)).to_numpy()
        return m

    # -------------------------------------------------------------------------------- k-NN

    def _ids_for(self, n_rows: int) -> pd.Index:
        """Which index a matrix's rows belong to. Matched by length: segment-keyed matrices have
        one row per segment, shot-keyed ones (the text embeddings) one per shot. A matrix that
        matches neither -- a scratch matrix a caller stuffed into `emb` -- gets positional ids
        rather than an IndexError from indexing the wrong frame."""
        if n_rows == len(self.segments):
            return self.segments.index
        if n_rows == len(self.shots):
            return self.shots.index.astype(str)
        return pd.RangeIndex(n_rows).astype(str)

    def knn(
        self, matrix: str, qvec: np.ndarray, k: int, mask: np.ndarray | None = None
    ) -> list[tuple[str, float]]:
        M = self.emb[matrix]
        q = np.asarray(qvec, dtype=np.float32).ravel()
        q = q / (float(np.linalg.norm(q)) or 1.0)
        sims = M @ q
        if mask is not None:
            sims = np.where(mask, sims, -np.inf)
        k = min(int(k), sims.size)
        if k <= 0:
            return []
        # argpartition first: full sort of every row to return ten of them is wasted work, and
        # this is the call the 200 ms budget is spent in.
        part = np.argpartition(-sims, k - 1)[:k]
        idx = part[np.argsort(-sims[part], kind="stable")]
        ids = self._ids_for(M.shape[0])
        return [(str(ids[i]), float(sims[i])) for i in idx if np.isfinite(sims[i])]
