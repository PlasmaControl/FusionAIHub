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
from functools import cached_property
from pathlib import Path

import numpy as np
import pandas as pd

from ..schema import Range, ShotRecord


def _empty(dtypes: dict[str, str]) -> pd.DataFrame:
    """A zero-row frame with exactly these columns and dtypes, in this order."""
    return pd.DataFrame({name: pd.Series(dtype=dt) for name, dt in dtypes.items()})


# The three optional label tables' schemas are defined once, by the modules that WRITE them, and
# imported here lazily -- inside the functions rather than at module scope, because
# `ideate.labels.claims` imports `ideate.shotdb.text`, and a top-level import would close that
# circle through this package's own `__init__`.


def empty_events() -> pd.DataFrame:
    """No events, labelmaker's own columns and dtypes."""
    from labelmaker.events import schema as events_schema

    return _empty({name: events_schema.DTYPES[name] for name in events_schema.COLUMNS})


def empty_labels_wide() -> pd.DataFrame:
    """No label summaries, `labels.join`'s columns and dtypes."""
    from ..labels.join import LABELS_WIDE_DTYPES

    return _empty(LABELS_WIDE_DTYPES)


def empty_text_claims() -> pd.DataFrame:
    """No claims, `labels.claims`'s columns and dtypes."""
    from ..labels.claims import CLAIMS_DTYPES

    return _empty(CLAIMS_DTYPES)


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
        events: pd.DataFrame | None = None,
        labels_wide: pd.DataFrame | None = None,
        text_claims: pd.DataFrame | None = None,
        event_sources: pd.DataFrame | None = None,
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
        # The three label tables (`ideate labels join`). Optional on disk and never None here:
        # a database built before the join, or one whose join found no events, gets the EMPTY
        # TYPED frame rather than None, so a caller filters `db.events` without first asking
        # whether there is a table -- and gets zero rows, which is what "nobody has looked yet"
        # honestly is. The distinction that matters ("did a detector cover this shot?") is not
        # carried by the frame's existence but by the coverage columns on its rows.
        self.events = empty_events() if events is None else events
        self.labels_wide = empty_labels_wide() if labels_wide is None else labels_wide
        self.text_claims = empty_text_claims() if text_claims is None else text_claims
        from ..labels.event_sources import empty_sources

        self.event_sources = empty_sources() if event_sources is None else event_sources
        self.has_event_sources = event_sources is not None
        #: name -> message for an optional label table that exists but could not be read.
        self.load_errors: dict[str, str] = {}
        meta = shots[["shot", "run_id", "regime", "verdict", "operational"]].set_index("shot")
        # A left join preserves row order, which is what keeps self.segments aligned row-for-row
        # with emb["scalar"]; every lookup in this class relies on that.
        self.segments = segments.join(meta, on="shot", rsuffix="_shot")
        self._phenomenon_evidence: dict = {}

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
        # The label tables are optional AND may be unreadable (a torn or foreign parquet). An
        # unreadable one is recorded, not fatal: the core tables still load, and the reader
        # that needs the table (`mcp.tools.get_events`, `retrieval.phenomena`) reports the
        # failure in its own words -- an eager raise here turned a corrupt `events.parquet`
        # into a protocol error on every MCP call, including ones that never touch events.
        label_tables: dict[str, pd.DataFrame] = {}
        load_errors: dict[str, str] = {}
        for name in ("events", "labels_wide", "text_claims", "event_sources"):
            p = db_dir / f"{name}.parquet"
            if not p.exists():
                continue
            try:
                label_tables[name] = pd.read_parquet(p)
            except Exception as exc:  # noqa: BLE001 - reported to the caller, never swallowed
                load_errors[name] = f"{type(exc).__name__}: {exc}"
        db = cls(db_dir, shots, segments, emb, pca, manifest, shapes, windows, **label_tables)
        db.load_errors = load_errors
        return db

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

    @cached_property
    def _phenomenon_config(self):
        from ..retrieval import phenomena as ph

        return ph._config()

    def phenomenon_evidence(self, shot: int, phenomenon: str, segment: str):
        """Evidence cached for this loaded snapshot, shared by masks and retrieval channels."""
        from ..retrieval import phenomena as ph

        key = (int(shot), phenomenon, segment)
        if key not in self._phenomenon_evidence:
            self._phenomenon_evidence[key] = ph.evidence(
                shot, phenomenon, self, segment, label_floor=self._phenomenon_config[2],
            )
        return self._phenomenon_evidence[key]

    @cached_property
    def _label_tokens(self) -> list[frozenset[str]]:
        """One token set per segment, built once on first use of a label filter.

        Phenomena require observed intervals. Sources name both the event producer and evidence
        kind within the segment. Label summaries are shot scoped; their operating point is the
        published table threshold, a registered threshold, or the detection label floor.
        """
        from ..retrieval import phenomena as ph

        registry = ph.registry()
        floor = self._phenomenon_config[2]
        thresholds = {
            ref.key: floor if ref.thr is None else ref.thr
            for entry in registry.values() for ref in entry.labels
        }
        labels: dict[int, set[str]] = {}
        for row in self.labels_wide.to_dict("records"):
            key = f"{row['slug']}/{row['label']}"
            threshold = ph._f(row.get("thr"))
            if threshold is None:
                threshold = thresholds.get(key)
            p = ph._f(row.get("max_valid"))
            if threshold is not None and p is not None and row["n_valid"] > 0 and p >= threshold:
                labels.setdefault(int(row["shot"]), set()).add(f"label:{key}")
        events = {int(shot): frame.to_dict("records") for shot, frame in self.events.groupby("shot")}
        out = []
        for row in self.segments[["shot", "segment", "t0_ms", "t1_ms", "operational", "regime"]].itertuples():
            tokens = set(row.operational) | {row.regime} | labels.get(int(row.shot), set())
            window = (row.t0_ms / 1000.0, row.t1_ms / 1000.0)
            rows = [r for r in events.get(int(row.shot), ()) if ph._overlaps(r, window)]
            for event in rows:
                tokens.update((f"source:{event['source']}", f"source:{event['evidence_kind']}"))
            observed_sources = {r["source"] for r in rows if r["evidence_kind"] in ph.OBSERVED_KINDS}
            for pid, entry in registry.items():
                if observed_sources.intersection(entry.sources):
                    ev = self.phenomenon_evidence(row.shot, pid, row.segment)
                    if ev.intervals:
                        tokens.add(f"phenomenon:{pid}")
            out.append(frozenset(tokens))
        return out

    def _avoid_coverage(self, segment: str, avoid: Iterable[str], notes=None) -> np.ndarray:
        """Require relevant observed coverage, explaining excluded states at report time."""
        from ..retrieval import phenomena as ph

        keep = np.ones(len(self.segments), dtype=bool)
        for token in sorted(set(avoid)):
            if not token.startswith("phenomenon:"):
                continue
            pid = ph._avoid_ids([token])[0]
            counts: dict[str, int] = {}
            details: set[str] = set()
            n_observed = 0
            for i, row in enumerate(self.segments[["shot", "segment"]].itertuples(index=False)):
                if row.segment != segment:
                    continue
                ev = self.phenomenon_evidence(row.shot, pid, segment)
                if ev.intervals:
                    n_observed += 1
                    details.update(c for c in ev.caveats if c in ph.EVENT_CAVEATS.values())
                if ev.coverage_state != "observed":
                    keep[i] = False
                    counts[ev.coverage_state] = counts.get(ev.coverage_state, 0) + 1
                    details.update(c for c in ev.caveats if (
                        "coverage unknown" in c or "no detector registered" in c or "could not read" in c
                    ))
                elif ev.coverage_partial and f"phenomenon:{pid}" not in self._label_tokens[i]:
                    details.update(c for c in ev.caveats if "coverage" in c or "covered only" in c)
            if notes is not None:
                if n_observed:
                    notes.append(ph.AVOID_DROPPED.format(
                        token=token, n=n_observed, title=ph.registry()[pid].title,
                    ))
                notes.extend(
                    f"--avoid {token}: excluded {n} {segment} segment(s) with {state} coverage; "
                    "absence is not evidence"
                    for state, n in sorted(counts.items())
                )
                notes.extend(sorted(details))
        return keep

    def label_filter_caveats(self, segment: str, avoid: Iterable[str]) -> list[str]:
        notes: list[str] = []
        self._avoid_coverage(segment, avoid, notes)
        return notes

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
            labels = self._label_tokens
            if req:
                m &= np.array([req <= lab for lab in labels], dtype=bool)
            if avoid:
                m &= np.array([not (avoid & lab) for lab in labels], dtype=bool)
                m &= self._avoid_coverage(segment, avoid)
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
