"""Retrieval channels: one function each, `(QueryState, ShotDB) -> [(segment_id, score)]`.

`CHANNELS` at the bottom is the registry, and it is a plain dict on purpose: adding a channel is
one function plus one line in it, with nothing else in this package to read first. `rank.py`
iterates the dict and never names a channel.

Every channel applies `hard_filter` itself rather than receiving a mask. That recomputes the
filter once per channel (~1 ms on the 420-row database, and it is a boolean pass over one column
per constraint, so it stays cheap as the database grows). The alternative -- filtering after
fusion -- is wrong, not merely slower: a channel returns its top k, so masked-out rows would eat
the k budget and a query with a narrow filter would come back short or empty.

Copied, with attribution, from shotsearch (same author, MIT): the BM25 tokenizer regex from
`index/bm25_store.py`, and `split_negatives`/`text_matches_negative` from `query/negation.py`.
Copies, not imports -- shotsearch is a separate project and not a dependency of this one.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import math
import os
import re
from collections import Counter

import numpy as np

from ..flags.rules import actuator_columns
from ..schema import QueryState, Range
from ..shotdb.store import ShotDB

# How many candidates a channel returns before fusion. Bigger than any n a user asks for, so RRF
# sees a real ranking from each channel rather than the same handful of rows.
CANDIDATES = 200


# ------------------------------------------------------------------------------- hard filtering


def hard_filter(q: QueryState, db: ShotDB) -> np.ndarray:
    """Boolean mask over `db.segments` rows: segment name, ranges, labels, exclusions.

    `ShotDB.mask` does the work (it is the tested implementation of these semantics, including
    the rule that a NaN never satisfies a range) and raises KeyError on a column the database
    does not have, which the CLI turns into a message naming the column."""
    return db.mask(
        segment=q.segment,
        constraints=q.constraints,
        require_labels=q.require_labels,
        avoid_labels=q.avoid_labels,
        exclude_shots=q.exclude_shots,
        exclude_runs=q.exclude_runs,
    )


def nan_excluded(q: QueryState, db: ShotDB) -> dict[str, int]:
    """Per constrained column, how many rows of the requested segment it dropped for having no
    value at all -- as opposed to a value outside the range.

    This is not bookkeeping. "Never recorded" and "outside your range" are different answers to
    a physicist, and without this count a filter on a column that half the database is missing
    looks like a physics result rather than a coverage hole.
    """
    in_segment = (db.segments["segment"] == q.segment).to_numpy(dtype=bool)
    out: dict[str, int] = {}
    for col in q.constraints:
        if col not in db.segments.columns:
            continue
        missing = ~np.isfinite(db.segments[col].to_numpy(dtype=float))
        n = int(np.count_nonzero(in_segment & missing))
        if n:
            out[col] = n
    return out


# --------------------------------------------------------------------------------- query vector


def _actuator_column(name: str, db: ShotDB) -> str | None:
    """`nbi.total` / `ech.LUKE` -> the segments column that holds it, or a column name as given.

    `flags.rules.actuator_columns` is the one resolver, built from configs/ideate/actuators.yaml,
    so a new actuator system is reachable from a query the moment it is in the registry -- and so
    the value after `--actuator` names the same column here as in the operating-limit rules. This
    used to try `_mean` first while the rules took `_peak`, making one flag two quantities. A key
    the registry knows whose column this database lacks is None, never silently another stat.
    """
    if name in db.segments.columns:
        return name
    col = actuator_columns().get(name)
    return col if col is not None and col in db.segments.columns else None


def _midpoint(rng: Range) -> float | None:
    """A range's representative value: the middle when both bounds are given, otherwise the one
    bound there is -- a one-sided constraint still says roughly where to look."""
    if rng.lo is not None and rng.hi is not None:
        return 0.5 * (rng.lo + rng.hi)
    return rng.lo if rng.lo is not None else rng.hi


def target_values(q: QueryState, db: ShotDB) -> dict[str, float]:
    """Column -> the value this query asks for, from constraint midpoints and actuator settings."""
    out: dict[str, float] = {}
    for col, rng in q.constraints.items():
        mid = _midpoint(rng)
        if mid is not None and col in db.segments.columns:
            out[col] = mid
    for name, value in q.actuators.items():
        col = _actuator_column(name, db)
        if col is not None:
            out[col] = value
    return out


def query_z(q: QueryState, db: ShotDB) -> tuple[np.ndarray, list[str]] | None:
    """The z-scored feature vector a specs query asks for: z on the dimensions it named, 0 on
    every other one, in the frozen PCA's own column order. None when it named none of them.

    Zero is the right filler because the PCA was fitted on z-scored features, so 0 is the fitted
    population mean of each dimension -- "no preference" rather than a guess.
    """
    cols: list[str] = db.pca.get("feature_cols") or []
    mean = np.asarray(db.pca.get("mean") or [], dtype=np.float64)
    scale = np.asarray(db.pca.get("scale") or [], dtype=np.float64)
    # `build.build` assigns `feature_cols` onto `empty_pca()` when it had <= 2 rows to fit on, so
    # a column list with no fitted mean/scale behind it is "no PCA", not a malformed one. Gating
    # on `cols` alone raised IndexError at `mean[i]` on a two-shot database.
    if not cols or mean.size < len(cols) or scale.size < len(cols):
        return None
    width = len(cols) + int(db.pca.get("n_shape") or 0)
    pos = {c: i for i, c in enumerate(cols)}
    z = np.zeros(width, dtype=np.float64)
    named: list[str] = []
    for col, value in target_values(q, db).items():
        i = pos.get(col)
        if i is None:
            continue
        z[i] = (value - mean[i]) / scale[i]
        named.append(col)
    return (z, named) if named else None


def _project_z(z: np.ndarray, pca: dict) -> np.ndarray:
    """Whiten-project an already-z-scored row, exactly as `build.project_scalar` does after its
    impute-and-z step, and L2-normalize so cosine is a dot product."""
    var = np.asarray(pca["explained_variance"], dtype=np.float64)
    var = np.where(var > 0, var, 1.0)
    e = z @ np.asarray(pca["components"], dtype=np.float64).T / np.sqrt(var)
    n = float(np.linalg.norm(e))
    return (e / (n or 1.0)).astype(np.float32)


def ref_row(q: QueryState, db: ShotDB) -> str | None:
    """The reference shot's row in the requested segment, if the database holds it."""
    if q.ref_shot is None:
        return None
    seg_id = f"{q.ref_shot}:{q.segment}"
    return seg_id if seg_id in db.segments.index else None


# -------------------------------------------------------------------------------------- text


_TOKEN_RE = re.compile(r"[a-z0-9]+(?:[-=/_][a-z0-9]+)*")  # shotsearch/index/bm25_store.py


def tokenize(text: str) -> list[str]:
    """Lowercase + split, keeping physics tokens whole: n=2, q95, qh-mode, beta_n."""
    return _TOKEN_RE.findall(text.lower())


# shotsearch/query/negation.py. "no" only triggers in the "no ... shots" form, so it does not
# fire on a stray "no" inside operator prose.
_NEG_RES = [
    re.compile(r"(?i)\bbut\s+no\b\s+(.+?)(?:[.;]|$)"),
    re.compile(r"(?i)\bbut\s+not\b\s+(.+?)(?:[.;]|$)"),
    re.compile(r"(?i)\bwithout\b\s+(.+?)(?:[.;]|$)"),
    re.compile(r"(?i)\bexcluding\b\s+(.+?)(?:[.;]|$)"),
    re.compile(r"(?i)\bexcept(?:\s+for)?\b\s+(.+?)(?:[.;]|$)"),
    re.compile(r"(?i)\bno\s+(.+?\bshots?\b)"),
]
_TAIL = re.compile(r"\s*\b(?:shots?|discharges?)\b\s*$", re.IGNORECASE)


def _normalize(term: str) -> str:
    t = re.sub(r"[^0-9a-z]+", " ", term.lower()).strip()
    return _TAIL.sub("", t).strip()


def split_negatives(text: str) -> tuple[str, list[str]]:
    """(positive text, negated terms). No trigger -> (text, [])."""
    earliest: tuple[int, str] | None = None
    for rgx in _NEG_RES:
        m = rgx.search(text)
        if m and (earliest is None or m.start() < earliest[0]):
            earliest = (m.start(), m.group(1))
    if earliest is None:
        return text.strip(), []
    out: list[str] = []
    for part in re.split(r"\b(?:and|or)\b|[,/]", earliest[1], flags=re.IGNORECASE):
        term = _normalize(part)
        if term and term not in out:
            out.append(term)
    return text[: earliest[0]].strip(" ,;."), out


def text_matches_negative(haystack: str, negatives: list[str]) -> bool:
    """True if any negated term appears in `haystack` as whole words (punctuation/case-insensitive).

    Whole words, not a substring: `--negative H-mode` must not drop every QH-mode shot, and a
    negated "elm" must not match "helmet" (both did, when this was a plain `in`). Both sides are
    normalised the same way -- lowercase, runs of non-alphanumerics to one space -- so a
    hyphenated or slashed term matches however the operator punctuated it.
    """
    words = " ".join(re.sub(r"[^0-9a-z]+", " ", haystack.lower()).split())
    padded = f" {words} "
    for neg in negatives:
        term = " ".join(re.sub(r"[^0-9a-z]+", " ", neg.lower()).split())
        if term and f" {term} " in padded:
            return True
    return False


def positive_and_negatives(q: QueryState) -> tuple[str | None, list[str]]:
    """The query's positive text and every negative it carries -- the explicit `--negative`
    flags plus the ones written into the sentence ("... but no QH-mode shots")."""
    negatives = [_normalize(n) for n in q.negatives]
    if not q.text:
        return None, [n for n in negatives if n]
    positive, implicit = split_negatives(q.text)
    for term in implicit:
        if term not in negatives:
            negatives.append(term)
    return (positive or None), [n for n in negatives if n]


def _shot_text(db: ShotDB, shot: int) -> str:
    row = db.shots.loc[shot]
    return f"{row.get('text_mp') or ''} {row.get('text_log') or ''}"


def _segment_rows(db: ShotDB, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(row positions of the surviving segments, their shots' positions in the shots table)."""
    rows = np.flatnonzero(mask)
    shot_pos = {int(s): i for i, s in enumerate(db.shots.index)}
    keep, shots = [], []
    for r in rows:
        p = shot_pos.get(int(db.segments["shot"].iloc[r]))
        if p is not None:
            keep.append(r)
            shots.append(p)
    return np.asarray(keep, dtype=int), np.asarray(shots, dtype=int)


# ------------------------------------------------------------------------------------ channels


def scalar_knn(q: QueryState, db: ShotDB) -> list[tuple[str, float]]:
    """Cosine k-NN in the frozen 24-dimensional scalar PCA space.

    The query point is the reference shot's own embedding row when there is one -- no re-projection,
    so a shot is exactly identical to itself -- and otherwise the constraint/actuator vector
    projected through the same PCA.
    """
    if "scalar" not in db.emb or db.emb["scalar"].size == 0:
        return []
    seg_id = ref_row(q, db)
    if seg_id is not None:
        qvec = db.emb["scalar"][db.segments.index.get_loc(seg_id)]
    else:
        built = query_z(q, db)
        if built is None:
            return []
        qvec = _project_z(built[0], db.pca)
    if not np.any(qvec):
        return []
    mask = hard_filter(q, db)
    return db.knn("scalar", qvec, CANDIDATES, mask)


def text_knn(q: QueryState, db: ShotDB) -> list[tuple[str, float]]:
    """MiniLM cosine over the two per-shot texts, scored as max(mp, log).

    max, not mean: a shot matches the query if *either* the experiment's intent or what the
    operators wrote about it matches. Averaging would punish a shot whose mini-proposal is a
    one-line title, which is most of them.
    """
    for key in ("text_log", "text_mp"):
        if key not in db.emb or db.emb[key].size == 0:
            return []
    positive, negatives = positive_and_negatives(q)
    if positive:
        # The same guard cli.py sets, repeated because this package is importable on its own: on
        # a node with no outbound route sentence_transformers does not fail when it reaches into
        # huggingface_hub, it hangs for minutes inside httpx.connect_tcp (measured in Task 12; the
        # same load takes ~5 s offline). IDEATE_HF_ONLINE=1 opts out, which is what a first run
        # on a machine with no cached checkpoint needs.
        if os.environ.get("IDEATE_HF_ONLINE") != "1":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from ..shotdb.text import embed_texts

        qvec = embed_texts([positive])[0]
    else:
        seg_id = ref_row(q, db)
        if seg_id is None or q.ref_shot not in db.shots.index:
            return []
        p = db.shots.index.get_loc(q.ref_shot)
        qvec = db.emb["text_log"][p]
        if not np.any(qvec):
            qvec = db.emb["text_mp"][p]
    if not np.any(qvec):
        return []
    qvec = np.asarray(qvec, dtype=np.float32).ravel()
    sims = np.maximum(db.emb["text_log"] @ qvec, db.emb["text_mp"] @ qvec)
    rows, shots = _segment_rows(db, hard_filter(q, db))
    out: list[tuple[str, float]] = []
    for r, p in zip(rows, shots, strict=True):
        # A shot with no text at all embeds as a zero row and scores exactly 0.0. That is not a
        # weak match, it is no evidence, and it must not outrank a genuine negative cosine.
        if not np.any(db.emb["text_log"][p]) and not np.any(db.emb["text_mp"][p]):
            continue
        shot = int(db.shots.index[p])
        if negatives and text_matches_negative(_shot_text(db, shot), negatives):
            continue
        out.append((str(db.segments.index[r]), float(sims[p])))
    out.sort(key=lambda kv: -kv[1])
    return out[:CANDIDATES]


def bm25(q: QueryState, db: ShotDB) -> list[tuple[str, float]]:
    """Okapi BM25 (k1=1.5, b=0.75) over `text_log`, the operators' own words.

    It answers what the dense channel cannot: an exact term. "n=3" and "GASB" are rare tokens
    MiniLM smooths away, and a physicist typing one of them means it literally.

    Query text only. Using a reference shot's whole log as the query was considered and left out:
    a ~400-token document query matches on whatever it happens to share with everything, which is
    a different (and much noisier) thing than what --ref means here.

    The index is built per call. At 105 documents that is under a millisecond; at the few thousand
    this database is heading for it is still small enough not to need a persisted index.
    """
    positive, negatives = positive_and_negatives(q)
    if not positive:
        return []
    qtok = tokenize(positive)
    if not qtok:
        return []
    rows, shots = _segment_rows(db, hard_filter(q, db))
    if rows.size == 0:
        return []
    texts = [str(db.shots["text_log"].iloc[p] or "") for p in shots]
    corpus = [tokenize(t) for t in texts]
    scores = bm25_scores(corpus, qtok)
    out: list[tuple[str, float]] = []
    for i, (r, p) in enumerate(zip(rows, shots, strict=True)):
        if scores[i] <= 0:  # no query term occurs; BM25 has nothing to say about this shot
            continue
        shot = int(db.shots.index[p])
        if negatives and text_matches_negative(_shot_text(db, shot), negatives):
            continue
        out.append((str(db.segments.index[r]), float(scores[i])))
    out.sort(key=lambda kv: -kv[1])
    return out[:CANDIDATES]


def bm25_scores(
    corpus: list[list[str]], query: list[str], k1: float = 1.5, b: float = 0.75
) -> np.ndarray:
    """Okapi BM25 with the usual +0.5 smoothed IDF, one score per document."""
    n = len(corpus)
    if n == 0 or not query:
        return np.zeros(n, dtype=np.float64)
    lengths = np.array([len(d) for d in corpus], dtype=np.float64)
    avgdl = float(lengths.mean()) or 1.0
    tf = [Counter(d) for d in corpus]
    scores = np.zeros(n, dtype=np.float64)
    denom_len = k1 * (1.0 - b + b * lengths / avgdl)
    for term in set(query):
        df = sum(1 for c in tf if term in c)
        if df == 0:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        f = np.array([c.get(term, 0) for c in tf], dtype=np.float64)
        scores += idf * f * (k1 + 1.0) / (f + denom_len)
    return scores


def ignite_knn(q: QueryState, db: ShotDB) -> list[tuple[str, float]]:
    """Waveform similarity from the IGNITE codec embeddings: which past shots' diagnostics LOOKED
    like the reference shot's, as the foundation model's frozen encoders see them.

    The query is the reference shot's own row (its segment embedding), or, with `ref_window_ms`,
    the mean of its stored 250 ms windows that overlap that interval -- a fragment query. The
    embedding is a concatenation of per-modality blocks (manifest `ignite.dims`) with NaN where a
    shot lacks the modality, so similarity is the mean over the modalities BOTH shots have of the
    per-modality cosine (each block centred on the database mean and L2-normalised, so a 256-dim
    spectro block and a 128-dim Thomson block count once each); a shot sharing no modality with
    the query is not a candidate. Nothing here needs the model: it reads what `ideate build` wrote.
    """
    meta = db.manifest.get("ignite", {})
    emb = db.emb.get("ignite_seg")
    dims = meta.get("dims") or []
    if (
        meta.get("status") != "ok"
        or emb is None
        or emb.shape[0] != len(db.segments)
        or sum(dims) != emb.shape[1]
    ):
        return []
    qvec = _ignite_query(q, db, emb)
    if qvec is None or not np.isfinite(qvec).any():
        return []
    scores = modality_cosine(emb, qvec, dims)
    ok = hard_filter(q, db) & np.isfinite(scores)
    ref = ref_row(q, db)
    if ref is not None:
        ok[db.segments.index.get_loc(ref)] = False  # the query row itself is not an answer
    idx = np.flatnonzero(ok)
    if idx.size == 0:
        return []
    order = idx[np.argsort(-scores[idx], kind="stable")][:CANDIDATES]
    ids = db.segments.index
    return [(str(ids[i]), float(scores[i])) for i in order]


def _ignite_query(q: QueryState, db: ShotDB, emb: np.ndarray) -> np.ndarray | None:
    if q.ref_shot is None:
        return None
    if q.ref_window_ms is not None:
        win = getattr(db, "windows", None)
        wmat = db.emb.get("ignite_win")
        if win is None or wmat is None or wmat.shape[0] != len(win):
            return None
        t0, t1 = q.ref_window_ms
        sel = (win["shot"] == q.ref_shot) & (win["t1_ms"] > t0) & (win["t0_ms"] < t1)
        rows = wmat[sel.to_numpy()].astype(np.float32)
        if rows.shape[0] == 0:
            return None
        finite = np.isfinite(rows)
        n = finite.sum(axis=0)
        return np.where(n > 0, np.where(finite, rows, 0.0).sum(axis=0) / np.maximum(n, 1), np.nan)
    seg_id = ref_row(q, db)
    return None if seg_id is None else emb[db.segments.index.get_loc(seg_id)]


def modality_cosine(M: np.ndarray, qvec: np.ndarray, dims: list[int]) -> np.ndarray:
    """Per row of M, the mean over shared modalities of the per-block cosine to `qvec`; NaN where
    the row and the query share no modality. Each block is centred on the column means of the
    finite rows -- pre-FSQ features carry a common offset that would otherwise push every cosine
    toward 1 -- and L2-normalised, so every modality weighs the same regardless of its width."""
    total = np.zeros(M.shape[0], dtype=np.float64)
    shared = np.zeros(M.shape[0], dtype=np.int64)
    off = 0
    for d in dims:
        block, qb = M[:, off : off + d], qvec[off : off + d]
        off += d
        rows_ok = np.isfinite(block).all(axis=1)
        if not np.isfinite(qb).all() or not rows_ok.any():
            continue
        mu = block[rows_ok].mean(axis=0)
        B = block[rows_ok] - mu
        B /= np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-12)
        qn = qb - mu
        qn /= max(float(np.linalg.norm(qn)), 1e-12)
        total[rows_ok] += B @ qn
        shared[rows_ok] += 1
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(shared > 0, total / np.maximum(shared, 1), np.nan)


def phenomenon(q: QueryState, db: ShotDB) -> list[tuple[str, float]]:
    """Resolved phenomenon evidence, ordered by the same tiers and score as `locate`.

    With several phenomena, the strongest evidence tier leads and the resolver-weighted scores
    break ties. A query with no resolved phenomenon casts no RRF vote.
    """
    from . import phenomena as ph

    positive, negatives = positive_and_negatives(q)
    resolved = ph.resolve(positive)
    if not resolved:
        return []
    weights, saturation, _floor = db._phenomenon_config
    out = []
    for row in db.segments.loc[hard_filter(q, db), ["shot", "segment"]].itertuples():
        if negatives and text_matches_negative(_shot_text(db, int(row.shot)), negatives):
            continue
        evidence = [
            (db.phenomenon_evidence(int(row.shot), pid, row.segment), weight)
            for pid, weight in resolved
        ]
        evidence = [(ev, weight) for ev, weight in evidence if ph.has_evidence(ev)]
        if not evidence:
            continue
        tier = max(ph._tier(ev) for ev, _ in evidence)
        value = sum(weight * ph.score(ev, weights, saturation) for ev, weight in evidence)
        out.append((str(row.Index), value, tier, int(row.shot)))
    out.sort(key=lambda item: (-item[2], -item[1], item[3]))
    return [(sid, value) for sid, value, _tier, _shot in out[:CANDIDATES]]


# The registry. One function, one line here, and rank.py picks it up; weights live in
# configs/ideate/retrieval.yaml under `retrieval.weights` and default to 1.0 for a channel not
# named.
CHANNELS = {
    "scalar_knn": scalar_knn,
    "text_knn": text_knn,
    "bm25": bm25,
    "ignite_knn": ignite_knn,
    "phenomenon": phenomenon,
}
