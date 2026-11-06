"""Read a few MDSplus nodes of a DIII-D shot from FDP without downloading whole tree files.

FDP serves each shot's MDSplus trees as files (``<view>/mdsplus/shots/<tree>_<shot>.{tree,
characteristics,datafile}``), and its OSDF caches answer HTTP byte ranges. The ``.tree`` and
``.characteristics`` files (under 1 MB) say where each node's record lies in the
``.datafile``. This module fetches those two files and only the record bytes the wanted nodes
need into a sparse local datafile of the full size. MDSplus then reads the nodes as if the
whole tree were on disk.

A record of more than one 32 kB chunk lies before its ``rfa`` (the NCI's record file address
points to the last chunk; strace on 185786, 2026-11-03): the span fetched is
``[rfa - rlength - rlength / 1000 - 4096, rfa + 65536)``. Nodes a record refers to (an
expression such as ``DENV2F = scale * (PL1V2F - PL2V2F / lamrat)``) are fetched in further
rounds. A segmented node, or a read that still fails, falls back to the whole datafile.

Runs inside ``fdp -D d3d run`` (login node), which sets ``BEARER_TOKEN``. The token is sent
only to the FDP cache and never logged.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

# The OSDF director's caches for fdp-d3d (2026-11-03). On that afternoon osdf1.chic, then the first, sent
# headers and then nothing for 30 s on every request while the others served the same files in under a
# second, so a failed attempt moves this process on to the next cache. On 2026-11-06 it still took 23 s
# for a 73 kB file that dtn-pas.cinc served in 0.1 s, so it is now the last.
CACHES = (
    "https://dtn-pas.cinc.nrp.internet2.edu:8443",
    "https://osdf1.newy32aoa.nrp.internet2.edu:8443",
    "https://osdf-cache-01.gwave.ics.psu.edu:8443",
    "https://mghpcc-cache.nationalresearchplatform.org:8443",
    "https://osdf1.chic.nrp.internet2.edu:8443",
)
_cache = [0]  # index of this process's current cache
VIEW = "/fdp-d3d/views/shots/{prefix}/{shot}/v1/mdsplus/shots"
#: EFIT and the other code trees are not in the shot views but in FDP's code archive, in folders
#: named by the shot number's digit pairs (fdp's default_tree_path ``codes/~t/~j~i/~h~g/~f~e/~d~c``:
#: efit01_185704 is ``codes/efit01/00/00/18/57/``).
CODES = "/fdp-d3d/archives/mdsplus/codes/{tree}/{d}"
CODE_TREE = re.compile(r"(efit\w*|zipfit\d+)")
BEFORE_SLACK, AFTER = 4096, 65536
ROUNDS = 6
PARALLEL = 6  # range requests in flight for one shot
ATTEMPTS = 6  # 42 s of backoff: the login node loses its route for ~20 s at times (errno 101)
SMALL_RECORD = 4096  # bytes: an expression, not samples


class AuthError(RuntimeError):
    pass


class Missing(RuntimeError):
    """The view has no such tree file (HTTP 404)."""


def _url(shot: int, name: str) -> str:
    """The file's path on the cache; ``_get`` puts the current cache in front."""
    tree = name.rsplit("_", 1)[0].lower()
    if CODE_TREE.fullmatch(tree):
        d = f"{shot:010d}"
        return CODES.format(tree=tree, d="/".join(d[i:i + 2] for i in range(0, 8, 2))) + "/" + name
    return VIEW.format(prefix=shot // 100, shot=shot) + "/" + name


def _get(url: str, rng: tuple[int, int] | None = None, timeout: int = 120) -> tuple[bytes, int | None]:
    """The body and, for a range request, the file's total size (Content-Range)."""
    headers = {"Authorization": f"Bearer {os.environ['BEARER_TOKEN']}"}
    if rng is not None:
        headers["Range"] = f"bytes={rng[0]}-{rng[1] - 1}"
    err: Exception | None = None
    for attempt in range(ATTEMPTS):
        used = _cache[0]
        try:
            req = urllib.request.Request(CACHES[used] + url, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                body = r.read()
                total = None
                cr = r.headers.get("Content-Range")
                if cr:
                    total = int(cr.rsplit("/", 1)[1])
                if rng is not None and r.status != 206:
                    raise RuntimeError(f"no range support: HTTP {r.status}")
                return body, total
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise AuthError(f"HTTP {e.code} for {url.rsplit('/', 1)[1]}") from None
            if e.code == 404:
                raise Missing(url.rsplit("/", 1)[1]) from None
            if e.code == 416:  # an empty datafile
                return b"", 0
            err = e
        except Exception as e:  # noqa: BLE001 - network faults are retried
            err = e
        if _cache[0] == used:  # the shot's other threads may have moved on already
            _cache[0] = (used + 1) % len(CACHES)
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"{url.rsplit('/', 1)[1]}: {type(err).__name__}: {err}"[:300])


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[list[int]] = []
    for a, b in sorted(spans):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return [(a, b) for a, b in out]


class SparseTrees:
    """The shot's tree files under ``dest`` (one folder per shot), filled as nodes need them."""

    def __init__(self, shot: int, dest: Path):
        self.shot, self.dest = shot, Path(dest)
        self.dest.mkdir(parents=True, exist_ok=True)
        self.size: dict[str, int] = {}
        self.have: dict[str, list[tuple[int, int]]] = {}
        self.whole: set[str] = set()
        self.bytes = 0
        self.missing: dict[str, str] = {}  # what every stage_nodes call on these files found
        self.ptdata: list[str] = []

    def write_info(self) -> None:
        """``staged.json``, which ``read_nodes`` reads: the trees fetched so far and the findings."""
        (self.dest / "staged.json").write_text(
            json.dumps({"shot": self.shot, "bytes": self.bytes, "whole": sorted(self.whole),
                        "trees": sorted(self.size), "missing": self.missing, "ptdata": self.ptdata}))

    def _file(self, tree: str, ext: str) -> Path:
        return self.dest / f"{tree}_{self.shot}.{ext}"

    def open_files(self, tree: str) -> None:
        """The tree's .tree and .characteristics, and an empty sparse datafile of the right size."""
        tree = tree.lower()
        if tree in self.size:
            return
        for ext in ("tree", "characteristics"):
            f = self._file(tree, ext)
            if not f.exists():
                body, _ = _get(_url(self.shot, f.name))
                f.write_bytes(body)
                self.bytes += len(body)
        data = self._file(tree, "datafile")
        _, total = _get(_url(self.shot, data.name), (0, 1))
        total = total or 0
        with open(data, "wb") as g:
            g.truncate(total)
        self.size[tree], self.have[tree] = total, []
        os.environ[f"{tree}_path"] = str(self.dest)

    def fill(self, tree: str, spans: list[tuple[int, int]]) -> None:
        tree = tree.lower()
        if tree in self.whole:
            return
        size = self.size[tree]
        spans = [(max(0, a), min(size, b)) for a, b in spans if min(size, b) > max(0, a)]
        todo = []
        for a, b in _merge(spans):
            if not any(x <= a and b <= y for x, y in self.have[tree]):
                todo.append((a, b))
        if not todo:
            return
        url = _url(self.shot, self._file(tree, "datafile").name)
        with ThreadPoolExecutor(min(len(todo), PARALLEL)) as pool:
            bodies = list(pool.map(lambda ab: _get(url, ab)[0], todo))
        with open(self._file(tree, "datafile"), "r+b") as g:
            for (a, _b), body in zip(todo, bodies, strict=True):
                g.seek(a)
                g.write(body)
                self.bytes += len(body)
        self.have[tree] = _merge(self.have[tree] + todo)

    def fill_whole(self, tree: str) -> None:
        tree = tree.lower()
        if tree in self.whole:
            return
        f = self._file(tree, "datafile")
        body, _ = _get(_url(self.shot, f.name), timeout=1800)
        f.write_bytes(body)
        self.bytes += len(body)
        self.whole.add(tree)


def _span(node) -> tuple[int, int] | None:
    rlen = int(node.rlength)
    if rlen <= 0:
        return None  # no data, or held in the NCI itself
    rfa = int(node.rfa)
    return rfa - rlen - rlen // 1000 - BEFORE_SLACK, rfa + AFTER


def _refs(d, seen: set[int]):
    """Tree nodes a record names, without evaluating it."""
    import MDSplus

    if d is None or id(d) in seen:
        return
    seen.add(id(d))
    if isinstance(d, (MDSplus.TreeNode, MDSplus.TreePath)):
        yield d
        return
    if isinstance(d, MDSplus.Compound):
        for a in d.getDescs() if hasattr(d, "getDescs") else d.args:
            yield from _refs(a, seen)
    elif isinstance(d, (MDSplus.List, MDSplus.Apd)):
        for a in d:
            yield from _refs(a, seen)


_SUBTREE = re.compile(r"\\([A-Za-z_][A-Za-z0-9_]*)::")


def stage_nodes(shot: int, dest: Path, tags: dict[str, list[str]], whole: bool = False,
                trees: SparseTrees | None = None) -> dict:
    """Fetch what reading ``tags`` needs into ``dest``; nothing is decoded but the small
    records whose references are followed. ``whole`` fetches whole datafiles instead (the
    fallback for a shot whose sparse read failed). Returns ``{"bytes", "whole", "missing",
    "ptdata"}``: ``missing`` maps a tag to why it cannot be read (no tree file, no such node);
    ``ptdata`` lists the referenced nodes whose record is a ``PTDATA`` call (before ~2010 the
    filterscopes are ``PTDATA2("PHDnn", $SHOT)``), which these files cannot serve. Raises
    ``AuthError`` on a refused token.

    ``trees``, the ``SparseTrees`` of an earlier call on the same ``dest``, adds nodes to the
    files fetched so far (a new one would truncate the datafiles); ``staged.json`` then holds
    every call's trees and findings, and ``missing`` and ``ptdata`` this call's."""
    import MDSplus

    st = trees if trees is not None else SparseTrees(shot, dest)
    missing: dict[str, str] = {}
    ptdata: list[str] = []
    for tree, names in tags.items():
        try:
            st.open_files(tree)
        except Missing as e:
            for n in names:
                missing[n] = f"no tree file {e}"
            continue
        T = _open(st, tree, shot)
        queue = []
        for n in names:
            try:
                queue.append(T.getNode(n))
            except Exception as e:  # noqa: BLE001 - a tag this era does not have
                missing[n] = f"no node: {type(e).__name__}"
        if whole:
            for node in queue:
                st.fill_whole(node.tree.name.lower())
        done: set[str] = set()
        for _ in range(ROUNDS):
            batch, queue = [n for n in queue if n.getFullPath() not in done], []
            if not batch:
                break
            per_tree: dict[str, list] = {}
            for n in batch:
                done.add(n.getFullPath())
                tname = n.tree.name.lower()
                st.open_files(tname)
                if n.getNumSegments() > 0:
                    st.fill_whole(tname)
                    continue
                sp = _span(n)
                if sp:
                    per_tree.setdefault(tname, []).append(sp)
            for tname, spans in per_tree.items():
                st.fill(tname, spans)
            for n in batch:
                try:
                    rec = n.getRecord()
                except Exception:  # noqa: BLE001 - an empty node
                    continue
                # an expression record is short; a PTDATA call in it (old shots, alone or inside
                # an expression such as PTDATA2("PHD02", $SHOT) / \FS02:GAIN_PC) means the
                # samples are in PTDATA files, not this tree
                text = str(rec) if 0 < int(n.rlength) < SMALL_RECORD else ""
                if "PTDATA" in text.upper():  # its other references (calibrations) are still fetched
                    ptdata.append(f"{n.getFullPath()}: {text[:80]}")
                for r in _refs(rec, set()):
                    try:
                        queue.append(r if isinstance(r, MDSplus.TreeNode) else T.getNode(str(r)))
                    except Exception:  # noqa: BLE001, S112 - a name that is not a node
                        continue
    st.missing.update(missing)
    st.ptdata += ptdata
    st.write_info()
    return {"bytes": st.bytes, "whole": sorted(st.whole), "missing": missing, "ptdata": ptdata}


def _open(st: SparseTrees, tree: str, shot: int):
    """The tree, fetching the subtrees it names as their absence is reported."""
    import MDSplus

    for _ in range(4):
        try:
            return MDSplus.Tree(tree, shot, "readonly")
        except MDSplus.mdsExceptions.TreeException as e:
            found = [s.lower() for s in _SUBTREE.findall(str(e))]
            if not found:
                raise
            for s in found:
                st.open_files(s)
    return MDSplus.Tree(tree, shot, "readonly")


def read_nodes(shot: int, dest: Path, tags: dict[str, list[str]]) -> dict[str, tuple]:
    """``{tag: (data, dim)}`` from a staged folder, no network; a tag that cannot be read maps
    to ``(reason,)``."""
    import MDSplus
    import numpy as np

    dest = Path(dest)
    info = json.loads((dest / "staged.json").read_text())
    for t in info["trees"]:
        os.environ[f"{t}_path"] = str(dest)
    out: dict[str, tuple] = {}
    for tree, names in tags.items():
        if tree not in info["trees"]:
            for n in names:
                out[n] = (info["missing"].get(n, f"no tree file {tree}"),)
            continue
        try:
            T = MDSplus.Tree(tree, shot, "readonly")
        except Exception as e:  # noqa: BLE001
            for n in names:
                out[n] = (f"tree open failed: {type(e).__name__}: {e}"[:200],)
            continue
        for n in names:
            if n in info["missing"]:
                out[n] = (info["missing"][n],)
                continue
            try:
                node = T.getNode(n)
                out[n] = (np.asarray(node.data()), np.asarray(node.dim_of().data()))
            except Exception as e:  # noqa: BLE001
                out[n] = (f"read failed: {type(e).__name__}: {e}"[:200],)
    return out
