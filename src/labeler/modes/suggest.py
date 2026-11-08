"""The `mhd_modes` suggestion tables from omnimode's port-v6 labels.

The same files `labeler.events.candidates --method mhd_modes` writes for v5: one table an
event, `shot, category, t_start, t_end, confidence` (ms; 0 absent, 1 present, 2 uncertain,
3 not observable, each shot's rows tiling its window), beside a `.meta.json`, at
`<root>/suggestions/mhd_modes/<version>/<event>_suggest_mhd_modes_<version>.csv`. The review
page opens on one through `review/source.json` (`point`). What differs from v5 is the source
(omnimode's port-v6 tracks, not the FusionAIHub cache), the owner's rules (`labeler.modes.tracks`:
harmonics dropped, |n| 1-3) and the subcategories: every class a labelled mode carries, with
the five v5 children kept so no existing subcategory pointer is left without a table.

A version folder is never written into twice (`write` refuses one that holds a table), as
`candidates` refuses an existing version.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from labeler.config import atomic_path
from labeler.events import suggestions
from labeler.events.review import labels as review_labels
from labeler.mhd import rows

from . import cells as C
from . import tracks as T

METHOD = "mhd_modes"
VERSION = "v6port"
#: v5's subcategories, kept in every version so a pointer at one never dangles.
V5_CLASSES = ("1/1", "2/1", "3/1", "3/2", "4/3")
CATEGORIES = {0: "absent", 1: "present", 2: "uncertain", 3: "not_observable"}
#: Why a listed shot has no rows (`load`'s `skipped`): no files, a failed run, no window.
NO_LABELS = "no labels computed (no port-v6 files)"
FAILED = "labeller failed"
NO_WINDOW = "no window"


def table_file(root, event: str, version: str = VERSION) -> Path:
    name = f"{event}_suggest_{METHOD}_{version}.csv"
    return Path(root) / "suggestions" / METHOD / version / name


def git_sha(repo: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        ).stdout.strip()
    except OSError:
        return None
    return out or None


def load(labels_dir, shots) -> tuple[list, dict[int, str], dict]:
    """Read every listed shot; `(ready, skipped, dropped)`.

    `ready` is `[(shot, rule-applied Shot)]` for the shots with files and a window; `skipped`
    maps the others to why (`NO_LABELS`, `FAILED`, `NO_WINDOW`); `dropped` is `{shot: apply_rules counts}`.
    A shot is read only when it is listed: the caller's list is the seal."""
    ready, skipped, dropped = [], {}, {}
    for shot in sorted({int(s) for s in shots}):
        got = T.read_shot(labels_dir, shot)
        if got is None:
            skipped[shot] = NO_LABELS
        elif got.info.get("model_error") or str(got.info.get("status", "")).startswith(
            "read error"
        ):
            skipped[shot] = (
                f"{FAILED}: {got.info.get('model_error') or got.info.get('status')}"
            )
        elif not got.info.get("window"):
            skipped[shot] = f"{NO_WINDOW}: {got.info.get('status')}"
        else:
            ruled, counts = T.apply_rules(got)
            ready.append((shot, ruled))
            dropped[shot] = counts
    return ready, skipped, dropped


def subcategories(ready) -> dict[str, str]:
    """`{event: class}`: the classes the ready shots' labelled modes carry, and v5's five."""
    names = set(V5_CLASSES)
    for _, got in ready:
        names |= set(T.classes(got))
    return C.children(names)


def tables(ready, kids: dict[str, str]) -> dict[str, list]:
    """Each event's rows (`[shot, category, t_start, t_end, confidence]`) over the ready shots."""
    out: dict[str, list] = {event: [] for event in (*C.PARENTS, *kids)}
    for shot, got in ready:
        for event, cells in C.shot_cells(got, kids).items():
            out[event] += cells.rows(shot)
    return out


def rule_text(event: str, kids: dict[str, str]) -> dict:
    """What the table's states mean, as v5's sidecar says it (its `rule` block)."""
    common = {
        "equilibrium": (
            "the shot's own tree in omnimode's info record (EFIT02ER first, else EFIT02 or "
            "EFIT01); each track's resonance gate used its slices"
        ),
        "harmonics": "tracks with harm true are dropped before any state is set (owner rule 1)",
        "scope": "|n| = 1, 2, 3 (the port emits no other n; a later redesign adds n>3 and n=0)",
        "span": "frames padded +-2 ms, merged across gaps of up to 20 ms, spans under 20 ms dropped",
        "crash": "labeler.mhd.rows' crash table and q = 1 gate are not applied (not omnimode's)",
    }
    if event in C.PARENTS:
        return {
            "n": C.PARENTS[event],
            "present": (
                f"every track with n = {C.PARENTS[event]} but fixed-frequency lines and "
                "harmonics, labelled or not (confidence the largest track coherence in the span)"
            ),
            "not_observable": (
                "outside the Mirnov record's finite stretch and over its NaN gaps; every event "
                "on a shot whose Mirnov cannot be read"
            ),
            "absent": "the rest",
            **common,
        }
    name = kids[event]
    return {
        "m/n": name,
        "present": (
            f"tracks with resonance status m/n and label {name} (confidence P(resonant)), "
            "inside the parent's present time"
        ),
        "uncertain": (
            f"tracks whose status is uncertain, no rotation or q_min above and whose 90% "
            f"credible set holds {name} (confidence its posterior); on an EFIT01 shot also "
            f"q_min-above tracks whose candidate is {name}"
        ),
        "not_observable": (
            "the parent's present time on a shot with no equilibrium tree, or with no slice "
            "within 30 ms, or over tracks the model could not run on; and the parent's "
            "not-observable time"
        ),
        "absent": "the rest",
        **common,
    }


def sidecar(
    event, kids, ready, skipped, dropped, frame, *, labels_dir, version, repo
) -> dict:
    """The table's `.meta.json` (the fields of v5's, with the port's own in place of the cache)."""
    infos = [got.info for _, got in ready]
    freeze = Counter(str(i.get("freeze")) for i in infos)
    trees = Counter(str((i.get("inputs") or {}).get("tree")) for i in infos)
    states = {
        name: int(((frame.category == code) * (frame.t_end - frame.t_start)).sum())
        for code, name in CATEGORIES.items()
    }
    return {
        "event": event,
        "method": METHOD,
        "version": version,
        "made_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "made_by": "scripts/labeler/mode_fdp_label.py suggest",
        "git_sha": git_sha(repo),
        "categories": {str(k): v for k, v in CATEGORIES.items()},
        "rule": rule_text(event, kids),
        "source": {
            "labels": "omnimode port-v6 (<shot>.json, .tracks.parquet, .frames.parquet)",
            "port_of": "labeler.mhd v6 (omnimode's port; same label model and rules)",
            "labels_dir": str(labels_dir),
            "freeze": dict(freeze),
            "harmonics": "hidden (tracks with harm true are not in any table)",
            "n_scope": "|n| = 1, 2, 3",
            "status": "proposed, not checked; to be superseded by the labeller redesign",
        },
        "totals_ms": states,
        "trees": dict(trees),
        "status_counts": dict(Counter(str(i.get("status")) for i in infos)),
        "skipped": {str(k): v for k, v in sorted(skipped.items())},
        "per_shot": {
            str(shot): {
                "window": got.info.get("window"),
                "tree": (got.info.get("inputs") or {}).get("tree"),
                "tracks_kept": dropped[shot]["kept"],
                "harmonics_dropped": dropped[shot]["harmonics"],
                "out_of_scope_dropped": dropped[shot]["out_of_scope"],
            }
            for shot, got in ready
        },
    }


def outside_taxonomy(kids: dict[str, str], taxonomy) -> list[str]:
    """The subcategories of `kids` that `taxonomy` (a `labeler.events.taxonomy.Taxonomy`) does
    not name: the review page would not offer them, so they need an `events.yaml` entry."""
    return [event for event in kids if not taxonomy.is_child(event)]


def write(
    labels_dir,
    shots,
    root,
    *,
    version: str = VERSION,
    repo: Path,
    taxonomy=None,
) -> dict:
    """Write every event's table and sidecar under `root`; returns the summary.

    Refuses when the version folder already holds a table. `shots` is the caller's list.
    With `taxonomy` given, a class the labelled modes carry that it does not name is refused
    before anything is written (`ValueError`, naming the events)."""
    folder = Path(root) / "suggestions" / METHOD / version
    held = (
        sorted(folder.glob(f"*_suggest_{METHOD}_{version}.csv"))
        if folder.is_dir()
        else []
    )
    if held:
        raise FileExistsError(
            f"{folder} already holds {len(held)} tables; a version is never written into twice"
        )
    ready, skipped, dropped = load(labels_dir, shots)
    kids = subcategories(ready)
    if taxonomy is not None and (unnamed := outside_taxonomy(kids, taxonomy)):
        raise ValueError(
            f"subcategories {unnamed} are not in events.yaml's taxonomy: add them first"
        )
    summary = {"events": {}, "shots": len(ready), "skipped": skipped, "classes": kids}
    for event, record in tables(ready, kids).items():
        path = table_file(root, event, version)
        frame = suggestions.write_table(path, record, {})  # validated, atomic, sorted
        meta = sidecar(
            event,
            kids,
            ready,
            skipped,
            dropped,
            frame,
            labels_dir=labels_dir,
            version=version,
            repo=repo,
        )
        meta |= {"rows": len(frame), "shots": int(frame.shot.nunique())}
        with atomic_path(path.with_suffix(".meta.json")) as tmp:
            tmp.write_text(json.dumps(meta, indent=1) + "\n")
        summary["events"][event] = {
            "path": str(path),
            "rows": len(frame),
            "present_runs": int((frame.category == rows.PRESENT).sum()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    return summary


def point(events_dir, root, kids: dict[str, str], *, version: str = VERSION) -> dict:
    """Point each event's review page at its table (`labels.write_pointer`, FusionAIHub's own
    mechanism, which writes `review/source.json` with `set_at` and `set_by`): the three parents
    at `<events>/<event>/review/` and each subcategory at `<events>/<parent>/subcategories/<m>/
    review/`, a subcategory folder made when it is missing. Returns each pointer's entry."""
    entries = {}
    for event in (*C.PARENTS, *kids):
        parent, _, key = event.partition(".")
        folder = Path(events_dir) / parent / ("subcategories" if key else "") / key
        entries[event] = review_labels.write_pointer(
            folder, table_file(root, event, version), method=METHOD, version=version
        )
    return entries
