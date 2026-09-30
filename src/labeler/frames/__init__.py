"""Frame models for the events beside AE (round three, Part B; spec §3.1-§3.3).

One `EventSpec` per event (D36) says what its method reads, learns and must
reach:
- `roles`: the review-store rows it reads, each matched by its title's prefix,
  so a suffix such as ", clipped to its plasma range" does not matter;
- `target`: the labels it learns (`targets` turns each into per-bin states);
- `sub_ms`, `bin_ms` and `pool`: its sub-frames, and the bins its loss is on,
  to which the 10 ms frames' logits are pooled by their maximum (onsets) or
  their mean (states);
- `required_groups`: the corpus or cache groups a shot needs (`raw.record_tier`);
- `baselines` and `bar`: what it is scored against, and the bar (spec §3.4);
- `threshold_rule`: what the threshold is picked on val to maximise (F3): the
  present class's F1 ("f1"), or the mean of both classes' F1 ("macro_f1",
  H-mode's, whose H1 asks for F1(H) and F1(L));
- `balance_crops`: whether training centres half of each shot's crops on
  absent time and half on present (F11): the sawteeth's, whose shots nearly all
  hold some (`labeler.frames.train.crop_windows`).

v2 (F1): the split, its meta, the owner's frozen saves and the features are
versioned. v1's stay where v1 wrote them (`frames/shots/<method>.csv`,
`frames/features/<method>/`); any other version's go under
`frames/shots/<version>/` and `frames/features/<version>/<method>/`. The stores
are not versioned: they hold no targets.

`lmode_frames` has no spec of its own (D38): it is `hmode_frames`' model, read
as 1 - P(H) on the labelled bins (`DERIVED`).

A roster shot reads its review store (`spectrograms/<event>/<shot>.h5`). Every
other shot's store is built under `frames/stores/` (`store_path`), never into
`spectrograms/`, which the review page serves.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from ..config import Paths
from ..events import rosters

VERSION = "v2"
#: The version whose files sit unversioned, where v1 wrote them.
V1 = "v1"
SEED = 20260923


@dataclass(frozen=True)
class Role:
    """A store row a method reads: the row whose title starts with `title`.

    `kind` is "image", "trace" or "modes" (an image row whose bytes are
    `verify.mode_bytes`' codes); an image is pooled to `groups` frequency groups,
    and a modes row gives one level per n, `groups` n. An `optional` role's row
    may be missing. Roles may share a name: each of the sawtooth's four ECE rows
    is an "ece" role.
    """

    name: str
    title: str
    kind: str
    optional: bool = False
    groups: int = 1
    #: The width the role contributes: the row's fixed channel count, or 1 when
    #: pooled; what a missing optional row is zero-filled to.
    channels: int = 1
    #: Channels reduced to one feature set, so their count does not change the width.
    pooled: bool = False


@dataclass(frozen=True)
class EventSpec:
    """One event's frame model (spec §3.1-§3.2).

    `bar` maps each criterion to its conditions `(quantity, op, threshold)`, all
    of which must hold: a bare quantity is the method's own score, and
    `lo(q - baseline)` the lower end of the interval on its difference with a
    baseline's (spec §3.4).
    """

    method: str
    event: str
    store_event: str
    roles: tuple[Role, ...]
    target: str
    sub_ms: float
    bin_ms: float
    pool: str
    required_groups: tuple[str, ...]
    baselines: tuple[str, ...]
    #: Left out of the hash, so a spec can still be a key.
    bar: dict = field(hash=False)
    #: "f1" (the present class's) or "macro_f1" (the mean of F1(H) and F1(L)).
    threshold_rule: str = "f1"
    #: Half of each train shot's crops centred on absent bins, half on present.
    balance_crops: bool = False


SPECS: dict[str, EventSpec] = {
    "elm_frames": EventSpec(
        method="elm_frames",
        event="edge_localized_mode",
        store_event="edge_localized_mode",
        roles=(Role("dalpha", "D-alpha FS", "trace"),),
        target="hiro_onsets",
        sub_ms=2.0,
        bin_ms=50.0,
        pool="max",
        required_groups=("filterscopes",),
        baselines=("elm_onsets", "elm_clock", "always"),
        bar={
            "E1": (
                ("f1", ">=", 0.80),
                ("precision", ">=", 0.70),
                ("recall", ">=", 0.70),
            ),
            "E2": (("lo(f1 - elm_onsets)", ">=", -0.03),),
            "E3": (("lo(f1 - always)", ">", 0.0),),
        },
    ),
    "hmode_frames": EventSpec(
        method="hmode_frames",
        event="high_confinement_mode",
        store_event="high_confinement_mode",
        roles=(
            Role("dalpha", "D-alpha filterscopes", "trace", pooled=True),
            Role("nbi", "NBI power", "trace", optional=True),
        ),
        target="jalal_butt_hl",
        sub_ms=10.0,
        bin_ms=50.0,
        pool="mean",
        required_groups=("filterscopes",),
        baselines=("dalpha_lh", "always"),
        bar={
            "H1": (("f1(H)", ">=", 0.95), ("f1(L)", ">=", 0.70)),
            "H2": (("lo(f1(H) - always)", ">", 0.0),),
        },
        threshold_rule="macro_f1",
    ),
    "ntm_frames": EventSpec(
        method="ntm_frames",
        event="neoclassical_tearing_mode",
        store_event="neoclassical_tearing_mode",
        roles=(
            Role("power", "MPI66M322D power", "image", groups=32),
            Role("modes", "toroidal n, MPI66M probes", "modes", groups=10),
        ),
        target="tearing_archive",
        sub_ms=5.0,
        bin_ms=50.0,
        pool="mean",
        required_groups=("mirnov",),
        baselines=("always",),
        bar={
            "N1": (
                ("f1", ">=", 0.70),
                ("precision", ">=", 0.60),
                ("recall", ">=", 0.60),
            ),
            "N2": (("lo(f1 - always)", ">", 0.0),),
        },
    ),
    "sawtooth_frames": EventSpec(
        method="sawtooth_frames",
        event="sawtooth_oscillation",
        store_event="sawtooth_oscillation",
        roles=(
            Role("ece", "ECE Te, ch 20-23", "trace", channels=4),
            Role("ece", "ECE Te, ch 24-27", "trace", channels=4),
            Role("ece", "ECE Te, ch 28-31", "trace", channels=4),
            Role("ece", "ECE Te, ch 32-35", "trace", channels=4),
            Role("sxr", "SXR", "trace", optional=True, channels=4),
        ),
        target="ece_sawtooth_v3",
        sub_ms=2.0,
        bin_ms=10.0,
        pool="mean",
        required_groups=("ece",),
        baselines=("always",),
        bar={
            "S1": (("f1", ">=", 0.85),),
            "S2": (("lo(f1 - always)", ">", 0.0),),
        },
        balance_crops=True,
    ),
}
#: Methods with no model of their own: `(the method whose model they read, the
#: event they label)` (D38).
DERIVED = {"lmode_frames": ("hmode_frames", "low_confinement_mode")}


def _frames(paths: Paths) -> Path:
    return paths.root / "frames"


def _shots(paths: Paths, version: str) -> Path:
    base = _frames(paths) / "shots"
    return base if version == V1 else base / version


def shots_file(paths: Paths, method: str, version: str = VERSION) -> Path:
    """The method's shots and their split."""
    return _shots(paths, version) / f"{method}.csv"


def shots_meta_file(paths: Paths, method: str, version: str = VERSION) -> Path:
    """The split's metadata, D60's `labelled_shots` among it."""
    return _shots(paths, version) / f"{method}.json"


def owner_file(paths: Paths, method: str, version: str = VERSION) -> Path:
    """The owner's saves as the split froze them: v1's `owner` split (D40), and
    from v2 the labels laid over the original's (F2)."""
    return _shots(paths, version) / f"{method}.owner.csv"


def grid_path(paths: Paths, event: str, shot: int) -> Path:
    """A format grid: `<label tables>/<event>/format/shots/<shot>.npz`."""
    return paths.label_tables / event / "format" / "shots" / f"{int(shot)}.npz"


def roster_shots(paths: Paths, event: str) -> frozenset[int]:
    """The shots of `event`'s review roster, the label tables' `<event>/shots.csv`;
    none without one."""
    path = rosters.roster_path(event, root=paths.label_tables)
    if not path.is_file():
        return frozenset()
    stat = path.stat()
    return _roster_shots(path, stat.st_mtime_ns, stat.st_ino, stat.st_size)


@lru_cache(maxsize=16)
def _roster_shots(path, _mtime_ns, _ino, _size) -> frozenset[int]:
    """Cached per file version, as `review.labels` caches its tables."""
    return frozenset(int(shot) for shot in rosters.read_roster(path).shot)


def stores_dir(paths: Paths, spec: EventSpec) -> Path:
    """Where the stores of the shots outside the roster are built."""
    return _frames(paths) / "stores" / spec.store_event


def store_path(paths: Paths, spec: EventSpec, shot: int) -> Path:
    """A roster shot's review store, else the one built for this package."""
    if int(shot) in roster_shots(paths, spec.store_event):
        return paths.spectrogram_file(spec.store_event, shot)
    return stores_dir(paths, spec) / f"{int(shot)}.h5"


def features_dir(paths: Paths, method: str, version: str = VERSION) -> Path:
    """Each shot's features, as the model reads them."""
    base = _frames(paths) / "features"
    return base / method if version == V1 else base / version / method


def model_dir(paths: Paths, method: str, version: str = VERSION) -> Path:
    """The trained model, its split and its scores."""
    return paths.models / method / version


def summary_file(paths: Paths, method: str, version: str = VERSION) -> Path:
    """The extension's summary, beside its suggestion table."""
    return paths.root / "suggestions" / method / version / "summary.csv"


def gallery_dir(paths: Paths, spec: EventSpec, version: str = VERSION) -> Path:
    """One picture per shot."""
    return paths.root / "gallery" / spec.event / f"{spec.method}-{version}"
