"""The 14 TokaMark tasks as typed specs, the benchmark's shot splits and the published scores.

Everything that comes from TokaMark is read from TokaMark's own configuration, so the
specs cannot drift from the benchmark. `tokamark` is imported inside the functions that
need it: this module imports without it (the pixi environment has none), and
`PUBLISHED`, `ALL_TASKS`, the dataclasses and `best_published` never need it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

ALL_TASKS: tuple[str, ...] = tuple(
    f"task_{g}-{i}" for g, n in ((1, 3), (2, 3), (3, 3), (4, 5)) for i in range(1, n + 1)
)
PROBE_TASKS = ("task_1-1", "task_1-2", "task_2-1", "task_3-1", "task_4-1", "task_4-4")
STAGE0_TASKS = PROBE_TASKS + ("task_4-5",)
SPECTROGRAM_SIGNALS = (
    "magnetics-b_field_tor_probe_cc_field",
    "magnetics-b_field_pol_probe_omv_voltage",
)

# Published task NRMSE (lower is better), measured on TokaMark's random-split test shots.
# "mean", "persistence", "tokamind_paper": /models/tokamind/compare_to_paper.py, PAPER[task]
#   elements 0 (Mean), 1 (Persistence) and 6 (FT-Base); None = not reported there.
# "tokamind_rerun", "gentoka": the table in docs/models/ignite-mast-design.md, columns
#   "TokaMind re-run by GenToka" and "GenToka". Copied on 2026-11-22.
_PUBLISHED_ROWS = {
    # task: (mean, persistence, tokamind_paper, tokamind_rerun, gentoka)
    "task_1-1": (0.757, None, 0.150, 0.1363, 0.0941),
    "task_1-2": (0.998, None, 0.043, 0.0439, 0.0412),
    "task_1-3": (0.927, None, 0.122, 0.1200, 0.1156),
    "task_2-1": (0.817, None, 0.181, 0.1289, 0.0880),
    "task_2-2": (0.991, None, 0.043, 0.0408, 0.0370),
    "task_2-3": (0.923, None, 0.092, 0.0866, 0.0828),
    "task_3-1": (0.984, 0.536, 0.346, 0.3477, 0.3311),
    "task_3-2": (0.630, 0.365, 0.269, 0.2529, 0.2522),
    "task_3-3": (0.841, None, 0.220, 0.2415, 0.2147),
    "task_4-1": (0.511, 0.325, 0.213, 0.2122, 0.2298),
    "task_4-2": (0.511, 0.325, 0.214, 0.2143, 0.2302),
    "task_4-3": (0.731, None, 0.080, 0.0781, 0.0691),
    "task_4-4": (1.027, 0.718, 0.349, 0.3470, 0.3116),
    "task_4-5": (0.653, 0.870, 0.651, 0.7377, 0.7487),
}
_PUBLISHED_COLUMNS = ("mean", "persistence", "tokamind_paper", "tokamind_rerun", "gentoka")
PUBLISHED: dict[str, dict[str, float | None]] = {
    t: dict(zip(_PUBLISHED_COLUMNS, row)) for t, row in _PUBLISHED_ROWS.items()
}

_ROLES = ("input", "actuator", "output")


@dataclass(frozen=True)
class SignalSpec:
    name: str                      # "<source>-<signal>"
    dt: float                      # seconds
    values_shape: tuple[int, ...]  # without time
    mean: float                    # TokaMark's statistics file
    std: float

    def standardize(self, x):
        """TokaMark-standardized units. Plain arithmetic: numpy arrays and torch tensors alike."""
        return (x - self.mean) / self.std

    def unstandardize(self, z):
        return z * self.std + self.mean


@dataclass(frozen=True)
class TaskSpec:
    name: str
    markovian: bool
    input_length: float  # seconds
    output_length: float
    delta: float
    stride: float
    inputs: tuple[SignalSpec, ...]
    actuators: tuple[SignalSpec, ...]
    outputs: tuple[SignalSpec, ...]
    # role ("input" | "actuator" | "output") -> signal name -> time stamps (TokaMark's
    # `length_in_time_stamps`); a hand-built spec passes them in.
    steps: Mapping[str, Mapping[str, int]]

    def n_out(self, s: SignalSpec) -> int:
        return self.steps["output"][s.name]

    def n_act(self, s: SignalSpec) -> int:
        return self.steps["actuator"][s.name]

    def n_in(self, s: SignalSpec, history_s: float | None = None) -> int:
        """Markovian tasks: TokaMark's count. Non-markovian: the history in use, in samples."""
        if self.markovian:
            return self.steps["input"][s.name]
        return round((history_s or self.input_length) / s.dt)


def load_task(name: str) -> TaskSpec:
    """Build the spec from TokaMark's task configuration, task metadata and statistics."""
    from tokamark.tasks import get_task_config, get_task_metadata

    config = get_task_config(name)
    meta = get_task_metadata(config)  # dt, values_shape, mean, std, length_in_time_stamps per signal
    window = config["task_window_segmenter"]

    def specs(role: str) -> tuple[SignalSpec, ...]:
        return tuple(
            SignalSpec(key, float(m["dt"]), tuple(int(d) for d in m["values_shape"]),
                       float(m["mean"]), float(m["std"]))
            for key, m in meta[role].items()
        )

    return TaskSpec(
        name=name,
        markovian=config["task_type"] == "markovian",
        input_length=float(window["input_length"]),
        output_length=float(window["output_length"]),
        delta=float(window["delta"]),
        stride=float(config["stride_window"]),
        inputs=specs("input"), actuators=specs("actuator"), outputs=specs("output"),
        steps={r: {k: int(m["length_in_time_stamps"]) for k, m in meta[r].items()} for r in _ROLES},
    )


def split_shots(split: str) -> list[int]:
    """Sorted shot numbers of a split of TokaMark's random-split file.

    "val_select" is the validation shots at even positions of the sorted list, "val_report"
    those at odd positions: models are selected on the first and reported on the second, so
    no reported number was selected on its own shots.
    """
    if split not in ("train", "val", "test", "val_select", "val_report"):
        raise ValueError(f"unknown split {split!r}")
    from tokamark.data_split import get_train_test_val_shots

    train, test, val = get_train_test_val_shots()  # TokaMark returns train, test, val
    if split == "train":
        return sorted(train)
    if split == "test":
        return sorted(test)
    val = sorted(val)
    return {"val": val, "val_select": val[0::2], "val_report": val[1::2]}[split]


def best_published(task: str) -> float:
    """The lowest of the TokaMind paper, the TokaMind re-run and GenToka scores."""
    row = PUBLISHED[task]
    return min(row[k] for k in ("tokamind_paper", "tokamind_rerun", "gentoka"))
