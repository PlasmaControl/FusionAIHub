"""The 14 TokaMark tasks as typed specs, the benchmark's shot splits and the published scores.

Everything that comes from TokaMark is read from TokaMark's own configuration, so the
specs cannot drift from the benchmark. `tokamark` is imported inside the functions that
need it: this module imports without it (the pixi environment has none), and
`PUBLISHED`, `ALL_TASKS`, the dataclasses and `best_published` never need it.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

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
    # Which windows the task is scored on: "task" (the task's own) or "persistence_reference"
    # (the windows of TokaMark's persistence script, see `persistence_reference`).
    windows: str = "task"

    def n_out(self, s: SignalSpec) -> int:
        return self.steps["output"][s.name]

    def n_act(self, s: SignalSpec, history_s: float | None = None) -> int:
        """Actuator samples: the input history plus the future part (delta + output length).

        For a non-markovian task `history_s` replaces the input length of the history part,
        like `n_in`; the future part does not change.
        """
        n = self.steps["actuator"][s.name]
        if self.markovian or history_s is None:
            return n
        return n - round(self.input_length / s.dt) + round(history_s / s.dt)

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

    inputs, actuators, outputs = specs("input"), specs("actuator"), specs("output")
    input_length, output_length, delta = (float(window[k]) for k in ("input_length", "output_length", "delta"))

    def length(role: str, s: SignalSpec) -> int:
        """Samples of a window of this role, TokaMark's formulas (`TokaMarkDataset._build_window`)."""
        n_in, n_out, n_delta = (round(x / s.dt) for x in (input_length, output_length, delta))
        return {"input": n_in, "actuator": n_in + n_delta + n_out, "output": n_out}[role]

    # `get_task_metadata` hands the same dictionary to every role of a signal, so the last role
    # that lists the signal overwrites `length_in_time_stamps` of the others: the input length of
    # a signal that is also an output holds the output length (found 2026-11-22: 15 inputs in 7
    # tasks, e.g. task_3-1 thomson_scattering-t_e input 10 instead of 1). The steps are therefore
    # computed from seconds and dt; TokaMark's number is only used as a check where it is reliable,
    # so that a change of its rule is noticed.
    steps: dict[str, dict[str, int]] = {}
    for role, specs_ in (("input", inputs), ("actuator", actuators), ("output", outputs)):
        steps[role] = {}
        for s in specs_:
            steps[role][s.name] = length(role, s)
            last_role = [r for r in _ROLES if s.name in meta[r]][-1]
            if role == last_role and meta[role][s.name]["length_in_time_stamps"] != steps[role][s.name]:
                raise ValueError(
                    f"{name}: {role} {s.name}: TokaMark's length_in_time_stamps "
                    f"{meta[role][s.name]['length_in_time_stamps']} differs from the {steps[role][s.name]} "
                    "computed from seconds and dt; has TokaMark's rule changed?"
                )
    return TaskSpec(
        name=name,
        markovian=config["task_type"] == "markovian",
        input_length=input_length,
        output_length=output_length,
        delta=delta,
        stride=float(config["stride_window"]),
        inputs=inputs, actuators=actuators, outputs=outputs,
        steps=steps,
    )


def persistence_reference(task: TaskSpec) -> TaskSpec:
    """The task as TokaMark's persistence script scores it (`scripts/persistence.py`, lines 348-381).

    The script edits the task's configuration before it builds the windows: markovian, inputs and
    outputs reduced to the signals that are both, no actuators. In sparse test mode TokaMark drops
    a window when all inputs and all actuators are entirely NaN, so without actuators a window needs
    a finite input and the first cut moves to the first valid input. The published persistence
    column is therefore scored on the windows that have a finite input. Input and output order is
    kept; `windows` is "persistence_reference" and the rest (name, lengths, delta, stride, steps)
    is the task's. Raises ``ValueError`` if no output is also an input.
    """
    input_names = {s.name for s in task.inputs}
    output_names = {s.name for s in task.outputs}
    common = input_names & output_names
    if not common:
        raise ValueError(f"task {task.name} has no output that is also an input; persistence is not defined")
    return replace(
        task,
        markovian=True,
        inputs=tuple(s for s in task.inputs if s.name in common),
        actuators=(),
        outputs=tuple(s for s in task.outputs if s.name in common),
        steps={
            "input": {n: v for n, v in task.steps["input"].items() if n in common},
            "actuator": {},
            "output": {n: v for n, v in task.steps["output"].items() if n in common},
        },
        windows="persistence_reference",
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
