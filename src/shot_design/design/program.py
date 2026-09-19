"""Editable, immutable shot programs and validated IGNITE input exports.

Public functions return JSON-ready preview dictionaries, except export_ignite,
which returns the atomically written artifact Path. All paths share _evaluate.
Source caches/corpus files are read-only; writes are designs and new seed outputs.
"""

from __future__ import annotations

import os
import re
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, field_validator

from ..config import Paths, load_yaml
from . import actuators as act
from .program_reference import comparison_reference, prepare_reference, reference

SEED_FRAMES = 20
MAX_PREDICTION_FRAMES = 80
ID_PATTERN = r"^[0-9a-f]{32}$"
Shot = Annotated[int, Field(strict=True, gt=0)]
Digest = Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class Vertex(BaseModel):
    model_config = ConfigDict(extra="forbid")
    t_s: FiniteFloat
    y: FiniteFloat


class ProposalBaseline(BaseModel):
    """Persist a physical-value proposal independently of later manual edits."""

    model_config = ConfigDict(extra="forbid")
    method: Literal["average"] = "average"
    shots: list[Shot] = Field(min_length=2, max_length=6)
    start_s: FiniteFloat
    end_s: FiniteFloat
    reference_digest: Digest
    curves: dict[str, list[Vertex]]
    skipped_channels: dict[str, str] = Field(default_factory=dict)

    @field_validator("shots")
    @classmethod
    def distinct_shots(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Proposal reference shots must be distinct")
        return value

    @field_validator("curves", "skipped_channels")
    @classmethod
    def native_channels(cls, value):
        unknown = set(value) - set(act.CHANNEL_NAMES)
        if unknown:
            raise ValueError(
                "Proposal requires native actuator keys: " + ", ".join(sorted(unknown))
            )
        return value


class DesignProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal["shot-design/1"] = "shot-design/1"
    id: Annotated[str, Field(pattern=ID_PATTERN)] | None = None
    created: str | None = None
    reference_shot: Shot
    comparison_shots: list[Shot] = Field(default_factory=list, max_length=5)
    start_s: FiniteFloat = 1.0
    end_s: FiniteFloat = 5.0
    notes: str = Field(default="", max_length=100_000)
    reference_digest: Digest | None = None
    proposal: ProposalBaseline | None = None
    edits: dict[str, list[Vertex]] = Field(default_factory=dict)
    units: dict[str, str] = Field(default_factory=dict)

    @field_validator("comparison_shots")
    @classmethod
    def distinct_comparisons(cls, value):
        if len(value) != len(set(value)):
            raise ValueError("Comparison shots must be distinct")
        return value


class ProgramValidationError(ValueError):
    def __init__(self, errors):
        self.errors = list(errors)
        super().__init__("; ".join(self.errors))


def _catalog() -> dict:
    """Native units/order; member names follow the corpus registry, not UI sorting."""
    config = load_yaml("actuators.yaml")
    systems = config["systems"]
    members = {v["group"]: v for v in config["corpus"].values()}
    units = {
        "ech_power": "W",
        "pinj": "W",
        "beam_voltage": "V",
        "tinj": "N m",
        "gas_flow": "Torr.L/s",
        "gas_raw": "V",
        "rmp": "A",
        "i_coil": "A",
    }
    labels = {
        "ech_power": "ECH power",
        "pinj": "NBI power",
        "beam_voltage": "Beam voltage",
        "tinj": "NBI torque",
        "gas_flow": "Calibrated gas flow",
        "gas_raw": "Gas valve",
        "rmp": "I-coil current",
        "i_coil": "Coil current",
    }
    out = {}
    for group in act.ACT_SPEC:
        entry = members[group.group]
        names = entry.get("members")
        if names is None and group.group in ("beam_voltage", "tinj"):
            names = members["pinj"]["members"]
        if names is None and group.group == "gas_flow":
            names = members["gas_raw"]["members"]
        if names is None and group.group == "rmp":
            names = members["i_coil"]["members"][6:]
        system = entry.get("system")
        if group.group == "rmp":
            system = members["i_coil"].get("system")
        limit = systems.get(system, {}).get("max")
        for i in range(group.n_channels):
            name = names[i] if names and i < len(names) else i
            out[f"{group.group}[{i}]"] = {
                "rows": [group.offset + i],
                "label": f"{labels[group.group]} {name}",
                "units": units[group.group],
                "limit": limit,
            }
    for key, group in [("nbi.total", "pinj"), ("ech.total", "ech_power")]:
        spec = act.group_spec(group)
        out[key] = {
            "rows": list(range(spec.offset, spec.offset + spec.n_channels)),
            "label": f"{key.split('.')[0].upper()} total power",
            "units": "W",
            "limit": None,
        }
    return out


def _window(program, frames, errors):
    start, end = program.start_s / act.FRAME_S, program.end_s / act.FRAME_S
    if not np.isfinite([start, end]).all():
        errors.append("Prediction window exceeds finite reference time bounds")
        return SEED_FRAMES, min(frames, 100)
    if not np.isclose(start, round(start), rtol=0, atol=1e-7) or not np.isclose(
        end, round(end), rtol=0, atol=1e-7
    ):
        errors.append("Prediction start/end must align to 50 ms frames")
    start, end = round(start), round(end)
    if start < SEED_FRAMES:
        errors.append("Prediction start must leave 20 earlier reference frames")
    if not 1 <= end - start <= MAX_PREDICTION_FRAMES:
        errors.append("Prediction window must contain 1 to 80 frames")
    if start >= frames or end > frames:
        errors.append(
            f"Prediction window exceeds reference cache bounds ({frames * 0.05:g} s)"
        )
    return start, end


def _points(values, indices):
    return [
        {"t_s": round(int(i) * act.FRAME_S, 10), "y": float(y)}
        for i, y in zip(indices, values, strict=True)
        if np.isfinite(y)
    ]


def _negative_power_changes(target, baseline):
    """Preserve unchanged measurement noise without allowing new negative demand."""
    same_baseline = np.isclose(target, baseline, rtol=1e-12, atol=1e-12)
    return (target < 0) & ((baseline >= 0) | ~same_baseline)


def _evaluate(program: DesignProgram, paths: Paths):
    program = DesignProgram.model_validate(program).model_copy(deep=True)
    catalog = _catalog()
    program.units = {**{k: v["units"] for k, v in catalog.items()}, **program.units}
    errors, model_errors, warnings, source_errors = [], [], [], []
    needs_seed = False
    result = {
        "program": None,
        "channels": [],
        "validation": None,
        "context_start_s": program.start_s - SEED_FRAMES * act.FRAME_S,
        "seed_frames": SEED_FRAMES,
        "frame_s": act.FRAME_S,
    }

    def finish(artifact=None):
        result["program"] = program.model_dump(mode="json")
        result["validation"] = {
            "errors": [*errors, *model_errors],
            "physical_errors": errors,
            "model_errors": model_errors,
            "warnings": warnings,
            "can_export": not errors and not model_errors,
            "can_save": not source_errors,
            "needs_seed": needs_seed and not source_errors,
        }
        return result, artifact, source_errors

    try:
        ref = reference(program.reference_shot, paths)
    except ValueError as exc:
        errors.append(str(exc))
        source_errors.append(str(exc))
        return finish()
    needs_seed = ref.needs_seed
    if ref.cache_error:
        model_errors.append(ref.cache_error)
    if program.reference_digest and program.reference_digest not in (
        ref.digest, ref.source_digest
    ):
        message = (
            "Reference source changed; explicitly reseed the design before exporting"
        )
        errors.append(message)
        source_errors.append(message)
    else:
        program.reference_digest = ref.digest
    start, end = _window(program, ref.controls.n_frames, errors)
    # Invalid windows still display a bounded trace so the draft remains editable.
    display_start = min(max(start, SEED_FRAMES), ref.controls.n_frames - 1)
    display_end = min(max(end, display_start + 1), ref.controls.n_frames)
    display_end = min(display_end, display_start + MAX_PREDICTION_FRAMES)
    context = display_start - SEED_FRAMES
    # Use the same decimal frame coordinates returned by _points, so an
    # unchanged vertex is sampled exactly even beside a steep power transition.
    times = np.round(np.arange(display_start, display_end) * act.FRAME_S, 10)
    prediction = ref.controls.raw[:, display_start:display_end].copy()
    proposal_curves = {}
    if program.proposal is not None:
        proposal_errors = []
        proposal = program.proposal
        if proposal.shots != [program.reference_shot, *program.comparison_shots]:
            proposal_errors.append(
                "Proposal reference shots changed; merge references again"
            )
        if (proposal.start_s, proposal.end_s) != (program.start_s, program.end_s):
            proposal_errors.append(
                "Proposal prediction window changed; merge references again"
            )
        if proposal.reference_digest != (ref.source_digest or ref.digest):
            proposal_errors.append(
                "Proposal reference source changed; merge references again"
            )
        errors.extend(proposal_errors)
        source_errors.extend(proposal_errors)
        if not proposal_errors:
            proposal_curves = proposal.curves
        if proposal.skipped_channels:
            warnings.append(
                "Average retained first-reference values for unavailable or "
                "incompatible channels: "
                + ", ".join(proposal.skipped_channels)
            )
    edited_rows = set()
    proposal_prediction = prediction.copy()
    for curves in (proposal_curves, program.edits):
        # A proposal and manual edits are separate layers. Totals use the member
        # ratios of the proposal, and overlap is rejected only within a layer.
        occupied = set()
        layer_baseline = prediction.copy()
        for key, vertices in curves.items():
            if key not in catalog:
                errors.append(f"Unknown actuator key: {key}")
                continue
            rows = catalog[key]["rows"]
            if occupied.intersection(rows):
                errors.append(f"{key}: total/member edits overlap")
            occupied.update(rows)
            if program.units[key] != catalog[key]["units"]:
                errors.append(f"{key}: units must be {catalog[key]['units']}")
                continue
            if not ref.available[rows, display_start:display_end].all():
                errors.append(
                    f"{key}: missing reference channel measurements "
                    "in prediction window"
                )
                continue
            tx = np.array([v.t_s for v in vertices])
            if not len(tx) or np.any(np.diff(tx) <= 0):
                errors.append(f"{key}: vertices must have strictly increasing times")
                continue
            if (
                tx[0] > program.start_s + 1e-9
                or tx[-1] < program.end_s - act.FRAME_S - 1e-9
            ):
                errors.append(
                    f"{key}: vertices must cover start through last prediction frame"
                )
                continue
            if tx[0] < program.start_s - 1e-9 or tx[-1] > program.end_s + 1e-9:
                errors.append(f"{key}: vertices must stay inside the prediction window")
                continue
            target = np.interp(times, tx, [v.y for v in vertices])
            if key.startswith(("pinj[", "ech_power[")) or key in (
                "nbi.total", "ech.total"
            ):
                baseline = layer_baseline[rows].sum(axis=0)
                # Preserve historical measurement noise at untouched frames. The
                # tolerance only accommodates interpolation roundoff of that same
                # negative baseline; newly negative demand is always rejected.
                if np.any(_negative_power_changes(target, baseline)):
                    errors.append(
                        f"{key}: edited injected power demand must be nonnegative (W)"
                    )
                    continue
            if len(rows) > 1:
                raw = layer_baseline[rows]
                total = raw.sum(axis=0)
                if np.any((total == 0) & (target != 0)):
                    errors.append(
                        f"{key}: nonzero demand has no active reference member split"
                    )
                    continue
                target = raw * np.divide(
                    target, total, out=np.ones_like(total), where=total != 0
                )
                if key in ("nbi.total", "ech.total"):
                    negative_changes = _negative_power_changes(target, raw)
                    invalid_members = np.flatnonzero(negative_changes.any(axis=1))
                    if len(invalid_members):
                        errors.extend(
                            f"{key}: split changes {act.CHANNEL_NAMES[rows[i]]} "
                            "to negative "
                            "power; edited member demand must be nonnegative (W)"
                            for i in invalid_members
                        )
                        continue
            else:
                target = target[None, :]
            if not np.isfinite(target).all():
                errors.append(f"{key}: interpolation exceeds finite physical values")
                continue
            prediction[rows] = target
            edited_rows.update(rows)
        if curves is proposal_curves:
            proposal_prediction = prediction.copy()
    output = ref.cache["actuators"][context:display_end].clone() if ref.cache else None
    mean, std = ref.controls.stats
    for row in sorted(edited_rows) if ref.cache else ():
        key = act.CHANNEL_NAMES[row]
        baseline = ref.controls.raw[row, display_start:display_end]
        if np.array_equal(prediction[row], baseline):
            continue
        if std[row] == 0:
            model_errors.append(
                f"{key}: changed values have zero-spread reference normalization"
            )
            continue
        expected = ref.controls.z[row].astype(np.float16)
        actual = ref.cache["actuators"][:, row].numpy()
        # Allow at most float16 rounding; differences like the known source-drift
        # coil caches are not compatible with the current reference statistics.
        if not np.allclose(expected, actual, rtol=0.001, atol=1e-5):
            model_errors.append(
                f"{key}: cache normalization disagrees with raw reference; rebuild cache"
            )
            continue
        with np.errstate(over="ignore", invalid="ignore"):
            z = (prediction[row] - mean[row]) / (std[row] + act.EPS)
        if not np.isfinite(z).all() or np.any(np.abs(z) > np.finfo(np.float16).max):
            model_errors.append(f"{key}: edited normalization would overflow float16")
            continue
        output[SEED_FRAMES:, row] = torch.tensor(z, dtype=torch.float16)
        if np.any(np.abs(z) > act.EXTRAPOLATION_Z):
            warnings.append(f"Edit {key} exceeds |z|=3; model extrapolation")
    baseline_z = (
        ref.cache["actuators"][context:display_end].numpy()
        if ref.cache else np.zeros((1, act.N_CHANNELS))
    )
    excursions = [
        act.CHANNEL_NAMES[i]
        for i in np.flatnonzero(
            np.max(np.abs(baseline_z), axis=0) > act.EXTRAPOLATION_Z
        )
    ]
    if excursions:
        warnings.append(
            "Baseline reference already exceeds |z|=3: " + ", ".join(excursions)
        )
    related = {
        "pinj": ("beam_voltage", "tinj"),
        "beam_voltage": ("pinj", "tinj"),
        "tinj": ("pinj", "beam_voltage"),
        "gas_raw": ("gas_flow",),
        "gas_flow": ("gas_raw",),
        "rmp": ("i_coil",),
        "i_coil": ("rmp",),
    }
    for group in act.ACT_SPEC:
        if not edited_rows.intersection(
            range(group.offset, group.offset + group.n_channels)
        ):
            continue
        for retained in related.get(group.group, ()):
            other = act.group_spec(retained)
            if (
                not set(range(other.offset, other.offset + other.n_channels))
                <= edited_rows
            ):
                warnings.append(
                    f"{group.group} edited; related {retained} controls retain reference values"
                )
    comparisons = []
    for shot in program.comparison_shots:
        try:
            comparisons.append((shot, comparison_reference(shot, paths, display_end)))
        except ValueError as exc:
            warnings.append(f"Comparison {shot} unavailable: {exc}")
    for key, info in catalog.items():
        rows = info["rows"]
        available = ref.available[rows, display_start:display_end].all()
        ref_indices = np.arange(context, display_end)
        ref_good = ref.available[rows, context:display_end].all(axis=0)
        values = ref.controls.raw[rows, context:display_end].sum(axis=0)
        entry = {
            "key": key,
            "label": info["label"],
            "units": info["units"],
            "editable": bool(available),
            "reason": None if available else "Missing reference channel measurements",
            "reference": _points(values[ref_good], ref_indices[ref_good]),
            "vertices": [],
            "baseline_vertices": [],
            "comparisons": [],
        }
        if available:
            entry["baseline_vertices"] = _points(
                proposal_prediction[rows].sum(axis=0),
                np.arange(display_start, display_end),
            )
        if key in program.edits:
            entry["vertices"] = [v.model_dump() for v in program.edits[key]]
        elif available:
            with np.errstate(over="ignore", invalid="ignore"):
                values = prediction[rows].sum(axis=0)
            if not np.isfinite(values).all():
                errors.append(f"{key}: edited total exceeds finite physical values")
            entry["vertices"] = _points(values, np.arange(display_start, display_end))
        for shot, comparison in comparisons:
            stop = min(display_end, comparison.controls.n_frames)
            if stop <= context:
                continue
            good = comparison.available[rows, context:stop].all(axis=0)
            vals = comparison.controls.raw[rows, context:stop].sum(axis=0)
            if good.any():
                entry["comparisons"].append(
                    {
                        "shot": shot,
                        "vertices": _points(vals[good], np.arange(context, stop)[good]),
                    }
                )
        if info["limit"] is not None and available:
            cap = float(info["limit"])
            baseline = ref.controls.raw[rows, display_start:display_end].sum(axis=0)
            if np.any(np.abs(baseline) > cap):
                warnings.append(
                    f"Baseline {key} exceeds configured provisional amplitude limit {cap:g} {info['units']}"
                )
            if set(rows).intersection(edited_rows) and np.any(
                np.abs(prediction[rows].sum(axis=0)) > cap
            ):
                warnings.append(
                    f"Edit {key} exceeds configured provisional amplitude limit {cap:g} {info['units']}"
                )
        result["channels"].append(entry)
    if program.edits or program.proposal:
        warnings.append(
            "Prediction diagnostic tokens describe the real reference, not ground truth for the edited scenario"
        )
    artifact = {
        "codes": {
            k: v[context:display_end].clone() for k, v in ref.cache["codes"].items()
        },
        "actuators": output,
        "n_frames": display_end - context,
        "vocabs": dict(ref.cache["vocabs"]),
    } if ref.cache else None
    return finish(artifact)


def preview(program: DesignProgram, paths: Paths) -> dict:
    return _evaluate(program, paths)[0]


def average_references(program: DesignProgram, paths: Paths) -> dict:
    """Snapshot equal physical means at matching absolute 50 ms frame times.

    A channel must be measured throughout the window by every donor. Otherwise
    the first reference remains intact and the omission is recorded. Power means
    that introduce negative demand are also skipped, never clipped. Diagnostic
    seeds and normalization always belong to that first reference; donors need
    only raw corpus measurements. Existing manual edits remain a separate layer.
    """
    program = DesignProgram.model_validate(program).model_copy(deep=True)
    shots = [program.reference_shot, *program.comparison_shots]
    if len(shots) < 2 or len(shots) != len(set(shots)):
        raise ProgramValidationError([
            "Average requires at least two distinct reference shots"
        ])
    try:
        ref = reference(program.reference_shot, paths)
    except ValueError as exc:
        raise ProgramValidationError([str(exc)]) from exc
    errors = []
    start, end = _window(program, ref.controls.n_frames, errors)
    if program.reference_digest and program.reference_digest not in (
        ref.digest, ref.source_digest
    ):
        errors.append("Reference source changed; explicitly reseed before merging")
    if errors:
        raise ProgramValidationError(errors)
    sources = [(program.reference_shot, ref)]
    for shot in program.comparison_shots:
        try:
            sources.append((shot, comparison_reference(shot, paths, end)))
        except ValueError as exc:
            raise ProgramValidationError([
                f"Cannot merge reference {shot}: {exc}"
            ]) from exc
    curves, skipped = {}, {}
    for row, key in enumerate(act.CHANNEL_NAMES):
        missing = [
            shot for shot, source in sources
            if not source.available[row, start:end].all()
        ]
        if missing:
            skipped[key] = (
                "Missing measurements in prediction window for shots: "
                + ", ".join(map(str, missing))
            )
            continue
        values = np.mean(
            [source.controls.raw[row, start:end] for _, source in sources], axis=0
        )
        if not np.isfinite(values).all():
            raise ProgramValidationError([
                f"{key}: average exceeds finite physical values"
            ])
        if key.startswith(("pinj[", "ech_power[")) and np.any(
            _negative_power_changes(values, ref.controls.raw[row, start:end])
        ):
            skipped[key] = (
                "Average would introduce or change negative power demand; "
                "retained first-reference measurements"
            )
            continue
        curves[key] = _points(values, np.arange(start, end))
    program.reference_digest = ref.digest
    program.proposal = ProposalBaseline(
        shots=shots,
        start_s=program.start_s,
        end_s=program.end_s,
        reference_digest=ref.source_digest or ref.digest,
        curves=curves,
        skipped_channels=skipped,
    )
    return preview(program, paths)


def prepare_ignite(program: DesignProgram, paths: Paths) -> dict:
    result, _, source_errors = _evaluate(program, paths)
    if source_errors:
        raise ProgramValidationError(source_errors)
    if result["validation"]["needs_seed"]:
        prepare_reference(program.reference_shot, paths)
    return preview(DesignProgram(**result["program"]), paths)


def _id_path(ident: str, root: Path, suffix: str) -> Path:
    if not re.fullmatch(ID_PATTERN, ident):
        raise FileNotFoundError("Unknown design revision")
    path = root / f"{ident}{suffix}"
    if path.is_symlink():
        raise FileNotFoundError("Unknown design revision")
    return path


def _atomic(path: Path, writer, *, immutable=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".design-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        if immutable:
            os.link(
                temp, path
            )  # publish a complete file without ever overwriting a revision
        else:
            os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def save(program: DesignProgram, paths: Paths) -> dict:
    result, _, source_errors = _evaluate(program, paths)
    if source_errors:
        raise ProgramValidationError(source_errors)
    saved = DesignProgram(**result["program"])
    saved.id = uuid.uuid4().hex
    saved.created = datetime.now(UTC).isoformat()
    path = _id_path(saved.id, paths.actuations_dir / "designs", ".json")
    _atomic(
        path,
        lambda f: f.write(saved.model_dump_json(indent=2).encode()),
        immutable=True,
    )
    result["program"] = saved.model_dump(mode="json")
    return result


def load_program(ident: str, paths: Paths) -> DesignProgram:
    path = _id_path(ident, paths.actuations_dir / "designs", ".json")
    program = DesignProgram.model_validate_json(path.read_text())
    if program.id != ident:
        raise ValueError("Stored design revision identity is inconsistent")
    return program


def load(ident: str, paths: Paths) -> dict:
    return preview(load_program(ident, paths), paths)


def list_programs(paths: Paths) -> list[dict]:
    revisions = []
    for path in (paths.actuations_dir / "designs").glob("*.json"):
        try:
            program = load_program(path.stem, paths)
        except (ValueError, OSError):
            continue
        revisions.append(
            program.model_dump(
                include={"id", "reference_shot", "created", "start_s", "end_s"}
            )
        )
    return sorted(revisions, key=lambda p: p["created"] or "", reverse=True)


def export_ignite(program: DesignProgram, paths: Paths) -> Path:
    result, artifact, _ = _evaluate(program, paths)
    if not result["validation"]["can_export"]:
        raise ProgramValidationError(result["validation"]["errors"])
    ident = result["program"]["id"] or uuid.uuid4().hex
    path = _id_path(ident, paths.ignite_inputs_dir / "designs", ".pt")
    _atomic(path, lambda f: torch.save(artifact, f))
    return path
