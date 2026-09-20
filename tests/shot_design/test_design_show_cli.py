"""`shot_design design show <ident>`: the saved-design revision, or, with
`--references` / `--actuation-csv`, the two files the Frontier demo script needs
(`references.md`, `actuation.csv`) built from the design's own saved HDF5 artifact.

The default view reads the editable JSON revision through `design.program.load` (real
corpus fixtures, via `program_source`/`draft` from test_program.py); `--references` and
`--actuation-csv` read only the sibling `outputs/<ident>.h5` written by
`design.assistant._write_hdf5`, so those two are tested against a small synthetic HDF5
with no corpus or retrieval machinery involved.
"""

from __future__ import annotations

import csv
import io
import json

import h5py
import numpy as np

from shot_design import cli
from shot_design.design import actuators as act

from .test_program import draft, program_source, service  # noqa: F401

IDENT = "b" * 32


def _write_design_artifact(paths, ident=IDENT, *, n_frames=4):
    """A minimal `outputs/<ident>.h5`, in the exact shape `_write_hdf5` writes."""
    metadata = {
        "prompt": "control tearing modes with ECCD",
        "goal": "suppress the n=2 NTM",
        "model": "gemma4:26b",
        "reference_shot": 195071,
        "comparison_shots": [193353],
        "baseline": "reference",
        "scales": {"ech.total": 1.3},
        "explanation": "Increase ECH power 30% to raise the deposited current density.",
        "retrieved_candidates": [
            {"shot": 195071, "score": 0.9123, "description": "NTM control shot"},
            {"shot": 193353, "score": 0.8456, "description": "AE control shot"},
        ],
        "checks": {"available_channels": 80, "warnings": []},
        "design_id": ident,
    }
    names = list(act.CHANNEL_NAMES)
    times = np.arange(n_frames) * act.FRAME_S + 1.0
    values = np.zeros((n_frames, len(names)), dtype=np.float32)
    values[:, 0] = np.arange(n_frames, dtype=np.float32)
    available = np.ones((n_frames, len(names)), dtype=bool)
    values[:, 1] = np.nan
    available[:, 1] = False
    units = ["W" for _ in names]

    out_dir = paths.data_root / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_dir / f"{ident}.h5", "w") as f:
        string = h5py.string_dtype("utf-8")
        f.create_dataset("time_s", data=times)
        f.create_dataset("actuators", data=values)
        f.create_dataset("available", data=available)
        f.create_dataset("channel_names", data=names, dtype=string)
        f.create_dataset("channel_units", data=units, dtype=string)
        f.create_dataset("metadata", data=json.dumps(metadata), dtype=string)
        f.create_dataset("program", data=json.dumps({"id": ident}), dtype=string)
    return metadata, names, times, values


def test_show_default_prints_the_saved_program(program_source, capsys):
    paths, _ = program_source
    saved = service().save(draft(notes="Request: x\n\nGemma: y"), paths)
    ident = saved["program"]["id"]

    rc = cli.main(["design", "show", ident])

    assert rc == 0
    out = capsys.readouterr().out
    assert ident in out
    assert "reference shot 990091" in out


def test_show_unknown_ident_fails_cleanly(program_source, capsys):
    paths, _ = program_source

    rc = cli.main(["design", "show", "0" * 32])

    assert rc == 1
    assert "0" * 32 in capsys.readouterr().err


def test_references_prints_prompt_explanation_and_candidates(paths, capsys):
    metadata, *_ = _write_design_artifact(paths)

    rc = cli.main(["design", "show", IDENT, "--references"])

    assert rc == 0
    out = capsys.readouterr().out
    assert metadata["prompt"] in out
    assert metadata["explanation"] in out
    assert str(metadata["reference_shot"]) in out
    assert str(metadata["comparison_shots"][0]) in out
    assert "195071" in out and "193353" in out


def test_actuation_csv_has_a_header_and_one_row_per_frame(paths, capsys):
    metadata, names, times, values = _write_design_artifact(paths, n_frames=4)

    rc = cli.main(["design", "show", IDENT, "--actuation-csv"])

    assert rc == 0
    out = capsys.readouterr().out
    rows = list(csv.reader(io.StringIO(out)))
    assert rows[0] == ["time_s", *names]
    assert len(rows) == 1 + len(times)
    assert float(rows[1][0]) == times[0]
    assert float(rows[1][1]) == values[0, 0]
    assert rows[1][2] == ""  # the unavailable channel is blank, not the literal "nan"


def test_show_missing_artifact_fails_cleanly(paths, capsys):
    rc = cli.main(["design", "show", "c" * 32, "--references"])

    assert rc == 1
    assert "c" * 32 in capsys.readouterr().err
