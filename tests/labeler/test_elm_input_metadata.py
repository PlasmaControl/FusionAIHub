"""Source metadata stays separate from numerical ELM input normalization."""

import hashlib
import json

import numpy as np

from labeler.elm import inputs, prepare


def _record(path, metadata=None):
    t_fs = np.arange(100 * 50) / 50.0
    t_int = np.arange(100 * 100) / 100.0
    arrays = {
        "t_fs_ms": t_fs,
        "filterscopes": np.full((3, t_fs.size), 1e15),
        "t_int_ms": t_int,
        "interferometer": np.full((2, t_int.size), 3e14),
    }
    if metadata is not None:
        arrays["source_metadata"] = json.dumps(metadata)
    np.savez(path, **arrays)
    return arrays


def test_legacy_cache_does_not_invent_density_units(tmp_path):
    path = tmp_path / "7.npz"
    _record(path)
    metadata = inputs.read_metadata(path)
    density = metadata["interferometer"]
    assert density["ordinate_units"] == [None, None]
    assert density["unit_status"] == "unverified: source ordinate metadata absent"
    assert density["input_divisor_native_units"] == 1e14
    assert (
        metadata["source_record"]["sha256"]
        == hashlib.sha256(path.read_bytes()).hexdigest()
    )


def test_source_metadata_survives_without_changing_input_values(tmp_path):
    path = tmp_path / "7.npz"
    source = {
        "interferometer": {
            "tree": "bci",
            "channels": [r"\BCI::DENV2F", r"\BCI::DENV3F"],
            "ordinate_units": ["source-unit", "source-unit"],
            "source_attrs": {"producer": "synthetic source"},
        }
    }
    record = _record(path, source)
    metadata = inputs.read_metadata(path)
    assert (
        metadata["interferometer"]["source_attrs"]
        == source["interferometer"]["source_attrs"]
    )
    assert metadata["interferometer"]["ordinate_units"] == [
        "source-unit",
        "source-unit",
    ]
    np.testing.assert_array_equal(
        inputs.read_channels(path),
        inputs.channels(
            record["t_fs_ms"],
            record["filterscopes"],
            record["t_int_ms"],
            record["interferometer"],
        ),
    )


def test_preparation_writes_metadata_separately_from_input_values(tmp_path):
    source, target = tmp_path / "7.npz", tmp_path / "7.npy"
    _record(source)
    shot, n = prepare._one((7, source, target))
    expected = inputs.read_channels(source)
    assert shot == 7 and n == expected.shape[1]
    np.testing.assert_array_equal(np.load(target), expected)
    metadata = json.loads(target.with_suffix(".metadata.json").read_text())
    assert metadata["source_record"]["path"] == str(source)
    assert metadata["interferometer"]["physical_rescaling_applied"] is False
