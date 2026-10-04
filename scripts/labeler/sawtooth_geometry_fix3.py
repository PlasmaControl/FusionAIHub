"""Audit ECE setup and acquire only missing essential sawtooth geometry.

Run ``fetch`` through the prescribed pixi ``fdp run`` wrapper. All fetched
records live under round4/saw/fix4; production stores remain read-only.
"""

from __future__ import annotations

import argparse
import errno
import json
import os
import shutil
import signal
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
DEFAULT_WORK = Path("/scratch/gpfs/EKOLEMEN/nc1514/labelmaker/round4/saw/fix4")
OUTPUT = REPO / "outputs/labeler/sawtooth/fix4"
ARCHIVE = Path("/scratch/gpfs/nc1514/omnimode/data")
FDP_MDS_RELAY = "fdp://fdp-d3d-origin.nationalresearchplatform.org:8443/mdsip"
EQ_LEAVES = (
    "gtime",
    "psirz",
    "qpsi",
    "fpol",
    "r",
    "z",
    "rmaxis",
    "zmaxis",
    "ssimag",
    "ssibry",
    "bdry",
    "nbdry",
)
MINIMAL_EQ_LEAVES = ("gtime", "fpol", "rmaxis", "bdry", "nbdry")
Q_INPUT_EQ_LEAVES = tuple(name for name in EQ_LEAVES if name not in ("bdry", "nbdry"))
EQ_RANKS = dict(zip(EQ_LEAVES, (1, 3, 2, 2, 1, 1, 1, 1, 1, 1, 3, 1), strict=True))


def save_json(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(record, indent=2, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", dir=os.environ["TMPDIR"], delete=False
    ) as temporary:
        temporary.write(serialized)
        temporary_path = Path(temporary.name)
    try:
        try:
            temporary_path.replace(path)
        except OSError as error:
            if error.errno != errno.EXDEV:
                raise
            shutil.copyfile(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def authentication_error(error):
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "unauthorized",
            "forbidden",
            "authentication",
            "auth token",
            "bearer",
            "login",
            "401",
            "403",
            "token expired",
        )
    )


def missing_efit_error(error):
    """Recognize missing trees/nodes, rather than transport or decoder errors."""
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "treenodata",
            "tree-e-nodata",
            "tree-w-nodata",
            "no data available",
            "treennf",
            "tree-w-nnf",
            "node not found",
            "treefopenr",
            "tree-e-fopenr",
            "error opening file for read-only",
            "no such tree",
        )
    )


def missing_efit_tree_error(error):
    message = str(error).lower()
    return any(
        token in message
        for token in (
            "treefopenr",
            "tree-e-fopenr",
            "error opening file for read-only",
            "no such tree",
        )
    )


def write_eq_batch(file, arrays, result):
    group = file.require_group("eq")
    group.attrs["source"] = "efit01"
    for name, values in arrays.items():
        if name not in group:
            dataset = group.create_dataset(name, data=values, compression="gzip")
            dataset.attrs["expression"] = rf"\efit01::top.results.geqdsk:{name}"
            dataset.attrs["acquisition"] = "packed_numeric_TDI_with_native_shape_header"
            dataset.attrs["transport"] = result.get("packed_transport", "FDP local TDI")
            result["fetched"].append(f"eq/{name}")
    file.flush()


def evaluate_packed(expression, shot, remote=False, include_height=False):
    """Evaluate TDI in the read-only FDP tree or through remote MDS transport."""
    from toksearch import MdsSignal
    from toksearch.signal.mds import (
        MdsConnectionRegistry,
        MdsLocalSignal,
        MdsTreeRegistry,
    )

    if include_height:
        MdsConnectionRegistry().open_tree(FDP_MDS_RELAY, "ELECTRONS", shot)

    request = MdsSignal(
        expression,
        "efit01",
        dims=[],
        fetch_units=False,
        location=FDP_MDS_RELAY if remote else None,
    )
    reader = request.sig
    if isinstance(reader, MdsLocalSignal):
        tree = MdsTreeRegistry().open_tree("efit01", shot, treepath=reader.treepath)
        return {"data": tree.tdiExecute(expression).data()}
    return request.fetch(shot)


def _alarm(signum, frame):
    raise TimeoutError("essential metadata fetch exceeded per-node timeout")


def unpack_minimal(values):
    """Decode explicit native MDS dimensions; never infer array order by size."""
    data = np.asarray(values, dtype=float).ravel()
    if len(data) < 6 or not np.isfinite(data[:6]).all():
        raise ValueError("invalid minimal metadata shape header")
    header = data[:6].astype(int)
    if not np.array_equal(header, data[:6]) or (header <= 0).any():
        raise ValueError("invalid minimal metadata shape sizes")
    nt, nf0, nf1, nb0, nb1, nb2 = header
    sizes = (nt, nf0 * nf1, nt, nt, nb0 * nb1 * nb2)
    if len(data) != 6 + sum(sizes):
        raise ValueError("minimal metadata payload length disagrees with header")
    parts, start = [], 6
    for size in sizes:
        parts.append(data[start : start + size])
        start += size
    # MDS arrays' first native dimension varies fastest. The native shapes
    # are included by TDI; the numpy data convention reverses those shapes.
    fpol = parts[1].reshape((nf1, nf0))
    boundary = parts[4].reshape((nb2, nb1, nb0))
    if fpol.shape[0] != nt or boundary.shape[0] != nt or boundary.shape[-1] != 2:
        raise ValueError("minimal metadata axes do not match measured EFIT contract")
    return dict(
        zip(
            MINIMAL_EQ_LEAVES,
            (parts[0], fpol, parts[2], boundary, parts[3]),
            strict=True,
        )
    )


def packed_minimal_expression():
    """One numeric TDI vector carrying only the five essential EFIT nodes."""
    nodes = {
        name: rf"DATA(\efit01::top.results.geqdsk:{name})" for name in MINIMAL_EQ_LEAVES
    }
    fields = [
        f"SIZE({nodes['gtime']})",
        f"SIZE({nodes['fpol']},0)",
        f"SIZE({nodes['fpol']},1)",
        f"SIZE({nodes['bdry']},0)",
        f"SIZE({nodes['bdry']},1)",
        f"SIZE({nodes['bdry']},2)",
        nodes["gtime"],
        nodes["fpol"],
        nodes["rmaxis"],
        nodes["nbdry"],
        nodes["bdry"],
    ]
    return "[" + ",".join(fields) + "]"


def packed_full_expression(leaves=EQ_LEAVES):
    """Carry every EFIT q=1 input with measured native dimension headers."""
    nodes = {name: rf"DATA(\efit01::top.results.geqdsk:{name})" for name in leaves}
    fields = [
        f"SIZE({nodes[name]},{axis})"
        for name in leaves
        for axis in range(EQ_RANKS[name])
    ]
    fields.extend(nodes[name] for name in leaves)
    return "[" + ",".join(fields) + "]"


def packed_height_expression():
    return packed_full_expression()[:-1] + r",DATA(\ECE::TOP.SETUP.ECEZH)]"


def unpack_height(values):
    values = np.asarray(values, dtype=float).ravel()
    if not len(values) or not np.isfinite(values[-1]) or abs(values[-1]) > 1:
        raise ValueError("invalid packed measured sightline height")
    return unpack_full(values[:-1]), float(values[-1])


def unpack_full(values, leaves=EQ_LEAVES):
    """Decode full EFIT arrays only when exact shapes and sizes agree."""
    data = np.asarray(values, dtype=float).ravel()
    width = sum(EQ_RANKS[name] for name in leaves)
    if len(data) < width or not np.isfinite(data[:width]).all():
        raise ValueError("invalid full metadata shape header")
    header = data[:width].astype(int)
    if not np.array_equal(header, data[:width]) or (header <= 0).any():
        raise ValueError("invalid full metadata shape sizes")
    arrays, shape_start, data_start = {}, 0, width
    for name in leaves:
        rank = EQ_RANKS[name]
        shape = tuple(header[shape_start : shape_start + rank][::-1])
        size = int(np.prod(shape))
        if data_start + size > len(data):
            raise ValueError("full metadata payload is shorter than shape header")
        arrays[name] = data[data_start : data_start + size].reshape(shape)
        shape_start += rank
        data_start += size
    if data_start != len(data):
        raise ValueError("full metadata payload is longer than shape header")
    nt = len(arrays["gtime"])
    if (
        arrays["psirz"].shape != (nt, len(arrays["z"]), len(arrays["r"]))
        or (
            "bdry" in arrays
            and (arrays["bdry"].shape[0] != nt or arrays["bdry"].shape[-1] != 2)
        )
        or any(arrays[name].shape[0] != nt for name in leaves if name not in ("r", "z"))
    ):
        raise ValueError("full metadata axes disagree with EFIT node contract")
    return arrays


def validate_batch(args):
    """Check one remote packed result against same-shot individual-node data."""
    shot = args.shots[0] if args.shots else 185805
    source = args.work / "geometry" / f"{shot}.h5"
    leaves = EQ_LEAVES if args.batch_full else MINIMAL_EQ_LEAVES
    expression = (
        packed_height_expression()
        if args.packed_height
        else packed_full_expression()
        if args.batch_full
        else packed_minimal_expression()
    )
    decoder = unpack_full if args.batch_full else unpack_minimal
    mode = "full" if args.batch_full else "minimal"
    report = {
        "shot": shot,
        "mode": mode,
        "transport": FDP_MDS_RELAY if args.remote_packed else "FDP local TDI",
        "individual_node_source": str(source),
        "verified": False,
    }
    signal.signal(signal.SIGALRM, _alarm)
    try:
        with h5py.File(source, "r", locking=False) as file:
            expected = {name: np.asarray(file[f"eq/{name}"]) for name in leaves}
            expected_height = (
                float(file["ecegeom/ECEZH"][()]) if args.packed_height else None
            )
        started = time.monotonic()
        signal.alarm(args.node_timeout)
        record = evaluate_packed(
            expression, shot, args.remote_packed, args.packed_height
        )
        signal.alarm(0)
        if args.packed_height:
            actual, height = unpack_height(record["data"])
            report["height_comparison"] = {
                "expected_m": expected_height,
                "packed_m": height,
                "matches": bool(abs(height - expected_height) <= 1e-7),
            }
        else:
            actual = decoder(record["data"])
        report["elapsed_s"] = time.monotonic() - started
        report["comparisons"] = {
            name: {
                "expected_shape": list(expected[name].shape),
                "packed_shape": list(actual[name].shape),
                "max_absolute_difference": float(
                    np.nanmax(abs(actual[name] - expected[name]))
                )
                if actual[name].shape == expected[name].shape
                else None,
                "matches": bool(
                    actual[name].shape == expected[name].shape
                    and np.allclose(
                        actual[name],
                        expected[name],
                        rtol=2e-6,
                        atol=2e-7,
                        equal_nan=True,
                    )
                ),
            }
            for name in leaves
        }
        report["verified"] = all(
            row["matches"] for row in report["comparisons"].values()
        )
        if args.packed_height:
            report["verified"] &= report["height_comparison"]["matches"]
    except Exception as error:  # noqa: BLE001 -- retain upstream validation failure
        signal.alarm(0)
        report["error"] = f"{type(error).__name__}: {error}"
        report["authentication_failed"] = authentication_error(error)
        if report["authentication_failed"]:
            save_json(args.work / "authentication_failed.json", report)
    from toksearch.signal.mds import MdsTreeRegistry

    MdsTreeRegistry().close_all_trees()
    name = (
        "batch_full_height_remote_validation.json"
        if args.packed_height
        else f"batch_{mode}_remote_validation.json"
        if args.remote_packed
        else f"batch_{mode}_validation.json"
    )
    save_json(args.work / name, report)
    save_json(OUTPUT / name, report)
    print(json.dumps(report), flush=True)
    if not report["verified"]:
        raise RuntimeError(f"packed {mode} metadata failed same-shot validation")


def height_audit(args):
    """Record measured sightline heights and blank-unit source contracts."""
    rows = []
    for source, folder in (
        ("archived", args.archive / "raw"),
        ("fetched", args.work / "geometry"),
    ):
        for path in sorted(folder.glob("*.h5")):
            with h5py.File(path, "r", locking=False) as file:
                if "ecegeom/ECEZH" not in file:
                    continue
                dataset = file["ecegeom/ECEZH"]
                rows.append(
                    {
                        "shot": int(path.stem),
                        "source": source,
                        "source_file": str(path),
                        "values": [
                            float(value) if np.isfinite(value) else None
                            for value in np.asarray(dataset).ravel()
                        ],
                        "units": str(dataset.attrs.get("units", "")),
                    }
                )
    report = {
        "height_source": "ELECTRONS ECE setup ECEZH; metres by setup contract",
        "by_shot": rows,
        "missing_fetched_heights": [
            int(path.stem)
            for path in sorted((args.work / "geometry").glob("*.h5"))
            if not any(
                row["shot"] == int(path.stem) and row["source"] == "fetched"
                for row in rows
            )
        ],
    }
    save_json(args.work / "ece_height_audit.json", report)
    if len(rows) > 500:
        from collections import Counter

        small = {key: value for key, value in report.items() if key != "by_shot"}
        small["source_records"] = str(args.work / "ece_height_audit.json")
        small["measured_record_counts"] = dict(Counter(row["source"] for row in rows))
        small["measured_height_counts"] = dict(
            Counter(f"{row['source']}:{json.dumps(row['values'])}" for row in rows)
        )
        save_json(OUTPUT / "ece_height_audit.json", small)
    else:
        save_json(OUTPUT / "ece_height_audit.json", report)
    print(
        json.dumps(
            {
                "measured_records": len(rows),
                "missing_fetched_heights": len(report["missing_fetched_heights"]),
            }
        ),
        flush=True,
    )


def validate_height(args):
    """Compare the remote scalar height to its downloaded same-shot value."""
    from toksearch import MdsSignal

    shot = args.shots[0] if args.shots else 185805
    source = args.work / "geometry" / f"{shot}.h5"
    report = {
        "shot": shot,
        "source_file": str(source),
        "transport": FDP_MDS_RELAY,
        "expression": r"\ECE::TOP.SETUP.ECEZH",
        "verified": False,
    }
    signal.signal(signal.SIGALRM, _alarm)
    try:
        with h5py.File(source, "r", locking=False) as file:
            expected = float(file["ecegeom/ECEZH"][()])
        started = time.monotonic()
        signal.alarm(args.node_timeout)
        record = MdsSignal(
            report["expression"],
            "ELECTRONS",
            dims=[],
            location=FDP_MDS_RELAY,
            fetch_units=False,
        ).fetch(shot)
        signal.alarm(0)
        values = np.asarray(record["data"]).ravel()
        if len(values) != 1:
            raise ValueError("ECEZH is not one scalar")
        actual = float(values[0])
        report.update(
            expected_height_m=expected,
            remote_height_m=actual,
            elapsed_s=time.monotonic() - started,
            verified=bool(np.isfinite(actual) and abs(actual - expected) <= 1e-7),
        )
    except Exception as error:  # noqa: BLE001 -- preserve upstream failure
        signal.alarm(0)
        report.update(
            error=f"{type(error).__name__}: {error}",
            authentication_failed=authentication_error(error),
        )
        if report["authentication_failed"]:
            save_json(args.work / "authentication_failed.json", report)
    save_json(args.work / "height_remote_validation.json", report)
    save_json(OUTPUT / "height_remote_validation.json", report)
    print(json.dumps(report), flush=True)
    if not report["verified"]:
        raise RuntimeError("remote measured sightline height validation failed")


def validate_ece(args):
    """Verify remote waveform data and clock against one cached same-shot node."""
    from toksearch import MdsSignal

    from labeler.events.verify import ECE_POINT, ECE_TREE

    shot = args.shots[0] if args.shots else 141182
    source = args.work / "geometry" / f"{shot}.h5"
    report = {
        "shot": shot,
        "channel": 0,
        "source_file": str(source),
        "transport": FDP_MDS_RELAY,
        "verified": False,
        "data_rtol": 2e-6,
        "data_atol_kev": 2e-7,
        "clock_atol_ms": 1e-5,
    }
    signal.signal(signal.SIGALRM, _alarm)
    try:
        with h5py.File(source, "r", locking=False) as file:
            dataset = file["ece/ECEVS01"]
            expected, attrs = np.asarray(dataset), dict(dataset.attrs)
        started = time.monotonic()
        signal.alarm(args.node_timeout)
        record = MdsSignal(
            ECE_POINT.format(channel=1), ECE_TREE, dims=["dim0"], location=FDP_MDS_RELAY
        ).fetch(shot)
        signal.alarm(0)
        values, clock = (
            np.asarray(record["data"]),
            np.asarray(record["dim0"], dtype=float),
        )
        modeled_clock = attrs["t0_ms"] + attrs["dt_ms"] * np.arange(len(expected))
        shape_ok = values.shape == expected.shape == clock.shape
        data_delta = float(np.max(abs(values - expected))) if shape_ok else None
        clock_delta = float(np.max(abs(clock - modeled_clock))) if shape_ok else None
        report.update(
            samples=len(values),
            local_dtype=str(expected.dtype),
            remote_dtype=str(values.dtype),
            data_max_abs_delta=data_delta,
            clock_max_abs_delta_ms=clock_delta,
            elapsed_s=time.monotonic() - started,
            verified=bool(
                shape_ok
                and np.allclose(values, expected, rtol=2e-6, atol=2e-7, equal_nan=True)
                and clock_delta <= 1e-5
            ),
        )
    except Exception as error:  # noqa: BLE001 -- preserve upstream failure
        signal.alarm(0)
        report.update(
            error=f"{type(error).__name__}: {error}",
            authentication_failed=authentication_error(error),
        )
        if report["authentication_failed"]:
            root = args.work.parent if args.work.name == "reference" else args.work
            save_json(root / "authentication_failed.json", report)
    save_json(args.work / "ece_remote_validation.json", report)
    save_json(OUTPUT / "ece_remote_validation.json", report)
    print(json.dumps(report), flush=True)
    if not report["verified"]:
        raise RuntimeError("remote ECE waveform validation failed")


def frequency_audit(args):
    """Check cached same-shot RF setups against the archived grid proof."""
    from labeler.sawtooth.geometry import audit_frequency_grid

    grid = audit_frequency_grid(args.archive)
    if not grid["verified"]:
        raise RuntimeError("archive fixed-grid channel join is unverified")
    expected = np.asarray(grid["frequency_hz"]) / 1e9
    rows = []
    for path in sorted((args.work / "geometry").glob("*.h5")):
        with h5py.File(path, "r", locking=False) as file:
            if "ecegeom/FREQ" not in file:
                continue
            values = np.asarray(file["ecegeom/FREQ"]).ravel()[:40]
            if len(values) != 40:
                continue
            difference = abs(values - expected)
            rows.append(
                {
                    "shot": int(path.stem),
                    "source_file": str(path),
                    "maximum_abs_delta_ghz": float(np.max(difference)),
                    "exception_channels": np.flatnonzero(difference > 0.001).tolist(),
                    "order_break_channels": np.flatnonzero(
                        np.diff(values) < 0
                    ).tolist(),
                }
            )
    report = {
        "archived_grid_proof": grid,
        "cached_setup_records": len(rows),
        "matches": sum(not row["exception_channels"] for row in rows),
        "exceptions": [row for row in rows if row["exception_channels"]],
        "by_shot": rows,
        "limitations": "cached setups check RF values; the channel identifier join is established by the archived ECE/FREQ proof",
    }
    save_json(args.work / "fetched_frequency_audit.json", report)
    save_json(OUTPUT / "fetched_frequency_audit.json", report)
    print(
        json.dumps(
            {
                "records": len(rows),
                "matches": report["matches"],
                "exception_shots": [row["shot"] for row in report["exceptions"]],
            }
        ),
        flush=True,
    )


def fetch_one(job):
    """One shot, preserving partial successes and explicit missing-node records."""
    from toksearch import MdsSignal

    from labeler.events.verify import ECE_POINT, ECE_TREE
    from labeler.features.resolve_fdp import _fetch_mds

    (
        shot,
        work,
        pace,
        timeout,
        ece,
        minimal,
        batch_minimal,
        batch_full,
        height_only,
        fixed_grid,
        remote_packed,
        packed_height,
    ) = job
    path = Path(work) / "geometry" / f"{shot}.h5"
    path.parent.mkdir(parents=True, exist_ok=True)
    result = {"shot": shot, "path": str(path), "fetched": [], "missing": {}}
    result["packed_transport"] = FDP_MDS_RELAY if remote_packed else "FDP local TDI"
    signal.signal(signal.SIGALRM, _alarm)
    targets = [
        ("eq", name, rf"\efit01::top.results.geqdsk:{name}", "efit01", ())
        for name in (MINIMAL_EQ_LEAVES if minimal else EQ_LEAVES)
    ]
    if not minimal:
        targets += [
            ("ecegeom", name, rf"\ECE::TOP.SETUP.{name}", "ELECTRONS", ())
            for name in (("ECEZH",) if fixed_grid else ("FREQ", "ECEZH"))
        ]
    if height_only:
        targets = [("ecegeom", "ECEZH", r"\ECE::TOP.SETUP.ECEZH", "ELECTRONS", ())]
    if ece:
        targets += [
            ("ece", f"ECEVS{i:02d}", ECE_POINT.format(channel=i), ECE_TREE, ("dim0",))
            for i in range(1, 41)
        ]
    with h5py.File(path, "a") as file:
        file.attrs["shot"] = shot
        batch_leaves = EQ_LEAVES if batch_full else MINIMAL_EQ_LEAVES
        if (batch_minimal or batch_full) and any(
            f"eq/{name}" not in file for name in batch_leaves
        ):
            expression = (
                packed_height_expression()
                if packed_height
                else packed_full_expression()
                if batch_full
                else packed_minimal_expression()
            )
            try:
                signal.alarm(timeout)
                record = evaluate_packed(expression, shot, remote_packed, packed_height)
                signal.alarm(0)
                arrays = (
                    unpack_height(record["data"])[0]
                    if packed_height
                    else unpack_full(record["data"])
                    if batch_full
                    else unpack_minimal(record["data"])
                )
                write_eq_batch(file, arrays, result)
                if packed_height and "ecegeom/ECEZH" not in file:
                    setup = file.require_group("ecegeom")
                    setup.attrs["source"] = "ELECTRONS"
                    dataset = setup.create_dataset(
                        "ECEZH", data=unpack_height(record["data"])[1]
                    )
                    dataset.attrs["expression"] = r"\ECE::TOP.SETUP.ECEZH"
                    dataset.attrs["transport"] = FDP_MDS_RELAY
                    dataset.attrs["acquisition"] = (
                        "packed_measured_ECE_sightline_height"
                    )
                    result["fetched"].append("ecegeom/ECEZH")
                result["batch_full" if batch_full else "batch_minimal"] = "decoded"
            except Exception as error:  # noqa: BLE001 -- upstream MDS exception classes
                signal.alarm(0)
                result["batch_error"] = f"{type(error).__name__}: {error}"
                if authentication_error(error):
                    result["authentication_failed"] = True
                elif missing_efit_error(error):
                    fallback_arrays = None
                    if batch_full and not missing_efit_tree_error(error):
                        try:
                            signal.alarm(timeout)
                            fallback = evaluate_packed(
                                packed_minimal_expression(), shot, remote_packed
                            )
                            signal.alarm(0)
                            fallback_arrays = unpack_minimal(fallback["data"])
                            write_eq_batch(file, fallback_arrays, result)
                            result["batch_full"] = "minimal_field_geometry_only"
                        except Exception as fallback_error:  # noqa: BLE001
                            signal.alarm(0)
                            result["minimal_fallback_error"] = (
                                f"{type(fallback_error).__name__}: {fallback_error}"
                            )
                            if authentication_error(fallback_error):
                                result["authentication_failed"] = True
                    if (
                        batch_full
                        and fallback_arrays is None
                        and not result.get("authentication_failed")
                        and not missing_efit_tree_error(error)
                    ):
                        try:
                            signal.alarm(timeout)
                            qrecord = evaluate_packed(
                                packed_full_expression(Q_INPUT_EQ_LEAVES),
                                shot,
                                remote_packed,
                            )
                            signal.alarm(0)
                            fallback_arrays = unpack_full(
                                qrecord["data"], Q_INPUT_EQ_LEAVES
                            )
                            write_eq_batch(file, fallback_arrays, result)
                            result["batch_full"] = "q_inputs_without_LCFS_boundary"
                        except Exception as qerror:  # noqa: BLE001
                            signal.alarm(0)
                            result["q_input_fallback_error"] = (
                                f"{type(qerror).__name__}: {qerror}"
                            )
                            if authentication_error(qerror):
                                result["authentication_failed"] = True
                    # Known missing bundles get at most three packed queries,
                    # checking q inputs independently of optional LCFS nodes.
                    # Do not repeat twelve individual absent-node queries.
                    for name in batch_leaves:
                        if f"eq/{name}" not in file:
                            result["missing"][f"eq/{name}"] = (
                                "not obtained after failed packed EFIT bundle(s): "
                                + result["batch_error"]
                            )
                    targets = [target for target in targets if target[0] != "eq"]
                    if fallback_arrays is None and not ece:
                        targets = []
        if result.get("authentication_failed"):
            targets = []
        for group, name, expression, tree, dims in targets:
            node = f"{group}/{name}"
            if node in file:
                continue
            try:
                signal.alarm(timeout)
                record = (
                    MdsSignal(
                        expression,
                        tree,
                        dims=dims,
                        location=FDP_MDS_RELAY,
                        fetch_units=bool(dims),
                    ).fetch(shot)
                    if remote_packed
                    else _fetch_mds(expression, tree, shot, dims=dims)
                    if dims
                    else MdsSignal(expression, tree, dims=[]).fetch(shot)
                )
                signal.alarm(0)
                values = np.asarray(record["data"])
                if not values.size:
                    raise ValueError("empty node")
                clock = None
                if dims:
                    clock = np.asarray(record[dims[0]], dtype=float)
                    if (
                        values.ndim != 1
                        or clock.shape != values.shape
                        or len(clock) < 2
                        or not np.isfinite(clock).all()
                        or not (np.diff(clock) > 0).all()
                    ):
                        raise ValueError("ECE waveform and time axes disagree")
                parent = file.require_group(group)
                parent.attrs["source"] = tree
                dataset = parent.create_dataset(
                    name, data=values, compression="gzip" if values.ndim else None
                )
                units = record.get("units") or {}
                dataset.attrs["units"] = str(
                    units.get("data", "") if isinstance(units, dict) else units
                )
                dataset.attrs["expression"] = expression
                if dims:
                    dataset.attrs.update(
                        t0_ms=float(clock[0]),
                        dt_ms=float(np.median(np.diff(clock))),
                        n=len(clock),
                    )
                file.flush()
                result["fetched"].append(node)
            except Exception as error:  # noqa: BLE001 -- upstream MDS exception classes
                signal.alarm(0)
                result["missing"][node] = f"{type(error).__name__}: {error}"
                if authentication_error(error):
                    result["authentication_failed"] = True
                    break
        file.attrs["fetch_missing_json"] = json.dumps(result["missing"])
    for tree in {"efit01", "ELECTRONS", ECE_TREE}:
        MdsSignal("", tree, dims=[]).cleanup_shot(shot)
    if result["fetched"]:
        time.sleep(pace)
    result["status"] = (
        "authentication_failed"
        if result.get("authentication_failed")
        else "partial"
        if result["missing"]
        else "complete"
    )
    return result


def fetch(args):
    if args.batch_minimal or args.batch_full:
        mode = "full" if args.batch_full else "minimal"
        proof_root = args.work.parent if args.work.name == "reference" else args.work
        proof = proof_root / (
            "batch_full_height_remote_validation.json"
            if args.packed_height
            else f"batch_{mode}_remote_validation.json"
            if args.remote_packed
            else f"batch_{mode}_validation.json"
        )
        if not proof.exists() or not json.loads(proof.read_text()).get("verified"):
            raise RuntimeError(
                f"run validate-batch for {mode} metadata before batch fetching"
            )
    shots = args.shots
    if args.cohort:
        shots += pd.read_csv(REPO / "data/events/catalog/cohort.csv").shot.tolist()
    if args.population_efit or args.population_ece:
        for path in sorted((args.prior_work / "shots").glob("*.json")):
            record = json.loads(path.read_text())
            selected = (
                record.get("qmin_available")
                if args.population_efit
                else not record.get("error") and bool(record.get("window_s"))
            )
            if selected:
                shots.append(int(record["shot"]))
    shots = sorted(set(map(int, shots)))
    manifest_name = (
        "geometry_height_fetch.json"
        if args.height_only
        else "geometry_minimal_fetch.json"
        if args.minimal
        else "geometry_fetch.json"
    )
    previous_path = args.work / manifest_name
    previous = json.loads(previous_path.read_text()) if previous_path.exists() else {}
    root = args.work.parent if args.work.name == "reference" else args.work
    for path in (
        root / "geometry_fetch.json",
        root / "geometry_minimal_fetch.json",
        root / "geometry_height_fetch.json",
        root / "reference/geometry_fetch.json",
        root / "authentication_failed.json",
    ):
        if path.exists() and json.loads(path.read_text()).get("authentication_failed"):
            raise RuntimeError(
                "previous fetch stopped on authentication; no further fetch"
            )
    results = {str(r["shot"]): r for r in previous.get("records", [])}
    jobs = iter(
        (
            shot,
            str(args.work),
            args.pace,
            args.node_timeout,
            args.ece,
            args.minimal,
            args.batch_minimal,
            args.batch_full,
            args.height_only,
            args.fixed_grid,
            args.remote_packed,
            args.packed_height,
        )
        for shot in shots
    )
    auth_failed = False
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        pending = {}
        for _ in range(args.workers):
            job = next(jobs, None)
            if job:
                pending[pool.submit(fetch_one, job)] = job[0]
        while pending:
            completed, _ = wait(pending, return_when=FIRST_COMPLETED)
            for future in completed:
                shot = pending.pop(future)
                try:
                    result = future.result()
                except Exception as error:  # noqa: BLE001 -- preserve worker failure
                    result = {
                        "shot": shot,
                        "status": "worker_failed",
                        "error": f"{type(error).__name__}: {error}",
                    }
                    result["authentication_failed"] = authentication_error(error)
                results[str(shot)] = result
                auth_failed |= result.get("authentication_failed", False)
                report = {
                    "shots_requested": shots,
                    "authentication_failed": auth_failed,
                    "workers": args.workers,
                    "pace_s": args.pace,
                    "mode": "minimal_field_axis_boundary"
                    if args.minimal
                    else "height_only"
                    if args.height_only
                    else "full_q1",
                    "expected_eq_nodes": (
                        ()
                        if args.height_only
                        else MINIMAL_EQ_LEAVES
                        if args.minimal
                        else EQ_LEAVES
                    ),
                    "records": list(results.values()),
                }
                save_json(previous_path, report)
                output_name = "reference_fetch.json" if args.ece else manifest_name
                if args.minimal or len(results) > 500:
                    from collections import Counter

                    small = {
                        key: value for key, value in report.items() if key != "records"
                    }
                    small["record_status_counts"] = dict(
                        Counter(row["status"] for row in results.values())
                    )
                    small["source_records"] = str(previous_path)
                    save_json(OUTPUT / output_name, small)
                else:
                    save_json(OUTPUT / output_name, report)
                print(
                    json.dumps(
                        {
                            "shot": shot,
                            "status": result["status"],
                            "nodes": len(result.get("fetched", [])),
                            "missing": result.get("missing", {}),
                            "batch": result.get(
                                "batch_full", result.get("batch_minimal")
                            ),
                            "batch_error": result.get("batch_error"),
                        }
                    ),
                    flush=True,
                )
                if not auth_failed:
                    job = next(jobs, None)
                    if job:
                        pending[pool.submit(fetch_one, job)] = job[0]
    if auth_failed:
        print("STOP: authentication failure; remaining shots not fetched", flush=True)


def audit(args):
    """Report fixed-grid evidence and checked/missing q=1 support per shot."""
    from collections import Counter

    from labeler.config import Paths
    from labeler.events import equilibrium
    from labeler.events.verify import NoDataError
    from labeler.features import resolve_archive
    from labeler.sawtooth.geometry import audit_frequency_grid, load_radius_geometry

    paths = Paths.from_env()
    shots = args.shots
    if args.cohort:
        shots += pd.read_csv(REPO / "data/events/catalog/cohort.csv").shot.tolist()
    if not shots:
        shots = [int(path.stem) for path in (args.work / "geometry").glob("*.h5")]
    grid = audit_frequency_grid(args.archive)
    records = []
    for shot in sorted(set(map(int, shots))):
        row = {"shot": shot, "local_scalars": {}, "qpsi_available": False}
        for name in ("bt", "r0", "qmin", "qpsi"):
            try:
                array = equilibrium.signal(shot, name, paths, fetch=False)
                row["local_scalars"][name] = "local_store"
            except (NoDataError, ValueError):
                archive_name = "qpsi" if name == "qmin" else name
                arrays, _ = resolve_archive.resolve(shot, [archive_name])
                array = arrays.get(archive_name)
                row["local_scalars"][name] = (
                    ("archive_qpsi_min" if name == "qmin" else "archive")
                    if array is not None
                    else "unavailable"
                )
            if name == "qpsi" and array is not None:
                row["qpsi_available"] = bool(np.isfinite(array.y).any())
        sources = [
            args.archive / "raw" / f"{shot}.h5",
            args.work / "geometry" / f"{shot}.h5",
        ]
        clock = np.arange(0, 6.01, 0.02)
        for path in sources:
            try:
                with h5py.File(path, "r", locking=False) as file:
                    if "eq/gtime" in file:
                        clock = np.asarray(file["eq/gtime"], dtype=float) / 1000
                        row["full_EFIT_source"] = str(path)
                        row["qpsi_available"] = "eq/qpsi" in file
                        break
            except OSError:
                continue
        radius, info = load_radius_geometry(
            shot,
            clock,
            48,
            paths,
            archive_root=args.archive,
            metadata_root=args.work / "geometry",
        )
        keep = (
            "status",
            "frequency_scope",
            "frequency_source",
            "frequency_order_supported",
            "frequency_order_break_channels",
            "near_duplicate_frequency_pairs",
            "axis_source",
            "field_product_source",
            "q1_checked",
            "q1_status",
            "q1_source",
            "q1_low_supported_samples",
            "q1_high_supported_samples",
            "q1_no_axis_connected_surface_slices",
            "lcfs_outer_supported_samples",
            "sightline_height_source",
            "same_shot_archive",
        )
        row["radius_geometry"] = {key: info[key] for key in keep if key in info}
        row["q1_major_radius_check_supported"] = bool(
            radius is not None
            and radius.q1_high_R_m is not None
            and np.isfinite(radius.q1_high_R_m).any()
        )
        row["geometry_samples"] = len(clock)
        records.append(row)
    report = {
        "archive_frequency_grid": grid,
        "shots": len(records),
        "qpsi_available_shots": sum(r["qpsi_available"] for r in records),
        "q1_checked_shots": sum(
            r["radius_geometry"].get("q1_checked", False) for r in records
        ),
        "q1_major_radius_supported_shots": sum(
            r["q1_major_radius_check_supported"] for r in records
        ),
        "radius_status_counts": dict(
            Counter(r["radius_geometry"]["status"] for r in records)
        ),
        "by_shot": records,
        "limitations": (
            "nominal vacuum second harmonic; geometric rho uses outer LCFS R "
            "and is not flux rho; EFIT01 q may be biased relative to MSE EFIT; "
            "checked profiles without an axis-connected q=1 crossing are "
            "reported separately from missing equilibrium data"
        ),
    }
    save_json(args.work / "geometry_metadata_audit.json", report)
    if len(records) > 500:
        small = {key: value for key, value in report.items() if key != "by_shot"}
        small["shots_audited"] = [row["shot"] for row in records]
        small["source_records"] = str(args.work / "geometry_metadata_audit.json")
        save_json(OUTPUT / "geometry_metadata_audit.json", small)
    else:
        save_json(OUTPUT / "geometry_metadata_audit.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "shots",
                    "qpsi_available_shots",
                    "q1_checked_shots",
                    "q1_major_radius_supported_shots",
                    "radius_status_counts",
                )
            }
        ),
        flush=True,
    )


def field_audit(args):
    """Inventory broad local scalar metadata and train-only F/Bt consistency.

    The empirical ratio is descriptive. It is not relabeled as a measured
    BT0 reference radius and is not used to replace per-shot EFIT F.
    """
    from collections import Counter

    from labeler.config import Paths
    from labeler.events import equilibrium
    from labeler.events.panels.ece_geometry import align_q
    from labeler.events.verify import NoDataError
    from labeler.features import resolve_archive

    paths = Paths.from_env()
    progress_path = args.work / "field_audit_progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {}
    if progress.get("prior_work") != str(args.prior_work):
        progress = {"prior_work": str(args.prior_work)}
    shots = progress.get("shots")
    if shots is None:
        shots = []
        for path in sorted((args.prior_work / "shots").glob("*.json")):
            record = json.loads(path.read_text())
            if not record.get("error") and record.get("window_s"):
                shots.append(int(record["shot"]))
        progress["shots"] = shots
        save_json(progress_path, progress)
        print(f"selected {len(shots)} successful ECE records", flush=True)
    archive_index = resolve_archive.shot_index()
    archived = {
        int(shot): set(fields) for shot, fields in progress.get("archived", {}).items()
    }
    if not progress.get("archive_complete"):
        for path in set(archive_index.values()):
            with h5py.File(path, "r", locking=False) as file:
                for shot in shots:
                    if archive_index.get(shot) != path:
                        continue
                    keys = set(file[str(shot)])
                    archived[shot] = {
                        name
                        for name, locator in (
                            ("bt", "bt"),
                            ("r0", "rmaxis_EFIT01"),
                            ("aminor", "aminor_EFIT01"),
                        )
                        if locator in keys
                    }
        progress.update(
            archived={str(shot): sorted(fields) for shot, fields in archived.items()},
            archive_complete=True,
        )
        save_json(progress_path, progress)
        print(f"inspected {len(archived)} scalar archive shots", flush=True)
    rows = progress.get("rows", [])
    completed_shots = {row["shot"] for row in rows}
    for shot in shots:
        if shot in completed_shots:
            continue
        found = {}
        for path in (
            paths.features_file(shot),
            paths.corpus_file(shot),
            paths.raw_cache / f"{shot}_processed.h5",
        ):
            if not path.exists():
                continue
            with h5py.File(path, "r", locking=False) as file:
                for name in ("bt", "r0", "aminor"):
                    if (
                        name in file
                        and "ydata" in file[name]
                        and file[name]["ydata"].shape[-1] > 1
                    ):
                        found.setdefault(name, "local_store")
        for name in archived.get(shot, set()):
            found.setdefault(name, "archive")
        rows.append({"shot": shot, "fields": found})
        if len(rows) % 250 == 0:
            progress["rows"] = rows
            save_json(progress_path, progress)
            print(f"inspected {len(rows)}/{len(shots)} local scalar stores", flush=True)
    progress["rows"] = rows
    save_json(progress_path, progress)
    cohort = pd.read_csv(REPO / "data/events/catalog/cohort.csv")
    train = set(cohort.loc[cohort.split == "train", "shot"])
    calibration = []
    for path in sorted((args.work / "geometry").glob("*.h5")):
        shot = int(path.stem)
        if shot not in train:
            continue
        try:
            try:
                bt = equilibrium.signal(shot, "bt", paths, fetch=False)
            except (NoDataError, ValueError):
                arrays, _ = resolve_archive.resolve(shot, ["bt"])
                if "bt" not in arrays:
                    continue
                bt = equilibrium.canonical(arrays["bt"], "bt")
            with h5py.File(path, "r", locking=False) as file:
                clock = np.asarray(file["eq/gtime"], dtype=float) / 1000
                product = np.asarray(file["eq/fpol"], dtype=float)[:, -1]
                axis = np.asarray(file["eq/rmaxis"], dtype=float)
            field = align_q(clock, bt.x, bt.y)[0]
            keep = (
                (clock > 0.5)
                & (abs(field) > 1)
                & (axis > 1.5)
                & (axis < 1.9)
                & np.isfinite(product)
            )
            ratio = np.abs(product[keep] / field[keep])
            if len(ratio) < 10:
                continue
            calibration.append(
                {
                    "shot": shot,
                    "split": "train",
                    "samples": len(ratio),
                    "effective_F_per_Bt_median_m": float(np.median(ratio)),
                    "effective_F_per_Bt_q05_m": float(np.quantile(ratio, 0.05)),
                    "effective_F_per_Bt_q95_m": float(np.quantile(ratio, 0.95)),
                    "bt_resolver": bt.attrs.get("resolver"),
                    "F_source": str(path),
                }
            )
        except (OSError, KeyError, ValueError):
            continue
    medians = [row["effective_F_per_Bt_median_m"] for row in calibration]
    report = {
        "population_selection": "successful prior fixed-rule records with ECE window",
        "shots": len(shots),
        "shots_selected": shots,
        "potential_local_bt_shots": sum("bt" in r["fields"] for r in rows),
        "potential_local_bt_axis_shots": sum(
            {"bt", "r0"} <= set(r["fields"]) for r in rows
        ),
        "potential_local_bt_axis_aminor_shots": sum(
            {"bt", "r0", "aminor"} <= set(r["fields"]) for r in rows
        ),
        "field_source_counts": dict(
            Counter(
                f"{name}:{source}" for r in rows for name, source in r["fields"].items()
            )
        ),
        "train_calibration_shots": len(calibration),
        "effective_F_per_Bt_shot_median_m": (
            float(np.median(medians)) if medians else None
        ),
        "train_calibration": calibration,
        "limitations": (
            "scalar availability is a metadata inventory, not finite waveform "
            "coverage; F/Bt is an empirical effective field-product scale, not "
            "proof of a physical BT0 reference radius; rmaxis+aminor is not the "
            "outer LCFS R because the magnetic axis is shifted; no inferred "
            "reference or approximate LCFS is substituted in labeling"
        ),
        "recommended_minimum_per_shot": list(MINIMAL_EQ_LEAVES),
    }
    save_json(args.work / "local_geometry_inventory.json", {**report, "by_shot": rows})
    save_json(OUTPUT / "field_geometry_audit.json", report)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "shots",
                    "potential_local_bt_shots",
                    "potential_local_bt_axis_shots",
                    "potential_local_bt_axis_aminor_shots",
                    "train_calibration_shots",
                    "effective_F_per_Bt_shot_median_m",
                )
            }
        ),
        flush=True,
    )


def reference(args):
    """Independent central-ECE derivative period/amplitude reference checks.

    This diagnostic never changes the frozen rule. Published figure windows
    are checked with a 40ms minimum peak spacing. Pre/post medians use the
    last/first 10% of each measured local cycle, excluding the 0.3ms edge.
    """
    from scipy.ndimage import gaussian_filter1d
    from scipy.signal import find_peaks, resample_poly

    from labeler.config import Paths
    from labeler.sawtooth.geometry import load_radius_geometry, select_core

    records = []
    for shot in args.shots or [141182, 141195]:
        path = args.work / "geometry" / f"{shot}.h5"
        result = {
            "shot": shot,
            "published_period_ms": [85, 5],
            "published_amplitude": [0.35, 0.02],
            "published_reference": {
                "source_digest": str(
                    REPO.parent / "FusionAIHub/.tmp/label_papers/Muscatello_ST.md"
                ),
                "endpoint_R_m": {"channel_0": 2.22, "channel_39": 1.44},
                "figure_2_central_te_range_kev": [2.5, 3.5],
                "condition_database_central_te_range_kev": [2.0, 5.0],
            },
            "source": str(path),
        }
        if not path.exists():
            result["status"] = "unavailable_local_reference"
            records.append(result)
            continue
        try:
            with h5py.File(path, "r", locking=False) as file:
                rows, clock = [], None
                for channel in range(40):
                    dataset = file[f"ece/ECEVS{channel + 1:02d}"]
                    dt = float(dataset.attrs["dt_ms"]) / 1000
                    t0 = float(dataset.attrs["t0_ms"]) / 1000
                    start = max(0, int(np.ceil((2.0 - t0) / dt)))
                    stop = min(len(dataset), int(np.ceil((5.1 - t0) / dt)))
                    factor = max(1, round(1 / (dt * 10000)))
                    values = np.asarray(dataset[start:stop], dtype=float)
                    row = resample_poly(values, 1, factor, padtype="reflect")
                    times = t0 + (start + factor * np.arange(len(row))) * dt
                    if clock is None:
                        clock = times
                    elif not np.allclose(clock, times, atol=1e-7):
                        raise ValueError("reference channel clocks disagree")
                    rows.append(row)
            values = np.asarray(rows)
            radius, geometry = load_radius_geometry(
                shot,
                clock,
                40,
                Paths.from_env(),
                archive_root=args.archive,
                metadata_root=args.work / "geometry",
            )
            core = select_core(values, radius=radius)
            central = core.info["central_channel"]
            trace = values[central]
            dt = float(np.median(np.diff(clock)))
            smooth = gaussian_filter1d(trace, max(1, 0.00015 / dt))
            edge = -np.gradient(smooth, dt)
            scale = max(1e-6, 1.4826 * np.median(abs(edge - np.median(edge))))
            picks, _ = find_peaks(
                edge / scale, height=6, distance=max(1, round(0.04 / dt))
            )
            windows = []
            for left, right in ((2.7, 3.0), (4.55, 4.8)):
                selected = picks[(clock[picks] >= left) & (clock[picks] <= right)]
                events = []
                for pick in selected:
                    stamp = float(clock[pick])
                    position = int(np.searchsorted(picks, pick))
                    previous_period = (
                        stamp - clock[picks[position - 1]] if position else 0.085
                    )
                    next_period = (
                        clock[picks[position + 1]] - stamp
                        if position + 1 < len(picks)
                        else 0.085
                    )
                    before = trace[
                        (clock >= stamp - 0.1 * previous_period)
                        & (clock < stamp - 0.0003)
                    ]
                    after = trace[
                        (clock >= stamp + 0.0003) & (clock < stamp + 0.1 * next_period)
                    ]
                    pre, post = float(np.median(before)), float(np.median(after))
                    amplitude = (pre - post) / pre if pre > 0 else None
                    events.append(
                        {
                            "time_s": stamp,
                            "amplitude": amplitude,
                            "pre_kev": pre,
                            "post_kev": post,
                            "edge_z": float(edge[pick] / scale),
                        }
                    )
                gaps = np.diff(clock[selected]) * 1000
                amplitudes = [
                    r["amplitude"] for r in events if r["amplitude"] is not None
                ]
                within = (clock >= left) & (clock <= right)
                band_levels = np.median(values[[15, 16]][:, within], axis=1)
                endpoint_radius = {}
                if radius is not None:
                    for channel, published_r in ((0, 2.22), (39, 1.44)):
                        locations = radius.R_m[channel, within]
                        locations = locations[np.isfinite(locations)]
                        endpoint_radius[f"channel_{channel}"] = (
                            {
                                "minimum_m": float(np.min(locations)),
                                "median_m": float(np.median(locations)),
                                "maximum_m": float(np.max(locations)),
                                "published_R_m": published_r,
                                "median_minus_published_m": float(
                                    np.median(locations) - published_r
                                ),
                            }
                            if len(locations)
                            else None
                        )
                windows.append(
                    {
                        "window_s": [left, right],
                        "central_median_te_kev": float(np.median(trace[within])),
                        "median_precrash_te_kev": float(
                            np.median([r["pre_kev"] for r in events])
                        )
                        if events
                        else None,
                        "channel_15_median_te_kev": float(band_levels[0]),
                        "channel_16_median_te_kev": float(band_levels[1]),
                        "channel_16_to_15_median_level_ratio": (
                            float(band_levels[1] / band_levels[0])
                            if band_levels[0] > 0
                            else None
                        ),
                        "nominal_endpoint_R_m": endpoint_radius,
                        "crashes": events,
                        "median_period_ms": float(np.median(gaps))
                        if len(gaps)
                        else None,
                        "median_amplitude": (
                            float(np.median(amplitudes)) if amplitudes else None
                        ),
                        "periods_ms": gaps.tolist(),
                    }
                )
            result.update(
                status="independent_reference_measurement",
                central_channel=central,
                core_geometry=core.info,
                radius_geometry=geometry,
                windows=windows,
                absolute_temperature_source_check=(
                    "retrieved nearest-axis TECEF precrash levels exceed the "
                    "published 2-5keV condition-database range; historical "
                    "calibration, band equivalence and optical assumptions "
                    "are unverified; no correction applied"
                    if any(
                        window["median_precrash_te_kev"] is not None
                        and window["median_precrash_te_kev"] > 5
                        for window in windows
                    )
                    else "reported retrieved levels; historical calibration, "
                    "band equivalence and optical assumptions are unverified"
                ),
                nearest_published_crash_ms=(
                    float(clock[picks[np.argmin(abs(clock[picks] - 2.8371))]] * 1000)
                    if shot == 141182 and len(picks)
                    else None
                ),
            )
            np.savez_compressed(
                args.work / f"reference_{shot}.npz",
                t=clock,
                y=values,
                central_channel=central,
                crash_s=clock[picks],
            )
        except (OSError, KeyError, ValueError) as error:
            result.update(status="reference_incomplete", error=str(error))
        records.append(result)
    report = {
        "method": "independent central-ECE derivative, frozen diagnostic parameters",
        "parameters": {
            "minimum_spacing_s": 0.04,
            "edge_z": 6,
            "gaussian_sigma_s": 0.00015,
            "cycle_pre_post_fraction": 0.1,
            "edge_exclusion_s": 0.0003,
        },
        "by_shot": records,
        "limitations": (
            "checks published figure windows and approximate period/amplitude; "
            "EFIT01 is not published MSE EFIT; nominal channel location has "
            "no relativistic or optical-depth correction; absolute retrieved "
            "TECEF levels differ from published central temperatures and "
            "historical calibration/band equivalence is unverified; no detector tuning"
        ),
    }
    save_json(args.work / "muscatello_reference.json", report)
    save_json(OUTPUT / "muscatello_reference.json", report)
    print(
        json.dumps(
            {"by_shot": [{"shot": r["shot"], "status": r["status"]} for r in records]}
        ),
        flush=True,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "fetch",
            "audit",
            "field-audit",
            "height-audit",
            "frequency-audit",
            "reference",
            "validate-batch",
            "validate-height",
            "validate-ece",
        ),
    )
    parser.add_argument("--work", type=Path, default=DEFAULT_WORK)
    parser.add_argument("--archive", type=Path, default=ARCHIVE)
    parser.add_argument("--shots", type=int, nargs="*", default=[])
    parser.add_argument("--cohort", action="store_true")
    parser.add_argument("--population-efit", action="store_true")
    parser.add_argument("--population-ece", action="store_true")
    parser.add_argument("--minimal", action="store_true")
    parser.add_argument("--batch-minimal", action="store_true")
    parser.add_argument("--batch-full", action="store_true")
    parser.add_argument("--height-only", action="store_true")
    parser.add_argument("--fixed-grid", action="store_true")
    parser.add_argument("--remote-packed", action="store_true")
    parser.add_argument("--packed-height", action="store_true")
    parser.add_argument("--prior-work", type=Path, default=DEFAULT_WORK.parent / "fix2")
    parser.add_argument("--ece", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--pace", type=float, default=1)
    parser.add_argument("--node-timeout", type=int, default=120)
    args = parser.parse_args()
    if not 1 <= args.workers <= 3 or args.pace < 1:
        parser.error("fetch requires 1..3 workers and pace >= 1 second")
    if args.minimal and (args.population_efit or args.ece):
        parser.error("EFIT-q checks and reference ECE require the full metadata bundle")
    if args.batch_minimal and not args.minimal:
        parser.error("--batch-minimal requires --minimal")
    if args.batch_full and args.minimal:
        parser.error("--batch-full requires full EFIT metadata")
    if args.packed_height and not (args.batch_full and args.remote_packed):
        parser.error("--packed-height requires --batch-full --remote-packed")
    if (
        args.stage == "fetch"
        and args.remote_packed
        and not (args.batch_full or args.batch_minimal)
    ):
        parser.error("--remote-packed requires a packed fetch mode")
    if args.height_only and (args.minimal or args.ece or args.batch_full):
        parser.error("--height-only cannot be combined with EFIT or ECE modes")
    if args.stage == "fetch":
        fetch(args)
    elif args.stage == "audit":
        audit(args)
    elif args.stage == "field-audit":
        field_audit(args)
    elif args.stage == "validate-batch":
        validate_batch(args)
    elif args.stage == "validate-height":
        validate_height(args)
    elif args.stage == "validate-ece":
        validate_ece(args)
    elif args.stage == "height-audit":
        height_audit(args)
    elif args.stage == "frequency-audit":
        frequency_audit(args)
    else:
        reference(args)


if __name__ == "__main__":
    main()
