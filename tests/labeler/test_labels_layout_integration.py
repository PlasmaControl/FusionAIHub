"""The committed format inputs retain the RWM CLI contract on both shot sets."""
import csv
import hashlib
import json
import runpy
import subprocess
from pathlib import Path

import pandas as pd
import yaml

from labeler import run
from labeler.events import databases as db
from labeler.events import schema

REPO = Path(__file__).resolve().parents[2]

AUTHORED_README_PREFIXES = {
    "alfven_eigenmode": (
        "Alfven Eigenmode", 3075,
        "55195688e799d0de44260e2f7ca6ff42cbcf673f6deb54573eb2642ee493ad4c",
    ),
    "detachment": (
        "Detachment", 2092,
        "12fb72534513e22d80d5705f6e0b3e49c7c9c60b78abf8343851ff8acdae1e3e",
    ),
    "edge_localized_mode": (
        "Edge Localized Mode", 2818,
        "4a1426530541c228eb189b355de43ba7dcab9a479c4c8a1d1c45a9d4e5bb4934",
    ),
    "high_confinement_mode": (
        "High Confinement Mode", 2303,
        "50a9e9770aa95ac3b83d72f709c6eaae17748b56babc92f90c23bd0733582347",
    ),
    "improved_energy_confinement_mode": (
        "Improved Energy Confinement Mode", 1867,
        "83e527284e6b8dc59642cad6507d70aa68186510710b3799a8de788078b96f04",
    ),
    "low_confinement_mode": (
        "Low Confinement Mode", 1952,
        "6c5526bac01851093f9b75918e2e4375db711a3f005dd9175d73f8a46519c528",
    ),
    "minimum_safety_factor": (
        "Minimum Safety Factor", 2669,
        "53a2cebb45c4ff7867d62180f4b87f31fa0a59e930fc5eec9f0a561a4ae4c3b3",
    ),
    "poloidal_beta": (
        "Poloidal Beta", 1295,
        "b651508b197c480142bc638649d75035e0d949aa579c34da4f7288550491a8f5",
    ),
    "resistive_wall_mode": (
        "Resistive Wall Mode", 2564,
        "08deff5c613b5976c4bcfff6be27b0626b1dc6bb4dd9650cba9c7dc5a25a24f3",
    ),
    "sawtooth_oscillation": (
        "Sawtooth Oscillation", 2584,
        "ba7b3b97cabca7d5a4019548084bf55cd8cc909bcf25cde95f9649aa6dcdb80d",
    ),
    "tearing_mode": (
        "Tearing Mode", 2996,
        "f42dcdeeb6d424ff399f42bf75407cbc731d218b85404950c052100b20d51f6f",
    ),
    "wide_pedestal_quiescent_h_mode": (
        "Wide Pedestal Quiescent H-Mode", 2927,
        "6423613c64c1d4bbcbd63975b800a4b3dc934de3973609524b3f535783c9eca1",
    ),
}

README_SECTIONS = [
    "Description", "Method", "Provenance", "Models", "Alias", "Reference",
    "Contact",
]


def test_databases_only_on_all_committed_rwm_shots_writes_56_events_33_sources(
    tmp_path, monkeypatch,
):
    labels = REPO / "data/events"
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(labels))
    shots = sorted(set().union(*(db.shots(s, labels) for s in db.load_manifest(labels))))
    assert run.main(["events", "--databases-only", "--root", str(tmp_path),
                     "--run-id", "rwm-integration", "--shots", *map(str, shots)]) == 0
    payload = json.loads((tmp_path / "runs/events/rwm-integration.json").read_text())
    assert payload["totals"]["n_events"] == 56
    assert payload["totals"]["n_source_records"] == 33
    assert payload["totals"]["n_shots_named"] == 33
    events = pd.concat([schema.read_events(tmp_path / f"events/{s}_events.parquet")
                        for s in shots])
    sources = pd.concat([schema.read_sources(tmp_path / f"events/{s}_sources.parquet")
                         for s in shots])
    assert len(events) == 56 and len(sources) == 33
    assert set(events.evidence_kind) == {"database"}
    assert events.confidence.isna().all()
    assert events.t_cov0_s.isna().all() and sources.t_cov1_s.isna().all()


def test_rwm_500_scan_exports_empty_table_with_completed_run_provenance(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("LABELMAKER_LABEL_TABLES", str(REPO / "data/events"))
    shot_list = REPO / "configs/shot_design/shot_lists/recommender_v1.yaml"
    shots = [r["shot"] for r in yaml.safe_load(shot_list.read_text())["shots"]]
    assert len(shots) == 500
    assert run.main(["events", "--databases-only", "--root", str(tmp_path),
                     "--run-id", "rwm-zero", "--shots", *map(str, shots)]) == 0
    main = runpy.run_path(str(REPO / "scripts/labeler/labels_extend.py"))["main"]
    out = tmp_path / "resistive_wall_mode/extend_rwm/recommender_v1.csv"
    assert main(["--category", "resistive_wall_mode", "--producer", "rwm",
                 "--shot-list", str(shot_list), "--root", str(tmp_path),
                 "--run-id", "rwm-zero", "--out", str(out)]) == 0
    assert pd.read_csv(out).empty
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["n_requested_shots"] == 500 and meta["n_rows"] == 0
    assert meta["made_from"][0]["run_id"] == "rwm-zero"
    assert meta["made_from"][0]["producer"] == "rwm"
    assert meta["full_events_root"] == "events"
    assert not meta["full_events_root"].startswith("/tmp")
    assert meta["run_metadata"] == "runs/events/rwm-zero.json"
    assert not list((tmp_path / "events").glob("*.parquet"))


def test_gitignore_allows_new_label_tables_to_be_tracked():
    # --no-index also checks tracked paths; the novel name catches future tables.
    result = subprocess.run([
        "git", "check-ignore", "--no-index", "data/events/tables.yaml",
        "data/events/future_category/raw/future_table.csv",
    ], cwd=REPO, capture_output=True, text=True, check=False)
    assert result.returncode == 1, result.stdout + result.stderr
    assert result.stdout == ""


def test_discrete_label_inventory_has_the_roadmap_input_schema():
    with (REPO / "data/events/discrete_labels.csv").open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == [
            "Name", "Category", "Database Exists", "Auto Classification Exists",
            "Priority", "Feasability", "Feasability Reason", "Proposed Method",
            "Link", "Completed", "Notes",
        ]
        rows = list(reader)
    assert len(rows) >= 37
    assert all(None not in row and row["Priority"] in {"1", "2", "3", "4", "5"}
               for row in rows)


def test_category_readmes_preserve_authored_prefixes_and_append_tables_last():
    events = REPO / "data/events"
    readmes = {path.parent.name: path for path in events.glob("*/README.md")}
    assert set(readmes) == set(AUTHORED_README_PREFIXES)

    for category, (title, expected_length, expected_sha256) in (
        AUTHORED_README_PREFIXES.items()
    ):
        content = readmes[category].read_bytes()
        marker = b"\n## Tables\n"
        assert content.count(marker) == 1, category
        authored, tables = content.split(marker)

        assert len(authored) == expected_length, category
        assert hashlib.sha256(authored).hexdigest() == expected_sha256, category
        assert tables.strip(), category
        assert b"\n## " not in tables, category

        lines = authored.decode().splitlines()
        headings = [line for line in lines if line.startswith(("# ", "## "))]
        sections = README_SECTIONS.copy()
        if category == "minimum_safety_factor":
            sections.insert(1, "Categories")
        assert headings == [f"# {title}", *(f"## {name}" for name in sections)]

        section_starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
        section_ends = [*section_starts[1:], len(lines)]
        for start, end in zip(section_starts, section_ends):
            assert "\n".join(lines[start + 1:end]).strip(), (category, lines[start])
