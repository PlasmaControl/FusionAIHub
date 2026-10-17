"""The labels-only Hugging Face release: blind shots withheld, gold and holdout shots
in their own splits everywhere, nothing identifying (roster placeholders included),
withheld tables out unless named, the grid encoded losslessly, nothing it uses left
out silently, the repo kept private and each release tagged once, no build or restore
that replaces what is not a release, and the tables restorable only as released."""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from labeler.config import Paths
from labeler.events import hf_export, taxonomy
from labeler.events.review import labels, silver

SHOT, OTHER, GOLD, BLIND, CHALLENGE_SHOT = 170815, 170816, 171234, 180001, 172000
NAME, LOGIN = "Jane Doe", "jdoe1"
HEADER = "shot,category,t_start,t_end,confidence,attrs\n"
CHALLENGE = "disruption/format/disruption_challenge_format_2026_v1"
PLACEHOLDER = "1,unverified,false,alice;bob,,EXAMPLE - replace with a real shot\n"


def _table(path, rows, meta=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        HEADER + "".join(f"{s},{c},{a},{b},{p},{x}\n" for s, c, a, b, p, x in rows)
    )
    if meta is not None:
        path.with_suffix(".meta.json").write_text(json.dumps(meta))
    return path


@pytest.fixture
def paths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        hf_export,
        "WITHHELD",
        {CHALLENGE: {"why": "permission is not a licence", "get": "ask its owners"}},
    )
    tables = tmp_path / "events"
    (tables / "catalog").mkdir(parents=True)
    (tables / "catalog" / "cohort.csv").write_text("shot\n")
    monkeypatch.setattr(
        silver,
        "read_cohort",
        lambda path: pd.DataFrame(
            {
                "shot": [BLIND, OTHER, GOLD, 190000],
                "split": ["test", "train", "train", "val"],
                "blind": [True, False, False, False],
            }
        ),
    )
    config = tmp_path / "config"
    config.mkdir()
    (config / "events.yaml").write_text(
        "taxonomy:\n  disruption:\n    display: Disruption\n"
        "raw_datasets:\n  - stem: theirs\n"
        f"    provenance: copied from {tmp_path}/raw/theirs.csv by a colleague\n"
    )
    (config / "best_models.yaml").write_text(
        "version: 1\nevents:\n  disruption:\n    model: rule\n    task: detector\n"
        '    headline: "phase-controlled AUROC 0.534: no skill"\n'
        "    scores:\n      reference: a rule\n      interval: exploratory\n"
        "      phase_controlled_auroc: {value: 0.534, lo: 0.4, hi: 0.6}\n"
        "    caveats: [no input senses it]\n"
        "    todo: a stream is changing it\n"
    )
    event = tables / "disruption"
    (event / "shots.csv").parent.mkdir()
    (event / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n"
        f"{SHOT},verified,false,{NAME},,\n{BLIND},unverified,false,,,\n"
        f"{OTHER},unverified,true,,,\n{PLACEHOLDER}"
    )
    crowd = '"{""iscrowd"": 1, ""secret_key"": ""x"", ""n"": 1}"'
    _table(event / "review" / "labels.csv", [(SHOT, 1, 100, 300, "", crowd)])
    (event / "review" / "history.jsonl").write_text(
        json.dumps(
            {
                "shot": SHOT,
                "reviewer": LOGIN,
                "name": NAME,
                "saved_at": "2026-10-01T00:00:00",
            }
        )
        + "\n"
    )
    meta = {
        "categories": {"0": "absent", "1": "present"},
        "category": "disruption",
        "made_from": [f"{tmp_path}/raw/theirs.csv"],
        "made_by": "formatter.py",
    }
    _table(
        event / "format" / "disruption_format_2026_v1.csv",
        [(s, 1, 50, 400, 1.0, crowd) for s in (SHOT, OTHER, GOLD, BLIND)],
        meta,
    )
    _table(
        event / "format" / "disruption_challenge_format_2026_v1.csv",
        [(CHALLENGE_SHOT, 1, 10, 20, 1.0, "")],
        meta,
    )
    suggestion = _table(
        tmp_path
        / "root"
        / "suggestions"
        / "rule"
        / "v1"
        / "disruption_suggest_rule_v1.csv",
        [(s, 1, 0, 500, 0.5, "") for s in (SHOT, OTHER, GOLD, BLIND)],
        {**meta, "method": "rule", "version": "v1"},
    )
    labels.write_pointer(event, suggestion, method="rule", version="v1")
    (tables / "silver").mkdir()
    (tables / "silver" / "silver_shots.csv").write_text(
        "shot,batch,order,pool,classes,sources,notes\n"
        f"{SHOT},1,1,corpus,,format,\n{OTHER},1,2,corpus,,format,\n"
    )
    (tables / "gold").mkdir()
    (tables / "gold" / "gold_shots.csv").write_text(
        f"shot,run_day,year,classes,sources,notes\n{GOLD},20260101,2026,,,\n"
    )
    return Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables,
        config_dir=config,
        raw_cache=tmp_path / "raw",
    )


def _texts(folder: Path):
    for path in sorted(p for p in folder.rglob("*") if p.is_file()):
        if path.suffix == ".parquet":
            frame = pd.read_parquet(path)
            yield path, frame, frame.astype(str).to_csv(index=False)
        else:
            yield path, None, path.read_text()


def test_release_withholds_blind_shots_and_scrubs_identity(paths, tmp_path):
    out = tmp_path / "release"
    report = hf_export.build(paths, out)
    assert report["problems"] == []
    assert report["withheld_blind_rows"] == 2  # the format and suggestion rows
    assert report["dropped_attrs"] == {"secret_key": 2}
    assert set(report["withheld"]) == {CHALLENGE}
    # the roster's placeholder row (alice;bob) is no reviewer and no shot
    assert hf_export.context(paths).reviewers == {NAME: "R1"}
    for path, frame, text in _texts(out):
        if frame is not None and "shot" in frame:
            assert not {BLIND, CHALLENGE_SHOT, 1} & set(frame.shot), path
        if frame is not None and {"shot", "split"} <= set(frame):
            splits = dict(zip(frame.shot, frame.split, strict=True))
            assert splits.get(GOLD, "test") == "test", path
            assert splits.get(OTHER, "holdout") == "holdout", path
        for leak in (str(BLIND), str(tmp_path), NAME, LOGIN, "secret_key", "alice"):
            assert leak not in text, (path, leak)
        assert "EXAMPLE" not in text, path
    assert CHALLENGE not in set(pd.read_parquet(out / "datasets.parquet").dataset)
    assert not (out / "tables" / f"{CHALLENGE}.csv").exists()
    release = json.loads((out / "release.json").read_text())
    assert release["release"] == "v1" and release["withheld"] == [CHALLENGE]

    reviewed = pd.read_parquet(out / "labels" / "reviewed" / "train.parquet")
    assert reviewed.reviewer.tolist() == ["R1"] and reviewed.iscrowd.tolist() == [True]
    for split, shot in (("train", SHOT), ("holdout", OTHER), ("test", GOLD)):
        part = pd.read_parquet(out / "labels" / "original" / f"{split}.parquet")
        assert part.shot.tolist() == [shot] and set(part.split) == {split}
    held = pd.read_parquet(out / "shots" / "holdout.parquet")
    assert held.shot.tolist() == [OTHER] and held.events.tolist() == ["disruption"]
    cohort = pd.read_parquet(out / "shots" / "cohort_split.parquet").set_index("shot")
    assert cohort.loc[GOLD, "split"] == "test" and bool(cohort.loc[GOLD, "gold"])
    assert (
        cohort.loc[OTHER, "split"] == "holdout" and cohort.loc[190000, "split"] == "val"
    )
    source = pd.concat(
        pd.read_parquet(p) for p in (out / "labels" / "source").glob("*")
    )
    assert set(source.layer) == {"source"} and set(source.category_name) == {"present"}
    roster = pd.read_csv(
        out / "tables" / "disruption" / "shots.csv", keep_default_na=False
    )
    assert roster.shot.tolist() == [SHOT, OTHER]
    assert roster.reviewers.tolist() == ["R1", ""]

    text = (out / "README.md").read_text()
    card = hf_export.front_matter(text)
    # no license chosen yet: no license field, no license file
    assert not {"license", "license_name", "license_link"} & set(card)
    assert not (out / "LICENSE.md").exists() and "## License\n\nNone yet" in text
    assert "## Withheld tables" in text and f"`{CHALLENGE}`: permission" in text
    shot_configs = {
        c["config_name"]: c["data_files"]
        for c in card["configs"]
        if c["data_files"][0]["path"].startswith("shots/")
    }
    assert set(shot_configs) == {
        "shots_silver",
        "shots_gold",
        "shots_holdout",
        "cohort_split",
    }
    assert all(len(files) == 1 for files in shot_configs.values())
    assert {c["config_name"] for c in card["configs"]} >= {
        "reviewed",
        "original",
        "source",
        "grid",
    }
    grid_config = next(c for c in card["configs"] if c["config_name"] == "grid")
    assert {f["split"] for f in grid_config["data_files"]} == {
        "train",
        "holdout",
        "test",
    }
    assert "**phase-controlled AUROC 0.534: no skill.**" in text
    assert "phase-controlled AUROC 0.534 [0.400, 0.600] (brackets: exploratory)" in text
    assert "Against: a rule." in text and "Caveat: no input senses it." in text
    assert "Review source (the table the review page opens on): `rule` v1." in text
    assert report["review_sources"]["disruption"]["version"] == "v1"
    assert "shot-bootstrap" not in text
    assert hf_export.pending(hf_export._signals(paths)) == {
        "disruption": "a stream is changing it"
    }

    for name, split, shots in (
        ("silver", "silver", (SHOT,)),
        ("holdout", "silver", (OTHER,)),
        ("gold", "gold", (GOLD,)),
    ):
        runs = pd.read_parquet(out / "grid" / f"{name}.parquet")
        assert set(runs.shot) == set(shots)
        for shot in shots:
            names, t0, values = hf_export.decode_grid(runs, shot)
            with np.load(out.parent / "release.work" / split / f"{shot}.npz") as grid:
                assert names == list(grid["events"]) and t0 == int(grid["t0_ms"])
                assert np.array_equal(values, grid["labels"])

    leak = out / "tables" / "disruption" / "extra.csv"
    leak.write_text(f"shot,notes\n{BLIND},{tmp_path}/x\n")
    pd.DataFrame({"shot": [GOLD], "split": ["train"]}).to_parquet(
        out / "shots" / "extra.parquet"
    )
    pd.DataFrame(
        {"shot": [BLIND], "notes": [f"by {NAME} from /u/jd/x.csv"]}
    ).to_parquet(out / "shots" / "leak.parquet")
    pd.DataFrame(
        {"shot": [OTHER], "split": ["holdout"], "note": [f"also seen on {BLIND}"]}
    ).to_parquet(out / "shots" / "note.parquet")  # the shot column is clean
    (out / "notes.txt").write_text(f"made by {LOGIN}\n{PLACEHOLDER}")
    shutil.copy(leak, out / "tables" / f"{CHALLENGE}.csv")
    problems = hf_export.verify(out, hf_export.context(paths))
    for name, what in (
        ("tables/disruption/extra.csv", "blind shots"),
        ("tables/disruption/extra.csv", "absolute path"),
        ("shots/extra.parquet", "shots in the wrong split"),
        ("shots/leak.parquet", "blind shots"),
        ("shots/leak.parquet", "absolute path /u/"),
        ("shots/leak.parquet", "names a login or reviewer"),
        ("shots/note.parquet", "blind shot numbers"),
        ("notes.txt", "names a login or reviewer"),
        ("notes.txt", "a roster placeholder row"),
        (CHALLENGE, "withheld, yet in the release"),
    ):
        assert any(p.startswith(f"{name}: {what}") for p in problems), (name, what)
    # only the string column gave the clean-shot-column file away
    assert [p for p in problems if p.startswith("shots/note.parquet")] == [
        f"shots/note.parquet: blind shot numbers ['{BLIND}']"
    ]


class FakeApi:
    def __init__(self, private, tags=()):
        self.private, self.tags, self.calls = private, list(tags), []

    def repo_exists(self, repo_id, repo_type):
        return self.private is not None

    def repo_info(self, repo_id, repo_type):
        return SimpleNamespace(private=self.private)

    def create_repo(self, repo_id, *, repo_type, private, exist_ok):
        self.calls.append(("create", private, exist_ok))
        self.private = private

    def upload_folder(self, **kwargs):
        self.calls.append(("upload", kwargs["repo_type"], kwargs["delete_patterns"]))
        return SimpleNamespace(oid="abc")

    def list_repo_refs(self, repo_id, repo_type):
        return SimpleNamespace(tags=[SimpleNamespace(name=t) for t in self.tags])

    def create_tag(self, repo_id, *, repo_type, tag, revision):
        self.calls.append(("tag", tag, revision))


def test_upload_needs_yes_a_private_repo_and_no_pending_models(paths, tmp_path):
    out = tmp_path / "release"
    hf_export.build(paths, out)
    ctx = hf_export.context(paths)
    script = (
        Path(__file__).resolve().parents[2] / "scripts" / "labeler" / "hf_publish.py"
    )
    spec = importlib.util.spec_from_file_location("hf_publish", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.main(["upload", "--dir", str(out)]) == 2

    with pytest.raises(hf_export.ReleaseError, match="todo"):
        hf_export.upload(out, ctx, api=FakeApi(None))
    public = FakeApi(False)
    with pytest.raises(hf_export.ReleaseError, match="public"):
        hf_export.upload(out, ctx, api=public, allow_pending=True)
    assert public.calls == []
    released = FakeApi(True, tags=["v1"])  # v1 is out: a rebuild needs a new name
    with pytest.raises(hf_export.ReleaseError, match="already has release v1"):
        hf_export.upload(out, ctx, api=released, allow_pending=True)
    assert released.calls == []
    fresh = FakeApi(None)
    result = hf_export.upload(out, ctx, api=fresh, allow_pending=True)
    assert fresh.calls == [
        ("create", True, True),
        ("upload", "dataset", "*"),
        ("tag", "v1", "abc"),
    ]
    assert result == {"repo": hf_export.REPO_ID, "commit": "abc", "tag": "v1"}


def test_include_ships_a_withheld_table_and_release_names_and_gold_are_checked(
    paths, tmp_path
):
    out = tmp_path / "release"
    with pytest.raises(hf_export.ReleaseError, match="not a name"):
        hf_export.build(paths, out, release="final")
    with pytest.raises(hf_export.ReleaseError, match="names no withheld"):
        hf_export.build(paths, out, include=["disruption/format/other"])
    report = hf_export.build(paths, out, include=[CHALLENGE], release="v2")
    assert report["withheld"] == {} and report["problems"] == []
    original = pd.read_parquet(out / "labels" / "original" / "train.parquet")
    assert set(original.dataset) >= {CHALLENGE}
    assert (out / "tables" / f"{CHALLENGE}.csv").is_file()
    release = json.loads((out / "release.json").read_text())
    assert release["release"] == "v2" and release["withheld"] == []
    assert "Release v2," in (out / "README.md").read_text()

    # no gold list: no gold file, config or grid, and the build still verifies
    (paths.label_tables / "gold" / "gold_shots.csv").unlink()
    report = hf_export.build(paths, out)
    assert report["gold_shots"] == 0 and report["problems"] == []
    assert not (out / "shots" / "gold.parquet").exists()
    assert not (out / "grid" / "gold.parquet").exists()
    names = {
        c["config_name"]
        for c in hf_export.front_matter((out / "README.md").read_text())["configs"]
    }
    assert "shots_gold" not in names and "shots_silver" in names


def test_tables_skip_rules_and_sawtooth_style_crowd_chunks(paths, tmp_path):
    extend = paths.label_tables / "disruption" / "extend_chunks"
    _table(extend / "bare.csv", [(175000, 1, 0, 10, 1.0, "")])  # no sidecar
    wide = extend / "wide.csv"
    wide.write_text("shot,category,t_start,t_end,confidence,extra\n175000,1,0,10,1,x\n")
    wide.with_suffix(".meta.json").write_text("{}")
    chunk = extend / "chunk-000.csv"
    pd.DataFrame(
        [
            (SHOT, 1, 600, 900, True, 1.0, '{"state": "present", "reason": {"a": 1}}'),
            (SHOT, 0, 900, 950, True, 1.0, '{"state": "absent"}'),
            (
                SHOT,
                1,
                700,
                700,
                False,
                1.0,
                '{"period_ms": 30.5, "uncertainty_reasons": [1]}',
            ),
            (SHOT, 3, 950, 990, True, 1.0, '{"state": "unassessed"}'),
            (BLIND, 1, 600, 900, True, 1.0, ""),
        ],
        columns=hf_export.CROWD_COLUMNS,
    ).to_csv(chunk, index=False)
    chunk.with_suffix(".meta.json").write_text(
        json.dumps(
            {
                "categories": {"0": "absent", "1": "present"},
                "producer": "saw_physics",
                "version": "2026_10_04",
            }
        )
    )
    kept = "disruption/format/disruption_format_2026_v1"
    found, skipped = hf_export.tables(paths, exclude=[kept])
    why = {item["dataset"]: item["why"] for item in skipped}
    assert why[kept] == "excluded at build" and kept not in {t.dataset for t in found}
    # an id that is no table of the labeler raises, as an unknown include does
    for call in (
        lambda: hf_export.tables(paths, exclude=["disruption/format/nothing"]),
        lambda: hf_export.build(
            paths, tmp_path / "never", exclude=["disruption/format/nothing"]
        ),
    ):
        with pytest.raises(hf_export.ReleaseError, match="--exclude names no table"):
            call()
    assert not (tmp_path / "never").exists()
    found, skipped = hf_export.tables(paths)
    why = {item["dataset"]: item["why"] for item in skipped}
    assert why["disruption/extend_chunks/bare"] == "no .meta.json sidecar"
    assert why["disruption/extend_chunks/wide"] == "not the interval schema"
    assert why[CHALLENGE].startswith("withheld: ")
    table = next(t for t in found if t.dataset.endswith("chunk-000"))
    assert (table.layer, table.method, table.version) == (
        "extend",
        "saw_physics",
        "2026_10_04",
    )
    frame, scrub = hf_export.rows(table, hf_export.context(paths))
    assert scrub == {
        "withheld_rows": 1,
        "dropped_attrs": ["reason", "uncertainty_reasons"],
    }
    assert [None if pd.isna(v) else bool(v) for v in frame.iscrowd] == [
        True,
        None,
        False,
        True,
    ]
    assert frame.category_name.tolist() == [
        "present",
        "absent",
        "present",
        "not_observable",  # no name in the sidecar or the event: the catalog state
    ]
    with pytest.raises(hf_export.ReleaseError, match="category 7 has no name"):
        hf_export._categories(table, [7])
    # a code the sidecar leaves out keeps the event's name; the sidecar wins elsewhere
    partial = dataclasses.replace(
        table, event="confinement", meta={"categories": {"1": "H (own name)"}}
    )
    names = hf_export._categories(partial, [0, 1, 5])
    assert (names["0"], names["1"], names["5"]) == (
        "absent",
        "H (own name)",
        "uncertain",
    )

    # the grid paints from the chunk and the release holds it: no allow_unshipped
    report = hf_export.build(paths, tmp_path / "release")
    assert report["unshipped"] == {}
    datasets = pd.read_parquet(tmp_path / "release" / "datasets.parquet")
    shipped = datasets.set_index("dataset").loc["disruption/extend_chunks/chunk-000"]
    assert bool(shipped.shipped) and shipped.grid_frames > 0


def test_scrub_and_verify_know_home_and_storage_paths():
    for path in (
        "/u/jd/a.csv",
        "/p/x/y",
        "/global/cfs/z",
        "/Users/jd/q",
        "~jd/w",
        "~/v",
    ):
        cleaned = hf_export.scrub_text(f"from {path} today")
        assert "jd" not in cleaned and hf_export.FORBIDDEN_PATH.search(f" {path}")
        assert not hf_export.FORBIDDEN_PATH.search(cleaned), cleaned
    for fine in (
        "m/n 2/1",
        "MDSplus/PTDATA",
        "$LABELER_ROOT/round4/x",
        "R1/R2",
        "a ~ b",
    ):
        assert not hf_export.FORBIDDEN_PATH.search(fine), fine


def test_build_refuses_unsafe_targets_and_flags_unshipped_tables(paths, tmp_path):
    for target in (
        paths.label_tables,
        tmp_path,
        paths.root,
        paths.label_tables / "release",
    ):
        with pytest.raises(hf_export.ReleaseError, match="refusing"):
            hf_export.build(paths, target)
    stray = tmp_path / "notes"
    stray.mkdir()
    (stray / "keep.txt").write_text("x")
    with pytest.raises(hf_export.ReleaseError, match="not a previous release"):
        hf_export.build(paths, stray)
    assert (stray / "keep.txt").read_text() == "x"

    # a table the grid paints from that has no sidecar, so the release cannot hold it
    chunk = paths.label_tables / "disruption" / "extend_chunks" / "chunk_000.csv"
    _table(chunk, [(SHOT, 1, 600, 900, 1.0, "")])
    out = tmp_path / "release"
    with pytest.raises(hf_export.ReleaseError, match="does not hold"):
        hf_export.build(paths, out)
    assert not out.exists() and not list(tmp_path.glob(".release.*"))
    for _ in range(2):  # the second build replaces the first release
        report = hf_export.build(paths, out, allow_unshipped=True)
    assert set(report["unshipped"]) == {"disruption/extend_chunks/chunk_000"}
    assert not list(tmp_path.glob(".release.*"))
    datasets = pd.read_parquet(out / "datasets.parquet").set_index("dataset")
    assert not datasets.loc["disruption/extend_chunks/chunk_000", "shipped"]
    assert datasets.loc["disruption/extend_chunks/chunk_000", "grid_frames"] == 30
    assert "### Not in this release" in (out / "README.md").read_text()
    with pytest.raises(hf_export.ReleaseError, match="not in the release"):
        hf_export.upload(
            out, hf_export.context(paths), api=FakeApi(None), allow_pending=True
        )


def test_pull_restores_tables_and_pointer_and_never_replaces(paths, tmp_path):
    out = tmp_path / "release"
    hf_export.build(paths, out)
    with pytest.raises(hf_export.ReleaseError, match="full label tables"):
        hf_export.restore(out, paths)
    clone = Paths(
        root=tmp_path / "clone_root",
        corpus=paths.corpus,
        text_root=paths.text_root,
        logs_jsonl=paths.logs_jsonl,
        label_tables=tmp_path / "clone",
        raw_cache=paths.raw_cache,
    )
    result = hf_export.restore(out, clone)
    assert result["pointers"] == 1 and result["copied"] == result["files"]
    # The release carries both config files for reference (paths scrubbed); the code
    # reads them from configs/labeler, so a restore never writes them into the tables.
    for name in ("events.yaml", "best_models.yaml"):
        assert (out / "tables" / name).is_file()
        assert not (clone.label_tables / name).exists()
    restored = (
        clone.label_tables / "disruption" / "format" / "disruption_format_2026_v1.csv"
    )
    assert (
        restored.read_bytes()
        == (out / "tables" / restored.relative_to(clone.label_tables)).read_bytes()
    )
    assert (
        labels.source_path(clone.label_tables / "disruption")
        == (
            clone.root
            / "suggestions"
            / "rule"
            / "v1"
            / "disruption_suggest_rule_v1.csv"
        ).resolve()
    )
    assert hf_export.restore(out, clone)["copied"] == 0
    restored.write_text(HEADER)
    with pytest.raises(hf_export.ReleaseError, match="differ"):
        hf_export.restore(out, clone)
    assert restored.read_text() == HEADER

    # a pointer already set elsewhere is a conflict too, and nothing is copied
    other = Paths(
        root=tmp_path / "other_root",
        corpus=paths.corpus,
        text_root=paths.text_root,
        logs_jsonl=paths.logs_jsonl,
        label_tables=tmp_path / "other",
        raw_cache=paths.raw_cache,
    )
    mine = _table(other.root / "suggestions" / "mine.csv", [(SHOT, 1, 0, 9, 1.0, "")])
    labels.write_pointer(
        other.label_tables / "disruption", mine, method="mine", version="v1"
    )
    with pytest.raises(hf_export.ReleaseError, match="differ"):
        hf_export.restore(out, other)
    assert not (other.label_tables / "disruption" / "format").exists()

    # a download is restored only as released (release.json's sha256), and an
    # earlier download of the same commit is reused only when it is exactly that
    api = SimpleNamespace(
        dataset_info=lambda repo_id, revision: SimpleNamespace(sha="c0")
    )
    pulled = tmp_path / "pull" / "c0"
    shutil.copytree(out, pulled)
    assert hf_export.download("x/y", tmp_path / "pull", api=api) == pulled
    (pulled / "tables" / "disruption" / "stale.csv").write_text("shot\n")
    with pytest.raises(hf_export.ReleaseError, match="earlier download"):
        hf_export.download("x/y", tmp_path / "pull", api=api)
    with pytest.raises(hf_export.ReleaseError, match="not in release.json"):
        hf_export.restore(pulled, other)
    (pulled / "tables" / "disruption" / "stale.csv").unlink()
    assert hf_export.check_files(pulled) == []
    changed = "tables/disruption/format/disruption_format_2026_v1.csv"
    (pulled / changed).write_text((pulled / changed).read_text() + "\n")
    assert hf_export.check_files(pulled) == [
        f"{changed}: sha256 differs from release.json"
    ]
    with pytest.raises(hf_export.ReleaseError, match="is not the release"):
        hf_export.restore(pulled, other)
    (pulled / changed).unlink()
    assert hf_export.check_files(pulled) == [f"{changed}: missing"]


def test_a_withheld_table_that_feeds_the_grid_fails_the_build(paths, tmp_path):
    key = (
        paths.label_tables / "disruption" / "format"
        / "disruption_challenge_format_2026_v1.csv"
    )  # fmt: skip
    meta = json.loads(key.with_suffix(".meta.json").read_text())
    _table(
        key, [(SHOT, 1, 10, 20, 1.0, "")], meta
    )  # a silver shot, before any other label
    out = tmp_path / "release"
    for allow in (False, True):  # allow_unshipped cannot cover a withheld table
        with pytest.raises(hf_export.ReleaseError, match="withheld tables"):
            hf_export.build(paths, out, allow_unshipped=allow)
    assert not out.exists() and not list(tmp_path.glob(".release.*"))
    # shipping it is the owner's call, by name
    report = hf_export.build(paths, out, include=[CHALLENGE])
    assert report["problems"] == [] and report["withheld"] == {}


def _five_columns(path, rows, meta):
    """A table with no attrs column, as the detachment suggestions are."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "shot,category,t_start,t_end,confidence\n"
        + "".join(f"{s},{c},{a},{b},\n" for s, c, a, b in rows)
    )
    path.with_suffix(".meta.json").write_text(json.dumps(meta))


def test_card_says_a_merged_source_is_not_a_label_set_and_what_category_zero_means(
    paths, tmp_path
):
    events = paths.events_manifest
    events.write_text(
        events.read_text().replace(
            "taxonomy:\n",
            "taxonomy:\n  detachment:\n    display: Detachment\n    review: false\n",
            1,
        )
    )
    (paths.label_tables / "detachment").mkdir()
    names = {
        "0": "unmarked (not assessed)",
        "1": "attached",
        "2": "detached",
        "3": "marfe (candidate)",
        "4": "uncertain",
    }
    base = {"event": "detachment", "categories": names, "bin_ms": 50.0}
    suggestions = paths.root / "suggestions"
    _five_columns(
        suggestions / "detach_te" / "v2" / "detachment_suggest_detach_te_v2.csv",
        [(SHOT, 1, 0, 50), (SHOT, 2, 50, 100)],
        {
            **base,
            "method": "detach_te",
            "version": "v2",
            "rule": {
                "te": "processed divertor Thomson Te",
                "states": "Te <= 5 eV detached (2), >= 10 eV attached (1), "
                "between uncertain (4)",
            },
        },
    )
    _five_columns(
        suggestions / "detach_vote" / "v3" / "detachment_suggest_detach_vote_v3.csv",
        [(SHOT, 1, 0, 50), (SHOT, 3, 50, 100)],
        {
            **base,
            "method": "detach_vote",
            "version": "v3",
            "rule": {"vote": "the vote first", "te": "Te where the vote has none"},
            "bins_from": {"detach_te": 4702, "detach_vote": 1665},
            "coverage": {"bins": 6367, "shots": 175},
        },
    )
    paths.best_models.write_text(
        "version: 1\nevents:\n  detachment:\n    model: detach_vote\n"
        "    task: rule\n    caveats: [a merge of two methods]\n"
    )
    found, _ = hf_export.tables(paths)
    (item,) = hf_export.mixed_sources(found)
    assert (item["dataset"], item["bins"], item["shots"]) == (
        "suggestions/detach_vote/v3/detachment_suggest_detach_vote_v3",
        6367,
        175,
    )
    assert item["bins_from"] == {"detach_te": 4702, "detach_vote": 1665}
    assert item["per_row_marker"] is False

    out = tmp_path / "release"
    report = hf_export.build(paths, out)
    assert report["problems"] == [] and report["mixed_sources"] == [item]
    text = (out / "README.md").read_text()
    assert (
        "- **Not a label set**: `suggestions/detach_vote/v3/"
        "detachment_suggest_detach_vote_v3` (detachment, tier source) merges the bins"
    ) in text
    core = (
        "of its 6,367 50 ms bins on 175 shots, 4,702 from `detach_te` (Te <= 5 eV "
        "detached (2), >= 10 eV attached (1), between uncertain (4)); 1,665 from "
        "`detach_vote`; its rows carry no per-row marker of which method set a bin"
    )
    assert core in text
    assert (
        "Merge rule, from its sidecar: the vote first; Te where the vote has none"
        in text
    )
    assert "`detach_te` v2, from its sidecar: processed divertor Thomson Te" in text
    # the best-model line for the event carries it too
    assert (
        "Caveat: a merge of two methods. The table that ships for this event, "
        "`detach_vote` v3, is not this model's label set: " + core
    ) in text

    # category 0 is absent only where the table assessed the event
    assert (
        "only where `category_name` says `absent`, and not even then for the events "
        "whose 0 only says nothing was marked: "
        "`confinement` (the owner saved no regime here); "
        "`detachment` (its tables name 0 `unmarked (not assessed)`; "
        "the owner saved no state here). The 10 ms grid already maps those 0s to -1"
    ) in text
    assert "explicit assessed-absent time" not in text
    assert report["zero_exceptions"]["detachment"]["names"] == [
        "unmarked (not assessed)"
    ]
    assert set(report["zero_exceptions"]) == {"confinement", "detachment"}


def test_a_default_class_is_in_the_taxonomy_the_zero_exceptions_and_the_card(paths):
    events = paths.events_manifest
    events.write_text(
        events.read_text().replace(
            "taxonomy:\n",
            "taxonomy:\n  detachment:\n    display: Detachment\n    default: 1\n",
            1,
        )
    )
    detachment = paths.label_tables / "detachment"
    detachment.mkdir()
    (detachment / "shots.csv").write_text(
        "shot,tier,holdout,reviewers,verified_on,notes\n"
    )
    tax = hf_export._taxonomy(paths, ["detachment", "disruption"])
    assert tax["events"]["detachment"]["default"] == 1
    assert "default" not in tax["events"]["disruption"]
    assert tax["grid"]["default"]["detachment"] == 1
    assert tax["grid"]["default"]["disruption"] == 0
    assert tax["grid"]["zero"]["detachment"].startswith(
        "attached (1), the event's default"
    )
    assert tax["grid"]["zero"]["disruption"].startswith("absent")
    found = hf_export.zero_exceptions([], taxonomy.load(events))
    assert set(found) == {"confinement", "detachment"}
    assert found["detachment"]["owner"].startswith("attached (1)")
    assert found["confinement"]["owner"] == silver.ZERO["confinement"]  # no default
    text = hf_export._zero_text(found)
    assert "`detachment` (attached (1), the event's default" in text
    assert "or, for an event with a default class, that class" in text
    # With no taxonomy to ask, the exceptions are the ones `silver.ZERO` names.
    assert {e: v["owner"] for e, v in hf_export.zero_exceptions([]).items()} == (
        silver.ZERO
    )


def test_card_names_the_review_source_and_checks_the_tables_an_entry_reads():
    def table(method, version, shipped=True):
        return {
            "dataset": f"suggestions/{method}/{version}/t",
            "method": method,
            "version": version,
            "shipped": shipped,
        }

    provenance = [
        table("saw_physics", "v4"),
        table("crash", "v3"),
        table("crash", "v4", shipped=False),
    ]
    sources = hf_export.review_sources(
        {
            "sawtooth_oscillation": "suggestions/saw_physics/v4/t",
            "elsewhere": "suggestions/rwm_onsets/v4/t",  # not in the release
        },
        provenance,
    )
    assert sources == {
        "sawtooth_oscillation": {
            "dataset": "suggestions/saw_physics/v4/t",
            "method": "saw_physics",
            "version": "v4",
        }
    }
    entry = {"model": "saw-hl3-full", "version": "2026_10_05"}
    entry["reads_tables"] = ["crash v3"]
    assert hf_export._source_text(entry, "sawtooth_oscillation", sources) == (
        "Review source (the table the review page opens on): `saw_physics` v4. "
        "Also reads: `crash` v3."
    )
    # a subcategory takes its event's source; the model's own name and version is
    # not said twice
    own = {"model": "saw_physics", "version": "v4"}
    assert hf_export._source_text(own, "sawtooth_oscillation.x", sources) == ""
    assert hf_export._source_text({"model": "m"}, "nothing", sources) == ""

    hf_export.check_reads_tables({"events": {"a": entry}}, provenance)
    for bad in ("crash v4", "crash v9", "crash"):  # unshipped, absent, malformed
        with pytest.raises(hf_export.ReleaseError, match="crash"):
            hf_export.check_reads_tables(
                {"events": {"a": {"reads_tables": [bad]}}}, provenance
            )
