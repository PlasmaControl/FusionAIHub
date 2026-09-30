"""S1's fix: lit pixels in the masks-full check, the manifests of masks-full and
dataset-full and the records that name them, the cross-validation ablation and
its summary, and v3's diagnosis."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from labeler.ae import full
from labeler.ae.xpower import (
    cv,
    diagnosis,
    evaluate,
    gallery,
    gallery_dir,
    model_dir,
    train,
)
from labeler.ae.xpower.data import BAND_KHZ, FULL_BAND_KHZ
from labeler.ae.xpower.model import FrameCNN, FrameCNNConfig
from labeler.config import Paths

from . import ae_tree, round3_tree
from .round3_tree import MHD_FRAMES
from .test_ae_xpower_cv import POOL, SPLITS, TEST, _designed, _fake_fit
from .test_round3_full import _clean
from .test_round3_xpower_v3 import _designed_whole


def _json(path):
    return json.loads(path.read_text())


def _files(root):
    return {p for p in root.rglob("*") if p.is_file()}


# ---------------------------------------------------------------- the check

T_V1 = -0.768 + 0.256 * np.arange(7820)
T_FULL = 0.1 + 0.256 * np.arange(12000)


def test_the_check_counts_each_records_lit_pixels_per_compared_column(tmp_path):
    _clean(tmp_path / "v1.npz", T_V1, [(200, 210, 500, 900), (80, 85, 1000, 1500)])
    lines = [(200, 210, 500, 900), (200, 210, 2500, 3000)]
    _clean(tmp_path / "full.npz", T_FULL, lines, notch_bin=300)
    got = full.check_shot(tmp_path / "v1.npz", tmp_path / "full.npz")
    assert set(got) == {"frames", "iou", "lit_per_column", "notch_v1", "notch_full"}
    n, per = got["frames"], got["lit_per_column"]
    ae, line = 10 * 400 / 0.256, 5 * 500 / 0.256  # lit pixels of the AE, of 40 kHz
    assert per["full"]["0-250"] == per["full"]["80-250"]  # after 2 s is not compared
    assert per["full"]["80-250"] == pytest.approx(ae / n, abs=0.01)
    assert per["v1"]["80-250"] == pytest.approx(ae / n, abs=0.01)
    assert per["v1"]["0-250"] == pytest.approx((ae + line) / n, abs=0.01)
    # The IoUs are the check's as before; whole-shot's pixels lie inside v1's.
    assert got["iou"]["80-250"] == pytest.approx(1.0, abs=0.01)
    assert got["iou"]["0-250"] == pytest.approx(ae / (ae + line), abs=0.01)
    assert got["iou"]["0-250"] == pytest.approx(
        per["full"]["0-250"] / per["v1"]["0-250"], abs=0.01
    )


def test_with_no_frame_compared_the_lit_pixels_are_nan(tmp_path):
    _clean(tmp_path / "v1.npz", T_V1, [(200, 210, 500, 900)])
    late = 2100.0 + 0.256 * np.arange(2000)
    _clean(tmp_path / "full.npz", late, [(200, 210, 2200, 2300)])
    got = full.check_shot(tmp_path / "v1.npz", tmp_path / "full.npz")
    assert got["frames"] == 0
    values = [v for record in got["lit_per_column"].values() for v in record.values()]
    assert len(values) == 4 and all(math.isnan(v) for v in values)


def test_check_md_prints_lit_pixels_per_column_beside_the_ious():
    none = {c: [] for c in full.CHANNEL_NAMES}
    new = {
        "shot": 172018,
        "split": "train",
        "frames": 7816,
        "iou": {"0-250": 0.3262, "80-250": 0.3211},
        "lit_per_column": {
            "v1": {"0-250": 0.581, "80-250": 0.36},
            "full": {"0-250": 0.5, "80-250": 0.404},
        },
        "notch_v1": none,
        "notch_full": none,
    }
    old = {k: v for k, v in new.items() if k != "lit_per_column"} | {"shot": 170720}
    text = full.check_md([new, old], full.check_bar([new, old]))
    lines = text.splitlines()
    head = next(x for x in lines if x.startswith("| shot |"))
    assert "| IoU 80-250 | lit per column 0-250 | lit per column 80-250 |" in head
    sep, row, older = lines[lines.index(head) + 1 : lines.index(head) + 4]
    assert sep.count("|") == head.count("|") == row.count("|") == older.count("|")
    assert row.startswith("| 172018 |")
    assert "| 0.326 | 0.321 | 0.58 / 0.50 | 0.36 / 0.40 |" in row
    assert "| 0.326 | 0.321 | - | - |" in older  # a record from before the columns
    assert "(v1 / whole shot)" in text


# ---------------------------------------------------------------- the manifests

T0 = datetime(2026, 9, 29, 6, 30, tzinfo=UTC)


def _at(path, seconds: float) -> None:
    """Give `path` the mtime T0 + `seconds`."""
    ts = (T0 + timedelta(seconds=seconds)).timestamp()
    os.utime(path, (ts, ts))


def _log(paths, job, stems, start, root=None):
    """Job `job`'s log: it ran from T0 + `start` s for 100 s."""
    begin = (T0 + timedelta(seconds=start)).timestamp()
    return round3_tree.job_log(paths, job, stems, begin, begin + 100, root=root)


@pytest.fixture
def made(tmp_path):
    """masks-full and dataset-full of two stems: 101_train by job 7001 (it ran
    0-100 s after T0) and 102_valid by job 7002 (200-300 s); the check's report
    by job 7003 (400-500 s)."""
    paths = Paths(root=tmp_path / "root")
    masks, dataset = full.masks_full_dir(paths), full.dataset_full_dir(paths)
    masks.mkdir(parents=True)
    dataset.mkdir(parents=True)
    for stem, when in (("101_train", 50), ("102_valid", 250)):
        for file in (
            masks / f"{stem}_clean.npz",
            masks / f"{stem}_probs.npz",
            dataset / f"{stem}.npz",
        ):
            file.write_bytes(file.name.encode() * 3)
            _at(file, when)
    for name in ("check.json", "check.md"):
        (masks / name).write_text("{}")
        _at(masks / name, 450)
    _log(paths, "7001", ["101_train"], 0)
    _log(paths, "7002", ["102_valid"], 200)
    _log(paths, "7003", [], 400)
    return paths


def _jobs(paths, *jobs):
    return [full.read_job(paths, j, "abc1234") for j in jobs or ("7001", "7002")]


def _folds_made(paths, seconds: float) -> None:
    """v3's first record, `cv/folds.json`, made at T0 + `seconds`."""
    file = model_dir(paths, "v3") / "cv" / "folds.json"
    file.parent.mkdir(parents=True, exist_ok=True)
    when = (T0 + timedelta(seconds=seconds)).isoformat(timespec="seconds")
    file.write_text(json.dumps({"made_at": when}))


def test_a_manifest_lists_each_data_file_with_the_job_that_wrote_it(made):
    masks = full.masks_full_dir(made)
    report = full.read_job(made, "7003", "def5678")
    record = full.write_manifest(made, masks, _jobs(made), report)
    assert sorted(record["files"]) == [
        "101_train_clean.npz",
        "101_train_probs.npz",
        "102_valid_clean.npz",
        "102_valid_probs.npz",
    ]
    entry, data = record["files"]["102_valid_probs.npz"], b"102_valid_probs.npz" * 3
    assert entry["job"] == "7002" and entry["bytes"] == len(data)
    assert entry["sha256"] == hashlib.sha256(data).hexdigest()
    assert entry["mtime"] == (T0 + timedelta(seconds=250)).isoformat()
    job = record["jobs"]["7001"]
    assert (job["commit"], job["files"], job["stems"]) == ("abc1234", 2, 1)
    assert job["log"] == "runs/slurm/7001.out"
    assert job["started"] == T0.isoformat() and job["host"] == "node01"
    assert record["report"]["job"] == "7003" and record["report"]["commit"] == "def5678"
    assert sorted(record["report"]["files"]) == ["check.json", "check.md"]
    assert record["directory"] == "ae/masks-full"
    saved = (masks / full.MANIFEST).read_bytes()
    assert json.loads(saved) == record
    assert full.manifest_sha256(masks) == hashlib.sha256(saved).hexdigest()
    other = full.write_manifest(made, full.dataset_full_dir(made), _jobs(made))
    assert sorted(other["files"]) == ["101_train.npz", "102_valid.npz"]
    assert other["report"] is None


def test_a_stem_written_again_is_the_job_that_wrote_it_last(made):
    _log(made, "7002", ["101_train", "102_valid"], 200)  # 7002 wrote 101 again
    masks = full.masks_full_dir(made)
    _at(masks / "101_train_clean.npz", 260)
    files = full.manifest_record(made, masks, _jobs(made))["files"]
    assert files["101_train_clean.npz"]["job"] == "7002"
    assert files["101_train_probs.npz"]["job"] == "7001"


def test_a_manifest_refuses_a_file_no_listed_job_wrote(made):
    masks, dataset = full.masks_full_dir(made), full.dataset_full_dir(made)
    with pytest.raises(ValueError, match="102_valid_clean.npz: written at .* by none"):
        full.manifest_record(made, masks, _jobs(made, "7001"))
    _at(dataset / "102_valid.npz", 150)  # between the two jobs
    with pytest.raises(ValueError, match="102_valid.npz: written at"):
        full.manifest_record(made, dataset, _jobs(made))
    _at(dataset / "102_valid.npz", 250)
    _log(made, "7004", ["103_train"], 600)
    with pytest.raises(ValueError, match="jobs 7004 wrote none"):
        full.manifest_record(made, dataset, _jobs(made, "7001", "7002", "7004"))
    (dataset / "102_valid.npz.tmp").write_bytes(b"")  # a write under way
    with pytest.raises(ValueError, match="not a data file"):
        full.manifest_record(made, dataset, _jobs(made))
    (masks / "check.json.tmp").write_bytes(b"")  # the check's, never an input
    assert len(full.manifest_record(made, masks, _jobs(made))["files"]) == 4
    report = full.read_job(made, "7003", "def5678")
    _at(masks / "check.md", 350)
    with pytest.raises(ValueError, match="not while job 7003 ran"):
        full.manifest_record(made, masks, _jobs(made), report)


def test_a_manifest_is_written_once(made):
    masks = full.masks_full_dir(made)
    first = full.write_manifest(made, masks, _jobs(made))
    saved = (masks / full.MANIFEST).read_bytes()
    digest = full.manifest_sha256(masks)
    assert full.write_manifest(made, masks, _jobs(made)) == first
    assert (masks / full.MANIFEST).read_bytes() == saved
    # A check run rewrites its report: no input, so the manifest still holds.
    (masks / "check.md").write_text("again")
    assert full.manifest_sha256(masks) == digest
    _at(masks / "101_train_probs.npz", 60)  # rewritten inside 7001's run
    with pytest.raises(FileExistsError, match="differs in files"):
        full.write_manifest(made, masks, _jobs(made))
    with pytest.raises(ValueError, match="changed 101_train_probs.npz"):
        full.manifest_sha256(masks)
    assert (masks / full.MANIFEST).read_bytes() == saved


def test_the_manifests_sha256_names_unlisted_missing_and_changed_files(made):
    dataset = full.dataset_full_dir(made)
    with pytest.raises(FileNotFoundError, match="no manifest"):
        full.manifest_sha256(dataset)
    full.write_manifest(made, dataset, _jobs(made))
    (dataset / "103_train.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="unlisted 103_train.npz"):
        full.manifest_sha256(dataset)
    (dataset / "103_train.npz").unlink()
    data = (dataset / "101_train.npz").read_bytes()
    (dataset / "101_train.npz").unlink()
    with pytest.raises(ValueError, match="missing 101_train.npz"):
        full.manifest_sha256(dataset)
    (dataset / "101_train.npz").write_bytes(data)  # the same bytes, a new mtime
    with pytest.raises(ValueError, match="changed 101_train.npz"):
        full.manifest_sha256(dataset)


def test_a_job_is_read_from_its_own_finished_log_under_this_root(made):
    job = full.read_job(made, "7002", "2dbd01c")
    assert job.stems == ("102_valid",) and job.host == "node01"
    assert job.started == T0 + timedelta(seconds=200)
    assert job.finished == T0 + timedelta(seconds=300)
    log = made.runs / "slurm" / "7002.out"
    assert job.log_sha256 == hashlib.sha256(log.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="not a commit"):
        full.read_job(made, "7002", "HEAD")
    log.rename(made.runs / "slurm" / "7005.out")  # says job 7002
    with pytest.raises(ValueError, match="not job 7005"):
        full.read_job(made, "7005", "2dbd01c")
    _log(made, "7006", ["102_valid"], 200, root="/elsewhere")
    with pytest.raises(ValueError, match="not job 7006"):
        full.read_job(made, "7006", "2dbd01c")
    _log(made, "7007", ["102_valid"], 200)
    unfinished = made.runs / "slurm" / "7007.out"
    unfinished.write_text(unfinished.read_text().rsplit("finished at", 1)[0])
    with pytest.raises(ValueError, match="not a finished job"):
        full.read_job(made, "7007", "2dbd01c")


def test_the_inputs_are_the_two_manifests_and_inputs_json_vouches_for_v3(made):
    masks, dataset = full.masks_full_dir(made), full.dataset_full_dir(made)
    full.write_manifest(made, masks, _jobs(made))
    full.write_manifest(made, dataset, _jobs(made))
    identity = full.inputs_identity(made, "v3")
    assert identity == {
        "masks_full_manifest_sha256": full.manifest_sha256(masks),
        "dataset_full_manifest_sha256": full.manifest_sha256(dataset),
    }
    assert full.inputs_identity(made, "v2") == {} == full.inputs_identity(made, "v1")
    record_file = made.root / "fold0.json"
    assert full.check_inputs(made, "v3", identity, record_file) == identity
    assert full.check_inputs(made, "v2", None, record_file) == {}
    with pytest.raises(ValueError, match="names no inputs"):
        full.check_inputs(made, "v3", None, record_file)
    other = dict.fromkeys(identity, "0" * 64)
    with pytest.raises(ValueError, match="fold0.json: made from other"):
        full.check_inputs(made, "v3", other, record_file)
    _folds_made(made, 1000)
    saved = full.write_inputs(made, "v3")
    assert _json(full.inputs_file(made, "v3")) == saved
    assert {k: saved[k] for k in identity} == identity
    assert saved["latest_file_mtime"] == (T0 + timedelta(seconds=250)).isoformat()
    assert saved["first_record"] == "models/ae_xpower/v3/cv/folds.json"
    assert full.write_inputs(made, "v3") == saved  # once; again, the same
    # A record made before the manifests names none: inputs.json stands for it.
    assert full.check_inputs(made, "v3", None, record_file) == identity
    with pytest.raises(ValueError, match="not scored over whole windows"):
        full.write_inputs(made, "v2")


def test_inputs_json_is_refused_for_files_written_after_the_first_record(made):
    for directory in (full.masks_full_dir(made), full.dataset_full_dir(made)):
        full.write_manifest(made, directory, _jobs(made))
    _folds_made(made, 200)  # 102_valid was written at 250 s
    with pytest.raises(ValueError, match="after"):
        full.write_inputs(made, "v3")
    assert not full.inputs_file(made, "v3").exists()


def test_the_manifest_command_writes_both_and_v3s_inputs(made, monkeypatch, capsys):
    monkeypatch.setenv("LABELER_ROOT", str(made.root))
    _folds_made(made, 1000)
    argv = ["--manifest", "--job", "7001=abc1234", "--job", "7002=abc1234"]
    argv += ["--check-job", "7003=def5678", "--version", "v3"]
    assert full.main(argv) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["masks-full"]["files"] == 4 and out["dataset-full"]["files"] == 2
    assert out["inputs"]["masks_full_manifest_sha256"] == out["masks-full"]["sha256"]
    manifest = _json(full.masks_full_dir(made) / full.MANIFEST)
    assert manifest["report"]["job"] == "7003"
    assert full.main(argv) == 0  # a no-op the second time
    assert json.loads(capsys.readouterr().out) == out


@pytest.mark.parametrize(
    "argv",
    [
        [],
        ["--manifest"],
        ["--check", "--manifest", "--job", "7001=abc1234"],
        ["--check", "--job", "7001=abc1234"],
        ["--check", "--version", "v3"],
        ["--manifest", "--job", "x=abc1234"],
        ["--manifest", "--job", "7008=abc1234"],  # no such log
    ],
)
def test_the_manifest_command_refuses_a_wrong_mode(made, monkeypatch, argv):
    monkeypatch.setenv("LABELER_ROOT", str(made.root))
    with pytest.raises(SystemExit) as error:
        full.main(argv)
    assert error.value.code == 2
    assert not (full.masks_full_dir(made) / full.MANIFEST).exists()


# ---------------------------------------------------------------- the ablation


def test_the_ablations_are_the_other_versions_bands_outside_the_own():
    got = {
        v: {n: (s["band"], s["mhd_weight"]) for n, s in names.items()}
        for v, names in cv.ABLATIONS.items()
    }
    assert got == {
        "v2": {"band0-mhd3": (FULL_BAND_KHZ, 3.0)},
        "v3": {"band80-mhd3": (BAND_KHZ, 3.0), "band80-mhd10": (BAND_KHZ, 10.0)},
    }
    assert cv.ABLATIONS["v2"]["band0-mhd3"] == train.candidates("v3")["band0-mhd3"]
    for version, names in cv.ABLATIONS.items():
        assert not set(names) & set(train.candidates(version))
        for name in names:
            assert cv.ablation_spec(version, name) == names[name]
    with pytest.raises(ValueError, match="one of version v3's own"):
        cv.ablation_spec("v3", "band0-mhd3")
    with pytest.raises(ValueError, match="not one of version v3's ablations"):
        cv.ablation_spec("v3", "band80-mhd30")
    with pytest.raises(ValueError, match="ablations: none"):
        cv.ablation_spec("v1", "band0-mhd99")
    paths = Paths(root=Path("/r"))
    where = Path("/r/runs/ae_xpower/ablation/v3/band80-mhd3")
    assert cv.ablation_dir(paths, "v3", "band80-mhd3") == where
    assert cv.ablation_dir(paths, "v3", "band80-mhd3", 10) == where / "pilot"


def test_an_ablation_needs_the_folds_its_version_chose_from(tmp_path):
    models = tmp_path / "v3"
    (models / "cv").mkdir(parents=True)
    (models / "chosen.json").write_text(json.dumps({"folds_sha256": "a"}))
    (models / "cv" / "choice.json").write_text(json.dumps({"folds_sha256": "b"}))
    with pytest.raises(ValueError, match="choice.json's folds_sha256 b"):
        cv._chosen_folds(models, "a")
    with pytest.raises(ValueError, match="chosen.json's folds_sha256 a"):
        cv._chosen_folds(models, "b")
    (models / "cv" / "choice.json").write_text(json.dumps({"folds_sha256": "a"}))
    cv._chosen_folds(models, "a")


def test_ablate_is_for_a_fold_task_with_the_default_seed(capsys):
    task = ["--version", "v3", "--candidate", "band80-mhd3", "--fold", "0", "--ablate"]
    for argv in (
        task + ["--seed", "20260924"],
        ["--version", "v3", "--choose", "--ablate"],
    ):
        with pytest.raises(SystemExit) as error:
            cv.main(argv)
        assert error.value.code == 2
    assert "--ablate" in capsys.readouterr().err


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/labeler/ae_xpower_cv.sbatch"


def test_the_cv_job_runs_an_ablation_candidates_five_folds():
    subprocess.run(["bash", "-n", str(SCRIPT)], check=True)
    text = SCRIPT.read_text()
    assert 'CANDIDATE="$ABLATE"' in text and '[ "$K" -lt 5 ]' in text
    assert "${ABLATE:+--ablate}" in text.split("\nsrun ", 1)[1]
    for version, names in cv.ABLATIONS.items():
        for name in names:
            assert f"VERSION={version} ABLATE={name} sbatch --array=0-4 <this>" in text
    for pilot in ("VERSION=v3 ABLATE=band80-mhd3", "VERSION=v2 ABLATE=band0-mhd3"):
        assert f"{pilot} PILOT=10 sbatch --array=0-4 <this>" in text
    assert "runs/ae_xpower/ablation/$VERSION/<c>/cv/<c>/" in text
    assert "python -m labeler.ae.xpower.diagnosis --ablation" in text


def _final_fit(shots, val, config, log=print):
    return (
        FrameCNN(FrameCNNConfig(width=4)),
        [{"epoch": 1, "val_f1": None, "kept": True}],
        None,
    )


def _index(paths, digest, f1s) -> None:
    """The gallery's index.csv: v3's test shots drawn from its chosen model,
    with `f1s` {shot: (whole window, 0-2 s)}, and one pool shot."""
    chosen = _json(model_dir(paths, "v3") / "chosen.json")
    rows = [
        {
            "shot": shot,
            "split": "test" if shot in f1s else "train",
            "window_start_ms": 0,
            "window_end_ms": 3000,
            "threshold": chosen["threshold"],
            "candidate": chosen["candidate"],
            "version": "v3",
            "snapshot_sha256": digest,
            "f1_0_2s": f1s.get(shot, (0, 0))[1],
            "scored_window": "whole",
            "f1_scored": f1s.get(shot, (0, 0))[0],
        }
        for shot in [*f1s, POOL[0]]
    ]
    file = gallery_dir(paths, "v3") / "index.csv"
    file.parent.mkdir(parents=True, exist_ok=True)
    with file.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=gallery.INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture(scope="module")
def made_v3(tmp_path_factory):
    """The whole-shot tree, its manifests, and on it: v3's pipeline (folds, 15
    fold tasks, the choice, the final model, the one test), v2's (folds, 15
    fold tasks, the choice, the final model), and the ablation's v3
    band80-mhd3 and v2 band0-mhd3 (all five folds; v3's band80-mhd10 not run).
    P(AE) is designed: `_designed_whole` for v3's folds, `_designed` for v2's."""
    base = tmp_path_factory.mktemp("s1fix")
    with pytest.MonkeyPatch.context() as mp:
        paths = round3_tree.build(base, SPLITS)
        round3_tree.manifests(paths)
        ae_tree.env(mp, paths)
        digests = {v: ae_tree.snapshot(paths, mp, v) for v in ("v3", "v2")}
        with pytest.MonkeyPatch.context() as run:
            _designed_whole(run)
            _designed(run)
            run.setattr(train, "fit", _fake_fit([]))
            for version in ("v3", "v2"):
                assert cv.main(["--version", version, "--folds"]) == 0
                for name in train.candidates(version):
                    for k in range(5):
                        task = ["--candidate", name, "--fold", str(k)]
                        assert cv.main(["--version", version, *task]) == 0
                assert cv.main(["--version", version, "--choose"]) == 0
            run.setattr(train, "fit", _final_fit)
            for version in ("v3", "v2"):
                assert train.main(["--version", version, "--from-cv"]) == 0
            run.setattr(train, "fit", _fake_fit([]))
            for version, name in (("v3", "band80-mhd3"), ("v2", "band0-mhd3")):
                for k in range(5):
                    task = ["--candidate", name, "--fold", str(k), "--ablate"]
                    assert cv.main(["--version", version, *task]) == 0
        mp.setattr(evaluate, "load_seldnet", lambda paths: round3_tree.AllFire())
        assert evaluate.main(["--test", "--version", "v3"]) == 0
        _index(paths, digests["v3"], {TEST[0]: (0.5, 0.9), TEST[1]: (0.8, 0.7)})
        yield paths, digests


def test_v3s_records_name_the_manifests_they_were_made_from(made_v3):
    paths, _ = made_v3
    identity = full.inputs_identity(paths, "v3")
    assert len(identity) == 2
    models = model_dir(paths, "v3")
    for name in train.candidates("v3"):
        for k in range(5):
            assert _json(cv._record_paths(models, name, k)[0])["inputs"] == identity
    chosen = _json(models / "chosen.json")
    assert _json(cv.cv_dir(models) / "choice.json")["inputs"] == identity
    assert chosen["inputs"] == identity
    out = models / chosen["candidate"]
    assert _json(out / "training.json")["inputs"] == identity
    assert train.load(out / "model.pt")[1]["inputs"] == identity
    assert _json(models / "evaluation.json")["meta"]["inputs"] == identity
    v2 = model_dir(paths, "v2")  # v2 reads neither directory: its records name none
    for record in (
        cv.cv_dir(v2) / "choice.json",
        v2 / "chosen.json",
        cv._record_paths(v2, "band80-mhd3", 0)[0],
        cv._record_paths(cv.ablation_dir(paths, "v2", "band0-mhd3"), "band0-mhd3", 0)[
            0
        ],
    ):
        assert "inputs" not in _json(record), record


def test_a_changed_masks_full_file_stops_every_read_of_v3s_records(made_v3):
    paths, _ = made_v3
    models = model_dir(paths, "v3")
    file = next(full.masks_full_dir(paths).glob("*_clean.npz"))
    st = file.stat()
    os.utime(file, ns=(st.st_atime_ns, st.st_mtime_ns + 10**9))
    try:
        with pytest.raises(ValueError, match="changed"):
            cv.candidate_frames(paths, "v3", "band0-mhd3")
        with pytest.raises(ValueError, match="changed"):
            cv.run_choose(paths, models, "v3")
        with pytest.raises(ValueError, match="changed"):
            train.train_from_cv(paths, models, version="v3")
        with pytest.raises(ValueError, match="changed"):
            evaluate.load_test(paths, models, "v3")
        with pytest.raises(ValueError, match="changed"):
            where = cv.ablation_dir(paths, "v3", "band80-mhd10")
            cv.run_fold(
                paths,
                where,
                candidate="band80-mhd10",
                fold=0,
                version="v3",
                ablate=True,
            )
        assert not cv._record_paths(where, "band80-mhd10", 0)[0].exists()
    finally:
        os.utime(file, ns=(st.st_atime_ns, st.st_mtime_ns))
    assert len(cv.candidate_frames(paths, "v3", "band0-mhd3")) == len(POOL)


def test_a_record_naming_no_inputs_stands_on_inputs_json(made_v3):
    paths, _ = made_v3
    models = model_dir(paths, "v3")
    record_file = cv._record_paths(models, "band0-mhd10", 2)[0]
    data = record_file.read_bytes()
    stripped = {k: v for k, v in json.loads(data).items() if k != "inputs"}
    record_file.write_text(json.dumps(stripped))
    try:
        with pytest.raises(ValueError, match="names no inputs"):
            cv.candidate_frames(paths, "v3", "band0-mhd10")
        full.inputs_file(paths, "v3").write_text(
            json.dumps(full.inputs_identity(paths, "v3"))
        )
        assert len(cv.candidate_frames(paths, "v3", "band0-mhd10")) == len(POOL)
        full.inputs_file(paths, "v3").write_text(
            json.dumps(dict.fromkeys(full.inputs_identity(paths, "v3"), "0" * 64))
        )
        with pytest.raises(ValueError, match="inputs.json: made from other"):
            cv.candidate_frames(paths, "v3", "band0-mhd10")
    finally:
        record_file.write_bytes(data)
        full.inputs_file(paths, "v3").unlink(missing_ok=True)


def test_an_ablation_fold_is_trained_as_the_versions_own_on_its_folds(made_v3):
    paths, _ = made_v3
    v3, v2 = model_dir(paths, "v3"), model_dir(paths, "v2")
    where = cv.ablation_dir(paths, "v3", "band80-mhd3")
    assert where == paths.runs / "ae_xpower" / "ablation" / "v3" / "band80-mhd3"
    same = (
        "shots",
        "stop_shots",
        "train_shots",
        "stop_fold",
        "train_folds",
        "folds_sha256",
        "labels_sha256",
        "source_sha256",
        "config",
        "seed",
        "pilot",
        "inputs",
    )
    for k in range(5):
        record = _json(cv._record_paths(where, "band80-mhd3", k)[0])
        own = _json(cv._record_paths(v3, "band0-mhd3", k)[0])
        assert record["ablation"] is True and "ablation" not in own
        assert (record["version"], record["candidate"]) == ("v3", "band80-mhd3")
        assert record["band_khz"] == [80.0, 250.0] and record["mhd_weight"] == 3.0
        assert {key: record[key] for key in same} == {key: own[key] for key in same}
        assert (
            record["config"]
            == _json(cv._record_paths(v2, "band80-mhd3", k)[0])["config"]
        )
        # v2's ablation: its 0-2 s folds, its source table, no masks-full.
        other = _json(
            cv._record_paths(
                cv.ablation_dir(paths, "v2", "band0-mhd3"), "band0-mhd3", k
            )[0]
        )
        v2own = _json(cv._record_paths(v2, "band80-mhd3", k)[0])
        assert other["band_khz"] == [0.0, 250.0] and other["ablation"] is True
        assert {key: other.get(key) for key in same} == {
            key: v2own.get(key) for key in same
        }
        assert other["source_sha256"] == evaluate.source_table(paths).sha256
    for version, name in (("v3", "band80-mhd3"), ("v2", "band0-mhd3")):
        assert not (cv.cv_dir(model_dir(paths, version)) / name).exists()


def test_an_ablation_writes_nowhere_else_and_never_again(made_v3):
    paths, _ = made_v3
    before = {p: p.read_bytes() for p in _files(paths.root / "models")}
    where = cv.ablation_dir(paths, "v3", "band80-mhd3")
    record = cv._record_paths(where, "band80-mhd3", 0)[0]
    kept = record.read_bytes()
    task = {"fold": 0, "version": "v3", "ablate": True}
    with pytest.raises(ValueError, match="writes only under"):
        cv.run_fold(paths, model_dir(paths, "v3"), candidate="band80-mhd3", **task)
    pilot = paths.runs / "ae_xpower" / "pilot" / "v3"
    with pytest.raises(ValueError, match="writes only under"):
        cv.run_fold(paths, pilot, candidate="band80-mhd3", pilot=10, **task)
    own = cv.ablation_dir(paths, "v3", "band0-mhd3")
    with pytest.raises(ValueError, match="own candidates"):
        cv.run_fold(paths, own, candidate="band0-mhd3", **task)
    with pytest.raises(FileExistsError, match="trained once; nothing is replaced"):
        cv.run_fold(paths, where, candidate="band80-mhd3", **task)
    with pytest.raises(ValueError, match="default seed only"):
        cv.run_fold(paths, where, candidate="band80-mhd3", seed=7, **task)
    assert record.read_bytes() == kept
    assert {p: p.read_bytes() for p in _files(paths.root / "models")} == before


def test_an_ablation_pilot_writes_under_its_own_pilot(made_v3, monkeypatch):
    paths, _ = made_v3
    monkeypatch.setattr(train, "fit", _fake_fit([]))
    task = ["--version", "v3", "--candidate", "band80-mhd10", "--fold", "1"]
    assert cv.main([*task, "--ablate", "--pilot", "10"]) == 0
    where = cv.ablation_dir(paths, "v3", "band80-mhd10", 10)
    record = _json(cv._record_paths(where, "band80-mhd10", 1)[0])
    assert record["pilot"] == 10 and record["ablation"] is True
    assert len(record["shots"]) == 2 and record["config"]["epochs"] == cv.PILOT_EPOCHS
    full_run = cv.ablation_dir(paths, "v3", "band80-mhd10")
    assert cv._lacking(full_run, "band80-mhd10") == [0, 1, 2, 3, 4]


def test_candidate_frames_reads_an_ablation_as_the_choice_reads_a_record(made_v3):
    paths, _ = made_v3
    frames = cv.candidate_frames(paths, "v3", "band80-mhd3")
    own = cv.candidate_frames(paths, "v3", "band0-mhd3")
    assert sorted(f.shot for f in frames) == sorted(f.shot for f in own) == POOL
    assert all(len(f.owner) == 300 for f in frames)
    with pytest.raises(FileNotFoundError, match="folds \\[0, 1, 2, 3, 4\\] missing"):
        cv.candidate_frames(paths, "v3", "band80-mhd10")
    where = cv.ablation_dir(paths, "v3", "band80-mhd3")
    record_file = cv._record_paths(where, "band80-mhd3", 3)[0]
    data = record_file.read_bytes()
    record_file.write_text(json.dumps(json.loads(data) | {"mhd_weight": 10.0}))
    try:
        with pytest.raises(ValueError, match="mhd_weight differs"):
            cv.candidate_frames(paths, "v3", "band80-mhd3")
    finally:
        record_file.write_bytes(data)


def test_the_summary_puts_the_ablation_beside_the_versions_own(made_v3):
    paths, _ = made_v3
    s = diagnosis.ablation_summary(paths)
    runs = s["runs"]
    assert list(runs) == [f"{v}/{n}" for v in ("v3", "v2") for n in diagnosis.RUNS[v]]
    assert (runs["v3/band80-mhd10"]["status"], runs["v3/band80-mhd10"]["lacking"]) == (
        "not run",
        [0, 1, 2, 3, 4],
    )
    assert all(r["status"] == "done" for k, r in runs.items() if k != "v3/band80-mhd10")
    assert s["folds_sha256"]["v2"] == s["folds_sha256"]["v3"]
    assert s["snapshots"]["frames_differing"] == 0
    assert runs["v3/band0-mhd3"]["frames"]["scored"] == len(POOL) * 300
    assert runs["v3/band0-mhd3"]["frames_before_2s"]["scored"] == len(POOL) * 200
    assert "rows_before_2s" not in runs["v2/band80-mhd3"]
    # P(AE) as designed: at 0.80, v3's band0-mhd3 and v2's band80-mhd3 fire on
    # every MHD frame (0.83), v3's band80-mhd3 on none (it is not band0-mhd3)
    # and v2's band0-mhd3 on nothing (0.72).
    factors = {
        (e["a"][0] + "/" + e["a"][1], e["b"][0] + "/" + e["b"][1]): e
        for e in s["factors"]
    }
    band3 = factors[("v3/band80-mhd3", "v3/band0-mhd3")]
    assert band3["status"] == "done"
    assert band3["at"]["0.80"]["fp_rate_mhd"] == [0.0, 1.0]
    assert band3["at"]["0.80"]["d_fp_rate_mhd"] == 1.0
    window3 = factors[("v2/band80-mhd3", "v3/band80-mhd3")]
    assert window3["b_frames"] == "before"
    assert window3["at"]["0.80"]["fp_rate_mhd"] == [1.0, 0.0]
    assert factors[("v3/band80-mhd10", "v3/band0-mhd10")]["status"] == "not run"
    assert factors[("v2/band80-mhd10", "v3/band80-mhd10")]["status"] == "not run"
    for measure in ("peak F1", "MHD FP"):
        verdict = s["verdict"][measure]
        assert set(verdict["mean_abs"]) == {"band", "window"}
        assert verdict["larger"] in ("band", "window")
    # The MHD split: masks-full lights 98 kHz, in TokEye's AE band, on every
    # MHD frame, so all of v3's MHD frames hold lit AE-band pixels.
    for key, r in runs.items():
        if "mhd_split" in r:
            split = r["mhd_split"]
            total = sum(split[c]["frames"] for c in diagnosis.AE_LIT)
            assert total == r["frames"]["mhd_absent"], key
    v3split = runs["v3/band0-mhd3"]["mhd_split"]
    assert v3split["at least half"]["frames"] == len(POOL) * len(MHD_FRAMES)
    threshold = _json(cv.cv_dir(model_dir(paths, "v3")) / "choice.json")["threshold"]
    fired = len(POOL) * len(MHD_FRAMES) if threshold <= 0.83 else 0
    assert v3split["threshold"] == threshold
    assert v3split["at least half"]["fp"] == fired
    text = diagnosis.summary_md(s)
    for heading in (
        "## Frontier on v3's folds (whole window): F1 / MHD FP",
        "## v3's runs scored on their frames before 2 s: F1 / MHD FP",
        "## Frontier on v2's folds (0-2 s): F1 / MHD FP",
        "## Factors (b minus a)",
        "## MHD false positives by TokEye's AE band",
    ):
        assert heading in text
    assert "| v3/band80-mhd10 (ablation) | 80-250 | 10 | not run" in text
    before = _files(paths.root)
    assert diagnosis.main(["--ablation"]) == 0
    out = paths.runs / "ae_xpower" / "ablation" / "summary.md"
    assert _files(paths.root) - before == {out}
    assert out.read_text() == text


def test_the_diagnosis_counts_v3s_test_frames_as_its_record_does(made_v3):
    paths, _ = made_v3
    models = model_dir(paths, "v3")
    evaluation = _json(models / "evaluation.json")
    choice = _json(cv.cv_dir(models) / "choice.json")
    d = diagnosis.diagnosis(paths, "v3")
    n = d["shot_counts"]
    assert (n["whole"], n["whole_present"]) == (600, 200)
    assert (n["whole"], n["whole_present"]) == (
        evaluation["frames"]["scored"],
        evaluation["frames"]["present"],
    )
    assert (n["table"], n["before"], n["after"]) == (400, 400, 200)
    assert (n["before_present"], n["after_present"]) == (120, 80)
    assert n["table_only_present"] == n["before_only_present"] == 0
    before, after = d["test"]["recall"]["before"], d["test"]["recall"]["after"]
    assert before[0] == before[1] == d["test"]["recall"]["table_0_2s"]
    assert after[0] == after[1]
    assert d["oof"]["all"]["scored"] == choice["frames"]["scored"] == len(POOL) * 300
    assert d["oof"]["before"]["scored"] == len(POOL) * 200
    assert d["oof"]["after"]["scored"] == len(POOL) * 100
    assert d["oof"]["all"]["recall"] == choice["chosen_row"]["recall"]
    assert d["test"]["always_f1"] == evaluation["methods"]["always"]["f1"]
    low = evaluation["differences"]["f1_minus_always"]["low"]
    assert d["need"]["a3"] == pytest.approx(-low)
    assert [s["shot"] for s in d["shots"]] == TEST  # 111 at 0.5, then 112 at 0.8
    assert d["long_window"] == {"shots": 2, "lower_whole": 1, "of": 2}
    assert [r["threshold"] for r in d["curve"]] == [float(t) for t in train.THRESHOLDS]
    text = diagnosis.diagnosis_md(d)
    assert "## For the owner's list (Task 3.1)" in text
    assert (
        "- v4's test design: stratify the test split by window length, or score "
        "the frames after 2 s separately." in text
    )
    assert f"| {d['threshold']:.2f} (chosen) |" in text
    kept = (models / "evaluation.json").read_bytes()
    before_files = _files(paths.root)
    assert diagnosis.main(["--version", "v3"]) == 0
    assert _files(paths.root) - before_files == {models / "diagnosis.md"}
    assert (models / "diagnosis.md").read_text() == text
    assert (models / "evaluation.json").read_bytes() == kept


def test_the_diagnosis_is_for_a_whole_window_version_and_the_chosen_models_gallery(
    made_v3, capsys
):
    paths, digests = made_v3
    with pytest.raises(ValueError, match="not scored over whole windows"):
        diagnosis.diagnosis(paths, "v2")
    index = gallery_dir(paths, "v3") / "index.csv"
    kept = index.read_bytes()
    index.write_text(kept.decode().replace(digests["v3"], "0" * 64))
    try:
        with pytest.raises(ValueError, match="names another snapshot"):
            diagnosis.diagnosis(paths, "v3")
    finally:
        index.write_bytes(kept)
    with pytest.raises(SystemExit):
        diagnosis.main(["--version", "v2"])
    assert "not scored over whole windows" in capsys.readouterr().err
