"""The post-hoc record: MHD false positives by place, read after the one test."""

from __future__ import annotations

import json

import numpy as np
import pytest

from labeler.ae.xpower import evaluate, model_dir, posthoc

from .test_ae_xpower_evaluate import _frames
from .test_ae_xpower_final import _final_and_v1


def test_each_place_counts_its_own_mhd_frames():
    """Present 10-19 and 25-29 (a gap inside AE at 20-24); MHD at 5-7 (lead-in),
    21-22 (inside) and 35 (after); a second shot with no present frame, MHD at
    1-3. The model errs on 6, 21 and 2; SELDNet on every MHD frame."""
    owner = np.zeros(40, int)
    owner[10:20] = owner[25:30] = 1
    mhd = np.zeros(40, bool)
    mhd[[5, 6, 7, 21, 22, 35]] = True
    ours = np.zeros(40, bool)
    ours[[6, 21]] = True
    quiet = np.zeros(40, int)
    quiet_mhd = np.zeros(40, bool)
    quiet_mhd[1:4] = True
    quiet_ours = np.zeros(40, bool)
    quiet_ours[2] = True
    shots = [
        _frames(1, owner, mhd, ae_xpower=ours, seldnet=mhd),
        _frames(2, quiet, quiet_mhd, ae_xpower=quiet_ours, seldnet=quiet_mhd),
    ]
    found = posthoc.by_place(shots, ("ae_xpower", "seldnet"))
    counts = {
        place: (e["mhd_frames"], e["shots"], e["methods"]["ae_xpower"]["fp"])
        for place, e in found.items()
    }
    assert counts == {
        "lead_in": (3, 1, 1),
        "inside": (2, 1, 1),
        "after": (1, 1, 0),
        "no_present": (3, 1, 1),
    }
    assert all(e["methods"]["seldnet"]["fp"] == e["mhd_frames"] for e in found.values())
    inside = found["inside"]
    assert inside["methods"]["ae_xpower"]["rate"]["value"] == pytest.approx(0.5)
    assert inside["minus_seldnet"]["ae_xpower"]["value"] == pytest.approx(-0.5)
    assert "seldnet" not in inside["minus_seldnet"]


@pytest.fixture
def tested(tmp_path, monkeypatch):
    """v2's final model, tested once; the paths."""
    paths = _final_and_v1(tmp_path, monkeypatch)
    assert evaluate.main(["--test", "--version", "v2"]) == 0
    return paths


def test_the_record_refuses_a_test_record_naming_other_inputs(tested):
    """Each sha256 the test record names (model, chosen, split, labels, source
    table) must be what is read now, or nothing is written."""
    file = model_dir(tested, "v2") / "evaluation.json"
    saved = file.read_text()
    keys = (
        "model_sha256",
        "chosen_sha256",
        "split_sha256",
        "labels_copy_sha256",
        "labels_sha256",
        "source_sha256",
    )
    for key in keys:
        record = json.loads(saved)
        record["meta"][key] = "0" * 64
        file.write_text(json.dumps(record))
        with pytest.raises(ValueError, match=f"evaluation.json: {key} "):
            posthoc.run(tested, "v2")
    file.write_text(saved)
    assert not list(model_dir(tested, "v2").glob("posthoc.*"))
    assert not (tested.runs / "ae_xpower" / "posthoc").exists()


def test_the_record_never_writes_the_test_record(tested):
    models = model_dir(tested, "v2")
    test = [models / "evaluation.json", models / "evaluation.md"]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in test]
    assert posthoc.main(["--version", "v2"]) == 0
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in test] == before
    record = json.loads((models / "posthoc.json").read_text())
    assert record["post_hoc"] == posthoc.POST_HOC
    assert posthoc.POST_HOC in (models / "posthoc.md").read_text()
    evaluation = json.loads(test[0].read_text())
    assert record["frames"] == evaluation["frames"]
    placed = sum(e["mhd_frames"] for e in record["places"].values())
    assert placed == evaluation["frames"]["mhd_absent"]  # every MHD frame, once
    assert record["seed_study"]["status"] == "not run"


def test_a_pilot_writes_only_under_runs(tested):
    assert posthoc.main(["--version", "v2", "--limit", "1"]) == 0
    pilot = tested.runs / "ae_xpower" / "posthoc" / "v2"
    record = json.loads((pilot / "posthoc.json").read_text())
    assert record["limit"] == 1 and record["frames"]["shots"] == 1
    assert (pilot / "posthoc.md").is_file()
    assert not list(model_dir(tested, "v2").glob("posthoc.*"))
