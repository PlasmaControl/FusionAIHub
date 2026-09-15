"""End to end against the real DIII-D stores: staged HDF5 in, a database and a printed shot out.

Everything else in the suite runs on synthetic fixtures, which is what makes them fast and
deterministic -- and also what makes them unable to catch a wrong node name, a staged column that
is named but empty, or a logbook record whose shape does not match the parser. This module is the
one that touches the real thing. It skips cleanly, as a whole, when the EKOLEMEN stores are not
mounted, so `pytest` still passes on a laptop.

Every number asserted here was measured in this environment against
/scratch/gpfs/EKOLEMEN/d3d_fusion_data/161172.h5, not copied from a plan. Nothing is written
anywhere but a pytest tmp directory: SHOT_DESIGN_DATA_ROOT is redirected, which moves raw_dir, db_dir
and text_cache_dir while leaving the read-only stores where they are.


Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from shot_design import cli, config
from shot_design.schema import ShotSummary
from shot_design.shotdb import build, text

STAGED_DIR = Path("/scratch/gpfs/EKOLEMEN/d3d_fusion_data")
GATE = STAGED_DIR / "160904.h5"
SHOT, OTHER = 161172, 160904
_paths = config.load_paths()
GATES = [GATE, STAGED_DIR / f"{SHOT}.h5", _paths.logs_jsonl, _paths.qh_database_csv, _paths.shotsummary_raw_dir]

pytestmark = [
    pytest.mark.real_data,
    pytest.mark.skipif(not all(p.exists() for p in GATES), reason="real staged/text/QH stores not mounted"),
]


def _stub_embeddings(mp) -> None:
    """MiniLM is exercised by test_minilm_loads_offline below; the build tests are about the
    HDF5 and logbook paths, and loading a transformer per test would dominate their runtime."""
    mp.setattr(text, "embed_texts", lambda texts: np.zeros((len(texts), 384), np.float32))


@pytest.fixture(scope="module")
def real_paths(tmp_path_factory):
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("SHOT_DESIGN_DATA_ROOT", str(tmp_path_factory.mktemp("ideate-real")))
        mp.delenv("SHOT_DESIGN_PATHS", raising=False)
        paths = config.load_paths()
        for d in (paths.raw_dir, paths.db_dir, paths.text_cache_dir):
            d.mkdir(parents=True, exist_ok=True)
        assert paths.staged_raw_dir == STAGED_DIR  # the redirect must not move the read-only store
        yield paths


@pytest.fixture(scope="module")
def real_db(real_paths):
    """One CLI build of two staged shots, shared by the tests below."""
    with pytest.MonkeyPatch.context() as mp:
        _stub_embeddings(mp)
        assert cli.main(["build", "--shots", str(SHOT), str(OTHER), "--workers", "1", "--no-encode"]) == 0
    # The stub is undone before the tests run: it must not leak into the MiniLM test below.
    return real_paths


def test_build_record_from_the_staged_store(real_paths):
    text.build_logs_subset(real_paths, {SHOT})
    rec, shapes = build.build_record(SHOT, real_paths, build.load_build_cfg())

    assert [s.name for s in rec.segments] == ["full", "ramp_up", "flat_top", "ramp_down"]
    flat = rec.segment("flat_top")
    assert (round(flat.t0_ms, 1), round(flat.t1_ms, 1)) == (529.5, 5199.0)
    assert flat.raw["ip_mean"] == pytest.approx(985281.3, rel=1e-6)  # 0.985 MA
    assert flat.raw["bt_mean"] == pytest.approx(1.9496199, rel=1e-6)  # T
    assert flat.raw["pnbi_total_mean"] == pytest.approx(8207485.5, rel=1e-6)  # W

    # EFIT scalars are read from a staged file, which does not record which EFIT run produced
    # them -- so the value is real and its provenance is explicitly assumed.
    assert flat.derived["betan_mean"] == pytest.approx(2.2799129, rel=1e-6)
    assert rec.derived_provenance["betan"].tool == "EFIT"
    assert rec.derived_provenance["betan"].assumed is True

    # The two halves of the status rule, on real data: a beam the staged file carries is present;
    # q95 is named in the staged q_psi group's block_items, holds nothing, and is listed in that
    # group's missing_channels -- the staged producer's miss, which EFIT01 can still serve, so it
    # is `pending` (a fetch to-do), never `unavailable`.
    assert rec.coverage["pnbi_15L"] == "present"
    assert rec.coverage["q95"] == "pending"
    # The five corpus/labeler-only signals (nbi_torque_total, nbi_voltage_mean,
    # gasflow_total, rmp_total, pcbcoil) have no d3d_fusion_data address at all, so on this
    # reader they are `unavailable` -- not `pending`, which would promise a fetch could get them.
    assert set(rec.coverage.values()) == {"present", "pending", "unavailable"}
    assert {n for n, v in rec.coverage.items() if v == "unavailable"} == {
        "nbi_torque_total", "nbi_voltage_mean", "gasflow_total", "rmp_total", "pcbcoil",
    }  # fmt: skip
    assert set(rec.raw_sources.values()) == {"staged"}  # nothing fetched into this tmp raw_dir

    assert rec.human.run_id == "20150113A" and rec.human.mpid == "2014-21-19"
    assert str(rec.shot_date) == "2015-01-13" and rec.campaign == "2014_2015"
    assert rec.human.mp_title == "Optimizing pedestal height in hybrid regimes"
    assert rec.human.verdict == "good"
    assert rec.labels.regime == "H" and rec.labels.regime_source == "text"
    assert rec.outcome.ip_target_hit is True
    assert rec.outcome.end_reason == "programmed_rampdown"
    assert {s.shape for s in shapes.values()} == {(140,)}  # 7 waveforms x 20 points


def test_the_qh_database_cannot_label_this_era(real_paths):
    """`regime_source: "database"` is unreachable for these shots and the test says so out loud:
    QH_Database.csv covers 173694-175544, which does not intersect the 2014-15 staged store or
    configs/shot_design/shot_lists/poc_v1.yaml. Regimes here come from the logbook text or from
    geometry."""
    qh = build._qh_shots(str(real_paths.qh_database_csv))
    assert (min(qh), max(qh)) == (173694, 175544)
    assert not qh & set(config.load_shot_list("poc_v1"))


def test_cli_show_prints_the_real_record(real_db, capsys):
    capsys.readouterr()
    assert cli.main(["show", str(SHOT)]) == 0
    out = capsys.readouterr().out
    assert f"Shot {SHOT}   2015-01-13   run 20150113A" in out
    assert "MP 2014-21-19  Optimizing pedestal height in hybrid regimes" in out
    assert "Ip 985 kA, Bt 1.95 T, PNBI 8.21 MW" in out  # describe.segment_line, same as query
    assert "[flat_top] 530-5199 ms" in out
    assert "chief operator: Plasma shot; ok" in out  # verbatim, from parse_log_entries
    assert "EFIT01 tree=efit01 (run id assumed" in out
    assert "pending (19)" in out and "q95" in out


def test_cli_export_round_trips_through_shotsummary(real_db, tmp_path, capsys):
    capsys.readouterr()
    assert cli.main([
        "export", str(SHOT), str(OTHER), "--format", "json", "--out", str(tmp_path / "e.json")
    ]) == 0  # fmt: skip
    rows = json.loads((tmp_path / "e.json").read_text())
    summaries = [ShotSummary.model_validate(r) for r in rows]
    assert [s.shot for s in summaries] == [SHOT, OTHER]
    assert summaries[0].mpid == "2014-21-19"
    assert summaries[0].flat_top["ip_mean"] == pytest.approx(985281.3, rel=1e-6)
    assert summaries[0].description.startswith(
        'Shot 161172 (2015-01-13, run 20150113A, MP 2014-21-19 "Optimizing pedestal height in hybrid regimes").'
    )
    assert 0.0 < summaries[0].coverage_fraction <= 1.0


def test_cli_coverage_table_over_the_real_build(real_db, capsys):
    capsys.readouterr()
    assert cli.main(["coverage"]) == 0
    out = capsys.readouterr().out
    assert "85 fields over 2 shots" in out
    assert "[nbi]" in out and "[ech]" in out and "[gas]" in out and "[coil_rmp]" in out
    assert "q95" in out


def test_minilm_loads_offline_from_the_local_cache():
    """The operational fact a demo trips over: sentence_transformers calls huggingface_hub even
    with the checkpoint cached, and on a node with no outbound route that call hangs rather than
    failing. shot_design.cli defaults HF_HUB_OFFLINE=1 so this returns in seconds."""
    import os

    assert os.environ.get("HF_HUB_OFFLINE") == "1"  # set at shot_design.cli import
    cache = Path.home() / ".cache" / "huggingface" / "hub"
    if not list(cache.glob("models--sentence-transformers--all-MiniLM-L6-v2")):
        pytest.skip("MiniLM checkpoint is not in the local HF cache")
    vecs = text.embed_texts(["wide pedestal QH-mode at low torque", ""])
    assert vecs.shape == (2, 384)
    assert np.linalg.norm(vecs[0]) == pytest.approx(1.0, abs=1e-5)
    assert not vecs[1].any()  # an empty text is a zero row, never a random one
