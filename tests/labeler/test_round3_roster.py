"""The roster interpreter figure (D51): one non-blind roster shot with corpus CO2,
AE from the frame model and SegNet over its corpus rows, the other phenomena from
the frame models' suggestion tables, drawn as suggestions in a scratch build."""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest

from labeler import frames
from labeler.ae.seg import train as seg_train
from labeler.ae.seg.model import SegNet, SegNetConfig
from labeler.ae.xpower.data import FULL_BAND_KHZ
from labeler.ae.xpower.evaluate import chosen_model
from labeler.config import DEFAULT_LABEL_TABLES, sha256_of
from labeler.events import suggestions
from labeler.events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from labeler.events.review.labels import Label
from labeler.paper import AE, ORDER, paper_dir, roster, shots, title

from . import ae_tree

SHOT = 198672  # non-blind in the frozen cohort, 2024, 9 s of corpus CO2
OTHER = 198806  # non-blind, 2024, 9 s of corpus CO2
BLIND = 198955  # blind, 9 s of corpus CO2
NO_CO2 = 186636  # non-blind, no corpus CO2
WINDOW = (0, 1000)
NTM, HMODE = "neoclassical_tearing_mode", "high_confinement_mode"
ELM, SAW = "edge_localized_mode", "sawtooth_oscillation"
ELM_ROWS = [(0, 200, 0), (200, 600, PRESENT), (600, 1000, 0)]
SAW_ROWS = [(0, 100, NOT_OBSERVABLE), (100, 400, UNCERTAIN), (400, 1000, PRESENT)]


def _says(net, rows, first, n, *, band, context=20):
    """P(AE) 1 over 300-900 ms and 0 elsewhere, whatever the rows."""
    frame = first + np.arange(n)
    prob = ((frame >= 30) & (frame < 90)).astype(np.float32)
    return prob, np.ones(n, dtype=bool)


def _table(paths, category: str, rows_by_shot: dict) -> None:
    """A frame model's v1 suggestion table holding `rows_by_shot`."""
    path = roster.table_file(paths, category)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(suggestions.COLUMNS)]
    for shot, spans in rows_by_shot.items():
        lines += [f"{shot},{c},{a},{b},0.9" for a, b, c in spans]
    path.write_text("\n".join(lines) + "\n")


def _v2_seg(paths):
    """SegNet v2's model.pt at threshold 0: every pixel of 0-250 kHz is AE."""
    out = paths.root / "models" / "ae_seg" / "v2"
    seg_train.save(
        out,
        SegNet(SegNetConfig(width=4)),
        threshold=0.0,
        split={},
        history=[],
        config=seg_train.TrainConfig(width=4),
        inputs={},
        band_khz=FULL_BAND_KHZ,
        version="v2",
    )
    return out / "model.pt"


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The frozen cohort, SHOT's corpus CO2 to 1.1 s, a v3 frame model, SegNet v2,
    the ELM and sawtooth frame models' tables with SHOT, H-mode's without it, and
    none for NTM."""
    paths = ae_tree.build(tmp_path, {SHOT: "valid"})
    cohort = paths.label_tables / "catalog" / "cohort.csv"
    cohort.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEFAULT_LABEL_TABLES / "catalog" / "cohort.csv", cohort)
    ae_tree.corpus(tmp_path, [SHOT], seconds=(-0.1, 1.1), tone_s=(0.3, 0.9))
    ae_tree.chosen(paths, {SHOT: "test"}, "v3")
    _v2_seg(paths)
    _table(paths, ELM, {SHOT: ELM_ROWS})
    _table(paths, SAW, {SHOT: SAW_ROWS})
    _table(paths, HMODE, {OTHER: [(0, 1000, PRESENT)]})
    ae_tree.env(monkeypatch, paths)
    monkeypatch.setattr(roster, "probabilities", _says)
    return paths


def test_the_tracks_are_the_frame_models_in_the_papers_order():
    assert list(roster.TABLES) == [c for c in ORDER if c != AE]
    assert set(roster.TABLES.values()) <= set(frames.SPECS)
    assert roster.TABLE_VERSION == "v1"
    assert (roster.AE_VERSION, roster.SEG_VERSION) == ("v3", "v2")


def test_the_candidates_are_the_non_blind_roster_with_corpus_co2(tree):
    found = roster.candidates(tree)
    numbers = [c.shot for c in found]
    assert len(found) == 153 and numbers == sorted(numbers)
    assert SHOT in numbers and OTHER in numbers
    assert BLIND not in numbers and NO_CO2 not in numbers
    assert roster.Candidate(SHOT, 2024, (9, 5935)) in found


def test_each_table_is_read_by_shot_and_a_missing_one_is_none(tree):
    tables = roster.read_tables(tree)
    assert list(tables) == list(roster.TABLES)
    assert tables[NTM] is None and SHOT not in tables[HMODE]
    assert tables[HMODE][OTHER] == Label(WINDOW, ((0, 1000, PRESENT),))
    assert tables[ELM][SHOT] == Label(WINDOW, ((200, 600, PRESENT),))
    assert roster.present_ms(tables[SAW][SHOT]) == 600
    assert roster.present_ms(None) == 0


def test_the_pick_takes_the_most_phenomena_then_the_most_time_then_the_shot():
    def label(*spans):
        return Label(WINDOW, tuple(spans))

    found = [roster.Candidate(s, 2024, WINDOW) for s in (1, 2, 3, 4)]
    tables = {
        NTM: None,
        HMODE: {},
        ELM: {1: label((0, 900, PRESENT)), 2: label((0, 20, PRESENT))},
        SAW: {
            2: label((0, 20, PRESENT)),
            3: label((0, 50, PRESENT)),
            4: label((0, 30, PRESENT), (30, 900, UNCERTAIN)),
        },
    }
    assert roster.present_ms(tables[SAW][4]) == 30, "uncertain time is not present"
    assert roster.pick(found, tables).shot == 2, "two phenomena beat one of 900 ms"
    tables[ELM][3] = label((0, 10, PRESENT))
    assert roster.pick(found, tables).shot == 3, "two phenomena, 60 ms against 40"
    tables[ELM][2] = label((0, 40, PRESENT))
    assert roster.pick(found, tables).shot == 2, "two, 60 ms each: the lower shot"
    with pytest.raises(ValueError):
        roster.pick([], tables)


def _shot(paths):
    return roster.roster_shot(
        paths,
        roster.Candidate(SHOT, 2024, WINDOW),
        model_file=chosen_model(paths.root / "models" / "ae_xpower" / "v3"),
        seg_file=paths.root / "models" / "ae_seg" / "v2" / "model.pt",
        tables=roster.read_tables(paths),
    )


def test_the_shot_runs_ae_over_its_corpus_rows_and_reads_the_tables(tree):
    s = _shot(tree)
    assert (s.first, len(s.prob), s.threshold) == (0, 100, 0.5)
    assert list(s.tracks) == list(ORDER)
    assert np.flatnonzero(s.tracks[AE] == PRESENT).tolist() == list(range(30, 90))
    assert np.flatnonzero(s.tracks[ELM] == PRESENT).tolist() == list(range(20, 60))
    saw = s.tracks[SAW]
    assert (saw[:10] == NOT_OBSERVABLE).all() and (saw[10:40] == UNCERTAIN).all()
    assert (saw[40:] == PRESENT).all()
    assert (s.tracks[NTM], s.tracks[HMODE]) == (roster.NO_TABLE, roster.NOT_APPLIED)
    assert s.image.shape == s.mask.shape and s.image.shape[1] == s.grid.n
    assert s.mask.all(), "SegNet v2 at threshold 0 over 0-250 kHz"
    assert s.grid.dt_ms == pytest.approx(shots.PICTURE_LEVEL * 0.256, rel=0.05)


def _labels(fig) -> set[str]:
    return {
        name
        for legend in fig.legends
        for name in (t.get_text() for t in legend.get_texts())
    }


def test_the_figure_says_suggestions_and_has_no_band_line(tree, tmp_path):
    s = _shot(tree)
    fig = roster.draw(s, tmp_path / "roster")
    spec, *tracks = fig.axes
    heading = "shot 198672 (2024): the models' suggestions, not reviewed"
    assert spec.get_title() == heading
    assert list(spec.lines) == [], "no 80 kHz line"
    assert [ax.get_ylabel() for ax in tracks] == [title(c) for c in ORDER]
    by_category = dict(zip(ORDER, tracks, strict=True))
    for category, text in ((NTM, roster.NO_TABLE), (HMODE, roster.NOT_APPLIED)):
        assert [t.get_text() for t in by_category[category].texts] == [text]
        assert not by_category[category].collections, category
    for category in (AE, ELM, SAW):
        assert by_category[category].collections, category
    names = _labels(fig)
    assert {"suggested: present", "suggested: uncertain", shots.MASK_LABEL} <= names
    assert "suggested: not observable" in names
    assert not any(n.startswith("owner") for n in names), "nothing here is reviewed"
    assert (tmp_path / "roster.pdf").is_file() and (tmp_path / "roster.png").is_file()


def test_main_refuses_the_papers_directory_and_a_shot_off_the_list(tree, tmp_path):
    for out in (paper_dir(tree), paper_dir(tree) / "roster"):
        with pytest.raises(roster.ScratchOnly):
            roster.main(["--out", str(out)])
    assert not paper_dir(tree).exists()
    for shot in (BLIND, NO_CO2):
        with pytest.raises(ValueError, match=f"{shot}: not a non-blind roster shot"):
            roster.main(["--out", str(tmp_path / "b"), "--shot", str(shot)])
    assert not (tmp_path / "b").exists()


def test_main_picks_draws_and_records(tree, tmp_path, monkeypatch, capsys):
    # OTHER suggests H-mode alone, 1000 ms; SHOT suggests ELMs and sawteeth
    found = [roster.Candidate(s, 2024, WINDOW) for s in (OTHER, SHOT)]
    monkeypatch.setattr(roster, "candidates", lambda paths: found)
    out = tmp_path / "scratch" / "paper-round-three-b"
    assert roster.main(["--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["shot"], printed["out"]) == (SHOT, str(out))
    record = json.loads((out / roster.MANIFEST).read_text())
    assert record["shot"] == SHOT and record["pick_rule"] == roster.PICK_RULE
    assert record["window"] == list(WINDOW) and record["candidates"] == 2
    model = chosen_model(tree.root / "models" / "ae_xpower" / "v3")
    assert record["ae_model"] == {"path": str(model), "sha256": sha256_of(model)}
    assert record["seg_model"]["path"].endswith("models/ae_seg/v2/model.pt")

    def pinned(category):
        path = roster.table_file(tree, category)
        return {"path": str(path), "sha256": sha256_of(path)}

    assert record["tables"] == {
        "ntm_frames": None,
        "hmode_frames": pinned(HMODE),
        "elm_frames": pinned(ELM),
        "sawtooth_frames": pinned(SAW),
    }
    assert record["tier"] == "suggestions"
    assert (out / f"{roster.STEM}.pdf").is_file()
    assert (out / f"{roster.STEM}.png").is_file()

    assert roster.main(["--out", str(out), "--shot", str(SHOT)]) == 0
    record = json.loads((out / roster.MANIFEST).read_text())
    assert record["pick_rule"] == roster.NAMED == "named by --shot"
