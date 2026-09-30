"""The paper's interpreter figure, `fig_interpreter` (`labeler.paper.roster`): one
non-blind roster shot with corpus CO2, AE from the frame model and SegNet over its
corpus rows (the mask over the whole picture), the signals the frame models read
(F10), the other phenomena from the frame models' suggestion tables, drawn as
suggestions; by the paper build, and by the roster CLI into a scratch build."""

from __future__ import annotations

import builtins
import io
import json
import os
import shutil
from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pytest
from matplotlib.colors import to_rgba

from labeler import frames
from labeler.ae.seg import train as seg_train
from labeler.ae.seg.model import SegNet, SegNetConfig
from labeler.ae.seg.poi import ae_pixels
from labeler.ae.xpower.data import FULL_BAND_KHZ
from labeler.ae.xpower.evaluate import chosen_model
from labeler.config import DEFAULT_LABEL_TABLES, sha256_of
from labeler.events import suggestions
from labeler.events.catalog.states import NOT_OBSERVABLE, PRESENT, UNCERTAIN
from labeler.events.review import rows
from labeler.events.review.labels import Label
from labeler.paper import AE, ORDER, PAGE_IN, build, paper_dir, roster, shots, title

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
    """A frame model's `frames.VERSION` suggestion table holding `rows_by_shot`."""
    path = roster.table_file(paths, category)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [",".join(suggestions.COLUMNS)]
    for shot, spans in rows_by_shot.items():
        lines += [f"{shot},{c},{a},{b},0.9" for a, b, c in spans]
    path.write_text("\n".join(lines) + "\n")


def _seg(paths, band=FULL_BAND_KHZ, version="v3"):
    """SegNet's model.pt at threshold 0, its blob naming `band`: every pixel at
    or above its threshold."""
    out = paths.root / "models" / "ae_seg" / version
    seg_train.save(
        out,
        SegNet(SegNetConfig(width=4)),
        threshold=0.0,
        split={},
        history=[],
        config=seg_train.TrainConfig(width=4),
        inputs={},
        band_khz=band,
        version=version,
    )
    return out / "model.pt"


#: The fixture stores' grid: 1 ms columns, -100 to 1200 ms.
GRID = rows.Grid(-100.0, 1.0, 1300)
MODES = {"n": [1, 2, 3], "levels": 85, "colours": ["#ff0000", "#00aa00", "#0000ff"]}


def _trace(name, title, channels, units="", level=1.0):
    values = np.full((2, channels, GRID.n), level, np.float32)
    values[0] -= 0.5  # each column's minimum under its maximum
    return rows.TraceRow(name, title, values, y_units=units)


def _stores(paths) -> dict:
    """SHOT's review stores: the NTM's Mirnov power and n map, the ELM's
    PCPHD03 and FS01 D-alpha, the sawteeth's first two ECE groups and no SXR;
    no H-mode store. Each event's path."""
    power = np.zeros((50, GRID.n), np.uint8)
    power[10:20, 300:700] = 200  # 20-40 kHz, 200-600 ms
    codes = np.zeros((50, GRID.n), np.uint8)
    codes[10:20, 300:700] = 84 * 3 + 1  # the top level, n=2
    codes[40, 900:903] = 84 * 3  # three noise pixels of n=1, left out of the key
    image = {"y0": 0.5, "dy": 2.0, "y_units": "kHz", "z_lo": 0.0, "z_hi": 1.0}
    built = {
        NTM: [
            rows.ImageRow("power", "MPI66M322D power", power, z_units="dB", **image),
            rows.ImageRow(
                "modes",
                "toroidal n, MPI66M probes",
                codes,
                z_units="",
                modes=MODES,
                **image,
            ),
        ],
        ELM: [
            _trace("pcphd03", "D-alpha PCPHD03", 1),
            _trace("fs", "D-alpha FS01, the ELM spans' channel, clipped", 1),
        ],
        SAW: [
            _trace("ece0", "ECE Te, ch 20-23 (5 ms median)", 4, "keV", 1.0),
            _trace("ece1", "ECE Te, ch 24-27 (5 ms median)", 4, "keV", 2.0),
        ],
    }
    out = {}
    for event, found in built.items():
        path = paths.spectrogram_file(event, SHOT)
        path.parent.mkdir(parents=True, exist_ok=True)
        rows.write(path, GRID, found)
        out[event] = path
    return out


@pytest.fixture
def tree(tmp_path, monkeypatch):
    """The frozen cohort, SHOT's corpus CO2 to 1.1 s, a v2 frame model, SegNet v3
    (the paper's two), the ELM and sawtooth frame models' tables with SHOT,
    H-mode's without it, and none for NTM."""
    paths = ae_tree.build(tmp_path, {SHOT: "valid"})
    cohort = paths.label_tables / "catalog" / "cohort.csv"
    cohort.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(DEFAULT_LABEL_TABLES / "catalog" / "cohort.csv", cohort)
    ae_tree.corpus(tmp_path, [SHOT], seconds=(-0.1, 1.1), tone_s=(0.3, 0.9))
    ae_tree.chosen(paths, {SHOT: "test"}, "v2")
    _seg(paths)
    _table(paths, ELM, {SHOT: ELM_ROWS})
    _table(paths, SAW, {SHOT: SAW_ROWS})
    _table(paths, HMODE, {OTHER: [(0, 1000, PRESENT)]})
    ae_tree.env(monkeypatch, paths)
    monkeypatch.setattr(roster, "probabilities", _says)
    return paths


def test_the_tracks_are_the_frame_models_in_the_papers_order():
    assert list(roster.TABLES) == [c for c in ORDER if c != AE]
    assert set(roster.TABLES.values()) <= set(frames.SPECS)
    assert roster.TABLE_VERSION == frames.VERSION == "v2"
    assert (roster.AE_VERSION, roster.SEG_VERSION) == ("v2", "v3")
    assert build.INTERPRETER_SEG_VERSION == roster.SEG_VERSION
    assert roster.STEM == "fig_interpreter" and "fig_interpreter" in build.PRODUCTS
    assert not [f for f in build.OWNED if "roster" in f], "one interpreter figure"


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


def _shot(paths, seg_file=None):
    seg_file = seg_file or paths.root / "models" / "ae_seg" / "v3" / "model.pt"
    model_file = chosen_model(paths.root / "models" / "ae_xpower" / "v2")
    return roster.roster_shot(
        paths,
        roster.Candidate(SHOT, 2024, WINDOW),
        model=shots.Model.load(model_file, {}),
        segmentation=shots.Segmentation.load(seg_file),
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
    assert s.mask.all(), "SegNet at threshold 0 over the whole picture"
    assert s.seg_band == roster.whole_band(s.y0, s.dy, s.image.shape[0])
    assert s.grid.dt_ms == pytest.approx(shots.PICTURE_LEVEL * 0.256, rel=0.05)


def test_the_mask_is_segnet_over_the_whole_spectrogram(tree):
    # A blob naming 80-250 kHz, v1's band, at threshold 0: the roster's mask
    # holds the rows below 80 kHz too; the paper's figures keep the blob's band.
    seg_file = _seg(tree, band=(80.0, 250.0), version="v1")
    s = _shot(tree, seg_file)
    freqs = s.y0 + np.arange(s.image.shape[0]) * s.dy
    assert (freqs < 80).any() and s.mask[freqs < 80].all()
    assert s.mask.all()
    lo, hi = s.seg_band
    assert lo < freqs[0] and freqs[-1] < hi
    banded = ae_pixels(np.zeros_like(s.mask, float), 0.0, s.y0, s.dy, (80.0, 250.0))
    assert not banded[freqs < 80].any(), "the band the paper's figures keep"
    assert shots.ROSTER_MASK_LABEL == "segmentation"
    assert shots.MASK_LABEL == "segmentation: AE"


def test_the_examples_figure_keeps_its_mask_label(tmp_path):
    n = 100
    s = shots.AEShot(
        shot=1,
        split="test",
        grid=rows.Grid(0.0, 10.0, n),
        image=np.zeros((25, n), np.uint8),
        y0=5.0,
        dy=10.0,
        first=0,
        owner=np.zeros(n, np.int8),
        prob=np.zeros(n),
        threshold=0.5,
        f1=float("nan"),
        mask=np.ones((25, n), bool),
    )
    names = _labels(shots.draw_examples([s], tmp_path / "examples"))
    assert shots.MASK_LABEL in names and shots.ROSTER_MASK_LABEL not in names


def _labels(fig) -> set[str]:
    return {
        name
        for legend in fig.legends
        for name in (t.get_text() for t in legend.get_texts())
    }


def test_the_figure_says_suggestions_and_has_no_band_line(tree, tmp_path):
    s = _shot(tree)
    fig = roster.draw(s, tmp_path / "roster")
    spec, *rest = fig.axes
    panels, tracks = rest[: len(roster.PANELS)], rest[len(roster.PANELS) :]
    heading = "shot 198672 (2024): the models' suggestions, not reviewed"
    assert spec.get_title() == heading
    assert list(spec.lines) == [], "no 80 kHz line"
    # No store: every signal panel says so, and nothing is fetched.
    said = [[t.get_text() for t in ax.texts] for ax in panels]
    assert said == [[shots.NO_DATA.format(p.title)] for p in roster.PANELS]
    assert s.stores == dict.fromkeys((NTM, ELM, HMODE, SAW))
    assert [ax.get_ylabel() for ax in tracks] == [title(c) for c in ORDER]
    by_category = dict(zip(ORDER, tracks, strict=True))
    for category, text in ((NTM, roster.NO_TABLE), (HMODE, roster.NOT_APPLIED)):
        assert [t.get_text() for t in by_category[category].texts] == [text]
        assert not by_category[category].collections, category
    for category in (AE, ELM, SAW):
        assert by_category[category].collections, category
    names = _labels(fig)
    assert {"suggested: present", "suggested: uncertain"} <= names
    assert shots.ROSTER_MASK_LABEL in names and shots.MASK_LABEL not in names
    assert "suggested: not observable" in names
    assert not any(n.startswith("owner") for n in names), "nothing here is reviewed"
    assert (tmp_path / "roster.pdf").is_file() and (tmp_path / "roster.png").is_file()


def test_the_signal_panels_are_the_rows_the_frame_models_read(tree, tmp_path):
    stores = _stores(tree)
    s = _shot(tree)
    assert s.stores == {
        NTM: stores[NTM],
        ELM: stores[ELM],
        HMODE: None,
        SAW: stores[SAW],
    }
    by_title = {g.panel.title: g for g in s.signals}
    assert list(by_title) == [p.title for p in roster.PANELS]
    labels = [g.label for g in s.signals]
    assert labels == [
        "Mirnov\n(kHz)",
        "n\n(kHz)",
        "D-alpha\nFS01",
        "NBI",
        "ECE Te\n(keV)",
        "SXR",
    ]
    assert by_title["NBI power"].text == "no NBI power data", "no H-mode store"
    assert by_title["SXR"].text == "no SXR data", "no SXR row"
    # FS01, never PCPHD03, though PCPHD03 comes first in the store.
    [dalpha] = by_title["D-alpha FS"].rows
    assert dalpha.meta["title"].startswith("D-alpha FS01")
    ece = by_title["ECE Te"].rows
    assert [r.meta["title"][:16] for r in ece] == [
        "ECE Te, ch 20-23",
        "ECE Te, ch 24-27",
    ]
    # Each panel's rows are a frame model's roles, matched by title prefix.
    for panel in roster.PANELS:
        assert panel.roles and all(r.title.startswith(panel.title) for r in panel.roles)
    # Read over the figure's range, one column a store column here (1 ms).
    [power] = by_title["MPI66M322D power"].rows
    assert (power.t0, power.t1) == (-50.0, 1050.0) and power.values.shape == (50, 1100)
    fig = roster.draw(s, tmp_path / "roster")
    spec, *rest = fig.axes
    panels = dict(zip(by_title, rest[: len(roster.PANELS)], strict=True))
    assert all(ax.get_xlim() == spec.get_xlim() for ax in rest), "one time axis"
    assert len(panels["MPI66M322D power"].images) == 1
    mirnov = panels["MPI66M322D power"].images[0]
    assert mirnov.get_cmap().name == spec.images[0].get_cmap().name == "inferno"
    assert panels["MPI66M322D power"].get_ylim() == (0.0, 99.5)
    modes = panels["toroidal n"]
    rgb = modes.images[0].get_array()
    assert np.allclose(rgb[15, 400], [0.0, 0xAA / 255, 0.0]), "n=2 at its top level"
    assert np.allclose(rgb[0, 0], 0.0), "no mode is black"
    assert [t.get_text() for t in modes.get_legend().get_texts()] == ["n=2"]
    white = to_rgba(modes.get_legend().get_texts()[0].get_color())
    assert white == to_rgba("white"), "the key is white on the black map"
    assert len(panels["D-alpha FS"].collections) == 1
    ece_ax = panels["ECE Te"]
    assert len(ece_ax.collections) == 8, "two groups of four channels"
    key = [t.get_text() for t in ece_ax.get_legend().get_texts()]
    assert key == ["ch 20-23", "ch 24-27"]
    assert [t.get_text() for t in panels["SXR"].texts] == ["no SXR data"]
    names = _labels(fig)
    assert not any(n.startswith(("n=", "ch ")) for n in names), "the keys stay"
    width, height = fig.get_size_inches()
    assert width == PAGE_IN and height > 3.4


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
    monkeypatch.setattr(roster, "candidates", lambda paths, snap=None: found)
    out = tmp_path / "scratch" / "paper-round-three-b"
    assert roster.main(["--out", str(out)]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["shot"], printed["out"]) == (SHOT, str(out))
    record = json.loads((out / roster.MANIFEST).read_text())
    assert record["shot"] == SHOT and record["pick_rule"] == roster.PICK_RULE
    assert record["window"] == list(WINDOW) and record["candidates"] == 2
    model = chosen_model(tree.root / "models" / "ae_xpower" / "v2")
    assert record["ae_model"] == {"path": str(model), "sha256": sha256_of(model)}
    assert (record["ae_version"], record["seg_version"]) == ("v2", "v3")
    assert record["seg_model"]["path"].endswith("models/ae_seg/v3/model.pt")
    assert record["seg_band_khz"] == list(_shot(tree).seg_band)
    assert record["tables_version"] == frames.VERSION
    assert record["corpus"] == str(tree.corpus_file(SHOT)), "by path: not hashed"

    def pinned(category):
        path = roster.table_file(tree, category)
        return {"path": str(path), "sha256": sha256_of(path)}

    assert record["tables"] == {
        "ntm_frames": None,
        "hmode_frames": pinned(HMODE),
        "elm_frames": pinned(ELM),
        "sawtooth_frames": pinned(SAW),
    }
    assert record["stores"] == dict.fromkeys((NTM, ELM, HMODE, SAW))
    assert record["tier"] == "suggestions"
    assert (out / f"{roster.STEM}.pdf").is_file()
    assert (out / f"{roster.STEM}.png").is_file()

    stores = _stores(tree)
    assert roster.main(["--out", str(out), "--shot", str(SHOT)]) == 0
    record = json.loads((out / roster.MANIFEST).read_text())
    assert record["pick_rule"] == roster.NAMED == "named by --shot"
    assert record["stores"] == {
        event: None
        if event not in stores
        else {
            "path": str(stores[event]),
            "sha256": sha256_of(stores[event]),
        }
        for event in (NTM, ELM, HMODE, SAW)
    }


def test_the_build_draws_the_roster_figure_as_fig_interpreter(tree, tmp_path):
    stores = _stores(tree)
    out = tmp_path / "paper"
    argv = ["--out", str(out), "--version", "v2", "--seg-version", "v1"]
    assert build.main([*argv, "--shot", str(SHOT)]) == 0
    manifest = json.loads((out / build.MANIFEST).read_text())
    files = build.files_of("fig_interpreter")
    assert manifest["products"]["fig_interpreter"] == files
    assert all((out / f).is_file() for f in files)
    assert not [k for k in manifest if k.startswith("interpreter_")], "the old keys"
    got, pinned = manifest["interpreter"], manifest["inputs"]
    assert (got["shot"], got["year"], got["window"]) == (SHOT, 2024, [9, 5935])
    assert (got["pick_rule"], got["candidates"]) == (roster.NAMED, 153)
    assert (got["ae_version"], got["seg_version"]) == ("v2", "v3")
    assert manifest["seg_version"] == "v1", "the AE figures' SegNet, apart"
    model = chosen_model(tree.root / "models" / "ae_xpower" / "v2")
    assert got["ae_model"] == pinned["ae_model"], "read once, fig_examples' too"
    assert got["ae_model"]["path"] == str(model)
    assert got["seg_model"] == pinned["interpreter_seg_model"]
    assert got["seg_model"]["path"].endswith("/models/ae_seg/v3/model.pt")
    assert "seg_model" not in pinned, "no SegNet v1 in this tree"
    assert got["seg_band_khz"] == list(_shot(tree).seg_band)
    assert got["tables_version"] == frames.VERSION
    assert got["tables"] == {
        method: None if category == NTM else pinned[roster.table_key(method)]
        for category, method in roster.TABLES.items()
    }
    assert got["stores"] == {
        event: pinned.get(roster.store_key(event, SHOT))
        for event in (NTM, ELM, HMODE, SAW)
    }
    assert got["stores"][HMODE] is None
    assert got["stores"][SAW] == {
        "path": str(stores[SAW]),
        "sha256": sha256_of(stores[SAW]),
    }
    assert pinned["cohort"]["path"] == str(tree.label_tables / "catalog" / "cohort.csv")
    assert (got["tier"], got["corpus"]) == ("suggestions", str(tree.corpus_file(SHOT)))
    assert manifest["partial"]["fig_interpreter"] == [
        {
            "reason": build.NO_TRACK_TABLE,
            "phenomenon": NTM,
            "missing": [str(roster.table_file(tree, NTM))],
        },
        {
            "reason": build.NO_PANEL_STORE.format("no NBI power data"),
            "panel": "NBI power",
            "missing": [str(tree.spectrogram_file(HMODE, SHOT))],
        },
        {
            "reason": build.NO_PANEL_ROW.format("no SXR data"),
            "panel": "SXR",
            "store": str(stores[SAW]),
        },
    ]
    assert manifest["consistent"] is True
    # The roster CLI's preview records the same, less its commit.
    preview = tmp_path / "preview"
    assert (
        roster.main(["--out", str(preview), "--version", "v2", "--shot", str(SHOT)])
        == 0
    )
    record = json.loads((preview / roster.MANIFEST).read_text())
    assert got == {k: v for k, v in record.items() if k != "git"}
    assert sorted(p.name for p in preview.iterdir()) == [
        "fig_interpreter.pdf",
        "fig_interpreter.png",
        roster.MANIFEST,
    ]


def test_the_build_picks_the_roster_shot_or_skips_one_off_the_list(tree, tmp_path):
    manifest = build.build(tree, tmp_path / "a", version="v2")
    got = manifest["interpreter"]
    assert (got["shot"], got["pick_rule"]) == (SHOT, roster.PICK_RULE), (
        "ELMs and sawteeth beat OTHER's H-mode alone"
    )
    assert "fig_interpreter" in manifest["products"]
    for shot in (BLIND, NO_CO2):
        manifest = build.build(tree, tmp_path / "b", version="v2", shot=shot)
        assert manifest["interpreter"] is None
        assert "fig_interpreter" not in manifest["products"]
        assert manifest["skipped"]["fig_interpreter"] == {
            "reason": f"{shot}: not a non-blind roster shot with corpus CO2",
            "missing": [],
        }


def _opened(monkeypatch) -> Counter:
    """Every file opened for reading, by resolved path, through Python, pathlib,
    pandas, torch and h5py, as test_paper_build's spy counts them."""
    opened: Counter = Counter()

    def wrap(real):
        def spy(file, mode="r", *args, **kwargs):
            if isinstance(file, str | os.PathLike) and not set(mode) & set("wax+"):
                opened[str(Path(file).resolve())] += 1
            return real(file, mode, *args, **kwargs)

        return spy

    monkeypatch.setattr(builtins, "open", wrap(builtins.open))
    monkeypatch.setattr(io, "open", wrap(io.open))
    monkeypatch.setattr(h5py, "File", wrap(h5py.File))
    return opened


def test_the_build_pins_every_file_the_figure_reads_but_the_corpus(
    tree, tmp_path, monkeypatch
):
    _stores(tree)
    opened = _opened(monkeypatch)
    out = tmp_path / "paper"
    manifest = build.build(tree, out, shot=SHOT, version="v2")
    assert "fig_interpreter" in manifest["products"]
    under = str(tmp_path.resolve())
    read = {p for p in opened if p.startswith(under) and not p.startswith(str(out))}
    pinned = {str(Path(v["path"]).resolve()) for v in manifest["inputs"].values()}
    corpus = str(tree.corpus_file(SHOT).resolve())
    assert corpus in read and corpus not in pinned, "read in place, by path alone"
    assert read - {corpus} <= pinned, sorted(read - {corpus} - pinned)
    twice = {p: n for p, n in opened.items() if p in pinned and n != 2}
    assert not twice, f"each input read once to draw, once to check: {twice}"
    stores = {roster.store_key(e, SHOT) for e in (NTM, ELM, SAW)}
    assert stores <= set(manifest["inputs"]), "each store through the snapshot"
    assert manifest["consistent"] is True
