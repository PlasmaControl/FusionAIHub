"""The frame models' example figures (`fig_examples_<phenomenon>`, X1) and the
gallery drawing they share."""

from __future__ import annotations

import io
import json
import math

import numpy as np
import pytest
from matplotlib.colors import to_hex
from matplotlib.figure import Figure

from labeler import frames
from labeler.frames import evaluate, gallery
from labeler.frames import train as frames_train
from labeler.frames.targets import ABSENT, PRESENT_T, UNCERTAIN_T, UNKNOWN
from labeler.paper import build, frame_examples, roc, shots
from labeler.paper.snapshot import Snapshot

from .test_round3_frames_apply import elm, tree  # noqa: F401  (fixtures)
from .test_round3_frames_evaluate import TEST

METHOD = "elm_frames"


@pytest.fixture
def scored(elm):  # noqa: F811
    """The ELM model's tree: its split, features, model and evaluation."""
    return elm


NAME = "fig_examples_elm"


def test_the_products_are_named_by_phenomenon_and_owned():
    names = {m: frame_examples.figure_name(m) for m in roc.SELECTED.values()}
    assert [names[m] for m in roc.SELECTED.values() if m != roc.AE_METHOD] == [
        "fig_examples_ntm",
        "fig_examples_hmode",
        "fig_examples_elm",
        "fig_examples_sawtooth",
    ]
    for name in names.values():
        if name != names[roc.AE_METHOD]:
            assert build.PRODUCTS[name] == build.FIGURE
            assert {name + ".pdf", name + ".png"} <= build.OWNED


def test_only_shots_with_a_present_bin_are_ranked(scored):
    data, _ = frame_examples.read_features(scored, METHOD, TEST)
    found = frame_examples.present_shots(data)
    for shot, blob in data.items():
        with np.load(io.BytesIO(blob)) as z:
            assert (shot in found) == bool((z["states"] == PRESENT_T).any())
    assert "present bin" in frame_examples.RULE


def test_the_picks_are_pick_examples_over_the_f1_without_nan():
    f1 = {1: 0.9, 2: math.nan, 3: 0.5, 4: 0.1, 5: 0.7}
    assert frame_examples.pick(f1) == shots.pick_examples(f1)
    assert frame_examples.pick(f1) == [1, 3, 4]
    assert frame_examples.pick({7: 0.2, 8: math.nan}) == [7]


def _frame_shot(k: int) -> frame_examples.FrameShot:
    spec = frames.SPECS[METHOD]
    n_frames = 60  # 5 ms frames: 300 ms, 6 bins of 50 ms
    per = round(spec.bin_ms / 10)
    x = np.linspace(0, 1, 5 * n_frames * 3, dtype=np.float32).reshape(-1, n_frames * 5)
    channels = sum(frames.features._width(r) for r in spec.roles)
    x = np.tile(x[:1], (channels, 1))
    bins = np.arange(n_frames // per) * spec.bin_ms
    states = np.array([ABSENT, PRESENT_T, UNCERTAIN_T, UNKNOWN, ABSENT, ABSENT])
    prob = np.repeat([0.1, 0.9, 0.5, np.nan, 0.2, 0.05], per)
    return frame_examples.FrameShot(
        190000 + k, x, 0, bins, states, prob, 0.6, [1.0, 0.5, 0.0][k]
    )


def test_each_figure_draws_three_shots_with_input_target_and_p(tmp_path):
    spec = frames.SPECS[METHOD]
    fig = frame_examples.draw_examples(
        spec, [_frame_shot(k) for k in range(3)], tmp_path / "f"
    )
    assert (tmp_path / "f.pdf").is_file() and (tmp_path / "f.png").is_file()
    axes = [ax for ax in fig.axes]
    assert len(axes) == 9
    for k in range(3):
        rows, strip, model = axes[3 * k : 3 * k + 3]
        assert len(rows.images) == 1, "the input rows"
        assert (
            rows.get_title()
            == f"shot {190000 + k} (test): F1 (50 ms bins) "
            + (["1.00", "0.50", "0.00"][k])
        )
        assert len(strip.patches) >= 3, "the target runs, absent blank"
        assert model.get_ylim() == (0, 1)
        [threshold] = [ln for ln in model.lines if ln.get_linestyle() == ":"]
        assert list(threshold.get_ydata()) == [0.6, 0.6], "the model's threshold"
    strip = axes[1]
    [unknown] = [p for p in strip.patches if p.get_label() == "target: unknown"]
    assert to_hex(unknown.get_facecolor()) == "#9a9a9a", "not-observable grey"
    assert unknown.get_linewidth() == 0, "no outline"
    assert not [p for p in strip.patches if "absent" in p.get_label()], "blank"
    [legend] = fig.legends
    labels = [t.get_text() for t in legend.get_texts()]
    assert labels == [
        "target: present",
        "target: uncertain",
        "target: unknown",
        frame_examples.MODEL_LABEL,
        frame_examples.THRESHOLD_LABEL,
    ]


def test_an_optional_role_s_flag_channel_is_named():
    spec = frames.SPECS["hmode_frames"]
    optional = [r for r in spec.roles if r.optional]
    assert optional, "H-mode has an optional role"
    fig = Figure()
    ax = fig.subplots()
    width = sum(frames.features._width(r) for r in spec.roles)
    gallery.draw_rows(ax, spec, np.zeros((width, 10)), 0, 50)
    names = [t.get_text() for t in ax.get_yticklabels()]
    assert names == [
        n
        for r in spec.roles
        for n in ([r.name, f"{r.name} seen"] if r.optional else [r.name])
    ]


def test_roles_of_one_name_share_a_tick():
    spec = frames.SPECS["sawtooth_frames"]
    fig = Figure()
    ax = fig.subplots()
    width = sum(frames.features._width(r) for r in spec.roles)
    gallery.draw_rows(ax, spec, np.zeros((width, 10)), 0, 50)
    names = [t.get_text() for t in ax.get_yticklabels()]
    assert names.count("ece") == 1 and names[-1].endswith("seen")
    edges = np.cumsum([0, *[frames.features._width(r) for r in spec.roles]])
    ece = [i for i, r in enumerate(spec.roles) if r.name == "ece"]
    centre = (edges[ece[0]] + edges[ece[-1] + 1]) / 2
    assert centre in list(ax.get_yticks())
    assert len(ax.lines) == len(spec.roles) - 1, "a white line between blocks"


def test_the_gallery_draws_the_same_artists(tmp_path, monkeypatch):
    """`gallery.draw`, now built of `draw_rows`, `draw_target` and `draw_prob`,
    draws what it drew: the rows' image, one span per target run, the shaded
    frames, P, the dotted threshold and the four named keys."""
    spec = frames.SPECS[METHOD]
    s = _frame_shot(0)
    drawn = []
    monkeypatch.setattr(
        Figure,
        "savefig",
        lambda fig, path, **kw: (drawn.append(fig), path.write_bytes(b"")),
    )
    gallery.draw(
        tmp_path / "g.jpg",
        title="t",
        spec=spec,
        x=s.x,
        first=0,
        prob=s.prob,
        threshold=0.6,
        target=(s.bins, s.states),
    )
    [fig] = drawn
    rows, strip, model = fig.axes
    assert rows.images[0].get_extent() == [0, 300, s.x.shape[0], 0]
    assert rows.get_ylabel() == "features" and strip.get_ylabel() == "target"
    assert len(strip.patches) == 5, "one span per run of a state"
    assert len(model.patches) == 1, "the run of frames at or above 0.6"
    np.testing.assert_array_equal(model.lines[0].get_ydata(), s.prob)
    assert list(model.lines[1].get_ydata()) == [0.6, 0.6]
    assert [t.get_text() for t in fig.legends[0].get_texts()] == [
        "present",
        "absent",
        "uncertain",
        "unknown",
    ]
    assert fig._suptitle.get_text() == "t"
    assert model.get_xlabel() == "time (ms)" and model.get_xlim() == (0, 300)


def _read(paths, method, evaluation_only: bool = False):
    found = build.inputs(paths, "v2")
    snap = Snapshot(paths.root / "snap")
    evaluation = json.loads(found[build.evaluation_key(method)].read_bytes())
    return found, snap, evaluation


def _call(paths, tmp_path, evaluation, snap, found, examples=3):
    made, partial = {}, {}

    def figure(name, draw, *args):
        draw(*args, tmp_path / name)
        made[name] = build.files_of(name)

    def lacking(products, reason, **what):
        for product in products:
            partial.setdefault(product, []).append({"reason": reason, **what})

    why, record = build._frame_examples(
        paths, found, snap, METHOD, NAME, evaluation, examples, figure, lacking
    )
    return why, record, made, partial


def test_the_build_draws_the_frame_figure_and_records_the_picks(
    scored,
    tmp_path,
):
    found, snap, evaluation = _read(scored, METHOD)
    why, record, made, partial = _call(scored, tmp_path, evaluation, snap, found)
    assert why is None and made == {NAME: [NAME + ".pdf", NAME + ".png"]}
    data, _ = frame_examples.read_features(scored, METHOD, TEST)
    present = sorted(frame_examples.present_shots(data))
    assert present and sorted(record["example_shots"]) == present, "with a present bin"
    assert record["example_rule"] == frame_examples.RULE
    assert record["test_shots"] == 2 and record["features_missing"] == {}
    assert record["threshold"] == 0.5
    assert partial == {
        NAME: [
            {
                "reason": build.FRAME_FEW.format(n=3),
                "shots": record["example_shots"],
            }
        ]
    }, "fewer than 3 shots is partial"
    # each F1 is the evaluation's own over the scored bins
    read, _ = evaluate.read_shots(scored, METHOD, TEST)
    model, _ = frames_train.load(frames.model_dir(scored, METHOD) / "model.pt")
    for s in (s for s in read if s.shot in record["example_shots"]):
        prob = evaluate.model_probs(model, frames.SPECS[METHOD], s)
        f1 = evaluate.score(prob, s.states, 0.5, s.observed)["f1"]
        assert record["shot_f1"][str(s.shot)] == build._number(
            math.nan if f1 is None else f1
        )
    # the drawn shots' features are pinned as inputs
    for s in record["example_shots"]:
        path, sha = snap.pinned[f"frames_features_{METHOD}_{s}"]
        assert path == frames.features_dir(scored, METHOD) / f"{s}.npz"
        assert sha == record["shot_features_sha256"][str(s)]
    for what in ("model", "split"):
        assert build.frame_example_key(what, METHOD) in snap.pinned, what


def test_the_threshold_is_the_blobs(scored, tmp_path, monkeypatch):
    found, snap, evaluation = _read(scored, METHOD)
    load = frames_train.load

    def loaded(source):
        model, blob = load(source)
        return model, blob | {"threshold": 0.123}

    monkeypatch.setattr(frames_train, "load", loaded)
    why, record, _, partial = _call(scored, tmp_path, evaluation, snap, found)
    assert why is None and record["threshold"] == 0.123
    reasons = [e["reason"] for e in partial[NAME]]
    assert build.FRAME_THRESHOLD in reasons, "the evaluation scored another"


def test_the_figure_is_skipped_without_its_model_or_evaluation(
    scored,
    tmp_path,
):
    found, snap, evaluation = _read(scored, METHOD)
    model = found[build.frame_example_key("model", METHOD)]
    model.rename(model.with_name("gone.pt"))
    why, record, made, _ = _call(scored, tmp_path, evaluation, snap, found)
    assert record is None and made == {}
    assert why == {"reason": build.MISSING, "missing": [str(model)]}
    model.with_name("gone.pt").rename(model)
    why, record, *_ = _call(scored, tmp_path, None, snap, found)
    assert why["reason"] == build.NO_FRAME_EVALUATION
    other = dict(evaluation, model=dict(evaluation["model"], sha256="0" * 64))
    snap = Snapshot(scored.root / "snap2")
    why, record, *_ = _call(scored, tmp_path, other, snap, found)
    assert why == {"reason": build.FRAME_OTHER.format(what="model"), "missing": []}


def test_the_figure_is_skipped_with_no_features(scored, tmp_path):
    found, snap, evaluation = _read(scored, METHOD)
    for path in frames.features_dir(scored, METHOD).glob("*.npz"):
        path.unlink()
    why, record, made, _ = _call(scored, tmp_path, evaluation, snap, found)
    assert record is None and made == {}
    assert why["reason"] == build.NO_FRAME_SHOT
