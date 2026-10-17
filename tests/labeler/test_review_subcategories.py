"""Subcategories: a label inside its parent's label; the taxonomy, I-mode, markers."""

from __future__ import annotations

import json

import h5py
import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler.ae.seg import whole
from labeler.config import Paths, store_event
from labeler.events import panels, taxonomy
from labeler.events.review import build as review_build
from labeler.events.review import labels
from labeler.events.review import rows as review_rows
from labeler.events.review.rows import Grid, ImageRow, TraceRow
from labeler.events.ui.app import COOKIE, create_app

ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,,,\n"
    "170816,unverified,false,,,\n"
    "170817,unverified,false,,,\n"
)
TAXONOMY = """\
taxonomy:
  mhd_n1:
    display: "MHD n=1"
    group: "MHD mode"
    subcategories:
      m1: {display: "1/1 (sawtooth, fishbone)", m: 1, n: 1}
      m2: {display: "2/1 (tearing)", m: 2, n: 1}
  mhd_n2:
    display: "MHD n=2"
    group: "MHD mode"
    subcategories:
      m3: {display: "3/2 (tearing)", m: 3, n: 2}
  mhd_n3:
    display: "MHD n=3"
    group: "MHD mode"
    subcategories:
      m4: {display: "4/3 (tearing)", m: 4, n: 3}
  alfven_eigenmode:
    display: "Alfvén eigenmode (AE)"
    subcategories:
      tae: {display: "TAE"}
      rsae: {display: "RSAE"}
  confinement: {display: "Confinement regime"}
"""
EVENTS = (
    "mhd_n1", "mhd_n2", "mhd_n3", "alfven_eigenmode", "confinement", "disruption",
    "sawtooth_oscillation", "fishbone", "neoclassical_tearing_mode",
    "improved_energy_confinement_mode",
)
SHOT = 170815


@pytest.fixture(autouse=True)
def _no_builder_for_m1(monkeypatch):
    """`mhd_n1.m1` is its parent's store until a builder is registered for it, and the
    panel work that registers one runs beside these tests: they decide it themselves."""
    monkeypatch.delitem(panels.BUILDERS, "mhd_n1.m1", raising=False)
    monkeypatch.delitem(review_build.BUILDERS, "mhd_n1.m1", raising=False)


@pytest.fixture
def tables(tmp_path):
    root = tmp_path / "events"
    root.mkdir()
    (root / "events.yaml").write_text(TAXONOMY)
    for event in EVENTS:
        (root / event).mkdir()
        (root / event / "shots.csv").write_text(ROSTER)
    return root


@pytest.fixture
def paths(tables, tmp_path):
    return Paths(
        root=tmp_path / "root", corpus=tmp_path / "corpus",
        text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=tables, config_dir=tables, raw_cache=tmp_path / "raw",
    )


@pytest.fixture
def client(paths):
    transport = TestClient(create_app(paths=paths, token="secret"))
    transport.cookies.set(COOKIE, "secret")
    return transport


def _store(paths, event, shot=SHOT):
    """A built store for `event` (the store event: a subcategory shares its parent's)."""
    values = np.zeros((2, 1, 2000), dtype=np.float32)
    path = paths.spectrogram_file(event, shot)
    version = review_build.PANEL_VERSIONS.get(event)
    revision = review_build.PANEL_REVISIONS.get(event)
    review_rows.write(
        path, Grid(0.0, 1.0, 2000),
        [ImageRow("im", "Image", np.zeros((4, 2000), "uint8"), y0=0.0, dy=1.0,
                  y_units="kHz", z_lo=0.0, z_hi=1.0, z_units="dB"),
         TraceRow("p1", "Trace", values)],
        event=event, **({} if version is None else {"panel_version": version}),
        **({} if revision is None else {"panel_revision": revision}),
    )
    return path


def _post(client, event, intervals, iscrowd=None, shot=SHOT, window=(0, 2000)):
    body = {"event": event, "shot": shot, "window": list(window),
            "intervals": intervals, "overlap_edit": True}
    if iscrowd is not None:
        body["iscrowd"] = iscrowd
    return client.post("/api/label", json=body)


def _parent(client, intervals=((200, 400, 1), (800, 1000, 2)), iscrowd=(0, 1),
            event="mhd_n1", shot=SHOT):
    saved = _post(client, event, [list(span) for span in intervals],
                  list(iscrowd) if iscrowd is not None else None, shot=shot)
    assert saved.status_code == 200, saved.text
    return saved.json()["saved"]


def _child_dir(paths, child):
    return taxonomy.load(paths.events_manifest).child_dir(paths.label_tables, child)


# ---- the taxonomy ----


def test_the_taxonomy_names_children_groups_and_directories(tables):
    tree = taxonomy.load(tables / "events.yaml")
    assert tree.children("mhd_n1") == ["mhd_n1.m1", "mhd_n1.m2"]
    assert tree.parent("mhd_n1.m2") == "mhd_n1" and tree.parent("mhd_n1") is None
    assert tree.is_child("alfven_eigenmode.tae") and not tree.is_child("mhd_n1")
    assert tree.display("mhd_n1") == "MHD n=1"
    assert tree.display("mhd_n1.m1") == "1/1 (sawtooth, fishbone)"
    assert tree.display("disruption") == "disruption"
    assert tree.group("mhd_n3") == "MHD mode" and tree.group("mhd_n3.m4") == "MHD mode"
    assert tree.group("alfven_eigenmode") is None
    assert tree.child_dir(tables, "alfven_eigenmode.rsae") == (
        tables / "alfven_eigenmode" / "subcategories" / "rsae"
    )
    with pytest.raises(KeyError):
        tree.child_dir(tables, "mhd_n1")


def test_the_taxonomy_is_reread_when_the_file_changes_and_refuses_a_bad_entry(tables):
    assert taxonomy.load(tables / "events.yaml") is taxonomy.load(tables / "events.yaml")
    manifest = tables / "events.yaml"
    manifest.write_text(TAXONOMY + "  fishbone: {display: Fishbone}\n")
    assert taxonomy.load(tables / "events.yaml").display("fishbone") == "Fishbone"
    manifest.write_text("taxonomy:\n  mhd_n1:\n    subcategories:\n      '../x': {}\n")
    with pytest.raises(taxonomy.TaxonomyError, match="not a plain name"):
        taxonomy.load(tables / "events.yaml")
    manifest.write_text("taxonomy:\n  disruption: {review: 'no'}\n")
    with pytest.raises(taxonomy.TaxonomyError, match="review must be true or false"):
        taxonomy.load(tables / "events.yaml")
    manifest.unlink()
    assert taxonomy.load(tables / "events.yaml").children("mhd_n1") == []


def test_the_shipped_taxonomy_lists_the_mhd_modes_and_ae_children():
    tree = taxonomy.load(Paths().events_manifest)
    assert tree.children("mhd_n1") == ["mhd_n1.m1", "mhd_n1.m2", "mhd_n1.m3"]
    assert tree.children("mhd_n2") == ["mhd_n2.m3"]
    assert tree.children("mhd_n3") == ["mhd_n3.m4"]
    assert tree.children("alfven_eigenmode") == [
        "alfven_eigenmode.tae", "alfven_eigenmode.rsae",
    ]
    assert tree.group("mhd_n2") == "MHD mode"
    assert not tree.reviewed("disruption") and not tree.reviewed("vertical_displacement_event")
    assert tree.reviewed("locked_mode") and tree.reviewed("mhd_n1.m1")


# ---- a child is labelled inside its parent ----


def test_a_child_save_is_cut_to_the_parents_spans_in_both_lanes(client, paths):
    _store(paths, "mhd_n1")
    parent = _parent(client, intervals=((204, 397, 1), (803, 996, 2)))
    assert parent["intervals"] == [[200, 400, 1], [800, 1000, 2]]
    response = _post(
        client, "mhd_n1.m1",
        [[101, 304, 1], [353, 904, 1], [957, 1200, 2]], [0, 1, 0],
        window=(100, 1100),
    )
    assert response.status_code == 200, response.text
    clipped = {
        "window": [0, 2000],
        "intervals": [[200, 300, 1], [350, 400, 1], [800, 900, 1], [960, 1000, 2]],
        "iscrowd": [0, 1, 1, 0],
    }
    assert response.json()["saved"] == clipped, "spans outside the parent vanish"
    assert response.json()["row"]["state"] == "changed"
    opened = client.get(f"/api/shot?event=mhd_n1.m1&shot={SHOT}").json()
    assert opened["saved"] == clipped
    assert opened["allowed"] == [[200, 400], [800, 1000]]
    assert opened["parent"] == {
        "event": "mhd_n1", "display_name": "MHD n=1", "label": parent,
    }
    assert opened["categories"] == {"1": "present", "2": "uncertain"}
    directory = _child_dir(paths, "mhd_n1.m1")
    assert directory == paths.label_tables / "mhd_n1" / "subcategories" / "m1"
    assert labels.read_saved(directory)[SHOT].as_json() == clipped
    entry = labels.read_history(directory)[-1]
    assert entry["parent"] == "mhd_n1" and len(entry["parent_sha256"]) == 64
    assert entry["parent_saved_at"] == labels.read_history(
        paths.label_tables / "mhd_n1")[-1]["saved_at"]
    assert labels.read_saved(paths.label_tables / "mhd_n1")[SHOT].as_json() == parent, (
        "the parent's own label is untouched")


def test_a_child_is_409_until_the_parent_is_saved(client, paths):
    _store(paths, "mhd_n1")
    for response in (
        client.get(f"/api/shot?event=mhd_n1.m1&shot={SHOT}"),
        _post(client, "mhd_n1.m1", [[100, 300, 1]]),
    ):
        assert response.status_code == 409
        assert response.json()["error"] == "review MHD n=1 first"
    assert not _child_dir(paths, "mhd_n1.m1").exists(), "nothing was written"
    _parent(client)
    assert client.get(f"/api/shot?event=mhd_n1.m1&shot={SHOT}").status_code == 200


def test_a_smaller_parent_hides_child_spans_and_a_later_save_cannot_bring_them_back(
    client, paths,
):
    _store(paths, "mhd_n1")
    _parent(client)
    assert _post(client, "mhd_n1.m1", [[300, 380, 1], [850, 950, 1]]).status_code == 200
    directory = _child_dir(paths, "mhd_n1.m1")
    _parent(client, intervals=[(200, 400, 1)], iscrowd=[0])
    opened = client.get(f"/api/shot?event=mhd_n1.m1&shot={SHOT}").json()
    assert opened["allowed"] == [[200, 400]]
    assert opened["saved"]["intervals"] == [[300, 380, 1]], "hidden on read"
    assert (850, 950, 1) in labels.read_saved(directory)[SHOT].intervals, "file kept"
    # A client that still draws the old span, or the parent grown back, resurrects nothing.
    stale = _post(client, "mhd_n1.m1", [[300, 380, 1], [850, 950, 1]])
    assert stale.json()["saved"]["intervals"] == [[300, 380, 1]]
    assert labels.read_saved(directory)[SHOT].intervals == ((300, 380, 1),)
    _parent(client)
    regrown = client.get(f"/api/shot?event=mhd_n1.m1&shot={SHOT}").json()
    assert regrown["saved"]["intervals"] == [[300, 380, 1]]


def test_the_child_queue_is_the_parents_saved_non_empty_shots(client, paths):
    _parent(client, shot=SHOT)
    absent = _post(client, "mhd_n1", [], shot=170816)
    assert absent.status_code == 200 and absent.json()["saved"]["intervals"] == []
    queue = client.get("/api/queue?event=mhd_n1.m1").json()
    assert {key: queue[key] for key in ("shots", "resume", "list")} == {
        "shots": [{"shot": SHOT, "tier": "gold", "state": "unreviewed",
                   "saved_at": None}],
        "resume": SHOT, "list": "roster",
    }, "unsaved 170817 and all-absent 170816 are not offered, and nothing crashes"
    assert [row["shot"] for row in client.get("/api/queue?event=mhd_n1").json()["shots"]
            ] == [SHOT, 170816, 170817]
    _store(paths, "mhd_n1", 170816)
    assert client.get("/api/shot?event=mhd_n1.m1&shot=170816").json()["allowed"] == []
    assert _post(client, "mhd_n1.m1", [[100, 300, 1]]).status_code == 200
    row = client.get("/api/queue?event=mhd_n1.m1").json()["shots"][0]
    assert row["state"] == "changed" and row["saved_at"]


def test_two_children_overlap_freely_on_the_same_shot(client, paths):
    _store(paths, "mhd_n1")
    _parent(client, intervals=[(200, 800, 1)], iscrowd=None)
    first = _post(client, "mhd_n1.m1", [[300, 500, 1]])
    second = _post(client, "mhd_n1.m2", [[400, 700, 2]])
    assert first.status_code == second.status_code == 200
    for child, expected in (("mhd_n1.m1", [[300, 500, 1]]), ("mhd_n1.m2", [[400, 700, 2]])):
        saved = client.get(f"/api/shot?event={child}&shot={SHOT}").json()["saved"]
        assert saved["intervals"] == expected
    events = {row["event"]: row for row in client.get("/api/events").json()["events"]}
    assert events["mhd_n1"]["subcategories"] == [
        {"event": "mhd_n1.m1", "display_name": "1/1 (sawtooth, fishbone)",
         "n_reviewed": 1},
        {"event": "mhd_n1.m2", "display_name": "2/1 (tearing)", "n_reviewed": 1},
    ]
    assert events["mhd_n1"]["n_reviewed"] == 1, "the parent's count is its own"


def test_an_ae_child_is_cut_to_ae_and_reads_the_parents_masks_read_only(client, paths):
    _store(paths, "alfven_eigenmode")
    _parent(client, intervals=[(200, 400, 1)], iscrowd=None, event="alfven_eigenmode")
    response = _post(client, "alfven_eigenmode.rsae", [[100, 300, 1]])
    assert response.json()["saved"]["intervals"] == [[200, 300, 1]]
    opened = client.get(f"/api/shot?event=alfven_eigenmode.rsae&shot={SHOT}").json()
    assert opened["parent"]["display_name"] == "Alfvén eigenmode (AE)"
    assert [r["name"] for r in opened["rows"]] == ["im", "p1"]
    assert client.get(f"/api/masks?event=alfven_eigenmode.rsae&shot={SHOT}"
                      ).status_code == 404, "no pseudo-mask built for this shot"
    mask = whole.review_file(paths, SHOT)
    mask.parent.mkdir(parents=True, exist_ok=True)
    mask.write_bytes(b"")
    refused = client.post("/api/masks", json={
        "event": "alfven_eigenmode.rsae", "shot": SHOT, "pseudo_sha256": "0" * 64,
        "revision": 0, "rejected": [],
    })
    assert refused.status_code == 404
    assert "reject regions on alfven_eigenmode" in refused.json()["error"]


def test_only_a_name_in_the_taxonomy_is_a_child(client, paths):
    _store(paths, "mhd_n1")
    _parent(client)
    for event in ("mhd_n1.m9", "mhd_n1.", ".m1", "mhd_n1.m1.m1", "mhd_n1.m1/..",
                  "mhd_n1/subcategories/m1", "../mhd_n1.m1", "disruption.m1"):
        for route in ("queue", "shot"):
            response = client.get(f"/api/{route}", params={"event": event, "shot": SHOT})
            assert response.status_code == 404, (event, response.text)
    assert _post(client, "mhd_n1.m9", [[100, 300, 1]]).status_code == 404
    off_roster = client.get("/api/shot?event=mhd_n1.m1&shot=999999")
    assert off_roster.status_code == 404
    assert _post(client, "mhd_n1.m1", [[100, 300, 7]]).status_code == 400


# ---- the event list, folding ----


def test_events_lists_the_top_level_events_with_groups_and_subcategories(client):
    events = {row["event"]: row for row in client.get("/api/events").json()["events"]}
    assert sorted(events) == [
        "alfven_eigenmode", "confinement", "disruption", "mhd_n1", "mhd_n2", "mhd_n3",
    ], "no child, and none of the folded events"
    for name, child in (("mhd_n1", "mhd_n1.m1"), ("mhd_n2", "mhd_n2.m3"),
                        ("mhd_n3", "mhd_n3.m4")):
        assert events[name]["group"] == "MHD mode"
        assert child in [sub["event"] for sub in events[name]["subcategories"]]
    assert events["mhd_n1"]["display_name"] == "MHD n=1"
    assert events["mhd_n3"]["subcategories"] == [
        {"event": "mhd_n3.m4", "display_name": "4/3 (tearing)", "n_reviewed": 0},
    ]
    assert events["alfven_eigenmode"]["group"] is None
    assert [sub["display_name"] for sub in events["alfven_eigenmode"]["subcategories"]
            ] == ["TAE", "RSAE"]
    assert events["disruption"] == {
        "event": "disruption", "display_name": "disruption", "group": None,
        "n_shots": 3, "n_reviewed": 0,
        "categories": {"1": "present", "2": "uncertain"}, "subcategories": [],
    }
    assert events["mhd_n1"]["categories"] == {"1": "present", "2": "uncertain"}


def test_an_event_marked_review_false_is_not_offered(client, paths):
    manifest = paths.events_manifest
    manifest.write_text(TAXONOMY + "  disruption: {display: Disruption, review: false}\n")
    events = [row["event"] for row in client.get("/api/events").json()["events"]]
    assert "disruption" not in events and "mhd_n1" in events


@pytest.mark.parametrize("folded,target", [
    ("sawtooth_oscillation", "mhd_n1"), ("fishbone", "mhd_n1"),
    ("neoclassical_tearing_mode", "mhd_n1"),
    ("improved_energy_confinement_mode", "confinement"),
])
def test_a_folded_event_is_refused_naming_the_event_it_is_reviewed_as(
    client, folded, target,
):
    for route in ("queue", "shot", "history"):
        response = client.get(f"/api/{route}", params={"event": folded, "shot": SHOT})
        assert response.status_code == 404
        assert f"reviewed as {target}" in response.json()["error"]
    assert _post(client, folded, [[100, 300, 1]]).status_code == 404


# ---- I-mode ----


def test_confinement_offers_imode_as_category_6(client, paths):
    events = {row["event"]: row for row in client.get("/api/events").json()["events"]}
    assert events["confinement"]["categories"] == {
        "1": "high", "2": "low", "3": "qh", "4": "wpqh", "5": "uncertain",
        "6": "imode",
    }
    saved = _post(client, "confinement", [[200, 400, 6], [500, 600, 1]])
    assert saved.status_code == 200, saved.text
    assert saved.json()["saved"]["intervals"] == [[200, 400, 6], [500, 600, 1]]
    assert _post(client, "confinement", [[200, 400, 7]]).status_code == 400
    assert labels.categories("confinement")[6] == "imode"
    assert _store(paths, "high_confinement_mode").is_file()
    opened = client.get(f"/api/shot?event=confinement&shot={SHOT}")
    assert opened.status_code == 200 and opened.json()["saved"]["intervals"][0][2] == 6


# ---- which store a event reads ----


def test_a_subcategory_shares_its_parents_store_unless_it_has_its_own_builder(
    paths, monkeypatch,
):
    assert store_event("mhd_n2") == store_event("mhd_n3") == "mhd_n1"
    assert store_event("mhd_n1.m2") == "mhd_n1"
    assert store_event("mhd_n1.m1") == "mhd_n1"
    assert store_event("alfven_eigenmode.tae") == "alfven_eigenmode"
    assert store_event("confinement") == "high_confinement_mode"
    assert store_event("disruption") == "disruption"
    assert paths.spectrogram_file("mhd_n1.m2", 5).parts[-2:] == ("mhd_n1", "5.h5")
    monkeypatch.setitem(panels.BUILDERS, "mhd_n1.m1", object())
    assert store_event("mhd_n1.m1") == "mhd_n1.m1"
    assert store_event("mhd_n1.m2") == "mhd_n1"
    assert paths.spectrogram_file("mhd_n1.m1", 5).parts[-2:] == ("mhd_n1.m1", "5.h5")


def test_a_review_builder_makes_a_subcategory_its_own_store_too(monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "alfven_eigenmode.tae", _fake_builder)
    assert store_event("alfven_eigenmode.tae") == "alfven_eigenmode.tae"
    assert store_event("alfven_eigenmode.rsae") == "alfven_eigenmode"


def test_a_missing_panels_registry_is_not_an_error(monkeypatch):
    monkeypatch.delattr(panels, "BUILDERS")
    monkeypatch.delitem(review_build.BUILDERS, "mhd_n1.m1", raising=False)
    assert store_event("mhd_n1.m1") == "mhd_n1"


def test_a_child_with_its_own_recipe_is_stale_by_its_own_version(
    paths, monkeypatch,
):
    monkeypatch.setitem(panels.BUILDERS, "mhd_n1.m1", object())
    path = paths.spectrogram_file("mhd_n1.m1", SHOT)
    assert not review_build.current(path, "mhd_n1.m1")
    values = np.zeros((2, 1, 10), dtype=np.float32)
    for version, expected in ((None, False), (0, False), (1, True)):
        review_rows.write(path, Grid(0.0, 1.0, 10), [TraceRow("p", "T", values)],
                          event="mhd_n1.m1",
                          **({} if version is None else {"panel_version": version}))
        assert bool(review_build.current(path, "mhd_n1.m1")) is expected
    assert review_build.PANEL_VERSIONS["mhd_n1"] == 1
    assert review_build.PANEL_VERSIONS["mhd_n1.m1"] == 1
    assert review_build.PANEL_VERSIONS["edge_localized_mode"] == 4
    assert review_build.PANEL_VERSIONS["high_confinement_mode"] == 1
    assert review_build.PANEL_VERSIONS["detachment"] == 10
    assert review_build.PANEL_VERSIONS["resistive_wall_mode"] == 2
    assert review_build.PANEL_VERSIONS["sawtooth_oscillation"] == 1  # folded: untouched
    assert review_build.PANEL_REVISIONS == {"detachment": 1}


def test_confinement_stores_are_version_checked_as_the_h_mode_event(paths):
    path = paths.spectrogram_file("confinement", SHOT)
    assert path == paths.spectrogram_file("high_confinement_mode", SHOT)
    values = np.zeros((2, 1, 10), dtype=np.float32)
    review_rows.write(path, Grid(0.0, 1.0, 10), [TraceRow("p", "T", values)],
                      event="high_confinement_mode")
    assert not review_build.current(path, "confinement"), "built before version 1"
    review_rows.write(path, Grid(0.0, 1.0, 10), [TraceRow("p", "T", values)],
                      event="high_confinement_mode", panel_version=1)
    assert review_build.current(path, "confinement")


# ---- markers ----


def _fake_builder(event, shot, paths):
    values = np.zeros((2, 1, 2000), dtype=np.float32)
    return Grid(0.0, 1.0, 2000), [TraceRow("p1", "Trace", values)], {"params": {}}


def test_markers_travel_from_the_build_to_the_shot_view(client, paths, monkeypatch):
    calls = []

    def markers(shot, *, t_range, paths):
        calls.append((shot, t_range))
        return [{"label": "crash", "t_ms": [900.0, 300.5, float("nan")]},
                {"label": "onset", "t_ms": []}]

    monkeypatch.setitem(review_build.BUILDERS, "mhd_n1", _fake_builder)
    monkeypatch.setattr(panels, "MARKERS", {"mhd_n1": markers}, raising=False)
    _parent(client)
    # mhd_n2 is mhd_n1's store: one build, markers and all.
    built = review_build.build("mhd_n2", SHOT, paths)
    assert built == paths.spectrogram_file("mhd_n1", SHOT)
    assert calls == [(SHOT, (0.0, 2000.0))]
    expected = [{"label": "crash", "t_ms": [300.5, 900.0]},
                {"label": "onset", "t_ms": []}]
    with h5py.File(built, "r") as store:
        assert json.loads(store.attrs["markers"]) == expected
        assert store.attrs["panel_version"] == 1 and store.attrs["event"] == "mhd_n1"
    assert review_rows.meta(built)["markers"] == expected
    for event in ("mhd_n1", "mhd_n1.m1"):
        opened = client.get(f"/api/shot?event={event}&shot={SHOT}")
        assert opened.status_code == 200, opened.text
        assert opened.json()["markers"] == expected
    assert calls == [(SHOT, (0.0, 2000.0))], "the page never rebuilt it"


def test_a_store_without_markers_reads_back_none(client, paths, monkeypatch):
    monkeypatch.setitem(review_build.BUILDERS, "mhd_n1", _fake_builder)
    monkeypatch.delattr(panels, "MARKERS", raising=False)
    built = review_build.build("mhd_n1", SHOT, paths)
    with h5py.File(built, "r") as store:
        assert "markers" not in store.attrs
    _parent(client)
    assert client.get(f"/api/shot?event=mhd_n1&shot={SHOT}").json()["markers"] == []


def test_a_failing_markers_function_fails_the_build_instead_of_storing_none(
    paths, monkeypatch,
):
    def broken(shot, *, t_range, paths):
        raise RuntimeError("no crash times")

    monkeypatch.setitem(review_build.BUILDERS, "mhd_n1", _fake_builder)
    monkeypatch.setattr(panels, "MARKERS", {"mhd_n1": broken}, raising=False)
    with pytest.raises(RuntimeError, match="no crash times"):
        review_build.build("mhd_n1", SHOT, paths)
    assert not paths.spectrogram_file("mhd_n1", SHOT).exists()
