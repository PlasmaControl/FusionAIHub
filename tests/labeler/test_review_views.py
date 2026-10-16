"""Saved views (`review.views`): save, list, rename and delete, the etag conflicts, the
size and name limits, the presets, and that the store is never left half written."""

from __future__ import annotations

import json
import os
import threading

import pytest

from labeler.config import Paths
from labeler.events.review import views
from labeler.events.review.views import ViewError

EVENT = "mhd_n1"
LAYOUT = {"v": 1, "rows": [{"id": "cat:ip", "h": 80}]}
PRESETS = """
presets:
  mhd_n1:
    - {name: "q and Te", layout: {v: 1, rows: []}}
    - {name: broken}
"""


@pytest.fixture
def paths(tmp_path):
    config = tmp_path / "config"
    config.mkdir()
    (config / "views.yaml").write_text(PRESETS)
    return Paths(root=tmp_path / "root", corpus=tmp_path / "corpus",
                 text_root=tmp_path / "text", logs_jsonl=tmp_path / "logs.jsonl",
                 label_tables=tmp_path / "events", config_dir=config,
                 raw_cache=tmp_path / "raw")


def _post(paths, **body):
    return views.change(paths, {"event": EVENT, "reviewer": "Ana", **body})


def _save(paths, name="Mine", layout=LAYOUT, etag=None, **extra):
    return _post(paths, op="save", name=name, layout=layout, etag=etag, **extra)


def _etag(found, name):
    return next(v["etag"] for v in found if v["name"] == name)


def _refused(status, call, *args, **kwargs):
    with pytest.raises(ViewError) as caught:
        call(*args, **kwargs)
    assert caught.value.status == status
    return caught.value.message


def test_save_then_list_own_views_by_name_then_presets(paths):
    _save(paths, name="zeta")
    found = _save(paths, name="Alpha", layout={"v": 1, "rows": []})
    assert [v["name"] for v in found] == ["Alpha", "zeta", "q and Te"]
    assert found[2]["preset"] is True and "preset" not in found[0]
    assert found[1]["layout"] == LAYOUT and len(found[1]["etag"]) == 16
    assert views.list_views(paths, EVENT, " Ana ") == found
    assert views.list_views(paths, EVENT, None) == found[2:]  # no name: presets only
    assert views.list_views(paths, "ae", "Ana") == []
    stored = json.loads(views.store_path(paths).read_text())
    assert set(stored["Ana"][EVENT]["zeta"]) == {"layout", "etag", "saved_at"}


def test_saving_over_a_view_needs_its_etag(paths):
    first = _etag(_save(paths), "Mine")
    assert _refused(409, _save, paths) == views.CONFLICT  # no etag
    _refused(409, _save, paths, etag="0" * 16)  # another tab's
    second = _etag(_save(paths, layout={"v": 2}, etag=first), "Mine")
    assert second != first
    _refused(409, _save, paths, etag=first)  # the first tab is now stale
    _refused(409, _save, paths, name="New", etag=first)  # a new view has no etag


def test_rename_and_delete(paths):
    etag = _etag(_save(paths), "Mine")
    _save(paths, name="Other")
    _refused(409, _post, paths, op="rename", name="Mine", new_name="Other", etag=etag)
    _refused(409, _post, paths, op="rename", name="Mine", new_name="q and te",
             etag=etag)  # a preset's name, in any case
    _refused(409, _post, paths, op="rename", name="Mine", new_name="Kept", etag="x")
    found = _post(paths, op="rename", name="Mine", new_name="Kept", etag=etag)
    assert [v["name"] for v in found] == ["Kept", "Other", "q and Te"]
    assert _etag(found, "Kept") == etag
    _refused(404, _post, paths, op="delete", name="Mine", etag=etag)
    _refused(409, _post, paths, op="delete", name="Kept", etag=None)
    _post(paths, op="delete", name="Kept", etag=etag)
    found = _post(paths, op="delete", name="Other", etag=_etag(found, "Other"))
    assert [v["name"] for v in found] == ["q and Te"]
    assert json.loads(views.store_path(paths).read_text()) == {}  # emptied, pruned


def test_a_layout_over_64_kb_is_refused(paths):
    big = {"v": 1, "pad": "x" * views.LAYOUT_MAX}
    assert "64 KB" in _refused(400, _save, paths, layout=big)
    _refused(400, _save, paths, layout=[1, 2])
    _refused(400, _save, paths, layout=None)
    assert not views.store_path(paths).exists()


@pytest.mark.parametrize("name", [None, "", "   ", "Default", " default ",
                                  "a\x00b", "line\nbreak", "x" * 65, 3])
def test_a_bad_view_name_is_refused(paths, name):
    _refused(400, _save, paths, name=name)


def test_a_bad_request_is_refused(paths):
    _refused(400, views.change, paths, [])
    _refused(400, _post, paths, op="copy", name="Mine")
    _refused(400, views.change, paths, {"op": "save", "reviewer": "Ana", "name": "x",
                                        "layout": {}})
    message = _refused(400, views.change, paths, {"op": "save", "event": EVENT,
                                                  "name": "x", "layout": {}})
    assert message == "Pick your name to save views"
    _refused(400, views.change, paths, {"op": "save", "event": EVENT,
                                        "reviewer": "a\x07", "name": "x",
                                        "layout": {}})
    _refused(400, views.change, paths, {"op": "save", "event": EVENT,
                                        "reviewer": ["Ana"], "name": "x",
                                        "layout": {}})
    etag = _etag(_save(paths), "Mine")
    _refused(400, _post, paths, op="rename", name="Mine", new_name="Default",
             etag=etag)
    _refused(400, _post, paths, op="rename", name="Mine", new_name=None, etag=etag)


def test_a_preset_cannot_be_overwritten(paths):
    assert "preset" in _refused(400, _save, paths, name="Q AND TE")


def test_presets_skip_malformed_entries_and_follow_the_file(paths):
    assert [v["name"] for v in views.presets(paths, EVENT)] == ["q and Te"]
    (paths.config_dir / "views.yaml").write_text("presets: {mhd_n1: [{name: B, "
                                                 "layout: {v: 1}}], ae: []}\n")
    assert [v["name"] for v in views.presets(paths, EVENT)] == ["B"]
    (paths.config_dir / "views.yaml").write_text(":\n  - [")
    assert views.presets(paths, EVENT) == []
    (paths.config_dir / "views.yaml").unlink()
    assert views.presets(paths, EVENT) == []


def test_the_repo_presets_load(tmp_path):
    from pathlib import Path

    config = Path(__file__).parents[2] / "configs" / "labeler"
    found = views.presets(Paths(root=tmp_path, config_dir=config), EVENT)
    assert found and all(v["preset"] and v["layout"]["v"] == 1 for v in found)


def test_concurrent_saves_keep_the_store_whole(paths):
    errors = []

    def save(i):
        try:
            _save(paths, name=f"v{i:02d}")
        except Exception as error:  # noqa: BLE001 - the test reports any failure
            errors.append(error)

    threads = [threading.Thread(target=save, args=(i,)) for i in range(16)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    stored = json.loads(views.store_path(paths).read_text())
    assert sorted(stored["Ana"][EVENT]) == [f"v{i:02d}" for i in range(16)]
    leftovers = [p.name for p in views.store_path(paths).parent.iterdir()]
    assert sorted(leftovers) == ["views.json", "views.json.lock"]


def test_a_failed_write_leaves_the_old_store(paths, monkeypatch):
    _save(paths)
    before = views.store_path(paths).read_text()

    def broken(*args):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    with pytest.raises(OSError):
        _save(paths, name="Second")
    assert views.store_path(paths).read_text() == before
    assert not list(views.store_path(paths).parent.glob(".*.tmp"))


def test_a_malformed_reviewer_entry_is_left_as_it_is(paths):
    path = views.store_path(paths)
    path.parent.mkdir(parents=True)
    for data in ({"Ana": [1]}, {"Ana": {EVENT: "x"}}):
        text = json.dumps(data)
        path.write_text(text)
        assert "malformed" in _refused(500, _save, paths)
        assert path.read_text() == text
        assert views.list_views(paths, EVENT, "Ana") == views.presets(paths, EVENT)


def test_an_unreadable_store_is_left_as_it_is(paths):
    path = views.store_path(paths)
    path.parent.mkdir(parents=True)
    for text in ("{not json", "[1, 2]"):
        path.write_text(text)
        assert "left as it is" in _refused(500, _save, paths)
        _refused(500, views.list_views, paths, EVENT, "Ana")
        assert path.read_text() == text
