"""A pseudo-mask's regions on the review page, the reviewer's word on each, and
TokEye's whole-shot layer under them."""

from __future__ import annotations

import base64
import getpass
import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from labeler.ae.seg import regions, whole
from labeler.ae.seg.pseudo import IGNORE, PseudoMask
from labeler.config import Paths
from labeler.events.review.rows import Grid
from labeler.events.ui.app import COOKIE, create_app

from . import ae_tree, round3_tree

ROSTER = (
    "shot,tier,holdout,reviewers,verified_on,notes\n"
    "170815,gold,false,alice,2026-01-01,\n"
    "178642,unverified,true,,,\n"
)
SOURCE = "shot,category,t_start,t_end,confidence\n170815,0,0,2000,\n178642,0,0,2000,\n"
SHOT = 170815


def _mask() -> PseudoMask:
    mask = np.zeros((257, 300), dtype=np.uint8)
    mask[:82] = IGNORE
    mask[100:102, 10:15] = 1  # region 1: ten pixels
    mask[[200, 201, 202], [50, 51, 52]] = 1  # region 2: a diagonal, one region
    mask[250, 280:] = IGNORE
    return PseudoMask(SHOT, -10.0, 2.048, 0.0, 500 / 512, mask)


@pytest.fixture
def paths(tmp_path):
    directory = tmp_path / "events" / "alfven_eigenmode"
    (directory / "format").mkdir(parents=True)
    (directory / "format" / "alfven_eigenmode_format_2026_v1.csv").write_text(SOURCE)
    (directory / "shots.csv").write_text(ROSTER)
    found = Paths(
        root=tmp_path / "root",
        corpus=tmp_path / "corpus",
        text_root=tmp_path / "text",
        logs_jsonl=tmp_path / "logs.jsonl",
        label_tables=directory.parent,
        raw_cache=tmp_path / "raw",
    )
    _mask().save(whole.review_file(found, SHOT))
    return found


@pytest.fixture
def client(paths):
    transport = TestClient(create_app(paths=paths, token="secret"))
    transport.cookies.set(COOKIE, "secret")
    return transport


def _url(shot=SHOT, event="alfven_eigenmode"):
    return f"/api/masks?event={event}&shot={shot}"


def _body(client, **change):
    view = client.get(_url()).json()
    body = {
        "event": "alfven_eigenmode",
        "shot": SHOT,
        "pseudo_sha256": view["pseudo_sha256"],
        "revision": view.get("revision", 0),
        "rejected": [2],
        "name": " Ada ",
    }
    return {**body, **change}


def test_regions_are_numbered_in_label_order_with_their_runs_and_boxes():
    one, two = regions.describe(_mask())
    assert (one["id"], one["pixels"], two["id"], two["pixels"]) == (1, 10, 2, 3)
    assert one["runs"] == [[100, 10, 5], [101, 10, 5]]
    assert two["runs"] == [[200, 50, 1], [201, 51, 1], [202, 52, 1]]
    assert one["t0_ms"] == pytest.approx(-10.0 + 10 * 2.048)
    assert one["t1_ms"] == pytest.approx(-10.0 + 15 * 2.048)
    assert one["f0_khz"] == pytest.approx(99.5 * 500 / 512)
    assert one["f1_khz"] == pytest.approx(101.5 * 500 / 512)


def test_a_rejected_region_becomes_background_unless_the_decision_is_stale():
    pm = _mask()
    decision = {"rejected": [2], "pseudo_sha256": "a" * 64}
    kept = regions.reviewed_mask(pm, decision, "a" * 64)
    assert (kept[100:102, 10:15] == 1).all()
    assert (kept[[200, 201, 202], [50, 51, 52]] == 0).all()
    assert (kept[:82] == IGNORE).all() and (kept[250, 280:] == IGNORE).all()
    assert (regions.reviewed_mask(pm, decision, "b" * 64) == pm.mask).all()
    assert (regions.reviewed_mask(pm, None, "a" * 64) == pm.mask).all()


def test_a_region_rejected_on_the_page_s_mask_is_background_in_another_mask():
    page = _mask()
    target = np.zeros_like(page.mask)
    target[100:102, 10:15] = 1
    target[200, 50] = 1
    target[201, 51] = IGNORE  # the reviewer's word wins where the mask is unsure
    other = PseudoMask(SHOT, -10.0, 2.048, 0.0, 500 / 512, target)
    moved = regions.transfer(target, other, page, {"rejected": [2]})
    assert (moved[100:102, 10:15] == 1).all(), "region 1 is kept"
    assert (moved[[200, 201, 202], [50, 51, 52]] == 0).all()
    assert target[201, 51] == IGNORE, "the target is not changed in place"
    for moved_grid in (
        PseudoMask(SHOT, -8.0, 2.048, 0.0, 500 / 512, target),
        PseudoMask(SHOT, -10.0, 2.048, 0.0, 500 / 512, target[:, :200]),
    ):
        with pytest.raises(ValueError, match="grid"):
            regions.transfer(moved_grid.mask, moved_grid, page, {"rejected": [2]})


def test_the_file_a_decision_was_made_on_holds_while_its_sha256_does(paths):
    file = whole.review_file(paths, SHOT)
    sha = regions.file_sha256(file)
    page = {"pseudo": whole.REVIEW, "pseudo_sha256": sha, "rejected": [2]}
    assert regions.decided_file(paths, SHOT, page) == file
    assert regions.decided_file(paths, SHOT, {}) == regions.pseudo_file(paths, SHOT)
    assert regions.decided_file(
        paths, SHOT, {"pseudo": "pseudo-v3"}
    ) == regions.pseudo_file(paths, SHOT, "v3")
    assert regions.clicked_name(SHOT, page) == f"pseudo-v1-full/{SHOT}.npz"
    assert regions.clicked_mask(paths, SHOT, page, "pseudo-v3") == (file, None)
    assert regions.clicked_mask(paths, SHOT, page, whole.REVIEW) == (None, None)
    assert regions.clicked_mask(paths, SHOT, None, "pseudo-v3") == (None, None)
    stale = {**page, "pseudo_sha256": "0" * 64}
    path, why = regions.clicked_mask(paths, SHOT, stale, "pseudo-v3")
    assert path is None and "earlier" in why and str(file) in why
    path, why = regions.clicked_mask(paths, SHOT, {**page, "pseudo": "x"}, "v3")
    assert path is None and "unknown mask set" in why
    file.unlink()
    path, why = regions.clicked_mask(paths, SHOT, page, "pseudo-v3")
    assert path is None and "gone" in why


def test_a_rerun_keeps_an_unchanged_review_mask_and_its_sha256(tmp_path, monkeypatch):
    paths = round3_tree.build(tmp_path, {101: "train"}, tokeye_dt=0.256)
    ae_tree.env(monkeypatch, paths)
    assert whole.main([]) == 0
    file = whole.review_file(paths, 101)
    before = file.stat()
    assert whole.main([]) == 0
    assert file.stat().st_ino == before.st_ino, "not rewritten"
    pm = PseudoMask.load(file)
    assert whole.unchanged(file, pm)
    changed = pm.mask.copy()
    changed[0, 0] = 1 - changed[0, 0] if changed[0, 0] in (0, 1) else 0
    assert not whole.unchanged(file, PseudoMask(**{**vars(pm), "mask": changed}))
    assert not whole.unchanged(file, PseudoMask(**{**vars(pm), "t0_ms": 1.0}))
    assert not whole.unchanged(tmp_path / "none.npz", pm)


def test_the_page_reads_the_regions_and_saves_a_rejection_with_the_name(client, paths):
    view = client.get(_url()).json()
    assert [r["id"] for r in view["regions"]] == [1, 2]
    assert view["rejected"] == [] and view["stale"] is False
    assert view["last_save"] is None
    assert view["revision"] == 0
    assert view["grid"] == {"t0_ms": -10.0, "dt_ms": 2.048, "n": 300}
    saved = client.post("/api/masks", json=_body(client))
    assert saved.status_code == 200
    assert saved.json()["rejected"] == [2]
    assert saved.json()["last_save"]["name"] == "Ada"
    assert saved.json()["revision"] == 1
    assert saved.json()["last_save"]["reviewer"] == getpass.getuser()
    assert client.get(_url()).json()["rejected"] == [2]
    log = regions.log_path(paths.label_tables / "alfven_eigenmode")
    [line] = log.read_text().splitlines()
    entry = json.loads(line)
    assert entry["pseudo_sha256"] == view["pseudo_sha256"]
    assert entry["pseudo"] == "pseudo-v1-full" and entry["shot"] == SHOT


def test_the_last_save_is_the_decision_and_the_earlier_ones_stay(client, paths):
    client.post("/api/masks", json=_body(client))
    client.post("/api/masks", json=_body(client, rejected=[], name=None))
    assert client.get(_url()).json()["rejected"] == []
    log = regions.log_path(paths.label_tables / "alfven_eigenmode")
    assert len(log.read_text().splitlines()) == 2
    assert client.get(_url()).json()["revision"] == 2


def test_a_decision_on_an_older_pseudo_mask_is_stale(client, paths):
    client.post("/api/masks", json=_body(client))
    changed = _mask()
    changed.mask[120, 20:30] = 1
    changed.save(whole.review_file(paths, SHOT))
    view = client.get(_url()).json()
    assert view["rejected"] == [] and view["stale"] is True
    assert view["last_save"]["rejected"] == [2]
    assert view["revision"] == 1


def test_a_save_must_name_the_revision_it_replaces(client, paths):
    body = _body(client)
    assert client.post("/api/masks", json=body).status_code == 200
    stale = client.post("/api/masks", json={**body, "rejected": [1]})
    assert stale.status_code == 409
    assert "mask decisions changed" in stale.json()["error"]
    assert client.get(_url()).json()["rejected"] == [2]
    log = regions.log_path(paths.label_tables / "alfven_eigenmode")
    assert len(log.read_text().splitlines()) == 1
    fresh = client.post("/api/masks", json=_body(client, rejected=[1, 2]))
    assert fresh.status_code == 200 and fresh.json()["revision"] == 2


@pytest.mark.parametrize("revision", [None, -1])
def test_revision_is_required_and_nonnegative(client, revision):
    body = _body(client, revision=revision)
    if revision is None:
        del body["revision"]
    assert client.post("/api/masks", json=body).status_code == 422


@pytest.mark.parametrize(
    ("change", "status"),
    [
        ({"pseudo_sha256": "0" * 64}, 409),
        ({"rejected": [3]}, 400),
        ({"rejected": [0]}, 400),
        ({"name": "a\x07b"}, 400),
        ({"shot": 178642}, 404),
        ({"shot": 999}, 404),
        ({"event": "sawtooth_oscillation"}, 404),
    ],
)
def test_a_bad_save_is_refused_and_writes_nothing(client, paths, change, status):
    assert client.post("/api/masks", json=_body(client, **change)).status_code == status
    assert not regions.log_path(paths.label_tables / "alfven_eigenmode").exists()


def test_a_shot_without_a_pseudo_mask_is_404_and_the_route_is_gated(client):
    assert client.get(_url(shot=178642)).status_code == 404
    assert client.get(_url(event="nothing")).status_code == 404
    assert TestClient(client.app).get(_url()).status_code == 401


def _layer() -> np.ndarray:
    """TokEye lit on two chords in page bin 100 over 5121-6139 ms, on one chord in 151."""
    t = np.arange(2500, 3000) * 2.048 + 1.0
    clean = np.zeros((4, 512, len(t)), dtype=bool)
    clean[:2, 198] = True  # page bin k takes TokEye bins 2k - 2 and 2k - 1
    clean[3, 300] = True
    return whole.build(Grid(0.0, 2.048, 4000), 257, (t, clean, None))


def test_the_layer_lights_where_two_chords_agree_past_the_label_window():
    lit = _layer()
    assert lit.shape == (257, 4000)
    assert lit[100, 2500:3000].all() and lit.sum() == 500


def test_the_page_reads_the_layer_bit_packed_along_time(client, paths):
    lit = _layer()
    whole.save(whole.layer_file(paths, SHOT), Grid(0.0, 2.048, 4000), 0.0, 0.977, lit)
    view = client.get(f"/api/tokeye?event=alfven_eigenmode&shot={SHOT}").json()
    assert view["grid"] == {"t0_ms": 0.0, "dt_ms": 2.048, "n": 4000}
    assert (view["n_y"], view["y0_khz"], view["dy_khz"]) == (257, 0.0, 0.977)
    bits = np.frombuffer(base64.b64decode(view["bits"]), dtype=np.uint8)
    drawn = np.unpackbits(bits.reshape(257, -1), axis=1, count=4000).astype(bool)
    assert np.array_equal(drawn, lit)


def test_the_page_reads_the_whole_window_mask_not_pseudo_v1(client, paths):
    other = PseudoMask(SHOT, -10.0, 2.048, 0.0, 500 / 512, np.zeros((257, 300), "u1"))
    other.save(regions.pseudo_file(paths, SHOT))  # pseudo-v1's: SegNet v1's input
    view = client.get(_url()).json()
    assert view["pseudo"] == whole.REVIEW and len(view["regions"]) == 2
    assert view["pseudo_sha256"] == regions.file_sha256(whole.review_file(paths, SHOT))


def test_a_shot_without_a_layer_is_404_and_the_route_is_gated(client):
    url = "/api/tokeye?event={}&shot={}"
    assert client.get(url.format("alfven_eigenmode", 178642)).status_code == 404
    assert client.get(url.format("nothing", SHOT)).status_code == 404
    assert (
        TestClient(client.app).get(url.format("alfven_eigenmode", SHOT)).status_code
        == 401
    )
