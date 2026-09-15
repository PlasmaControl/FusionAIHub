"""Offline blurbs: what the model sees, what it may say, and what happens when it says more.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from shot_design.llm.client import Reply
from shot_design.retrieval import blurb
from shot_design.schema import ResultItem
from shot_design.shotdb import build, store

from .test_build_store import stub_embeddings  # noqa: F401

CFG = {"blurb": {"model": "quality", "max_words": 70, "prompt_version": 1}}


class FakeClient:
    def __init__(self, content):
        self.content, self.calls = content, []

    def available(self):
        return True, ""

    def chat(self, messages, **kw):
        self.calls.append((messages, kw))
        return Reply(content=self.content, tool_calls=[], model="fake")


@pytest.fixture
def rec(paths, staged_shot_a, text_fixtures, stub_embeddings):  # noqa: F811
    build.build([staged_shot_a], paths, build.load_build_cfg(), workers=1, encode=False)
    return store.ShotDB.load(paths.db_dir).get(staged_shot_a)


def test_source_text_carries_title_quotes_and_outcome_but_no_scalar_table(rec):
    s = blurb.source_text(rec)
    assert rec.human.run_title in s
    assert "Good shot, ran through to rampdown" in s
    assert "ip_mean" not in s and "pnbi_15L_peak" not in s


def test_gate_accepts_a_faithful_blurb(rec):
    src = blurb.source_text(rec)
    cand = f"Shot {rec.shot} was part of the QH-mode access study at low torque. It ran through to rampdown."
    assert blurb.gate(src, cand, rec, 70) is None


def test_gate_rejects_an_invented_number_or_shot(rec):
    src = blurb.source_text(rec)
    assert "number" in blurb.gate(src, f"Shot {rec.shot} reached 2.4 MA.", rec, 70)
    assert "shot" in blurb.gate(src, f"Shot {rec.shot} repeated 175000.", rec, 70)


def test_gate_rejects_an_invented_shot_in_the_2xxxxx_band(rec):
    """Shot references used to be `1\\d{5}`: an invented 203500 read as an ordinary number and
    the gate reported it (if at all) as the wrong kind of fabrication."""
    src = blurb.source_text(rec)
    reason = blurb.gate(src, f"Shot {rec.shot} repeated 203500.", rec, 70)
    assert reason is not None and reason.startswith("shot not in source") and "203500" in reason


def test_gate_rejects_a_quoted_candidate(rec):
    src = blurb.source_text(rec)
    # The blurb never quotes the operators (that is `describe.best_quote`'s job on the card), so
    # a candidate containing a quotation mark is rejected outright, regardless of its numbers or
    # shot references.
    cand = f'Shot {rec.shot} was, as the operator put it, "everything broke" in the end.'
    reason = blurb.gate(src, cand, rec, 70)
    assert reason == "quotation marks"


def test_gate_rejects_too_long(rec):
    src = blurb.source_text(rec)
    assert "words" in blurb.gate(src, "word " * 71, rec, 70)


def test_system_prompt_requires_naming_a_bad_ending():
    prompt = blurb._system(70)
    assert "disrupt" in prompt and "quench" in prompt and "terminated early" in prompt


def test_make_uses_the_model_and_falls_back_on_failure(rec):
    good = FakeClient(
        f"Shot {rec.shot} studied QH-mode access at low NBI torque. It ran through to rampdown."
    )
    b = blurb.make(rec, good, CFG)
    assert b.source == "llm" and b.text == good.content
    prompt = good.calls[0][0]
    assert prompt[0]["role"] == "system" and str(rec.shot) in prompt[1]["content"]
    bad = FakeClient(f"Shot {rec.shot} reached 3.1 MA.")
    b = blurb.make(rec, bad, CFG)
    assert b.source == "template" and "number" in b.reason and b.text == blurb.fallback(rec)
    b = blurb.make(rec, None, CFG)
    assert b.source == "template" and b.reason == "no model"


def test_shots_parquet_has_blurb_columns_after_build(
    paths,
    staged_shot_a,
    text_fixtures,
    stub_embeddings,  # noqa: F811
):
    build.build([staged_shot_a], paths, build.load_build_cfg(), workers=1, encode=False)
    df = pd.read_parquet(paths.db_dir / "shots.parquet")
    assert {"blurb", "blurb_source"} <= set(df.columns)
    assert df["blurb_source"].iloc[0] == "template" and df["blurb"].iloc[0]


def test_write_blurbs_backfills_only_missing_and_is_atomic(
    paths,
    staged_shot_a,
    staged_shot_b,
    text_fixtures,
    stub_embeddings,  # noqa: F811
):
    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False
    )
    # No numbers and no quotes, so the same reply passes the gate for either shot: this test is
    # about the backfill's bookkeeping, not about the gate (which has its own tests above).
    client = FakeClient("The session studied QH-mode access at low NBI torque.")
    n = build.write_blurbs(paths, client, only_missing=True)
    assert n == 2 and len(client.calls) == 2
    df = pd.read_parquet(paths.db_dir / "shots.parquet")
    assert set(df["blurb_source"]) == {"llm"}
    assert not list(paths.db_dir.glob("*.part"))
    assert build.write_blurbs(paths, client, only_missing=True) == 0


def test_write_blurbs_updates_the_manifest_counts(
    paths,
    staged_shot_a,
    staged_shot_b,
    text_fixtures,
    stub_embeddings,  # noqa: F811
):
    """The backfill rewrites shots.parquet, so the manifest's blurb counts -- the only place a
    reader can see how much of the database the model actually wrote -- have to follow it, or
    `shot_design blurb` leaves a database claiming 0 llm blurbs while every row has one."""
    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False
    )
    before = json.loads((paths.db_dir / "manifest.json").read_text())
    assert before["blurbs"] == {"llm": 0, "template": 2}
    build.write_blurbs(paths, FakeClient("The session studied QH-mode access at low torque."), True)
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    df = pd.read_parquet(paths.db_dir / "shots.parquet")
    assert manifest["blurbs"] == {
        k: int((df["blurb_source"] == k).sum()) for k in ("llm", "template")
    }
    assert manifest["blurbs"] == {"llm": 2, "template": 0}
    assert manifest["n_shots"] == before["n_shots"]  # nothing else was rewritten
    assert not list(paths.db_dir.glob("*.part"))


def test_result_item_carries_the_blurb(
    paths,
    staged_shot_a,
    staged_shot_b,
    text_fixtures,
    stub_embeddings,  # noqa: F811
):
    from shot_design.retrieval import rank
    from shot_design.schema import QueryState

    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False
    )
    db = store.ShotDB.load(paths.db_dir)
    items = rank.search(QueryState(ref_shot=staged_shot_a), db).items
    assert items and isinstance(items[0], ResultItem) and items[0].blurb
