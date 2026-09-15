"""Offline blurbs: what the model sees, what it may say, and what happens when it says more.

Ported from shot-recommender-system (shotrec) @565d548.
"""

from __future__ import annotations

import json

import httpx
import pandas as pd
import pytest

from shot_design import cli, config
from shot_design.llm.client import LLMClient
from shot_design.retrieval import blurb
from shot_design.schema import ResultItem
from shot_design.shotdb import build, store

from .test_build_store import stub_embeddings  # noqa: F401

CFG = {
    "provider": "ollama",
    "base_url": "http://llm.test",
    "models": {"quality": "gemma4:26b", "fast": "gemma4:e4b"},
    "reasoning_effort": "none",
    "blurb": {"model": "quality", "max_words": 90, "prompt_version": 5},
}
GOOD = (
    "The session studied QH-mode access at low torque. It ran through to rampdown. "
    "No notable findings were logged."
)


class FakeClient(LLMClient):
    def __init__(self, content, paths, cfg=None):
        self.content, self.calls = content, []

        def handler(request):
            self.calls.append(json.loads(request.content))
            return httpx.Response(200, json={
                "model": "gemma4:26b",
                "choices": [{"message": {"content": self.content}}],
            })

        super().__init__(cfg or CFG, paths, transport=httpx.MockTransport(handler))


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
    cand = GOOD
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
    prompt = blurb._system(90)
    assert "disrupt" in prompt and "quench" in prompt and "terminated early" in prompt
    assert "exactly three plain sentences" in prompt
    assert "whether it succeeded" in prompt
    assert "third" in prompt and "interesting finding" in prompt
    assert "operator entries" in prompt and "shot brief" in prompt
    assert "No notable findings were logged." in prompt
    assert "90 words" in prompt


def test_make_uses_the_model_and_falls_back_on_failure(rec, paths):
    good = FakeClient(GOOD, paths)
    b = blurb.make(rec, good, CFG)
    assert b.source == "llm" and b.text == good.content
    prompt = good.calls[0]["messages"]
    assert prompt[0]["role"] == "system" and str(rec.shot) in prompt[1]["content"]
    assert "(prompt v5)" in prompt[0]["content"]
    assert good.calls[0]["reasoning_effort"] == "none"
    bad = FakeClient(f"Shot {rec.shot} reached 3.1 MA.", paths)
    # Clear the previous deterministic reply before trying a different fake server answer.
    for cached in paths.llm_cache_dir.rglob("*.json"):
        cached.unlink()
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
    assert {"blurb", "blurb_source", "blurb_model", "blurb_prompt_version"} <= set(df.columns)
    assert df["blurb_source"].iloc[0] == "template" and df["blurb"].iloc[0]
    assert df["blurb_model"].iloc[0] == "gemma4:26b"
    assert df["blurb_prompt_version"].iloc[0] == 5
    assert pd.api.types.is_integer_dtype(df["blurb_prompt_version"])


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
    client = FakeClient(GOOD, paths)
    n = build.write_blurbs(paths, client, only_missing=True)
    assert n == 2 and len(client.calls) == 2
    df = pd.read_parquet(paths.db_dir / "shots.parquet")
    assert set(df["blurb_source"]) == {"llm"}
    assert set(df["blurb_model"]) == {"gemma4:26b"}
    assert set(df["blurb_prompt_version"]) == {5}
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
    provenance = {"model": "gemma4:26b", "prompt_version": 5}
    assert before["blurbs"] == {"llm": 0, "template": 2, **provenance}
    build.write_blurbs(paths, FakeClient(GOOD, paths), True)
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    df = pd.read_parquet(paths.db_dir / "shots.parquet")
    assert manifest["blurbs"] == {
        **{k: int((df["blurb_source"] == k).sum()) for k in ("llm", "template")},
        **provenance,
    }
    assert manifest["blurbs"] == {"llm": 2, "template": 0, **provenance}
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


@pytest.mark.parametrize("candidate", [
    GOOD,
    "Study low torque! It ran through rampdown? No notable findings were logged.",
    "Study low torque. It held 1.2 MA. No notable findings were logged.",
    "Study low torque. It held current in kA. No notable findings were logged.",
])
def test_gate_counts_sentences_without_splitting_decimals(rec, candidate):
    assert blurb.gate(blurb.source_text(rec) + " Current 1.2 MA.", candidate, rec, 90) is None


@pytest.mark.parametrize(("candidate", "count"), [
    ("Study low torque. It ran through rampdown.", 2),
    (GOOD + " The plasma was steady.", 4),
    ("Study low torque", 0),
    ("Study low torque. It ran through rampdown. No notable findings were logged", 2),
    (GOOD + " Unfinished claim", 3),
])
def test_gate_rejects_wrong_sentence_count_or_unfinished_text(rec, candidate, count):
    reason = blurb.gate(blurb.source_text(rec), candidate, rec, 90)
    assert reason is not None and "three" in reason and str(count) in reason


@pytest.mark.parametrize("version", [4, None, "absent", 5, 6])
def test_write_blurbs_retries_stale_or_unknown_versions(rec, paths, version):
    table = paths.db_dir / "shots.parquet"
    df = pd.read_parquet(table)
    df["blurb"], df["blurb_source"] = "Old answer.", "llm"
    if version != "absent":
        df["blurb_prompt_version"] = pd.Series([version], index=df.index, dtype="Int64")
    elif "blurb_prompt_version" in df:
        df = df.drop(columns="blurb_prompt_version")
    df.to_parquet(table)
    client = FakeClient(GOOD, paths)
    expected = int(version in (4, None, "absent"))
    assert build.write_blurbs(paths, client) == expected
    assert len(client.calls) == expected
    updated = pd.read_parquet(table)
    assert updated["blurb_prompt_version"].iloc[0] == (5 if expected else version)
    assert updated["blurb"].iloc[0] == (GOOD if expected else "Old answer.")


def test_write_blurbs_uses_client_config_for_prompt_and_provenance(rec, paths):
    custom = {**CFG, "blurb": {"model": "fast", "max_words": 90, "prompt_version": 6}}
    client = FakeClient(GOOD, paths, cfg=custom)
    assert build.write_blurbs(paths, client) == 1
    sent = client.calls[0]
    assert sent["model"] == "gemma4:e4b"
    assert "prompt v6" in sent["messages"][0]["content"]
    df = pd.read_parquet(paths.db_dir / "shots.parquet")
    assert df["blurb_model"].iloc[0] == "gemma4:e4b"
    assert df["blurb_prompt_version"].iloc[0] == 6
    manifest = json.loads((paths.db_dir / "manifest.json").read_text())
    assert manifest["blurbs"] == {
        "llm": 1, "template": 0, "model": "gemma4:e4b", "prompt_version": 6,
    }


@pytest.mark.parametrize("candidate", [GOOD, "Only one sentence."])
def test_cli_dry_run_limits_calls_prints_candidate_gate_final_and_writes_nothing(
    paths, staged_shot_a, staged_shot_b, text_fixtures, stub_embeddings,  # noqa: F811
    monkeypatch, capsys, candidate,
):
    from shot_design.llm import client as client_module

    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False,
    )
    capsys.readouterr()
    client = FakeClient(candidate, paths)
    monkeypatch.setattr(client_module, "LLMClient", lambda: client)
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns)
              for p in paths.data_root.rglob("*") if p.is_file()}
    assert cli.main(["blurb", "--all", "--dry-run", "--limit", "1"]) == 0
    output = capsys.readouterr().out
    assert f"Shot {staged_shot_a}" in output and str(staged_shot_b) not in output
    rec = store.ShotDB.load(paths.db_dir).get(staged_shot_a)
    assert f"Source text: {len(blurb.source_text(rec))} chars" in output
    assert f"Candidate: {candidate}" in output
    assert "Gate: PASS" in output if candidate == GOOD else "Gate: FAIL" in output
    assert f"Final (llm): {GOOD}" in output if candidate == GOOD else (
        f"Final (template): {blurb.fallback(rec)}" in output
    )
    assert len(client.calls) == 1
    after = {p: (p.read_bytes(), p.stat().st_mtime_ns)
             for p in paths.data_root.rglob("*") if p.is_file()}
    assert after == before  # includes request cache, manifests, tables and sidecars


def test_write_blurbs_limit_applies_after_missing_filter(
    paths, staged_shot_a, staged_shot_b, text_fixtures, stub_embeddings,  # noqa: F811
):
    build.build(
        [staged_shot_a, staged_shot_b], paths, build.load_build_cfg(), workers=1, encode=False,
    )
    client = FakeClient(GOOD, paths)
    assert build.write_blurbs(paths, client, limit=1) == 1
    assert build.write_blurbs(paths, client, limit=1) == 1
    assert len(client.calls) == 2
    assert set(pd.read_parquet(paths.db_dir / "shots.parquet")["blurb_source"]) == {"llm"}


def test_cli_rejects_negative_limit(paths, capsys):
    assert cli.main(["blurb", "--limit", "-1"]) == 2
    assert "non-negative" in capsys.readouterr().err


def test_zero_limit_writes_nothing(rec, paths):
    client = FakeClient(GOOD, paths)
    table = paths.db_dir / "shots.parquet"
    before = table.stat().st_mtime_ns
    assert build.write_blurbs(paths, client, limit=0) == 0
    assert not client.calls and table.stat().st_mtime_ns == before


def test_default_ollama_without_endpoint_uses_template_without_a_request(rec, paths):
    def unexpected(request):
        pytest.fail("a missing endpoint must not open a socket")

    cfg = config.load_yaml("llm.yaml")
    assert cfg["provider"] == "ollama"
    client = LLMClient(cfg, paths, transport=httpx.MockTransport(unexpected))
    assert client.available()[0] is False
    assert "scripts/shot_design/serve_llm.sbatch" in client.available()[1]
    result = blurb.make(rec, client)
    assert result.source == "template" and result.reason == "no model"
    assert not list(paths.llm_cache_dir.rglob("*.json"))
