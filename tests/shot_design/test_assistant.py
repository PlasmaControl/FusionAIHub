"""Gemma proposals remain grounded in real controls and produce physical artifacts."""

# ruff: noqa: F811 - shared pytest fixtures

import json
from types import SimpleNamespace

import h5py
import httpx
import numpy as np
import pytest

from shot_design.llm.client import LLMClient
from shot_design.schema import ResultItem

from .test_program import SHOT, averaging_source, program_source  # noqa: F401


def model_client(
    paths,
    *,
    shot=SHOT,
    fail=False,
    scales=None,
    comparisons=None,
    baseline="reference",
    prose_first=False,
    fail_repair=False,
):
    payloads = [
        {
            "search_text": "tearing mode ECCD control",
            "goal": "Control tearing",
            "start_s": 1,
            "end_s": 5,
        },
        {
            "reference_shot": shot,
            "comparison_shots": comparisons or [],
            "baseline": baseline,
            "scales": scales if scales is not None else {"nbi.total": 1.2},
            "explanation": "Increase the measured beam waveforms by 20 percent for this candidate.",
        },
    ]
    if prose_first:
        payloads.insert(1, "This shot is a suitable measured reference.")
    if fail_repair:
        payloads[-1] = "Still prose instead of a structured proposal."
    replies = iter(payloads)

    def handler(request):
        if fail:
            return httpx.Response(503, json={"error": "model unavailable"})
        reply = next(replies)
        content = reply if isinstance(reply, str) else json.dumps(reply)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    return LLMClient(
        paths=paths,
        cfg={
            "provider": "ollama",
            "base_url": "http://gemma.invalid",
            "models": {"quality": "gemma4:26b"},
            "cache": False,
        },
        transport=httpx.MockTransport(handler),
    )


@pytest.fixture
def candidate_search(monkeypatch):
    from shot_design.design import assistant

    item = ResultItem(
        id=f"{SHOT}:flat_top",
        shot=SHOT,
        segment="flat_top",
        score=0.8,
        description="Measured beam reference",
    )
    monkeypatch.setattr(
        assistant.rank, "search", lambda query, db: SimpleNamespace(items=[item])
    )


def test_grounded_plan_saves_physical_hdf5_and_editable_revision(
    program_source, candidate_search
):
    from shot_design.design import assistant, program

    paths, _ = program_source
    events = []
    result = assistant.run_design(
        "Control tearing modes",
        paths,
        object(),
        client=model_client(paths),
        progress=lambda *args: events.append(args),
    )
    saved = program.load_program(result["design_id"], paths)
    assert saved.reference_shot == SHOT
    assert "nbi.total" in saved.edits
    with h5py.File(result["artifact_path"], "r") as f:
        assert f["actuators"].shape == (80, 88)
        np.testing.assert_allclose(f["time_s"][:3], [1.0, 1.05, 1.1])
        np.testing.assert_allclose(f["actuators"][:2, 12], [120, 240])
        assert f["channel_names"].asstr()[12] == "pinj[0]"
        assert f["channel_units"].asstr()[12] == "W"
        assert not f["available"][:, 47 + 1].any()
        assert np.isnan(f["actuators"][:, 47 + 1]).all()
        metadata = json.loads(f["metadata"].asstr()[()])
        assert metadata["prompt"] == "Control tearing modes"
        assert metadata["model"] == "gemma4:26b"
        assert metadata["reference_shot"] == SHOT
    assert [key for key, status, detail in events if status == "complete"] == [
        "interpret",
        "retrieve",
        "propose",
        "validate",
        "save",
    ]


def test_unavailable_gemma_never_creates_a_fake_success(program_source):
    from shot_design.design import assistant
    from shot_design.llm.client import LLMUnavailable

    paths, _ = program_source
    with pytest.raises(LLMUnavailable):
        assistant.run_design(
            "Control ELMs", paths, object(), client=model_client(paths, fail=True)
        )
    assert not list((paths.data_root / "outputs").glob("*.h5"))


def test_model_cannot_choose_a_shot_outside_retrieval(program_source, candidate_search):
    from shot_design.design import assistant

    paths, _ = program_source
    with pytest.raises(ValueError, match="retrieved"):
        assistant.run_design(
            "Control ELMs", paths, object(), client=model_client(paths, shot=123456)
        )
    assert not list((paths.data_root / "outputs").glob("*.h5"))


def test_missing_channel_edits_are_refused(program_source, candidate_search):
    from shot_design.design import assistant

    paths, _ = program_source
    with pytest.raises(ValueError, match="available"):
        assistant.run_design(
            "Control ELMs",
            paths,
            object(),
            client=model_client(paths, scales={"rmp[0]": 1.1}),
        )


def test_raw_only_reference_can_write_hdf5_without_claiming_ignite_ready(
    program_source, candidate_search
):
    from shot_design.design import assistant

    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    result = assistant.run_design(
        "Control tearing", paths, object(), client=model_client(paths)
    )
    assert result["checks"]["needs_seed"]
    assert not result["checks"]["can_export"]
    assert result["checks"]["hdf5_valid"]
    assert result["artifact_path"].endswith(".h5")


def test_model_selected_average_persists_before_its_scale_edits(
    averaging_source, monkeypatch
):
    from shot_design.design import assistant, program

    paths, _, _ = averaging_source
    items = [
        ResultItem(
            id=f"{shot}:flat_top",
            shot=shot,
            segment="flat_top",
            score=0.8,
            description="Measured reference",
        )
        for shot in (SHOT, SHOT + 1)
    ]
    monkeypatch.setattr(
        assistant.rank, "search", lambda query, db: SimpleNamespace(items=items)
    )
    result = assistant.run_design(
        "Average beam references",
        paths,
        object(),
        client=model_client(
            paths, comparisons=[SHOT + 1], baseline="average", scales={"pinj[0]": 1.1}
        ),
    )
    saved = program.load_program(result["design_id"], paths)
    assert saved.proposal.shots == [SHOT, SHOT + 1]
    assert "pinj[0]" in saved.edits
    with h5py.File(result["artifact_path"], "r") as file:
        assert file["actuators"][0, 12] == pytest.approx(220.0)


@pytest.mark.parametrize(
    "scales", [{"nbi.total": 2.0}, {"nbi.total": 1.1, "pinj[0]": 1.1}]
)
def test_out_of_bounds_and_conflicting_model_edits_cannot_be_saved(
    program_source, candidate_search, scales
):
    from shot_design.design import assistant

    paths, _ = program_source
    with pytest.raises(ValueError):
        assistant.run_design(
            "Adjust beam power",
            paths,
            object(),
            client=model_client(paths, scales=scales),
        )
    assert not list((paths.data_root / "outputs").glob("*.h5"))


def test_prose_model_response_gets_one_structured_repair(
    program_source, candidate_search
):
    from shot_design.design import assistant

    paths, _ = program_source
    result = assistant.run_design(
        "Control tearing", paths, object(), client=model_client(paths, prose_first=True)
    )
    with h5py.File(result["artifact_path"], "r") as file:
        assert file["actuators"][0, 12] == pytest.approx(120)
        metadata = json.loads(file["metadata"].asstr()[()])
        assert metadata["model_call_count"] == 3
        assert metadata["model_calls_cached"] == [False, False, False]


def test_generated_flat_edit_has_two_handles_and_matching_hdf5(
    program_source, candidate_search
):
    from shot_design.design import assistant, program

    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    result = assistant.run_design(
        "Increase ECH",
        paths,
        object(),
        client=model_client(paths, scales={"ech.total": 1.1}),
    )
    saved = program.load_program(result["design_id"], paths)
    assert len(saved.edits["ech.total"]) == 2
    assert saved.edits["ech.total"][0].t_s == 1
    assert saved.edits["ech.total"][-1].t_s == 4.95
    with h5py.File(result["artifact_path"], "r") as file:
        np.testing.assert_allclose(file["actuators"][:, :12], 3.3)
        metadata = json.loads(file["metadata"].asstr()[()])
        assert metadata["simplification"]["relative_peak_tolerance"] == 0.05


def test_generated_smooth_edit_is_bounded_by_five_percent_of_scaled_peak(
    program_source, candidate_search
):
    from shot_design.design import assistant, program

    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    measured = 1000 + 10 * np.sin(np.arange(120))
    wave = np.r_[np.repeat(measured, 50), measured[-1]]
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as file:
        file["pinj/ydata"][0] = wave
    result = assistant.run_design(
        "Increase one beam",
        paths,
        object(),
        client=model_client(paths, scales={"pinj[0]": 1.1}),
    )
    saved = program.load_program(result["design_id"], paths)
    assert len(saved.edits["pinj[0]"]) == 2
    target = measured[20:100] * 1.1
    with h5py.File(result["artifact_path"], "r") as file:
        assert np.max(np.abs(file["actuators"][:, 12] - target)) <= 0.05 * target.max()


def test_format_repair_stops_after_one_attempt(program_source, candidate_search):
    from shot_design.design import assistant

    paths, _ = program_source
    with pytest.raises(ValueError, match="The model returned an invalid"):
        assistant.run_design(
            "Control tearing",
            paths,
            object(),
            client=model_client(paths, prose_first=True, fail_repair=True),
        )
    assert not list((paths.data_root / "outputs").glob("*.h5"))


@pytest.mark.parametrize(
    ("cache_problem", "scales", "channel", "expected", "message"),
    [
        ("normalization", {"pinj[0]": 1.2}, 12, 120.0, "cache normalization"),
        ("constant", {"ech_power[0]": 1.1}, 0, 3.3, "zero-spread"),
        ("missing", {"pinj[0]": 1.2}, 12, 120.0, "seed cache"),
    ],
)
def test_model_input_errors_preserve_valid_physical_proposal(
    program_source, candidate_search, cache_problem, scales, channel, expected, message
):
    import torch

    from shot_design.design import assistant, program

    paths, cache = program_source
    cache_path = paths.data_root / "frame_codes" / f"{SHOT}.pt"
    if cache_problem == "normalization":
        cache["actuators"][:, 12] += 1
        torch.save(cache, cache_path)
    elif cache_problem == "missing":
        cache_path.unlink()
    result = assistant.run_design(
        "Propose measured actuator changes",
        paths,
        object(),
        client=model_client(paths, scales=scales),
    )
    checks = result["checks"]
    assert checks["physical_errors"] == []
    assert any(message in error for error in checks["model_errors"])
    assert not checks["can_export"]
    assert checks["hdf5_valid"]
    saved = program.load_program(result["design_id"], paths)
    with pytest.raises(program.ProgramValidationError, match=message):
        program.export_ignite(saved, paths)
    with h5py.File(result["artifact_path"], "r") as file:
        assert file["actuators"][0, channel] == pytest.approx(expected)
        metadata = json.loads(file["metadata"].asstr()[()])
        assert metadata["checks"]["model_errors"] == checks["model_errors"]


def test_retrieval_evidence_tells_gemma_which_controls_block_ignite(program_source):
    import torch

    from shot_design.design import assistant

    paths, cache = program_source
    cache["actuators"][:, 12] += 1
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    item = ResultItem(
        id=f"{SHOT}:flat_top",
        shot=SHOT,
        segment="flat_top",
        score=0.8,
        description="Measured reference",
    )
    candidate = assistant._candidate(
        item, paths, assistant.Intent(search_text="mode", goal="control")
    )
    packet = assistant._model_candidates([candidate])[0]
    assert "pinj[0]" in packet["ignite_incompatible_channels"]
    assert "nbi.total" in packet["ignite_incompatible_channels"]
    assert "ech_power[0]" in packet["ignite_incompatible_channels"]
    assert "pinj[1]" not in packet["ignite_incompatible_channels"]
