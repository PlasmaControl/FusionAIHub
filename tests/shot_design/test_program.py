"""Deterministic editor/export tests with actual corpus files and frame caches."""

import h5py
import numpy as np
import pytest
import torch

from shot_design.config import load_yaml
from shot_design.design.actuators import build_actuators
from shot_design.shotdb.corpus import CorpusReader

SHOT = 990091
# Verified against the shipped step13400 checkpoint and codec MANIFEST.json.
PRODUCTION_VOCABS = {
    "ece": 32768,
    "bes": 64000,
    "mhr": 32768,
    "co2": 32768,
    "tangtv_lower": 64000,
    "tangtv_upper": 64000,
    "ts_core_density": 1000,
    "ts_core_temp": 1000,
    "ts_tangential_density": 1000,
    "ts_tangential_temp": 1000,
    "cer_ti": 1000,
    "cer_rot": 1000,
    "mse": 1000,
    "filterscopes": 1000,
}


@pytest.fixture
def program_source(paths, monkeypatch):
    monkeypatch.delenv("SHOT_DESIGN_CORPUS", raising=False)
    monkeypatch.delenv("IDEATE_CORPUS", raising=False)
    root = paths.foundation_model_processed_dir
    root.mkdir(parents=True)
    # 120 complete frames, 50 samples/frame. Mean=150, std=50 for pinj[0].
    x = np.arange(6001) / 1000
    wave = np.repeat(np.tile([100.0, 200.0], 60), 50)
    wave = np.r_[wave, wave[-1]]
    with h5py.File(root / f"{SHOT}_processed.h5", "w") as f:
        for group, y in {
            "pinj": np.arange(1, 9)[:, None] * wave,
            "ech_power": np.full((12, 6001), 3.0),
            "gas_raw": np.vstack([wave, np.full(6001, np.nan)]),
        }.items():
            g = f.create_group(group)
            g["xdata"], g["ydata"] = x, y
    raw = build_actuators(SHOT, CorpusReader(root), 120)
    codes = {
        name: torch.arange(120 * spec["n_tok"], dtype=torch.int32).reshape(
            120, spec["n_tok"]
        )
        % 16
        for name, spec in load_yaml("ignite_modalities.yaml")["modalities"].items()
    }
    cache = {
        "codes": codes,
        "actuators": torch.tensor(raw.z.T, dtype=torch.float16),
        "vocabs": dict(PRODUCTION_VOCABS),
        "n_frames": 120,
    }
    cache_dir = paths.data_root / "frame_codes"
    cache_dir.mkdir()
    torch.save(cache, cache_dir / f"{SHOT}.pt")
    return paths, cache


def service():
    from shot_design.design import program

    return program


def draft(**kwargs):
    return service().DesignProgram(reference_shot=SHOT, **kwargs)


def vertices(factor=1.2, scale=1):
    return [
        {"t_s": i * 0.05, "y": factor * scale * (100 if i % 2 == 0 else 200)}
        for i in range(20, 100)
    ]


@pytest.fixture
def averaging_source(program_source):
    import shutil

    paths, cache = program_source
    donor = paths.foundation_model_processed_dir / f"{SHOT + 1}_processed.h5"
    shutil.copyfile(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", donor
    )
    with h5py.File(donor, "a") as f:
        f["pinj/ydata"][0] *= 3
        f["gas_raw/ydata"][0] *= 3
    return paths, cache, donor


def test_average_uses_physical_values_and_keeps_first_reference_seed(averaging_source):
    paths, cache, _ = averaging_source
    got = service().average_references(draft(comparison_shots=[SHOT + 1]), paths)
    assert got["validation"]["can_export"]
    assert got["program"]["edits"] == {}
    assert got["program"]["proposal"]["method"] == "average"
    assert got["program"]["proposal"]["shots"] == [SHOT, SHOT + 1]
    channels = {c["key"]: c for c in got["channels"]}
    assert channels["pinj[0]"]["vertices"][:2] == [
        {"t_s": 1.0, "y": 200.0}, {"t_s": 1.05, "y": 400.0}
    ]
    assert channels["gas_raw[0]"]["vertices"][0]["y"] == 200
    assert channels["nbi.total"]["vertices"][0]["y"] == 3700
    assert channels["pinj[0]"]["baseline_vertices"] == channels["pinj[0]"]["vertices"]
    out = torch.load(service().export_ignite(
        service().DesignProgram(**got["program"]), paths
    ), weights_only=True)
    assert torch.equal(out["actuators"][:20], cache["actuators"][:20])
    assert torch.equal(out["actuators"][:, 13:47], cache["actuators"][:100, 13:47])
    assert out["actuators"][20:22, 12].tolist() == pytest.approx([1, 5], abs=0.001)
    for key in out["codes"]:
        assert torch.equal(out["codes"][key], cache["codes"][key][:100])


def test_average_total_edit_uses_proposal_ratios_and_survives_reopen(averaging_source):
    paths, _, donor = averaging_source
    merged = service().average_references(draft(comparison_shots=[SHOT + 1]), paths)
    program = service().DesignProgram(**merged["program"])
    program.edits = {"nbi.total": [service().Vertex(**v) for v in vertices(2, 37)]}
    saved = service().save(program, paths)
    with h5py.File(donor, "a") as f:
        f["pinj/ydata"][0] *= 10
    reopened = service().load(saved["program"]["id"], paths)
    assert reopened["validation"]["can_export"]
    assert reopened["program"] == saved["program"]
    channels = {c["key"]: c for c in reopened["channels"]}
    assert channels["nbi.total"]["baseline_vertices"][0]["y"] == 3700
    assert channels["nbi.total"]["vertices"][0]["y"] == 7400
    assert channels["pinj[0]"]["vertices"][0]["y"] == 400
    out = torch.load(service().export_ignite(
        service().DesignProgram(**reopened["program"]), paths
    ), weights_only=True)
    assert out["actuators"][20, 12].item() == pytest.approx(5, abs=0.001)
    assert out["actuators"][20, 13].item() == pytest.approx(1, abs=0.001)


def test_average_missing_donor_channel_retains_first_and_reports_skip(averaging_source):
    paths, _, donor = averaging_source
    with h5py.File(donor, "a") as f:
        f["pinj/ydata"][0, 1000:1050] = np.nan
    got = service().average_references(draft(comparison_shots=[SHOT + 1]), paths)
    proposal = got["program"]["proposal"]
    assert "pinj[0]" not in proposal["curves"]
    assert str(SHOT + 1) in proposal["skipped_channels"]["pinj[0]"]
    channel = next(c for c in got["channels"] if c["key"] == "pinj[0]")
    assert channel["vertices"][0]["y"] == 100
    assert channel["vertices"][1]["y"] == 200
    assert "gas_raw[1]" in proposal["skipped_channels"]
    assert got["validation"]["can_export"]


@pytest.mark.parametrize("kwargs", [
    {"comparison_shots": []},
    {"comparison_shots": [SHOT]},
    {"comparison_shots": [SHOT + 9]},
    {"comparison_shots": [SHOT + 1], "start_s": 0.5},
    {"comparison_shots": [SHOT + 1], "end_s": 5.01},
])
def test_average_rejects_invalid_sources_and_windows(averaging_source, kwargs):
    paths, _, _ = averaging_source
    with pytest.raises(service().ProgramValidationError):
        service().average_references(draft(**kwargs), paths)


@pytest.mark.parametrize("updates", [
    {"reference_shot": SHOT + 1},
    {"comparison_shots": [SHOT + 2]},
    {"start_s": 1.05},
    {"end_s": 4.95},
])
def test_average_snapshot_cannot_be_reused_for_other_sources_or_window(
    averaging_source, updates
):
    paths, _, _ = averaging_source
    merged = service().average_references(draft(comparison_shots=[SHOT + 1]), paths)
    program = service().DesignProgram(**{**merged["program"], **updates})
    got = service().preview(program, paths)
    assert not got["validation"]["can_export"]
    assert any("proposal" in error.lower() for error in got["validation"]["errors"])


def test_average_keeps_uncached_editing_and_source_drift_checks(averaging_source):
    paths, _, _ = averaging_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    got = service().average_references(draft(comparison_shots=[SHOT + 1]), paths)
    assert got["validation"]["can_save"]
    assert got["validation"]["needs_seed"]
    saved = service().save(service().DesignProgram(**got["program"]), paths)
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        f["pinj/ydata"][0, 0] += 1
    stale = service().load(saved["program"]["id"], paths)
    assert not stale["validation"]["can_save"]
    with pytest.raises(service().ProgramValidationError, match="changed"):
        service().average_references(service().DesignProgram(**got["program"]), paths)


def test_average_covers_all_88_native_channels_with_equal_three_shot_weights(
    averaging_source,
):
    paths, cache, _ = averaging_source
    groups = [
        ("ech_power", 12), ("pinj", 8), ("beam_voltage", 8), ("tinj", 8),
        ("gas_flow", 11), ("gas_raw", 11), ("rmp", 12), ("i_coil", 18),
    ]
    wave = np.repeat(np.tile([100.0, 200.0], 60), 50)
    wave = np.r_[wave, wave[-1]]
    for shot, factor in [(SHOT, 1), (SHOT + 1, 3), (SHOT + 2, 5)]:
        with h5py.File(
            paths.foundation_model_processed_dir / f"{shot}_processed.h5", "w"
        ) as f:
            offset = 0
            for group, width in groups:
                g = f.create_group(group)
                g["xdata"] = np.arange(6001) / 1000
                g["ydata"] = (
                    np.arange(offset + 1, offset + width + 1)[:, None] * wave * factor
                )
                offset += width
    cache["actuators"] = torch.tensor(build_actuators(
        SHOT, CorpusReader(paths.foundation_model_processed_dir), 120
    ).z.T, dtype=torch.float16)
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    got = service().average_references(
        draft(comparison_shots=[SHOT + 1, SHOT + 2]), paths
    )
    assert got["validation"]["can_export"]
    assert not got["program"]["proposal"]["skipped_channels"]
    assert len(got["program"]["proposal"]["curves"]) == 88
    offset = 0
    for group, width in groups:
        for member in range(width):
            points = got["program"]["proposal"]["curves"][f"{group}[{member}]"]
            assert points[0]["y"] == 300 * (offset + member + 1)
        offset += width
    out = torch.load(service().export_ignite(
        service().DesignProgram(**got["program"]), paths
    ), weights_only=True)
    assert out["actuators"][20].tolist() == pytest.approx([3] * 88, abs=0.001)


def test_average_aligns_absolute_times_with_shifted_prediction_and_donor(
    averaging_source,
):
    paths, cache, donor = averaging_source
    with h5py.File(donor, "a") as f:
        f["pinj/xdata"][:] -= 0.05
    got = service().average_references(
        draft(comparison_shots=[SHOT + 1], start_s=2.0, end_s=3.0), paths
    )
    assert got["validation"]["can_export"]
    points = got["program"]["proposal"]["curves"]["pinj[0]"]
    assert points[:2] == [{"t_s": 2.0, "y": 350}, {"t_s": 2.05, "y": 250}]
    out = torch.load(service().export_ignite(
        service().DesignProgram(**got["program"]), paths
    ), weights_only=True)
    assert torch.equal(out["actuators"][:20], cache["actuators"][20:40])
    assert out["actuators"][20:22, 12].tolist() == pytest.approx([4, 2], abs=0.001)


def test_average_retains_normalization_and_manual_overlap_guards(averaging_source):
    paths, cache, donor = averaging_source
    cache["actuators"][:, 12] += 1
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    with h5py.File(donor, "a") as f:
        f["ech_power/ydata"][0] *= 2
    got = service().average_references(draft(
        comparison_shots=[SHOT + 1],
        edits={"nbi.total": vertices(scale=36), "pinj[0]": vertices()},
    ), paths)
    errors = " ".join(got["validation"]["errors"])
    assert "normalization" in errors
    assert "zero-spread" in errors
    assert "overlap" in errors
    assert not got["validation"]["can_export"]


def test_average_skips_changed_negative_power_without_clipping(averaging_source):
    paths, _, donor = averaging_source
    with h5py.File(donor, "a") as f:
        f["pinj/ydata"][0, 1000:1050] = -300
    got = service().average_references(draft(comparison_shots=[SHOT + 1]), paths)
    assert got["validation"]["can_export"]
    proposal = got["program"]["proposal"]
    assert "pinj[0]" not in proposal["curves"]
    assert "negative" in proposal["skipped_channels"]["pinj[0]"]
    points = next(c for c in got["channels"] if c["key"] == "pinj[0]")["vertices"]
    assert points[:2] == [{"t_s": 1.0, "y": 100}, {"t_s": 1.05, "y": 200}]


def test_average_short_donor_retains_full_first_reference_channel(averaging_source):
    paths, _, donor = averaging_source
    with h5py.File(donor, "a") as f:
        times, values = f["pinj/xdata"][:3001], f["pinj/ydata"][:, :3001]
        del f["pinj"]
        group = f.create_group("pinj")
        group["xdata"], group["ydata"] = times, values
    got = service().average_references(draft(comparison_shots=[SHOT + 1]), paths)
    assert got["validation"]["can_export"]
    proposal = got["program"]["proposal"]
    assert "pinj[0]" not in proposal["curves"]
    assert str(SHOT + 1) in proposal["skipped_channels"]["pinj[0]"]
    assert "gas_raw[0]" in proposal["curves"]
    points = next(c for c in got["channels"] if c["key"] == "pinj[0]")["vertices"]
    assert [points[i]["y"] for i in (0, 39, 40, 79)] == [100, 200, 100, 200]


def test_reference_replay_and_edit_preserve_complete_reference_stats(program_source):
    paths, cache = program_source
    out = torch.load(service().export_ignite(draft(), paths), weights_only=True)
    assert set(out) == {"codes", "actuators", "vocabs", "n_frames"}
    assert torch.equal(out["actuators"], cache["actuators"][:100])
    for name in out["codes"]:
        assert torch.equal(out["codes"][name], cache["codes"][name][:100])
    edited = draft(edits={"pinj[0]": vertices()})
    out = torch.load(service().export_ignite(edited, paths), weights_only=True)
    assert torch.equal(out["actuators"][:20], cache["actuators"][:20])
    assert torch.equal(out["actuators"][:, 13:], cache["actuators"][:100, 13:])
    assert out["actuators"][20, 12].item() == pytest.approx(-0.6, abs=0.001)
    assert out["actuators"][21, 12].item() == pytest.approx(1.8, abs=0.001)


def test_shifted_window_slices_tokens_and_context_together(program_source):
    paths, cache = program_source
    out = torch.load(
        service().export_ignite(draft(start_s=2, end_s=6), paths), weights_only=True
    )
    assert torch.equal(out["actuators"], cache["actuators"][20:120])
    assert torch.equal(out["codes"]["ece"], cache["codes"]["ece"][20:120])


def test_total_preserves_member_ratio(program_source):
    paths, _ = program_source
    out = torch.load(
        service().export_ignite(draft(edits={"nbi.total": vertices(scale=36)}), paths),
        weights_only=True,
    )
    assert out["actuators"][20, 12:20].tolist() == pytest.approx([-0.6] * 8, abs=0.001)
    assert out["actuators"][21, 12:20].tolist() == pytest.approx([1.8] * 8, abs=0.001)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"start_s": 0.5}, "20"),
        ({"end_s": 5.01}, "50 ms"),
        ({"end_s": 6}, "80"),
        ({"start_s": 3, "end_s": 7}, "reference"),
        ({"edits": {"unknown": vertices()}}, "Unknown"),
        ({"edits": {"nbi.total": vertices(), "pinj[0]": vertices()}}, "overlap"),
        ({"edits": {"pinj[0]": [{"t_s": 2, "y": 100}]}}, "cover"),
        ({"edits": {"ech_power[0]": vertices()}}, "zero-spread"),
        ({"edits": {"i_coil[0]": vertices()}}, "missing"),
        ({"edits": {"gas_raw[1]": vertices()}}, "missing"),
        ({"edits": {"gas_raw[2]": vertices()}}, "missing"),
        ({"edits": {"pinj[0]": vertices(factor=1e10)}}, "float16"),
    ],
)
def test_invalid_drafts_are_visible_but_cannot_export(program_source, kwargs, message):
    paths, _ = program_source
    result = service().preview(draft(**kwargs), paths)
    assert not result["validation"]["can_export"]
    assert message.lower() in " ".join(result["validation"]["errors"]).lower()
    with pytest.raises(service().ProgramValidationError):
        service().export_ignite(draft(**kwargs), paths)


def test_missing_channels_are_not_displayed_as_measured_zero(program_source):
    paths, _ = program_source
    payload = service().preview(draft(), paths)
    channels = {c["key"]: c for c in payload["channels"]}
    assert len(channels) == 90
    assert channels["pinj[0]"]["editable"]
    assert channels["pinj[0]"]["units"] == "W"
    for key in ["gas_raw[1]", "gas_raw[2]", "i_coil[0]"]:
        assert not channels[key]["editable"]
        assert channels[key]["reference"] == []
        assert channels[key]["reason"]
    assert len(channels["pinj[0]"]["reference"]) == 100
    assert len(channels["pinj[0]"]["vertices"]) == 80


def test_save_immutable_roundtrip_and_digest_drift(program_source):
    paths, _ = program_source
    first = service().save(draft(notes="A draft", edits={"pinj[0]": vertices()}), paths)
    ident = first["program"]["id"]
    loaded = service().load(ident, paths)
    assert loaded == first
    second = service().save(service().DesignProgram(**first["program"]), paths)
    assert second["program"]["id"] != ident
    assert len(service().list_programs(paths)) == 2
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        f["pinj/ydata"][0, 0] = 99
    stale = service().load(ident, paths)
    assert not stale["validation"]["can_export"]
    assert "changed" in " ".join(stale["validation"]["errors"])
    with pytest.raises(service().ProgramValidationError):
        service().save(service().DesignProgram(**first["program"]), paths)


def test_missing_comparison_is_warning_and_reference_is_still_exportable(
    program_source,
):
    paths, _ = program_source
    payload = service().preview(draft(comparison_shots=[990099]), paths)
    assert payload["validation"]["can_export"]
    assert "990099" in " ".join(payload["validation"]["warnings"])


def test_bad_cache_normalization_blocks_only_edited_channel(program_source):
    paths, cache = program_source
    cache["actuators"][:, 12] += 1
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    assert service().preview(draft(), paths)["validation"]["can_export"]
    got = service().preview(draft(edits={"pinj[0]": vertices()}), paths)
    assert "normalization" in " ".join(got["validation"]["errors"])


def test_missing_seed_cache_is_actionable(program_source):
    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    got = service().preview(draft(), paths)
    assert not got["validation"]["can_export"]
    assert "cache" in " ".join(got["validation"]["errors"])


def test_uncached_reference_can_be_edited_saved_and_reopened(program_source):
    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    got = service().preview(draft(edits={"pinj[0]": vertices()}), paths)
    channel = next(c for c in got["channels"] if c["key"] == "pinj[0]")
    assert channel["editable"]
    assert channel["vertices"][0]["y"] == 120
    assert got["validation"]["can_save"]
    assert got["validation"]["needs_seed"]
    saved = service().save(service().DesignProgram(**got["program"]), paths)
    reopened = service().load(saved["program"]["id"], paths)
    assert reopened["program"] == saved["program"]
    with pytest.raises(service().ProgramValidationError, match="cache"):
        service().export_ignite(service().DesignProgram(**saved["program"]), paths)


def test_raw_draft_survives_seed_preparation_without_losing_edits(
    program_source, monkeypatch
):
    from shot_design.design import seed

    paths, cache = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    saved = service().save(draft(edits={"pinj[0]": vertices()}, notes="My plan"), paths)

    def encode(shot, *, reader, out_dir, paths, **kwargs):
        assert shot == SHOT
        assert reader.corpus_dir == paths.foundation_model_processed_dir
        out = out_dir / f"{shot}.pt"
        torch.save(cache, out)
        return out

    monkeypatch.setattr(seed, "encode_frame_codes", encode)
    prepared = service().prepare_ignite(service().DesignProgram(**saved["program"]), paths)
    assert prepared["validation"]["can_export"]
    assert not prepared["validation"]["needs_seed"]
    assert prepared["program"]["edits"] == saved["program"]["edits"]
    assert prepared["program"]["notes"] == "My plan"
    reopened = service().load(saved["program"]["id"], paths)
    assert reopened["validation"]["can_export"]
    out = torch.load(service().export_ignite(
        service().DesignProgram(**reopened["program"]), paths
    ), weights_only=True)
    assert out["actuators"][20, 12].item() == pytest.approx(-0.6, abs=0.001)
    assert torch.equal(out["actuators"][:20], cache["actuators"][:20])
    assert not (paths.data_root / "frame_codes" / f"{SHOT}.pt").exists()


def test_raw_source_changes_still_invalidate_drafts(program_source):
    paths, _ = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()
    saved = service().save(draft(), paths)
    with h5py.File(paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a") as f:
        f["pinj/ydata"][0, 0] += 1
    got = service().load(saved["program"]["id"], paths)
    assert not got["validation"]["can_save"]
    assert not got["validation"]["needs_seed"]
    assert "changed" in " ".join(got["validation"]["errors"])


def test_generated_seed_is_bound_to_the_corpus_it_encoded(program_source, monkeypatch):
    from shot_design.design import seed

    paths, cache = program_source
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").unlink()

    def encode(shot, *, out_dir, **kwargs):
        output = out_dir / f"{shot}.pt"
        torch.save(cache, output)
        return output

    monkeypatch.setattr(seed, "encode_frame_codes", encode)
    assert service().prepare_ignite(draft(), paths)["validation"]["can_export"]
    with h5py.File(paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a") as f:
        f["pinj/ydata"][0, 0] += 1
    fresh = service().preview(draft(), paths)
    assert fresh["channels"]
    assert fresh["validation"]["needs_seed"]
    assert not fresh["validation"]["can_export"]


def test_native_units_are_saved_and_wrong_edit_units_rejected(program_source):
    paths, _ = program_source
    got = service().save(draft(), paths)
    assert got["program"]["units"]["pinj[0]"] == "W"
    got = service().preview(
        draft(edits={"pinj[0]": vertices()}, units={"pinj[0]": "MW"}), paths
    )
    assert not got["validation"]["can_export"]
    assert "units" in " ".join(got["validation"]["errors"])


def test_total_cannot_turn_on_idle_reference_members(program_source):
    paths, cache = program_source
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        f["pinj/ydata"][:, 1000:1050] = 0
    cache["actuators"] = torch.tensor(
        build_actuators(
            SHOT, CorpusReader(paths.foundation_model_processed_dir), 120
        ).z.T,
        dtype=torch.float16,
    )
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    got = service().preview(draft(edits={"nbi.total": vertices(scale=36)}), paths)
    assert not got["validation"]["can_export"]
    assert "no active" in " ".join(got["validation"]["errors"])


@pytest.mark.parametrize(
    "kind",
    [
        "frames",
        "actuator_shape",
        "nonfinite",
        "missing_modality",
        "token_shape",
        "token_dtype",
        "vocab",
        "extra_key",
    ],
)
def test_corrupt_seed_caches_are_not_exported(program_source, kind):
    paths, cache = program_source
    if kind == "frames":
        cache["n_frames"] = 119
    elif kind == "actuator_shape":
        cache["actuators"] = cache["actuators"][:, :-1]
    elif kind == "nonfinite":
        cache["actuators"][0, 0] = float("nan")
    elif kind == "missing_modality":
        del cache["codes"]["ece"]
    elif kind == "token_shape":
        cache["codes"]["ece"] = cache["codes"]["ece"][:, :-1]
    elif kind == "token_dtype":
        cache["codes"]["ece"] = cache["codes"]["ece"].to(torch.int64)
    elif kind == "vocab":
        cache["vocabs"]["ece"] = 1
    else:
        cache["extra"] = 1
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    got = service().preview(draft(), paths)
    assert not got["validation"]["can_export"]
    assert "cache" in " ".join(got["validation"]["errors"]).lower()


def test_token_changes_invalidate_saved_reference(program_source):
    paths, cache = program_source
    saved = service().save(draft(), paths)
    cache["codes"]["ece"][0, 0] = 1
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    got = service().load(saved["program"]["id"], paths)
    assert "changed" in " ".join(got["validation"]["errors"])


def test_bundle_fallback_and_comparison_overlays(program_source):
    import shutil

    from shot_design.shotdb.ignite import bundle_dir

    paths, cache = program_source
    fallback = bundle_dir(paths) / "frame_codes"
    fallback.mkdir(parents=True)
    (paths.data_root / "frame_codes" / f"{SHOT}.pt").rename(fallback / f"{SHOT}.pt")
    torch.save(cache, fallback / f"{SHOT + 1}.pt")
    shutil.copyfile(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5",
        paths.foundation_model_processed_dir / f"{SHOT + 1}_processed.h5",
    )
    got = service().preview(draft(comparison_shots=[SHOT + 1]), paths)
    assert got["validation"]["can_export"]
    channel = next(c for c in got["channels"] if c["key"] == "pinj[0]")
    assert channel["comparisons"][0]["vertices"] == channel["reference"]


def test_related_controls_and_extrapolation_are_explicit(program_source):
    paths, _ = program_source
    got = service().preview(draft(edits={"pinj[0]": vertices(factor=5)}), paths)
    warnings = " ".join(got["validation"]["warnings"])
    assert "tinj" in warnings and "beam_voltage" in warnings
    assert "Edit pinj[0]" in warnings and "extrapolation" in warnings


def test_repeated_preview_uses_cached_reference_arrays(program_source, monkeypatch):
    paths, _ = program_source
    original = service().preview(draft(), paths)

    def unexpected_read(*args, **kwargs):
        raise AssertionError("Raw data reread on unchanged preview")

    monkeypatch.setattr(CorpusReader, "read", unexpected_read)
    assert service().preview(draft(), paths) == original


def test_extreme_finite_times_are_validation_errors(program_source):
    paths, _ = program_source
    got = service().preview(draft(start_s=1e308, end_s=1e308), paths)
    assert not got["validation"]["can_export"]


def test_extreme_member_edits_cannot_create_nonfinite_json(program_source):
    import json

    paths, _ = program_source
    edits = {
        f"pinj[{i}]": [{"t_s": 1, "y": 1e308}, {"t_s": 4.95, "y": 1e308}]
        for i in range(8)
    }
    got = service().preview(draft(edits=edits), paths)
    assert not got["validation"]["can_export"]
    json.dumps(got, allow_nan=False)


def test_comparisons_only_need_raw_waveforms_not_seed_tokens(program_source):
    import shutil

    paths, _ = program_source
    shutil.copyfile(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5",
        paths.foundation_model_processed_dir / f"{SHOT + 1}_processed.h5",
    )
    got = service().preview(draft(comparison_shots=[SHOT + 1]), paths)
    channel = next(c for c in got["channels"] if c["key"] == "pinj[0]")
    assert channel["comparisons"][0]["shot"] == SHOT + 1


def test_failed_export_preserves_existing_artifact(program_source, monkeypatch):
    paths, _ = program_source
    saved = service().save(draft(), paths)
    program = service().DesignProgram(**saved["program"])
    path = service().export_ignite(program, paths)
    before = path.read_bytes()

    def failed_write(value, stream):
        stream.write(b"incomplete")
        raise OSError("Simulated full disk")

    monkeypatch.setattr(torch, "save", failed_write)
    with pytest.raises(OSError, match="full disk"):
        service().export_ignite(program, paths)
    assert path.read_bytes() == before
    assert not list(path.parent.glob(".design-*"))


def test_symlink_and_traversal_revision_ids_are_rejected(program_source, tmp_path):
    paths, _ = program_source
    saved = service().save(draft(), paths)
    ident = saved["program"]["id"]
    path = paths.actuations_dir / "designs" / f"{ident}.json"
    external = tmp_path / "outside.json"
    path.rename(external)
    path.symlink_to(external)
    for candidate in [ident, "../outside", "/tmp/outside", "a" * 31]:
        with pytest.raises(FileNotFoundError):
            service().load(candidate, paths)
    assert service().list_programs(paths) == []


def test_provisional_limits_distinguish_baseline_and_edit(program_source, monkeypatch):
    paths, _ = program_source
    original = service().load_yaml

    def limited(name):
        cfg = original(name)
        if name == "actuators.yaml":
            cfg["systems"]["nbi"]["max"] = 180
        return cfg

    monkeypatch.setattr(service(), "load_yaml", limited)
    base = service().preview(draft(), paths)
    assert any("Baseline pinj[0]" in w for w in base["validation"]["warnings"])
    assert not any("Edit pinj[0]" in w for w in base["validation"]["warnings"])
    changed = service().preview(draft(edits={"pinj[0]": vertices()}), paths)
    assert any(
        "Edit pinj[0]" in w and "provisional" in w
        for w in changed["validation"]["warnings"]
    )


def test_five_comparisons_do_not_reread_raws_on_every_preview(
    program_source, monkeypatch
):
    import shutil

    paths, _ = program_source
    for shot in range(SHOT + 1, SHOT + 6):
        shutil.copyfile(
            paths.foundation_model_processed_dir / f"{SHOT}_processed.h5",
            paths.foundation_model_processed_dir / f"{shot}_processed.h5",
        )
    program = draft(comparison_shots=list(range(SHOT + 1, SHOT + 6)))
    first = service().preview(program, paths)

    def no_read(*args, **kwargs):
        raise AssertionError(
            "Comparison arrays were evicted between identical previews"
        )

    monkeypatch.setattr(CorpusReader, "read", no_read)
    assert service().preview(program, paths) == first


def test_rmp_subset_inherits_configured_coil_cap(program_source, monkeypatch):
    paths, cache = program_source
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        group = f.create_group("rmp")
        group["xdata"] = f["pinj/xdata"][()]
        # This channel crosses 180 A already; the second crosses only after edit.
        low_high = f["pinj/ydata"][0]
        group["ydata"] = np.vstack([low_high, low_high / 2] + [low_high] * 10)
    cache["actuators"] = torch.tensor(
        build_actuators(
            SHOT, CorpusReader(paths.foundation_model_processed_dir), 120
        ).z.T,
        dtype=torch.float16,
    )
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    original = service().load_yaml

    def limited(name):
        cfg = original(name)
        if name == "actuators.yaml":
            cfg["systems"]["coil_rmp"]["max"] = 180
        return cfg

    monkeypatch.setattr(service(), "load_yaml", limited)
    baseline = service().preview(draft(), paths)
    assert any(
        "Baseline rmp[0]" in w and "provisional" in w
        for w in baseline["validation"]["warnings"]
    )
    assert not any("Baseline rmp[1]" in w for w in baseline["validation"]["warnings"])
    edited = service().preview(draft(edits={"rmp[1]": vertices(factor=1.2)}), paths)
    assert edited["validation"]["can_export"]
    assert any(
        "Edit rmp[1]" in w and "provisional" in w
        for w in edited["validation"]["warnings"]
    )


@pytest.mark.parametrize("timestamp", [np.inf, np.nan, -np.inf])
def test_nonfinite_trailing_pad_time_is_actionable(program_source, timestamp):
    paths, _ = program_source
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        f["pinj/ydata"][:, -1] = np.nan
        f["pinj/xdata"][-1] = timestamp
    got = service().preview(draft(), paths)
    assert not got["validation"]["can_export"]
    assert "pinj" in " ".join(got["validation"]["errors"])
    assert "time" in " ".join(got["validation"]["errors"])


def test_comparison_with_nonfinite_pad_time_is_only_a_warning(program_source):
    import shutil

    paths, _ = program_source
    comparison = paths.foundation_model_processed_dir / f"{SHOT + 1}_processed.h5"
    shutil.copyfile(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", comparison
    )
    with h5py.File(comparison, "a") as f:
        f["pinj/ydata"][:, -1] = np.nan
        f["pinj/xdata"][-1] = np.inf
    got = service().preview(draft(comparison_shots=[SHOT + 1]), paths)
    assert got["validation"]["can_export"]
    assert any(
        str(SHOT + 1) in w and "time" in w for w in got["validation"]["warnings"]
    )


@pytest.mark.parametrize("key", ["pinj[0]", "ech_power[0]", "nbi.total", "ech.total"])
def test_changed_negative_power_demand_cannot_export(program_source, key):
    paths, _ = program_source
    got = service().preview(
        draft(edits={key: [{"t_s": 1, "y": -100}, {"t_s": 4.95, "y": -100}]}), paths
    )
    assert not got["validation"]["can_export"]
    assert any(
        key in error and "nonnegative" in error for error in got["validation"]["errors"]
    )
    with pytest.raises(service().ProgramValidationError):
        service().export_ignite(
            draft(edits={key: [{"t_s": 1, "y": -100}, {"t_s": 4.95, "y": -100}]}), paths
        )


@pytest.mark.parametrize("key", ["pinj[0]", "nbi.total"])
def test_unchanged_negative_baseline_power_does_not_block_other_edits(
    program_source, key
):
    paths, cache = program_source
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        f["pinj/ydata"][:, 1000:1050] = -1
        f["pinj/ydata"][:, 1150:1200] = -1
    cache["actuators"] = torch.tensor(
        build_actuators(
            SHOT, CorpusReader(paths.foundation_model_processed_dir), 120
        ).z.T,
        dtype=torch.float16,
    )
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    baseline = service().preview(draft(), paths)
    assert baseline["validation"]["can_export"]
    channel = next(c for c in baseline["channels"] if c["key"] == key)
    points = channel["vertices"]
    assert points[0]["y"] < 0
    points[1]["y"] *= 1.2
    got = service().preview(draft(edits={key: points}), paths)
    assert got["validation"]["can_export"]
    out = torch.load(
        service().export_ignite(draft(edits={key: points}), paths), weights_only=True
    )
    assert torch.equal(out["actuators"][20], cache["actuators"][20])
    assert torch.equal(out["actuators"][23], cache["actuators"][23])


@pytest.mark.parametrize(
    ("key", "group", "width"),
    [
        ("nbi.total", "pinj", 8),
        ("ech.total", "ech_power", 12),
    ],
)
def test_total_split_cannot_change_negative_member_noise(
    program_source, key, group, width
):
    paths, cache = program_source
    with h5py.File(
        paths.foundation_model_processed_dir / f"{SHOT}_processed.h5", "a"
    ) as f:
        # Every member has spread, so normalization cannot mask the power guard.
        f[f"{group}/ydata"][:] = np.arange(1, width + 1)[:, None] * f["pinj/ydata"][0]
        f[f"{group}/ydata"][0, 1000:1050] = -100
    cache["actuators"] = torch.tensor(
        build_actuators(
            SHOT, CorpusReader(paths.foundation_model_processed_dir), 120
        ).z.T,
        dtype=torch.float16,
    )
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    baseline = service().preview(draft(), paths)
    assert baseline["validation"]["can_export"]
    points = next(c for c in baseline["channels"] if c["key"] == key)["vertices"]
    assert points[0]["y"] > 0
    # An edit at a different frame leaves the historical negative member intact.
    points[1]["y"] *= 1.2
    unchanged_noise = draft(edits={key: points})
    assert service().preview(unchanged_noise, paths)["validation"]["can_export"]
    out = torch.load(service().export_ignite(unchanged_noise, paths), weights_only=True)
    assert torch.equal(out["actuators"][20], cache["actuators"][20])
    # This positive total would turn the same -100 W member into -120 W.
    points[0]["y"] *= 1.2
    changed_noise = draft(edits={key: points})
    got = service().preview(changed_noise, paths)
    assert not got["validation"]["can_export"]
    assert any(
        f"{group}[0]" in error and "nonnegative" in error
        for error in got["validation"]["errors"]
    )
    with pytest.raises(service().ProgramValidationError):
        service().export_ignite(changed_noise, paths)


@pytest.mark.parametrize(
    ("modality", "wrong_vocab"),
    [
        ("ece", 16),
        ("ece", 1000),
        ("bes", 32768),
        ("cer_ti", 64000),
    ],
)
def test_wrong_codec_generation_refuses_export_even_with_in_range_tokens(
    program_source,
    modality,
    wrong_vocab,
):
    paths, cache = program_source
    cache["vocabs"][modality] = wrong_vocab
    assert cache["codes"][modality].max() < wrong_vocab
    torch.save(cache, paths.data_root / "frame_codes" / f"{SHOT}.pt")
    got = service().preview(draft(), paths)
    assert not got["validation"]["can_export"]
    assert any(
        modality in error and "vocabulary" in error and "production" in error
        for error in got["validation"]["errors"]
    )
    with pytest.raises(service().ProgramValidationError):
        service().export_ignite(draft(), paths)
