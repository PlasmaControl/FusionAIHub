"""Step 5 guard tests for E2E foundation-model integration of the
spectrogram modality.

Mirrors the Phase C ``test_video_integration.py`` G1/G4-style checks for
the spectrogram path:

* **S1** — when a ``kind="spectrogram"`` diagnostic is added, every
  spectrogram ``TokenSlice`` must lie inside the diagnostic prefix
  (``slice.stop <= model.n_diag_tokens``) so ``rollout.py:149`` sees
  it.
* **S2** — spectrogram tokens come **before** video tokens in the
  layout when both are present, matching the
  ``[slow_ts | fast_ts | spectro | video | actuators]`` ordering set
  by ``train_e2e_stage1.build_configs``. Adding either modality must
  not perturb the other's slice.
* **S3** — a TS-only state_dict loads cleanly into a TS+spectro model;
  only ``diag_tokenizers.{spec}.*`` and ``diag_heads.{spec}.*`` are
  reported missing, nothing unexpected.

The G2/G3 byte-identity guards (TS-only path unchanged when
``--use_spectro`` is empty) are already covered by
``test_video_integration.py::test_no_video_*`` — adding the
spectrogram code path doesn't run unless ``use_spectro`` is non-empty,
so the same fixture continues to pin the TS-only state_dict and
forward output.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from tokamak_foundation_model.e2e.model import (
    ActuatorConfig,
    DiagnosticConfig,
    E2EFoundationModel,
)


FIXTURE_PATH = Path(__file__).parent / "fixtures" / "no_video_forward.pt"


# ── Step-5 capability probe ─────────────────────────────────────────────


def _spectro_kind_supported() -> bool:
    """``E2EFoundationModel.__init__`` accepts ``kind="spectrogram"``."""
    cfg = DiagnosticConfig(
        name="x", kind="spectrogram", n_channels=1, window_samples=8,
        freq_bins=8, spectrogram_patch_size=(4, 4),
    )
    try:
        cfg.n_tokens()
    except ValueError:
        return False
    return True


def _explicit_loader_available() -> bool:
    try:
        from tokamak_foundation_model.e2e import (  # noqa: F401
            checkpoint as _ckpt,
        )
        return hasattr(_ckpt, "load_state_dict_explicit")
    except ImportError:
        return False


SPECTRO_SUPPORTED = _spectro_kind_supported()
LOADER_AVAILABLE = _explicit_loader_available()


# Plan-locked spectrogram defaults.
SPECTRO_FREQ_BINS = 512
SPECTRO_TIME_FRAMES = 98
# (name, n_channels, (F_p, T_p))
SPECTRO_CONFIGS: list[tuple[str, int, tuple[int, int]]] = [
    ("ece", 40, (32, 8)),
    ("co2", 4, (64, 8)),
    ("bes", 16, (32, 8)),
]


# ── Fixture loading ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture {FIXTURE_PATH} not present — run "
            "`pixi run python scripts/capture_no_video_fixture.py` "
            "to create it."
        )
    return torch.load(FIXTURE_PATH, weights_only=False)


def _ts_diags_from_fixture(fixture) -> list[DiagnosticConfig]:
    return [DiagnosticConfig(**d) for d in fixture["config"]["diagnostics"]]


def _build_with_spectro(
    fixture, names: list[str], with_video: bool = False,
) -> E2EFoundationModel:
    cfg = fixture["config"]
    torch.manual_seed(fixture["seed"])
    diags = _ts_diags_from_fixture(fixture)
    by_name = {n: (n, c, p) for n, c, p in SPECTRO_CONFIGS}
    for name in names:
        n_ch, p = by_name[name][1], by_name[name][2]
        diags.append(
            DiagnosticConfig(
                name=name, kind="spectrogram",
                n_channels=n_ch, window_samples=SPECTRO_TIME_FRAMES,
                freq_bins=SPECTRO_FREQ_BINS,
                spectrogram_patch_size=p,
            )
        )
    if with_video:
        diags.append(
            DiagnosticConfig(
                name="tangtv", kind="video",
                n_channels=2, window_samples=3,
                height=120, width=360, video_patch_size=(3, 12, 12),
            )
        )
    acts = [ActuatorConfig(**a) for a in cfg["actuators"]]
    return E2EFoundationModel(
        diagnostics=diags,
        actuators=acts,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
    )


# ── S1 — spectrogram tokens live in the diagnostic prefix ──────────────


@pytest.mark.skipif(
    not SPECTRO_SUPPORTED,
    reason="DiagnosticConfig.kind='spectrogram' unsupported",
)
@pytest.mark.parametrize("name", ["ece", "co2", "bes"])
def test_spectrogram_tokens_in_diagnostic_prefix(fixture, name):
    """Every spectrogram TokenSlice must satisfy
    ``slice.stop <= n_diag_tokens`` so rollout's contiguous
    diagnostic prefix slice picks it up.
    """
    model = _build_with_spectro(fixture, [name])
    spec_slices = [s for s in model.token_layout if s.name == name]
    assert spec_slices, f"no TokenSlice for {name}"
    for s in spec_slices:
        assert s.is_diagnostic, f"{name} slice must be flagged is_diagnostic"
        assert s.slice_.stop <= model.n_diag_tokens, (
            f"{name} tokens at {s.slice_} fall outside the diagnostic "
            f"prefix [:n_diag_tokens={model.n_diag_tokens}]"
        )


# ── S2 — token ordering: TS | spectro | video | actuators ──────────────


@pytest.mark.skipif(
    not SPECTRO_SUPPORTED,
    reason="DiagnosticConfig.kind='spectrogram' unsupported",
)
def test_layout_order_ts_then_spectro_then_video(fixture):
    """When TS, spectro, and video coexist, the diagnostic-prefix
    layout is ``[ts... | spectro... | video...]`` followed by actuator
    slices. Each spectro slice must precede the tangtv slice.
    """
    model = _build_with_spectro(
        fixture, names=["ece", "co2", "bes"], with_video=True,
    )
    diag_slices = [s for s in model.token_layout if s.is_diagnostic]
    by_kind = {}
    for cfg in model.diagnostics:
        by_kind[cfg.name] = cfg.kind
    # Build the (start, kind, name) ordering.
    ordered = [(s.slice_.start, by_kind[s.name], s.name) for s in diag_slices]
    ordered.sort()  # by start
    # Find the kind sequence; must be all ts, then all spectro, then all video
    seen_spectro = False
    seen_video = False
    for _, kind, name in ordered:
        if kind in ("slow_ts", "fast_ts"):
            assert not seen_spectro and not seen_video, (
                f"TS modality {name!r} appears after spectro/video"
            )
        elif kind == "spectrogram":
            seen_spectro = True
            assert not seen_video, (
                f"spectro {name!r} appears after a video modality"
            )
        elif kind == "video":
            seen_video = True


# ── S3 — TS-only checkpoint loads cleanly into TS+spectro ──────────────


@pytest.mark.skipif(
    not SPECTRO_SUPPORTED,
    reason="DiagnosticConfig.kind='spectrogram' unsupported",
)
@pytest.mark.skipif(
    not LOADER_AVAILABLE,
    reason="load_state_dict_explicit missing",
)
@pytest.mark.parametrize("names", [["ece"], ["ece", "co2", "bes"]])
def test_load_old_checkpoint_into_spectro_model_succeeds(fixture, names):
    """TS-only state -> TS+spectrogram model: only spectrogram keys are
    missing, nothing unexpected. Same contract Phase C uses for video.
    """
    from tokamak_foundation_model.e2e.checkpoint import (
        load_state_dict_explicit,
    )

    # Save the TS-only state_dict from a freshly-built TS-only model so
    # the test doesn't depend on the live fixture file containing
    # weights (the fixture currently records *keys* + a saved forward
    # output; that's enough since the loader checks key contracts).
    cfg = fixture["config"]
    torch.manual_seed(fixture["seed"])
    ts_only = E2EFoundationModel(
        diagnostics=_ts_diags_from_fixture(fixture),
        actuators=[ActuatorConfig(**a) for a in cfg["actuators"]],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
    )
    saved_state = ts_only.state_dict()

    with_spectro = _build_with_spectro(fixture, names)
    allowed = tuple(
        f"{prefix}{name}." for prefix in (
            "diag_tokenizers.", "diag_heads.",
        )
        for name in names
    )
    # Should NOT raise.
    load_state_dict_explicit(
        with_spectro, saved_state, allowed_missing_prefixes=allowed,
    )


# ── S4 — Stage 2 trainer split helper ─────────────────────────────────


def test_split_spectro_target_by_step_shapes():
    """``split_spectro_target_by_step`` returns K windows of exactly
    ``trunc_t`` frames each. ``trunc_t`` must match
    ``SpectrogramTokenizer.trunc_t`` (= ``window_samples // T_p * T_p``)
    so the per-step target shape lines up with the head's recon shape.
    """
    from scripts.training.train_e2e_stage2_delta import (
        split_spectro_target_by_step,
    )
    # Realistic STFT target shape: (B, C, F, ~977 frames for K=10).
    # trunc_t=96 mirrors the standard window_samples=98, T_p=8 config.
    target = torch.randn(2, 4, 512, 977)
    windows = split_spectro_target_by_step(target, k_steps=10, trunc_t=96)
    assert len(windows) == 10
    for w in windows:
        assert w.shape == (2, 4, 512, 96)


def test_split_spectro_target_by_step_raises_when_too_short():
    """Target shorter than ``K * trunc_t`` raises — silently truncating
    to fewer than K windows would mismatch the rollout's K-step loop."""
    from scripts.training.train_e2e_stage2_delta import (
        split_spectro_target_by_step,
    )
    target = torch.randn(1, 1, 512, 100)         # K * trunc_t = 960 > 100
    with pytest.raises(ValueError, match="K \\* trunc_t"):
        split_spectro_target_by_step(target, k_steps=10, trunc_t=96)


# ── S5 — Stage 1 forward_batch end-to-end shape contract ──────────────


@pytest.mark.skipif(
    not SPECTRO_SUPPORTED,
    reason="DiagnosticConfig.kind='spectrogram' unsupported",
)
def test_stage1_forward_batch_with_spectrogram_loss_is_finite(fixture):
    """End-to-end shape contract for the Stage 1 trainer's spectrogram
    branch. Catches the regression where the dataloader's
    98-frame spectrogram target was passed un-truncated against the
    head's 96-frame reconstruction (broadcast error in masked MAE).

    Constructs a TS+spectro model + a synthetic batch mimicking the
    dataloader contract, then calls ``forward_batch`` and
    ``compute_step_loss``. Loss must be finite with non-trivial
    gradient pathways. We stub ``cer_ti`` channel masks for the
    masked-MAE path and gate spectro presence on.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "train_e2e_stage1", "scripts/training/train_e2e_stage1.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)

    cfg = fixture["config"]
    diags = _ts_diags_from_fixture(fixture)
    diags.append(
        DiagnosticConfig(
            name="ece", kind="spectrogram",
            n_channels=40, window_samples=SPECTRO_TIME_FRAMES,
            freq_bins=SPECTRO_FREQ_BINS,
            spectrogram_patch_size=(32, 8),
        )
    )
    acts = [ActuatorConfig(**a) for a in cfg["actuators"]]
    torch.manual_seed(fixture["seed"])
    model = E2EFoundationModel(
        diagnostics=diags, actuators=acts,
        d_model=cfg["d_model"], n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"], mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
    )

    B = 2
    batch = {"inputs": {}, "targets": {}}
    for d_cfg in diags:
        if d_cfg.kind == "slow_ts":
            x = torch.randn(B, d_cfg.n_channels, d_cfg.window_samples)
            batch["inputs"][d_cfg.name] = x
            batch["targets"][d_cfg.name] = torch.randn_like(x)
        elif d_cfg.kind == "fast_ts":
            x = torch.randn(B, d_cfg.n_channels, d_cfg.window_samples)
            batch["inputs"][d_cfg.name] = x
            batch["targets"][d_cfg.name] = torch.randn_like(x)
        elif d_cfg.kind == "spectrogram":
            x = torch.randn(
                B, d_cfg.n_channels, d_cfg.freq_bins, d_cfg.window_samples,
            )
            batch["inputs"][d_cfg.name] = x
            batch["targets"][d_cfg.name] = torch.randn_like(x)
            batch["inputs"][f"{d_cfg.name}_valid"] = torch.tensor([1, 1])
            batch["targets"][f"{d_cfg.name}_valid"] = torch.tensor([1, 1])
    for a_cfg in acts:
        batch["targets"][a_cfg.name] = torch.randn(
            B, a_cfg.n_channels, a_cfg.window_samples,
        )

    loss, per_modality = m.compute_step_loss(model, batch, torch.device("cpu"))
    assert torch.isfinite(loss).item(), f"loss={loss.item()} not finite"
    assert "ece" in per_modality, "spectrogram modality missing from loss dict"
    assert per_modality["ece"] == per_modality["ece"]   # not NaN
    loss.backward()


@pytest.mark.skipif(
    not SPECTRO_SUPPORTED,
    reason="DiagnosticConfig.kind='spectrogram' unsupported",
)
@pytest.mark.skipif(
    not LOADER_AVAILABLE,
    reason="load_state_dict_explicit missing",
)
def test_loader_rejects_missing_spectrogram_when_not_allowed(fixture):
    """If we add spectrograms but forget to declare their prefixes in
    ``allowed_missing_prefixes``, the explicit loader must raise — same
    safety contract video has.
    """
    from tokamak_foundation_model.e2e.checkpoint import (
        load_state_dict_explicit,
    )

    cfg = fixture["config"]
    torch.manual_seed(fixture["seed"])
    ts_only = E2EFoundationModel(
        diagnostics=_ts_diags_from_fixture(fixture),
        actuators=[ActuatorConfig(**a) for a in cfg["actuators"]],
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
    )
    saved_state = ts_only.state_dict()

    with_spectro = _build_with_spectro(fixture, ["ece"])
    with pytest.raises(RuntimeError, match=r"[Mm]issing"):
        load_state_dict_explicit(
            with_spectro, saved_state, allowed_missing_prefixes=(),
        )
