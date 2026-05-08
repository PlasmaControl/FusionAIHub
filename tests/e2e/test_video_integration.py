"""Step 5 guard tests for E2E foundation-model integration of the video
modality.

Five tests pin the contracts the user explicitly flagged as
regression-risk in ``docs/phase_c_step1_status.md`` §12:

* **G1** — when a ``kind="video"`` diagnostic is added, every video
  ``TokenSlice`` must lie inside the diagnostic prefix
  (``slice.stop <= model.n_diag_tokens``) so ``rollout.py:149`` sees
  it.
* **G2** — the model built from the fixture's TS-only diagnostics
  list has *exactly* the set of ``state_dict()`` keys captured before
  any Step-5 edit. Catches accidental renames / new TS keys.
* **G3** — same TS-only model, fed the saved input, reproduces the
  saved output **byte-for-byte**. Catches silent perturbations of
  the TS forward path.
* **G4** — a TS-only checkpoint loads cleanly into a model that also
  has a video diagnostic; only ``diag_tokenizers.tangtv.*`` /
  ``diag_heads.tangtv.*`` are reported missing, nothing unexpected.
* **G5** — an unexpected key in the loaded state must raise; the new
  loader is not allowed to silently drop renamed TS keys.

G2 and G3 should pass on the *current* (pre-Step-5) code as a
sanity check that the fixture is consistent with the live tree. G1,
G4, G5 require Step-5 features and are skipped until those land.
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


# ── Step-5 capability probes ────────────────────────────────────────────


def _video_kind_supported() -> bool:
    """``E2EFoundationModel.__init__`` accepts ``kind="video"``."""
    cfg = DiagnosticConfig(
        name="x", kind="video", n_channels=1, window_samples=1,
        height=1, width=1, video_patch_size=(1, 1, 1),
    )
    try:
        cfg.n_tokens()
    except ValueError:
        return False
    return True


def _explicit_loader_available() -> bool:
    """A factored ``load_state_dict_explicit`` exists in the e2e package."""
    try:
        from tokamak_foundation_model.e2e import (  # noqa: F401
            checkpoint as _ckpt,
        )
        return hasattr(_ckpt, "load_state_dict_explicit")
    except ImportError:
        return False


VIDEO_SUPPORTED = _video_kind_supported()
LOADER_AVAILABLE = _explicit_loader_available()


# ── Fixture loading ─────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE_PATH.exists():
        pytest.skip(
            f"Fixture {FIXTURE_PATH} not present — run "
            "`pixi run python scripts/capture_no_video_fixture.py` to create it."
        )
    return torch.load(FIXTURE_PATH, weights_only=False)


def _build_no_video_model_from_fixture(fixture) -> E2EFoundationModel:
    """Recreate the exact TS-only model that produced the fixture."""
    cfg = fixture["config"]
    torch.manual_seed(fixture["seed"])
    diags = [DiagnosticConfig(**d) for d in cfg["diagnostics"]]
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


# ── G2 — state_dict keys identical ──────────────────────────────────────


def test_no_video_state_dict_keys_identical(fixture):
    """The TS-only model's state_dict keys must match the fixture exactly.

    A diff here means someone renamed / added / removed a TS key
    without regenerating the fixture deliberately. See the
    "WHEN TO REGENERATE" comment at the top of
    ``scripts/capture_no_video_fixture.py``.
    """
    model = _build_no_video_model_from_fixture(fixture)
    live_keys = sorted(model.state_dict().keys())
    saved_keys = list(fixture["state_dict_keys"])
    extra = sorted(set(live_keys) - set(saved_keys))
    missing = sorted(set(saved_keys) - set(live_keys))
    assert not extra, f"unexpected new keys in state_dict: {extra}"
    assert not missing, f"keys disappeared from state_dict: {missing}"
    assert live_keys == saved_keys, (
        "state_dict key order changed (might break older checkpoints)"
    )


# ── G3 — forward output bitwise identical ───────────────────────────────


def test_no_video_forward_bitwise_identical(fixture):
    """Same model, same input → byte-identical output as captured."""
    model = _build_no_video_model_from_fixture(fixture).eval()
    inp = fixture["input"]
    saved_output = fixture["output"]

    with torch.no_grad():
        live_output = model(
            inp["diag_inputs"],
            inp["act_inputs"],
            inp["step_index"],
            inp["time_offset_s"],
        )

    assert set(live_output.keys()) == set(saved_output.keys())
    for name, saved_t in saved_output.items():
        live_t = live_output[name]
        assert live_t.shape == saved_t.shape, (
            f"{name}: shape changed {tuple(saved_t.shape)} -> {tuple(live_t.shape)}"
        )
        assert torch.equal(live_t, saved_t), (
            f"{name}: forward output drifted from fixture; "
            "TS forward path was perturbed."
        )


# ── G1 — video tokens live in the diagnostic prefix ────────────────────


@pytest.mark.skipif(
    not VIDEO_SUPPORTED,
    reason="Step 5 not yet implemented: DiagnosticConfig.kind='video' unsupported",
)
def test_video_tokens_in_diagnostic_prefix(fixture):
    """Every video TokenSlice must satisfy slice.stop <= n_diag_tokens.

    The rollout code at ``rollout.py:149`` propagates diagnostic tokens
    via a contiguous slice ``[:, :n_diag_tokens]``. Video tokens must
    sit inside that prefix.
    """
    cfg = fixture["config"]
    diags = [DiagnosticConfig(**d) for d in cfg["diagnostics"]]
    diags.append(
        DiagnosticConfig(
            name="tangtv", kind="video",
            n_channels=2, window_samples=3,
            height=120, width=360, video_patch_size=(3, 12, 12),
        )
    )
    acts = [ActuatorConfig(**a) for a in cfg["actuators"]]
    model = E2EFoundationModel(
        diagnostics=diags,
        actuators=acts,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
    )

    video_slices = [
        s for s in model.token_layout if s.name == "tangtv"
    ]
    assert video_slices, "no TokenSlice for tangtv"
    for s in video_slices:
        assert s.is_diagnostic, "tangtv slice must be flagged is_diagnostic"
        assert s.slice_.stop <= model.n_diag_tokens, (
            f"tangtv tokens at {s.slice_} fall outside the diagnostic "
            f"prefix [:n_diag_tokens={model.n_diag_tokens}]"
        )


# ── G4 — old TS-only checkpoint loads cleanly into a TS+video model ────


@pytest.mark.skipif(
    not VIDEO_SUPPORTED,
    reason="Step 5 not yet implemented: DiagnosticConfig.kind='video' unsupported",
)
@pytest.mark.skipif(
    not LOADER_AVAILABLE,
    reason="Step 5 not yet implemented: load_state_dict_explicit missing",
)
def test_load_old_checkpoint_into_video_model_succeeds(fixture):
    """TS-only state -> TS+video model: only tangtv keys are missing,
    nothing unexpected.
    """
    from tokamak_foundation_model.e2e.checkpoint import (
        load_state_dict_explicit,
    )

    cfg = fixture["config"]
    ts_only = _build_no_video_model_from_fixture(fixture)
    saved_state = ts_only.state_dict()

    diags = [DiagnosticConfig(**d) for d in cfg["diagnostics"]]
    diags.append(
        DiagnosticConfig(
            name="tangtv", kind="video",
            n_channels=2, window_samples=3,
            height=120, width=360, video_patch_size=(3, 12, 12),
        )
    )
    acts = [ActuatorConfig(**a) for a in cfg["actuators"]]
    with_video = E2EFoundationModel(
        diagnostics=diags,
        actuators=acts,
        d_model=cfg["d_model"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        mlp_ratio=cfg["mlp_ratio"],
        dropout=cfg["dropout"],
    )

    # Should NOT raise — only tangtv keys missing, nothing unexpected.
    load_state_dict_explicit(
        with_video,
        saved_state,
        allowed_missing_prefixes=(
            "diag_tokenizers.tangtv.",
            "diag_heads.tangtv.",
        ),
    )


# ── G5 — unexpected key in state must raise ────────────────────────────


@pytest.mark.skipif(
    not LOADER_AVAILABLE,
    reason="Step 5 not yet implemented: load_state_dict_explicit missing",
)
def test_load_with_unexpected_key_raises(fixture):
    """A renamed / extra key must trip the explicit loader.

    If we tolerate unexpected keys we can't catch silent renames in
    the TS path during a Phase C edit.
    """
    from tokamak_foundation_model.e2e.checkpoint import (
        load_state_dict_explicit,
    )

    model = _build_no_video_model_from_fixture(fixture)
    state = model.state_dict()
    # Inject an unexpected key.
    state["this_key_does_not_exist_in_the_model"] = torch.tensor(0.0)

    with pytest.raises(RuntimeError, match=r"[Uu]nexpected"):
        load_state_dict_explicit(
            model, state, allowed_missing_prefixes=()
        )