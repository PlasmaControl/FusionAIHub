"""
Unit tests for dynamics rollout health.

Catches architectural issues (fixed-point attractors, actuator
insensitivity, gradient vanishing, state independence) using random
tensors — no data or training required.

Run with:
    pixi run pytest tests/test_dynamics_rollout.py -v
"""

import pytest
import torch
import torch.nn.functional as F

from tokamak_foundation_model.models.latent_feature_space.foundation_model import (
    PerceiverFoundationModel,
)
from tokamak_foundation_model.models.latent_feature_space.perceiver_components import (
    _DynamicsCrossAttentionBlock,
    CrossAttentionDynamics,
)

ACTUATOR_CONFIGS = {
    "pin": {"target_fs": 10000, "n_channels": 8, "patch_len": 200},
    "tin": {"target_fs": 10000, "n_channels": 8, "patch_len": 200},
    "beam_voltage": {"target_fs": 10000, "n_channels": 8, "patch_len": 200},
    "ech_power": {"target_fs": 10000, "n_channels": 4, "patch_len": 200,
                  "channels_to_use": [5, 7, 8, 10]},
    "gas_flow": {"target_fs": 10000, "n_channels": 7, "patch_len": 200,
                 "channels_to_use": [0, 1, 2, 3, 4, 6, 7]},
    "rmp": {"target_fs": 10000, "n_channels": 11, "patch_len": 200,
            "channels_to_use": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
}

MOD_CONFIGS = {
    "ts_core_temp": {"d_lat": 32, "n_tokens": 16},
    "mse": {"d_lat": 32, "n_tokens": 16},
}

D_MODEL = 64
N_LATENT = 16
N_HEADS = 4
N_STEPS = 8


def _make_model():
    return PerceiverFoundationModel(
        modality_configs=MOD_CONFIGS,
        d_model=D_MODEL,
        n_latent=N_LATENT,
        encoder_layers=1,
        processor_layers=1,
        decoder_layers=1,
        dynamics_layers=1,
        n_heads=N_HEADS,
        dropout=0.0,
        dynamics_type="cross_attention",
        actuator_configs=ACTUATOR_CONFIGS,
        ema_decay=0.996,
    )


def _random_ae_latents(B=2):
    return {name: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
            for name, cfg in MOD_CONFIGS.items()}


def _random_actuators(B=2):
    return {name: torch.randn(
                B,
                len(acfg.get("channels_to_use", range(acfg["n_channels"]))),
                5000)
            for name, acfg in ACTUATOR_CONFIGS.items()}


def _run_rollout(model, B=2, n_steps=N_STEPS):
    """Run a rollout and return latents and deltas at each step."""
    lat_ctx = _random_ae_latents(B)
    act_ctx = _random_actuators(B)
    act = _random_actuators(B)

    latent = model.encode(lat_ctx, act_ctx)
    latents = [latent]
    deltas = []

    for k in range(n_steps):
        prev = latent
        latent = model.dynamics(
            latent, act, act, offset_ms=500 + k * 500, dt_ms=500)
        deltas.append(latent - prev)
        latents.append(latent)

    return latents, deltas, act


# ============================================================
# Section 1: Delta Health
# ============================================================


class TestDeltaHealth:
    """Verify that the dynamics produces non-trivial, diverse deltas."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_delta_nonzero_every_step(self):
        """Each dynamics step must produce a delta with non-trivial L2 norm.

        At random init, each delta should have magnitude comparable to the
        latent (both are ~sqrt(d_model) due to LayerNorm).  A near-zero
        delta means the architecture structurally suppresses change.
        """
        _, deltas, _ = _run_rollout(self.model)

        for k, delta in enumerate(deltas):
            norm = delta.norm(dim=-1).mean().item()
            assert norm > 0.1, (
                f"Step {k}: delta L2 norm={norm:.4f} — "
                f"dynamics produces near-zero delta"
            )

    @torch.no_grad()
    def test_delta_magnitude_does_not_collapse(self):
        """||delta_k|| should not decay more than 10x over the rollout.

        Post-norm self-attention bounds delta magnitude, but it should
        not systematically shrink across steps.  A decay ratio < 0.1
        means the dynamics is contracting.
        """
        _, deltas, _ = _run_rollout(self.model)

        norms = [d.norm(dim=-1).mean().item() for d in deltas]
        ratio = norms[-1] / max(norms[0], 1e-8)

        assert ratio > 0.1, (
            f"Delta magnitude collapsed: first={norms[0]:.4f}, "
            f"last={norms[-1]:.4f}, ratio={ratio:.4f}"
        )

    @torch.no_grad()
    def test_delta_directions_are_diverse(self):
        """Consecutive deltas should not all point in the same direction.

        Mean cosine similarity between delta_k and delta_{k+1} should be
        well below 1.0.  If deltas are collinear, the rollout is just
        linear extrapolation — it can't represent nonlinear plasma evolution.
        """
        B = 2
        _, deltas, _ = _run_rollout(self.model, B=B)

        cos_sims = []
        for i in range(1, len(deltas)):
            cos = F.cosine_similarity(
                deltas[i].reshape(B, -1),
                deltas[i - 1].reshape(B, -1), dim=1)
            cos_sims.append(cos.mean().item())

        mean_cos = sum(cos_sims) / len(cos_sims)
        assert mean_cos < 0.97, (
            f"Deltas are too collinear: mean cos_sim={mean_cos:.4f} — "
            f"rollout degenerates to linear extrapolation"
        )

    @torch.no_grad()
    def test_delta_not_proportional_to_latent(self):
        """Delta should not be a scalar multiple of the current latent.

        If delta_k ∝ latent_k, the dynamics is just scaling the state,
        not predicting meaningful change.  Check that the component of
        delta orthogonal to latent is substantial.
        """
        B = 2
        latents, deltas, _ = _run_rollout(self.model, B=B)

        for k, delta in enumerate(deltas):
            lat = latents[k]  # state before this delta
            lat_flat = lat.reshape(B, -1)
            delta_flat = delta.reshape(B, -1)

            # Project delta onto latent direction
            lat_norm = lat_flat / lat_flat.norm(dim=1, keepdim=True).clamp(min=1e-8)
            proj = (delta_flat * lat_norm).sum(dim=1, keepdim=True) * lat_norm
            ortho = delta_flat - proj

            # Orthogonal component should be substantial
            ortho_ratio = ortho.norm(dim=1).mean() / delta_flat.norm(dim=1).mean()
            assert ortho_ratio > 0.3, (
                f"Step {k}: delta is too aligned with latent "
                f"(orthogonal ratio={ortho_ratio:.3f}). "
                f"Dynamics is just scaling the state."
            )


# ============================================================
# Section 2: Actuator Sensitivity
# ============================================================


class TestActuatorSensitivity:
    """Verify that actuator inputs meaningfully affect the dynamics."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_different_actuators_diverge(self):
        """Same starting latent, different actuators → diverging trajectories.

        After N_STEPS, the Euclidean distance between trajectories must
        be non-trivial.
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act_a = _random_actuators(B)

        latent_a = self.model.encode(lat_ctx, act_ctx)
        latent_b = latent_a.clone()

        for k in range(N_STEPS):
            act_b = _random_actuators(B)
            latent_a = self.model.dynamics(
                latent_a, act_a, act_a, offset_ms=500 + k * 500, dt_ms=500)
            latent_b = self.model.dynamics(
                latent_b, act_b, act_b, offset_ms=500 + k * 500, dt_ms=500)

        dist = (latent_a - latent_b).norm(dim=-1).mean().item()
        assert dist > 0.1, (
            f"Distance={dist:.4f} — dynamics ignores actuators"
        )

    @torch.no_grad()
    def test_actuator_change_changes_delta(self):
        """The SAME initial state with different actuators must produce
        different single-step deltas.

        This is a tighter version of the trajectory test: even at step 0,
        different actuators must produce different deltas.
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act_a = _random_actuators(B)
        act_b = _random_actuators(B)

        latent = self.model.encode(lat_ctx, act_ctx)

        out_a = self.model.dynamics(
            latent, act_a, act_a, offset_ms=500, dt_ms=500)
        out_b = self.model.dynamics(
            latent, act_b, act_b, offset_ms=500, dt_ms=500)

        delta_a = out_a - latent
        delta_b = out_b - latent

        dist = (delta_a - delta_b).norm(dim=-1).mean().item()
        assert dist > 0.01, (
            f"Delta distance={dist:.6f} — single-step dynamics ignores "
            f"actuator differences"
        )


# ============================================================
# Section 3: State Dependence
# ============================================================


class TestStateDependence:
    """Verify that delta = f(state, actuators), not g(actuators) alone.

    The fusion MLP concatenates [act_info, latent_current] — verify
    that the latent_current half actually affects the output.
    """

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_different_states_different_deltas(self):
        """Same actuators + different initial states → different deltas.

        Uses directly constructed latents (not encoder outputs) to test
        the dynamics in isolation.  The encoder squashes input differences
        at random init, which is expected — this test bypasses that.
        """
        B = 2
        act = _random_actuators(B)

        # Construct two clearly different latent states directly
        latent_a = torch.randn(B, N_LATENT, D_MODEL)
        latent_b = torch.randn(B, N_LATENT, D_MODEL)

        out_a = self.model.dynamics(
            latent_a, act, act, offset_ms=500, dt_ms=500)
        out_b = self.model.dynamics(
            latent_b, act, act, offset_ms=500, dt_ms=500)

        delta_a = out_a - latent_a
        delta_b = out_b - latent_b

        cos = F.cosine_similarity(
            delta_a.reshape(B, -1), delta_b.reshape(B, -1), dim=1)

        assert cos.mean().item() < 0.95, (
            f"cos_sim={cos.mean():.4f} — deltas are nearly identical for "
            f"different states.  The dynamics is state-independent."
        )

    def test_jacobian_of_delta_wrt_state(self):
        """∂delta/∂latent must have non-trivial Frobenius norm.

        If the Jacobian is near-zero, the dynamics output doesn't depend
        on the input state (fixed-point attractor).

        NOTE: We use MSE against a random target, NOT .sum(), because the
        dynamics self-attention uses post-norm LayerNorm whose output has
        zero mean per token — making .sum() trivially zero with zero
        gradient regardless of input.
        """
        B = 1
        act = _random_actuators(B)

        # Use directly constructed latent (bypass encoder)
        latent = torch.randn(B, N_LATENT, D_MODEL, requires_grad=True)
        target = torch.randn(B, N_LATENT, D_MODEL)

        out = self.model.dynamics(
            latent, act, act, offset_ms=500, dt_ms=500)
        delta = out - latent

        # Use MSE loss — .sum() gives zero gradient through LayerNorm
        loss = F.mse_loss(delta, target)
        loss.backward()
        grad = latent.grad

        assert grad is not None, "No gradient flowed to latent input"

        grad_norm = grad.norm().item()
        assert grad_norm > 1e-4, (
            f"Jacobian too small: grad_norm={grad_norm:.6f} — "
            f"dynamics delta barely depends on state"
        )


# ============================================================
# Section 4: Component Integrity (vs README spec)
# ============================================================


class TestComponentIntegrity:
    """Verify individual components match the README spec."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)

    @torch.no_grad()
    def test_cross_attention_no_query_passthrough(self):
        """_DynamicsCrossAttentionBlock: output must NOT contain a residual
        from the query input.

        If we pass in queries Q and context C, the output should be
        derived from C (via V), not from Q.  Specifically, if we use
        orthogonal Q and C, the output should be closer to C than to Q.
        """
        d = 64
        B, N_q, N_c = 2, 8, 12
        block = _DynamicsCrossAttentionBlock(d, n_heads=4, dropout=0.0)
        block.eval()

        # Create queries and context with very different statistics
        queries = torch.randn(B, N_q, d) * 10  # large magnitude
        context = torch.randn(B, N_c, d) * 0.1  # small magnitude

        output = block(queries, context)

        # If there's no query residual, the output magnitude should be
        # determined by the context (V), not the queries.
        # With LayerNorm(attn_out), magnitude is ~1 regardless.
        # The key test: output should NOT track query magnitude.
        q_corr = F.cosine_similarity(
            output.reshape(B, -1), queries.reshape(B, -1), dim=1)

        assert q_corr.abs().mean().item() < 0.5, (
            f"Output correlates with queries: cos_sim={q_corr.mean():.4f} — "
            f"cross-attention has accidental query residual"
        )

    @torch.no_grad()
    def test_cross_attention_output_varies_with_queries(self):
        """Different queries to the same context → different outputs.

        Even though there's no query residual, the attention ROUTING
        should depend on queries (Q-K alignment).
        """
        d = 64
        B, N_q, N_c = 2, 8, 12
        block = _DynamicsCrossAttentionBlock(d, n_heads=4, dropout=0.0)
        block.eval()

        context = torch.randn(B, N_c, d)
        queries_a = torch.randn(B, N_q, d)
        queries_b = torch.randn(B, N_q, d)

        out_a = block(queries_a, context)
        out_b = block(queries_b, context)

        dist = (out_a - out_b).norm(dim=-1).mean().item()
        assert dist > 0.01, (
            f"Distance={dist:.6f} — cross-attention ignores queries "
            f"(output is the same regardless of Q)"
        )

    @torch.no_grad()
    def test_fusion_mlp_uses_state(self):
        """Zeroing the state half of the fusion input must change output.

        The fusion MLP takes [act_info; latent_current; latent_prev; step_embed].
        If we replace latent_current with zeros, the output should
        change significantly.
        """
        model = _make_model()
        model.eval()
        dynamics = model.dynamics

        B = 2
        d = D_MODEL
        act_info = torch.randn(B, N_LATENT, d)
        latent = torch.randn(B, N_LATENT, d)
        latent_prev = torch.randn(B, N_LATENT, d)
        step_embed = torch.randn(B, N_LATENT, d)
        zeros = torch.zeros(B, N_LATENT, d)

        out_with_state = dynamics.fusion_net(
            torch.cat([act_info, latent, latent_prev, step_embed], dim=-1))
        out_without_state = dynamics.fusion_net(
            torch.cat([act_info, zeros, latent_prev, step_embed], dim=-1))

        dist = (out_with_state - out_without_state).norm(dim=-1).mean().item()
        assert dist > 0.1, (
            f"Fusion distance={dist:.4f} — fusion MLP ignores state input"
        )

    @torch.no_grad()
    def test_fusion_mlp_uses_actuator_info(self):
        """Zeroing the actuator half of the fusion input must change output."""
        model = _make_model()
        model.eval()
        dynamics = model.dynamics

        B = 2
        d = D_MODEL
        act_info = torch.randn(B, N_LATENT, d)
        latent = torch.randn(B, N_LATENT, d)
        latent_prev = torch.randn(B, N_LATENT, d)
        step_embed = torch.randn(B, N_LATENT, d)
        zeros = torch.zeros(B, N_LATENT, d)

        out_with_act = dynamics.fusion_net(
            torch.cat([act_info, latent, latent_prev, step_embed], dim=-1))
        out_without_act = dynamics.fusion_net(
            torch.cat([zeros, latent, latent_prev, step_embed], dim=-1))

        dist = (out_with_act - out_without_act).norm(dim=-1).mean().item()
        assert dist > 0.1, (
            f"Fusion distance={dist:.4f} — fusion MLP ignores actuator input"
        )

    @torch.no_grad()
    def test_decoder_differentiates_latent_states(self):
        """The Perceiver decoder must produce different AE tokens for
        different latent inputs.

        If the decoder ignores the latent (e.g., just returns its own
        learned queries), decoded signals would be constant regardless
        of dynamics output.
        """
        model = _make_model()
        model.eval()

        B = 2
        lat_a = torch.randn(B, N_LATENT, D_MODEL)
        lat_b = torch.randn(B, N_LATENT, D_MODEL)

        dec_a = model.decode(lat_a)
        dec_b = model.decode(lat_b)

        for name in dec_a:
            dist = (dec_a[name] - dec_b[name]).norm(dim=-1).mean().item()
            assert dist > 0.01, (
                f"Decoder output for '{name}' doesn't change with latent "
                f"(dist={dist:.6f})"
            )


# ============================================================
# Section 5: Gradient Health
# ============================================================


class TestGradientHealth:
    """Verify gradients flow properly through the rollout."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()

    def test_gradient_flows_through_rollout(self):
        """Gradient from step N loss must reach dynamics parameters."""
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)
        target = torch.randn(B, N_LATENT, D_MODEL)

        self.model.train()
        latent = self.model.encode(lat_ctx, act_ctx)

        for k in range(N_STEPS):
            latent = self.model.dynamics(
                latent, act, act, offset_ms=500 + k * 500, dt_ms=500)

        # Use MSE loss (not .sum()) to avoid LayerNorm zero-sum artifact
        loss = F.mse_loss(latent, target)
        loss.backward()

        grad_norm = 0.0
        for p in self.model.dynamics.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item()

        assert grad_norm > 0, "No gradient reached dynamics parameters"

    def test_gradient_reaches_encoder(self):
        """Gradient from dynamics output must reach encoder parameters.

        The dynamics input comes from the encoder.  If gradient doesn't
        flow back through, encoder weights are effectively frozen even
        when they shouldn't be.
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)
        target = torch.randn(B, N_LATENT, D_MODEL)

        self.model.train()
        latent = self.model.encode(lat_ctx, act_ctx)
        latent = self.model.dynamics(
            latent, act, act, offset_ms=500, dt_ms=500)

        # Use MSE loss (not .sum()) to avoid LayerNorm zero-sum artifact
        loss = F.mse_loss(latent, target)
        loss.backward()

        # Check encoder parameters (not the dynamics' own actuator tokenizer)
        encoder_grad_norm = 0.0
        for p in self.model.encoder.parameters():
            if p.grad is not None:
                encoder_grad_norm += p.grad.norm().item()

        assert encoder_grad_norm > 0, (
            "No gradient reached encoder parameters from dynamics output"
        )

    def test_no_vanishing_gradient_over_rollout(self):
        """Per-step gradient magnitude should not decay exponentially.

        Compute loss at step k only, check that gradient magnitude to
        dynamics parameters doesn't vanish for large k.
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)
        target = torch.randn(B, N_LATENT, D_MODEL)

        grad_norms_per_step = []

        for target_step in [0, N_STEPS // 2, N_STEPS - 1]:
            self.model.zero_grad()
            self.model.train()
            latent = self.model.encode(lat_ctx, act_ctx)

            for k in range(target_step + 1):
                latent = self.model.dynamics(
                    latent, act, act, offset_ms=500 + k * 500, dt_ms=500)

            # Use MSE loss (not .sum()) to avoid LayerNorm zero-sum artifact
            loss = F.mse_loss(latent, target)
            loss.backward()

            gn = sum(p.grad.norm().item()
                     for p in self.model.dynamics.parameters()
                     if p.grad is not None)
            grad_norms_per_step.append(gn)

        # Gradient at last step should be at least 1% of first step
        ratio = grad_norms_per_step[-1] / max(grad_norms_per_step[0], 1e-8)
        assert ratio > 0.01, (
            f"Gradient vanishes over rollout: step_0={grad_norms_per_step[0]:.4f}, "
            f"step_{N_STEPS-1}={grad_norms_per_step[-1]:.4f}, ratio={ratio:.6f}"
        )


# ============================================================
# Section 6: Signal-Space Validation
# ============================================================


class TestSignalSpace:
    """Verify that decoded predictions are healthy."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_decoded_outputs_differ_across_steps(self):
        """Decoded AE tokens at different rollout steps must not be identical.

        This is the ground-truth test for copy behavior: even if latent-
        space metrics look OK, the decoded signals must actually change.
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)

        latent = self.model.encode(lat_ctx, act_ctx)

        decoded_steps = []
        for k in range(N_STEPS):
            latent = self.model.dynamics(
                latent, act, act, offset_ms=500 + k * 500, dt_ms=500)
            ae_tok = self.model.decode(latent)
            flat = torch.cat(
                [t.reshape(B, -1) for t in ae_tok.values()], dim=1)
            decoded_steps.append(flat)

        # Check pairwise distances between decoded steps
        cors = []
        for i in range(1, len(decoded_steps)):
            cos = F.cosine_similarity(
                decoded_steps[i], decoded_steps[i - 1], dim=1)
            cors.append(cos.mean().item())

        mean_cor = sum(cors) / len(cors)
        assert mean_cor < 0.995, (
            f"Mean decoded correlation={mean_cor:.4f} — "
            f"rollout produces identical signals at every step"
        )

    @torch.no_grad()
    def test_decoded_trajectory_spans_space(self):
        """The decoded trajectory should not be confined to a low-rank subspace.

        Stack all decoded outputs into a matrix and check its effective
        rank (number of singular values > 10% of the largest).
        If rank ≈ 1, the trajectory is a line (linear extrapolation).
        """
        B = 1
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)

        latent = self.model.encode(lat_ctx, act_ctx)

        decoded_steps = []
        for k in range(N_STEPS):
            latent = self.model.dynamics(
                latent, act, act, offset_ms=500 + k * 500, dt_ms=500)
            ae_tok = self.model.decode(latent)
            flat = torch.cat(
                [t.reshape(1, -1) for t in ae_tok.values()], dim=1)
            decoded_steps.append(flat.squeeze(0))

        # Stack: [N_STEPS, D_decoded]
        traj = torch.stack(decoded_steps, dim=0)
        # Center
        traj = traj - traj.mean(dim=0, keepdim=True)

        # SVD
        _, S, _ = torch.linalg.svd(traj, full_matrices=False)
        # Effective rank: singular values > 10% of largest
        threshold = 0.1 * S[0]
        eff_rank = (S > threshold).sum().item()

        assert eff_rank >= 2, (
            f"Trajectory effective rank={eff_rank} — "
            f"decoded predictions lie on a line (linear extrapolation). "
            f"Singular values: {S[:5].tolist()}"
        )

    @torch.no_grad()
    def test_dynamics_changes_decoder_output_vs_context(self):
        """decode(dynamics(encode(ctx))) must differ from decode(encode(ctx)).

        This directly tests that the dynamics step actually CHANGES the
        decoded output compared to just encoding and decoding the context.
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)

        latent_ctx = self.model.encode(lat_ctx, act_ctx)
        dec_ctx = self.model.decode(latent_ctx)

        latent_pred = self.model.dynamics(
            latent_ctx, act, act, offset_ms=500, dt_ms=500)
        dec_pred = self.model.decode(latent_pred)

        for name in dec_ctx:
            dist = (dec_ctx[name] - dec_pred[name]).norm(dim=-1).mean().item()
            assert dist > 0.01, (
                f"'{name}': dynamics doesn't change decoded output "
                f"(dist={dist:.6f})"
            )


# ============================================================
# Section 7: Rollout Accumulation
# ============================================================


class TestRolloutAccumulation:
    """Verify that multi-step rollout accumulates meaningfully."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_total_displacement_grows_with_steps(self):
        """The total latent displacement from context should grow with
        the number of rollout steps (at least sub-linearly).

        If displacement saturates immediately, the dynamics has a
        fixed-point attractor near the context.
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)

        latent_0 = self.model.encode(lat_ctx, act_ctx)
        latent = latent_0.clone()

        displacements = []
        for k in range(N_STEPS):
            latent = self.model.dynamics(
                latent, act, act, offset_ms=500 + k * 500, dt_ms=500)
            disp = (latent - latent_0).norm(dim=-1).mean().item()
            displacements.append(disp)

        # Displacement at step N should be larger than at step 1
        assert displacements[-1] > displacements[0], (
            f"Displacement doesn't grow: step_1={displacements[0]:.4f}, "
            f"step_{N_STEPS}={displacements[-1]:.4f}"
        )

        # Should grow by at least 2x over the rollout
        growth = displacements[-1] / max(displacements[0], 1e-8)
        assert growth > 2.0, (
            f"Displacement grows too slowly: "
            f"step_1={displacements[0]:.4f}, "
            f"step_{N_STEPS}={displacements[-1]:.4f}, "
            f"growth={growth:.2f}x"
        )

    @torch.no_grad()
    def test_rollout_not_periodic(self):
        """The rollout should not cycle back to previous states.

        Check that distance from context monotonically increases
        (or at least doesn't decrease significantly).
        """
        B = 2
        lat_ctx = _random_ae_latents(B)
        act_ctx = _random_actuators(B)
        act = _random_actuators(B)

        latent_0 = self.model.encode(lat_ctx, act_ctx)
        latent = latent_0.clone()

        prev_disp = 0.0
        decreases = 0
        for k in range(N_STEPS):
            latent = self.model.dynamics(
                latent, act, act, offset_ms=500 + k * 500, dt_ms=500)
            disp = (latent - latent_0).norm(dim=-1).mean().item()
            if disp < prev_disp * 0.9:  # Allow 10% tolerance
                decreases += 1
            prev_disp = disp

        assert decreases <= N_STEPS // 4, (
            f"Displacement decreased {decreases}/{N_STEPS} steps — "
            f"rollout is periodic or contracting"
        )