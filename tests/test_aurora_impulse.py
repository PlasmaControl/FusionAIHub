"""
Impulse tests for the Aurora-inspired tokamak foundation model.

Inject a single non-zero input ("impulse") and trace how the signal
propagates through each module.  Much more informative than random inputs
because you can verify causality, information flow, and mixing behavior.

Run with:
    pixi run pytest tests/test_aurora_impulse.py -v -s
"""

import pytest
import torch
import torch.nn.functional as F
from copy import deepcopy
import matplotlib.pyplot as plt

from tokamak_foundation_model.models.aurora.backbone import (
    BackboneBlock,
    LatentBackbone,
)
from tokamak_foundation_model.models.aurora.encoder_decoder import (
    PerceiverDecoder,
    PerceiverEncoder,
)
from tokamak_foundation_model.models.aurora.foundation_model import (
    TokamakFoundationModel,
)
from tokamak_foundation_model.models.latent_feature_space.modality_tokenizer import (
    ActuatorTokenizer,
    ModalityTokenizer,
)

# ── Test dimensions ────────────────────────────────────────────────────────

B = 2
D = 32
N_L = 8
N_HEADS = 4
N_BLOCKS = 2

MODALITY_CONFIGS = {
    "filterscopes": {"n_tokens": 4, "d_lat": 16},
    "ts_core_temp": {"n_tokens": 3, "d_lat": 8},
    "mse": {"n_tokens": 4, "d_lat": 16},
}

ACTUATOR_CONFIGS = {
    "pin": {"target_fs": 10000, "n_channels": 2, "patch_len": 10},
    "beam_voltage": {"target_fs": 10000, "n_channels": 4, "patch_len": 10},
}

N_TOTAL = sum(cfg["n_tokens"] for cfg in MODALITY_CONFIGS.values())
T_SAMPLES = 50


# ── Helpers ────────────────────────────────────────────────────────────────


def zero_ae_tokens():
    return {m: torch.zeros(B, cfg["n_tokens"], cfg["d_lat"])
            for m, cfg in MODALITY_CONFIGS.items()}


def zero_actuators():
    return {a: torch.zeros(B, cfg["n_channels"], T_SAMPLES)
            for a, cfg in ACTUATOR_CONFIGS.items()}


def per_token_norms(x):
    """(B, N, D) → (N,) average norm per token position."""
    return x.norm(dim=-1).mean(dim=0)


def per_modality_norms(ae_tokens):
    """Dict of AE tokens → dict of scalar norms."""
    return {m: v.norm().item() for m, v in ae_tokens.items()}


def _make_model():
    return TokamakFoundationModel(
        modality_configs=MODALITY_CONFIGS,
        d_model=D, n_latent=N_L, n_heads=N_HEADS,
        encoder_cross_layers=1, encoder_self_layers=1,
        backbone_blocks=N_BLOCKS, decoder_layers=1,
        mlp_ratio=2.0, dropout=0.0,
        actuator_configs=ACTUATOR_CONFIGS,
    )


def _do_rollout(model, ae_tokens, actuators, n_steps):
    """Simple rollout using the same actuators at every step."""
    act_pairs = [(actuators, actuators)] * n_steps
    return model.rollout(ae_tokens, act_pairs, n_steps=n_steps)


# ═══════════════════════════════════════════════════════════════════════════
# 1. MODALITY TOKENIZER — single modality impulse
# ═══════════════════════════════════════════════════════════════════════════


class TestModalityTokenizerImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.tokenizer = ModalityTokenizer(MODALITY_CONFIGS, d_model=D)

    def test_impulse_in_single_modality(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8) * 10.0  # strong impulse
        out = self.tokenizer(ae_tok)
        norms = per_token_norms(out)

        max_norm = norms.max().item()
        min_norm = norms.min().item()

        print(f"  Token norms: {norms.tolist()}")
        print(f"  Max/min ratio: {max_norm / (min_norm + 1e-8):.1f}")

        assert max_norm > min_norm * 1.5, (
            "Impulse modality tokens should be larger than zero-input tokens")

    def test_zero_modalities_still_nonzero(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)
        out = self.tokenizer(ae_tok)
        norms = per_token_norms(out)
        assert norms.min() > 0, (
            "Some tokens exactly zero — modality embedding missing?")

    def test_impulse_in_each_modality_produces_different_output(self):
        """Impulse in filterscopes vs mse should produce different tokenizer output."""
        ae_a = zero_ae_tokens()
        ae_a["filterscopes"] = torch.ones(B, 4, 16) * 10.0

        ae_b = zero_ae_tokens()
        ae_b["mse"] = torch.ones(B, 4, 16) * 10.0

        out_a = self.tokenizer(ae_a)
        out_b = self.tokenizer(ae_b)

        cos_sim = F.cosine_similarity(
            out_a.flatten(1), out_b.flatten(1), dim=1).mean()

        print(f"  Cos sim (filterscopes vs mse impulse): {cos_sim:.4f}")
        assert cos_sim < 0.999, (
            "Different modality impulses produce identical output")


# ═══════════════════════════════════════════════════════════════════════════
# 2. ACTUATOR TOKENIZER — single actuator impulse
# ═══════════════════════════════════════════════════════════════════════════


class TestActuatorTokenizerImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.tokenizer = ActuatorTokenizer(ACTUATOR_CONFIGS, d_model=D)

    def test_actuator_impulse_direction(self):
        out_zero = self.tokenizer(zero_actuators(), offset_ms=0.0)

        actuators = zero_actuators()
        actuators["beam_voltage"] = torch.ones(B, 4, T_SAMPLES)
        out_impulse = self.tokenizer(actuators, offset_ms=0.0)

        cos_sim = F.cosine_similarity(
            out_zero.flatten(1), out_impulse.flatten(1), dim=1).mean()

        print(f"  Cos sim (zero vs impulse): {cos_sim:.4f}")
        assert cos_sim < 0.99, "Actuator impulse didn't change output direction"

    def test_step_vs_ramp(self):
        step = zero_actuators()
        step["beam_voltage"] = torch.ones(B, 4, T_SAMPLES)

        ramp = zero_actuators()
        ramp["beam_voltage"] = torch.linspace(
            0, 1, T_SAMPLES).expand(B, 4, T_SAMPLES)

        out_step = self.tokenizer(step, offset_ms=0.0)
        out_ramp = self.tokenizer(ramp, offset_ms=0.0)

        cos_sim = F.cosine_similarity(
            out_step.flatten(1), out_ramp.flatten(1), dim=1).mean()

        print(f"  Cos sim (step vs ramp): {cos_sim:.4f}")
        assert cos_sim < 0.99, (
            "Step and ramp produce identical tokens — Conv1d not working")


# ═══════════════════════════════════════════════════════════════════════════
# 3. PERCEIVER ENCODER — single token impulse
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceiverEncoderImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.encoder = PerceiverEncoder(
            d_model=D, n_latent_queries=N_L,
            n_cross_layers=1, n_self_layers=1, n_heads=N_HEADS)

    def test_impulse_spreads_to_all_queries(self):
        inp = torch.zeros(B, N_TOTAL, D)
        inp[:, 5, :] = 10.0

        latent = self.encoder(inp)
        norms = per_token_norms(latent)

        print(f"  Latent query norms: {norms.tolist()}")
        n_active = (norms > 0.01).sum().item()
        print(f"  Active queries: {n_active}/{N_L}")

        assert n_active == N_L, (
            f"Only {n_active}/{N_L} queries activated")

    def test_baseline_vs_impulse(self):
        """Adding a strong impulse to one token should change the encoder output."""
        inp_base = torch.randn(B, N_TOTAL, D) * 0.1  # small baseline
        latent_base = self.encoder(inp_base)

        inp_impulse = inp_base.clone()
        inp_impulse[:, 5, :] += 50.0  # strong impulse on top
        latent_impulse = self.encoder(inp_impulse)

        diff_norm = (latent_impulse - latent_base).norm().item()
        print(f"  Impulse contribution norm: {diff_norm:.8f}")
        # At random init, Perceiver learned queries dominate — the impulse
        # effect is small but must be non-zero (cross-attention is working).
        assert diff_norm > 0.1, "Impulse barely affected encoder output — check norm_kv"


# ═══════════════════════════════════════════════════════════════════════════
# 4. BACKBONE BLOCK — impulse mixing
# ═══════════════════════════════════════════════════════════════════════════


class TestBackboneBlockImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.block = BackboneBlock(d_model=D, n_heads=N_HEADS, mlp_ratio=4.0)

    def test_self_attention_spreads_impulse(self):
        latent = torch.zeros(B, N_L, D)
        latent[:, 3, :] = 5.0
        act = torch.zeros(B, 5, D)

        out = self.block(latent, act)
        norms = per_token_norms(out)

        print(f"  Per-token norms after block: {norms.tolist()}")
        n_active = (norms > 0.01).sum().item()
        assert n_active == N_L, (
            f"Only {n_active}/{N_L} tokens active — self-attention not mixing")

    def test_impulse_position_retains_highest_norm(self):
        latent = torch.zeros(B, N_L, D)
        latent[:, 3, :] = 5.0
        act = torch.zeros(B, 5, D)

        out = self.block(latent, act)
        norms = per_token_norms(out)

        impulse_norm = norms[3].item()
        other_max = torch.cat([norms[:3], norms[4:]]).max().item()

        print(f"  Impulse position norm: {impulse_norm:.3f}")
        print(f"  Max other norm: {other_max:.3f}")

        assert impulse_norm > other_max, (
            "Impulse position lost advantage — residual connection broken?")

    def test_cross_attention_to_actuators(self):
        latent = torch.zeros(B, N_L, D)
        act = torch.randn(B, 5, D) * 5.0

        out = self.block(latent, act)
        norms = per_token_norms(out)

        print(f"  Token norms (zero latent, active actuators): {norms.tolist()}")
        assert norms.min() > 0.01, (
            "Some tokens zero despite active actuators — cross-attention broken")

    def test_actuator_vs_no_actuator(self):
        latent = torch.randn(B, N_L, D)

        out_no_act = self.block(latent, torch.zeros(B, 5, D))
        out_with_act = self.block(latent, torch.randn(B, 5, D) * 5.0)

        diff = (out_with_act - out_no_act).norm().item()
        print(f"  Output difference from actuators: {diff:.4f}")
        assert diff > 0.1, "Actuators had no effect on backbone block output"


# ═══════════════════════════════════════════════════════════════════════════
# 5. FULL BACKBONE — impulse propagation through depth
# ═══════════════════════════════════════════════════════════════════════════


class TestBackboneImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.backbone = LatentBackbone(
            d_model=D, n_blocks=N_BLOCKS, n_heads=N_HEADS, mlp_ratio=4.0)

    def test_progressive_mixing(self):
        latent = torch.zeros(B, N_L, D)
        latent[:, 3, :] = 5.0
        act = torch.zeros(B, 5, D)

        intermediate_cvs = []

        def hook_fn(module, input, output):
            norms = per_token_norms(output)
            cv = (norms.std() / (norms.mean() + 1e-8)).item()
            intermediate_cvs.append(cv)

        handles = [b.register_forward_hook(hook_fn)
                   for b in self.backbone.blocks]

        self.backbone(latent, act, step_index=0)

        for h in handles:
            h.remove()

        print(f"  Per-block norm CV: {intermediate_cvs}")

        if len(intermediate_cvs) >= 2:
            assert intermediate_cvs[-1] <= intermediate_cvs[0] * 1.5, (
                "Signal not mixing — later blocks have higher variance")

    def test_step_embedding_changes_output(self):
        latent = torch.zeros(B, N_L, D)
        latent[:, 3, :] = 5.0
        act = torch.zeros(B, 5, D)

        out_0 = self.backbone(latent, act, step_index=0)
        out_7 = self.backbone(latent, act, step_index=7, offset_ms=3500.0)

        cos_sim = F.cosine_similarity(
            out_0.flatten(1), out_7.flatten(1), dim=1).mean()

        print(f"  Cos sim (step 0 vs step 7): {cos_sim:.4f}")
        assert cos_sim < 0.99, "Step embedding has no effect on output"


# ═══════════════════════════════════════════════════════════════════════════
# 6. PERCEIVER DECODER — single latent token impulse
# ═══════════════════════════════════════════════════════════════════════════


class TestDecoderImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        oq = {m: cfg["n_tokens"] for m, cfg in MODALITY_CONFIGS.items()}
        self.decoder = PerceiverDecoder(
            d_model=D, output_queries_config=oq,
            n_layers=1, n_heads=N_HEADS)

    def test_impulse_reaches_all_modalities(self):
        latent_zero = torch.zeros(B, N_L, D)
        latent_impulse = torch.zeros(B, N_L, D)
        latent_impulse[:, 3, :] = torch.ones(D) * 5.0

        out_zero = self.decoder(latent_zero)
        out_impulse = self.decoder(latent_impulse)

        for m in MODALITY_CONFIGS:
            diff = (out_impulse[m] - out_zero[m]).norm().item()
            cos = F.cosine_similarity(
                out_impulse[m].flatten(1), out_zero[m].flatten(1), dim=1).mean()
            print(f"{m}: diff_norm={diff:.4f}, cos_sim={cos:.4f}")

        norms = {m: v.norm().item() for m, v in out_impulse.items()}

        print(f"  Per-modality output norms: {norms}")
        for m, norm in norms.items():
            assert norm > 0.01, (
                f"Modality {m} got zero output from latent impulse")

    def test_modalities_produce_different_outputs(self):
        latent = torch.zeros(B, N_L, D)
        latent[:, 3, :] = 5.0

        out = self.decoder(latent)

        if "filterscopes" in out and "mse" in out:
            cos_sim = F.cosine_similarity(
                out["filterscopes"].flatten(1),
                out["mse"].flatten(1), dim=1).mean()

            print(f"  Cos sim (filterscopes vs mse): {cos_sim:.4f}")
            assert cos_sim < 0.95, (
                "Different modalities decode identically")

    def test_baseline_vs_impulse(self):
        """Adding a strong impulse should change decoder output."""
        lat_base = torch.randn(B, N_L, D) * 0.1  # small baseline
        lat_impulse = lat_base.clone()
        lat_impulse[:, 3, :] += 50.0

        out_base = self.decoder(lat_base)
        out_impulse = self.decoder(lat_impulse)

        total_diff = 0.0
        for m in MODALITY_CONFIGS:
            diff = (out_impulse[m] - out_base[m]).norm().item()
            print(f"  {m}: impulse contribution = {diff:.8f}")
            total_diff += diff
        # At random init the effect is small but must be non-zero.
        assert total_diff > 0.1, "Impulse barely affected decoder output — check norm_kv"


# ═══════════════════════════════════════════════════════════════════════════
# 7. FULL MODEL — cross-modality information transfer
# ═══════════════════════════════════════════════════════════════════════════


class TestFullModelImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_single_modality_activates_all_outputs(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)
        act = zero_actuators()

        out = self.model.forward(ae_tok, act, act, step_index=0)
        norms = per_modality_norms(out)

        print(f"  Output norms (ts_core_temp impulse):")
        for m, norm in norms.items():
            print(f"    {m}: {norm:.4f}")

        for m, norm in norms.items():
            assert norm > 0.001, (
                f"{m} has zero output despite ts_core_temp input")

    def test_different_input_modalities_give_different_outputs(self):
        ae_a = zero_ae_tokens()
        ae_a["filterscopes"] = torch.ones(B, 4, 16)

        ae_b = zero_ae_tokens()
        ae_b["ts_core_temp"] = torch.ones(B, 3, 8)
        act = zero_actuators()

        # 1. Tokenizer
        diag_a = self.model.modality_tokenizer(ae_a)
        diag_b = self.model.modality_tokenizer(ae_b)
        print(f"After tokenizer: cos_sim={F.cosine_similarity(diag_a.flatten(1), diag_b.flatten(1), dim=1).mean():.6f}")

        # 2. Encoder
        act_tok = self.model.actuator_tokenizer(act, offset_ms=0.0)
        enc_input_a = torch.cat([diag_a, act_tok], dim=1)
        enc_input_b = torch.cat([diag_b, act_tok], dim=1)
        latent_a = self.model.encoder(enc_input_a)
        latent_b = self.model.encoder(enc_input_b)
        print(f"After encoder: cos_sim={F.cosine_similarity(latent_a.flatten(1), latent_b.flatten(1), dim=1).mean():.6f}")

        # 3. Backbone
        bb_a = self.model.backbone(latent_a, act_tok, step_index=0)
        bb_b = self.model.backbone(latent_b, act_tok, step_index=0)
        print(f"After backbone: cos_sim={F.cosine_similarity(bb_a.flatten(1), bb_b.flatten(1), dim=1).mean():.6f}")

        # 4. Decoder
        dec_a = self.model.decoder(bb_a)
        dec_b = self.model.decoder(bb_b)
        for m in MODALITY_CONFIGS:
            cos = F.cosine_similarity(dec_a[m].flatten(1), dec_b[m].flatten(1), dim=1).mean()
            print(f"After decoder {m}: cos_sim={cos:.6f}")

        # 5. Output projections (if they exist)
        out_a = self.model.forward(ae_a, act, act, step_index=0)
        out_b = self.model.forward(ae_b, act, act, step_index=0)
        for m in MODALITY_CONFIGS:
            cos = F.cosine_similarity(out_a[m].flatten(1), out_b[m].flatten(1), dim=1).mean()
            print(f"Final output {m}: cos_sim={cos:.6f}")

        # At random init, encoder squashes differences. Check that
        # outputs are at least not numerically identical.
        for m in MODALITY_CONFIGS:
            cos_sim = F.cosine_similarity(
                out_a[m].flatten(1), out_b[m].flatten(1), dim=1).mean()
            print(f"  {m}: cos_sim = {cos_sim:.4f}")

        # At least one modality should show substantial difference
        min_cos = min(
            F.cosine_similarity(out_a[m].flatten(1), out_b[m].flatten(1), dim=1).mean()
            for m in MODALITY_CONFIGS)
        assert min_cos < 0.95, "All modalities produce nearly identical output regardless of input"

    def test_training_breaks_output_symmetry(self):
        """After a few reconstruction steps, the model must distinguish inputs."""
        model = _make_model()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ae_a = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                for m, cfg in MODALITY_CONFIGS.items()}
        ae_b = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                for m, cfg in MODALITY_CONFIGS.items()}
        act = zero_actuators()

        for step in range(50):
            optimizer.zero_grad()
            out_a = model.forward(ae_a, act, act, step_index=0)
            out_b = model.forward(ae_b, act, act, step_index=0)
            loss = sum(
                F.mse_loss(out_a[m], ae_a[m]) + F.mse_loss(out_b[m], ae_b[m])
                for m in MODALITY_CONFIGS)
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            out_a = model.forward(ae_a, act, act, step_index=0)
            out_b = model.forward(ae_b, act, act, step_index=0)

        for m in MODALITY_CONFIGS:
            cos = F.cosine_similarity(
                out_a[m].flatten(1), out_b[m].flatten(1), dim=1).mean()
            print(f"  {m}: cos_sim after training = {cos:.4f}")

        max_cos = max(
            F.cosine_similarity(
                out_a[m].flatten(1), out_b[m].flatten(1), dim=1).mean()
            for m in MODALITY_CONFIGS)
        assert max_cos < 0.9, (
            f"Model still can't distinguish inputs after 50 training steps "
            f"(max cos_sim={max_cos:.4f})")

    @torch.no_grad()
    def test_actuator_impulse_changes_output(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        out_no_act = self.model.forward(
            ae_tok, zero_actuators(), zero_actuators(), step_index=0)

        act = zero_actuators()
        act["beam_voltage"] = torch.ones(B, 4, T_SAMPLES) * 5.0
        out_with_act = self.model.forward(ae_tok, act, act, step_index=0)

        total_diff = sum(
            (out_with_act[m] - out_no_act[m]).norm().item()
            for m in MODALITY_CONFIGS)

        for m in MODALITY_CONFIGS:
            diff = (out_with_act[m] - out_no_act[m]).norm().item()
            print(f"  {m}: actuator effect = {diff:.4f}")

        assert total_diff > 0.01, "Actuators had no effect on model output"

    @torch.no_grad()
    def test_output_not_identical_to_input(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        out = self.model.forward(
            ae_tok, zero_actuators(), zero_actuators(), step_index=0)

        cos_sim = F.cosine_similarity(
            ae_tok["ts_core_temp"].flatten(1),
            out["ts_core_temp"].flatten(1), dim=1).mean()

        print(f"  Input/output cos_sim for ts_core_temp: {cos_sim:.4f}")
        assert cos_sim < 0.99, "Output ≈ input — model is learning identity"


# ═══════════════════════════════════════════════════════════════════════════
# 8. ROLLOUT — impulse propagation across autoregressive steps
# ═══════════════════════════════════════════════════════════════════════════


class TestRolloutImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_signal_spreads_across_steps(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        preds = _do_rollout(self.model, ae_tok, zero_actuators(), n_steps=8)

        print(f"\n  Rollout impulse propagation:")
        for k, pred in enumerate(preds):
            norms = per_modality_norms(pred)
            print(f"    Step {k}: {norms}")

        last_norms = per_modality_norms(preds[-1])
        for m, norm in last_norms.items():
            assert norm > 0.001, (
                f"{m} still zero at step 8 — signal not propagating")

    @torch.no_grad()
    def test_no_modality_collapse(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        preds = _do_rollout(self.model, ae_tok, zero_actuators(), n_steps=8)
        last = preds[-1]

        if "filterscopes" in last and "mse" in last:
            cos_sim = F.cosine_similarity(
                last["filterscopes"].flatten(1),
                last["mse"].flatten(1), dim=1).mean()

            print(f"  Step 8 cos_sim (filterscopes vs mse): {cos_sim:.4f}")
            assert cos_sim < 0.99, (
                "Modalities converged to same output")

    @torch.no_grad()
    def test_consecutive_steps_differ(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        preds = _do_rollout(self.model, ae_tok, zero_actuators(), n_steps=4)

        for k in range(len(preds) - 1):
            for m in MODALITY_CONFIGS:
                cos = F.cosine_similarity(
                    preds[k][m].flatten(1),
                    preds[k + 1][m].flatten(1), dim=1).mean()
                print(f"  Step {k}→{k+1}, {m}: cos_sim={cos:.4f}")

            max_cos = max(
                F.cosine_similarity(
                    preds[k][m].flatten(1),
                    preds[k + 1][m].flatten(1), dim=1).mean()
                for m in MODALITY_CONFIGS)
            assert max_cos < 0.99, (
                f"Steps {k} and {k+1} too similar (cos_sim={max_cos:.4f})")

    @torch.no_grad()
    def test_no_explosion_from_impulse(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        preds = _do_rollout(self.model, ae_tok, zero_actuators(), n_steps=8)

        total_norms = [sum(v.norm().item() for v in p.values()) for p in preds]
        print(f"  Total norms per step: {[f'{n:.2f}' for n in total_norms]}")

        if total_norms[0] > 0:
            ratio = total_norms[-1] / total_norms[0]
            assert ratio < 100, f"Output exploded: ratio = {ratio:.1f}"

    @torch.no_grad()
    def test_no_collapse_from_impulse(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        preds = _do_rollout(self.model, ae_tok, zero_actuators(), n_steps=8)

        total_norms = [sum(v.norm().item() for v in p.values()) for p in preds]
        assert total_norms[-1] > total_norms[0] * 0.01, (
            f"Output collapsed: {total_norms[-1]:.4f} vs {total_norms[0]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. GRADIENT IMPULSE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestGradientImpulse:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()

    def test_gradient_from_one_modality_loss_reaches_all_parameters(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)

        out = self.model.forward(
            ae_tok, zero_actuators(), zero_actuators(), step_index=0)

        # Loss only on filterscopes (different modality than input)
        loss = out["filterscopes"].sum()
        loss.backward()

        n_with_grad = 0
        n_total = 0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                n_total += 1
                if param.grad is not None and param.grad.abs().sum() > 0:
                    n_with_grad += 1

        # Not all params get gradients: per-modality decoder blocks only
        # get gradients when their modality is in the loss.  Check that
        # shared params (encoder, backbone) all get gradients.
        print(f"  Parameters with gradients: {n_with_grad}/{n_total}")

        # Encoder and backbone must have gradients
        for name, param in self.model.encoder.named_parameters():
            if param.requires_grad:
                assert param.grad is not None and param.grad.abs().sum() > 0, (
                    f"Encoder param {name} missing gradient")
        for name, param in self.model.backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None and param.grad.abs().sum() > 0, (
                    f"Backbone param {name} missing gradient")

    def test_two_step_gradient_with_impulse(self):
        ae_tok = zero_ae_tokens()
        ae_tok["ts_core_temp"] = torch.ones(B, 3, 8)
        act = zero_actuators()

        pred1 = self.model.forward(ae_tok, act, act, step_index=0)
        pred2 = self.model.forward(pred1, act, act, step_index=1)

        loss = pred2["mse"].sum()
        loss.backward()

        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in self.model.modality_tokenizer.parameters())
        assert has_grad, (
            "Tokenizer got no gradients through 2-step impulse rollout")


class TestPerceiverBottleneck:
    """Check if the Perceiver roundtrip preserves differences between timesteps."""

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    @torch.no_grad()
    def test_roundtrip_preserves_temporal_difference(self):
        """Encode two different AE token sets, decode them.
        The decoded cos_sim should be close to the raw cos_sim."""
        ae_t0 = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                 for m, cfg in MODALITY_CONFIGS.items()}
        ae_t1 = {m: ae_t0[m] + torch.randn_like(ae_t0[m]) * 0.3  # 30% perturbation
                 for m in MODALITY_CONFIGS}

        out_t0 = self.model.forward(ae_t0, zero_actuators(), zero_actuators(), step_index=0)
        out_t1 = self.model.forward(ae_t1, zero_actuators(), zero_actuators(), step_index=0)

        for m in MODALITY_CONFIGS:
            raw_cos = F.cosine_similarity(
                ae_t0[m].flatten(1), ae_t1[m].flatten(1), dim=1).mean()
            roundtrip_cos = F.cosine_similarity(
                out_t0[m].flatten(1), out_t1[m].flatten(1), dim=1).mean()

            print(f"  {m}: raw_cos={raw_cos:.4f}, roundtrip_cos={roundtrip_cos:.4f}")

            # Roundtrip should not push cos_sim much closer to 1.0
            # If raw_cos is 0.95 and roundtrip_cos is 0.999, the bottleneck is killing changes
            gap = roundtrip_cos - raw_cos
            assert gap < 0.05, (
                f"{m}: bottleneck smoothed away temporal difference "
                f"(raw={raw_cos:.4f}, roundtrip={roundtrip_cos:.4f})")

    def test_roundtrip_after_training_preserves_temporal_difference(self):
        """After brief training, the model must preserve temporal differences."""
        model = _make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ae_t0 = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                 for m, cfg in MODALITY_CONFIGS.items()}
        ae_t1 = {m: ae_t0[m] + torch.randn_like(ae_t0[m]) * 0.3
                 for m in MODALITY_CONFIGS}
        act = zero_actuators()

        for step in range(500):
            optimizer.zero_grad()
            out_t0 = model.forward(ae_t0, act, act, step_index=0)
            out_t1 = model.forward(ae_t1, act, act, step_index=0)
            loss = sum(
                F.mse_loss(out_t0[m], ae_t0[m]) + F.mse_loss(out_t1[m], ae_t1[m])
                for m in MODALITY_CONFIGS)
            loss.backward()
            optimizer.step()
            print(f"  Step {step}: loss={loss.item():.6f}")

        with torch.no_grad():
            out_t0 = model.forward(ae_t0, act, act, step_index=0)
            out_t1 = model.forward(ae_t1, act, act, step_index=0)

            for m in MODALITY_CONFIGS:
                raw_cos = F.cosine_similarity(
                    ae_t0[m].flatten(1), ae_t1[m].flatten(1), dim=1).mean()
                roundtrip_cos = F.cosine_similarity(
                    out_t0[m].flatten(1), out_t1[m].flatten(1), dim=1).mean()
                gap = roundtrip_cos - raw_cos
                print(f"  {m}: raw={raw_cos:.4f}, roundtrip={roundtrip_cos:.4f}, gap={gap:.4f}")
                assert gap < 0.05, (
                    f"{m}: bottleneck persists after training (gap={gap:.4f})")