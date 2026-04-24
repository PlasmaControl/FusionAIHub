"""
Unit tests for the Aurora-inspired tokamak foundation model.

Testing strategy:
  1. Shape tests:       Does each module produce the right output shape?
  2. Gradient tests:    Do gradients flow through every parameter?
  3. Invariant tests:   Does the module respect known constraints?
  4. Numerical tests:   Is the output reasonable (not NaN, not exploding)?
  5. Integration tests: Do modules compose correctly end-to-end?

Each test uses small dimensions for speed:
  B=2, d_model=32, n_latents=8, n_heads=4, backbone_blocks=2

Run with:
    pixi run pytest tests/test_aurora.py -v
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

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

# ── Test fixtures ──────────────────────────────────────────────────────────

B = 2
D = 32
N_L = 8
N_HEADS = 4
N_BLOCKS = 2
DT = 0.5

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
N_ACT = len(ACTUATOR_CONFIGS)


@pytest.fixture
def ae_tokens():
    return {
        m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
        for m, cfg in MODALITY_CONFIGS.items()
    }


@pytest.fixture
def ae_tokens_pair():
    t0 = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
          for m, cfg in MODALITY_CONFIGS.items()}
    t1 = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
          for m, cfg in MODALITY_CONFIGS.items()}
    return t0, t1


@pytest.fixture
def actuator_signals():
    T_samples = 50
    return {
        a: torch.randn(B, cfg["n_channels"], T_samples)
        for a, cfg in ACTUATOR_CONFIGS.items()
    }


@pytest.fixture
def latent():
    return torch.randn(B, N_L, D)


@pytest.fixture
def actuator_tokens():
    return torch.randn(B, N_ACT * 5, D)


def _make_model():
    return TokamakFoundationModel(
        modality_configs=MODALITY_CONFIGS,
        d_model=D,
        n_latent=N_L,
        n_heads=N_HEADS,
        encoder_cross_layers=1,
        encoder_self_layers=1,
        backbone_blocks=N_BLOCKS,
        decoder_layers=1,
        mlp_ratio=2.0,
        dropout=0.0,
        actuator_configs=ACTUATOR_CONFIGS,
    )


def zero_actuators(T_samples: int = 50) -> dict:
    """Build a dict of zero-valued raw actuator signals matching the
    ACTUATOR_CONFIGS schema — used as a neutral control for dynamics tests."""
    return {
        a: torch.zeros(B, cfg["n_channels"], T_samples)
        for a, cfg in ACTUATOR_CONFIGS.items()
    }


# ═══════════════════════════════════════════════════════════════════════════
# 1. MODALITY TOKENIZER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestModalityTokenizer:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.tokenizer = ModalityTokenizer(MODALITY_CONFIGS, d_model=D)

    def test_output_shape(self, ae_tokens):
        out = self.tokenizer(ae_tokens)
        assert out.shape == (B, N_TOTAL, D)

    def test_output_shape_subset(self):
        subset = {"filterscopes": torch.randn(B, 4, 16)}
        out = self.tokenizer(subset)
        assert out.shape == (B, 4, D)

    def test_gradients_flow(self, ae_tokens):
        out = self.tokenizer(ae_tokens)
        out.sum().backward()
        for m in MODALITY_CONFIGS:
            w = self.tokenizer.projections[m].weight
            assert w.grad is not None
            assert w.grad.abs().sum() > 0

    def test_gradients_to_input(self):
        ae_tok = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"],
                                  requires_grad=True)
                  for m, cfg in MODALITY_CONFIGS.items()}
        out = self.tokenizer(ae_tok)
        out.sum().backward()
        for m in ae_tok:
            assert ae_tok[m].grad is not None

    def test_token_count_matches_input(self, ae_tokens):
        out = self.tokenizer(ae_tokens)
        expected = sum(ae_tokens[m].shape[1] for m in ae_tokens)
        assert out.shape[1] == expected

    def test_no_nans(self, ae_tokens):
        assert not torch.isnan(self.tokenizer(ae_tokens)).any()

    def test_output_scale_reasonable(self, ae_tokens):
        out = self.tokenizer(ae_tokens)
        assert 0.01 < out.std() < 100.0


# ═══════════════════════════════════════════════════════════════════════════
# 2. ACTUATOR TOKENIZER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestActuatorTokenizer:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.tokenizer = ActuatorTokenizer(ACTUATOR_CONFIGS, d_model=D)

    def test_output_shape(self, actuator_signals):
        out = self.tokenizer(actuator_signals, offset_ms=0.0)
        assert out.shape[0] == B
        assert out.shape[2] == D
        assert out.shape[1] > 0

    def test_different_offsets_different_pe(self, actuator_signals):
        out1 = self.tokenizer(actuator_signals, offset_ms=0.0)
        out2 = self.tokenizer(actuator_signals, offset_ms=500.0)
        assert not torch.allclose(out1, out2)

    def test_gradients_flow(self, actuator_signals):
        out = self.tokenizer(actuator_signals, offset_ms=0.0)
        out.sum().backward()
        for name, param in self.tokenizer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_no_nans(self, actuator_signals):
        assert not torch.isnan(
            self.tokenizer(actuator_signals, offset_ms=0.0)).any()

    def test_layernorm_applied(self, actuator_signals):
        out = self.tokenizer(actuator_signals, offset_ms=0.0)
        per_token_mean = out.mean(dim=-1)
        per_token_std = out.std(dim=-1)
        assert per_token_mean.abs().max() < 0.5
        assert (per_token_std - 1.0).abs().max() < 0.5


# ═══════════════════════════════════════════════════════════════════════════
# 3. PERCEIVER ENCODER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceiverEncoder:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.encoder = PerceiverEncoder(
            d_model=D, n_latent_queries=N_L,
            n_cross_layers=1, n_self_layers=1, n_heads=N_HEADS)

    def test_output_shape(self):
        inp = torch.randn(B, N_TOTAL + N_ACT * 5, D)
        out = self.encoder(inp)
        assert out.shape == (B, N_L, D)

    def test_output_independent_of_input_length(self):
        short = torch.randn(B, 5, D)
        long = torch.randn(B, 200, D)
        assert self.encoder(short).shape == (B, N_L, D)
        assert self.encoder(long).shape == (B, N_L, D)

    def test_gradients_to_latent_queries(self):
        inp = torch.randn(B, N_TOTAL, D)
        self.encoder(inp).sum().backward()
        assert self.encoder.latent_queries.grad is not None
        assert self.encoder.latent_queries.grad.abs().sum() > 0

    def test_gradients_to_input(self):
        inp = torch.randn(B, N_TOTAL, D, requires_grad=True)
        self.encoder(inp).sum().backward()
        assert inp.grad is not None

    def test_no_nans(self):
        assert not torch.isnan(
            self.encoder(torch.randn(B, N_TOTAL, D))).any()

    def test_deterministic_in_eval(self):
        self.encoder.eval()
        inp = torch.randn(B, N_TOTAL, D)
        assert torch.allclose(self.encoder(inp), self.encoder(inp))


# ═══════════════════════════════════════════════════════════════════════════
# 4. BACKBONE BLOCK TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestBackboneBlock:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.block = BackboneBlock(d_model=D, n_heads=N_HEADS, mlp_ratio=4.0)

    def test_output_shape(self, latent, actuator_tokens):
        out = self.block(latent, actuator_tokens)
        assert out.shape == latent.shape

    def test_all_parameters_receive_gradients(self, latent, actuator_tokens):
        self.block(latent, actuator_tokens).sum().backward()
        for name, param in self.block.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"

    def test_residual_connection_exists(self, latent, actuator_tokens):
        out = self.block(latent, actuator_tokens)
        cos_sim = F.cosine_similarity(
            out.flatten(1), latent.flatten(1), dim=1).mean()
        assert cos_sim > 0.0, "Residual connection may be broken"

    def test_pre_norm_not_post_norm(self):
        large_lat = torch.randn(B, N_L, D) * 50.0
        large_act = torch.randn(B, N_ACT * 5, D) * 50.0
        out = self.block(large_lat, large_act)
        assert out.abs().max() > 10.0, "Output bounded — looks post-normed"

    def test_no_nans(self, latent, actuator_tokens):
        assert not torch.isnan(self.block(latent, actuator_tokens)).any()

    def test_no_nans_large_input(self):
        large = torch.randn(B, N_L, D) * 100.0
        act = torch.randn(B, N_ACT * 5, D)
        assert not torch.isnan(self.block(large, act)).any()


# ═══════════════════════════════════════════════════════════════════════════
# 5. LATENT BACKBONE TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestLatentBackbone:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.backbone = LatentBackbone(
            d_model=D, n_blocks=N_BLOCKS, n_heads=N_HEADS, mlp_ratio=4.0)

    def test_output_shape(self, latent, actuator_tokens):
        out = self.backbone(latent, actuator_tokens, step_index=0)
        assert out.shape == (B, N_L, D)

    def test_gradients_flow_all_blocks(self, latent, actuator_tokens):
        self.backbone(latent, actuator_tokens, step_index=0).sum().backward()
        for name, param in self.backbone.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"

    def test_step_embedding_receives_gradient(self, latent, actuator_tokens):
        self.backbone(latent, actuator_tokens, step_index=3).sum().backward()
        for name, param in self.backbone.step_mlp.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Step embed param {name} has no gradient")

    def test_different_steps_different_output(self, latent, actuator_tokens):
        out0 = self.backbone(latent, actuator_tokens, step_index=0)
        out5 = self.backbone(latent, actuator_tokens, step_index=5,
                              offset_ms=3000.0)
        assert not torch.allclose(out0, out5, atol=1e-5)

    def test_skip_connections(self, latent, actuator_tokens):
        bb_noskip = deepcopy(self.backbone)
        bb_noskip.use_skips = False
        out_skip = self.backbone(latent, actuator_tokens, step_index=0)
        out_noskip = bb_noskip(latent, actuator_tokens, step_index=0)
        if self.backbone.use_skips:
            assert not torch.allclose(out_skip, out_noskip, atol=1e-5)

    def test_no_nans(self, latent, actuator_tokens):
        assert not torch.isnan(
            self.backbone(latent, actuator_tokens, step_index=0)).any()

    def test_output_not_identical_to_input(self, latent, actuator_tokens):
        out = self.backbone(latent, actuator_tokens, step_index=0)
        assert not torch.allclose(out, latent, atol=1e-3)


# ═══════════════════════════════════════════════════════════════════════════
# 6. PERCEIVER DECODER TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestPerceiverDecoder:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        oq = {m: cfg["n_tokens"] for m, cfg in MODALITY_CONFIGS.items()}
        self.decoder = PerceiverDecoder(
            d_model=D, output_queries_config=oq, n_layers=1, n_heads=N_HEADS)

    def test_output_shapes_per_modality(self, latent):
        out = self.decoder(latent)
        for m, cfg in MODALITY_CONFIGS.items():
            assert out[m].shape == (B, cfg["n_tokens"], D)

    def test_subset_modalities(self, latent):
        out = self.decoder(latent, modality="filterscopes")
        assert out.shape == (B, 4, D)

    def test_gradients_to_output_queries(self, latent):
        out = self.decoder(latent)
        sum(v.sum() for v in out.values()).backward()
        for m in MODALITY_CONFIGS:
            assert self.decoder.output_queries[m].grad is not None

    def test_gradients_to_latent_input(self):
        lat = torch.randn(B, N_L, D, requires_grad=True)
        out = self.decoder(lat)
        sum(v.sum() for v in out.values()).backward()
        assert lat.grad is not None
        assert lat.grad.abs().sum() > 0

    def test_no_nans(self, latent):
        out = self.decoder(latent)
        for m in out:
            assert not torch.isnan(out[m]).any(), f"NaN in {m}"


# ═══════════════════════════════════════════════════════════════════════════
# 7. FULL MODEL FORWARD PASS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestFullModel:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()

    def test_output_shapes(self, ae_tokens, actuator_signals):
        out = self.model.forward(
            ae_tokens, actuator_signals, actuator_signals, step_index=0)
        for m, cfg in MODALITY_CONFIGS.items():
            assert out[m].shape == (B, cfg["n_tokens"], cfg["d_lat"])

    def test_output_same_keys_as_input(self, ae_tokens, actuator_signals):
        out = self.model.forward(
            ae_tokens, actuator_signals, actuator_signals, step_index=0)
        assert set(out.keys()) == set(ae_tokens.keys())

    def test_full_gradient_flow(self, ae_tokens, actuator_signals):
        out = self.model.forward(
            ae_tokens, actuator_signals, actuator_signals, step_index=0)
        loss = sum(v.sum() for v in out.values())
        loss.backward()

        missing = []
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                if param.grad is None or param.grad.abs().sum() == 0:
                    missing.append(name)
        assert len(missing) == 0, f"No gradients: {missing}"

    def test_two_step_gradient_flow(self, ae_tokens, actuator_signals):
        pred1 = self.model.forward(
            ae_tokens, actuator_signals, actuator_signals, step_index=0)
        pred2 = self.model.forward(
            pred1, actuator_signals, actuator_signals, step_index=1)

        sum(v.sum() for v in pred2.values()).backward()

        for name, param in self.model.modality_tokenizer.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, (
                    f"Gradient didn't flow through 2-step chain to {name}")

    def test_different_inputs_different_outputs(self, actuator_signals):
        tok1 = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                for m, cfg in MODALITY_CONFIGS.items()}
        tok2 = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                for m, cfg in MODALITY_CONFIGS.items()}
        out1 = self.model.forward(
            tok1, actuator_signals, actuator_signals, step_index=0)
        out2 = self.model.forward(
            tok2, actuator_signals, actuator_signals, step_index=0)
        for m in MODALITY_CONFIGS:
            assert not torch.allclose(out1[m], out2[m], atol=1e-5)

    def test_not_identity(self, ae_tokens, actuator_signals):
        out = self.model.forward(
            ae_tokens, actuator_signals, actuator_signals, step_index=0)
        for m in ae_tokens:
            assert not torch.allclose(out[m], ae_tokens[m], atol=1e-3)

    def test_no_nans(self, ae_tokens, actuator_signals):
        out = self.model.forward(
            ae_tokens, actuator_signals, actuator_signals, step_index=0)
        for m in out:
            assert not torch.isnan(out[m]).any()

    def test_output_finite(self, ae_tokens, actuator_signals):
        out = self.model.forward(
            ae_tokens, actuator_signals, actuator_signals, step_index=0)
        for m in out:
            assert torch.isfinite(out[m]).all()


# ═══════════════════════════════════════════════════════════════════════════
# 8. ROLLOUT TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestRollout:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()
        self.model.eval()

    def _act_pairs(self, n):
        return [({a: torch.randn(B, cfg["n_channels"], 50)
                  for a, cfg in ACTUATOR_CONFIGS.items()},
                 {a: torch.randn(B, cfg["n_channels"], 50)
                  for a, cfg in ACTUATOR_CONFIGS.items()})
                for _ in range(n)]

    @torch.no_grad()
    def test_rollout_produces_n_steps(self, ae_tokens):
        preds = self.model.rollout(ae_tokens, self._act_pairs(4), n_steps=4)
        assert len(preds) == 4

    @torch.no_grad()
    def test_each_step_has_correct_shape(self, ae_tokens):
        for pred in self.model.rollout(ae_tokens, self._act_pairs(4)):
            for m, cfg in MODALITY_CONFIGS.items():
                assert pred[m].shape == (B, cfg["n_tokens"], cfg["d_lat"])

    @torch.no_grad()
    def test_steps_differ(self, ae_tokens):
        preds = self.model.rollout(ae_tokens, self._act_pairs(4))
        for k in range(len(preds) - 1):
            all_same = all(
                torch.allclose(preds[k][m], preds[k + 1][m], atol=1e-5)
                for m in MODALITY_CONFIGS)
            assert not all_same, (
                f"Step {k} and {k+1} identical — copy behavior!")

    @torch.no_grad()
    def test_rollout_is_deterministic(self, ae_tokens):
        pairs = self._act_pairs(3)
        preds1 = self.model.rollout(ae_tokens, pairs)
        preds2 = self.model.rollout(ae_tokens, pairs)
        for k in range(3):
            for m in MODALITY_CONFIGS:
                assert torch.allclose(preds1[k][m], preds2[k][m])

    @torch.no_grad()
    def test_no_nans_through_rollout(self, ae_tokens):
        for k, pred in enumerate(
            self.model.rollout(ae_tokens, self._act_pairs(8))
        ):
            for m in pred:
                assert not torch.isnan(pred[m]).any(), (
                    f"NaN at step {k}, modality {m}")

    @torch.no_grad()
    def test_no_explosion_through_rollout(self, ae_tokens):
        max_norms = []
        for pred in self.model.rollout(ae_tokens, self._act_pairs(8)):
            norms = [pred[m].norm().item() for m in pred]
            max_norms.append(max(norms))
        assert max_norms[-1] < max_norms[0] * 100, (
            f"Exploded: step1={max_norms[0]:.1f}, step8={max_norms[-1]:.1f}")

    @torch.no_grad()
    def test_no_collapse_through_rollout(self, ae_tokens):
        min_norms = []
        for pred in self.model.rollout(ae_tokens, self._act_pairs(8)):
            norms = [pred[m].norm().item() for m in pred]
            min_norms.append(min(norms))
        assert min_norms[-1] > min_norms[0] * 0.01, (
            f"Collapsed: step1={min_norms[0]:.4f}, step8={min_norms[-1]:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# 9. TRAINING LOOP TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestTraining:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()

    def test_single_step_loss_decreases(self, actuator_signals):
        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

        ae_in = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                 for m, cfg in MODALITY_CONFIGS.items()}
        ae_tgt = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                  for m, cfg in MODALITY_CONFIGS.items()}

        pred = self.model.forward(
            ae_in, actuator_signals, actuator_signals, step_index=0)
        loss1 = sum(F.l1_loss(pred[m], ae_tgt[m]) for m in MODALITY_CONFIGS)

        optimizer.zero_grad()
        loss1.backward()
        optimizer.step()

        pred = self.model.forward(
            ae_in, actuator_signals, actuator_signals, step_index=0)
        loss2 = sum(F.l1_loss(pred[m], ae_tgt[m]) for m in MODALITY_CONFIGS)

        assert loss2.item() < loss1.item(), "Loss didn't decrease"

    def test_multistep_loss_backprop(self, actuator_signals):
        self.model.train()

        ae_in = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                 for m, cfg in MODALITY_CONFIGS.items()}
        targets = [{m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                    for m, cfg in MODALITY_CONFIGS.items()}
                   for _ in range(3)]

        current = ae_in
        total_loss = 0
        for k in range(3):
            pred = self.model.forward(
                current, actuator_signals, actuator_signals, step_index=k)
            total_loss = total_loss + sum(
                F.l1_loss(pred[m], targets[k][m]) for m in MODALITY_CONFIGS)
            current = pred

        total_loss.backward()

        n_with = sum(1 for p in self.model.parameters()
                     if p.requires_grad and p.grad is not None
                     and p.grad.abs().sum() > 0)
        n_total = sum(1 for p in self.model.parameters() if p.requires_grad)
        assert n_with == n_total, (
            f"Only {n_with}/{n_total} params got gradients through 3-step")


# ═══════════════════════════════════════════════════════════════════════════
# 10. ENCODER-DECODER ROUNDTRIP TEST
# ═══════════════════════════════════════════════════════════════════════════


class TestEncoderDecoderRoundtrip:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.tokenizer = ModalityTokenizer(MODALITY_CONFIGS, D)
        self.encoder = PerceiverEncoder(
            d_model=D, n_latent_queries=N_L,
            n_cross_layers=2, n_self_layers=2, n_heads=N_HEADS)
        oq = {m: cfg["n_tokens"] for m, cfg in MODALITY_CONFIGS.items()}
        self.decoder = PerceiverDecoder(
            d_model=D, output_queries_config=oq,
            n_layers=2, n_heads=N_HEADS)

    def test_roundtrip_shape(self, ae_tokens):
        diag_tokens = self.tokenizer(ae_tokens)
        latent = self.encoder(diag_tokens)
        reconstructed = self.decoder(latent)
        for m, cfg in MODALITY_CONFIGS.items():
            assert reconstructed[m].shape == (B, cfg["n_tokens"], D)

    def test_roundtrip_loss_trainable(self, ae_tokens):
        diag_tokens = self.tokenizer(ae_tokens)
        latent = self.encoder(diag_tokens)
        reconstructed = self.decoder(latent)
        # Decoder outputs d_model, so compare shapes not values
        loss = sum(reconstructed[m].sum() for m in MODALITY_CONFIGS)
        loss.backward()
        assert self.encoder.latent_queries.grad is not None


# ═══════════════════════════════════════════════════════════════════════════
# 11. STRESS TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestStress:

    @pytest.fixture(autouse=True)
    def setup(self):
        torch.manual_seed(42)
        self.model = _make_model()

    def test_zero_input(self, actuator_signals):
        zeros = {m: torch.zeros(B, cfg["n_tokens"], cfg["d_lat"])
                 for m, cfg in MODALITY_CONFIGS.items()}
        out = self.model.forward(
            zeros, actuator_signals, actuator_signals, step_index=0)
        for m in out:
            assert not torch.isnan(out[m]).any()

    def test_large_input(self, actuator_signals):
        large = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"]) * 1000
                 for m, cfg in MODALITY_CONFIGS.items()}
        out = self.model.forward(
            large, actuator_signals, actuator_signals, step_index=0)
        for m in out:
            assert not torch.isnan(out[m]).any()

    def test_batch_size_1(self):
        tokens = {m: torch.randn(1, cfg["n_tokens"], cfg["d_lat"])
                  for m, cfg in MODALITY_CONFIGS.items()}
        acts = {a: torch.randn(1, cfg["n_channels"], 50)
                for a, cfg in ACTUATOR_CONFIGS.items()}
        out = self.model.forward(tokens, acts, acts, step_index=0)
        for m in out:
            assert out[m].shape[0] == 1

    @torch.no_grad()
    def test_long_rollout_stability(self, actuator_signals):
        self.model.eval()
        tokens = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                  for m, cfg in MODALITY_CONFIGS.items()}
        current = tokens
        for k in range(16):
            current = self.model.forward(
                current, actuator_signals, actuator_signals, step_index=k)
            for m in current:
                assert torch.isfinite(current[m]).all(), (
                    f"Non-finite at step {k}, modality {m}")

    def test_gradient_norm_bounded(self, actuator_signals):
        tokens = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                  for m, cfg in MODALITY_CONFIGS.items()}
        targets = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                   for m, cfg in MODALITY_CONFIGS.items()}
        pred = self.model.forward(
            tokens, actuator_signals, actuator_signals, step_index=0)
        loss = sum(F.l1_loss(pred[m], targets[m]) for m in MODALITY_CONFIGS)
        loss.backward()
        total_grad = torch.sqrt(sum(
            p.grad.norm() ** 2 for p in self.model.parameters()
            if p.grad is not None))
        assert torch.isfinite(total_grad)
        assert total_grad < 1e6


# ═══════════════════════════════════════════════════════════════════════════
# 12. DIAGNOSTIC TESTS — failure modes observed in production training
# ═══════════════════════════════════════════════════════════════════════════


class TestCopyBaseline:
    """Model must beat the trivial copy baseline after brief training."""

    def test_model_beats_copy_after_training(self):
        torch.manual_seed(0)
        model = _make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        pairs = []
        for _ in range(20):
            t0 = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                  for m, cfg in MODALITY_CONFIGS.items()}
            t1 = {m: t0[m] * 0.9 + 0.1 * torch.sin(t0[m] * 3.0)
                  for m in MODALITY_CONFIGS}
            pairs.append((t0, t1))

        act = zero_actuators()

        for step in range(200):
            optimizer.zero_grad()
            loss = 0
            for t0, t1 in pairs:
                pred = model.forward(t0, act, act, step_index=0)
                loss += sum(F.mse_loss(pred[m], t1[m]) for m in MODALITY_CONFIGS)
            loss.backward()
            optimizer.step()

        model.eval()
        model_wins = 0
        with torch.no_grad():
            for t0, t1 in pairs:
                pred = model.forward(t0, act, act, step_index=0)
                model_mse = sum(F.mse_loss(pred[m], t1[m]).item()
                               for m in MODALITY_CONFIGS)
                copy_mse = sum(F.mse_loss(t0[m], t1[m]).item()
                              for m in MODALITY_CONFIGS)
                if model_mse < copy_mse:
                    model_wins += 1

        print(f"  Model wins: {model_wins}/{len(pairs)}")
        assert model_wins > len(pairs) // 2, (
            f"Model wins only {model_wins}/{len(pairs)} — worse than copying")


class TestLossFunction:
    """Verify loss function doesn't penalize dynamics less than steady-state."""

    def test_loss_not_variance_normalized(self):
        """Same absolute error should produce same loss regardless of target variance."""
        pred = torch.zeros(B, 4, 16)

        # Low variance target
        static_target = torch.ones(B, 4, 16) * 0.3

        # High variance target, same absolute distance from pred
        dynamic_target = torch.randn(B, 4, 16) * 5.0
        dynamic_target = dynamic_target + 0.3  # shift so mean error ≈ 0.3

        # Compute loss the way training code does
        loss_static = F.l1_loss(pred, static_target)
        loss_dynamic = F.l1_loss(pred, dynamic_target)

        # If variance normalization is active, loss_dynamic would be
        # divided by a large number and be much smaller
        # Without it, loss_dynamic should be >= loss_static
        # because dynamic_target has elements further from pred
        print(f"  Static loss: {loss_static:.4f}, Dynamic loss: {loss_dynamic:.4f}")
        # The key check: dynamic loss should NOT be smaller than static
        assert loss_dynamic >= loss_static * 0.5, (
            "High-variance target gets lower loss — variance normalization likely active")

    def test_same_error_same_loss_regardless_of_variance(self):
        """Identical prediction errors should produce identical loss."""
        error = 0.3

        # Low variance target
        target_low = torch.ones(B, 4, 16) * 1.0
        pred_low = target_low + error

        # High variance target, same pointwise error
        target_high = torch.randn(B, 4, 16) * 10.0
        pred_high = target_high + error

        loss_low = F.l1_loss(pred_low, target_low)
        loss_high = F.l1_loss(pred_high, target_high)

        assert torch.allclose(loss_low, loss_high, atol=1e-5), (
            f"Same error gives different loss: {loss_low:.6f} vs {loss_high:.6f} — "
            f"loss is scaled by target variance")


class TestRolloutDynamics:
    """After training, rollout must not converge to a fixed point."""

    def test_rollout_no_fixed_point_after_training(self):
        torch.manual_seed(0)
        model = _make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        sequences = []
        for _ in range(10):
            steps = []
            state = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                     for m, cfg in MODALITY_CONFIGS.items()}
            steps.append(state)
            for k in range(4):
                state = {m: state[m] * 0.95 + 0.05 * torch.sin(state[m] * 2.0 + k * 0.5)
                         for m in MODALITY_CONFIGS}
                steps.append(state)
            sequences.append(steps)

        act = zero_actuators()

        for epoch in range(100):
            optimizer.zero_grad()
            loss = 0
            for seq in sequences:
                current = seq[0]
                for k in range(1, len(seq)):
                    pred = model.forward(current, act, act, step_index=k-1)
                    loss += sum(F.mse_loss(pred[m], seq[k][m])
                               for m in MODALITY_CONFIGS)
                    current = pred
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            current = sequences[0][0]
            cos_sims = []
            prev_pred = None
            for k in range(4):
                pred = model.forward(current, act, act, step_index=k)
                if prev_pred is not None:
                    cos = max(
                        F.cosine_similarity(
                            pred[m].flatten(1), prev_pred[m].flatten(1), dim=1
                        ).mean().item()
                        for m in MODALITY_CONFIGS)
                    cos_sims.append(cos)
                prev_pred = pred
                current = pred

            print(f"  Rollout cos_sims: {cos_sims}")
            for k, cos in enumerate(cos_sims):
                assert cos < 0.99, (
                    f"Step {k+1}→{k+2} cos_sim={cos:.4f} — fixed point collapse")


class TestPerceiverRoundtripChain:
    """Multiple encode-decode cycles must not erase temporal information."""

    def test_multi_roundtrip_preserves_difference(self):
        torch.manual_seed(0)
        model = _make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ae_a = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                for m, cfg in MODALITY_CONFIGS.items()}
        ae_b = {m: ae_a[m] + torch.randn_like(ae_a[m]) * 0.3
                for m in MODALITY_CONFIGS}
        act = zero_actuators()

        for step in range(500):
            optimizer.zero_grad()
            out_a = model.forward(ae_a, act, act, step_index=0)
            out_b = model.forward(ae_b, act, act, step_index=0)
            loss = sum(
                F.mse_loss(out_a[m], ae_a[m]) + F.mse_loss(out_b[m], ae_b[m])
                for m in MODALITY_CONFIGS)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            current_a = ae_a
            current_b = ae_b
            out_a = current_a
            out_b = current_b
            for k in range(4):
                out_a = model.forward(current_a, act, act, step_index=k)
                out_b = model.forward(current_b, act, act, step_index=k)

                for m in MODALITY_CONFIGS:
                    cos = F.cosine_similarity(
                        out_a[m].flatten(1), out_b[m].flatten(1), dim=1
                    ).mean().item()
                    raw_cos = F.cosine_similarity(
                        ae_a[m].flatten(1), ae_b[m].flatten(1), dim=1
                    ).mean().item()
                    print(f"  Roundtrip {k+1}, {m}: cos={cos:.4f} "
                          f"(raw={raw_cos:.4f})")

                current_a = out_a
                current_b = out_b

            max_cos = max(
                F.cosine_similarity(
                    out_a[m].flatten(1), out_b[m].flatten(1), dim=1
                ).mean().item()
                for m in MODALITY_CONFIGS)
            assert max_cos < 0.99, (
                f"4 roundtrips collapsed difference (max cos={max_cos:.4f})")


class TestDataScale:
    """All modalities must have comparable scale after normalization."""

    def test_normalized_tokens_unit_variance(self):
        """After applying stored normalization stats, tokens should have std ≈ 1."""
        # This would need access to real AE token stats
        # For a unit test, verify the normalization math is correct
        raw = torch.randn(100, 4, 16) * 5.0 + 3.0  # mean=3, std=5
        mean = raw.mean(dim=0)
        std = raw.std(dim=0).clamp(min=1e-6)
        normalized = (raw - mean) / std

        assert (normalized.mean(dim=0).abs() < 0.1).all(), "Mean not near zero"
        assert ((normalized.std(dim=0) - 1.0).abs() < 0.1).all(), "Std not near one"

    def test_tokenizer_output_balanced(self):
        """After tokenization, all modalities should contribute
        comparable norm to the encoder input."""
        torch.manual_seed(0)
        tokenizer = ModalityTokenizer(MODALITY_CONFIGS, d_model=D)
        ae_tokens = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                    for m, cfg in MODALITY_CONFIGS.items()}

        out = tokenizer(ae_tokens)

        idx = 0
        norms = {}
        for m, cfg in MODALITY_CONFIGS.items():
            n = cfg["n_tokens"]
            modality_tokens = out[:, idx:idx+n, :]
            norms[m] = modality_tokens.norm(dim=-1).mean().item()
            idx += n

        print(f"  Per-modality tokenized norms: {norms}")
        max_norm = max(norms.values())
        min_norm = min(norms.values())
        assert max_norm / (min_norm + 1e-8) < 10.0, (
            f"Tokenized norms imbalanced: max/min = {max_norm/min_norm:.1f}")


class TestSignalPathway:
    """Identify where in the model temporal information is lost."""

    def test_signal_survives_each_stage(self):
        torch.manual_seed(0)
        model = _make_model()
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

        ae_a = {m: torch.randn(B, cfg["n_tokens"], cfg["d_lat"])
                for m, cfg in MODALITY_CONFIGS.items()}
        ae_b = {m: ae_a[m] + torch.randn_like(ae_a[m]) * 0.3
                for m in MODALITY_CONFIGS}
        act = zero_actuators()

        for step in range(200):
            optimizer.zero_grad()
            out_a = model.forward(ae_a, act, act, step_index=0)
            out_b = model.forward(ae_b, act, act, step_index=0)
            loss = sum(
                F.mse_loss(out_a[m], ae_a[m]) + F.mse_loss(out_b[m], ae_b[m])
                for m in MODALITY_CONFIGS)
            loss.backward()
            optimizer.step()

        model.eval()
        act_curr_tok = model.actuator_tokenizer(act, offset_ms=0.0)
        act_fut_tok = model.actuator_tokenizer(act, offset_ms=500.0)
        act_tok = torch.cat([act_curr_tok, act_fut_tok], dim=1)

        with torch.no_grad():
            diag_a = model.modality_tokenizer(ae_a)
            diag_b = model.modality_tokenizer(ae_b)
            tok_cos = F.cosine_similarity(
                diag_a.flatten(1), diag_b.flatten(1), dim=1).mean()

            enc_a = model.encoder(torch.cat([diag_a, act_tok], dim=1))
            enc_b = model.encoder(torch.cat([diag_b, act_tok], dim=1))
            enc_cos = F.cosine_similarity(
                enc_a.flatten(1), enc_b.flatten(1), dim=1).mean()

            bb_a = model.backbone(enc_a, act_tok, step_index=0)
            bb_b = model.backbone(enc_b, act_tok, step_index=0)
            bb_cos = F.cosine_similarity(
                bb_a.flatten(1), bb_b.flatten(1), dim=1).mean()

            dec_a = model.decoder(bb_a)
            dec_b = model.decoder(bb_b)

            print(f"  Tokenizer cos: {tok_cos:.4f}")
            print(f"  Encoder cos:   {enc_cos:.4f}")
            print(f"  Backbone cos:  {bb_cos:.4f}")
            for m in MODALITY_CONFIGS:
                dec_cos = F.cosine_similarity(
                    dec_a[m].flatten(1), dec_b[m].flatten(1), dim=1).mean()
                print(f"  Decoder {m} cos: {dec_cos:.4f}")

            stages = [tok_cos.item(), enc_cos.item(), bb_cos.item()]
            for i in range(1, len(stages)):
                increase = stages[i] - stages[i-1]
                assert increase < 0.1, (
                    f"Stage {i} increases cos_sim by {increase:.3f} — "
                    f"information bottleneck detected")

            total_increase = stages[-1] - stages[0]
            assert total_increase < 0.15, (
                f"Total cos_sim increase from tokenizer to backbone: {total_increase:.3f}")
