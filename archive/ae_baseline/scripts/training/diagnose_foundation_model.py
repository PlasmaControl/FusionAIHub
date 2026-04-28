"""Per-modality diagnostic for the foundation model.

Loads a trained foundation model checkpoint and computes per-modality MSEs
to identify where filterscope information is lost:
- AE token variance (how much info the AE tokens carry)
- Roundtrip MSE: encode(target) -> decode -> compare to target AE tokens
- Prediction MSE: encode(ctx) -> dynamics -> decode -> compare to target AE tokens
- Copy MSE: encode(ctx) -> decode -> compare to target AE tokens (no dynamics)

If roundtrip MSE is high -> Perceiver encode/decode is the bottleneck.
If roundtrip MSE is low but pred MSE is high -> dynamics is the bottleneck.
"""
import argparse
import logging
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from tokamak_foundation_model.data.multi_file_dataset import (
    TokamakMultiFileDataset, make_dataloader)
from tokamak_foundation_model.models.latent_feature_space.foundation_model import (
    PerceiverFoundationModel)

# Import configs and helpers from train_foundation_model
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_foundation_model import (
    DIAGNOSTIC_CONFIGS, ACTUATOR_CONFIGS, DT_S, WINDOW_S, CHUNK_S,
    load_ae, split_window, encode_batch,
    actuator_context_window, actuator_step_windows,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Foundation model per-modality diagnostic")
    parser.add_argument("--checkpoint", required=True, help="Path to foundation model checkpoint")
    parser.add_argument("--data_dir", default="/scratch/gpfs/EKOLEMEN/foundation_model/")
    parser.add_argument("--stats_path", default="/projects/EKOLEMEN/foundation_model/preprocessing_stats.pt")
    parser.add_argument("--ae_checkpoint_dir", default="/projects/EKOLEMEN/foundation_model/")
    parser.add_argument("--max_files", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--n_batches", type=int, default=5, help="Number of val batches to evaluate")
    args = parser.parse_args()

    # --- Load checkpoint metadata ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    saved_args = ckpt.get("args", {})
    modality_configs_saved = ckpt.get("modality_configs", {})

    logger.info(f"Checkpoint epoch: {ckpt.get('epoch', '?')}")
    logger.info(f"  d_model={saved_args.get('d_model')}, n_latent={saved_args.get('n_latent')}")
    logger.info(f"  dynamics_type={saved_args.get('dynamics_type')}")
    logger.info(f"  zero_actuators={saved_args.get('zero_actuators')}")

    # --- Load AE models ---
    ae_ckpt_dir = Path(args.ae_checkpoint_dir)
    ae_models = {}
    for name, cfg in DIAGNOSTIC_CONFIGS.items():
        ckpt_path = ae_ckpt_dir / f"{name}_{cfg['model_type']}" / "checkpoint_best.pth"
        if ckpt_path.exists():
            ae_models[name] = load_ae(name, cfg, ckpt_path)

    active_diagnostics = {k: v for k, v in DIAGNOSTIC_CONFIGS.items() if k in ae_models}
    logger.info(f"Active diagnostics: {list(active_diagnostics.keys())}")

    # --- Build foundation model ---
    modality_configs = modality_configs_saved or {
        name: {"d_lat": cfg["d_lat"], "n_tokens": cfg["n_tokens"]}
        for name, cfg in active_diagnostics.items()
    }
    n_actuators = sum(cfg["n_channels"] for cfg in ACTUATOR_CONFIGS.values())
    dynamics_type = saved_args.get("dynamics_type", "cross_attention")

    model = PerceiverFoundationModel(
        modality_configs=modality_configs,
        d_model=saved_args.get("d_model", 256),
        n_latent=saved_args.get("n_latent", 128),
        n_actuators=n_actuators,
        encoder_layers=saved_args.get("encoder_layers", 1),
        processor_layers=saved_args.get("processor_layers", 1),
        decoder_layers=saved_args.get("decoder_layers", 2),
        decoder_self_attn_layers=saved_args.get("decoder_self_attn_layers", 0),
        dynamics_layers=saved_args.get("dynamics_layers", 2),
        n_heads=saved_args.get("n_heads", 8),
        dropout=0.0,  # eval mode
        dynamics_type=dynamics_type,
        actuator_configs=(ACTUATOR_CONFIGS if dynamics_type == "cross_attention" else None),
        ema_decay=saved_args.get("ema_decay", 0.996),
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    model.eval()
    logger.info(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # --- Build validation dataset ---
    stats = torch.load(args.stats_path, weights_only=False)
    all_signals = list(active_diagnostics.keys()) + list(ACTUATOR_CONFIGS.keys())

    data_dir = Path(args.data_dir)
    all_files = sorted(data_dir.glob("*_processed.h5"))
    random.seed(42)
    random.shuffle(all_files)
    if args.max_files:
        all_files = all_files[:args.max_files]
    n_val = max(1, int(0.1 * len(all_files)))
    val_files = all_files[:n_val]

    val_ds = TokamakMultiFileDataset(
        val_files,
        lengths_cache_path="lengths_diag_val.pt",
        preprocessing_stats=stats,
        input_signals=all_signals,
        chunk_duration_s=CHUNK_S,
        prediction_mode=False,
    )
    val_loader = make_dataloader(
        val_ds, batch_size=args.batch_size,
        num_workers=args.num_workers, shuffle=False,
        pin_memory=True,
    )

    # --- Accumulate per-modality metrics ---
    # For each modality, track:
    #   token_var:     variance of AE tokens (how much info they carry)
    #   roundtrip_mse: encode(target) -> decode -> MSE vs target AE tokens
    #   pred_mse:      encode(ctx) -> dynamics -> decode -> MSE vs target AE tokens
    #   copy_mse:      decode(encode(ctx)) -> MSE vs target AE tokens (no dynamics)
    metrics = {name: {"token_var": 0., "roundtrip_mse": 0.,
                      "pred_mse": 0., "copy_mse": 0., "n": 0}
               for name in active_diagnostics}

    use_cross_attn = dynamics_type == "cross_attention"

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= args.n_batches:
                break

            batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v
                     for k, v in batch.items()}

            # Split signals into context + 1 target window
            ctx_signals = {}
            tgt_signals = {}
            for name, cfg in active_diagnostics.items():
                if name not in batch:
                    continue
                ctx, tgts = split_window(batch[name], cfg["target_fs"], n_rollout=1)
                ctx_signals[name] = ctx
                tgt_signals[name] = tgts[0]

            if not ctx_signals:
                continue

            # Actuator extraction
            if use_cross_attn:
                act_ctx = actuator_context_window(batch, ACTUATOR_CONFIGS, stats)
                act_step_pairs = actuator_step_windows(
                    batch, ACTUATOR_CONFIGS, stats, n_rollout=1)
            else:
                act_ctx = None

            # AE encode context and target
            lat_ctx = encode_batch(ae_models, ctx_signals)
            lat_tgt = encode_batch(ae_models, tgt_signals)

            # --- Roundtrip: encode target -> decode (no dynamics) ---
            lat_tgt_perceiver = model.encode(lat_tgt, act_ctx)
            ae_tokens_roundtrip = model.decode(lat_tgt_perceiver)

            # --- Prediction: encode ctx -> dynamics -> decode ---
            lat_ctx_perceiver = model.encode(lat_ctx, act_ctx)
            if use_cross_attn:
                act_curr_sig, act_fut_sig = act_step_pairs[0]
                offset_ms = WINDOW_S * 1000
                lat_pred = model.dynamics(
                    lat_ctx_perceiver, act_curr_sig, act_fut_sig,
                    offset_ms=offset_ms, dt_ms=DT_S * 1000)
            else:
                from train_foundation_model import actuator_vectors
                act_pairs = actuator_vectors(batch, ACTUATOR_CONFIGS, stats, n_rollout=1)
                act_curr, act_fut = act_pairs[0]
                lat_pred = model.dynamics(lat_ctx_perceiver, act_curr, act_fut)
            ae_tokens_pred = model.decode(lat_pred)

            # --- Copy baseline: decode(encode(ctx)) vs target ---
            ae_tokens_copy = model.decode(lat_ctx_perceiver)

            # Compute per-modality metrics
            for name in active_diagnostics:
                if name not in lat_tgt:
                    continue
                tgt_tokens = lat_tgt[name]  # [B, n_tokens, d_lat]

                # Token variance
                var = tgt_tokens.var().item()

                # Roundtrip MSE
                rt_mse = F.mse_loss(ae_tokens_roundtrip[name], tgt_tokens).item()

                # Prediction MSE
                pr_mse = F.mse_loss(ae_tokens_pred[name], tgt_tokens).item()

                # Copy MSE (context tokens decoded vs target tokens)
                cp_mse = F.mse_loss(ae_tokens_copy[name], tgt_tokens).item()

                metrics[name]["token_var"] += var
                metrics[name]["roundtrip_mse"] += rt_mse
                metrics[name]["pred_mse"] += pr_mse
                metrics[name]["copy_mse"] += cp_mse
                metrics[name]["n"] += 1

            logger.info(f"  Batch {i+1}/{args.n_batches} processed")

    # --- Print results ---
    logger.info("\n" + "=" * 100)
    logger.info(f"{'Modality':<25s} {'TokenVar':>10s} {'Roundtrip':>10s} "
                f"{'Prediction':>10s} {'Copy':>10s} {'RT/Var':>10s} {'Pred/Var':>10s}")
    logger.info("-" * 100)

    for name in active_diagnostics:
        m = metrics[name]
        n = max(m["n"], 1)
        tv = m["token_var"] / n
        rt = m["roundtrip_mse"] / n
        pr = m["pred_mse"] / n
        cp = m["copy_mse"] / n
        rt_ratio = rt / max(tv, 1e-8)
        pr_ratio = pr / max(tv, 1e-8)

        logger.info(f"{name:<25s} {tv:10.6f} {rt:10.6f} {pr:10.6f} "
                    f"{cp:10.6f} {rt_ratio:10.4f} {pr_ratio:10.4f}")

    logger.info("=" * 100)
    logger.info("\nInterpretation:")
    logger.info("  RT/Var close to 0: Perceiver encode->decode preserves info well")
    logger.info("  RT/Var close to 1: Perceiver loses most information (bottleneck)")
    logger.info("  Pred/Var >> RT/Var: dynamics is the bottleneck")
    logger.info("  Copy ~ Pred: dynamics not learning (just copying context)")


if __name__ == "__main__":
    main()
