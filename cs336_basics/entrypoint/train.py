"""
Training script for Transformer Language Models.

Trains a transformer model on pre-tokenized data with configurable architecture,
optimization, and logging settings. Supports checkpointing and MLflow tracking.

Usage:
    # Use a predefined configuration
    uv run python -m cs336_basics.entrypoint.train --config gpt2-small

    # Use tiny config for testing
    uv run python -m cs336_basics.entrypoint.train --config tiny

    # Override specific parameters
    uv run python -m cs336_basics.entrypoint.train --config gpt2-small --batch_size 64 --lr 0.001

    # Resume from checkpoint
    uv run python -m cs336_basics.entrypoint.train --config tinystories --resume_from_checkpoint checkpoints/checkpoint_step_500

Available configs: tiny, tinystories, efficient, gpt2-small, gpt2-medium, gpt2-large

Note: Requires pre-tokenized .npy files. Use tokenize_dataset.py to prepare data.
"""
import argparse
import os
import time
import math
from dataclasses import asdict
import json
import logging
import traceback
import numpy as np
import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch


def to_bits(loss_nats): return loss_nats / math.log(2.0)


def to_ppl(loss_nats): return math.exp(loss_nats)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

from cs336_basics.modules import TransformerLM, cross_entropy_loss, gradient_clipping
from cs336_basics.optimizers import AdamW, SGD, get_lr_cosine_schedule
from cs336_basics.dataloaders import load_inputs_target_from_np_dataset
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint

from cs336_basics.entrypoint.train_config import (
    TrainConfig,
    get_config,
    list_configs,
    print_config_info
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a Transformer Language Model using predefined configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"Available configs: {', '.join(list_configs())}"
    )

    # Config selection (required)
    parser.add_argument("--config", type=str, required=True,
                        choices=list_configs(),
                        help="Select a predefined configuration")

    # Optional: resume from checkpoint
    parser.add_argument("--resume_from_checkpoint", type=str, default=None,
                        help="Path to checkpoint to resume from")

    # Optional: MLflow settings
    parser.add_argument("--use_mlflow", action="store_true",
                        help="Use MLflow for logging")
    parser.add_argument("--mlflow_run_name", type=str, default=None,
                        help="MLflow run name (optional)")

    args = parser.parse_args()

    # Load the predefined configuration
    cfg = get_config(args.config)
    logger.info(f"Using configuration: {args.config}")

    # Override only the essential runtime parameters
    if args.resume_from_checkpoint:
        cfg.resume_from_checkpoint = args.resume_from_checkpoint

    if args.use_mlflow:
        cfg.use_mlflow = args.use_mlflow

    if args.mlflow_run_name:
        cfg.mlflow_run_name = args.mlflow_run_name

    # Print configuration summary
    print_config_info(cfg)

    return cfg


def load_data_with_memmap(file_path: str, dtype=np.uint16):
    """Load tokenized data using memory mapping for efficiency."""
    npy_path = file_path.replace('.txt', '.npy')
    if not os.path.exists(npy_path):
        logger.error(
            f"No pre-tokenized data found in {npy_path}. Consider pre-tokenizing your data for faster loading.")
        raise FileNotFoundError(f"Pre-tokenized data not found: {npy_path}")

    logger.info(f"Loading pre-tokenized data from {npy_path}")
    return np.load(npy_path, mmap_mode='r')


def load_data_with_memmap_and_assert(max_vocab_size: int, file_path: str, dtype=np.uint16):
    data = load_data_with_memmap(file_path, dtype)
    min_id, max_id = int(data.min()), int(data.max())
    assert max_id < max_vocab_size, (
        f"Token id {max_id} exceeds vocab_size {max_vocab_size}. "
        f"Retokenize or set vocab_size to cover the data (e.g., 50257)."
    )
    return data


def evaluate(model: nn.Module, eval_data: np.ndarray, cfg: TrainConfig):
    """Evaluate the model on validation data."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for _ in range(cfg.eval_steps):
            inputs, targets = load_inputs_target_from_np_dataset(
                eval_data, cfg.batch_size, cfg.context_length, cfg.device
            )

            inputs = inputs.to(cfg.device).long()
            targets = targets.to(cfg.device).long()

            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)
            total_loss += loss.item()

    model.train()
    return total_loss / cfg.eval_steps


def main():
    cfg = parse_args()

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.seed)
    torch.autograd.set_detect_anomaly(True)

    # Create checkpoint directory if it doesn't exist
    os.makedirs(cfg.checkpoint_dir, exist_ok=True)

    # Initialize MLflow if requested and log parameters
    if cfg.use_mlflow:
        mlflow.set_tracking_uri(cfg.mlflow_tracking_uri)
        mlflow.set_experiment(cfg.mlflow_experiment_name)
        mlflow.start_run(run_name=cfg.mlflow_run_name)
        for key, value in asdict(cfg).items():
            if isinstance(value, (int, float, str, bool)):
                mlflow.log_param(key, value)

    # Load data with memory mapping
    logger.info("Loading training data...")
    train_data = load_data_with_memmap_and_assert(cfg.vocab_size, cfg.train_path)
    logger.info(f"Training data shape: {train_data.shape}")

    logger.info("Loading validation data...")
    valid_data = load_data_with_memmap_and_assert(cfg.vocab_size, cfg.valid_path)
    logger.info(f"Validation data shape: {valid_data.shape}")

    # Initialize model
    logger.info(f"Initializing model with {cfg.num_layers} layers...")
    model = TransformerLM(
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        d_ff=cfg.d_ff,
        theta=cfg.rope_theta,
        max_seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        num_layers=cfg.num_layers
    ).to(cfg.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Initialize optimizer
    if cfg.optimizer_type == "adamw":
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps
        )
    else:  # sgd
        optimizer = SGD(model.parameters(), lr=cfg.lr)

    # Resume from checkpoint if specified
    start_step = 0
    if cfg.resume_from_checkpoint:
        logger.info(f"Resuming from checkpoint: {cfg.resume_from_checkpoint}")
        start_step = load_checkpoint(cfg.resume_from_checkpoint, model, optimizer)
        logger.info(f"Resumed from step {start_step}")

    # Training loop
    logger.info(f"Starting training on {cfg.device}...")
    logger.info(f"Batch size: {cfg.batch_size}, Context length: {cfg.context_length}")
    logger.info(f"Max steps: {cfg.max_steps}")

    model.train()
    losses = []
    start_time = time.time()

    try:
        for step in range(start_step, cfg.max_steps):
            inputs, targets = load_inputs_target_from_np_dataset(
                train_data, cfg.batch_size, cfg.context_length, cfg.device
            )
            assert torch.allclose(inputs[:, 1:], targets[:, :-1]), \
                "Targets must be the next token (inputs shifted right by 1). Check your dataloader."

            # Move to device and ensure correct dtype
            inputs = inputs.to(cfg.device).long()
            targets = targets.to(cfg.device).long()

            # Adjust learning rate if using schedule
            if cfg.use_lr_schedule:
                lr = get_lr_cosine_schedule(
                    step, cfg.max_lr, cfg.min_lr,
                    cfg.warmup_iters, cfg.cosine_cycle_iters
                )
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr
            else:
                lr = cfg.lr

            # Forward pass
            optimizer.zero_grad()
            logits = model(inputs)
            loss = cross_entropy_loss(logits, targets)

            # Backward pass
            loss.backward()

            # Gradient clipping
            if cfg.use_gradient_clipping:
                gradient_clipping(model.parameters(), cfg.max_grad_norm)

            # Optimizer step
            optimizer.step()

            # Track losses
            losses.append(loss.item())

            # Logging
            if (step + 1) % cfg.log_interval == 0:
                avg_loss = np.mean(losses[-cfg.log_interval:])
                elapsed = time.time() - start_time
                tokens_per_sec = (step + 1 - start_step) * cfg.batch_size * cfg.context_length / elapsed

                logger.info(f"Step {step + 1}/{cfg.max_steps} | Loss: {avg_loss:.4f} nats |"
                            f" {to_bits(avg_loss):.4f} bits | PPL: {to_ppl(avg_loss):.2f} | "
                            f"LR: {lr:.2e} | Tokens/sec: {tokens_per_sec:.0f}")

                if cfg.use_mlflow:
                    mlflow.log_metrics({
                        "train_loss": avg_loss,
                        "learning_rate": lr,
                        "tokens_per_sec": tokens_per_sec
                    }, step=step + 1)

            # Evaluation
            if (step + 1) % cfg.eval_interval == 0:
                val_loss = evaluate(model, valid_data, cfg)
                logger.info(f"Validation loss at step {step + 1}: {val_loss:.4f}")

                if cfg.use_mlflow:
                    mlflow.log_metrics({
                        "val_loss": val_loss
                    }, step=step + 1)

            # Checkpointing
            if (step + 1) % cfg.checkpoint_interval == 0:
                checkpoint_path = os.path.join(cfg.checkpoint_dir, f"checkpoint_step_{step + 1}")
                logger.info(f"Saving checkpoint to {checkpoint_path}")

                # Create checkpoint directory
                os.makedirs(checkpoint_path, exist_ok=True)

                # Save model and optimizer states
                torch.save(model.state_dict(), os.path.join(checkpoint_path, "model.pt"))
                torch.save(optimizer.state_dict(), os.path.join(checkpoint_path, "optimizer.pt"))

                # Save configuration and step
                config_dict = {
                    "iteration": step + 1,
                    "config": asdict(cfg)
                }
                with open(os.path.join(checkpoint_path, "config.json"), 'w') as f:
                    json.dump(config_dict, f, indent=2)

                logger.info(f"Checkpoint saved successfully")

                # Also log checkpoint to MLflow if enabled
                if cfg.use_mlflow:
                    mlflow.log_artifacts(checkpoint_path, f"checkpoints/step_{step + 1}")

    except Exception as e:
        # Log the error
        logger.error(f"Training failed with error: {e}")
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

        # Write error to file
        error_file = "error.txt"
        with open(error_file, 'w') as f:
            f.write(f"Training failed at step {step if 'step' in locals() else 'initialization'}\n")
            f.write(f"Error: {str(e)}\n\n")
            f.write(f"Full traceback:\n{traceback.format_exc()}\n\n")
            f.write(f"Configuration:\n{json.dumps(asdict(cfg), indent=2)}\n")

        # Upload error file to MLflow if enabled
        if cfg.use_mlflow:
            logger.info("Uploading error.txt to MLflow...")
            mlflow.log_artifact(error_file, artifact_path="errors")
            mlflow.set_tag("training_status", "failed")
            mlflow.set_tag("error_message", str(e)[:250])  # MLflow has a limit on tag length
            mlflow.end_run(status="FAILED")

        # Re-raise the exception
        raise

    # Final evaluation
    final_val_loss = evaluate(model, valid_data, cfg)
    logger.info("Training completed!")
    logger.info(f"Final validation loss: {final_val_loss:.4f}")
    logger.info(f"Total time: {time.time() - start_time:.1f} seconds")

    # Save final checkpoint
    final_checkpoint_path = os.path.join(cfg.checkpoint_dir, "final_checkpoint")
    logger.info(f"Saving final checkpoint to {final_checkpoint_path}")
    os.makedirs(final_checkpoint_path, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(final_checkpoint_path, "model.pt"))
    torch.save(optimizer.state_dict(), os.path.join(final_checkpoint_path, "optimizer.pt"))

    config_dict = {
        "iteration": cfg.max_steps,
        "config": asdict(cfg),
        "final_val_loss": final_val_loss
    }
    with open(os.path.join(final_checkpoint_path, "config.json"), 'w') as f:
        json.dump(config_dict, f, indent=2)

    if cfg.use_mlflow:
        # Log final metrics
        mlflow.log_metrics({
            "final_val_loss": final_val_loss,
            "total_training_time": time.time() - start_time
        })

        # Log the model artifact
        logger.info("Logging model to MLflow...")
        mlflow.pytorch.log_model(
            model,
            "model",
            registered_model_name=f"transformer_lm_{cfg.d_model}d_{cfg.num_layers}l"
        )

        # Log checkpoint directory as artifact
        mlflow.log_artifacts(final_checkpoint_path, "final_checkpoint")

        # End the MLflow run
        mlflow.end_run()

    logger.info("Training script completed successfully!")


if __name__ == '__main__':
    main()
