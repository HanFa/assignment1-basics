"""
Simple hyperparameter sweep for learning rate optimization.

Usage:
    # Basic learning rate sweep
    uv run python -m cs336_basics.entrypoint.sweep --config tiny --lr_values 1e-4 5e-4 1e-3 5e-3 1e-2

    # With custom max_steps for faster iteration
    uv run python -m cs336_basics.entrypoint.sweep --config tiny --max_steps 5000
"""
import argparse
import json
import os
from dataclasses import asdict, replace
from typing import List, Dict, Any
import logging

import numpy as np
import torch
import torch.nn as nn

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

from cs336_basics.modules import TransformerLM
from cs336_basics.optimizers import AdamW, SGD
from cs336_basics.entrypoint.train_config import get_config, list_configs
from cs336_basics.entrypoint.train import (
    load_data_with_memmap_and_assert,
    train_model
)


def run_learning_rate_sweep(base_config, lr_values: List[float],
                           train_data: np.ndarray, valid_data: np.ndarray) -> Dict[str, Any]:
    """Run sweep over different learning rates and return results."""

    results = []
    best_lr = None
    best_val_loss = float('inf')

    logger.info(f"\nStarting learning rate sweep with {len(lr_values)} values")
    logger.info(f"Learning rates to test: {lr_values}")
    logger.info(f"Max steps per trial: {base_config.max_steps}\n")

    for i, lr in enumerate(lr_values):
        logger.info(f"\n{'='*50}")
        logger.info(f"Trial {i+1}/{len(lr_values)}: LR = {lr:.2e}")
        logger.info(f"{'='*50}")

        # Create config with new learning rate
        config = replace(base_config, lr=lr)

        # Set seed for reproducibility
        torch.manual_seed(config.seed + i)
        np.random.seed(config.seed + i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(config.seed + i)

        # Initialize model
        model = TransformerLM(
            d_model=config.d_model,
            num_heads=config.num_heads,
            d_ff=config.d_ff,
            theta=config.rope_theta,
            max_seq_len=config.max_seq_len,
            vocab_size=config.vocab_size,
            num_layers=config.num_layers
        ).to(config.device)

        # Initialize optimizer
        if config.optimizer_type == "adamw":
            optimizer = AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=config.weight_decay,
                betas=config.betas,
                eps=config.eps
            )
        else:
            optimizer = SGD(model.parameters(), lr=lr)

        try:
            trial_results = train_model(
                model=model,
                optimizer=optimizer,
                train_data=train_data,
                valid_data=valid_data,
                cfg=config,
                start_step=0,
                use_mlflow=True
            )

            val_loss = trial_results['best_val_loss']
            logger.info(f"Trial completed - Best Val Loss: {val_loss:.4f}")

            results.append({
                'lr': lr,
                'best_val_loss': val_loss,
                'final_val_loss': trial_results['final_val_loss'],
                'final_train_loss': trial_results['final_train_loss']
            })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_lr = lr

        except Exception as e:
            logger.error(f"Trial failed: {e}")
            results.append({
                'lr': lr,
                'status': 'failed',
                'error': str(e)
            })

        # Clean up GPU memory
        del model
        del optimizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        'results': results,
        'best_lr': best_lr,
        'best_val_loss': best_val_loss
    }


def main():
    parser = argparse.ArgumentParser(description="Simple learning rate sweep")

    parser.add_argument("--config", type=str, required=True,
                        choices=list_configs(),
                        help="Base configuration to use")

    parser.add_argument("--lr_values", type=float, nargs='+',
                        default=[1e-4, 5e-4, 1e-3, 5e-3, 1e-2],
                        help="Learning rates to sweep (default: 1e-4 5e-4 1e-3 5e-3 1e-2)")

    parser.add_argument("--max_steps", type=int, default=5000,
                        help="Max training steps per trial (default: 5000)")

    parser.add_argument("--output_dir", type=str, default="sweep_results",
                        help="Directory to save results")

    args = parser.parse_args()

    # Load base configuration
    config = get_config(args.config)

    # Override settings for faster sweep
    config.max_steps = args.max_steps
    config.log_interval = max(10, args.max_steps // 100)
    config.eval_interval = max(100, args.max_steps // 20)
    config.checkpoint_interval = args.max_steps + 1  # Don't checkpoint during sweep

    logger.info(f"Base config: {args.config}")
    logger.info(f"Max steps: {config.max_steps}")

    # Load data
    logger.info("Loading data...")
    train_data = load_data_with_memmap_and_assert(
        config.vocab_size, config.train_path
    )
    valid_data = load_data_with_memmap_and_assert(
        config.vocab_size, config.valid_path
    )

    # Run sweep
    sweep_results = run_learning_rate_sweep(
        config, args.lr_values, train_data, valid_data
    )

    # Print summary
    print("\n" + "="*60)
    print("SWEEP RESULTS SUMMARY")
    print("="*60)

    print("\nAll results:")
    for result in sweep_results['results']:
        if 'best_val_loss' in result:
            print(f"  LR={result['lr']:.2e}: Val Loss={result['best_val_loss']:.4f}")
        else:
            print(f"  LR={result['lr']:.2e}: FAILED")

    if sweep_results['best_lr']:
        print(f"\nBest learning rate: {sweep_results['best_lr']:.2e}")
        print(f"Best validation loss: {sweep_results['best_val_loss']:.4f}")

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, "lr_sweep_results.json")
    with open(output_file, 'w') as f:
        json.dump(sweep_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()