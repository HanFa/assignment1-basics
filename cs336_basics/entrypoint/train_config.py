"""
Predefined training configurations for different model sizes.
"""

from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class TrainConfig:
    """Base training configuration with sensible defaults."""

    # Dataset configurations
    train_path: str = "data/TinyStoriesV2-GPT4-train.npy"
    valid_path: str = "data/TinyStoriesV2-GPT4-valid.npy"
    tokenizer_path: Optional[str] = None

    # Model configurations
    d_model: int = 768
    num_heads: int = 12
    d_ff: int = 3072
    rope_theta: float = 10000.0
    max_seq_len: int = 1024
    vocab_size: int = 10000
    num_layers: int = 12

    # Optimizer configurations
    optimizer_type: str = "adamw"
    lr: float = 6e-4
    weight_decay: float = 0.1
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8

    # Learning rate schedule
    use_lr_schedule: bool = False
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 2000
    cosine_cycle_iters: int = 100000

    # Gradient clipping
    use_gradient_clipping: bool = True
    max_grad_norm: float = 1.0

    # Training configurations
    batch_size: int = 32
    context_length: int = 1024
    max_steps: int = 100000
    eval_interval: int = 500
    eval_steps: int = 100
    log_interval: int = 10

    # Checkpoint configurations
    checkpoint_dir: str = "checkpoints"
    checkpoint_interval: int = 500
    resume_from_checkpoint: Optional[str] = None

    # Logging configurations
    use_mlflow: bool = True
    mlflow_tracking_uri: str = "http://mlflow.sutroplanet.com"
    mlflow_experiment_name: str = "cs336-transformer"
    mlflow_run_name: Optional[str] = None

    # Device configuration
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Random seed
    seed: int = 42


@dataclass
class SampleConfig(TrainConfig):
    """Sample configuration from the pdf spec."""
    d_model: int = 512
    num_heads: int = 16
    d_ff: int = 1344
    num_layers: int = 4
    max_seq_len: int = 256
    batch_size: int = 64
    context_length: int = 256
    lr: float = 6e-4
    max_lr: float = 6e-4
    min_lr: float = 6e-5
    warmup_iters: int = 2000
    mlflow_experiment_name: str = "sample-model-testing"


@dataclass
class TinyConfig(TrainConfig):
    """Tiny configuration for testing and debugging (< 10M parameters)."""

    d_model: int = 256
    num_heads: int = 8
    d_ff: int = 1024
    num_layers: int = 6
    max_seq_len: int = 512
    batch_size: int = 64
    context_length: int = 256
    max_steps: int = 10000
    eval_interval: int = 100
    lr: float = 5e-3
    max_lr: float = 5e-3
    min_lr: float = 1e-4
    warmup_iters: int = 500
    mlflow_experiment_name: str = "tiny-model-testing"


CONFIG_REGISTRY = {
    "sample": SampleConfig,
    "tiny": TinyConfig,
}


def get_config(name: str) -> TrainConfig:
    if name not in CONFIG_REGISTRY:
        available = ", ".join(CONFIG_REGISTRY.keys())
        raise ValueError(f"Unknown config: {name}. Available configs: {available}")

    return CONFIG_REGISTRY[name]()


def list_configs():
    return list(CONFIG_REGISTRY.keys())


def print_config_info(config: TrainConfig):
    params = estimate_params(config)

    print(f"Configuration Summary:")
    print(f"  Model dimension: {config.d_model}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Attention heads: {config.num_heads}")
    print(f"  FFN dimension: {config.d_ff}")
    print(f"  Vocabulary size: {config.vocab_size}")
    print(f"  Max sequence length: {config.max_seq_len}")
    print(f"  Estimated parameters: {params / 1e6:.1f}M")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.lr}")
    print(f"  Training steps: {config.max_steps}")


def estimate_params(config: TrainConfig) -> int:
    embedding_params = config.vocab_size * config.d_model
    layer_params = (
            4 * config.d_model * config.d_model +  # Attention (Q, K, V, O projections)
            3 * config.d_model * config.d_ff +  # SwiGLU FFN (weight1, weight2, weight3)
            2 * config.d_model  # 2 LayerNorms (RMSNorm weights)
    )
    output_params = config.d_model * config.vocab_size
    total_params = (
            embedding_params +
            config.num_layers * layer_params +
            output_params
    )

    return total_params
