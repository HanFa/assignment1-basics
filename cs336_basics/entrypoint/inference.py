"""
Interactive inference script for trained Transformer Language Models.

Usage:
    # Run inference with a checkpoint
    uv run python -m cs336_basics.entrypoint.inference --checkpoint checkpoints/final_checkpoint
"""
import argparse
import json
import os
import torch
import torch.nn.functional as F
from typing import Optional

from cs336_basics.modules import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def load_model_from_checkpoint(checkpoint_path: str, device: str = 'cuda'):
    config_path = os.path.join(checkpoint_path, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")

    with open(config_path, 'r') as f:
        config_dict = json.load(f)

    cfg = config_dict['config']
    model = TransformerLM(
        d_model=cfg['d_model'],
        num_heads=cfg['num_heads'],
        d_ff=cfg['d_ff'],
        theta=cfg.get('rope_theta', 10000.0),
        max_seq_len=cfg['max_seq_len'],
        vocab_size=cfg['vocab_size'],
        num_layers=cfg['num_layers']
    ).to(device)
    model_path = os.path.join(checkpoint_path, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, cfg


@torch.no_grad()
def generate(
        model: TransformerLM,
        prompt_tokens: torch.Tensor,
        eos_idx: int,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: str = 'cuda'
) -> torch.Tensor:
    model.eval()
    tokens = prompt_tokens.to(device)

    for _ in range(max_length - len(prompt_tokens)):
        logits = model(tokens.unsqueeze(0))
        logits = logits[0, -1, :] / temperature
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = -float('Inf')

        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')

        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=0)
        if next_token.item() == eos_idx:
            break

    return tokens


def interactive_inference(
        model: TransformerLM,
        tokenizer: Optional[Tokenizer],
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        device: str = 'cuda'
):
    print("\n" + "=" * 60)
    print("Interactive Text Generation")
    print("=" * 60)
    print(f"Max length: {max_length}, Temperature: {temperature}")
    if top_k: print(f"Top-k: {top_k}")
    if top_p: print(f"Top-p: {top_p}")
    print("\nType 'quit' or 'exit' to stop.")
    print("=" * 60 + "\n")

    eos = '<|endoftext|>'
    eos_idx = tokenizer.bytes_to_idx[eos.encode('utf-8')]

    while True:
        prompt = input("Enter prompt: ")
        if prompt.lower() in ['quit', 'exit']:
            break

        prompt_tokens = torch.tensor(tokenizer.encode(prompt), dtype=torch.long)
        output_tokens = generate(
            model, prompt_tokens, eos_idx, max_length, temperature, top_k, top_p, device
        )
        output_text = tokenizer.decode(output_tokens.cpu().numpy().tolist())
        print("Generated text:")
        print("-" * 40)
        print(output_text)
        print("-" * 40 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Interactive inference with trained models")

    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to checkpoint directory")
    parser.add_argument("--tokenizer-vocab", type=str, default="pickles/vocab.pickle",
                        help="Path to tokenizer vocab file (optional)")
    parser.add_argument("--tokenizer-merges", type=str, default="pickles/merges.pickle",
                        help="Path to tokenizer merges file (optional)")
    parser.add_argument("--max_length", type=int, default=256,
                        help="Maximum generation length")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (lower = more focused)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="Top-k sampling (optional)")
    parser.add_argument("--top_p", type=float, default=None,
                        help="Top-p (nucleus) sampling (optional)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on")

    args = parser.parse_args()

    print(f"\nLoading model from {args.checkpoint}...")
    model, config = load_model_from_checkpoint(args.checkpoint, args.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully!")
    print(f"  Architecture: {config['num_layers']} layers, {config['d_model']} dim")
    print(f"  Parameters: {total_params:,}")
    print(f"  Device: {args.device}")

    tokenizer = Tokenizer.from_files(args.tokenizer_vocab, args.tokenizer_merges)
    print(f"Tokenizer loaded (vocab size: {len(tokenizer.vocab)})")

    interactive_inference(
        model, tokenizer,
        args.max_length, args.temperature, args.top_k, args.top_p,
        args.device
    )


if __name__ == "__main__":
    main()
