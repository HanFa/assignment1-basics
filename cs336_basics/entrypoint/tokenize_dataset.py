#!/usr/bin/env python3
"""
This script loads the tokenizer from hardcoded pickle files:
  - Vocabulary: pickles/vocab_owt.pickle
  - Merges: pickles/merge_owt.pickle

Example usage:
  # Tokenize with default settings (uses pickles/vocab_owt.pickle and pickles/merge_owt.pickle)
  uv run python -m cs336_basics.entrypoint.tokenize_dataset --train_path data/TinyStoriesV2-GPT4-train.txt --valid_path data/TinyStoriesV2-GPT4-valid.txt
"""

import argparse
import os
import time
import logging
import json
import numpy as np
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

from cs336_basics.tokenizer import Tokenizer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tokenize TinyStories dataset for training"
    )
    parser.add_argument(
        "--train_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-train.txt",
        help="Path to training text file"
    )
    parser.add_argument(
        "--valid_path",
        type=str,
        default="data/TinyStoriesV2-GPT4-valid.txt",
        help="Path to validation text file"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=10 * 1024 * 1024,  # 10MB chunks
        help="Size of text chunks to process at once"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="uint16",
        choices=["uint8", "uint16", "uint32"],
        help="Data type for tokens (uint16 supports vocab up to 65k)"
    )

    return parser.parse_args()


def load_tokenizer() -> tuple[Tokenizer, int]:
    """Load the tokenizer from vocab and merges pickle files."""
    vocab_path = "pickles/vocab_owt.pickle"
    merges_path = "pickles/merges_owt.pickle"

    logger.info(f"Loading tokenizer from:")
    logger.info(f"  Vocab: {vocab_path}")
    logger.info(f"  Merges: {merges_path}")

    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocab file not found: {vocab_path}")
    if not os.path.exists(merges_path):
        raise FileNotFoundError(f"Merges file not found: {merges_path}")

    tokenizer = Tokenizer.from_files(vocab_path, merges_path)
    vocab_size = len(tokenizer.vocab)
    logger.info(f"Tokenizer loaded. Vocabulary size: {vocab_size}")
    return tokenizer, vocab_size


def tokenize_file(
        input_path: str,
        output_path: str,
        tokenizer: Tokenizer,
        chunk_size: int,
        dtype: str
) -> dict:
    """Tokenize a single file and save as .npy."""

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input file not found: {input_path}")

    file_size = os.path.getsize(input_path)
    logger.info(f"Processing: {input_path} ({file_size / (1024 ** 2):.2f} MB)")

    all_tokens = []
    num_documents = 0
    start_time = time.time()

    with open(input_path, 'r', encoding='utf-8') as f:
        with tqdm(total=file_size, unit='B', unit_scale=True, desc="Tokenizing") as pbar:
            buffer = ""

            while True:
                chunk = f.read(chunk_size)
                if not chunk:
                    # Process remaining buffer
                    if buffer.strip():
                        # Split by double newlines (document boundaries in TinyStories)
                        documents = buffer.split('\n\n')
                        for doc in documents:
                            if doc.strip():
                                tokens = tokenizer.encode(doc.strip())
                                all_tokens.extend(tokens)
                                # Add a special separator token if available
                                if hasattr(tokenizer, 'eos_token_id'):
                                    all_tokens.append(tokenizer.eos_token_id)
                                num_documents += 1
                    break

                buffer += chunk

                # Process complete documents (separated by double newlines)
                parts = buffer.split('\n\n')

                # Keep the last part in buffer (might be incomplete)
                buffer = parts[-1]

                # Process complete documents
                for doc in parts[:-1]:
                    if doc.strip():
                        tokens = tokenizer.encode(doc.strip())
                        all_tokens.extend(tokens)
                        # Add a special separator token if available
                        if hasattr(tokenizer, 'eos_token_id'):
                            all_tokens.append(tokenizer.eos_token_id)
                        num_documents += 1

                pbar.update(len(chunk))

    # Convert to numpy array
    logger.info(f"Converting {len(all_tokens):,} tokens to numpy array...")
    tokens_array = np.array(all_tokens, dtype=getattr(np, dtype))

    # Save to file
    logger.info(f"Saving to {output_path}")
    np.save(output_path, tokens_array)

    # Calculate statistics
    elapsed = time.time() - start_time
    stats = {
        'input_file': input_path,
        'output_file': output_path,
        'num_tokens': len(all_tokens),
        'num_documents': num_documents,
        'file_size_mb': file_size / (1024 ** 2),
        'output_size_mb': tokens_array.nbytes / (1024 ** 2),
        'compression_ratio': file_size / tokens_array.nbytes,
        'processing_time': elapsed,
        'tokens_per_second': len(all_tokens) / elapsed,
        'dtype': dtype
    }

    logger.info(f"Tokenization complete:")
    logger.info(f"  Documents: {num_documents:,}")
    logger.info(f"  Tokens: {len(all_tokens):,}")
    logger.info(f"  Time: {elapsed:.2f}s")
    logger.info(f"  Speed: {stats['tokens_per_second']:.0f} tokens/s")
    logger.info(f"  Compression: {stats['compression_ratio']:.2f}x")

    return stats


def main():
    """Main tokenization pipeline."""
    args = parse_args()

    tokenizer, vocab_size = load_tokenizer()
    dtype_max = np.iinfo(getattr(np, args.dtype)).max
    if vocab_size > dtype_max:
        logger.error(f"Vocabulary size {vocab_size} exceeds {args.dtype} max ({dtype_max})")
        logger.error(f"Use --dtype uint32 for large vocabularies")
        return 1

    all_stats = {}

    if args.train_path:
        train_output = args.train_path.replace('.txt', '.npy')
        logger.info("=" * 60)
        logger.info("Tokenizing training data...")
        logger.info("=" * 60)

        try:
            stats = tokenize_file(
                args.train_path,
                train_output,
                tokenizer,
                args.chunk_size,
                args.dtype
            )
            all_stats['train'] = stats
        except Exception as e:
            logger.error(f"Failed to tokenize training data: {e}")
            return 1

    # Process validation data
    if args.valid_path:
        valid_output = args.valid_path.replace('.txt', '.npy')
        logger.info("=" * 60)
        logger.info("Tokenizing validation data...")
        logger.info("=" * 60)

        try:
            stats = tokenize_file(
                args.valid_path,
                valid_output,
                tokenizer,
                args.chunk_size,
                args.dtype
            )
            all_stats['valid'] = stats
        except Exception as e:
            logger.error(f"Failed to tokenize validation data: {e}")
            return 1

    # Save metadata
    metadata = {
        'vocab_pickle': 'pickles/vocab_owt.pickle',
        'merges_pickle': 'pickles/merge_owt.pickle',
        'vocab_size': vocab_size,
        'dtype': args.dtype,
        'stats': all_stats,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }

    metadata_path = 'data/tokenization_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata saved to {metadata_path}")

    logger.info("=" * 60)
    logger.info("TOKENIZATION SUMMARY")
    logger.info("=" * 60)

    total_tokens = sum(s['num_tokens'] for s in all_stats.values())
    total_docs = sum(s['num_documents'] for s in all_stats.values())
    total_time = sum(s['processing_time'] for s in all_stats.values())

    logger.info(f"Total documents processed: {total_docs:,}")
    logger.info(f"Total tokens generated: {total_tokens:,}")
    logger.info(f"Total processing time: {total_time:.2f}s")
    logger.info(f"Average speed: {total_tokens / total_time:.0f} tokens/s")

    for split_name, stats in all_stats.items():
        logger.info(f"\n{split_name.upper()} split:")
        logger.info(f"  Output: {stats['output_file']}")
        logger.info(f"  Size: {stats['output_size_mb']:.2f} MB")
        logger.info(f"  Tokens: {stats['num_tokens']:,}")

    logger.info("\nTokenization completed successfully!")
    logger.info("You can now use these .npy files in your training script for faster loading.")

    return 0


if __name__ == '__main__':
    exit(main())
