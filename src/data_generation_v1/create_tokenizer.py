#!/usr/bin/env python3
"""
Create BPE tokenizer with hand-made vocab for WM1 project.
Handles ASCII characters (excluding space) with no multi-character tokens.

Usage:
    python src/create_tokenizer.py configs/tokenizers/wm1_ascii_tokenizer.yaml [--overwrite]
"""

import argparse
import yaml
import json
import shutil
from pathlib import Path
import sys
sys.path.append('.')
from src.utils import init_directory
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import Split
from transformers import PreTrainedTokenizerFast


def create_tokenizer(config, save_path):
    """Create BPE tokenizer with ASCII chars (no multi-char tokens)."""
    
    special_tokens = config.get('special_tokens', {})
    
    # Build vocab: special tokens first, then chars
    vocab = {}
    token_id = 0
    
    # 1. Special tokens (in specific order for consistency)
    for key in ['pad_token', 'unk_token', 'bos_token', 'eos_token']:
        if special_tokens.get(key):
            vocab[special_tokens[key]] = token_id
            token_id += 1
    
    # 2. Single ASCII characters (from custom_chars)
    if config.get('custom_chars'):
        custom_chars = config.get('custom_chars', '')
        for char in custom_chars:
            if char not in vocab:
                vocab[char] = token_id
                token_id += 1
    
    # 3. Additional tokens if specified (we don't use any for WM1)
    additional_tokens = config.get('additional_tokens', [])
    merges = []
    for token in additional_tokens:
        if token not in vocab:
            vocab[token] = token_id
            token_id += 1
            # Add merge rules for multi-char tokens
            if len(token) == 2:
                merges.append((token[0], token[1]))
    
    print(f"Vocab size: {len(vocab)}")
    print(f"Special tokens: {[special_tokens.get(k) for k in ['pad_token', 'unk_token', 'bos_token', 'eos_token'] if special_tokens.get(k)]}")
    print(f"Merge rules: {len(merges)}")
    
    # Create BPE tokenizer with vocab and merges
    tokenizer = Tokenizer(BPE(
        vocab=vocab, 
        merges=merges, 
        unk_token=special_tokens.get('unk_token', '<unk>')
    ))
    
    # Pre-tokenizer: Split on space delimiter and REMOVE it
    # Space is ONLY a delimiter, not a token in our vocabulary
    tokenizer.pre_tokenizer = Split(
        pattern=' ',  # Split on space delimiter
        behavior='removed'  # Remove the delimiter - it's not a token!
    )
    
    # Save the raw tokenizer
    tokenizer.save(str(save_path / "tokenizer.json"))
    
    # Create HF wrapper for compatibility
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=str(save_path / "tokenizer.json"),
        model_max_length=512,  # Reasonable max for distance calculations
        **special_tokens
    )
    
    # Save HF version
    hf_tokenizer.save_pretrained(str(save_path))
    
    # Also save a simple JSON mapping for backward compatibility
    vocab_mapping = {
        "vocab": vocab,
        "id_to_token": {v: k for k, v in vocab.items()},
        "special_tokens": special_tokens,
        "vocab_size": len(vocab)
    }
    
    with open(save_path / "vocab_mapping.json", 'w') as f:
        json.dump(vocab_mapping, f, indent=2)
    
    print(f"Saved tokenizer to {save_path}")
    print(f"  - tokenizer.json: Raw tokenizer")
    print(f"  - tokenizer_config.json: HF config")
    print(f"  - vocab_mapping.json: Simple vocab mapping")
    
    return hf_tokenizer


def test_tokenizer(tokenizer):
    """Test the tokenizer with WM1-style inputs."""
    test_cases = [
        "dist(c_1234,c_5678)=90",
        "<bos>dist(c_42,c_99)=123<eos>",
        "Hello, World!",  # Test punctuation
        "x=y+z*w",  # Test math symbols
    ]
    
    print("\nTesting tokenizer:")
    for text in test_cases:
        # Encode
        encoded = tokenizer.encode(text)
        # Decode
        decoded = tokenizer.decode(encoded)
        print(f"  Input:   '{text}'")
        print(f"  Encoded: {encoded[:10]}{'...' if len(encoded) > 10 else ''}")
        print(f"  Decoded: '{decoded}'")
        print(f"  Length:  {len(encoded)} tokens")
        print()


def main():
    parser = argparse.ArgumentParser(description='Create tokenizer from YAML configuration')
    parser.add_argument('config_path', type=str, help='Path to YAML configuration file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing tokenizer')
    parser.add_argument('--debug', action='store_true', help='Debug mode with testing')
    
    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config_path}")
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate config
    if 'output_dir' not in config:
        raise ValueError("FATAL: 'output_dir' is required in config")
    
    # Initialize output directory
    output_path = init_directory(config['output_dir'], overwrite=args.overwrite)
    
    # Copy config to output directory for reproducibility
    config_copy_path = output_path / 'config.yaml'
    shutil.copy(args.config_path, config_copy_path)
    print(f"Copied config to {config_copy_path}")
    
    # Create tokenizer
    tokenizer = create_tokenizer(config, output_path)
    
    # Test if in debug mode
    if args.debug:
        test_tokenizer(tokenizer)
    
    print("\nTokenizer created successfully!")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()