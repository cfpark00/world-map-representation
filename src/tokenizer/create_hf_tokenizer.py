"""
Create a HuggingFace-compatible character-level tokenizer for WM_1 experiments.
"""

import json
import os
from typing import Optional
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.tokenizer.tokenizer_config import VOCAB


def create_hf_tokenizer(save_path: Optional[str] = None):
    """
    Create a HuggingFace tokenizer by saving the necessary files.
    """
    if not save_path:
        save_path = "outputs/tokenizer/wm1_tokenizer"
    
    os.makedirs(save_path, exist_ok=True)
    
    # Save vocab.json
    vocab_path = os.path.join(save_path, "vocab.json")
    with open(vocab_path, 'w') as f:
        json.dump(VOCAB, f, indent=2)
    
    # Create tokenizer.json
    tokenizer_json = {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": [
            {"id": 0, "content": "<pad>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 1, "content": "<bos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True},
            {"id": 2, "content": "<eos>", "single_word": False, "lstrip": False, "rstrip": False, "normalized": False, "special": True}
        ],
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Sequence",
            "pretokenizers": [
                {
                    "type": "Split",
                    "pattern": {"String": ""},
                    "behavior": "Removed",
                    "invert": False
                }
            ]
        },
        "post_processor": None,
        "decoder": {
            "type": "Replace",
            "pattern": {"String": " "},
            "content": ""
        },
        "model": {
            "type": "WordLevel",
            "vocab": VOCAB,
            "unk_token": "<pad>"
        }
    }
    
    tokenizer_json_path = os.path.join(save_path, "tokenizer.json")
    with open(tokenizer_json_path, 'w') as f:
        json.dump(tokenizer_json, f, indent=2)
    
    # Save special_tokens_map.json
    special_tokens_map = {
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>"
    }
    special_tokens_path = os.path.join(save_path, "special_tokens_map.json")
    with open(special_tokens_path, 'w') as f:
        json.dump(special_tokens_map, f, indent=2)
    
    # Save tokenizer_config.json
    tokenizer_config = {
        "tokenizer_class": "PreTrainedTokenizerFast",
        "bos_token": "<bos>",
        "eos_token": "<eos>",
        "pad_token": "<pad>",
        "model_max_length": 512,
        "padding_side": "right",
        "truncation_side": "right",
        "clean_up_tokenization_spaces": False
    }
    config_path = os.path.join(save_path, "tokenizer_config.json")
    with open(config_path, 'w') as f:
        json.dump(tokenizer_config, f, indent=2)
    
    print(f"Tokenizer files saved to {save_path}")
    
    # Load and return it
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(save_path)
    
    return tokenizer


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Create HuggingFace tokenizer for WM1')
    parser.add_argument('--save-path', type=str, default='outputs/tokenizer/wm1_tokenizer',
                       help='Path to save the tokenizer')
    
    args = parser.parse_args()
    
    # Create and save tokenizer
    tokenizer = create_hf_tokenizer(args.save_path)
    
    print(f"\nTokenizer ready! Load it with:")
    print(f">>> from transformers import AutoTokenizer")
    print(f">>> tokenizer = AutoTokenizer.from_pretrained('{args.save_path}')")