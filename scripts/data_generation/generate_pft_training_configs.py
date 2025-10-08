#!/usr/bin/env python3
"""Generate training configs for all pftX-Y combinations."""

import os
from pathlib import Path

# Map numbers to task names
TASK_MAP = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

def generate_training_config(x, y):
    """Generate a training config for pftX-Y."""
    return f"""# Output directory - all outputs go here
output_dir: "data/experiments/pft{x}-{y}"

# Dataset - PADDED VERSION
dataset:
  path: "data/datasets/pft{x}-{y}"
  max_sequence_length: 256

# Tokenizer
tokenizer_path: "data/tokenizers/default_tokenizer"

# Model (Qwen2.5-like)
model:
  vocab_size: 98  # New ASCII tokenizer vocab size
  hidden_size: 128
  num_hidden_layers: 6
  num_attention_heads: 4
  intermediate_size: 512
  init_scale: 0.1  # Standard deviation for weight initialization (GPT-2 style)
  ckpt: data/experiments/pt1-1/checkpoints/final

# Training
training:
  batch_size: 128  # Larger batch size since we have 120K samples
  eval_batch_size: 64
  num_epochs: 30
  optimizer: "adamw"
  learning_rate: 1e-5
  weight_decay: 0.01
  scheduler: "linear_with_warmup"
  warmup_steps: 50  # More warmup for larger dataset
  seed: 42

# Checkpointing
checkpointing:
  save_strategy: "steps"
  save_steps: 0.025
  eval_strategy: "steps"
  eval_steps: 0.005

logging:
  logging_steps: 10
  report_to: "none"

# Evaluation settings for the primary task
eval:
  {TASK_MAP[x]}:
    cities_csv: "data/datasets/cities/cities.csv"
"""

def main():
    output_dir = Path("/n/home12/cfpark00/WM_1/configs/training/pftset")

    # Generate all combinations
    for x in range(1, 8):
        for y in range(1, 8):
            if x != y:
                config_path = output_dir / f"pft{x}-{y}.yaml"
                if not config_path.exists():  # Don't overwrite existing configs
                    print(f"Creating {config_path}")
                    config_path.write_text(generate_training_config(x, y))
                else:
                    print(f"Skipping {config_path} (already exists)")

if __name__ == "__main__":
    main()