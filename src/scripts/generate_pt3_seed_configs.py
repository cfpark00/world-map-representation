#!/usr/bin/env python3
"""
Generate training configs for pt3-{1-8} with seed1 and seed2.
"""

import yaml
from pathlib import Path

# Base config template
base_config = {
    "checkpointing": {
        "eval_steps": 0.005,
        "eval_strategy": "steps",
        "save_steps": 0.025,
        "save_strategy": "steps"
    },
    "dataset": {
        "max_sequence_length": 256
    },
    "eval": {
        "randomwalk": {
            "cities_csv": "data/datasets/cities/cities.csv"
        }
    },
    "logging": {
        "logging_steps": 10,
        "report_to": "none"
    },
    "model": {
        "hidden_size": 128,
        "init_scale": 0.1,
        "intermediate_size": 512,
        "num_attention_heads": 4,
        "num_hidden_layers": 6,
        "vocab_size": 98
    },
    "tokenizer_path": "data/tokenizers/default_tokenizer",
    "training": {
        "batch_size": 128,
        "eval_batch_size": 64,
        "learning_rate": 0.0003,
        "num_epochs": 14,
        "optimizer": "adamw",
        "scheduler": "linear_with_warmup",
        "warmup_steps": 50,
        "weight_decay": 0.01
    }
}

def generate_pt3_configs():
    """Generate pt3-{1-8} configs for seed1 and seed2."""
    base_dir = Path("configs/revision/exp5/pt3_seed")

    for pt3_num in range(1, 9):
        pt3_dir = base_dir / f"pt3-{pt3_num}"
        pt3_dir.mkdir(parents=True, exist_ok=True)

        for seed in [1, 2]:
            config = base_config.copy()
            config["dataset"]["path"] = f"data/datasets/pt3-{pt3_num}"
            config["output_dir"] = f"data/experiments/revision/exp5/pt3-{pt3_num}_seed{seed}"
            config["training"]["seed"] = seed

            config_path = pt3_dir / f"pt3-{pt3_num}_seed{seed}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            print(f"Created: {config_path}")

    print(f"\nGenerated 16 configs (8 pt3 variants × 2 seeds)")

if __name__ == "__main__":
    generate_pt3_configs()
