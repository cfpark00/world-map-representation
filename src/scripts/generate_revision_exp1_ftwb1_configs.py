#!/usr/bin/env python3
"""
Generate training configs for revision/exp1 FTWB1 models.
FTWB1 = single-task fine-tuning with warmup and bias (7 tasks × 3 seeds = 21 models).

These are needed to compute expected generalization baseline for the paper.
"""

from pathlib import Path
import yaml

# Base paths
CONFIG_BASE = Path("/configs/revision/exp1/training")
EXPERIMENT_BASE = Path("data/experiments/revision/exp1")

# Seeds
SEEDS = [1, 2, 3]

# Task mapping for ftwb1 (single tasks)
TASK_MAPPING = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

def create_ftwb1_config(seed: int, exp_num: int) -> dict:
    """Create training config for a ftwb1 experiment."""

    task = TASK_MAPPING[exp_num]
    experiment_name = f"pt1_seed{seed}_ftwb1-{exp_num}"

    config = {
        "checkpointing": {
            "eval_steps": 0.005,
            "eval_strategy": "steps",
            "save_steps": 0.025,
            "save_strategy": "steps"
        },
        "dataset": {
            "max_sequence_length": 256,
            "path": f"data/datasets/ftwb1-{exp_num}"
        },
        "logging": {
            "logging_steps": 10,
            "report_to": "none"
        },
        "model": {
            "ckpt": str(EXPERIMENT_BASE / f"pt1_seed{seed}" / "checkpoints" / "final"),
            "hidden_size": 128,
            "init_scale": 0.1,
            "intermediate_size": 512,
            "num_attention_heads": 4,
            "num_hidden_layers": 6,
            "vocab_size": 98
        },
        "output_dir": str(EXPERIMENT_BASE / experiment_name),
        "tokenizer_path": "data/tokenizers/default_tokenizer",
        "training": {
            "batch_size": 128,
            "eval_batch_size": 64,
            "learning_rate": 1e-5,
            "num_epochs": 30,
            "optimizer": "adamw",
            "scheduler": "linear_with_warmup",
            "seed": seed,
            "warmup_steps": 50,
            "weight_decay": 0.01
        }
    }

    return config

def main():
    print("Generating FTWB1 training configs for revision/exp1...")

    # Create base config directory
    CONFIG_BASE.mkdir(parents=True, exist_ok=True)

    configs_created = 0

    # Generate configs for each seed
    for seed in SEEDS:
        seed_dir = CONFIG_BASE / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Create 7 ftwb1 configs (one per task)
        for exp_num in range(1, 8):
            task = TASK_MAPPING[exp_num]
            experiment_name = f"pt1_seed{seed}_ftwb1-{exp_num}"

            config = create_ftwb1_config(seed, exp_num)

            # Save config
            config_path = seed_dir / f"ftwb1-{exp_num}_{task}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)

            configs_created += 1

        print(f"  Created 7 configs for seed{seed} (ftwb1-1 through ftwb1-7)")

    print("\n" + "="*60)
    print(f"TOTAL: Created {configs_created} training configs")
    print(f"Location: {CONFIG_BASE}")
    print("="*60)
    print("\nTask mapping:")
    for exp_num, task in TASK_MAPPING.items():
        print(f"  ftwb1-{exp_num}: {task}")

if __name__ == "__main__":
    main()
