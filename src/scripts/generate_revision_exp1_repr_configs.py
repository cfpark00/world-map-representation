#!/usr/bin/env python3
"""
Generate representation extraction configs for revision/exp1 experiments.
Creates configs for extracting representations from trained models.

For exp1: We have ftwb2 models (2 tasks per model) and base models.
We extract representations for layer 5, using distance task with firstcity_last_and_trans format.
"""

from pathlib import Path
import yaml

# Base paths
CONFIG_BASE = Path("/configs/revision/exp1/representation_extraction")
EXPERIMENT_BASE = Path("data/experiments/revision/exp1")

# Seeds
SEEDS = [1, 2, 3]

# Training data for FTWB2 experiments (2 tasks each)
TRAINING_DATA_2TASK = {
    1: ["distance", "trianglearea"],
    2: ["angle", "compass"],
    3: ["inside", "perimeter"],
    4: ["crossing", "distance"],
    5: ["trianglearea", "angle"],
    6: ["compass", "inside"],
    7: ["perimeter", "crossing"],
    8: ["angle", "distance"],
    9: ["compass", "trianglearea"],
    10: ["angle", "inside"],
    11: ["compass", "perimeter"],
    12: ["crossing", "inside"],
    13: ["distance", "perimeter"],
    14: ["crossing", "trianglearea"],
    15: ["compass", "distance"],
    16: ["inside", "trianglearea"],
    17: ["angle", "perimeter"],
    18: ["compass", "crossing"],
    19: ["distance", "inside"],
    20: ["perimeter", "trianglearea"],
    21: ["angle", "crossing"],
}

def create_repr_config(experiment_name: str, task: str, seed: int) -> dict:
    """Create representation extraction config for a specific experiment and task."""

    output_dir = f"/data/experiments/revision/exp1/{experiment_name}/analysis_higher/{task}_firstcity_last_and_trans_l5"

    config = {
        "cities_csv": "data/datasets/cities/cities.csv",
        "device": "cuda",
        "experiment_dir": str(EXPERIMENT_BASE / experiment_name),
        "layers": [5],  # Extract layer 5 representations
        "method": {
            "name": "linear"
        },
        "n_test_cities": 1250,
        "n_train_cities": 3250,
        "output_dir": output_dir,
        "perform_pca": True,
        "probe_test": "region:.* && city_id:^[1-9][0-9]{3,}$",
        "probe_train": "region:.* && city_id:^[1-9][0-9]{3,}$",
        "prompt_format": f"{task}_firstcity_last_and_trans",
        "save_repr_ckpts": [-2],  # Last checkpoint
        "seed": 42,
    }

    return config

def main():
    print("Generating representation extraction configs for revision/exp1...")

    # Create base config directory
    CONFIG_BASE.mkdir(parents=True, exist_ok=True)

    configs_created = 0

    # Generate configs for each seed
    for seed in SEEDS:
        seed_dir = CONFIG_BASE / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # Base model - extract using distance task (most common)
        base_exp = f"pt1_seed{seed}"
        base_dir = seed_dir / "base"
        base_dir.mkdir(parents=True, exist_ok=True)

        config = create_repr_config(base_exp, "distance", seed)
        config_path = base_dir / "distance_firstcity_last_and_trans_l5.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        configs_created += 1

        print(f"  Created config for {base_exp}")

        # FTWB2 models - extract using first trained task
        for exp_num in range(1, 22):
            ftwb2_exp = f"pt1_seed{seed}_ftwb2-{exp_num}"
            ftwb2_dir = seed_dir / f"ftwb2-{exp_num}"
            ftwb2_dir.mkdir(parents=True, exist_ok=True)

            # Use the first task from training data
            trained_tasks = TRAINING_DATA_2TASK[exp_num]
            task = trained_tasks[0]

            config = create_repr_config(ftwb2_exp, task, seed)
            config_path = ftwb2_dir / f"{task}_firstcity_last_and_trans_l5.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs_created += 1

            if exp_num % 7 == 0:
                print(f"  Created configs for ftwb2-1 through ftwb2-{exp_num}")

        print(f"Completed seed{seed}: {1 + 21} configs\n")

    print("="*60)
    print(f"TOTAL: Created {configs_created} representation extraction configs")
    print(f"Location: {CONFIG_BASE}")
    print("="*60)
    print("\nNote: Each config extracts layer 5 representations using the first")
    print("      trained task for each model (distance for base models).")

if __name__ == "__main__":
    main()
