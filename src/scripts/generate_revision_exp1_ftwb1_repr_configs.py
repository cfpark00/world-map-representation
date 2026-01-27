#!/usr/bin/env python3
"""
Generate representation extraction configs for revision/exp1 FTWB1 models.
Creates 1 config per model (extracts using trained task) × 21 models = 21 configs.
"""

from pathlib import Path
import yaml

# Base paths
CONFIG_BASE = Path("/configs/revision/exp1/representation_extraction")
EXPERIMENT_BASE = Path("data/experiments/revision/exp1")

# Seeds
SEEDS = [1, 2, 3]

# Task mapping for ftwb1
TASK_MAPPING = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

def create_repr_config(experiment_name: str, task: str, seed: int) -> dict:
    """Create representation extraction config for a specific experiment and task."""

    output_dir = f"/data/experiments/revision/exp1/{experiment_name}/analysis_higher/{task}_firstcity_last_and_trans_l5"

    config = {
        "cities_csv": "data/datasets/cities/cities.csv",
        "device": "cuda",
        "experiment_dir": str(EXPERIMENT_BASE / experiment_name),
        "layers": [5],
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
        "save_repr_ckpts": [-2],
        "seed": 42,
    }

    return config

def main():
    print("Generating FTWB1 representation extraction configs for revision/exp1...")

    configs_created = 0

    # Generate configs for each seed
    for seed in SEEDS:
        seed_dir = CONFIG_BASE / f"seed{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True)

        # FTWB1 models - extract using trained task
        for exp_num in range(1, 8):
            ftwb1_exp = f"pt1_seed{seed}_ftwb1-{exp_num}"
            ftwb1_dir = seed_dir / f"ftwb1-{exp_num}"
            ftwb1_dir.mkdir(parents=True, exist_ok=True)

            # Use the trained task
            task = TASK_MAPPING[exp_num]

            config = create_repr_config(ftwb1_exp, task, seed)
            config_path = ftwb1_dir / f"{task}_firstcity_last_and_trans_l5.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs_created += 1

        print(f"  Created 7 configs for seed{seed} (ftwb1-1 through ftwb1-7)")

    print("\n" + "="*60)
    print(f"TOTAL: Created {configs_created} representation extraction configs")
    print(f"Location: {CONFIG_BASE}")
    print("="*60)

if __name__ == "__main__":
    main()
