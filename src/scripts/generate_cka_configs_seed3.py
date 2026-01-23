#!/usr/bin/env python3
"""
Generate CKA analysis configs for all seed3 comparisons (28x28 matrix).
This will create configs for:
- seed3 vs seed3 (same and different tasks)
- seed3 vs seed2 (all combinations)
- seed3 vs seed1 (all combinations)
- seed3 vs original (all combinations)
"""

import yaml
from pathlib import Path
from itertools import combinations_with_replacement

# Task configurations
tasks = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

# Seed configurations
seeds = ["", "_seed1", "_seed2", "_seed3"]  # original, seed1, seed2, seed3

# Base paths for representations
repr_paths = {
    1: "distance_firstcity_last_and_trans",
    2: "trianglearea_firstcity_last_and_trans",
    3: "angle_firstcity_last_and_trans",
    4: "compass_firstcity_last_and_trans",
    5: "inside_firstcity_last_and_trans",
    6: "perimeter_firstcity_last_and_trans",
    7: "crossing_firstcity_last_and_trans"
}

def generate_seed3_configs():
    """Generate CKA configs only for comparisons involving seed3."""
    base_config_dir = Path("configs/revision/exp4/cka_cross_seed")
    base_config_dir.mkdir(parents=True, exist_ok=True)

    config_count = 0

    # Generate all pairs of (task, seed) combinations where at least one is seed3
    models = []
    for task_id in range(1, 8):
        for seed in seeds:
            models.append((task_id, seed))

    # Only create configs where at least one model is seed3
    for i, (task1, seed1) in enumerate(models):
        for j, (task2, seed2) in enumerate(models):
            if i > j:  # Skip duplicates (only upper triangle + diagonal)
                continue

            # Skip if neither is seed3
            if seed1 != "_seed3" and seed2 != "_seed3":
                continue

            model1_name = f"pt1-{task1}{seed1}"
            model2_name = f"pt1-{task2}{seed2}"

            # Create config for layers 4, 5, and 6
            for layer in [4, 5, 6]:
                config = {
                    "center_kernels": True,
                    "checkpoint_steps": [328146],
                    "city_filter": "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$",
                    "exp1": {
                        "name": model1_name,
                        "repr_dir": f"data/experiments/revision/exp4/{model1_name}/analysis_higher/{repr_paths[task1]}_l{layer}/representations",
                        "task": tasks[task1]
                    },
                    "exp2": {
                        "name": model2_name,
                        "repr_dir": f"data/experiments/revision/exp4/{model2_name}/analysis_higher/{repr_paths[task2]}_l{layer}/representations",
                        "task": tasks[task2]
                    },
                    "kernel_type": "linear",
                    "layer": layer,
                    "output_dir": f"/n/home12/cfpark00/WM_1/data/experiments/revision/exp4/cka_analysis/{model1_name}_vs_{model2_name}/layer{layer}",
                    "save_timeline_plot": False,
                    "use_gpu": True
                }

                # Create output directory
                config_dir = base_config_dir / f"{model1_name}_vs_{model2_name}"
                config_dir.mkdir(parents=True, exist_ok=True)

                # Save config
                config_path = config_dir / f"layer{layer}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False)

                config_count += 1

    print(f"Generated {config_count} seed3 CKA configs ({config_count // 3} pairs × 3 layers)")
    return config_count

if __name__ == "__main__":
    count = generate_seed3_configs()
    print(f"Total configs: {count}")
