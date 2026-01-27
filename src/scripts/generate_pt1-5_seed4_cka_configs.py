#!/usr/bin/env python3
"""
Generate CKA configs for pt1-5_seed4 vs all other experiments.
This replaces pt1-5_seed3 configs since seed2 fails for the inside task.
"""

import yaml
from pathlib import Path

# All experiments to compare against
experiments = [
    # Original experiments
    ("pt1-1", "distance"),
    ("pt1-2", "trianglearea"),
    ("pt1-3", "angle"),
    ("pt1-4", "compass"),
    ("pt1-5", "inside"),
    ("pt1-6", "perimeter"),
    ("pt1-7", "crossing"),
    # Seed1 experiments
    ("pt1-1_seed1", "distance"),
    ("pt1-2_seed1", "trianglearea"),
    ("pt1-3_seed1", "angle"),
    ("pt1-4_seed1", "compass"),
    ("pt1-5_seed1", "inside"),
    ("pt1-6_seed1", "perimeter"),
    ("pt1-7_seed1", "crossing"),
    # Seed2 experiments (excluding pt1-5_seed2 since it failed)
    ("pt1-1_seed2", "distance"),
    ("pt1-2_seed2", "trianglearea"),
    ("pt1-3_seed2", "angle"),
    ("pt1-4_seed2", "compass"),
    ("pt1-6_seed2", "perimeter"),
    ("pt1-7_seed2", "crossing"),
    # Seed3 experiments (pt1-5_seed3 will be replaced by seed4)
    ("pt1-1_seed3", "distance"),
    ("pt1-2_seed3", "trianglearea"),
    ("pt1-3_seed3", "angle"),
    ("pt1-4_seed3", "compass"),
    ("pt1-5_seed3", "inside"),
    ("pt1-6_seed3", "perimeter"),
    ("pt1-7_seed3", "crossing"),
]

# Add pt1-5_seed4 vs itself
experiments.append(("pt1-5_seed4", "inside"))

layers = [3, 4, 5, 6]
base_dir = Path("configs/revision/exp4/cka_cross_seed")

def get_prompt_format(task):
    return f"{task}_firstcity_last_and_trans"

def get_experiment_path(exp_name):
    """Get the correct experiment path based on experiment name."""
    # Original experiments (no seed suffix) are in data/experiments/
    # Seeded experiments are in data/experiments/revision/exp4/
    if '_seed' not in exp_name:
        return f"data/experiments/{exp_name}"
    else:
        return f"data/experiments/revision/exp4/{exp_name}"

def create_cka_config(exp1_name, exp1_task, exp2_name, exp2_task, layer):
    """Create a CKA config comparing two experiments."""

    # Determine pair name (alphabetically sorted for consistency)
    if exp1_name < exp2_name:
        pair_name = f"{exp1_name}_vs_{exp2_name}"
        first_exp, first_task = exp1_name, exp1_task
        second_exp, second_task = exp2_name, exp2_task
    else:
        pair_name = f"{exp2_name}_vs_{exp1_name}"
        first_exp, first_task = exp2_name, exp2_task
        second_exp, second_task = exp1_name, exp1_task

    config = {
        "center_kernels": True,
        "checkpoint_steps": [328146],
        "city_filter": "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$",
        "exp1": {
            "name": first_exp,
            "repr_dir": f"{get_experiment_path(first_exp)}/analysis_higher/{get_prompt_format(first_task)}_l{layer}/representations",
            "task": first_task
        },
        "exp2": {
            "name": second_exp,
            "repr_dir": f"{get_experiment_path(second_exp)}/analysis_higher/{get_prompt_format(second_task)}_l{layer}/representations",
            "task": second_task
        },
        "kernel_type": "linear",
        "layer": layer,
        "output_dir": f"/data/experiments/revision/exp4/cka_analysis/{pair_name}/layer{layer}",
        "save_timeline_plot": False,
        "use_gpu": True
    }

    return pair_name, config

# Generate configs
print("Generating CKA configs for pt1-5_seed4...")

seed4_exp = ("pt1-5_seed4", "inside")
configs_created = 0

for other_exp, other_task in experiments:
    # Create configs for all layers
    for layer in layers:
        pair_name, config = create_cka_config(seed4_exp[0], seed4_exp[1],
                                              other_exp, other_task, layer)

        # Create output directory
        output_dir = base_dir / pair_name
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write config
        config_file = output_dir / f"layer{layer}.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        configs_created += 1

print(f"Created {configs_created} configs ({len(experiments)} pairs × {len(layers)} layers)")
print(f"Configs are in: {base_dir}/")
