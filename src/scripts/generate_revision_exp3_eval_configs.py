#!/usr/bin/env python3
"""
Generate evaluation configs for revision/exp3 experiments.
Creates configs for:
- Base models: pt1_wide, pt1_narrow (2 models)
- FTWB1 models: pt1_wide_ftwb{1-7}, pt1_narrow_ftwb{1-7} (14 models)
- FTWB2 models: pt1_wide_ftwb2-{2,4,9,12,13,15}, pt1_narrow_ftwb2-{2,4,9,12,13,15} (12 models)

Total: 28 models × 15 evaluations each = 420 configs
"""

from pathlib import Path
import yaml

# Base paths
CONFIG_BASE = Path("/n/home12/cfpark00/datadir/WM_1/configs/revision/exp3/eval")
EXPERIMENT_BASE = Path("data/experiments/revision/exp3")
DATASET_BASE = Path("data/datasets")

# All tasks to evaluate
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]

# Model types
MODEL_TYPES = ["wide", "narrow"]

# FTWB2 experiment numbers (subset of 21 total)
FTWB2_EXPS = [2, 4, 9, 12, 13, 15]

def create_eval_config(experiment_name: str, task: str, is_atlantis: bool = True) -> dict:
    """Create evaluation config for a specific experiment and task."""

    if is_atlantis:
        dataset_path = DATASET_BASE / f"{task}_100k_atlantis_required"
        output_dir = EXPERIMENT_BASE / experiment_name / "evals" / f"atlantis_{task}"
    else:
        dataset_path = DATASET_BASE / f"{task}_1M_no_atlantis"
        output_dir = EXPERIMENT_BASE / experiment_name / "evals" / task

    config = {
        "checkpoints": "last",
        "dataset_path": str(dataset_path),
        "device": "cuda",
        "do_sample": False,
        "eval_batch_size": 512,
        "experiment_dir": str(EXPERIMENT_BASE / experiment_name),
        "max_generation_length": 256,
        "output_dir": str(output_dir),
        "plot_log_scale": False,
        "save_full_results": False,
        "seed": 42,
        "temperature": 0.0,
        "top_k": 1,
    }

    return config

def create_multitask_eval_config(experiment_name: str) -> dict:
    """Create multi-task evaluation config."""

    output_dir = EXPERIMENT_BASE / experiment_name / "evals" / "multi_task"

    config = {
        "checkpoints": "last",
        "cities_csv": "/n/home12/cfpark00/WM_1/data/datasets/cities/cities.csv",
        "dataset_path": "/n/home12/cfpark00/WM_1/data/datasets/multitask_pt1",
        "device": "cuda",
        "do_sample": False,
        "eval_batch_size": 512,
        "experiment_dir": str(EXPERIMENT_BASE / experiment_name),
        "max_generation_length": 256,
        "output_dir": str(output_dir),
        "plot_log_scale": False,
        "save_full_results": False,
        "seed": 42,
        "temperature": 0.0,
        "top_k": 1,
    }

    return config

def main():
    print("Generating evaluation configs for revision/exp3...")

    # Create base config directory
    CONFIG_BASE.mkdir(parents=True, exist_ok=True)

    configs_created = 0

    # Generate configs for each model type (wide, narrow)
    for model_type in MODEL_TYPES:
        type_dir = CONFIG_BASE / model_type
        type_dir.mkdir(parents=True, exist_ok=True)

        # Base model configs
        base_exp = f"pt1_{model_type}"
        base_dir = type_dir / "base"
        base_dir.mkdir(parents=True, exist_ok=True)

        # Atlantis tasks
        for task in TASKS:
            config = create_eval_config(base_exp, task, is_atlantis=True)
            config_path = base_dir / f"atlantis_{task}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs_created += 1

        # Normal tasks
        for task in TASKS:
            config = create_eval_config(base_exp, task, is_atlantis=False)
            config_path = base_dir / f"{task}.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs_created += 1

        # Multi-task
        config = create_multitask_eval_config(base_exp)
        config_path = base_dir / "multi_task.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)
        configs_created += 1

        print(f"  Created {len(TASKS)*2 + 1} configs for {base_exp}")

        # FTWB1 model configs (1-7)
        for exp_num in range(1, 8):
            ftwb1_exp = f"pt1_{model_type}_ftwb{exp_num}"
            ftwb1_dir = type_dir / f"ftwb1-{exp_num}"
            ftwb1_dir.mkdir(parents=True, exist_ok=True)

            # Atlantis tasks
            for task in TASKS:
                config = create_eval_config(ftwb1_exp, task, is_atlantis=True)
                config_path = ftwb1_dir / f"atlantis_{task}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                configs_created += 1

            # Normal tasks
            for task in TASKS:
                config = create_eval_config(ftwb1_exp, task, is_atlantis=False)
                config_path = ftwb1_dir / f"{task}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                configs_created += 1

            # Multi-task
            config = create_multitask_eval_config(ftwb1_exp)
            config_path = ftwb1_dir / "multi_task.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs_created += 1

        print(f"  Created {7 * (len(TASKS)*2 + 1)} configs for {model_type} ftwb1-{1} through ftwb1-{7}")

        # FTWB2 model configs (selected experiments)
        for exp_num in FTWB2_EXPS:
            ftwb2_exp = f"pt1_{model_type}_ftwb2-{exp_num}"
            ftwb2_dir = type_dir / f"ftwb2-{exp_num}"
            ftwb2_dir.mkdir(parents=True, exist_ok=True)

            # Atlantis tasks
            for task in TASKS:
                config = create_eval_config(ftwb2_exp, task, is_atlantis=True)
                config_path = ftwb2_dir / f"atlantis_{task}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                configs_created += 1

            # Normal tasks
            for task in TASKS:
                config = create_eval_config(ftwb2_exp, task, is_atlantis=False)
                config_path = ftwb2_dir / f"{task}.yaml"
                with open(config_path, 'w') as f:
                    yaml.dump(config, f, default_flow_style=False, sort_keys=False)
                configs_created += 1

            # Multi-task
            config = create_multitask_eval_config(ftwb2_exp)
            config_path = ftwb2_dir / "multi_task.yaml"
            with open(config_path, 'w') as f:
                yaml.dump(config, f, default_flow_style=False, sort_keys=False)
            configs_created += 1

        print(f"  Created {len(FTWB2_EXPS) * (len(TASKS)*2 + 1)} configs for {model_type} ftwb2 experiments")

        print(f"Completed {model_type}: {(1 + 7 + len(FTWB2_EXPS)) * (len(TASKS)*2 + 1)} configs\n")

    print("="*60)
    print(f"TOTAL: Created {configs_created} evaluation configs")
    print(f"Location: {CONFIG_BASE}")
    print("="*60)
    print(f"\nBreakdown:")
    print(f"  Wide base: 15 configs")
    print(f"  Wide FTWB1 (7 models): {7 * 15} configs")
    print(f"  Wide FTWB2 (6 models): {len(FTWB2_EXPS) * 15} configs")
    print(f"  Narrow base: 15 configs")
    print(f"  Narrow FTWB1 (7 models): {7 * 15} configs")
    print(f"  Narrow FTWB2 (6 models): {len(FTWB2_EXPS) * 15} configs")
    print(f"  Total: {2 * (1 + 7 + len(FTWB2_EXPS)) * 15} configs")

if __name__ == "__main__":
    main()
