#!/usr/bin/env python3
"""
Generate batch training scripts for revision/exp1 FTWB1 models.
Creates scripts to train 21 models (3 seeds × 7 tasks).
"""

from pathlib import Path

# Base paths
SCRIPTS_BASE = Path("/scripts/revision/exp1/training")
CONFIG_BASE = Path("configs/revision/exp1/training")

# Seeds
SEEDS = [1, 2, 3]

# Task mapping
TASK_MAPPING = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

def main():
    print("Generating FTWB1 training scripts for revision/exp1...")

    # Create scripts directory
    SCRIPTS_BASE.mkdir(parents=True, exist_ok=True)

    scripts_created = 0

    # Generate individual scripts for each seed
    for seed in SEEDS:
        lines = ["#!/bin/bash"]

        for exp_num in range(1, 8):
            task = TASK_MAPPING[exp_num]
            config_path = CONFIG_BASE / f"seed{seed}" / f"ftwb1-{exp_num}_{task}.yaml"
            lines.append(f"uv run python src/training/train.py {config_path} --overwrite")

        script_name = f"train_seed{seed}_ftwb1_all.sh"
        script_path = SCRIPTS_BASE / script_name

        # Write script
        with open(script_path, 'w') as f:
            f.write("\n".join(lines) + "\n")

        # Make executable
        script_path.chmod(0o755)

        print(f"  Created {script_name}: 7 training runs")
        scripts_created += 1

    # Create master script to run all seeds sequentially
    master_lines = ["#!/bin/bash"]
    for seed in SEEDS:
        master_lines.append(f"bash scripts/revision/exp1/training/train_seed{seed}_ftwb1_all.sh")

    master_path = SCRIPTS_BASE / "train_all_ftwb1_sequential.sh"
    with open(master_path, 'w') as f:
        f.write("\n".join(master_lines) + "\n")
    master_path.chmod(0o755)

    print(f"  Created train_all_ftwb1_sequential.sh: master script")
    scripts_created += 1

    print("\n" + "="*60)
    print(f"TOTAL: Created {scripts_created} training scripts")
    print(f"Location: {SCRIPTS_BASE}")
    print("="*60)
    print("\nScripts created:")
    print("  - train_seed1_ftwb1_all.sh (7 models)")
    print("  - train_seed2_ftwb1_all.sh (7 models)")
    print("  - train_seed3_ftwb1_all.sh (7 models)")
    print("  - train_all_ftwb1_sequential.sh (master)")
    print(f"\nTotal models to train: 21 (3 seeds × 7 tasks)")

if __name__ == "__main__":
    main()
