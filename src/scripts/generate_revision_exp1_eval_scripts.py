#!/usr/bin/env python3
"""
Generate 9 batch bash scripts for running revision/exp1 evaluations.

Organization (3 seeds × 3 batches):
- seed1: base+ftwb2-1~7, ftwb2-8~14, ftwb2-15~21
- seed2: base+ftwb2-1~7, ftwb2-8~14, ftwb2-15~21
- seed3: base+ftwb2-1~7, ftwb2-8~14, ftwb2-15~21
"""

from pathlib import Path

# Base paths
SCRIPTS_BASE = Path("/scripts/revision/exp1/eval")
CONFIG_BASE = Path("configs/revision/exp1/eval")

# All tasks to evaluate
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]

# Seeds
SEEDS = [1, 2, 3]

def generate_batch_script(seed: int, batch: int, exp_range: list, include_base: bool = False):
    """
    Generate a batch script for a specific seed and experiment range.

    Args:
        seed: Seed number (1, 2, or 3)
        batch: Batch number (1, 2, or 3)
        exp_range: List of ftwb2 experiment numbers (e.g., [1, 2, 3, 4, 5, 6, 7])
        include_base: Whether to include base model evaluation
    """

    lines = ["#!/bin/bash"]

    # Add base model evaluations if this is batch 1
    if include_base:
        # Atlantis tasks
        for task in TASKS:
            config_path = CONFIG_BASE / f"seed{seed}" / "base" / f"atlantis_{task}.yaml"
            lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
        # Normal tasks
        for task in TASKS:
            config_path = CONFIG_BASE / f"seed{seed}" / "base" / f"{task}.yaml"
            lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
        # Multi-task
        config_path = CONFIG_BASE / f"seed{seed}" / "base" / "multi_task.yaml"
        lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")

    # Add ftwb2 model evaluations
    for exp_num in exp_range:
        # Atlantis tasks
        for task in TASKS:
            config_path = CONFIG_BASE / f"seed{seed}" / f"ftwb2-{exp_num}" / f"atlantis_{task}.yaml"
            lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
        # Normal tasks
        for task in TASKS:
            config_path = CONFIG_BASE / f"seed{seed}" / f"ftwb2-{exp_num}" / f"{task}.yaml"
            lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
        # Multi-task
        config_path = CONFIG_BASE / f"seed{seed}" / f"ftwb2-{exp_num}" / "multi_task.yaml"
        lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")

    return "\n".join(lines) + "\n"

def main():
    print("Generating batch evaluation scripts for revision/exp1...")

    # Create scripts directory
    SCRIPTS_BASE.mkdir(parents=True, exist_ok=True)

    scripts_created = 0

    # Define batches: [exp_nums]
    batches = [
        (1, [1, 2, 3, 4, 5, 6, 7], True),   # Batch 1: base + ftwb2-1~7
        (2, [8, 9, 10, 11, 12, 13, 14], False),  # Batch 2: ftwb2-8~14
        (3, [15, 16, 17, 18, 19, 20, 21], False),  # Batch 3: ftwb2-15~21
    ]

    # Generate scripts for each seed
    for seed in SEEDS:
        for batch_num, exp_range, include_base in batches:
            # Generate script content
            script_content = generate_batch_script(seed, batch_num, exp_range, include_base)

            # Determine script name
            if batch_num == 1:
                script_name = f"eval_seed{seed}_base_ftwb2-1-7.sh"
            elif batch_num == 2:
                script_name = f"eval_seed{seed}_ftwb2-8-14.sh"
            else:
                script_name = f"eval_seed{seed}_ftwb2-15-21.sh"

            script_path = SCRIPTS_BASE / script_name

            # Write script
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Make executable
            script_path.chmod(0o755)

            # Count commands (7 atlantis + 7 normal + 1 multi_task = 15 per model)
            num_models = len(exp_range) + (1 if include_base else 0)
            num_commands = num_models * 15

            print(f"  Created {script_name}: {num_commands} evaluations")
            scripts_created += 1

    print("\n" + "="*60)
    print(f"TOTAL: Created {scripts_created} batch scripts")
    print(f"Location: {SCRIPTS_BASE}")
    print("="*60)
    print("\nBatch organization (15 evals per model):")
    print("  Seed1: base+1-7 (120 evals), 8-14 (105 evals), 15-21 (105 evals)")
    print("  Seed2: base+1-7 (120 evals), 8-14 (105 evals), 15-21 (105 evals)")
    print("  Seed3: base+1-7 (120 evals), 8-14 (105 evals), 15-21 (105 evals)")
    print(f"\nTotal evaluations: {3 * (120 + 105 + 105)} = 990")

if __name__ == "__main__":
    main()
