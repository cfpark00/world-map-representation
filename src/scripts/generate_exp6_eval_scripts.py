#!/usr/bin/env python3
"""
Generate batch bash scripts for running revision/exp6 evaluations.

Organization:
- eval_pt1.sh: pt1 only (15 evals)
- eval_ftwb1_all.sh: ftwb1-1 to ftwb1-7 (105 evals)
- eval_ftwb2_part1.sh: ftwb2-1 to ftwb2-7 (105 evals)
- eval_ftwb2_part2.sh: ftwb2-8 to ftwb2-14 (105 evals)
- eval_ftwb2_part3.sh: ftwb2-15 to ftwb2-21 (105 evals)
- eval_all.sh: master script

Total: 5 scripts + 1 master = 6 scripts
"""

from pathlib import Path

SCRIPTS_BASE = Path("/scripts/revision/exp6/eval")
CONFIG_BASE = Path("configs/revision/exp6/eval")

TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]


def add_model_evals(lines, model_config_dir):
    """Add evaluation commands for a single model."""
    for task in TASKS:
        config_path = CONFIG_BASE / model_config_dir / f"atlantis_{task}.yaml"
        lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
    for task in TASKS:
        config_path = CONFIG_BASE / model_config_dir / f"{task}.yaml"
        lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
    config_path = CONFIG_BASE / model_config_dir / "multi_task.yaml"
    lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")


def main():
    print("Generating evaluation scripts for revision/exp6...")

    SCRIPTS_BASE.mkdir(parents=True, exist_ok=True)

    # Remove old scripts
    for old_script in SCRIPTS_BASE.glob("*.sh"):
        old_script.unlink()

    script_names = []

    # 1. eval_pt1.sh (15 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, "pt1")
    script_name = "eval_pt1.sh"
    script_path = SCRIPTS_BASE / script_name
    with open(script_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    script_names.append(script_name)
    print(f"  Created {script_name}: 15 evaluations")

    # 2. eval_ftwb1_all.sh (105 evals)
    lines = ["#!/bin/bash"]
    for i in range(1, 8):
        add_model_evals(lines, f"ftwb1-{i}")
    script_name = "eval_ftwb1_all.sh"
    script_path = SCRIPTS_BASE / script_name
    with open(script_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    script_names.append(script_name)
    print(f"  Created {script_name}: 105 evaluations")

    # 3. eval_ftwb2_part1.sh (ftwb2-1 to ftwb2-7, 105 evals)
    lines = ["#!/bin/bash"]
    for i in range(1, 8):
        add_model_evals(lines, f"ftwb2-{i}")
    script_name = "eval_ftwb2_part1.sh"
    script_path = SCRIPTS_BASE / script_name
    with open(script_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    script_names.append(script_name)
    print(f"  Created {script_name}: 105 evaluations")

    # 4. eval_ftwb2_part2.sh (ftwb2-8 to ftwb2-14, 105 evals)
    lines = ["#!/bin/bash"]
    for i in range(8, 15):
        add_model_evals(lines, f"ftwb2-{i}")
    script_name = "eval_ftwb2_part2.sh"
    script_path = SCRIPTS_BASE / script_name
    with open(script_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    script_names.append(script_name)
    print(f"  Created {script_name}: 105 evaluations")

    # 5. eval_ftwb2_part3.sh (ftwb2-15 to ftwb2-21, 105 evals)
    lines = ["#!/bin/bash"]
    for i in range(15, 22):
        add_model_evals(lines, f"ftwb2-{i}")
    script_name = "eval_ftwb2_part3.sh"
    script_path = SCRIPTS_BASE / script_name
    with open(script_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    script_path.chmod(0o755)
    script_names.append(script_name)
    print(f"  Created {script_name}: 105 evaluations")

    # Master script
    lines = ["#!/bin/bash"]
    for name in script_names:
        lines.append(f"bash scripts/revision/exp6/eval/{name}")
    master_name = "eval_all.sh"
    master_path = SCRIPTS_BASE / master_name
    with open(master_path, 'w') as f:
        f.write("\n".join(lines) + "\n")
    master_path.chmod(0o755)
    print(f"  Created {master_name}: master script")

    print()
    print("=" * 60)
    print(f"TOTAL: Created 6 scripts")
    print(f"Location: {SCRIPTS_BASE}")
    print("=" * 60)
    print(f"\nBreakdown:")
    print(f"  eval_pt1.sh: 15 evals")
    print(f"  eval_ftwb1_all.sh: 105 evals")
    print(f"  eval_ftwb2_part1.sh: 105 evals")
    print(f"  eval_ftwb2_part2.sh: 105 evals")
    print(f"  eval_ftwb2_part3.sh: 105 evals")
    print(f"  Total: 435 evaluations")


if __name__ == "__main__":
    main()
