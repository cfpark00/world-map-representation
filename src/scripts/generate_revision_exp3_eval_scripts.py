#!/usr/bin/env python3
"""
Generate batch bash scripts for running revision/exp3 evaluations.

Organization (chunked into ~30 evals per script, i.e., 2 models):
- Wide: 7 scripts (base+ftwb1-1, ftwb1-2+3, ftwb1-4+5, ftwb1-6+7, ftwb2-2+4, ftwb2-9+12, ftwb2-13+15)
- Narrow: 7 scripts (same pattern)

Total: 14 scripts + 2 master scripts
"""

from pathlib import Path

# Base paths
SCRIPTS_BASE = Path("/n/home12/cfpark00/datadir/WM_1/scripts/revision/exp3/eval")
CONFIG_BASE = Path("configs/revision/exp3/eval")

# All tasks to evaluate
TASKS = ["distance", "trianglearea", "angle", "compass", "inside", "perimeter", "crossing"]

# Model types
MODEL_TYPES = ["wide", "narrow"]

# FTWB2 experiment numbers
FTWB2_EXPS = [2, 4, 9, 12, 13, 15]

def add_model_evals(lines, model_type, model_name):
    """Add evaluation commands for a single model."""
    # Atlantis tasks
    for task in TASKS:
        config_path = CONFIG_BASE / model_type / model_name / f"atlantis_{task}.yaml"
        lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
    # Normal tasks
    for task in TASKS:
        config_path = CONFIG_BASE / model_type / model_name / f"{task}.yaml"
        lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")
    # Multi-task
    config_path = CONFIG_BASE / model_type / model_name / "multi_task.yaml"
    lines.append(f"uv run python src/eval/evaluate_checkpoints.py {config_path} --overwrite")

def generate_chunked_scripts(model_type):
    """Generate chunked scripts for a model type."""

    scripts = []

    # Chunk 1: base + ftwb1-1 (30 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, model_type, "base")
    add_model_evals(lines, model_type, "ftwb1-1")
    scripts.append(("eval_{}_base_ftwb1-1.sh".format(model_type), lines, 30))

    # Chunk 2: ftwb1-2 + ftwb1-3 (30 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, model_type, "ftwb1-2")
    add_model_evals(lines, model_type, "ftwb1-3")
    scripts.append(("eval_{}_ftwb1-2_ftwb1-3.sh".format(model_type), lines, 30))

    # Chunk 3: ftwb1-4 + ftwb1-5 (30 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, model_type, "ftwb1-4")
    add_model_evals(lines, model_type, "ftwb1-5")
    scripts.append(("eval_{}_ftwb1-4_ftwb1-5.sh".format(model_type), lines, 30))

    # Chunk 4: ftwb1-6 + ftwb1-7 (30 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, model_type, "ftwb1-6")
    add_model_evals(lines, model_type, "ftwb1-7")
    scripts.append(("eval_{}_ftwb1-6_ftwb1-7.sh".format(model_type), lines, 30))

    # Chunk 5: ftwb2-2 + ftwb2-4 (30 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, model_type, "ftwb2-2")
    add_model_evals(lines, model_type, "ftwb2-4")
    scripts.append(("eval_{}_ftwb2-2_ftwb2-4.sh".format(model_type), lines, 30))

    # Chunk 6: ftwb2-9 + ftwb2-12 (30 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, model_type, "ftwb2-9")
    add_model_evals(lines, model_type, "ftwb2-12")
    scripts.append(("eval_{}_ftwb2-9_ftwb2-12.sh".format(model_type), lines, 30))

    # Chunk 7: ftwb2-13 + ftwb2-15 (30 evals)
    lines = ["#!/bin/bash"]
    add_model_evals(lines, model_type, "ftwb2-13")
    add_model_evals(lines, model_type, "ftwb2-15")
    scripts.append(("eval_{}_ftwb2-13_ftwb2-15.sh".format(model_type), lines, 30))

    return scripts

def generate_master_script(model_type, script_names):
    """Generate master script that runs all chunks."""
    lines = ["#!/bin/bash"]
    for name in script_names:
        lines.append(f"bash scripts/revision/exp3/eval/{name}")
    return "\n".join(lines) + "\n"

def main():
    print("Generating chunked evaluation scripts for revision/exp3...")
    print("Chunking strategy: ~30 evals (2 models) per script\n")

    # Create scripts directory
    SCRIPTS_BASE.mkdir(parents=True, exist_ok=True)

    scripts_created = 0
    total_evals = 0

    # Generate scripts for each model type
    for model_type in MODEL_TYPES:
        print(f"Processing {model_type}...")

        chunks = generate_chunked_scripts(model_type)
        script_names = []

        for script_name, lines, num_evals in chunks:
            script_path = SCRIPTS_BASE / script_name

            script_content = "\n".join(lines) + "\n"
            with open(script_path, 'w') as f:
                f.write(script_content)
            script_path.chmod(0o755)

            print(f"  Created {script_name}: {num_evals} evaluations")
            script_names.append(script_name)
            scripts_created += 1
            total_evals += num_evals

        # Create master script
        master_script = generate_master_script(model_type, script_names)
        master_name = f"eval_{model_type}_all.sh"
        master_path = SCRIPTS_BASE / master_name

        with open(master_path, 'w') as f:
            f.write(master_script)
        master_path.chmod(0o755)

        print(f"  Created {master_name}: master script")
        scripts_created += 1
        print()

    print("="*60)
    print(f"TOTAL: Created {scripts_created} scripts")
    print(f"Location: {SCRIPTS_BASE}")
    print("="*60)
    print("\nChunking breakdown:")
    print("  Per model type: 7 chunked scripts (30 evals each) + 1 master")
    print("  Wide: 7 chunks + 1 master = 8 scripts")
    print("  Narrow: 7 chunks + 1 master = 8 scripts")
    print(f"  Total scripts: 16")
    print(f"  Total evaluations: {total_evals}")

if __name__ == "__main__":
    main()
