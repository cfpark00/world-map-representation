#!/usr/bin/env python3
"""
Generate batch bash scripts for running revision/exp1 representation extraction.

Organization (similar to eval scripts):
- seed1: base+ftwb2-1~7, ftwb2-8~14, ftwb2-15~21
- seed2: base+ftwb2-1~7, ftwb2-8~14, ftwb2-15~21
- seed3: base+ftwb2-1~7, ftwb2-8~14, ftwb2-15~21
"""

from pathlib import Path

# Base paths
SCRIPTS_BASE = Path("/n/home12/cfpark00/datadir/WM_1/scripts/revision/exp1/representation_extraction")
CONFIG_BASE = Path("configs/revision/exp1/representation_extraction")

# Seeds
SEEDS = [1, 2, 3]

# Training data for determining which task to extract
TRAINING_DATA_2TASK = {
    1: "distance",
    2: "angle",
    3: "inside",
    4: "crossing",
    5: "trianglearea",
    6: "compass",
    7: "perimeter",
    8: "angle",
    9: "compass",
    10: "angle",
    11: "compass",
    12: "crossing",
    13: "distance",
    14: "crossing",
    15: "compass",
    16: "inside",
    17: "angle",
    18: "compass",
    19: "distance",
    20: "perimeter",
    21: "angle",
}

def generate_batch_script(seed: int, batch: int, exp_range: list, include_base: bool = False):
    """
    Generate a batch script for representation extraction.

    Args:
        seed: Seed number (1, 2, or 3)
        batch: Batch number (1, 2, or 3)
        exp_range: List of ftwb2 experiment numbers
        include_base: Whether to include base model extraction
    """

    lines = [
        "#!/bin/bash",
        "cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1"
    ]

    # Add base model extraction if this is batch 1
    if include_base:
        config_path = CONFIG_BASE / f"seed{seed}" / "base" / "distance_firstcity_last_and_trans_l5.yaml"
        lines.append(f"uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite")

    # Add ftwb2 model extractions
    for exp_num in exp_range:
        task = TRAINING_DATA_2TASK[exp_num]
        config_path = CONFIG_BASE / f"seed{seed}" / f"ftwb2-{exp_num}" / f"{task}_firstcity_last_and_trans_l5.yaml"
        lines.append(f"uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite")

    return "\n".join(lines) + "\n"

def main():
    print("Generating batch representation extraction scripts for revision/exp1...")

    # Create scripts directory
    SCRIPTS_BASE.mkdir(parents=True, exist_ok=True)

    scripts_created = 0

    # Define batches
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
                script_name = f"extract_seed{seed}_base_ftwb2-1-7.sh"
            elif batch_num == 2:
                script_name = f"extract_seed{seed}_ftwb2-8-14.sh"
            else:
                script_name = f"extract_seed{seed}_ftwb2-15-21.sh"

            script_path = SCRIPTS_BASE / script_name

            # Write script
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Make executable
            script_path.chmod(0o755)

            # Count models
            num_models = len(exp_range) + (1 if include_base else 0)

            print(f"  Created {script_name}: {num_models} extractions")
            scripts_created += 1

    print("\n" + "="*60)
    print(f"TOTAL: Created {scripts_created} batch scripts")
    print(f"Location: {SCRIPTS_BASE}")
    print("="*60)
    print("\nBatch organization:")
    print("  Seed1: base+1-7 (8 extractions), 8-14 (7 extractions), 15-21 (7 extractions)")
    print("  Seed2: base+1-7 (8 extractions), 8-14 (7 extractions), 15-21 (7 extractions)")
    print("  Seed3: base+1-7 (8 extractions), 8-14 (7 extractions), 15-21 (7 extractions)")
    print(f"\nTotal extractions: {3 * (8 + 7 + 7)} = 66")

if __name__ == "__main__":
    main()
