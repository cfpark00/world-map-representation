#!/usr/bin/env python3
"""
Generate bash scripts for running representation extraction and PCA timeline
for revision/exp1 FTWB1 and FTWB2 models.
"""

from pathlib import Path

# Base paths
CONFIG_BASE = Path("/configs")
SCRIPT_BASE = Path("/scripts/revision/exp1")
REPR_CONFIG_DIR = CONFIG_BASE / "revision" / "exp1" / "representation_extraction"
PCA_CONFIG_DIR = CONFIG_BASE / "revision" / "exp1" / "pca_timeline"

# FTWB1 task mapping
FTWB1_TASKS = {
    1: "distance",
    2: "trianglearea",
    3: "angle",
    4: "compass",
    5: "inside",
    6: "perimeter",
    7: "crossing"
}

# Training data for FTWB2 experiments
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

def generate_repr_scripts():
    """Generate representation extraction scripts."""
    print("Generating representation extraction scripts...")

    repr_script_dir = SCRIPT_BASE / "representation_extraction"
    repr_script_dir.mkdir(parents=True, exist_ok=True)

    # Individual scripts for each seed
    for seed_name in ['original', 'seed1', 'seed2', 'seed3']:
        # FTWB1
        ftwb1_lines = ["#!/bin/bash\n"]
        for ftwb1_num in range(1, 8):
            task = FTWB1_TASKS[ftwb1_num]
            if seed_name == 'original':
                exp_name = f"pt1_ftwb1-{ftwb1_num}"
            else:
                exp_name = f"pt1_{seed_name}_ftwb1-{ftwb1_num}"

            config_path = REPR_CONFIG_DIR / exp_name / f"{task}_firstcity_last_and_trans_l5.yaml"
            ftwb1_lines.append(f"uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite\n")

        ftwb1_script = repr_script_dir / f"extract_{seed_name}_ftwb1.sh"
        with open(ftwb1_script, 'w') as f:
            f.writelines(ftwb1_lines)
        ftwb1_script.chmod(0o755)

        # FTWB2
        ftwb2_lines = ["#!/bin/bash\n"]
        for ftwb2_num in range(1, 22):
            task = TRAINING_DATA_2TASK[ftwb2_num][0]
            exp_name = f"pt1_{seed_name}_ftwb2-{ftwb2_num}"

            config_path = REPR_CONFIG_DIR / exp_name / f"{task}_firstcity_last_and_trans_l5.yaml"
            ftwb2_lines.append(f"uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite\n")

        ftwb2_script = repr_script_dir / f"extract_{seed_name}_ftwb2.sh"
        with open(ftwb2_script, 'w') as f:
            f.writelines(ftwb2_lines)
        ftwb2_script.chmod(0o755)

    # Master scripts
    master_ftwb1 = repr_script_dir / "extract_all_ftwb1.sh"
    master_ftwb1_lines = ["#!/bin/bash\n"]
    for seed_name in ['original', 'seed1', 'seed2', 'seed3']:
        master_ftwb1_lines.append(f"bash scripts/revision/exp1/representation_extraction/extract_{seed_name}_ftwb1.sh\n")
    with open(master_ftwb1, 'w') as f:
        f.writelines(master_ftwb1_lines)
    master_ftwb1.chmod(0o755)

    master_ftwb2 = repr_script_dir / "extract_all_ftwb2.sh"
    master_ftwb2_lines = ["#!/bin/bash\n"]
    for seed_name in ['seed1', 'seed2', 'seed3']:
        master_ftwb2_lines.append(f"bash scripts/revision/exp1/representation_extraction/extract_{seed_name}_ftwb2.sh\n")
    with open(master_ftwb2, 'w') as f:
        f.writelines(master_ftwb2_lines)
    master_ftwb2.chmod(0o755)

    master_all = repr_script_dir / "extract_all.sh"
    master_all_lines = ["#!/bin/bash\n"]
    master_all_lines.append("bash scripts/revision/exp1/representation_extraction/extract_all_ftwb1.sh\n")
    master_all_lines.append("bash scripts/revision/exp1/representation_extraction/extract_all_ftwb2.sh\n")
    with open(master_all, 'w') as f:
        f.writelines(master_all_lines)
    master_all.chmod(0o755)

    print(f"  Created 8 seed-specific scripts (4 FTWB1 + 4 FTWB2)")
    print(f"  Created 3 master scripts")
    return 11

def generate_pca_scripts():
    """Generate PCA timeline visualization scripts."""
    print("\nGenerating PCA timeline scripts...")

    pca_script_dir = SCRIPT_BASE / "pca_timeline"
    pca_script_dir.mkdir(parents=True, exist_ok=True)

    # Individual scripts for each seed
    for seed_name in ['original', 'seed1', 'seed2', 'seed3']:
        # FTWB1
        ftwb1_lines = ["#!/bin/bash\n"]
        for ftwb1_num in range(1, 8):
            task = FTWB1_TASKS[ftwb1_num]
            if seed_name == 'original':
                exp_name = f"pt1_ftwb1-{ftwb1_num}"
            else:
                exp_name = f"pt1_{seed_name}_ftwb1-{ftwb1_num}"

            config_dir = PCA_CONFIG_DIR / exp_name
            for suffix in ["_l5.yaml", "_l5_na.yaml", "_l5_raw.yaml"]:
                config_path = config_dir / f"{task}_firstcity_last_and_trans{suffix}"
                ftwb1_lines.append(f"uv run python src/analysis/visualize_pca_3d_timeline.py {config_path} --overwrite\n")

        ftwb1_script = pca_script_dir / f"pca_{seed_name}_ftwb1.sh"
        with open(ftwb1_script, 'w') as f:
            f.writelines(ftwb1_lines)
        ftwb1_script.chmod(0o755)

        # FTWB2
        ftwb2_lines = ["#!/bin/bash\n"]
        for ftwb2_num in range(1, 22):
            task = TRAINING_DATA_2TASK[ftwb2_num][0]
            exp_name = f"pt1_{seed_name}_ftwb2-{ftwb2_num}"

            config_dir = PCA_CONFIG_DIR / exp_name
            for suffix in ["_l5.yaml", "_l5_na.yaml", "_l5_raw.yaml"]:
                config_path = config_dir / f"{task}_firstcity_last_and_trans{suffix}"
                ftwb2_lines.append(f"uv run python src/analysis/visualize_pca_3d_timeline.py {config_path} --overwrite\n")

        ftwb2_script = pca_script_dir / f"pca_{seed_name}_ftwb2.sh"
        with open(ftwb2_script, 'w') as f:
            f.writelines(ftwb2_lines)
        ftwb2_script.chmod(0o755)

    # Master scripts
    master_ftwb1 = pca_script_dir / "pca_all_ftwb1.sh"
    master_ftwb1_lines = ["#!/bin/bash\n"]
    for seed_name in ['original', 'seed1', 'seed2', 'seed3']:
        master_ftwb1_lines.append(f"bash scripts/revision/exp1/pca_timeline/pca_{seed_name}_ftwb1.sh\n")
    with open(master_ftwb1, 'w') as f:
        f.writelines(master_ftwb1_lines)
    master_ftwb1.chmod(0o755)

    master_ftwb2 = pca_script_dir / "pca_all_ftwb2.sh"
    master_ftwb2_lines = ["#!/bin/bash\n"]
    for seed_name in ['seed1', 'seed2', 'seed3']:
        master_ftwb2_lines.append(f"bash scripts/revision/exp1/pca_timeline/pca_{seed_name}_ftwb2.sh\n")
    with open(master_ftwb2, 'w') as f:
        f.writelines(master_ftwb2_lines)
    master_ftwb2.chmod(0o755)

    master_all = pca_script_dir / "pca_all.sh"
    master_all_lines = ["#!/bin/bash\n"]
    master_all_lines.append("bash scripts/revision/exp1/pca_timeline/pca_all_ftwb1.sh\n")
    master_all_lines.append("bash scripts/revision/exp1/pca_timeline/pca_all_ftwb2.sh\n")
    with open(master_all, 'w') as f:
        f.writelines(master_all_lines)
    master_all.chmod(0o755)

    print(f"  Created 8 seed-specific scripts (4 FTWB1 + 4 FTWB2)")
    print(f"  Created 3 master scripts")
    return 11

def main():
    print("="*60)
    print("Generating Revision Exp1 Execution Scripts")
    print("="*60)

    repr_scripts = generate_repr_scripts()
    pca_scripts = generate_pca_scripts()

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Total representation extraction scripts: {repr_scripts}")
    print(f"Total PCA timeline scripts: {pca_scripts}")
    print(f"Total scripts: {repr_scripts + pca_scripts}")
    print(f"\nScripts saved to:")
    print(f"  - {SCRIPT_BASE / 'representation_extraction'}")
    print(f"  - {SCRIPT_BASE / 'pca_timeline'}")

if __name__ == "__main__":
    main()
