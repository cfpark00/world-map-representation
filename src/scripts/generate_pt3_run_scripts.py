#!/usr/bin/env python3
"""Generate bash scripts for PT3 representation extraction and PCA timeline."""

from pathlib import Path

# PT3 task combinations - use first task from each triple
PT3_TASKS = {
    1: 'distance',
    2: 'compass',
    3: 'crossing',
    4: 'angle',
    5: 'perimeter',
    6: 'trianglearea',
    7: 'inside',
}

SEEDS = [1, 2]
LAYER = 5

def create_repr_extraction_scripts():
    """Create representation extraction scripts."""
    base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')
    script_dir = base_dir / 'scripts/revision/exp2/pt3_seed/extract_representations'
    script_dir.mkdir(parents=True, exist_ok=True)

    # Create individual scripts for each variant
    for variant_num in PT3_TASKS.keys():
        task_name = PT3_TASKS[variant_num]
        lines = ["#!/bin/bash"]
        lines.append("cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1")

        for seed in SEEDS:
            config_path = f'configs/revision/exp2/pt3_seed/extract_representations/pt3-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{LAYER}.yaml'
            lines.append(f'uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite')

        script_path = script_dir / f'extract_pt3-{variant_num}.sh'
        script_path.write_text('\n'.join(lines) + '\n')
        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    # Create master script
    lines = ["#!/bin/bash"]
    lines.append("cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1")
    for variant_num in PT3_TASKS.keys():
        lines.append(f'bash scripts/revision/exp2/pt3_seed/extract_representations/extract_pt3-{variant_num}.sh')

    script_path = script_dir / 'extract_all_pt3.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

def create_pca_timeline_scripts():
    """Create PCA timeline scripts."""
    base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')
    script_dir = base_dir / 'scripts/revision/exp2/pt3_seed/pca_timeline'
    script_dir.mkdir(parents=True, exist_ok=True)

    # Create scripts for each PCA type
    for pca_type in ['mixed', 'raw', 'na']:
        lines = ["#!/bin/bash"]

        for variant_num in PT3_TASKS.keys():
            task_name = PT3_TASKS[variant_num]
            for seed in SEEDS:
                config_path = f'configs/revision/exp2/pt3_seed/pca_timeline/pt3-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{LAYER}_{pca_type}.yaml'
                lines.append(f'uv run python src/analysis/visualize_pca_3d_timeline.py {config_path} --overwrite')

        script_path = script_dir / f'pca_pt3_{pca_type}.sh'
        script_path.write_text('\n'.join(lines) + '\n')
        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    # Create master script
    lines = ["#!/bin/bash"]
    lines.append("bash scripts/revision/exp2/pt3_seed/pca_timeline/pca_pt3_mixed.sh")
    lines.append("bash scripts/revision/exp2/pt3_seed/pca_timeline/pca_pt3_raw.sh")
    lines.append("bash scripts/revision/exp2/pt3_seed/pca_timeline/pca_pt3_na.sh")

    script_path = script_dir / 'pca_pt3_all.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

def main():
    print("Creating PT3 representation extraction scripts...")
    create_repr_extraction_scripts()

    print("\nCreating PT3 PCA timeline scripts...")
    create_pca_timeline_scripts()

    print("\nSummary:")
    print("  Representation extraction scripts: 8 (7 variants + 1 master)")
    print("  PCA timeline scripts: 4 (3 types + 1 master)")
    print(f"\nNote: PT3-8 excluded (not trained yet)")

if __name__ == '__main__':
    main()
