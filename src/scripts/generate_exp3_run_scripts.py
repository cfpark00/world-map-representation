#!/usr/bin/env python3
"""Generate bash scripts for exp3 representation extraction and PCA timeline."""

from pathlib import Path

# Task mappings
TASKS = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing'
}

def create_repr_extraction_scripts():
    """Create representation extraction scripts."""
    base_dir = Path('')
    script_dir = base_dir / 'scripts/revision/exp3/representation_extraction'
    script_dir.mkdir(parents=True, exist_ok=True)

    # 1. Extract base models (pt1_wide, pt1_narrow)
    lines = ["#!/bin/bash"]
    lines.append("cd ")

    for model_type in ['pt1_wide', 'pt1_narrow']:
        config_path = f'configs/revision/exp3/representation_extraction/{model_type}/distance_firstcity_last_and_trans_l5.yaml'
        lines.append(f'uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite')

    script_path = script_dir / 'extract_base_models.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # 2. Extract wide ftwb models
    lines = ["#!/bin/bash"]
    lines.append("cd ")

    for task_num in range(1, 8):
        task_name = TASKS[task_num]
        config_path = f'configs/revision/exp3/representation_extraction/pt1_wide_ftwb{task_num}/{task_name}_firstcity_last_and_trans_l5.yaml'
        lines.append(f'uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite')

    script_path = script_dir / 'extract_wide_ftwb.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # 3. Extract narrow ftwb models
    lines = ["#!/bin/bash"]
    lines.append("cd ")

    for task_num in range(1, 8):
        task_name = TASKS[task_num]
        config_path = f'configs/revision/exp3/representation_extraction/pt1_narrow_ftwb{task_num}/{task_name}_firstcity_last_and_trans_l5.yaml'
        lines.append(f'uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite')

    script_path = script_dir / 'extract_narrow_ftwb.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # 4. Extract all
    lines = ["#!/bin/bash"]
    lines.append("cd ")
    lines.append("bash scripts/revision/exp3/representation_extraction/extract_base_models.sh")
    lines.append("bash scripts/revision/exp3/representation_extraction/extract_wide_ftwb.sh")
    lines.append("bash scripts/revision/exp3/representation_extraction/extract_narrow_ftwb.sh")

    script_path = script_dir / 'extract_all.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

def create_pca_timeline_scripts():
    """Create PCA timeline scripts."""
    base_dir = Path('')
    script_dir = base_dir / 'scripts/revision/exp3/pca_timeline'
    script_dir.mkdir(parents=True, exist_ok=True)

    # 1. PCA for base models (mixed and raw)
    for pca_type in ['mixed', 'raw']:
        lines = ["#!/bin/bash"]

        for model_type in ['pt1_wide', 'pt1_narrow']:
            config_path = f'configs/revision/exp3/pca_timeline/{model_type}_{pca_type}/{model_type}_distance_firstcity_last_and_trans_l5.yaml'
            lines.append(f'uv run python src/analysis/visualize_pca_3d_timeline.py {config_path} --overwrite')

        script_path = script_dir / f'pca_base_models_{pca_type}.sh'
        script_path.write_text('\n'.join(lines) + '\n')
        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    # 2. PCA for wide ftwb (mixed, raw, na)
    for pca_type in ['mixed', 'raw', 'na']:
        lines = ["#!/bin/bash"]

        for task_num in range(1, 8):
            task_name = TASKS[task_num]
            config_path = f'configs/revision/exp3/pca_timeline/pt1_wide_ftwb{task_num}_{pca_type}/pt1_wide_ftwb{task_num}_{task_name}_firstcity_last_and_trans_l5.yaml'
            lines.append(f'uv run python src/analysis/visualize_pca_3d_timeline.py {config_path} --overwrite')

        script_path = script_dir / f'pca_wide_ftwb_{pca_type}.sh'
        script_path.write_text('\n'.join(lines) + '\n')
        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    # 3. PCA for narrow ftwb (mixed, raw, na)
    for pca_type in ['mixed', 'raw', 'na']:
        lines = ["#!/bin/bash"]

        for task_num in range(1, 8):
            task_name = TASKS[task_num]
            config_path = f'configs/revision/exp3/pca_timeline/pt1_narrow_ftwb{task_num}_{pca_type}/pt1_narrow_ftwb{task_num}_{task_name}_firstcity_last_and_trans_l5.yaml'
            lines.append(f'uv run python src/analysis/visualize_pca_3d_timeline.py {config_path} --overwrite')

        script_path = script_dir / f'pca_narrow_ftwb_{pca_type}.sh'
        script_path.write_text('\n'.join(lines) + '\n')
        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    # 4. Create master scripts
    # All base models
    lines = ["#!/bin/bash"]
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_base_models_mixed.sh")
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_base_models_raw.sh")

    script_path = script_dir / 'pca_base_models_all.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # All wide ftwb
    lines = ["#!/bin/bash"]
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_mixed.sh")
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_raw.sh")
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_na.sh")

    script_path = script_dir / 'pca_wide_ftwb_all.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # All narrow ftwb
    lines = ["#!/bin/bash"]
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_narrow_ftwb_mixed.sh")
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_narrow_ftwb_raw.sh")
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_narrow_ftwb_na.sh")

    script_path = script_dir / 'pca_narrow_ftwb_all.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # All PCA
    lines = ["#!/bin/bash"]
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_base_models_all.sh")
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_wide_ftwb_all.sh")
    lines.append("bash scripts/revision/exp3/pca_timeline/pca_narrow_ftwb_all.sh")

    script_path = script_dir / 'pca_all.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

def main():
    print("Creating representation extraction scripts...")
    create_repr_extraction_scripts()

    print("\nCreating PCA timeline scripts...")
    create_pca_timeline_scripts()

    print("\nSummary:")
    print("  Representation extraction scripts: 4")
    print("  PCA timeline scripts: 11 (2 base + 3 wide + 3 narrow + 3 master)")

if __name__ == '__main__':
    main()
