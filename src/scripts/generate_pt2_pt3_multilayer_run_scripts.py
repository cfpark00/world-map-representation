#!/usr/bin/env python3
"""Generate bash scripts for PT2/PT3 multi-layer representation extraction."""

from pathlib import Path

PT2_TASKS = {
    1: 'distance', 2: 'angle', 3: 'inside', 4: 'crossing',
    5: 'trianglearea', 6: 'compass', 7: 'perimeter',
}

PT3_TASKS = {
    1: 'distance', 2: 'compass', 3: 'crossing', 4: 'angle',
    5: 'perimeter', 6: 'trianglearea', 7: 'inside',
}

SEEDS = [1, 2]
LAYERS = [3, 4, 5, 6]

def main():
    base_dir = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')

    # PT2 Scripts
    pt2_script_dir = base_dir / 'scripts/revision/exp2/pt2_seed/extract_representations_multilayer'
    pt2_script_dir.mkdir(parents=True, exist_ok=True)

    # Create individual layer scripts for PT2
    for layer in LAYERS:
        lines = ["#!/bin/bash"]
        lines.append("cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1")

        for variant_num in PT2_TASKS.keys():
            task_name = PT2_TASKS[variant_num]
            for seed in SEEDS:
                # Skip layer 5 for seed1 (already exists)
                if seed == 1 and layer == 5:
                    continue
                config_path = f'configs/revision/exp2/pt2_seed/extract_representations_multilayer/pt2-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{layer}.yaml'
                lines.append(f'uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite')

        script_path = pt2_script_dir / f'extract_pt2_layer{layer}.sh'
        script_path.write_text('\n'.join(lines) + '\n')
        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    # PT2 master script
    lines = ["#!/bin/bash"]
    lines.append("cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1")
    for layer in LAYERS:
        lines.append(f'bash scripts/revision/exp2/pt2_seed/extract_representations_multilayer/extract_pt2_layer{layer}.sh')

    script_path = pt2_script_dir / 'extract_pt2_all_layers.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # PT3 Scripts
    pt3_script_dir = base_dir / 'scripts/revision/exp2/pt3_seed/extract_representations_multilayer'
    pt3_script_dir.mkdir(parents=True, exist_ok=True)

    # Create individual layer scripts for PT3
    for layer in LAYERS:
        lines = ["#!/bin/bash"]
        lines.append("cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1")

        for variant_num in PT3_TASKS.keys():
            task_name = PT3_TASKS[variant_num]
            for seed in SEEDS:
                config_path = f'configs/revision/exp2/pt3_seed/extract_representations_multilayer/pt3-{variant_num}_seed{seed}_{task_name}_firstcity_last_and_trans_l{layer}.yaml'
                lines.append(f'uv run python src/analysis/analyze_representations_higher.py {config_path} --overwrite')

        script_path = pt3_script_dir / f'extract_pt3_layer{layer}.sh'
        script_path.write_text('\n'.join(lines) + '\n')
        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    # PT3 master script
    lines = ["#!/bin/bash"]
    lines.append("cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1")
    for layer in LAYERS:
        lines.append(f'bash scripts/revision/exp2/pt3_seed/extract_representations_multilayer/extract_pt3_layer{layer}.sh')

    script_path = pt3_script_dir / 'extract_pt3_all_layers.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    # Combined master script
    combined_script_dir = base_dir / 'scripts/revision/exp2'
    lines = ["#!/bin/bash"]
    lines.append("cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1")
    lines.append("bash scripts/revision/exp2/pt2_seed/extract_representations_multilayer/extract_pt2_all_layers.sh")
    lines.append("bash scripts/revision/exp2/pt3_seed/extract_representations_multilayer/extract_pt3_all_layers.sh")

    script_path = combined_script_dir / 'extract_pt2_pt3_all_multilayer.sh'
    script_path.write_text('\n'.join(lines) + '\n')
    script_path.chmod(0o755)
    print(f"Created: {script_path}")

    print("\nSummary:")
    print("  PT2 scripts: 4 layer scripts + 1 master = 5 scripts")
    print("  PT3 scripts: 4 layer scripts + 1 master = 5 scripts")
    print("  Combined master: 1 script")
    print("  Total: 11 scripts")

if __name__ == '__main__':
    main()
