"""
Generate bash scripts for all representation extractions.
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path('')
sys.path.insert(0, str(project_root))

import argparse


# Task mappings
TASK_NAMES = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing',
}


def generate_extraction_script(config_path, script_path, overwrite=True):
    """Generate a single extraction bash script."""
    overwrite_flag = " --overwrite" if overwrite else ""
    script_content = f"""#!/bin/bash
cd 
uv run python src/scripts/extract_and_save_representations.py {config_path}{overwrite_flag}
"""

    with open(script_path, 'w') as f:
        f.write(script_content)

    # Make executable
    script_path.chmod(0o755)


def main(layers=[3, 4, 5, 6], include_original=True, include_seed1=True, base_dir=None):
    """
    Generate all representation extraction bash scripts.
    """
    if base_dir is None:
        base_dir = Path('')
    else:
        base_dir = Path(base_dir)

    config_base = base_dir / 'configs' / 'analysis_v2' / 'representation_extraction'
    script_base = base_dir / 'scripts' / 'analysis_v2' / 'repr_extraction'
    script_base.mkdir(parents=True, exist_ok=True)

    script_count = 0

    # Generate scripts for original PT1 experiments
    if include_original:
        print("Generating scripts for original PT1 experiments...")
        for task_id, task_name in TASK_NAMES.items():
            exp_name = f'pt1-{task_id}'

            for layer in layers:
                config_path = config_base / 'original' / exp_name / f'{task_name}_last_and_trans_l{layer}.yaml'

                if not config_path.exists():
                    print(f"  Warning: {config_path} does not exist, skipping")
                    continue

                script_name = f'extract_{exp_name}_{task_name}_l{layer}.sh'
                script_path = script_base / 'original' / script_name
                script_path.parent.mkdir(parents=True, exist_ok=True)

                # Make path relative to project root
                rel_config_path = config_path.relative_to(base_dir)

                generate_extraction_script(rel_config_path, script_path)
                script_count += 1

    # Generate scripts for seed1 PT1 experiments
    if include_seed1:
        print("Generating scripts for seed1 PT1 experiments...")
        for task_id, task_name in TASK_NAMES.items():
            exp_name = f'pt1-{task_id}_seed1'

            for layer in layers:
                config_path = config_base / 'seed1' / exp_name / f'{task_name}_last_and_trans_l{layer}.yaml'

                if not config_path.exists():
                    print(f"  Warning: {config_path} does not exist, skipping")
                    continue

                script_name = f'extract_{exp_name}_{task_name}_l{layer}.sh'
                script_path = script_base / 'seed1' / script_name
                script_path.parent.mkdir(parents=True, exist_ok=True)

                # Make path relative to project root
                rel_config_path = config_path.relative_to(base_dir)

                generate_extraction_script(rel_config_path, script_path)
                script_count += 1

    # Generate master script to run all extractions
    master_script_path = script_base / 'run_all_extractions.sh'
    with open(master_script_path, 'w') as f:
        f.write('#!/bin/bash\n')
        f.write('# Run all representation extractions\n')
        f.write('cd \n\n')

        if include_original:
            f.write('echo "Extracting representations for original PT1 experiments..."\n')
            for task_id, task_name in TASK_NAMES.items():
                exp_name = f'pt1-{task_id}'
                for layer in layers:
                    script_name = f'extract_{exp_name}_{task_name}_l{layer}.sh'
                    f.write(f'bash scripts/analysis_v2/repr_extraction/original/{script_name}\n')
            f.write('\n')

        if include_seed1:
            f.write('echo "Extracting representations for seed1 PT1 experiments..."\n')
            for task_id, task_name in TASK_NAMES.items():
                exp_name = f'pt1-{task_id}_seed1'
                for layer in layers:
                    script_name = f'extract_{exp_name}_{task_name}_l{layer}.sh'
                    f.write(f'bash scripts/analysis_v2/repr_extraction/seed1/{script_name}\n')

    master_script_path.chmod(0o755)

    print(f"\nGenerated {script_count} extraction scripts in {script_base}")
    print(f"Master script: {master_script_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate representation extraction bash scripts')
    parser.add_argument('--layers', type=str, default='3,4,5,6',
                       help='Comma-separated layer numbers (default: 3,4,5,6)')
    parser.add_argument('--original-only', action='store_true',
                       help='Only generate scripts for original experiments')
    parser.add_argument('--seed1-only', action='store_true',
                       help='Only generate scripts for seed1 experiments')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='Base WM_1 directory (default: auto-detect)')

    args = parser.parse_args()

    layers = [int(x) for x in args.layers.split(',')]

    include_original = not args.seed1_only
    include_seed1 = not args.original_only

    main(
        layers=layers,
        include_original=include_original,
        include_seed1=include_seed1,
        base_dir=args.base_dir
    )
