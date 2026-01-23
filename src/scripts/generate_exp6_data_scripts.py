#!/usr/bin/env python3
"""
Generate bash scripts for Exp6 data generation.

Creates scripts to:
1. Generate city dataset with scattered Atlantis
2. Generate all 7x3=21 task datasets (with_atlantis, no_atlantis, atlantis_required)
3. Combine into multitask_pt1
4. Combine into FTWB1 datasets
5. Combine into FTWB2 datasets
"""
from pathlib import Path

TASKS = ['distance', 'trianglearea', 'angle', 'compass', 'inside', 'perimeter', 'crossing']
SCRIPT_PATH = Path('scripts/revision/exp6/data_generation')
CONFIG_BASE = 'configs/revision/exp6/data_generation'


def write_script(path: Path, content: str):
    """Write a bash script and make it executable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        f.write(content)
    path.chmod(0o755)
    print(f"  Created: {path}")


def main():
    SCRIPT_PATH.mkdir(parents=True, exist_ok=True)
    print("Generating Exp6 data generation scripts...")
    print(f"Script path: {SCRIPT_PATH}")
    print()

    # 1. City dataset script
    print("Creating city dataset script...")
    write_script(
        SCRIPT_PATH / 'gen_cities.sh',
        f'''#!/bin/bash
uv run python src/data_processing/create_city_dataset.py {CONFIG_BASE}/city_dataset_scattered_atlantis.yaml --overwrite
'''
    )

    # 2. Individual task dataset scripts
    print("\nCreating task dataset scripts...")

    # with_atlantis
    for task in TASKS:
        write_script(
            SCRIPT_PATH / f'gen_{task}_with_atlantis.sh',
            f'''#!/bin/bash
uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_1M_with_atlantis.yaml --overwrite
'''
        )

    # no_atlantis
    for task in TASKS:
        write_script(
            SCRIPT_PATH / f'gen_{task}_no_atlantis.sh',
            f'''#!/bin/bash
uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_1M_no_atlantis.yaml --overwrite
'''
        )

    # atlantis_required
    for task in TASKS:
        write_script(
            SCRIPT_PATH / f'gen_{task}_atlantis_required.sh',
            f'''#!/bin/bash
uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_100k_atlantis_required.yaml --overwrite
'''
        )

    # 3. Batch scripts for all tasks
    print("\nCreating batch task scripts...")

    # All with_atlantis
    write_script(
        SCRIPT_PATH / 'gen_all_with_atlantis.sh',
        '#!/bin/bash\n' + '\n'.join([
            f'uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_1M_with_atlantis.yaml --overwrite'
            for task in TASKS
        ]) + '\n'
    )

    # All no_atlantis
    write_script(
        SCRIPT_PATH / 'gen_all_no_atlantis.sh',
        '#!/bin/bash\n' + '\n'.join([
            f'uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_1M_no_atlantis.yaml --overwrite'
            for task in TASKS
        ]) + '\n'
    )

    # All atlantis_required
    write_script(
        SCRIPT_PATH / 'gen_all_atlantis_required.sh',
        '#!/bin/bash\n' + '\n'.join([
            f'uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_100k_atlantis_required.yaml --overwrite'
            for task in TASKS
        ]) + '\n'
    )

    # 4. Combination scripts
    print("\nCreating combination scripts...")

    # multitask_pt1
    write_script(
        SCRIPT_PATH / 'combine_multitask_pt1.sh',
        f'''#!/bin/bash
uv run python src/data_processing/combine_datasets.py {CONFIG_BASE}/ftset/combine_multitask_pt1.yaml --overwrite
'''
    )

    # FTWB1 (all 7)
    write_script(
        SCRIPT_PATH / 'combine_all_ftwb1.sh',
        '#!/bin/bash\n' + '\n'.join([
            f'uv run python src/data_processing/combine_datasets.py {CONFIG_BASE}/ftset/combine_ftwb1-{i}.yaml --overwrite'
            for i in range(1, 8)
        ]) + '\n'
    )

    # FTWB2 (all 21)
    write_script(
        SCRIPT_PATH / 'combine_all_ftwb2.sh',
        '#!/bin/bash\n' + '\n'.join([
            f'uv run python src/data_processing/combine_datasets.py {CONFIG_BASE}/ftset/combine_ftwb2-{i}.yaml --overwrite'
            for i in range(1, 22)
        ]) + '\n'
    )

    # 5. Master script that runs everything in order
    print("\nCreating master script...")
    master_script = f'''#!/bin/bash
echo "=== Exp6 Data Generation ==="
echo "Step 1: Generate city dataset with scattered Atlantis..."
uv run python src/data_processing/create_city_dataset.py {CONFIG_BASE}/city_dataset_scattered_atlantis.yaml --overwrite

echo ""
echo "Step 2: Generate task datasets with Atlantis (1M each)..."
'''
    for task in TASKS:
        master_script += f'uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_1M_with_atlantis.yaml --overwrite\n'

    master_script += '''
echo ""
echo "Step 3: Generate task datasets without Atlantis (1M each)..."
'''
    for task in TASKS:
        master_script += f'uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_1M_no_atlantis.yaml --overwrite\n'

    master_script += '''
echo ""
echo "Step 4: Generate task datasets with Atlantis required (100k each)..."
'''
    for task in TASKS:
        master_script += f'uv run python src/tasks/{task}.py {CONFIG_BASE}/{task}_100k_atlantis_required.yaml --overwrite\n'

    master_script += f'''
echo ""
echo "Step 5: Combine into multitask_pt1..."
uv run python src/data_processing/combine_datasets.py {CONFIG_BASE}/ftset/combine_multitask_pt1.yaml --overwrite

echo ""
echo "Step 6: Combine into FTWB1 datasets..."
'''
    for i in range(1, 8):
        master_script += f'uv run python src/data_processing/combine_datasets.py {CONFIG_BASE}/ftset/combine_ftwb1-{i}.yaml --overwrite\n'

    master_script += '''
echo ""
echo "Step 7: Combine into FTWB2 datasets..."
'''
    for i in range(1, 22):
        master_script += f'uv run python src/data_processing/combine_datasets.py {CONFIG_BASE}/ftset/combine_ftwb2-{i}.yaml --overwrite\n'

    master_script += '''
echo ""
echo "=== Exp6 Data Generation Complete ==="
'''
    write_script(SCRIPT_PATH / 'run_all_data_generation.sh', master_script)

    # Count scripts
    all_scripts = list(SCRIPT_PATH.glob('*.sh'))
    print(f"\n{'='*60}")
    print(f"Total scripts created: {len(all_scripts)}")
    print(f"  - 1 city dataset script")
    print(f"  - 21 individual task scripts (7 tasks x 3 variants)")
    print(f"  - 3 batch task scripts")
    print(f"  - 3 combination scripts (multitask_pt1, FTWB1, FTWB2)")
    print(f"  - 1 master script (run_all_data_generation.sh)")
    print(f"{'='*60}")
    print(f"\nTo generate all data, run:")
    print(f"  bash {SCRIPT_PATH}/run_all_data_generation.sh")


if __name__ == '__main__':
    main()
