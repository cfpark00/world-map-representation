#!/usr/bin/env python3
"""
Generate execution scripts for same-seed non-overlapping PT2/PT3 CKA (layers 3,4,6).
"""

from pathlib import Path

def main():
    base_path = Path(__file__).resolve().parents[2]
    config_dir = base_path / 'configs' / 'revision' / 'exp2' / 'cka_same_seed_nonoverlap_l346'
    script_dir = base_path / 'scripts' / 'revision' / 'exp2' / 'cka_same_seed_nonoverlap_l346'
    script_dir.mkdir(parents=True, exist_ok=True)

    # Get all config files
    config_files = sorted(config_dir.glob('cka_*.yaml'))
    print(f"Found {len(config_files)} config files")

    # Separate by type
    pt2_configs = [c for c in config_files if 'pt2-' in c.name]
    pt3_configs = [c for c in config_files if 'pt3-' in c.name]

    print(f"PT2 configs: {len(pt2_configs)}")
    print(f"PT3 configs: {len(pt3_configs)}")

    # Divide into 4 chunks
    n_chunks = 4
    all_configs = pt2_configs + pt3_configs
    chunk_size = len(all_configs) // n_chunks

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        if chunk_idx == n_chunks - 1:
            end_idx = len(all_configs)
        else:
            end_idx = start_idx + chunk_size

        chunk_configs = all_configs[start_idx:end_idx]

        # Create script
        script_path = script_dir / f'run_same_seed_nonoverlap_l346_chunk{chunk_idx + 1}.sh'
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            for config_path in chunk_configs:
                rel_path = config_path.relative_to(base_path)
                f.write(f'uv run python src/scripts/analyze_cka_pair.py {rel_path} --overwrite\n')

        script_path.chmod(0o755)
        print(f"Created chunk {chunk_idx + 1}: {len(chunk_configs)} configs")

    # Create master script
    master_script = script_dir / 'run_same_seed_nonoverlap_l346_all.sh'
    with open(master_script, 'w') as f:
        f.write('#!/bin/bash\n')
        for chunk_idx in range(n_chunks):
            f.write(f'bash scripts/revision/exp2/cka_same_seed_nonoverlap_l346/run_same_seed_nonoverlap_l346_chunk{chunk_idx + 1}.sh\n')

    master_script.chmod(0o755)
    print(f"Created master script")

    # Create layer-specific scripts
    for layer in [3, 4, 6]:
        layer_configs = [c for c in all_configs if f'_l{layer}.yaml' in c.name]

        script_path = script_dir / f'run_same_seed_nonoverlap_l{layer}.sh'
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            for config_path in layer_configs:
                rel_path = config_path.relative_to(base_path)
                f.write(f'uv run python src/scripts/analyze_cka_pair.py {rel_path} --overwrite\n')

        script_path.chmod(0o755)
        print(f"Created layer {layer} script: {len(layer_configs)} configs")

    print(f"\nTotal: {len(all_configs)} CKA calculations")
    print(f"Output directory: data/experiments/revision/exp2/cka_analysis_same_seed/")


if __name__ == '__main__':
    main()
