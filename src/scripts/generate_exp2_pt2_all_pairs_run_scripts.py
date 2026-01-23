#!/usr/bin/env python3
"""
Generate execution scripts for ALL PT2 CKA pairs, divided into 4 chunks.
"""

from pathlib import Path
import yaml


def main():
    base_path = Path(__file__).resolve().parents[2]
    config_dir = base_path / 'configs' / 'revision' / 'exp2' / 'cka_analysis_all'
    script_dir = base_path / 'scripts' / 'revision' / 'exp2' / 'cka_analysis_all'
    script_dir.mkdir(parents=True, exist_ok=True)

    # Get all config files
    config_files = sorted(config_dir.glob('cka_*.yaml'))
    print(f"Found {len(config_files)} config files")

    # Divide into 4 chunks
    n_chunks = 4
    chunk_size = len(config_files) // n_chunks

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        if chunk_idx == n_chunks - 1:
            # Last chunk gets remainder
            end_idx = len(config_files)
        else:
            end_idx = start_idx + chunk_size

        chunk_configs = config_files[start_idx:end_idx]

        # Create script
        script_path = script_dir / f'run_pt2_all_pairs_chunk{chunk_idx + 1}.sh'
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            for config_path in chunk_configs:
                rel_path = config_path.relative_to(base_path)
                f.write(f'uv run python src/scripts/analyze_cka_pair.py {rel_path} --overwrite\n')

        script_path.chmod(0o755)
        print(f"Created chunk {chunk_idx + 1}: {len(chunk_configs)} configs in {script_path.name}")

    # Create master script
    master_script = script_dir / 'run_pt2_all_pairs_all.sh'
    with open(master_script, 'w') as f:
        f.write('#!/bin/bash\n')
        for chunk_idx in range(n_chunks):
            f.write(f'bash scripts/revision/exp2/cka_analysis_all/run_pt2_all_pairs_chunk{chunk_idx + 1}.sh\n')

    master_script.chmod(0o755)
    print(f"Created master script: {master_script.name}")

    # Create layer-specific scripts
    layers = [3, 4, 5, 6]
    for layer in layers:
        layer_configs = [c for c in config_files if f'_l{layer}.yaml' in c.name]

        script_path = script_dir / f'run_pt2_all_pairs_l{layer}.sh'
        with open(script_path, 'w') as f:
            f.write('#!/bin/bash\n')
            for config_path in layer_configs:
                rel_path = config_path.relative_to(base_path)
                f.write(f'uv run python src/scripts/analyze_cka_pair.py {rel_path} --overwrite\n')

        script_path.chmod(0o755)
        print(f"Created layer {layer} script: {len(layer_configs)} configs")


if __name__ == '__main__':
    main()
