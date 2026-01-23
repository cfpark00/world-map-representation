#!/usr/bin/env python3
"""Generate bash scripts to run all PCA timeline visualizations."""

from pathlib import Path

# Configurations
configs = [
    ('seed1', 'seed1'),
    ('seed1_raw', 'seed1_raw'),
    ('seed2', 'seed2'),
    ('seed2_raw', 'seed2_raw'),
    ('seed3', 'seed3'),
    ('seed3_raw', 'seed3_raw'),
]

script_dir = Path("scripts/revision/exp4/pca_timeline")

for config_name, script_suffix in configs:
    config_dir = Path(f"configs/revision/exp4/pca_timeline/{config_name}")

    if not config_dir.exists():
        print(f"Skipping {config_name} (directory doesn't exist)")
        continue

    # Find all yaml files
    yaml_files = sorted(config_dir.glob("*.yaml"))

    if not yaml_files:
        print(f"Skipping {config_name} (no configs found)")
        continue

    # Create bash script
    script_file = script_dir / f"pca_timeline_{script_suffix}_all.sh"

    with open(script_file, 'w') as f:
        f.write("#!/bin/bash\n")
        for yaml_file in yaml_files:
            # Use path relative to project root
            rel_path = f"configs/revision/exp4/pca_timeline/{config_name}/{yaml_file.name}"
            f.write(f"uv run python src/analysis/visualize_pca_3d_timeline.py {rel_path} --overwrite\n")

    # Make executable
    script_file.chmod(0o755)

    print(f"Created: {script_file.name} ({len(yaml_files)} configs)")

print("\nDone!")
