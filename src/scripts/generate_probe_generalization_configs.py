#!/usr/bin/env python3
"""Generate probe generalization configs and scripts for all ftwb2 models."""

from pathlib import Path

# Base paths
config_dir = Path("configs/revision/exp1/probe_generalization")
script_dir = Path("scripts/revision/exp1/probe_generalization")
config_dir.mkdir(parents=True, exist_ok=True)
script_dir.mkdir(parents=True, exist_ok=True)

# Template
config_template = """# Probe generalization evaluation for pt1_seed1_ftwb2-{ftwb_id}
# Trains probe on non-Atlantis cities, evaluates on Atlantis cities

output_dir: data/experiments/revision/exp1/pt1_seed1_ftwb2-{ftwb_id}/probe_generalization/atlantis

# Path to representations directory
repr_dir: data/experiments/revision/exp1/pt1_seed1_ftwb2-{ftwb_id}/analysis_higher/angle_firstcity_last_and_trans_l5/representations

# Which checkpoint to use ("final", "last", or specific step number)
checkpoint: final

# Pattern for training cities (non-Atlantis)
probe_train: "region:^(?!Atlantis).*"

# Pattern for test cities (Atlantis only)
probe_test: "region:^Atlantis$"

# Number of cities to train on (rest used for baseline)
n_train: 4000
n_baseline: 100

# Probe method configuration
method:
  name: linear

seed: 42
"""

script_template = """#!/bin/bash
uv run python src/scripts/evaluate_probe_generalization.py configs/revision/exp1/probe_generalization/pt1_seed1_ftwb2-{ftwb_id}.yaml --overwrite
"""

# Generate for all 21 ftwb2 models
for ftwb_id in range(1, 22):
    # Config
    config_path = config_dir / f"pt1_seed1_ftwb2-{ftwb_id}.yaml"
    config_path.write_text(config_template.format(ftwb_id=ftwb_id))
    print(f"Created {config_path}")
    
    # Script
    script_path = script_dir / f"eval_pt1_seed1_ftwb2-{ftwb_id}.sh"
    script_path.write_text(script_template.format(ftwb_id=ftwb_id))
    script_path.chmod(0o755)
    print(f"Created {script_path}")

# Also create a master script to run all
master_script = """#!/bin/bash
# Run all probe generalization evaluations for ftwb2 models

for i in $(seq 1 21); do
    echo "Running ftwb2-$i..."
    bash scripts/revision/exp1/probe_generalization/eval_pt1_seed1_ftwb2-$i.sh
done

echo "All done!"
"""

master_path = script_dir / "run_all_ftwb2.sh"
master_path.write_text(master_script)
master_path.chmod(0o755)
print(f"\nCreated master script: {master_path}")

print(f"\nGenerated 21 configs and 21 scripts + 1 master script")
