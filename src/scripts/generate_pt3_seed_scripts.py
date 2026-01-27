#!/usr/bin/env python3
"""
Generate training scripts for pt3-{1-8} with seed1 and seed2.
"""

from pathlib import Path

def generate_pt3_scripts():
    """Generate bash scripts for pt3-{1-8} seed1 and seed2 training."""
    base_dir = Path("scripts/revision/exp5/pt3_seed")

    for pt3_num in range(1, 9):
        pt3_dir = base_dir / f"pt3-{pt3_num}"
        pt3_dir.mkdir(parents=True, exist_ok=True)

        for seed in [1, 2]:
            script_content = f"""#!/bin/bash
cd 
uv run python src/training/train.py configs/revision/exp5/pt3_seed/pt3-{pt3_num}/pt3-{pt3_num}_seed{seed}.yaml --overwrite
"""
            script_path = pt3_dir / f"pt3-{pt3_num}_seed{seed}.sh"
            with open(script_path, 'w') as f:
                f.write(script_content)

            # Make executable
            script_path.chmod(0o755)

            print(f"Created: {script_path}")

    print(f"\nGenerated 16 scripts (8 pt3 variants × 2 seeds)")

if __name__ == "__main__":
    generate_pt3_scripts()
