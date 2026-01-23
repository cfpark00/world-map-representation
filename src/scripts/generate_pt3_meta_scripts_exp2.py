#!/usr/bin/env python3
"""
Generate meta scripts for pt3-{1-8} in exp2.
"""

from pathlib import Path

# PT3 task names
pt3_tasks = {
    1: "distance+trianglearea+angle",
    2: "compass+inside+perimeter",
    3: "crossing+distance+trianglearea",
    4: "angle+compass+inside",
    5: "perimeter+crossing+distance",
    6: "trianglearea+angle+compass",
    7: "inside+perimeter+crossing",
    8: "distance+trianglearea+compass"
}

def generate_meta_scripts():
    """Generate run_pt3-X_all.sh meta scripts for exp2."""
    base_dir = Path("scripts/revision/exp2")

    for pt3_num in range(1, 9):
        script_content = f"""#!/bin/bash
# Meta script for pt3-{pt3_num} ({pt3_tasks[pt3_num]})
# Runs training for both seeds

set -e

echo "=========================================="
echo "PT3-{pt3_num}: {pt3_tasks[pt3_num]}"
echo "=========================================="
echo ""

# Seed 1
echo "=== Training pt3-{pt3_num} seed1 ==="
bash scripts/revision/exp2/pt3_seed/pt3-{pt3_num}/pt3-{pt3_num}_seed1.sh

echo ""
echo "=== Training pt3-{pt3_num} seed2 ==="
bash scripts/revision/exp2/pt3_seed/pt3-{pt3_num}/pt3-{pt3_num}_seed2.sh

echo ""
echo "=========================================="
echo "PT3-{pt3_num} ({pt3_tasks[pt3_num]}) COMPLETE!"
echo "=========================================="
"""
        script_path = base_dir / f"run_pt3-{pt3_num}_all.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)

        script_path.chmod(0o755)
        print(f"Created: {script_path}")

    print(f"\nGenerated 8 meta scripts for pt3-{'{1-8}'} in exp2")

if __name__ == "__main__":
    generate_meta_scripts()
