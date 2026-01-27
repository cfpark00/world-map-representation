#!/usr/bin/env python3
"""
Generate bash scripts to run CKA analysis for PT2/PT3 seed variants.
Creates layer-specific scripts for parallel execution.
"""

from pathlib import Path

def generate_master_script(prefix, layers, config_base, script_dir):
    """Generate master script that runs all layers for a prefix."""

    script_path = script_dir / f'run_{prefix}_seed_cka_all_layers.sh'

    content = f"""#!/bin/bash
# Master script to run all {prefix.upper()} seed CKA analysis across all layers
"""

    for layer in layers:
        content += f"bash scripts/revision/exp2/cka_analysis/run_{prefix}_seed_cka_l{layer}.sh\n"

    with open(script_path, 'w') as f:
        f.write(content)

    script_path.chmod(0o755)
    return script_path


def generate_layer_script(prefix, layer, pairs, seeds, config_base, script_dir):
    """Generate script for a specific prefix and layer."""

    script_path = script_dir / f'run_{prefix}_seed_cka_l{layer}.sh'

    # Count total configs (only unique seed pairs)
    n_seed_combos = len(seeds) * (len(seeds) - 1) // 2  # C(3,2) = 3
    total = len(pairs) * n_seed_combos

    content = f"""#!/bin/bash
# Run {prefix.upper()} seed CKA analysis for layer {layer}
# Non-overlapping pairs only, unique seed comparisons
# Total: {total} CKA calculations ({len(pairs)} pairs × {n_seed_combos} seed combinations)

cd 

echo "========================================="
echo "{prefix.upper()} Layer {layer} Seed CKA Analysis"
echo "Total: {total} calculations"
echo "========================================="

count=0
total={total}

"""

    # Generate run commands (only unique seed pairs)
    for var1, var2 in pairs:
        for i, seed1 in enumerate(seeds):
            for seed2 in seeds[i+1:]:  # Only unique pairs
                config_path = f"configs/revision/exp2/{prefix}_seed_cka/{prefix}-{var1}_vs_{prefix}-{var2}/layer{layer}_{seed1}_vs_{seed2}.yaml"

                content += f"""count=$((count + 1))
echo ""
echo "[$count/$total] {prefix}-{var1}_{seed1} vs {prefix}-{var2}_{seed2} (layer {layer})"
uv run python src/scripts/analyze_cka_pair.py {config_path}
"""

    content += f"""
echo ""
echo "========================================="
echo "{prefix.upper()} Layer {layer} CKA analysis complete!"
echo "Results: data/experiments/revision/exp2/cka_analysis/"
echo "========================================="
"""

    with open(script_path, 'w') as f:
        f.write(content)

    script_path.chmod(0o755)
    return script_path


def main():
    base_path = Path('')
    config_base = base_path / 'configs' / 'revision' / 'exp2'
    script_dir = base_path / 'scripts' / 'revision' / 'exp2' / 'cka_analysis'
    script_dir.mkdir(parents=True, exist_ok=True)

    layers = [3, 4, 5, 6]
    seeds = ['orig', '1', '2']

    # PT2 non-overlapping pairs (only 1-7, no seeds for pt2-8)
    pt2_pairs = [
        (1, 2), (1, 3), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 7),
        (3, 4), (3, 5),
        (4, 5), (4, 6),
        (5, 6), (5, 7),
        (6, 7),
    ]

    # PT3 non-overlapping pairs (only 1-7, no seeds for pt3-8)
    pt3_pairs = [
        (1, 2), (1, 7),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
    ]

    print("=" * 80)
    print("Generating CKA execution scripts for PT2/PT3 seed variants")
    print("=" * 80)

    # Generate PT2 scripts
    print("\nPT2 Scripts:")
    for layer in layers:
        script = generate_layer_script('pt2', layer, pt2_pairs, seeds, config_base, script_dir)
        print(f"  Created: {script.relative_to(base_path)}")

    pt2_master = generate_master_script('pt2', layers, config_base, script_dir)
    print(f"  Master:  {pt2_master.relative_to(base_path)}")

    # Generate PT3 scripts
    print("\nPT3 Scripts:")
    for layer in layers:
        script = generate_layer_script('pt3', layer, pt3_pairs, seeds, config_base, script_dir)
        print(f"  Created: {script.relative_to(base_path)}")

    pt3_master = generate_master_script('pt3', layers, config_base, script_dir)
    print(f"  Master:  {pt3_master.relative_to(base_path)}")

    # Generate combined master
    combined_master = script_dir / 'run_exp2_seed_cka_all.sh'
    with open(combined_master, 'w') as f:
        f.write("""#!/bin/bash
# Master script to run ALL exp2 seed CKA analysis (PT2 + PT3, all layers)

bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_all_layers.sh
bash scripts/revision/exp2/cka_analysis/run_pt3_seed_cka_all_layers.sh
""")
    combined_master.chmod(0o755)
    print(f"\nCombined Master: {combined_master.relative_to(base_path)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_seed_combos = len(seeds) * (len(seeds) - 1) // 2  # C(3,2) = 3 unique pairs
    pt2_total = len(pt2_pairs) * len(layers) * n_seed_combos
    pt3_total = len(pt3_pairs) * len(layers) * n_seed_combos
    print(f"PT2: {len(layers)} layer scripts + 1 master = {len(layers)+1} scripts")
    print(f"  Total PT2 CKA calculations: {pt2_total}")
    print(f"PT3: {len(layers)} layer scripts + 1 master = {len(layers)+1} scripts")
    print(f"  Total PT3 CKA calculations: {pt3_total}")
    print(f"\nGrand Total: {pt2_total + pt3_total} CKA calculations")
    print(f"  ({n_seed_combos} unique seed pairs: orig-vs-1, orig-vs-2, 1-vs-2)")
    print(f"\nAll scripts created in: {script_dir.relative_to(base_path)}/")
    print("\nUsage:")
    print(f"  Run all:       bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_all.sh")
    print(f"  Run PT2 only:  bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_all_layers.sh")
    print(f"  Run PT3 only:  bash scripts/revision/exp2/cka_analysis/run_pt3_seed_cka_all_layers.sh")
    print(f"  Run 1 layer:   bash scripts/revision/exp2/cka_analysis/run_pt2_seed_cka_l5.sh")


if __name__ == '__main__':
    main()
