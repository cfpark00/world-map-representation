#!/usr/bin/env python3
"""
Generate 4 balanced bash scripts for running exp2 seed CKA analysis in parallel.
Divides the 252 total calculations into 4 roughly equal chunks.
"""

from pathlib import Path

def main():
    base_path = Path('/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1')
    script_dir = base_path / 'scripts' / 'revision' / 'exp2' / 'cka_analysis'
    config_base = base_path / 'configs' / 'revision' / 'exp2'

    layers = [3, 4, 5, 6]
    seeds_unique = [('orig', '1'), ('orig', '2'), ('1', '2')]

    # PT2 non-overlapping pairs (14 pairs)
    pt2_pairs = [
        (1, 2), (1, 3), (1, 6), (1, 7),
        (2, 3), (2, 4), (2, 7),
        (3, 4), (3, 5),
        (4, 5), (4, 6),
        (5, 6), (5, 7),
        (6, 7),
    ]

    # PT3 non-overlapping pairs (7 pairs)
    pt3_pairs = [
        (1, 2), (1, 7),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        (6, 7),
    ]

    # Generate all configs as tuples (prefix, var1, var2, layer, seed1, seed2)
    all_configs = []

    # PT2
    for var1, var2 in pt2_pairs:
        for layer in layers:
            for seed1, seed2 in seeds_unique:
                config_path = f"configs/revision/exp2/pt2_seed_cka/pt2-{var1}_vs_pt2-{var2}/layer{layer}_{seed1}_vs_{seed2}.yaml"
                label = f"pt2-{var1}_{seed1} vs pt2-{var2}_{seed2} (L{layer})"
                all_configs.append((config_path, label))

    # PT3
    for var1, var2 in pt3_pairs:
        for layer in layers:
            for seed1, seed2 in seeds_unique:
                config_path = f"configs/revision/exp2/pt3_seed_cka/pt3-{var1}_vs_pt3-{var2}/layer{layer}_{seed1}_vs_{seed2}.yaml"
                label = f"pt3-{var1}_{seed1} vs pt3-{var2}_{seed2} (L{layer})"
                all_configs.append((config_path, label))

    total = len(all_configs)
    print(f"Total configs: {total}")

    # Divide into 4 chunks
    chunk_size = total // 4
    remainder = total % 4

    chunks = []
    start = 0
    for i in range(4):
        # Distribute remainder across first few chunks
        size = chunk_size + (1 if i < remainder else 0)
        chunks.append(all_configs[start:start + size])
        start += size

    # Verify
    print(f"Chunk sizes: {[len(c) for c in chunks]}")
    print(f"Total: {sum(len(c) for c in chunks)} (should be {total})")

    # Generate scripts
    for chunk_idx, chunk_configs in enumerate(chunks, 1):
        script_path = script_dir / f'run_exp2_seed_cka_chunk{chunk_idx}.sh'

        content = f"""#!/bin/bash
# Exp2 Seed CKA Analysis - Chunk {chunk_idx} of 4
# Total: {len(chunk_configs)} CKA calculations

cd /n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1

echo "========================================="
echo "Exp2 Seed CKA - Chunk {chunk_idx}/4"
echo "Total: {len(chunk_configs)} calculations"
echo "========================================="

count=0
total={len(chunk_configs)}

"""

        for config_path, label in chunk_configs:
            content += f"""count=$((count + 1))
echo ""
echo "[$count/$total] {label}"
uv run python src/scripts/analyze_cka_pair.py {config_path} --overwrite
"""

        content += f"""
echo ""
echo "========================================="
echo "Chunk {chunk_idx}/4 complete!"
echo "Results: data/experiments/revision/exp2/cka_analysis/"
echo "========================================="
"""

        with open(script_path, 'w') as f:
            f.write(content)

        script_path.chmod(0o755)
        print(f"Created: {script_path.relative_to(base_path)} ({len(chunk_configs)} calcs)")

    # Create master script
    master_path = script_dir / 'run_exp2_seed_cka_4chunks_parallel.sh'
    with open(master_path, 'w') as f:
        f.write("""#!/bin/bash
# Run all 4 chunks in parallel (if running on SLURM or separate terminals)
# Total: 252 CKA calculations divided into 4 balanced chunks

echo "Submit these 4 jobs in parallel:"
echo ""
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk1.sh"
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk2.sh"
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk3.sh"
echo "  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk4.sh"
echo ""
echo "Or run sequentially:"
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk1.sh
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk2.sh
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk3.sh
bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk4.sh
""")
    master_path.chmod(0o755)
    print(f"\nMaster: {master_path.relative_to(base_path)}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total configs: {total}")
    print(f"Chunks: 4")
    for i, chunk in enumerate(chunks, 1):
        print(f"  Chunk {i}: {len(chunk)} calculations")
    print(f"\nUsage (parallel):")
    print(f"  Terminal 1: bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk1.sh")
    print(f"  Terminal 2: bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk2.sh")
    print(f"  Terminal 3: bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk3.sh")
    print(f"  Terminal 4: bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk4.sh")
    print(f"\nOr sequential:")
    print(f"  bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_4chunks_parallel.sh")


if __name__ == '__main__':
    main()
