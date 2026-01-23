"""
Plot 14x14 Procrustes SIMILARITY matrix (1 - distance) for exp4 first-3-PCs analysis.
Uses magma colormap like CKA for consistency (higher = more similar).
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from matplotlib.colors import FuncNorm

def three_slope_mapping(x, breakpoint1=0.4, breakpoint2=0.6):
    """Piecewise linear mapping with three segments."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask1 = x <= breakpoint1
    result[mask1] = (x[mask1] / breakpoint1) * 0.2
    mask2 = (x > breakpoint1) & (x <= breakpoint2)
    result[mask2] = 0.2 + ((x[mask2] - breakpoint1) / (breakpoint2 - breakpoint1)) * 0.3
    mask3 = x > breakpoint2
    result[mask3] = 0.5 + ((x[mask3] - breakpoint2) / (1.0 - breakpoint2)) * 0.5
    return result

def three_slope_inverse(x, breakpoint1=0.4, breakpoint2=0.6):
    """Inverse mapping for three-slope normalization."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)
    mask1 = x <= 0.2
    result[mask1] = (x[mask1] / 0.2) * breakpoint1
    mask2 = (x > 0.2) & (x <= 0.5)
    result[mask2] = breakpoint1 + ((x[mask2] - 0.2) / 0.3) * (breakpoint2 - breakpoint1)
    mask3 = x > 0.5
    result[mask3] = breakpoint2 + ((x[mask3] - 0.5) / 0.5) * (1.0 - breakpoint2)
    return result

# Task mapping
TASK_NAMES = {
    1: 'distance',
    2: 'trianglearea',
    3: 'angle',
    4: 'compass',
    5: 'inside',
    6: 'perimeter',
    7: 'crossing',
}

# Create list of 14 model names
model_names = []
for task_id in range(1, 8):
    model_names.append(f'pt1-{task_id}')
    model_names.append(f'pt1-{task_id}_seed1')

print(f"Model order: {model_names}")

# Initialize 14x14 matrix with NaN
n = 14
proc_distance_matrix = np.full((n, n), np.nan)

# Base directory for Procrustes results
base_dir = Path('data/experiments/revision/exp4/procrustes_analysis_first3')

# Load Procrustes distances
loaded_count = 0
for i in range(n):
    for j in range(i, n):
        exp1_name = model_names[i]
        exp2_name = model_names[j]

        result_dir = base_dir / f'{exp1_name}_vs_{exp2_name}' / 'layer5'
        summary_file = result_dir / 'summary.json'

        if not summary_file.exists():
            result_dir = base_dir / f'{exp2_name}_vs_{exp1_name}' / 'layer5'
            summary_file = result_dir / 'summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            proc_dist = summary.get('final_procrustes', summary.get('mean_procrustes', np.nan))

            proc_distance_matrix[i, j] = proc_dist
            proc_distance_matrix[j, i] = proc_dist
            loaded_count += 1
            print(f"Loaded: {exp1_name} vs {exp2_name} = {proc_dist:.4f}")
        else:
            print(f"Missing: {exp1_name} vs {exp2_name}")

print(f"\nLoaded {loaded_count} / 105 Procrustes pairs")

# Convert distance to similarity: 1 - distance
proc_similarity_matrix = 1.0 - proc_distance_matrix

print(f"\nConverted to similarity (1 - distance)")
print(f"Similarity range: [{np.nanmin(proc_similarity_matrix):.4f}, {np.nanmax(proc_similarity_matrix):.4f}]")

# Create labels
labels = []
for i in range(1, 8):
    labels.append(f'pt1-{i}\n({TASK_NAMES[i]})')
    labels.append(f'pt1-{i}_s1\n({TASK_NAMES[i]})')

# Plot
fig, ax = plt.subplots(figsize=(14, 12))

masked_matrix = np.ma.masked_invalid(proc_similarity_matrix)

# Use three-slope normalization like CKA
norm = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)

# Plot with magma colormap
im = ax.imshow(masked_matrix, cmap='magma', norm=norm, aspect='auto')

# Colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Procrustes Similarity (First 3 PCs)\n1 - Distance', rotation=270, labelpad=25, fontsize=12)

# Ticks and labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(labels, fontsize=9)

# Grid
ax.set_xticks(np.arange(n) - 0.5, minor=True)
ax.set_yticks(np.arange(n) - 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Text annotations
for i in range(n):
    for j in range(n):
        if not np.isnan(proc_similarity_matrix[i, j]):
            text_color = 'white' if proc_similarity_matrix[i, j] < 0.6 else 'black'
            ax.text(j, i, f'{proc_similarity_matrix[i, j]:.2f}',
                   ha='center', va='center', color=text_color, fontsize=8)
        else:
            ax.text(j, i, '?',
                   ha='center', va='center', color='gray', fontsize=10, alpha=0.5)

# Section dividers
for i in range(1, 7):
    ax.axhline(y=i*2 - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(x=i*2 - 0.5, color='gray', linewidth=0.5, alpha=0.3)

# Title
title = f'14×14 Procrustes Similarity Matrix (Layer 5, First 3 PCs)\n'
title += f'Original + Seed1 Only | Loaded: {loaded_count}/105 pairs'
ax.set_title(title, fontsize=14, pad=20)

plt.tight_layout()

# Save
output_dir = Path('data/experiments/revision/exp4/procrustes_analysis_first3')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'procrustes_similarity_matrix_14x14_layer5_first3.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Save similarity matrix as CSV
df = pd.DataFrame(proc_similarity_matrix, index=model_names, columns=model_names)
csv_path = output_dir / 'procrustes_similarity_matrix_14x14_layer5_first3.csv'
df.to_csv(csv_path)
print(f"Saved matrix to: {csv_path}")

plt.close()

# ============================================================================
# Create 7x7 averaged similarity matrix
# ============================================================================
print("\n" + "="*80)
print("Creating 7x7 averaged Procrustes similarity matrix...")

proc_sim_7x7 = np.full((7, 7), np.nan)

for i in range(7):
    for j in range(7):
        sim_values = []
        for seed_i in [0, 1]:
            for seed_j in [0, 1]:
                if i == j and seed_i == seed_j:
                    continue
                idx_i = 2*i + seed_i
                idx_j = 2*j + seed_j
                val = proc_similarity_matrix[idx_i, idx_j]
                if not np.isnan(val):
                    sim_values.append(val)

        if len(sim_values) > 0:
            proc_sim_7x7[i, j] = np.mean(sim_values)
            if i == j:
                print(f"Task {i+1} vs Task {j+1} (diagonal): {len(sim_values)}/2 cross-seed values, avg={proc_sim_7x7[i, j]:.4f}")
            else:
                print(f"Task {i+1} vs Task {j+1}: {len(sim_values)}/4 values, avg={proc_sim_7x7[i, j]:.4f}")

# Task labels
task_labels_7x7 = [f'pt1-{i+1}\n({TASK_NAMES[i+1]})' for i in range(7)]

# Plot 7x7
fig2, ax2 = plt.subplots(figsize=(10, 9))

masked_matrix_7x7 = np.ma.masked_invalid(proc_sim_7x7)
norm_7x7 = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)
im2 = ax2.imshow(masked_matrix_7x7, cmap='magma', norm=norm_7x7, aspect='auto')

# Colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Procrustes Similarity (First 3 PCs)\n1 - Distance', rotation=270, labelpad=25, fontsize=12)

# Ticks
ax2.set_xticks(np.arange(7))
ax2.set_yticks(np.arange(7))
ax2.set_xticklabels(task_labels_7x7, fontsize=10, rotation=45, ha='right')
ax2.set_yticklabels(task_labels_7x7, fontsize=10)

# Grid
ax2.set_xticks(np.arange(7) - 0.5, minor=True)
ax2.set_yticks(np.arange(7) - 0.5, minor=True)
ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Annotations
for i in range(7):
    for j in range(7):
        if not np.isnan(proc_sim_7x7[i, j]):
            text_color = 'white' if proc_sim_7x7[i, j] < 0.6 else 'black'
            ax2.text(j, i, f'{proc_sim_7x7[i, j]:.2f}',
                   ha='center', va='center', color=text_color, fontsize=10)
        else:
            ax2.text(j, i, '?',
                   ha='center', va='center', color='gray', fontsize=12, alpha=0.5)

title2 = '7×7 Procrustes Similarity Matrix (Layer 5, First 3 PCs)\nAveraged across seed combinations'
ax2.set_title(title2, fontsize=14, pad=20)

plt.tight_layout()

# Save 7x7
output_path_7x7 = output_dir / 'procrustes_similarity_matrix_7x7_averaged_layer5_first3.png'
plt.savefig(output_path_7x7, dpi=300, bbox_inches='tight')
print(f"\nSaved 7x7 plot to: {output_path_7x7}")

# Save CSV
task_names_list = [f'pt1-{i+1}' for i in range(7)]
df_7x7 = pd.DataFrame(proc_sim_7x7, index=task_names_list, columns=task_names_list)
csv_path_7x7 = output_dir / 'procrustes_similarity_matrix_7x7_averaged_layer5_first3.csv'
df_7x7.to_csv(csv_path_7x7)
print(f"Saved 7x7 matrix to: {csv_path_7x7}")

plt.close()
print("\nDone!")
