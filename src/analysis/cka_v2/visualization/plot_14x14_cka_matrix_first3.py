"""
Plot 14x14 CKA matrix for exp4 first-3-PCs analysis (original + seed1 only).
Works with partial data and marks missing entries clearly.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from matplotlib.colors import FuncNorm

def three_slope_mapping(x, breakpoint1=0.4, breakpoint2=0.6):
    """
    Piecewise linear mapping with three segments:
    - 0.0 to breakpoint1: compressed (maps to 0.0-0.2)
    - breakpoint1 to breakpoint2: medium (maps to 0.2-0.5)
    - breakpoint2 to 1.0: expanded (maps to 0.5-1.0)
    """
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # Segment 1: 0.0 to 0.4 -> 0.0 to 0.2
    mask1 = x <= breakpoint1
    result[mask1] = (x[mask1] / breakpoint1) * 0.2

    # Segment 2: 0.4 to 0.6 -> 0.2 to 0.5
    mask2 = (x > breakpoint1) & (x <= breakpoint2)
    result[mask2] = 0.2 + ((x[mask2] - breakpoint1) / (breakpoint2 - breakpoint1)) * 0.3

    # Segment 3: 0.6 to 1.0 -> 0.5 to 1.0
    mask3 = x > breakpoint2
    result[mask3] = 0.5 + ((x[mask3] - breakpoint2) / (1.0 - breakpoint2)) * 0.5

    return result

def three_slope_inverse(x, breakpoint1=0.4, breakpoint2=0.6):
    """Inverse mapping for three-slope normalization."""
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float)

    # Inverse segment 1: 0.0 to 0.2 -> 0.0 to 0.4
    mask1 = x <= 0.2
    result[mask1] = (x[mask1] / 0.2) * breakpoint1

    # Inverse segment 2: 0.2 to 0.5 -> 0.4 to 0.6
    mask2 = (x > 0.2) & (x <= 0.5)
    result[mask2] = breakpoint1 + ((x[mask2] - 0.2) / 0.3) * (breakpoint2 - breakpoint1)

    # Inverse segment 3: 0.5 to 1.0 -> 0.6 to 1.0
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

# Create list of 14 model names in order (original + seed1 only)
# Order: distance (orig, seed1), trianglearea (orig, seed1), ...
model_names = []
for task_id in range(1, 8):
    model_names.append(f'pt1-{task_id}')
    model_names.append(f'pt1-{task_id}_seed1')

print(f"Model order: {model_names}")

# Initialize 14x14 matrix with NaN (missing data)
n = 14
cka_matrix = np.full((n, n), np.nan)

# Base directory for CKA results
base_dir = Path('data/experiments/revision/exp4/cka_analysis_first3')

# Load CKA results
loaded_count = 0
for i in range(n):
    for j in range(i, n):  # Upper triangle + diagonal
        exp1_name = model_names[i]
        exp2_name = model_names[j]

        # Try to load results (try both orderings since CKA is symmetric)
        result_dir = base_dir / f'{exp1_name}_vs_{exp2_name}' / 'layer5'
        summary_file = result_dir / 'summary.json'

        # If not found, try the reverse ordering
        if not summary_file.exists():
            result_dir = base_dir / f'{exp2_name}_vs_{exp1_name}' / 'layer5'
            summary_file = result_dir / 'summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            # Get final CKA value
            cka_val = summary.get('final_cka', summary.get('mean_cka', np.nan))

            # Fill both (i,j) and (j,i) for symmetry
            cka_matrix[i, j] = cka_val
            cka_matrix[j, i] = cka_val
            loaded_count += 1
            print(f"Loaded: {exp1_name} vs {exp2_name} = {cka_val:.4f}")
        else:
            print(f"Missing: {exp1_name} vs {exp2_name}")

print(f"\nLoaded {loaded_count} / 105 CKA pairs")
print(f"Missing: {105 - loaded_count} pairs")

# Create labels with task names
labels = []
for i in range(1, 8):
    labels.append(f'pt1-{i}\n({TASK_NAMES[i]})')
    labels.append(f'pt1-{i}_s1\n({TASK_NAMES[i]})')

# Plot
fig, ax = plt.subplots(figsize=(14, 12))

# Create a masked array to handle NaN values
masked_matrix = np.ma.masked_invalid(cka_matrix)

# Use custom three-slope normalization
norm = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)

# Plot heatmap with magma colormap
im = ax.imshow(masked_matrix, cmap='magma', norm=norm, aspect='auto')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('CKA Similarity (First 3 PCs)', rotation=270, labelpad=20, fontsize=12)

# Set ticks and labels
ax.set_xticks(np.arange(n))
ax.set_yticks(np.arange(n))
ax.set_xticklabels(labels, fontsize=9, rotation=45, ha='right')
ax.set_yticklabels(labels, fontsize=9)

# Add grid
ax.set_xticks(np.arange(n) - 0.5, minor=True)
ax.set_yticks(np.arange(n) - 0.5, minor=True)
ax.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Add text annotations
for i in range(n):
    for j in range(n):
        if not np.isnan(cka_matrix[i, j]):
            text_color = 'white' if cka_matrix[i, j] < 0.6 else 'black'
            ax.text(j, i, f'{cka_matrix[i, j]:.2f}',
                   ha='center', va='center', color=text_color, fontsize=8)
        else:
            ax.text(j, i, '?',
                   ha='center', va='center', color='gray', fontsize=10, alpha=0.5)

# Add thin section dividers between tasks (every 2 models)
for i in range(1, 7):
    ax.axhline(y=i*2 - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(x=i*2 - 0.5, color='gray', linewidth=0.5, alpha=0.3)

# Title
title = f'14×14 CKA Matrix (Layer 5, First 3 PCs)\n'
title += f'Original + Seed1 Only | Loaded: {loaded_count}/105 pairs | Missing: {105-loaded_count} pairs (marked with ?)'
ax.set_title(title, fontsize=14, pad=20)

plt.tight_layout()

# Save
output_dir = Path('data/experiments/revision/exp4/cka_analysis_first3')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'cka_matrix_14x14_layer5_first3.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Save the matrix as CSV for reference
df = pd.DataFrame(cka_matrix, index=model_names, columns=model_names)
csv_path = output_dir / 'cka_matrix_14x14_layer5_first3.csv'
df.to_csv(csv_path)
print(f"Saved matrix to: {csv_path}")

plt.close()

# ============================================================================
# Create 7x7 averaged matrix (average across seed combinations for each task pair)
# ============================================================================
print("\n" + "="*80)
print("Creating 7x7 averaged CKA matrix...")

# Initialize 7x7 matrix
cka_7x7 = np.full((7, 7), np.nan)

# For each task pair (i, j), average over all seed combinations
for i in range(7):
    for j in range(7):
        # Get indices in 14x14 matrix for this task pair
        # Task i: positions 2*i (orig), 2*i+1 (seed1)
        # Task j: positions 2*j (orig), 2*j+1 (seed1)

        cka_values = []
        for seed_i in [0, 1]:  # orig, seed1
            for seed_j in [0, 1]:  # orig, seed1
                # For diagonal (same task), only include cross-seed comparisons
                if i == j and seed_i == seed_j:
                    continue

                idx_i = 2*i + seed_i
                idx_j = 2*j + seed_j
                val = cka_matrix[idx_i, idx_j]
                if not np.isnan(val):
                    cka_values.append(val)

        # Average available values
        if len(cka_values) > 0:
            cka_7x7[i, j] = np.mean(cka_values)
            if i == j:
                print(f"Task {i+1} vs Task {j+1} (diagonal): {len(cka_values)}/2 cross-seed values, avg={cka_7x7[i, j]:.4f}")
            else:
                print(f"Task {i+1} vs Task {j+1}: {len(cka_values)}/4 values, avg={cka_7x7[i, j]:.4f}")

# Task labels for 7x7
task_labels_7x7 = [f'pt1-{i+1}\n({TASK_NAMES[i+1]})' for i in range(7)]

# Plot 7x7 matrix
fig2, ax2 = plt.subplots(figsize=(10, 9))

masked_matrix_7x7 = np.ma.masked_invalid(cka_7x7)
norm_7x7 = FuncNorm((three_slope_mapping, three_slope_inverse), vmin=0, vmax=1)
im2 = ax2.imshow(masked_matrix_7x7, cmap='magma', norm=norm_7x7, aspect='auto')

# Colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('CKA Similarity (First 3 PCs)', rotation=270, labelpad=20, fontsize=12)

# Ticks and labels
ax2.set_xticks(np.arange(7))
ax2.set_yticks(np.arange(7))
ax2.set_xticklabels(task_labels_7x7, fontsize=10, rotation=45, ha='right')
ax2.set_yticklabels(task_labels_7x7, fontsize=10)

# Grid
ax2.set_xticks(np.arange(7) - 0.5, minor=True)
ax2.set_yticks(np.arange(7) - 0.5, minor=True)
ax2.grid(which='minor', color='gray', linestyle='-', linewidth=0.5)

# Text annotations
for i in range(7):
    for j in range(7):
        if not np.isnan(cka_7x7[i, j]):
            text_color = 'white' if cka_7x7[i, j] < 0.6 else 'black'
            ax2.text(j, i, f'{cka_7x7[i, j]:.2f}',
                   ha='center', va='center', color=text_color, fontsize=10)
        else:
            ax2.text(j, i, '?',
                   ha='center', va='center', color='gray', fontsize=12, alpha=0.5)

# Title
title2 = '7×7 CKA Matrix (Layer 5, First 3 PCs)\nAveraged across seed combinations'
ax2.set_title(title2, fontsize=14, pad=20)

plt.tight_layout()

# Save 7x7
output_path_7x7 = output_dir / 'cka_matrix_7x7_averaged_layer5_first3.png'
plt.savefig(output_path_7x7, dpi=300, bbox_inches='tight')
print(f"\nSaved 7x7 plot to: {output_path_7x7}")

# Save 7x7 as CSV
task_names_list = [f'pt1-{i+1}' for i in range(7)]
df_7x7 = pd.DataFrame(cka_7x7, index=task_names_list, columns=task_names_list)
csv_path_7x7 = output_dir / 'cka_matrix_7x7_averaged_layer5_first3.csv'
df_7x7.to_csv(csv_path_7x7)
print(f"Saved 7x7 matrix to: {csv_path_7x7}")

plt.close()

# ============================================================================
# Create bar plot: Intra-task vs Inter-task CKA comparison
# ============================================================================
print("\n" + "="*80)
print("Creating bar plot: Intra-task vs Inter-task CKA comparison...")
print("Excluding crossing task (pt1-7) from analysis")

# Extract intra-task CKAs (same task, different seeds) from 14x14 matrix
# For each task (excluding crossing), compare orig vs seed1
intra_task_ckas = []
for task_id in range(6):  # Tasks 0-5 (pt1-1 to pt1-6), exclude crossing (task 6)
    idx_orig = 2 * task_id      # Original model for this task
    idx_seed1 = 2 * task_id + 1  # Seed1 model for this task
    val = cka_matrix[idx_orig, idx_seed1]
    if not np.isnan(val):
        intra_task_ckas.append(val)
        print(f"Intra-task: {model_names[idx_orig]} vs {model_names[idx_seed1]} = {val:.4f}")

# Extract inter-task CKAs (different tasks, all seed combinations) from 14x14 matrix
# For each pair of different tasks (excluding crossing), include all 4 seed combinations
inter_task_ckas = []
for task_i in range(6):  # Exclude crossing
    for task_j in range(6):  # Exclude crossing
        if task_i != task_j:  # Different tasks
            # All 4 seed combinations
            for seed_i in [0, 1]:  # orig, seed1
                for seed_j in [0, 1]:  # orig, seed1
                    idx_i = 2 * task_i + seed_i
                    idx_j = 2 * task_j + seed_j
                    val = cka_matrix[idx_i, idx_j]
                    if not np.isnan(val):
                        inter_task_ckas.append(val)

print(f"\nIntra-task CKAs (same task, different seeds): {len(intra_task_ckas)} values")
print(f"  Mean: {np.mean(intra_task_ckas):.4f}, Std: {np.std(intra_task_ckas):.4f}")
print(f"  Range: [{np.min(intra_task_ckas):.4f}, {np.max(intra_task_ckas):.4f}]")
print(f"\nInter-task CKAs (different tasks, all seeds): {len(inter_task_ckas)} values")
print(f"  Mean: {np.mean(inter_task_ckas):.4f}, Std: {np.std(inter_task_ckas):.4f}")
print(f"  Range: [{np.min(inter_task_ckas):.4f}, {np.max(inter_task_ckas):.4f}]")

# Create bar plot
fig3, ax3 = plt.subplots(figsize=(8, 6))

categories = [f'Intra-task\n(Same Task,\nDifferent Seeds)\nn={len(intra_task_ckas)}',
              f'Inter-task\n(Different Tasks,\nAll Seeds)\nn={len(inter_task_ckas)}']
means = [np.mean(intra_task_ckas), np.mean(inter_task_ckas)]
stds = [np.std(intra_task_ckas), np.std(inter_task_ckas)]

bars = ax3.bar(categories, means, yerr=stds, capsize=10,
               color=['#e74c3c', '#3498db'], alpha=0.7,
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.3f}\n±{std:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Labels and title
ax3.set_ylabel('Mean CKA Similarity', fontsize=12)
ax3.set_title('Intra-task vs Inter-task CKA Comparison\nLayer 5, First 3 PCs (Excluding Crossing Task)',
             fontsize=14, pad=15)
ax3.set_ylim([0, 1.0])
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save bar plot
output_path_bar = output_dir / 'cka_barplot_intra_vs_inter_layer5_first3.png'
plt.savefig(output_path_bar, dpi=300, bbox_inches='tight')
print(f"\nSaved bar plot to: {output_path_bar}")

plt.close()
print("\nDone!")
