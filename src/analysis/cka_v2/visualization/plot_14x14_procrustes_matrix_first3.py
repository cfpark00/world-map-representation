"""
Plot 14x14 Procrustes distance matrix for exp4 first-3-PCs analysis (original + seed1 only).
Works with partial data and marks missing entries clearly.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json

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
model_names = []
for task_id in range(1, 8):
    model_names.append(f'pt1-{task_id}')
    model_names.append(f'pt1-{task_id}_seed1')

print(f"Model order: {model_names}")

# Initialize 14x14 matrix with NaN (missing data)
n = 14
proc_matrix = np.full((n, n), np.nan)

# Base directory for Procrustes results
base_dir = Path('data/experiments/revision/exp4/procrustes_analysis_first3')

# Load Procrustes results
loaded_count = 0
for i in range(n):
    for j in range(i, n):  # Upper triangle + diagonal
        exp1_name = model_names[i]
        exp2_name = model_names[j]

        # Try to load results (try both orderings since Procrustes is symmetric)
        result_dir = base_dir / f'{exp1_name}_vs_{exp2_name}' / 'layer5'
        summary_file = result_dir / 'summary.json'

        # If not found, try the reverse ordering
        if not summary_file.exists():
            result_dir = base_dir / f'{exp2_name}_vs_{exp1_name}' / 'layer5'
            summary_file = result_dir / 'summary.json'

        if summary_file.exists():
            with open(summary_file, 'r') as f:
                summary = json.load(f)

            # Get final Procrustes distance
            proc_val = summary.get('final_procrustes', summary.get('mean_procrustes', np.nan))

            # Fill both (i,j) and (j,i) for symmetry
            proc_matrix[i, j] = proc_val
            proc_matrix[j, i] = proc_val
            loaded_count += 1
            print(f"Loaded: {exp1_name} vs {exp2_name} = {proc_val:.4f}")
        else:
            print(f"Missing: {exp1_name} vs {exp2_name}")

print(f"\nLoaded {loaded_count} / 105 Procrustes pairs")
print(f"Missing: {105 - loaded_count} pairs")

# Create labels with task names
labels = []
for i in range(1, 8):
    labels.append(f'pt1-{i}\n({TASK_NAMES[i]})')
    labels.append(f'pt1-{i}_s1\n({TASK_NAMES[i]})')

# Plot
fig, ax = plt.subplots(figsize=(14, 12))

# Create a masked array to handle NaN values
masked_matrix = np.ma.masked_invalid(proc_matrix)

# Use viridis colormap (lower is more similar, higher is more different)
im = ax.imshow(masked_matrix, cmap='viridis', aspect='auto', vmin=0, vmax=np.nanmax(proc_matrix))

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Procrustes Distance (First 3 PCs)\nLower = More Similar', rotation=270, labelpad=25, fontsize=12)

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
        if not np.isnan(proc_matrix[i, j]):
            # Light text for dark background, dark text for light background
            text_color = 'white' if proc_matrix[i, j] > np.nanmax(proc_matrix) / 2 else 'black'
            ax.text(j, i, f'{proc_matrix[i, j]:.2f}',
                   ha='center', va='center', color=text_color, fontsize=8)
        else:
            ax.text(j, i, '?',
                   ha='center', va='center', color='gray', fontsize=10, alpha=0.5)

# Add thin section dividers between tasks (every 2 models)
for i in range(1, 7):
    ax.axhline(y=i*2 - 0.5, color='gray', linewidth=0.5, alpha=0.3)
    ax.axvline(x=i*2 - 0.5, color='gray', linewidth=0.5, alpha=0.3)

# Title
title = f'14×14 Procrustes Distance Matrix (Layer 5, First 3 PCs)\n'
title += f'Original + Seed1 Only | Loaded: {loaded_count}/105 pairs | Missing: {105-loaded_count} pairs (marked with ?)'
ax.set_title(title, fontsize=14, pad=20)

plt.tight_layout()

# Save
output_dir = Path('data/experiments/revision/exp4/procrustes_analysis_first3')
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / 'procrustes_matrix_14x14_layer5_first3.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"\nSaved plot to: {output_path}")

# Save the matrix as CSV for reference
df = pd.DataFrame(proc_matrix, index=model_names, columns=model_names)
csv_path = output_dir / 'procrustes_matrix_14x14_layer5_first3.csv'
df.to_csv(csv_path)
print(f"Saved matrix to: {csv_path}")

plt.close()

# ============================================================================
# Create 7x7 averaged matrix
# ============================================================================
print("\n" + "="*80)
print("Creating 7x7 averaged Procrustes distance matrix...")

# Initialize 7x7 matrix
proc_7x7 = np.full((7, 7), np.nan)

# For each task pair (i, j), average over all seed combinations
for i in range(7):
    for j in range(7):
        proc_values = []
        for seed_i in [0, 1]:  # orig, seed1
            for seed_j in [0, 1]:  # orig, seed1
                # For diagonal (same task), only include cross-seed comparisons
                if i == j and seed_i == seed_j:
                    continue

                idx_i = 2*i + seed_i
                idx_j = 2*j + seed_j
                val = proc_matrix[idx_i, idx_j]
                if not np.isnan(val):
                    proc_values.append(val)

        # Average available values
        if len(proc_values) > 0:
            proc_7x7[i, j] = np.mean(proc_values)
            if i == j:
                print(f"Task {i+1} vs Task {j+1} (diagonal): {len(proc_values)}/2 cross-seed values, avg={proc_7x7[i, j]:.4f}")
            else:
                print(f"Task {i+1} vs Task {j+1}: {len(proc_values)}/4 values, avg={proc_7x7[i, j]:.4f}")

# Task labels for 7x7
task_labels_7x7 = [f'pt1-{i+1}\n({TASK_NAMES[i+1]})' for i in range(7)]

# Plot 7x7 matrix
fig2, ax2 = plt.subplots(figsize=(10, 9))

masked_matrix_7x7 = np.ma.masked_invalid(proc_7x7)
im2 = ax2.imshow(masked_matrix_7x7, cmap='viridis', aspect='auto', vmin=0, vmax=np.nanmax(proc_7x7))

# Colorbar
cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
cbar2.set_label('Procrustes Distance (First 3 PCs)\nLower = More Similar', rotation=270, labelpad=25, fontsize=12)

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
        if not np.isnan(proc_7x7[i, j]):
            text_color = 'white' if proc_7x7[i, j] > np.nanmax(proc_7x7) / 2 else 'black'
            ax2.text(j, i, f'{proc_7x7[i, j]:.2f}',
                   ha='center', va='center', color=text_color, fontsize=10)
        else:
            ax2.text(j, i, '?',
                   ha='center', va='center', color='gray', fontsize=12, alpha=0.5)

# Title
title2 = '7×7 Procrustes Distance Matrix (Layer 5, First 3 PCs)\nAveraged across seed combinations'
ax2.set_title(title2, fontsize=14, pad=20)

plt.tight_layout()

# Save 7x7
output_path_7x7 = output_dir / 'procrustes_matrix_7x7_averaged_layer5_first3.png'
plt.savefig(output_path_7x7, dpi=300, bbox_inches='tight')
print(f"\nSaved 7x7 plot to: {output_path_7x7}")

# Save 7x7 as CSV
task_names_list = [f'pt1-{i+1}' for i in range(7)]
df_7x7 = pd.DataFrame(proc_7x7, index=task_names_list, columns=task_names_list)
csv_path_7x7 = output_dir / 'procrustes_matrix_7x7_averaged_layer5_first3.csv'
df_7x7.to_csv(csv_path_7x7)
print(f"Saved 7x7 matrix to: {csv_path_7x7}")

plt.close()

# ============================================================================
# Create bar plot: Intra-task vs Inter-task Procrustes distance comparison
# ============================================================================
print("\n" + "="*80)
print("Creating bar plot: Intra-task vs Inter-task Procrustes distance comparison...")
print("Excluding crossing task (pt1-7) from analysis")

# Extract intra-task distances (same task, different seeds)
intra_task_dists = []
for task_id in range(6):  # Tasks 0-5 (pt1-1 to pt1-6), exclude crossing (task 6)
    idx_orig = 2 * task_id
    idx_seed1 = 2 * task_id + 1
    val = proc_matrix[idx_orig, idx_seed1]
    if not np.isnan(val):
        intra_task_dists.append(val)
        print(f"Intra-task: {model_names[idx_orig]} vs {model_names[idx_seed1]} = {val:.4f}")

# Extract inter-task distances (different tasks, all seed combinations)
inter_task_dists = []
for task_i in range(6):  # Exclude crossing
    for task_j in range(6):  # Exclude crossing
        if task_i != task_j:  # Different tasks
            for seed_i in [0, 1]:  # orig, seed1
                for seed_j in [0, 1]:  # orig, seed1
                    idx_i = 2 * task_i + seed_i
                    idx_j = 2 * task_j + seed_j
                    val = proc_matrix[idx_i, idx_j]
                    if not np.isnan(val):
                        inter_task_dists.append(val)

print(f"\nIntra-task distances (same task, different seeds): {len(intra_task_dists)} values")
print(f"  Mean: {np.mean(intra_task_dists):.4f}, Std: {np.std(intra_task_dists):.4f}")
print(f"  Range: [{np.min(intra_task_dists):.4f}, {np.max(intra_task_dists):.4f}]")
print(f"\nInter-task distances (different tasks, all seeds): {len(inter_task_dists)} values")
print(f"  Mean: {np.mean(inter_task_dists):.4f}, Std: {np.std(inter_task_dists):.4f}")
print(f"  Range: [{np.min(inter_task_dists):.4f}, {np.max(inter_task_dists):.4f}]")

# Create bar plot
fig3, ax3 = plt.subplots(figsize=(8, 6))

categories = [f'Intra-task\n(Same Task,\nDifferent Seeds)\nn={len(intra_task_dists)}',
              f'Inter-task\n(Different Tasks,\nAll Seeds)\nn={len(inter_task_dists)}']
means = [np.mean(intra_task_dists), np.mean(inter_task_dists)]
stds = [np.std(intra_task_dists), np.std(inter_task_dists)]

bars = ax3.bar(categories, means, yerr=stds, capsize=10,
               color=['#2ecc71', '#e67e22'], alpha=0.7,
               edgecolor='black', linewidth=1.5)

# Add value labels on bars
for i, (bar, mean, std) in enumerate(zip(bars, means, stds)):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{mean:.3f}\n±{std:.3f}',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

# Labels and title
ax3.set_ylabel('Mean Procrustes Distance', fontsize=12)
ax3.set_title('Intra-task vs Inter-task Procrustes Distance\nLayer 5, First 3 PCs (Excluding Crossing Task)\nLower = More Similar',
             fontsize=14, pad=15)
ax3.grid(True, alpha=0.3, axis='y')

plt.tight_layout()

# Save bar plot
output_path_bar = output_dir / 'procrustes_barplot_intra_vs_inter_layer5_first3.png'
plt.savefig(output_path_bar, dpi=300, bbox_inches='tight')
print(f"\nSaved bar plot to: {output_path_bar}")

plt.close()
print("\nDone!")
