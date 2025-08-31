#!/usr/bin/env python3
"""
Create figures for the emergent geographic representations blog post.
This script generates visualizations from the flagship experiment results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
import shutil

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Paths
experiment_dir = Path('/n/home12/cfpark00/WM_1/outputs/experiments/dist_100k_1M_20epochs')
analysis_dir = experiment_dir / 'analysis' / 'layers3_4_probe5000_train3000'
output_dir = Path('/n/home12/cfpark00/WM_1/reports/emergent-geographic-representations')
output_dir.mkdir(exist_ok=True, parents=True)

def copy_existing_figures():
    """Copy existing analysis figures to blog post directory."""
    
    # Copy the training dynamics plot
    dynamics_plot = analysis_dir / 'dynamics_plot.png'
    if dynamics_plot.exists():
        shutil.copy2(dynamics_plot, output_dir / 'training_dynamics.png')
        print("Copied training_dynamics.png")
    else:
        print(f"Warning: {dynamics_plot} not found")
    
    # Copy the world map evolution (use last frame as static image)
    world_map_gif = analysis_dir / 'world_map_evolution.gif'
    if world_map_gif.exists():
        # For now, just copy the GIF - in a real scenario we'd extract the last frame
        shutil.copy2(world_map_gif, output_dir / 'world_map_evolution.gif')
        print("Copied world_map_evolution.gif")
        
        # Create a static version note
        print("Note: For blog post, you may want to extract the final frame of the GIF")
    else:
        print(f"Warning: {world_map_gif} not found")

def create_representation_evolution_plot():
    """Create a plot showing the evolution of representation quality."""
    # Read the representation dynamics data
    csv_file = analysis_dir / 'representation_dynamics.csv'
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found")
        return
    
    df = pd.read_csv(csv_file)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Top plot: R² evolution
    steps = df['step']
    ax1.plot(steps, df['lon_test_r2'], 'b-', linewidth=2, label='Longitude R²', marker='o', markersize=4)
    ax1.plot(steps, df['lat_test_r2'], 'r-', linewidth=2, label='Latitude R²', marker='s', markersize=4)
    
    ax1.set_ylabel('Test R² Score', fontsize=12)
    ax1.set_title('Evolution of Geographic Representation Quality', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.2, 1.0)
    
    # Add milestone annotations
    ax1.axvline(x=15632, color='gray', linestyle='--', alpha=0.5)
    ax1.annotate('Rapid Emergence\n(R²>0.9)', xy=(15632, 0.9), xytext=(20000, 0.7),
                arrowprops=dict(arrowstyle='->', color='gray', alpha=0.7),
                fontsize=10, ha='center')
    
    # Bottom plot: Distance error
    ax2.plot(steps, df['mean_dist_error_km'], 'g-', linewidth=2, label='Mean Distance Error', marker='^', markersize=4)
    ax2.plot(steps, df['median_dist_error_km'], 'orange', linewidth=2, label='Median Distance Error', marker='d', markersize=4)
    
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Location Error (km)', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    
    # Add reference lines
    ax2.axhline(y=1000, color='red', linestyle=':', alpha=0.5, label='1000 km')
    ax2.axhline(y=10000, color='red', linestyle=':', alpha=0.3, label='10000 km')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'representation_evolution.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Created representation_evolution.png")

def create_probe_accuracy_plot():
    """Create a detailed plot of probe accuracy across training."""
    csv_file = analysis_dir / 'representation_dynamics.csv'
    if not csv_file.exists():
        print(f"Warning: {csv_file} not found")
        return
    
    df = pd.read_csv(csv_file)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = df['step']
    
    # Plot both train and test R² for comparison
    ax.plot(steps, df['lon_train_r2'], 'b--', alpha=0.7, label='Longitude R² (Train)', linewidth=1.5)
    ax.plot(steps, df['lon_test_r2'], 'b-', linewidth=2, label='Longitude R² (Test)', marker='o', markersize=3)
    ax.plot(steps, df['lat_train_r2'], 'r--', alpha=0.7, label='Latitude R² (Train)', linewidth=1.5)
    ax.plot(steps, df['lat_test_r2'], 'r-', linewidth=2, label='Latitude R² (Test)', marker='s', markersize=3)
    
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Geographic Coordinate Prediction Accuracy', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.2, 1.0)
    
    # Add final values as text
    final_lon_r2 = df['lon_test_r2'].iloc[-1]
    final_lat_r2 = df['lat_test_r2'].iloc[-1]
    
    ax.text(0.02, 0.98, f'Final Longitude R²: {final_lon_r2:.3f}\nFinal Latitude R²: {final_lat_r2:.3f}', 
            transform=ax.transAxes, verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='gray', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'probe_accuracy.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Created probe_accuracy.png")

def create_final_world_map_placeholder():
    """Create a placeholder for the world map visualization."""
    # This would normally extract the final frame from the GIF or recreate the world map
    # For now, create a placeholder that explains what should be here
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.text(0.5, 0.5, 'Final World Map Visualization\n\n' +
            'This should show:\n' +
            '• True city locations (colored by region)\n' +
            '• Model predictions connected by lines\n' +
            '• Geographic structure emergence\n' +
            '• Continental boundaries and clustering\n\n' +
            'Extract from world_map_evolution.gif or regenerate from analysis script',
            ha='center', va='center', fontsize=14, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor='lightblue', alpha=0.7))
    
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title('Model\'s Internal World Map (Final Checkpoint)', fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'final_world_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print("Created final_world_map.png (placeholder)")

def create_training_summary():
    """Create a summary of training results."""
    # Read final results
    eval_results_file = experiment_dir / 'eval_results.json'
    
    if eval_results_file.exists():
        import json
        with open(eval_results_file, 'r') as f:
            eval_results = json.load(f)
        
        print("\nTraining Summary:")
        print(f"Final eval loss: {eval_results['eval_loss']:.4f}")
        print(f"Mean distance error: {eval_results['eval_metric_mean']:.2f} km")
        print(f"Median distance error: {eval_results['eval_metric_median']:.2f} km")
        print(f"Valid predictions: {eval_results['eval_valid_count']}/{eval_results.get('eval_total', 'unknown')}")
    
    # Read representation results
    csv_file = analysis_dir / 'representation_dynamics.csv'
    if csv_file.exists():
        df = pd.read_csv(csv_file)
        final_row = df.iloc[-1]
        
        print(f"\nFinal Representation Quality:")
        print(f"Longitude R²: {final_row['lon_test_r2']:.4f}")
        print(f"Latitude R²: {final_row['lat_test_r2']:.4f}")
        print(f"Mean location error: {final_row['mean_dist_error_km']:.1f} km")
        print(f"Median location error: {final_row['median_dist_error_km']:.1f} km")

if __name__ == "__main__":
    print("Creating figures for emergent geographic representations blog post...")
    
    # Copy existing analysis figures
    copy_existing_figures()
    
    # Create new figures
    create_representation_evolution_plot()
    create_probe_accuracy_plot()
    create_final_world_map_placeholder()
    
    # Print summary
    create_training_summary()
    
    print(f"\nAll figures created successfully!")
    print(f"Location: {output_dir}")
    print(f"\nFigures for blog post:")
    for fig_file in output_dir.glob('*.png'):
        print(f"  - {fig_file.name}")
    for fig_file in output_dir.glob('*.gif'):
        print(f"  - {fig_file.name}")