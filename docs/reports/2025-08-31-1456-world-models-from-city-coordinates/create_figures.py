#!/usr/bin/env python3
"""
Create figures for the blog post about world models from city coordinates.
This script generates visualizations to support the narrative.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add project root to path
sys.path.append('/n/home12/cfpark00/WM_1')
from src.utils import load_cities_csv, extract_coordinates

# Set style
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'

# Output directory
output_dir = Path('/n/home12/cfpark00/WM_1/reports/world-models-from-city-coordinates')
output_dir.mkdir(exist_ok=True, parents=True)

def create_city_density_map():
    """Create world map showing city density distribution."""
    # Load cities
    df = load_cities_csv(None, '/n/home12/cfpark00/WM_1/outputs/datasets/cities_100k_plus_seed42.csv')
    
    # Extract coordinates directly since the column might be named differently
    if 'coordinates' in df.columns:
        coords = df['coordinates'].str.split(',', expand=True)
    elif 'Coordinates' in df.columns:
        coords = df['Coordinates'].str.split(',', expand=True)
    else:
        # Assume lat,lon are already separate columns
        pass
    
    if 'latitude' not in df.columns:
        df['latitude'] = coords[0].astype(float)
        df['longitude'] = coords[1].astype(float)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create hexbin for density
    hb = ax.hexbin(df['longitude'], df['latitude'], 
                   gridsize=50, cmap='YlOrRd', mincnt=1, alpha=0.8)
    
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Global Distribution of Cities (Population ≥ 100k)', fontsize=14, fontweight='bold')
    
    # Add colorbar
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label('Number of Cities', fontsize=11)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Add annotations for interesting regions
    ax.annotate('Dense: Japan', xy=(139, 35), xytext=(150, 45),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=10, ha='center')
    ax.annotate('Dense: Europe', xy=(10, 50), xytext=(10, 65),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=10, ha='center')
    ax.annotate('Sparse: Sahara', xy=(10, 20), xytext=(-20, 15),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=10, ha='center')
    ax.annotate('Sparse: Pacific', xy=(-140, 0), xytext=(-140, -30),
                arrowprops=dict(arrowstyle='->', color='black', alpha=0.5),
                fontsize=10, ha='center')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'city_density_map.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created city_density_map.png")

def create_batch_size_comparison():
    """Create visualization showing the batch size problem."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before (broken)
    dataset_size = 1051
    batch_size_broken = 512
    epochs = 100
    
    updates_per_epoch_broken = dataset_size // batch_size_broken
    total_updates_broken = updates_per_epoch_broken * epochs
    
    # After (working)
    batch_size_working = 64
    updates_per_epoch_working = dataset_size // batch_size_working
    total_updates_working = updates_per_epoch_working * epochs
    
    # Plot 1: Updates per epoch
    categories = ['Broken\n(BS=512)', 'Fixed\n(BS=64)']
    updates_per_epoch = [updates_per_epoch_broken, updates_per_epoch_working]
    colors = ['#ff6b6b', '#51cf66']
    
    bars1 = ax1.bar(categories, updates_per_epoch, color=colors, alpha=0.8)
    ax1.set_ylabel('Gradient Updates per Epoch', fontsize=12)
    ax1.set_title('Updates Per Epoch', fontsize=13, fontweight='bold')
    ax1.set_ylim(0, 20)
    
    # Add value labels on bars
    for bar, val in zip(bars1, updates_per_epoch):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val}', ha='center', fontsize=14, fontweight='bold')
    
    # Plot 2: Total updates
    total_updates = [total_updates_broken, total_updates_working]
    bars2 = ax2.bar(categories, total_updates, color=colors, alpha=0.8)
    ax2.set_ylabel('Total Gradient Updates (100 epochs)', fontsize=12)
    ax2.set_title('Total Training Updates', fontsize=13, fontweight='bold')
    ax2.set_ylim(0, 2000)
    
    # Add value labels
    for bar, val in zip(bars2, total_updates):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{val}', ha='center', fontsize=14, fontweight='bold')
    
    # Add annotation
    fig.text(0.5, 0.02, 'Dataset size: 1,051 samples | Training: 100 epochs', 
             ha='center', fontsize=11, style='italic')
    
    plt.suptitle('The Batch Size Problem: Why Our Model Couldn\'t Learn', 
                 fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'batch_size_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created batch_size_comparison.png")

def create_task_examples():
    """Create a figure showing the three task formats."""
    fig, axes = plt.subplots(3, 1, figsize=(10, 8))
    
    # Remove axes
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
    
    # Task 1: Distance
    axes[0].text(0.05, 0.7, 'Distance Prediction:', fontsize=14, fontweight='bold')
    axes[0].text(0.05, 0.3, 'Input:', fontsize=12, color='#666')
    axes[0].text(0.15, 0.3, 'dist(c_847,c_3924)=', fontsize=12, family='monospace', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0'))
    axes[0].text(0.55, 0.3, 'Output:', fontsize=12, color='#666')
    axes[0].text(0.65, 0.3, '2451', fontsize=12, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#e8f5e9'))
    
    # Task 2: Location
    axes[1].text(0.05, 0.7, 'Location Prediction:', fontsize=14, fontweight='bold')
    axes[1].text(0.05, 0.3, 'Input:', fontsize=12, color='#666')
    axes[1].text(0.15, 0.3, 'loc(c_847)=', fontsize=12, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0'))
    axes[1].text(0.55, 0.3, 'Output:', fontsize=12, color='#666')
    axes[1].text(0.65, 0.3, '4862,1787', fontsize=12, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#e8f5e9'))
    
    # Task 3: Random Walk
    axes[2].text(0.05, 0.7, 'Random Walk Generation:', fontsize=14, fontweight='bold')
    axes[2].text(0.05, 0.5, 'Input:', fontsize=12, color='#666')
    axes[2].text(0.15, 0.5, 'walk_200=', fontsize=12, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#f0f0f0'))
    axes[2].text(0.05, 0.2, 'Output:', fontsize=12, color='#666')
    axes[2].text(0.15, 0.2, 'c_847,c_912,c_445,c_1332...', fontsize=12, family='monospace',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='#e8f5e9'))
    
    plt.suptitle('Three Tasks of Increasing Complexity', fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / 'task_examples.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created task_examples.png")

def create_debugging_timeline():
    """Create a timeline showing our debugging journey."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Timeline events
    events = [
        ("Initial Training", "Model can't memorize\n1,000 locations", 0, '#ff6b6b'),
        ("Hypothesis 1", "Model too small?\nLearning rate?", 1, '#ffd43b'),
        ("Sanity Check", "Random4to8 dataset\nStill fails!", 2, '#fab005'),
        ("Hypothesis 2", "Many-to-one issue?\nBiased dataset", 3, '#ffd43b'),
        ("Breakthrough", "Batch size 512\nwith 1051 samples\n= 2 updates/epoch!", 4, '#51cf66'),
        ("Solution", "Batch size 64\n= 16 updates/epoch\nModel learns!", 5, '#51cf66'),
    ]
    
    # Plot timeline
    for i, (phase, desc, pos, color) in enumerate(events):
        # Main timeline point
        ax.scatter(pos, 0, s=200, c=color, zorder=3, edgecolors='black', linewidth=2)
        
        # Phase label above
        ax.text(pos, 0.15, phase, ha='center', fontsize=11, fontweight='bold')
        
        # Description below
        ax.text(pos, -0.25, desc, ha='center', fontsize=9, 
                multialignment='center', color='#333')
        
        # Connect with line
        if i < len(events) - 1:
            ax.plot([pos, pos+1], [0, 0], 'k-', alpha=0.3, linewidth=2)
    
    # Styling
    ax.set_xlim(-0.5, 5.5)
    ax.set_ylim(-0.5, 0.3)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    ax.set_title('The Debugging Journey: From Failure to Understanding', 
                fontsize=14, fontweight='bold', pad=20)
    
    # Add arrow showing time
    ax.annotate('', xy=(5.3, -0.4), xytext=(-0.3, -0.4),
                arrowprops=dict(arrowstyle='->', lw=1.5, color='gray'))
    ax.text(2.5, -0.45, 'Time →', ha='center', fontsize=10, color='gray', style='italic')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'debugging_timeline.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Created debugging_timeline.png")

if __name__ == "__main__":
    print("Creating figures for blog post...")
    
    # Create all figures
    create_city_density_map()
    create_batch_size_comparison()
    create_task_examples()
    create_debugging_timeline()
    
    print("\nAll figures created successfully!")
    print(f"Location: {output_dir}")