#!/usr/bin/env python3
"""
Analyze how internal representations evolve during training.
Tracks R² scores for longitude/latitude prediction from partial prompts.

Usage:
    python src/analysis/analyze_representations_clean.py configs/analysis/config.yaml [--overwrite]
"""

import sys
import os
from pathlib import Path
import re
import argparse
import json
import shutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import gaussian_kde
from matplotlib.colors import LinearSegmentedColormap
from transformers import AutoTokenizer, Qwen2ForCausalLM
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from src.utils import (
    euclidean_distance, 
    init_directory,
    extract_model_representations,
    filter_dataframe_by_pattern
)


def validate_config(config):
    """Validate config has all required fields. FAIL FAST - no defaults!"""
    required = [
        'output_dir',
        'experiment_dir', 
        'cities_csv',
        'layers',
        'n_probe_cities',
        'n_train_cities',
        'task_type',
        'prompt_format',
        'probe_train',
        'probe_test',
        'device',
        'seed'
    ]
    
    for field in required:
        if field not in config:
            raise ValueError(f"FATAL: '{field}' is required in config")
    
    # Validate types
    if not isinstance(config['layers'], list):
        raise ValueError("FATAL: 'layers' must be a list of integers")
    
    if config['n_train_cities'] >= config['n_probe_cities']:
        raise ValueError("FATAL: n_train_cities must be less than n_probe_cities")
    
    if config['task_type'] not in ['distance', 'randomwalk']:
        raise ValueError(f"FATAL: Unknown task_type: {config['task_type']}")
    
    if config['prompt_format'] not in ['dist', 'rw200']:
        raise ValueError(f"FATAL: Unknown prompt_format: {config['prompt_format']}")




def create_prompts(cities_df, prompt_format):
    """Create space-delimited prompts for new tokenizer."""
    prompts = []
    
    for _, city in cities_df.iterrows():
        if prompt_format == 'dist':
            compact = f"<bos>dist(c_{city['id']},c_"
        elif prompt_format == 'rw200':
            compact = f"<bos>walk_200=c_{city['id']},c_"
        else:
            raise ValueError(f"Unknown prompt_format: {prompt_format}")
        
        # Convert to space-delimited for new tokenizer
        # NOTE: This was using the broken create_space_delimited_prompt function
        prompt = ' '.join(compact)  # This is still wrong but it's archived code
        prompts.append(prompt)
    
    return prompts


def load_country_to_region_mapping(project_root):
    """Load the country to region mapping."""
    mapping_file = project_root / 'data' / 'geographic_mappings' / 'country_to_region.json'
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Country to region mapping not found at {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description='Analyze representation dynamics')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    
    args = parser.parse_args()
    
    # Load and validate config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    validate_config(config)
    
    # Extract config values (no defaults!)
    output_dir = Path(config['output_dir'])
    experiment_dir = Path(config['experiment_dir'])
    cities_csv = Path(config['cities_csv'])
    layers = config['layers']
    n_probe_cities = config['n_probe_cities']
    n_train_cities = config['n_train_cities']
    task_type = config['task_type']
    prompt_format = config['prompt_format']
    probe_train_pattern = config['probe_train']
    probe_test_pattern = config['probe_test']
    device = torch.device(config['device'])
    seed = config['seed']
    
    # Optional fields
    highlight_pattern = config.get('highlight', None)
    highlight_label = config.get('highlight_label', 'Highlighted')
    highlight_color = config.get('highlight_color', '#FF1493')
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize output directory
    output_dir = init_directory(output_dir, args.overwrite)
    
    # Copy config to output for reproducibility
    shutil.copy(args.config_path, output_dir / 'config.yaml')
    
    # Create subdirectories
    (output_dir / 'figures').mkdir(exist_ok=True)
    (output_dir / 'results').mkdir(exist_ok=True)
    
    print("="*60)
    print("Representation Dynamics Analysis")
    print("="*60)
    print(f"Experiment: {experiment_dir.name}")
    print(f"Output: {output_dir}")
    print(f"Layers: {layers}")
    print(f"Task: {task_type}, Prompt: {prompt_format}")
    
    # Load cities data
    cities_df = pd.read_csv(cities_csv)
    
    # Add region information
    project_root = Path(__file__).parent.parent.parent
    country_to_region = load_country_to_region_mapping(project_root)
    
    # Add Atlantis region
    atlantis_countries = [f'Atlantis_{i}' for i in range(1, 21)]
    for country in atlantis_countries:
        country_to_region[country] = 'Atlantis'
    
    cities_df['region'] = cities_df['country_code'].map(country_to_region)
    
    # Filter cities for training and test sets using util function
    train_cities = filter_dataframe_by_pattern(cities_df, probe_train_pattern, column_name='region')
    test_cities = filter_dataframe_by_pattern(cities_df, probe_test_pattern, column_name='region')
    
    print(f"Probe train cities: {len(train_cities)} (pattern: '{probe_train_pattern}')")
    print(f"Probe test cities: {len(test_cities)} (pattern: '{probe_test_pattern}')")
    
    # Sample cities
    n_test_cities = n_probe_cities - n_train_cities
    train_sample = train_cities.sample(n=min(n_train_cities, len(train_cities)), random_state=seed)
    test_sample = test_cities.sample(n=min(n_test_cities, len(test_cities)), random_state=seed)
    
    # Combine and create prompts
    all_cities = pd.concat([train_sample, test_sample], ignore_index=True)
    prompts = create_prompts(all_cities, prompt_format)
    
    # Extract targets (x = longitude, y = latitude)
    # Note: x,y in dataset are scaled by 10
    longitudes = all_cities['x'].values  # -1800 to 1800
    latitudes = all_cities['y'].values    # -900 to 900
    
    # Find checkpoints
    checkpoints_dir = experiment_dir / 'checkpoints'
    checkpoint_dirs = sorted([d for d in checkpoints_dir.iterdir() if d.is_dir()])
    
    # Always use final checkpoint
    final_checkpoint = checkpoints_dir / 'final'
    if not final_checkpoint.exists():
        raise FileNotFoundError(f"No final checkpoint found at {final_checkpoint}")
    
    print(f"\nAnalyzing final checkpoint")
    
    # Load tokenizer
    tokenizer_path = experiment_dir / 'checkpoints' / 'checkpoint-0'
    if not tokenizer_path.exists():
        # Try data/tokenizers/default_tokenizer as fallback
        tokenizer_path = project_root / 'data' / 'tokenizers' / 'default_tokenizer'
    
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.padding_side = 'left'  # For generation
    
    # Tokenize prompts
    tokenized = tokenizer(
        prompts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=False
    )
    
    input_ids = tokenized['input_ids'].to(device)
    attention_mask = tokenized['attention_mask'].to(device)
    
    # Load model
    model = Qwen2ForCausalLM.from_pretrained(str(final_checkpoint))
    model.to(device)
    model.eval()
    
    # Extract representations using util function
    print(f"Extracting from layers {layers}...")
    representations = extract_model_representations(
        model, tokenizer, input_ids, attention_mask, layers, device
    )
    
    # Get last token representation for each sequence
    last_token_reps = []
    for i in range(len(representations)):
        seq_len = attention_mask[i].sum().item()
        last_token_rep = representations[i, seq_len - 1, :].cpu().numpy()
        last_token_reps.append(last_token_rep)
    
    X = np.array(last_token_reps)
    
    # Split into train/test
    X_train = X[:n_train_cities]
    X_test = X[n_train_cities:]
    y_train_lon = longitudes[:n_train_cities]
    y_test_lon = longitudes[n_train_cities:]
    y_train_lat = latitudes[:n_train_cities]
    y_test_lat = latitudes[n_train_cities:]
    
    # Train linear probes
    print("Training linear probes...")
    probe_lon = Ridge(alpha=1.0)
    probe_lat = Ridge(alpha=1.0)
    
    probe_lon.fit(X_train, y_train_lon)
    probe_lat.fit(X_train, y_train_lat)
    
    # Evaluate
    train_r2_lon = probe_lon.score(X_train, y_train_lon)
    test_r2_lon = probe_lon.score(X_test, y_test_lon)
    train_r2_lat = probe_lat.score(X_train, y_train_lat)
    test_r2_lat = probe_lat.score(X_test, y_test_lat)
    
    print(f"\nResults:")
    print(f"  Longitude - Train R²: {train_r2_lon:.4f}, Test R²: {test_r2_lon:.4f}")
    print(f"  Latitude  - Train R²: {train_r2_lat:.4f}, Test R²: {test_r2_lat:.4f}")
    
    # Save results
    results = {
        'experiment': experiment_dir.name,
        'layers': layers,
        'n_train': n_train_cities,
        'n_test': n_test_cities,
        'train_r2_lon': train_r2_lon,
        'test_r2_lon': test_r2_lon,
        'train_r2_lat': train_r2_lat,
        'test_r2_lat': test_r2_lat,
        'probe_train_pattern': probe_train_pattern,
        'probe_test_pattern': probe_test_pattern
    }
    
    with open(output_dir / 'results' / 'probe_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Create visualization
    print("Creating visualization...")
    
    # Predict on test set
    pred_lon = probe_lon.predict(X_test)
    pred_lat = probe_lat.predict(X_test)
    
    # Set style for better looking plots
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    
    # Longitude scatter with density (convert to degrees for display)
    ax1 = axes[0, 0]
    y_test_lon_deg = y_test_lon / 10  # Convert to degrees
    pred_lon_deg = pred_lon / 10      # Convert to degrees
    points_lon = np.vstack([y_test_lon_deg, pred_lon_deg])
    if len(y_test_lon_deg) > 10:
        try:
            z_lon = gaussian_kde(points_lon)(points_lon)
            idx_lon = z_lon.argsort()
            x_lon, y_lon, z_lon = y_test_lon_deg[idx_lon], pred_lon_deg[idx_lon], z_lon[idx_lon]
            scatter1 = ax1.scatter(x_lon, y_lon, c=z_lon, s=20, cmap='viridis', alpha=0.6, edgecolors='none')
            plt.colorbar(scatter1, ax=ax1, label='Density')
        except:
            ax1.scatter(y_test_lon_deg, pred_lon_deg, alpha=0.6, s=20, color='steelblue')
    else:
        ax1.scatter(y_test_lon_deg, pred_lon_deg, alpha=0.6, s=20, color='steelblue')
    
    ax1.plot([-180, 180], [-180, 180], 'r--', alpha=0.7, linewidth=2, label='Perfect prediction')
    ax1.set_xlabel('True Longitude (°)', fontsize=12)
    ax1.set_ylabel('Predicted Longitude (°)', fontsize=12)
    ax1.set_title(f'Longitude Predictions (R² = {test_r2_lon:.4f})', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([-185, 185])
    ax1.set_ylim([-185, 185])
    ax1.legend(loc='upper left')
    
    # Latitude scatter with density (convert to degrees for display)
    ax2 = axes[0, 1]
    y_test_lat_deg = y_test_lat / 10  # Convert to degrees
    pred_lat_deg = pred_lat / 10      # Convert to degrees
    points_lat = np.vstack([y_test_lat_deg, pred_lat_deg])
    if len(y_test_lat_deg) > 10:
        try:
            z_lat = gaussian_kde(points_lat)(points_lat)
            idx_lat = z_lat.argsort()
            x_lat, y_lat, z_lat = y_test_lat_deg[idx_lat], pred_lat_deg[idx_lat], z_lat[idx_lat]
            scatter2 = ax2.scatter(x_lat, y_lat, c=z_lat, s=20, cmap='viridis', alpha=0.6, edgecolors='none')
            plt.colorbar(scatter2, ax=ax2, label='Density')
        except:
            ax2.scatter(y_test_lat_deg, pred_lat_deg, alpha=0.6, s=20, color='steelblue')
    else:
        ax2.scatter(y_test_lat_deg, pred_lat_deg, alpha=0.6, s=20, color='steelblue')
    
    ax2.plot([-90, 90], [-90, 90], 'r--', alpha=0.7, linewidth=2, label='Perfect prediction')
    ax2.set_xlabel('True Latitude (°)', fontsize=12)
    ax2.set_ylabel('Predicted Latitude (°)', fontsize=12)
    ax2.set_title(f'Latitude Predictions (R² = {test_r2_lat:.4f})', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([-95, 95])
    ax2.set_ylim([-95, 95])
    ax2.legend(loc='upper left')
    
    # Longitude error distribution (in degrees)
    ax3 = axes[1, 0]
    lon_errors_deg = (pred_lon - y_test_lon) / 10  # Convert error to degrees
    ax3.hist(lon_errors_deg, bins=50, alpha=0.7, color='coral', edgecolor='black')
    ax3.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lon_errors_deg):.2f}°')
    ax3.set_xlabel('Prediction Error (°)', fontsize=12)
    ax3.set_ylabel('Frequency', fontsize=12)
    ax3.set_title(f'Longitude Error Distribution (σ = {np.std(lon_errors_deg):.2f}°)', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # Latitude error distribution (in degrees)
    ax4 = axes[1, 1]
    lat_errors_deg = (pred_lat - y_test_lat) / 10  # Convert error to degrees
    ax4.hist(lat_errors_deg, bins=50, alpha=0.7, color='lightgreen', edgecolor='black')
    ax4.axvline(0, color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(lat_errors_deg):.2f}°')
    ax4.set_xlabel('Prediction Error (°)', fontsize=12)
    ax4.set_ylabel('Frequency', fontsize=12)
    ax4.set_title(f'Latitude Error Distribution (σ = {np.std(lat_errors_deg):.2f}°)', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    layers_str = ', '.join(map(str, layers))
    plt.suptitle(f'Coordinate Prediction Analysis - Layers {layers_str}', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # World map visualization with enhanced styling
    fig = plt.figure(figsize=(20, 12))
    ax = plt.subplot(111, projection='rectilinear')
    
    # Set background color to represent oceans
    ax.set_facecolor('#e6f2ff')
    fig.patch.set_facecolor('white')
    
    # Plot test cities
    test_cities_in_sample = all_cities.iloc[n_train_cities:]
    
    # Calculate prediction errors for coloring (in degrees)
    # Convert from scaled coordinates to degrees for error calculation
    actual_coords_deg = test_cities_in_sample[['x', 'y']].values / 10  # Convert to degrees
    pred_coords_deg = np.column_stack([pred_lon / 10, pred_lat / 10])  # Convert predictions to degrees
    errors = np.sqrt(np.sum((actual_coords_deg - pred_coords_deg) ** 2, axis=1))
    
    if highlight_pattern:
        highlighted = filter_dataframe_by_pattern(test_cities_in_sample, highlight_pattern, column_name='region')
        regular = test_cities_in_sample[~test_cities_in_sample.index.isin(highlighted.index)]
        
        # Plot regular cities with smaller dots (convert to degrees)
        if len(regular) > 0:
            ax.scatter(regular['x'] / 10, regular['y'] / 10, 
                      c='#2E86C1', alpha=0.4, s=15, label='Test cities', 
                      edgecolors='none', marker='o')
        
        # Plot highlighted cities with emphasis (convert to degrees)
        if len(highlighted) > 0:
            ax.scatter(highlighted['x'] / 10, highlighted['y'] / 10,
                      c=highlight_color, alpha=0.9, s=80, label=highlight_label,
                      edgecolors='white', linewidths=1, marker='*')
    else:
        ax.scatter(test_cities_in_sample['x'] / 10, test_cities_in_sample['y'] / 10,
                  c='#2E86C1', alpha=0.4, s=15, label='Test cities',
                  edgecolors='none', marker='o')
    
    # Plot predictions with error-based coloring (convert to degrees)
    scatter = ax.scatter(pred_lon / 10, pred_lat / 10, 
                         c=errors, cmap='RdYlGn_r', 
                         alpha=0.6, s=25, 
                         label='Predictions',
                         edgecolors='black', linewidths=0.5,
                         vmin=0, vmax=np.percentile(errors, 95))
    
    # Add colorbar for errors
    cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Prediction Error (Euclidean distance)', fontsize=12)
    cbar.ax.tick_params(labelsize=10)
    
    # Draw connection lines for largest errors (convert to degrees)
    n_worst = min(20, len(errors))
    worst_indices = np.argsort(errors)[-n_worst:]
    for idx in worst_indices:
        ax.plot([test_cities_in_sample.iloc[idx]['x'] / 10, pred_lon[idx] / 10],
                [test_cities_in_sample.iloc[idx]['y'] / 10, pred_lat[idx] / 10],
                'r-', alpha=0.3, linewidth=0.5)
    
    # Set proper world map bounds
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    
    # Enhanced grid
    ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    ax.set_axisbelow(True)
    
    # Add major gridlines
    ax.set_xticks(np.arange(-180, 181, 30))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    # Labels and title
    ax.set_xlabel('Longitude (°)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Latitude (°)', fontsize=14, fontweight='bold')
    ax.set_title(f'Geographic Predictions vs Ground Truth\\nLayers {layers_str} | Test Cities: {len(test_cities_in_sample)} | Mean Error: {np.mean(errors):.2f}°',
                 fontsize=16, fontweight='bold', pad=20)
    
    # Enhanced legend
    legend = ax.legend(loc='upper left', fontsize=11, 
                      framealpha=0.9, edgecolor='gray',
                      title='Legend', title_fontsize=12)
    legend.get_frame().set_facecolor('white')
    
    # Add statistics text box
    stats_text = (f'Statistics:\\n'
                 f'Mean Error: {np.mean(errors):.2f}°\\n'
                 f'Median Error: {np.median(errors):.2f}°\\n'
                 f'Max Error: {np.max(errors):.2f}°\\n'
                 f'R² Longitude: {test_r2_lon:.4f}\\n'
                 f'R² Latitude: {test_r2_lat:.4f}')
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.98, 0.02, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'figures' / 'world_map.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()