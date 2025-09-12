#!/usr/bin/env python3
"""
Analyze how internal representations evolve during training across all checkpoints.
Tracks R² scores for x/y coordinate prediction from partial prompts.
Generates plots and animated GIF showing evolution of predictions on world map.

Usage:
    python representation_dynamics.py <experiment_dir> <cities_csv> [--layers 3,4]
"""

import sys
import os
from pathlib import Path
import re
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, Qwen2ForCausalLM
from typing import List, Dict, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory, filter_dataframe_by_pattern
# from src.representation_extractor import RepresentationExtractor  # Not needed - using output_hidden_states directly
import json




# Load country to region mapping from JSON file
def load_region_mapping(mapping_path):
    """Load region mapping from JSON file."""
    with open(mapping_path, 'r') as f:
        return json.load(f)

# Global variable for region mapping (will be loaded from JSON)
country_to_region = {}

# Define region colors - using distinct colors for each region
region_colors = {
    'North America': '#2E7D32',     # Dark Green
    'South America': '#FDD835',     # Yellow  
    'Africa': '#D32F2F',           # Red
    'Western Europe': '#1976D2',    # Blue
    'Eastern Europe': '#795548',    # Brown
    'Middle East': '#F57C00',      # Orange
    'India': '#9C27B0',            # Purple
    'China': '#C62828',            # Dark Red
    'Korea': '#00ACC1',            # Cyan
    'Japan': '#E91E63',            # Pink
    'Southeast Asia': '#43A047',   # Light Green
    'Central Asia': '#FFB300',     # Amber
    'Oceania': '#00BCD4',          # Light Cyan
    'Unknown': '#9E9E9E',          # Gray
}


def analyze_checkpoint(checkpoint_path, step, partial_input_ids, partial_attention_mask, 
                       x_train, x_test, y_train, y_test, n_train_cities, 
                       device, layer_indices, return_predictions=False):
    """Analyze a single checkpoint and return R² scores"""
    
    # Load model
    model = Qwen2ForCausalLM.from_pretrained(checkpoint_path)
    model.eval()
    model = model.to(device)
    
    # Ensure inputs are on the same device as model
    partial_input_ids_device = partial_input_ids.to(device)
    partial_attention_mask_device = partial_attention_mask.to(device) if partial_attention_mask is not None else None
    
    # Get representations using output_hidden_states like the old working version
    with torch.no_grad():
        outputs = model(partial_input_ids_device, partial_attention_mask_device, output_hidden_states=True)
    
    # Extract and concatenate the specified layers
    layer_reps = []
    for idx in layer_indices:
        # hidden_states includes embedding layer at index 0, so layer N is at index N+1
        layer_reps.append(outputs.hidden_states[idx + 1])  # +1 because index 0 is embeddings
    
    # Concatenate layers if multiple
    if len(layer_reps) > 1:
        partial_representations = torch.cat(layer_reps, dim=-1)
    else:
        partial_representations = layer_reps[0]
    
    # Get last token representations (concatenating the last 3 tokens) - EXACTLY LIKE NOTEBOOK
    underscore_reps = partial_representations[:, -1, :]  # "_" (last token)
    c_reps = partial_representations[:, -2, :]           # "c" 
    comma_reps = partial_representations[:, -3, :]       # "," 
    
    # Concatenate all three representations
    partial_last_token_reps = torch.cat([comma_reps, c_reps, underscore_reps], dim=1)
    
    # Convert to numpy
    partial_reps_np = partial_last_token_reps.cpu().numpy()
    
    # Print shape after extraction
    print(f"Extracted representations shape: {partial_reps_np.shape}")
    
    # Split into train and test
    X_train_coord = partial_reps_np[:n_train_cities]
    X_test_coord = partial_reps_np[n_train_cities:]
    
    # Train x probe
    x_probe = Ridge(alpha=10.0)
    x_probe.fit(X_train_coord, x_train)
    x_train_pred = x_probe.predict(X_train_coord)
    x_test_pred = x_probe.predict(X_test_coord)
    
    # Train y probe
    y_probe = Ridge(alpha=10.0)
    y_probe.fit(X_train_coord, y_train)
    y_train_pred = y_probe.predict(X_train_coord)
    y_test_pred = y_probe.predict(X_test_coord)
    
    # Calculate metrics
    x_train_r2 = r2_score(x_train, x_train_pred)
    x_test_r2 = r2_score(x_test, x_test_pred)
    y_train_r2 = r2_score(y_train, y_train_pred)
    y_test_r2 = r2_score(y_test, y_test_pred)
    
    x_test_mae = mean_absolute_error(x_test, x_test_pred)
    y_test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Calculate Euclidean distance error
    pred_distances = np.sqrt((x_test - x_test_pred)**2 + (y_test - y_test_pred)**2)
    
    mean_dist_error = np.mean(pred_distances)
    median_dist_error = np.median(pred_distances)
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()
    
    result = {
        'step': step,
        'x_train_r2': x_train_r2,
        'x_test_r2': x_test_r2,
        'y_train_r2': y_train_r2,
        'y_test_r2': y_test_r2,
        'x_test_mae': x_test_mae,
        'y_test_mae': y_test_mae,
        'mean_dist_error': mean_dist_error,
        'median_dist_error': median_dist_error
    }
    
    if return_predictions:
        result['x_test_pred'] = x_test_pred
        result['y_test_pred'] = y_test_pred
        result['x_train_pred'] = x_train_pred
        result['y_train_pred'] = y_train_pred
    
    return result


def create_world_map_frame(x_pred, y_pred, x_true, y_true, 
                          test_city_info, step, r2_x, r2_y, mean_error):
    """Create a single frame for the world map animation with regions colored"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Get regions for test cities
    test_regions = []
    for city in test_city_info:
        country = city['country']
        # Use Unknown for unmapped countries (keep this one as it's needed for visualization)
        region = country_to_region.get(country, 'Unknown')
        test_regions.append(region)
    
    # Plot predicted test locations by region
    for region in region_colors.keys():
        # Get indices for this region
        region_mask = [r == region for r in test_regions]
        if sum(region_mask) > 0:
            region_xs_pred = x_pred[region_mask]
            region_ys_pred = y_pred[region_mask]
            region_xs_true = x_true[region_mask]
            region_ys_true = y_true[region_mask]
            
            # Plot true locations with smaller markers (scale by /10 for display)
            ax.scatter(region_xs_true / 10, region_ys_true / 10, 
                      s=15, alpha=0.3, c='gray',
                      edgecolors='none')
            
            # Plot predicted locations (scale by /10 for display)
            ax.scatter(region_xs_pred / 10, region_ys_pred / 10, 
                      s=30, alpha=0.7, c=region_colors[region],
                      label=f'{region} ({sum(region_mask)})', 
                      edgecolors='black', linewidth=0.3)
    
    # Add grid and reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)  # Equator
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)  # Prime Meridian
    
    # Calculate mean positions of predicted locations for each region
    region_label_positions = {}
    for region in region_colors.keys():
        # Get indices for this region
        region_mask = [r == region for r in test_regions]
        if sum(region_mask) > 0:
            region_xs_pred = x_pred[region_mask]
            region_ys_pred = y_pred[region_mask]
            # Calculate mean position of predictions for this region
            mean_x = np.mean(region_xs_pred) / 10  # Scale for display
            mean_y = np.mean(region_ys_pred) / 10  # Scale for display
            region_label_positions[region] = (mean_x, mean_y)
    
    # Add region labels at the mean predicted positions
    for region, (x, y) in region_label_positions.items():
        fontsize = 9 if 'Europe' in region else 10
        ax.text(x, y, region, fontsize=fontsize, fontweight='bold', 
               ha='center', va='center', alpha=0.6,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
    
    # Set limits and labels
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)
    
    # Add title with metrics
    ax.set_title(f'Step {step:,} | X R²: {r2_x:.3f} | Y R²: {r2_y:.3f} | Mean Error: {mean_error:.2f}', 
                fontsize=16, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, fontsize=8, 
             bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    
    # Add tick marks
    ax.set_xticks(range(-180, 181, 60))
    ax.set_yticks(range(-90, 91, 30))
    
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Analyze representation dynamics')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    
    args = parser.parse_args()
    
    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract config values
    output_dir = Path(config['output_dir'])
    experiment_dir = Path(config['experiment_dir'])
    cities_csv = Path(config['cities_csv'])
    layers = config['layers']
    n_probe_cities = config['n_probe_cities']
    n_train_cities = config['n_train_cities']
    device = torch.device(config['device'])
    seed = config['seed']
    # Required fields - no defaults
    task_type = config['task_type']
    prompt_format = config['prompt_format']
    probe_train_pattern = config['probe_train']
    probe_test_pattern = config['probe_test']
    
    # Optional checkpoint parameter - can be "final", a number, or None for all
    checkpoint_param = config.get('checkpoint', None)
    
    # Optional fields with explicit None
    highlight_pattern = config.get('highlight', None)
    highlight_label = config.get('highlight_label', None)
    highlight_color = config.get('highlight_color', None)
    
    # Initialize output directory
    output_dir = init_directory(output_dir, overwrite=args.overwrite)
    
    # Copy config to output
    import shutil
    shutil.copy(args.config_path, output_dir / 'config.yaml')
    
    # Setup paths from config
    layer_indices = layers
    checkpoints_dir = experiment_dir / 'checkpoints'
    training_config_path = experiment_dir / 'config.yaml'
    
    if not training_config_path.exists():
        print(f"Error: Training config not found at {training_config_path}")
        sys.exit(1)
    
    # Load training config for model info
    with open(training_config_path, 'r') as f:
        training_config = yaml.safe_load(f)
    
    experiment_name = experiment_dir.name
    
    print("="*60)
    print("Representation Dynamics Analysis")
    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Task type: {task_type}")
    print(f"Prompt format: {prompt_format}")
    if 'model' in training_config:
        print(f"Model layers: {training_config['model']['num_hidden_layers']}")
        print(f"Hidden size: {training_config['model']['hidden_size']}")
    print(f"Extracting from layers: {layer_indices}")
    
    # Handle checkpoint parameter
    checkpoint_dirs = []
    
    if checkpoint_param is not None:
        # Single checkpoint specified
        if checkpoint_param == "final":
            final_path = checkpoints_dir / 'final'
            if final_path.exists():
                # Try to get step from trainer_state.json
                trainer_state_path = final_path / 'trainer_state.json'
                if trainer_state_path.exists():
                    import json
                    with open(trainer_state_path, 'r') as f:
                        trainer_state = json.load(f)
                        step = trainer_state.get('global_step', 99999)
                else:
                    step = 99999  # Default for final
                checkpoint_dirs = [(step, final_path)]
                print(f"Using final checkpoint at step {step}")
            else:
                print(f"Error: Final checkpoint not found at {final_path}")
                sys.exit(1)
        else:
            # Numeric checkpoint specified
            checkpoint_path = checkpoints_dir / f'checkpoint-{checkpoint_param}'
            if checkpoint_path.exists():
                checkpoint_dirs = [(int(checkpoint_param), checkpoint_path)]
                print(f"Using checkpoint-{checkpoint_param}")
            else:
                print(f"Error: Checkpoint not found at {checkpoint_path}")
                sys.exit(1)
    else:
        # No checkpoint specified - analyze all checkpoints
        for item in sorted(os.listdir(checkpoints_dir)):
            if item.startswith('checkpoint-') and item != 'checkpoint-latest':
                match = re.match(r'checkpoint-(\d+)', item)
                if match:
                    step = int(match.group(1))
                    checkpoint_dirs.append((step, checkpoints_dir / item))
        
        # Add final checkpoint if it exists
        final_path = checkpoints_dir / 'final'
        if final_path.exists():
            # Get the max step from existing checkpoints
            max_step = max([s for s, _ in checkpoint_dirs]) if checkpoint_dirs else 0
            # Final checkpoint should be at max step (or we can assign it the true final step)
            checkpoint_dirs.append((max_step if max_step > 0 else 39080, final_path))
        
        # Sort by step number
        checkpoint_dirs = sorted(checkpoint_dirs, key=lambda x: x[0])
        
        print(f"\nFound {len(checkpoint_dirs)} checkpoints")
    
    print(f"Using device: {device}")
    
    # Load region mapping from config
    global country_to_region
    region_mapping_path = config.get('region_mapping_path', 'data/geographic_mappings/country_to_region.json')
    mapping_path = Path(region_mapping_path)
    if not mapping_path.is_absolute():
        mapping_path = project_root / mapping_path
    if mapping_path.exists():
        country_to_region = load_region_mapping(mapping_path)
        print(f"Loaded region mapping from {mapping_path}")
    else:
        print(f"Warning: Region mapping file not found at {mapping_path}")
    
    # Load tokenizer - no fallback
    tokenizer_path = checkpoints_dir / 'checkpoint-0'
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)
    
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.padding_side = 'left'  # For generation
    print(f"Tokenizer loaded from {tokenizer_path}")
    
    # Load cities data
    cities_df = pd.read_csv(cities_csv)
    print(f"Loaded {len(cities_df)} cities")
    
    # Use the scaled coordinates directly (x and y are already in the dataset)
    cities_df['row_id'] = cities_df['city_id']     # Use city_id as row_id
    
    # Apply filtering based on patterns
    if probe_train_pattern != '.*' or probe_test_pattern != '.*':
        train_cities = filter_dataframe_by_pattern(cities_df, probe_train_pattern, column_name='region')
        test_cities = filter_dataframe_by_pattern(cities_df, probe_test_pattern, column_name='region')
        print(f"Probe train cities: {len(train_cities)} (pattern: '{probe_train_pattern}')")
        print(f"Probe test cities: {len(test_cities)} (pattern: '{probe_test_pattern}')")
        
        # Sample from filtered sets
        n_test_cities = n_probe_cities - n_train_cities
        train_sample = train_cities.sample(n=min(n_train_cities, len(train_cities)), random_state=seed)
        test_sample = test_cities.sample(n=min(n_test_cities, len(test_cities)), random_state=seed)
        
        # Combine samples
        sampled_cities = pd.concat([train_sample, test_sample], ignore_index=True)
    else:
        # Original sampling without filtering
        np.random.seed(seed)
        sampled_city_indices = np.random.choice(len(cities_df), size=min(n_probe_cities, len(cities_df)), replace=False)
        sampled_cities = cities_df.iloc[sampled_city_indices]
    
    print(f"Sampled {len(sampled_cities)} cities for probing")
    print(f"Will use {n_train_cities} for training, {len(sampled_cities) - n_train_cities} for testing")
    
    # Create partial prompts ending at "_" after "c_"
    partial_prompts = []
    city_info = []
    
    for idx, city in sampled_cities.iterrows():
        # Create partial prompt based on prompt_format
        if prompt_format == 'dist':
            # Match training format exactly: "<bos> d i s t ( c _ X X X , c _"
            dist_str = f"dist(c_{city['row_id']},c_"
            spaced_str = ' '.join(dist_str)
            prompt = f"<bos> {spaced_str}"  # Space after <bos> to match training
        elif prompt_format == 'rw200':
            walk_str = f"walk_200=c_{city['row_id']},c_"
            spaced_str = ' '.join(walk_str)
            prompt = f"<bos> {spaced_str}"  # Space after <bos> to match training
        else:
            raise ValueError(f"Unknown prompt_format: {prompt_format}")
        partial_prompts.append(prompt)
        city_info.append({
            'row_id': city['row_id'],
            'name': city['asciiname'],
            'x': city['x'],
            'y': city['y'],
            'country': city['country_code']
        })
    
    print(f"Created {len(partial_prompts)} partial prompts")
    
    # Tokenize partial prompts with LEFT padding - EXACTLY LIKE NOTEBOOK
    tokenized_partial = tokenizer(
        partial_prompts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )
    
    partial_input_ids = tokenized_partial['input_ids'].to(device)
    partial_attention_mask = tokenized_partial['attention_mask'].to(device)
    
    print(f"Tokenized shape: {partial_input_ids.shape}")
    
    # Extract x and y as targets (scaled values)
    xs = np.array([c['x'] for c in city_info])
    ys = np.array([c['y'] for c in city_info])
    
    # Split into train and test
    x_train = xs[:n_train_cities]
    x_test = xs[n_train_cities:]
    y_train = ys[:n_train_cities]
    y_test = ys[n_train_cities:]
    
    # Get test city info for visualization
    test_city_info = city_info[n_train_cities:]
    
    # Analyze all checkpoints
    print("\n" + "="*60)
    print("Analyzing Checkpoints")
    print("="*60)
    
    results = []
    predictions_for_animation = []
    
    for step, checkpoint_path in tqdm(checkpoint_dirs, desc="Processing"):
        print(f"\nStep {step}:")
        
        try:
            # Get predictions for animation on selected checkpoints
            return_preds = (len(checkpoint_dirs) <= 20 or 
                          step % max(1, checkpoint_dirs[-1][0] // 10) == 0 or 
                          step == checkpoint_dirs[0][0] or 
                          step == checkpoint_dirs[-1][0])
            
            result = analyze_checkpoint(
                checkpoint_path, step,
                partial_input_ids, partial_attention_mask,
                x_train, x_test, y_train, y_test,
                n_train_cities, device, layer_indices,
                return_predictions=return_preds
            )
            
            # Try to get loss from trainer_state.json
            import json
            trainer_state_path = checkpoint_path / 'trainer_state.json'
            if trainer_state_path.exists():
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                    # Get the loss closest to this checkpoint step
                    losses = [(h['loss'], h['step']) for h in trainer_state.get('log_history', []) if 'loss' in h]
                    if losses:
                        # Find loss closest to but not after the checkpoint step
                        valid_losses = [l for l in losses if l[1] <= step]
                        if valid_losses:
                            result['loss'] = valid_losses[-1][0]
                        else:
                            result['loss'] = losses[0][0]  # Use first if no valid ones
                    else:
                        result['loss'] = None
            else:
                result['loss'] = None
                
            results.append(result)
            
            if return_preds:
                predictions_for_animation.append({
                    'step': step,
                    'x_pred': result['x_test_pred'],
                    'y_pred': result['y_test_pred'],
                    'x_r2': result['x_test_r2'],
                    'y_r2': result['y_test_r2'],
                    'mean_error': result['mean_dist_error']
                })
            
            print(f"  X R²: {result['x_test_r2']:.3f}, Y R²: {result['y_test_r2']:.3f}")
            print(f"  Mean dist error: {result['mean_dist_error']:.2f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    if not results:
        print("No successful checkpoint analyses!")
        return
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('step')
    
    # Use the output_dir from config
    analysis_dir = output_dir
    print(f"\nAnalysis directory: {analysis_dir}")
    
    # Save results
    output_csv = analysis_dir / 'representation_dynamics.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    print(results_df[['step', 'x_test_r2', 'y_test_r2', 'mean_dist_error']])
    
    # Create dynamics plot with vertically arranged subplots
    print("\n" + "="*60)
    print("Generating Dynamics Plot")
    print("="*60)
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Top subplot: Loss (left y-axis) and Mean Location Error (right y-axis)
    ax1 = axes[0]
    
    # Plot loss on left y-axis if available
    if 'loss' in results_df.columns and results_df['loss'].notna().any():
        color = 'tab:blue'
        ax1.plot(results_df['step'], results_df['loss'], color=color, linewidth=2, label='Loss')
        ax1.set_ylabel('Training Loss', color=color, fontsize=12)
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Add text with final loss
        final_loss = results_df['loss'].iloc[-1]
        if pd.notna(final_loss):
            ax1.text(0.02, 0.95, f'Final Loss: {final_loss:.3f}', 
                   transform=ax1.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        # If no loss data, just use the left axis for labeling
        ax1.set_ylabel('Training Loss (Not Available)', fontsize=12)
        ax1.text(0.5, 0.5, 'Loss data not available', 
               transform=ax1.transAxes, ha='center', va='center', alpha=0.5)
    
    # Create twin axis for distance error
    ax1_twin = ax1.twinx()
    color = 'tab:red'
    ax1_twin.plot(results_df['step'], results_df['mean_dist_error'], color=color, linewidth=2, label='Distance Error')
    ax1_twin.set_ylabel('Mean Location Error', color=color, fontsize=12)
    ax1_twin.tick_params(axis='y', labelcolor=color)
    ax1_twin.set_yscale('log')
    
    # Add reference lines for distance
    reference_distances = [10000, 5000, 2000, 1000, 500]
    for dist in reference_distances:
        if dist >= results_df['mean_dist_error'].min() and dist <= results_df['mean_dist_error'].max():
            ax1_twin.axhline(y=dist, color='gray', linestyle=':', alpha=0.2)
            ax1_twin.text(results_df['step'].max() * 1.01, dist, f'{dist}', 
                         va='center', ha='left', fontsize=8, alpha=0.5)
    
    # Add text with final error
    ax1_twin.text(0.98, 0.95, f'Final Error: {results_df["mean_dist_error"].iloc[-1]:.2f}', 
                 transform=ax1_twin.transAxes, ha='right', va='top', color=color,
                 bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax1.set_title('Training Loss & Location Error', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: R² Scores and Location Error
    ax2 = axes[1]
    
    # Plot R² scores on left y-axis
    ax2.plot(results_df['step'], results_df['x_test_r2'], 'b-', label='X R²', linewidth=2)
    ax2.plot(results_df['step'], results_df['y_test_r2'], 'r-', label='Y R²', linewidth=2)
    # Add average as a thicker line
    avg_r2 = (results_df['x_test_r2'] + results_df['y_test_r2']) / 2
    ax2.plot(results_df['step'], avg_r2, 'purple', label='Average R²', linewidth=2.5, alpha=0.7)
    
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Test R² Score', fontsize=12)
    ax2.set_title('Coordinate Prediction Performance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([-0.2, 1.0])
    
    # Add horizontal line at R²=0 for reference
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Create twin axis for haversine distance
    ax2_twin = ax2.twinx()
    color = 'tab:green'
    ax2_twin.plot(results_df['step'], results_df['mean_dist_error'], '--', 
                  color=color, linewidth=2, label='Location Error (km)', alpha=0.7)
    ax2_twin.set_ylabel('Mean Location Error', color=color, fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor=color)
    ax2_twin.set_yscale('log')
    
    # Set reasonable y-limits for distance
    max_dist = results_df['mean_dist_error'].max()
    min_dist = results_df['mean_dist_error'].min()
    ax2_twin.set_ylim([min_dist * 0.8, max_dist * 1.2])
    
    # Combine legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    # Add text with final values
    ax2.text(0.02, 0.95, f'Final R²:\nX: {results_df["x_test_r2"].iloc[-1]:.3f}\nY: {results_df["y_test_r2"].iloc[-1]:.3f}\nError: {results_df["mean_dist_error"].iloc[-1]:.2f}', 
           transform=ax2.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle(f'Representation Dynamics: {experiment_name}', fontsize=16, y=1.01)
    plt.tight_layout()
    
    # Save plot
    plot_path = analysis_dir / 'dynamics_plot.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"R² plot saved to {plot_path}")
    plt.close()
    
    # Create animated GIF of world map evolution
    if predictions_for_animation:
        print("\n" + "="*60)
        print("Generating World Map Animation")
        print("="*60)
        
        # Create frames
        frames = []
        for pred_data in tqdm(predictions_for_animation, desc="Creating frames"):
            fig = create_world_map_frame(
                pred_data['x_pred'], pred_data['y_pred'],
                x_test, y_test, test_city_info,
                pred_data['step'], pred_data['x_r2'], 
                pred_data['y_r2'], pred_data['mean_error']
            )
            frames.append(fig)
        
        # Save as GIF
        gif_path = analysis_dir / 'world_map_evolution.gif'
        
        # Save frames as individual images and then combine into GIF
        from PIL import Image
        images = []
        
        for i, fig in enumerate(frames):
            # Save figure to temporary file
            temp_path = analysis_dir / f'temp_frame_{i:03d}.png'
            fig.savefig(temp_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Load as PIL Image
            images.append(Image.open(temp_path))
        
        # Save as GIF
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=500,  # 500ms per frame
            loop=0
        )
        
        # Clean up temporary files
        for i in range(len(frames)):
            temp_path = analysis_dir / f'temp_frame_{i:03d}.png'
            if temp_path.exists():
                temp_path.unlink()
        
        print(f"World map animation saved to {gif_path}")
        
        # Also save the final frame as a static image
        if predictions_for_animation:
            final_pred = predictions_for_animation[-1]
            final_fig = create_world_map_frame(
                final_pred['x_pred'], final_pred['y_pred'],
                x_test, y_test, test_city_info,
                final_pred['step'], final_pred['x_r2'], 
                final_pred['y_r2'], final_pred['mean_error']
            )
            final_map_path = analysis_dir / 'world_map_final.png'
            final_fig.savefig(final_map_path, dpi=150, bbox_inches='tight')
            plt.close(final_fig)
            print(f"Final world map saved to {final_map_path}")
    
    # Print final statistics
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    
    initial = results_df.iloc[0]
    final = results_df.iloc[-1]
    
    print(f"\nInitial (Step {initial['step']}):")
    print(f"  X R²: {initial['x_test_r2']:.3f}")
    print(f"  Y R²:  {initial['y_test_r2']:.3f}")
    print(f"  Distance Error: {initial['mean_dist_error']:.2f}")
    
    print(f"\nFinal (Step {final['step']}):")
    print(f"  X R²: {final['x_test_r2']:.3f}")
    print(f"  Y R²:  {final['y_test_r2']:.3f}")
    print(f"  Distance Error: {final['mean_dist_error']:.2f}")
    
    print(f"\nImprovement:")
    print(f"  X R²: {final['x_test_r2'] - initial['x_test_r2']:+.3f}")
    print(f"  Y R²:  {final['y_test_r2'] - initial['y_test_r2']:+.3f}")
    print(f"  Distance Error: {final['mean_dist_error'] - initial['mean_dist_error']:+.2f}")
    print(f"\nOutputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()