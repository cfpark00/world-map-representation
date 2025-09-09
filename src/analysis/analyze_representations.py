#!/usr/bin/env python3
"""
Analyze how internal representations evolve during training across all checkpoints.
Tracks R² scores for longitude/latitude prediction from partial prompts.
Generates plots and animated GIF showing evolution of predictions on world map.

Usage:
    python analyze_representations.py <config_file.yaml> [--overwrite]
    
Example:
    python src/analysis/analyze_representations.py configs/analysis/dist_1M_analysis.yaml
    python src/analysis/analyze_representations.py configs/analysis/dist_1M_analysis.yaml --overwrite
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

from src.utils import euclidean_distance


def filter_cities_by_pattern(cities_df, pattern):
    """Filter cities DataFrame by regex pattern.
    
    Supports:
    - Direct regex on city names: "^Atlantis_"
    - Region-based filtering: "region:^(?!Africa).*"
    """
    if not pattern or pattern == '.*':
        return cities_df
    
    # Check if it's a region pattern
    if pattern.startswith('region:'):
        region_pattern = pattern[7:]  # Remove "region:" prefix
        mask = cities_df['region'].str.match(region_pattern, na=False)
    else:
        # Apply regex pattern to asciiname column
        mask = cities_df['asciiname'].str.match(pattern, na=False)
    
    return cities_df[mask]


class RepresentationExtractor:
    """Extract representations from specific transformer layers"""
    
    def __init__(self, model, layer_indices=None):
        """
        Initialize the extractor.
        
        Args:
            model: The transformer model
            layer_indices: Either a single int or a list of ints specifying which layers to extract.
                          If None, defaults to layer 4 (index 3).
        """
        self.model = model
        
        # Handle both single index and list of indices
        if layer_indices is None:
            self.layer_indices = [3]  # Default to layer 4 (0-indexed)
        elif isinstance(layer_indices, int):
            self.layer_indices = [layer_indices]
        else:
            self.layer_indices = list(layer_indices)
        
        # Sort indices to ensure consistent ordering
        self.layer_indices = sorted(self.layer_indices)
        
        # Storage for representations from each layer
        self.representations = {}
        self.hook_handles = []
        
    def create_hook_fn(self, layer_idx):
        """Create a hook function for a specific layer"""
        def hook_fn(module, input, output):
            # output is a tuple (hidden_states, ...)
            # We want the hidden states after the layer (residual stream)
            hidden_states = output[0]  # Shape: (batch_size, seq_len, hidden_size)
            # Ensure we keep the original shape
            if len(hidden_states.shape) == 2:
                # If it's 2D, we might need to unsqueeze
                # This shouldn't happen but let's be safe
                hidden_states = hidden_states.unsqueeze(0)
            self.representations[layer_idx] = hidden_states.detach().cpu()
        return hook_fn
        
    def register_hooks(self):
        """Register forward hooks on all specified layers"""
        for layer_idx in self.layer_indices:
            # Access the specific transformer layer
            layer = self.model.model.layers[layer_idx]
            hook_fn = self.create_hook_fn(layer_idx)
            handle = layer.register_forward_hook(hook_fn)
            self.hook_handles.append(handle)
        
    def remove_hooks(self):
        """Remove all hooks"""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles = []
            
    def extract(self, input_ids, attention_mask=None, concatenate=True):
        """
        Extract representations for given inputs.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            concatenate: If True and multiple layers, concatenate representations.
                        If False, return dict mapping layer_idx to representations.
        
        Returns:
            If single layer: tensor of shape (batch_size, seq_len, hidden_size)
            If multiple layers and concatenate=True: tensor of shape (batch_size, seq_len, hidden_size * n_layers)
            If multiple layers and concatenate=False: dict mapping layer_idx to tensors
        """
        self.representations = {}
        self.register_hooks()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
        
        # Get the captured representations
        reps = {idx: self.representations[idx].clone() for idx in self.layer_indices}
        
        self.remove_hooks()
        
        # Return based on configuration
        if len(self.layer_indices) == 1:
            # Single layer - return tensor directly
            return reps[self.layer_indices[0]]
        elif concatenate:
            # Multiple layers - concatenate along hidden dimension
            concatenated = torch.cat([reps[idx] for idx in self.layer_indices], dim=-1)
            return concatenated
        else:
            # Multiple layers - return dictionary
            return reps
    
    @property
    def layer_idx(self):
        """Backward compatibility - return first layer index"""
        return self.layer_indices[0]
    
    def __repr__(self):
        if len(self.layer_indices) == 1:
            return f"RepresentationExtractor(layer={self.layer_indices[0]})"
        else:
            return f"RepresentationExtractor(layers={self.layer_indices})"


# Load country to region mapping from JSON (single source of truth)
def load_country_to_region_mapping():
    """Load the country to region mapping from the JSON file."""
    mapping_file = project_root / 'data' / 'geographic_mappings' / 'country_to_region.json'
    
    if not mapping_file.exists():
        raise FileNotFoundError(f"Country to region mapping not found at {mapping_file}")
    
    with open(mapping_file, 'r') as f:
        return json.load(f)


# Load the mapping at module level (will be extended with additional labels if provided)
country_to_region = load_country_to_region_mapping()

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
    'Atlantis': '#FF1493',         # DeepPink - VERY visible hot pink
    'Unknown': '#9E9E9E',          # Gray
}


def analyze_checkpoint(checkpoint_path, step, partial_input_ids, partial_attention_mask, 
                       lon_train, lon_test, lat_train, lat_test, 
                       train_input_ids, train_attention_mask,
                       device, layer_indices, return_predictions=False):
    """Analyze a single checkpoint and return R² scores - EXACTLY LIKE THE NOTEBOOK"""
    
    # Load model
    model = Qwen2ForCausalLM.from_pretrained(checkpoint_path)
    model.eval()
    model = model.to(device)
    
    # Get representations using output_hidden_states instead of hooks
    with torch.no_grad():
        outputs = model(partial_input_ids, partial_attention_mask, output_hidden_states=True)
    
    # Extract and concatenate the specified layers
    layer_reps = []
    for idx in layer_indices:
        # hidden_states includes embedding layer at index 0, so layer N is at index N
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
    
    # Split into train and test
    # Use the actual number of training samples (may be less than n_train_cities if filtered)
    n_train = len(train_input_ids)
    X_train_coord = partial_reps_np[:n_train]
    X_test_coord = partial_reps_np[n_train:]
    
    # Train longitude probe
    lon_probe = Ridge(alpha=10.0)
    lon_probe.fit(X_train_coord, lon_train)
    lon_train_pred = lon_probe.predict(X_train_coord)
    lon_test_pred = lon_probe.predict(X_test_coord)
    
    # Train latitude probe
    lat_probe = Ridge(alpha=10.0)
    lat_probe.fit(X_train_coord, lat_train)
    lat_train_pred = lat_probe.predict(X_train_coord)
    lat_test_pred = lat_probe.predict(X_test_coord)
    
    # Calculate metrics
    lon_train_r2 = r2_score(lon_train, lon_train_pred)
    lon_test_r2 = r2_score(lon_test, lon_test_pred)
    lat_train_r2 = r2_score(lat_train, lat_train_pred)
    lat_test_r2 = r2_score(lat_test, lat_test_pred)
    
    lon_test_mae = mean_absolute_error(lon_test, lon_test_pred)
    lat_test_mae = mean_absolute_error(lat_test, lat_test_pred)
    
    # Calculate distance error
    pred_distances_km = []
    for i in range(len(lon_test_pred)):
        dist = euclidean_distance(lon_test[i], lat_test[i], lon_test_pred[i], lat_test_pred[i])
        pred_distances_km.append(dist)
    
    # Note: Since coordinates are scaled by 10, distances are also scaled
    # This is fine for relative comparisons but not absolute km values
    mean_dist_error = np.mean(pred_distances_km)
    median_dist_error = np.median(pred_distances_km)
    
    # Clean up model to free memory
    del model
    torch.cuda.empty_cache()
    
    result = {
        'step': step,
        'lon_train_r2': lon_train_r2,
        'lon_test_r2': lon_test_r2,
        'lat_train_r2': lat_train_r2,
        'lat_test_r2': lat_test_r2,
        'lon_test_mae': lon_test_mae,
        'lat_test_mae': lat_test_mae,
        'mean_dist_error_km': mean_dist_error,
        'median_dist_error_km': median_dist_error
    }
    
    if return_predictions:
        result['lon_test_pred'] = lon_test_pred
        result['lat_test_pred'] = lat_test_pred
        result['lon_train_pred'] = lon_train_pred
        result['lat_train_pred'] = lat_train_pred
    
    return result


def create_world_map_frame(lon_pred, lat_pred, lon_true, lat_true, 
                          test_city_info, lon_train_pred, lat_train_pred, lon_train, lat_train,
                          train_city_info, step, r2_lon, r2_lat, mean_error):
    """Create a single frame for the world map animation with regions colored"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Plot gray dots for TRAINING set only (true locations)
    ax.scatter(lon_train, lat_train, 
              s=15, alpha=0.3, c='gray',
              edgecolors='none', label='Training cities (true)')
    
    # Get regions for test cities
    test_regions = []
    unknown_cities = []
    for city in test_city_info:
        region = city['region']  # Use region directly from city data
        test_regions.append(region)
        if region == 'Unknown':
            unknown_cities.append(city)
    
    # Print warning if there are unknown cities
    if unknown_cities:
        print(f"\n[PLOT WARNING] Found {len(unknown_cities)} cities with Unknown region at step {step}:")
        for city in unknown_cities[:10]:  # Print first 10 unknown cities
            print(f"  - {city['name']}, {city['country']} (country_code: {city['country']})")
        if len(unknown_cities) > 10:
            print(f"  ... and {len(unknown_cities) - 10} more")
    
    # Plot predicted test locations by region
    for region in region_colors.keys():
        # Get indices for this region
        region_mask = [r == region for r in test_regions]
        if sum(region_mask) > 0:
            region_lons_pred = lon_pred[region_mask]
            region_lats_pred = lat_pred[region_mask]
            region_lons_true = lon_true[region_mask]
            region_lats_true = lat_true[region_mask]
            
            # Plot predicted locations - special handling for Atlantis
            if region == 'Atlantis':
                # Use stars with bigger size for Atlantis
                ax.scatter(region_lons_pred, region_lats_pred, 
                          s=100, alpha=0.8, c=region_colors[region],
                          marker='*',  # Star marker
                          label=f'{region} ({sum(region_mask)})', 
                          edgecolors='black', linewidth=0.5)
            else:
                # Normal dots for other regions
                ax.scatter(region_lons_pred, region_lats_pred, 
                          s=30, alpha=0.7, c=region_colors[region],
                          label=f'{region} ({sum(region_mask)})', 
                          edgecolors='black', linewidth=0.3)
    
    # Add grid and reference lines (coordinates scaled by 10)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)  # Equator
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)  # Prime Meridian
    
    # Calculate mean positions of predicted locations for each region
    region_label_positions = {}
    for region in region_colors.keys():
        # Get indices for this region
        region_mask = [r == region for r in test_regions]
        if sum(region_mask) > 0:
            region_lons_pred = lon_pred[region_mask]
            region_lats_pred = lat_pred[region_mask]
            # Calculate mean position of predictions for this region
            mean_lon = np.mean(region_lons_pred)
            mean_lat = np.mean(region_lats_pred)
            region_label_positions[region] = (mean_lon, mean_lat)
    
    # Add region labels at the mean predicted positions
    for region, (lon, lat) in region_label_positions.items():
        fontsize = 9 if 'Europe' in region else 10
        ax.text(lon, lat, region, fontsize=fontsize, fontweight='bold', 
               ha='center', va='center', alpha=0.6,
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))
    
    # Set limits and labels (coordinates are scaled by 10)
    ax.set_xlim(-1800, 1800)
    ax.set_ylim(-900, 900)
    ax.set_xlabel('X (Longitude × 10)', fontsize=12)
    ax.set_ylabel('Y (Latitude × 10)', fontsize=12)
    
    # Add title with metrics
    ax.set_title(f'Step {step:,} | Lon R²: {r2_lon:.3f} | Lat R²: {r2_lat:.3f} | Mean Error: {mean_error:.0f} km', 
                fontsize=16, pad=20)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, fontsize=8, 
             bbox_to_anchor=(0.02, 0.98), framealpha=0.9)
    
    # Add tick marks (scaled by 10)
    ax.set_xticks(range(-1800, 1801, 600))
    ax.set_yticks(range(-900, 901, 300))
    
    plt.tight_layout()
    
    return fig


def main():
    parser = argparse.ArgumentParser(description='Analyze representation dynamics across checkpoints')
    parser.add_argument('config_path', type=str,
                       help='Path to analysis configuration YAML file')
    parser.add_argument('--overwrite', action='store_true',
                       help='Overwrite existing analysis subdirectory (does not affect experiment itself)')
    
    args = parser.parse_args()
    
    # Load config from YAML file
    config_path = Path(args.config_path)
    if not config_path.exists():
        print(f"Error: Config file {config_path} does not exist!")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract configuration with defaults
    exp_dir = Path(config['exp_dir'])
    cities_csv = config['cities_csv']
    analysis_name = config.get('analysis_name', None)  # Explicit analysis directory name
    layers = config.get('layers', [3, 4])
    n_probe_cities = config.get('n_probe_cities', 5000)
    n_train_cities = config.get('n_train_cities', 3000)
    device_str = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    seed = config.get('seed', 42)
    task_type = config.get('task_type', None)
    prompt_format = config.get('prompt_format', '')
    test_mode = config.get('test', False)
    
    # New regex-based filtering
    probe_train_pattern = config.get('probe_train', '.*')  # Default: all cities
    probe_test_pattern = config.get('probe_test', '.*')    # Default: all cities
    highlight_pattern = config.get('highlight', None)     # Default: no highlighting
    highlight_label = config.get('highlight_label', 'Highlighted')
    highlight_color = config.get('highlight_color', '#FF1493')
    
    # For backward compatibility with command line style usage
    class Args:
        pass
    args = Args()
    args.exp_dir = str(exp_dir)
    args.cities_csv = cities_csv
    args.layers = layers
    args.n_probe_cities = n_probe_cities
    args.n_train_cities = n_train_cities
    args.device = device_str
    args.seed = seed
    args.task_type = task_type
    args.prompt_format = prompt_format
    args.test = test_mode
    
    # Parse layer indices
    layer_indices = args.layers
    overwrite = parser.parse_args().overwrite
    
    # Setup paths
    experiment_dir = Path(args.exp_dir)
    config_path = experiment_dir / 'config.yaml'
    checkpoints_dir = experiment_dir / 'checkpoints'
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_name = experiment_dir.name
    
    # Determine task type
    if args.task_type is None:
        # Infer from config
        task_type = config.get('task_type', 'distance')
    else:
        task_type = args.task_type
    
    # Determine prompt format based on task type if not specified
    if args.prompt_format == '':
        # Use task-appropriate default
        if task_type in ['distance', 'dist']:
            prompt_format = 'dist'
        elif task_type in ['randomwalk', 'rw', 'random_walk']:
            prompt_format = 'rw200'
        else:
            prompt_format = 'dist'  # fallback default
    else:
        prompt_format = args.prompt_format
    
    print("="*60)
    print("Representation Dynamics Analysis")
    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Config task type: {config.get('task_type', 'not specified')}")
    print(f"Analysis task type: {task_type}")
    print(f"Model layers: {config['model']['num_hidden_layers']}")
    print(f"Hidden size: {config['model']['hidden_size']}")
    print(f"Extracting from layers: {layer_indices}")
    print(f"Prompt format: {prompt_format}")
    
    # Test mode: print arguments and exit
    if args.test:
        print("\n" + "="*60)
        print("TEST MODE - Arguments")
        print("="*60)
        print(f"Raw args.prompt_format: '{args.prompt_format}'")
        print(f"Resolved prompt_format: '{prompt_format}'")
        print(f"Task type: '{task_type}'")
        print(f"Layers: {layer_indices}")
        print(f"Seed: {args.seed}")
        
        # Create a few test prompts to show what will be used
        print("\n" + "="*60)
        print("TEST MODE - Sample Prompts")
        print("="*60)
        test_city_ids = [100, 500, 1000]
        for city_id in test_city_ids:
            if prompt_format == 'dist':
                test_prompt = f"<bos>dist(c_{city_id},c_"
            elif prompt_format == 'rw200':
                test_prompt = f"<bos>walk_200=c_{city_id},c_"
            else:
                test_prompt = f"Unknown format: {prompt_format}"
            print(f"  City {city_id}: {test_prompt}")
        
        print("\n" + "="*60)
        print("TEST MODE COMPLETE - Exiting")
        print("="*60)
        return
    
    # Get all checkpoint directories
    checkpoint_dirs = []
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
    
    # Setup device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = Path(config['tokenizer_path'])
    if not tokenizer_path.is_absolute():
        # Assume relative to project root
        tokenizer_path = experiment_dir.parent.parent.parent / tokenizer_path
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.padding_side = 'left'  # For generation
    print(f"Tokenizer loaded from {tokenizer_path}")
    
    # Load cities data - EXACTLY LIKE NOTEBOOK
    cities_df = pd.read_csv(args.cities_csv)
    print(f"Loaded {len(cities_df)} cities")
    
    # Apply regex filtering to get train and test city pools
    train_cities_df = filter_cities_by_pattern(cities_df, probe_train_pattern)
    test_cities_df = filter_cities_by_pattern(cities_df, probe_test_pattern)
    
    print(f"Probe train cities (pattern '{probe_train_pattern}'): {len(train_cities_df)}")
    print(f"Probe test cities (pattern '{probe_test_pattern}'): {len(test_cities_df)}")
    
    # Sample cities for probing based on regex filtering
    np.random.seed(args.seed)
    n_train_cities = min(args.n_train_cities, len(train_cities_df))
    n_test_cities = min(args.n_probe_cities - n_train_cities, len(test_cities_df))
    
    print(f"Sampling {n_train_cities} training cities, {n_test_cities} test cities")
    
    # Sample from filtered pools
    train_sample_indices = np.random.choice(len(train_cities_df), size=n_train_cities, replace=False)
    test_sample_indices = np.random.choice(len(test_cities_df), size=n_test_cities, replace=False)
    
    train_sample = train_cities_df.iloc[train_sample_indices]
    test_sample = test_cities_df.iloc[test_sample_indices]
    
    # Create prompts for train + test cities
    all_cities = pd.concat([train_sample, test_sample], ignore_index=True)
    
    partial_prompts = []
    city_info = []
    unknown_count = 0
    unknown_examples = []
    
    for idx, city in all_cities.iterrows():
        # Create partial prompt based on format
        if prompt_format == 'dist':
            # "dist(c_XXX,c_" (comma + c + underscore)
            prompt = f"<bos>dist(c_{city['city_id']},c_"
        elif prompt_format == 'rw200':
            # "walk_200=c_XXX,c_" for random walk format
            prompt = f"<bos>walk_200=c_{city['city_id']},c_"
        else:
            raise ValueError(f"Unknown prompt format: {prompt_format}")
        partial_prompts.append(prompt)
        
        # Use original country code for world cities
        country_code = city.get('country_code', 'UNK')
        
        # Track unknown regions
        region = city['region']
        if region == 'Unknown':
            unknown_count += 1
            if len(unknown_examples) < 5:
                unknown_examples.append(f"{city['asciiname']} ({country_code})")
        
        city_info.append({
            'row_id': city['city_id'],
            'name': city['asciiname'],
            'longitude': city['x'],  # Note: x,y are scaled by 10
            'latitude': city['y'],    # Note: x,y are scaled by 10
            'country': country_code,
            'region': region  # Use region directly from CSV
        })
    
    print(f"Created {len(partial_prompts)} partial prompts ({n_train_cities} train + {n_test_cities} test)")
    
    # Report unknown regions
    if unknown_count > 0:
        print(f"\n[DATA WARNING] Found {unknown_count} cities with Unknown region:")
        for example in unknown_examples:
            print(f"  - {example}")
        if unknown_count > len(unknown_examples):
            print(f"  ... and {unknown_count - len(unknown_examples)} more")
    
    # Tokenize ALL prompts with LEFT padding
    tokenized_all = tokenizer(
        partial_prompts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=False  # Dataset already has special tokens
    )
    
    partial_input_ids = tokenized_all['input_ids'].to(device)
    partial_attention_mask = tokenized_all['attention_mask'].to(device)
    
    print(f"Tokenized shape: {partial_input_ids.shape}")
    
    # Extract longitude and latitude as targets
    longitudes = np.array([c['longitude'] for c in city_info])
    latitudes = np.array([c['latitude'] for c in city_info])
    
    # Split into train and test
    # Training: first n_train_cities, Test: everything after n_train_cities
    lon_train = longitudes[:n_train_cities]
    lon_test = longitudes[n_train_cities:]
    lat_train = latitudes[:n_train_cities]
    lat_test = latitudes[n_train_cities:]
    
    # Get train and test city info for visualization
    train_city_info = city_info[:n_train_cities]
    test_city_info = city_info[n_train_cities:]
    
    # For analyze_checkpoint, we need the training input IDs and masks
    train_input_ids = partial_input_ids[:n_train_cities]
    train_attention_mask = partial_attention_mask[:n_train_cities]
    
    # Analyze all checkpoints
    print("\n" + "="*60)
    print("Analyzing Checkpoints")
    print("="*60)
    
    results = []
    predictions_for_animation = []
    
    for step, checkpoint_path in tqdm(checkpoint_dirs, desc="Processing"):
        print(f"\nStep {step}:")
        
        try:
            # Always save predictions for animation
            return_preds = True
            
            result = analyze_checkpoint(
                checkpoint_path, step,
                partial_input_ids, partial_attention_mask,
                lon_train, lon_test, lat_train, lat_test,
                train_input_ids, train_attention_mask,
                device, layer_indices,
                return_predictions=return_preds
            )
            
            # Try to get loss and eval metrics from trainer_state.json
            trainer_state_path = checkpoint_path / 'trainer_state.json'
            if trainer_state_path.exists():
                with open(trainer_state_path, 'r') as f:
                    trainer_state = json.load(f)
                    log_history = trainer_state.get('log_history', [])
                    
                    # Get the loss closest to this checkpoint step
                    losses = [(h['loss'], h['step']) for h in log_history if 'loss' in h]
                    if losses:
                        # Find loss closest to but not after the checkpoint step
                        valid_losses = [l for l in losses if l[1] <= step]
                        if valid_losses:
                            result['loss'] = valid_losses[-1][0]
                        else:
                            result['loss'] = losses[0][0]  # Use first if no valid ones
                    else:
                        result['loss'] = None
                    
                    # Get eval metrics for both distance and randomwalk tasks
                    if task_type in ['distance', 'dist', 'randomwalk', 'rw']:
                        # Look for eval_metric_mean (distance error for distance, validity ratio for randomwalk)
                        eval_metrics = [(h.get('eval_metric_mean'), h.get('eval_metric_median'), h['step']) 
                                      for h in log_history if 'eval_metric_mean' in h]
                        if eval_metrics:
                            # Find metrics closest to but not after the checkpoint step
                            valid_metrics = [m for m in eval_metrics if m[2] <= step]
                            if valid_metrics:
                                result['eval_metric_mean'] = valid_metrics[-1][0]
                                result['eval_metric_median'] = valid_metrics[-1][1]
                            else:
                                result['eval_metric_mean'] = eval_metrics[0][0]
                                result['eval_metric_median'] = eval_metrics[0][1]
                        else:
                            result['eval_metric_mean'] = None
                            result['eval_metric_median'] = None
            else:
                result['loss'] = None
                if task_type in ['distance', 'dist', 'randomwalk', 'rw']:
                    result['eval_metric_mean'] = None
                    result['eval_metric_median'] = None
                
            results.append(result)
            
            if return_preds:
                predictions_for_animation.append({
                    'step': step,
                    'lon_pred': result['lon_test_pred'],
                    'lat_pred': result['lat_test_pred'],
                    'lon_train_pred': result['lon_train_pred'],
                    'lat_train_pred': result['lat_train_pred'],
                    'lon_r2': result['lon_test_r2'],
                    'lat_r2': result['lat_test_r2'],
                    'mean_error': result['mean_dist_error_km']
                })
            
            print(f"  Lon R²: {result['lon_test_r2']:.3f}, Lat R²: {result['lat_test_r2']:.3f}")
            print(f"  Mean dist error: {result['mean_dist_error_km']:.0f} km")
            
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
    
    # Create analysis output directory with subfolder for this specific configuration
    if not analysis_name:
        print("Error: 'analysis_name' is required in the config file")
        print("Please specify the analysis output directory name")
        sys.exit(1)
    
    # Use explicit analysis name from config
    analysis_subdir = analysis_name
    analysis_dir = experiment_dir / 'analysis' / analysis_subdir
    
    # Handle overwrite flag - only affects the analysis subdirectory
    if analysis_dir.exists():
        if overwrite:
            import shutil
            print(f"Warning: Overwriting existing analysis directory: {analysis_dir}")
            shutil.rmtree(analysis_dir)
        else:
            print(f"Error: Analysis directory already exists: {analysis_dir}")
            print(f"Use --overwrite to overwrite it (this will NOT affect the experiment itself)")
            sys.exit(1)
    
    analysis_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nAnalysis directory: {analysis_dir}")
    
    # Save results
    output_csv = analysis_dir / 'representation_dynamics.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    print(results_df[['step', 'lon_test_r2', 'lat_test_r2', 'mean_dist_error_km']])
    
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
    
    # Create twin axis for task-specific error metric
    ax1_twin = ax1.twinx()
    color = 'tab:red'
    
    if task_type in ['distance', 'dist'] and 'eval_metric_mean' in results_df.columns and results_df['eval_metric_mean'].notna().any():
        # For distance tasks, show actual distance prediction error
        ax1_twin.plot(results_df['step'], results_df['eval_metric_mean'], color=color, linewidth=2, label='Distance Prediction Error')
        ax1_twin.set_ylabel('Mean Distance Prediction Error (km)', color=color, fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1_twin.set_yscale('log')
        
        # Add reference lines for distance
        reference_distances = [10000, 5000, 2000, 1000, 500, 100, 50]
        for dist in reference_distances:
            if results_df['eval_metric_mean'].notna().any():
                min_val = results_df['eval_metric_mean'].min()
                max_val = results_df['eval_metric_mean'].max()
                if dist >= min_val and dist <= max_val:
                    ax1_twin.axhline(y=dist, color='gray', linestyle=':', alpha=0.2)
                    ax1_twin.text(results_df['step'].max() * 1.01, dist, f'{dist}km', 
                                 va='center', ha='left', fontsize=8, alpha=0.5)
        
        # Add text with final error
        final_metric = results_df['eval_metric_mean'].iloc[-1]
        if pd.notna(final_metric):
            ax1_twin.text(0.98, 0.95, f'Final Distance Error: {final_metric:.0f} km', 
                         transform=ax1_twin.transAxes, ha='right', va='top', color=color,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('Training Loss & Distance Prediction Error', fontsize=14)
    elif task_type in ['randomwalk', 'rw'] and 'eval_metric_mean' in results_df.columns and results_df['eval_metric_mean'].notna().any():
        # For random walk tasks, show validity ratio (0-1)
        ax1_twin.plot(results_df['step'], results_df['eval_metric_mean'], color=color, linewidth=2, label='Walk Validity Ratio')
        ax1_twin.set_ylabel('Mean Validity Ratio', color=color, fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor=color)
        # No log scale for ratio (0-1)
        
        # Set y-limits for ratio
        ax1_twin.set_ylim([0, 1.05])
        
        # Add reference lines for validity ratio
        reference_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]
        for ratio in reference_ratios:
            ax1_twin.axhline(y=ratio, color='gray', linestyle=':', alpha=0.2)
            ax1_twin.text(results_df['step'].max() * 1.01, ratio, f'{ratio:.0%}', 
                         va='center', ha='left', fontsize=8, alpha=0.5)
        
        # Add text with final validity ratio
        final_metric = results_df['eval_metric_mean'].iloc[-1]
        if pd.notna(final_metric):
            ax1_twin.text(0.98, 0.95, f'Final Validity: {final_metric:.1%}', 
                         transform=ax1_twin.transAxes, ha='right', va='top', color=color,
                         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('Training Loss & Walk Validity Ratio', fontsize=14)
    else:
        # For non-distance tasks or when eval_metric not available, show location reconstruction error
        ax1_twin.plot(results_df['step'], results_df['mean_dist_error_km'], color=color, linewidth=2, label='Location Error')
        ax1_twin.set_ylabel('Mean Location Error (km)', color=color, fontsize=12)
        ax1_twin.tick_params(axis='y', labelcolor=color)
        ax1_twin.set_yscale('log')
        
        # Add reference lines for distance
        reference_distances = [10000, 5000, 2000, 1000, 500]
        for dist in reference_distances:
            if dist >= results_df['mean_dist_error_km'].min() and dist <= results_df['mean_dist_error_km'].max():
                ax1_twin.axhline(y=dist, color='gray', linestyle=':', alpha=0.2)
                ax1_twin.text(results_df['step'].max() * 1.01, dist, f'{dist}km', 
                             va='center', ha='left', fontsize=8, alpha=0.5)
        
        # Add text with final error
        ax1_twin.text(0.98, 0.95, f'Final Location Error: {results_df["mean_dist_error_km"].iloc[-1]:.0f} km', 
                     transform=ax1_twin.transAxes, ha='right', va='top', color=color,
                     bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_title('Training Loss & Location Reconstruction Error', fontsize=14)
    ax1.grid(True, alpha=0.3)
    
    # Bottom subplot: R² Scores and Location Error
    ax2 = axes[1]
    
    # Plot R² scores on left y-axis
    ax2.plot(results_df['step'], results_df['lon_test_r2'], 'b-', label='Longitude R²', linewidth=2)
    ax2.plot(results_df['step'], results_df['lat_test_r2'], 'r-', label='Latitude R²', linewidth=2)
    # Add average as a thicker line
    avg_r2 = (results_df['lon_test_r2'] + results_df['lat_test_r2']) / 2
    ax2.plot(results_df['step'], avg_r2, 'purple', label='Average R²', linewidth=2.5, alpha=0.7)
    
    ax2.set_xlabel('Training Step', fontsize=12)
    ax2.set_ylabel('Test R² Score', fontsize=12)
    ax2.set_title('Coordinate Prediction Performance', fontsize=14)
    ax2.grid(True, alpha=0.3)
    
    # Auto-adjust y-limits based on data with some padding
    min_r2 = min(results_df['lon_test_r2'].min(), results_df['lat_test_r2'].min())
    max_r2 = max(results_df['lon_test_r2'].max(), results_df['lat_test_r2'].max())
    r2_range = max_r2 - min_r2
    padding = r2_range * 0.1 if r2_range > 0 else 0.1
    ax2.set_ylim([min_r2 - padding, max(1.0, max_r2 + padding)])
    
    # Add horizontal line at R²=0 for reference
    ax2.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    
    # Create twin axis for euclidean distance
    ax2_twin = ax2.twinx()
    color = 'tab:green'
    ax2_twin.plot(results_df['step'], results_df['mean_dist_error_km'], '--', 
                  color=color, linewidth=2, label='Location Error (km)', alpha=0.7)
    ax2_twin.set_ylabel('Mean Location Error (km)', color=color, fontsize=12)
    ax2_twin.tick_params(axis='y', labelcolor=color)
    ax2_twin.set_yscale('log')
    
    # Set reasonable y-limits for distance
    max_dist = results_df['mean_dist_error_km'].max()
    min_dist = results_df['mean_dist_error_km'].min()
    ax2_twin.set_ylim([min_dist * 0.8, max_dist * 1.2])
    
    # Combine legends from both axes
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right', fontsize=10)
    
    # Add text with final values
    ax2.text(0.02, 0.95, f'Final R²:\nLon: {results_df["lon_test_r2"].iloc[-1]:.3f}\nLat: {results_df["lat_test_r2"].iloc[-1]:.3f}\nError: {results_df["mean_dist_error_km"].iloc[-1]:.0f} km', 
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
        
        # Count total unknowns across all test cities
        total_unknowns = sum(1 for city in test_city_info if city['region'] == 'Unknown')
        if total_unknowns > 0:
            print(f"\n[ANIMATION WARNING] {total_unknowns} test cities will be plotted with 'Unknown' region")
            print("These will appear in gray color on the map")
        
        # Create frames
        frames = []
        for pred_data in tqdm(predictions_for_animation, desc="Creating frames"):
            fig = create_world_map_frame(
                pred_data['lon_pred'], pred_data['lat_pred'],
                lon_test, lat_test, test_city_info,
                pred_data['lon_train_pred'], pred_data['lat_train_pred'],
                lon_train, lat_train, train_city_info,
                pred_data['step'], pred_data['lon_r2'], 
                pred_data['lat_r2'], pred_data['mean_error']
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
        
        # Save as GIF with configurable pause on final frame
        frame_duration = 500  # 500ms per frame
        final_frame_pause = 2500  # Additional 2.5s pause on last frame
        
        # Create duration list for each frame
        durations = [frame_duration] * len(images)
        if len(durations) > 0:
            durations[-1] = frame_duration + final_frame_pause  # Last frame gets extra pause
        
        images[0].save(
            gif_path,
            save_all=True,
            append_images=images[1:],
            duration=durations,
            loop=0
        )
        
        # Save the final frame as a standalone PNG
        final_map_path = analysis_dir / 'final_world_map.png'
        if len(images) > 0:
            images[-1].save(final_map_path)
            print(f"Final world map saved to {final_map_path}")
        
        # Clean up temporary files
        for i in range(len(frames)):
            temp_path = analysis_dir / f'temp_frame_{i:03d}.png'
            if temp_path.exists():
                temp_path.unlink()
        
        print(f"World map animation saved to {gif_path}")
    
    # Print final statistics
    print("\n" + "="*60)
    print("Final Statistics")
    print("="*60)
    
    initial = results_df.iloc[0]
    final = results_df.iloc[-1]
    
    print(f"\nInitial (Step {initial['step']}):")
    print(f"  Longitude R²: {initial['lon_test_r2']:.3f}")
    print(f"  Latitude R²:  {initial['lat_test_r2']:.3f}")
    print(f"  Distance Error: {initial['mean_dist_error_km']:.0f} km")
    
    print(f"\nFinal (Step {final['step']}):")
    print(f"  Longitude R²: {final['lon_test_r2']:.3f}")
    print(f"  Latitude R²:  {final['lat_test_r2']:.3f}")
    print(f"  Distance Error: {final['mean_dist_error_km']:.0f} km")
    
    print(f"\nImprovement:")
    print(f"  Longitude R²: +{final['lon_test_r2'] - initial['lon_test_r2']:.3f}")
    print(f"  Latitude R²:  +{final['lat_test_r2'] - initial['lat_test_r2']:.3f}")
    print(f"  Distance Error: -{initial['mean_dist_error_km'] - final['mean_dist_error_km']:.0f} km")
    print(f"\nOutputs saved to: {analysis_dir}")


if __name__ == "__main__":
    main()