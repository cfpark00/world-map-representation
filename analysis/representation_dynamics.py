#!/usr/bin/env python3
"""
Analyze how internal representations evolve during training across all checkpoints.
Tracks R² scores for longitude/latitude prediction from partial prompts.
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

from src.utils import load_cities_csv, haversine


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


# Country to region mapping for visualization
country_to_region = {
    # North America
    'US': 'North America', 'CA': 'North America', 'MX': 'North America', 'GT': 'North America',
    'BZ': 'North America', 'SV': 'North America', 'HN': 'North America', 'NI': 'North America',
    'CR': 'North America', 'PA': 'North America', 'CU': 'North America', 'HT': 'North America',
    'DO': 'North America', 'JM': 'North America', 'TT': 'North America', 'BB': 'North America',
    'BS': 'North America', 'AG': 'North America', 'DM': 'North America', 'GD': 'North America',
    'KN': 'North America', 'LC': 'North America', 'VC': 'North America', 'GL': 'North America',
    'BM': 'North America', 'KY': 'North America', 'TC': 'North America', 'VG': 'North America',
    'VI': 'North America', 'PR': 'North America', 'AW': 'North America', 'CW': 'North America',
    'SX': 'North America', 'BQ': 'North America', 'AI': 'North America', 'MQ': 'North America',
    'GP': 'North America', 'MS': 'North America', 'BL': 'North America', 'MF': 'North America',
    'PM': 'North America',
    
    # South America
    'BR': 'South America', 'AR': 'South America', 'CL': 'South America', 'PE': 'South America',
    'CO': 'South America', 'VE': 'South America', 'EC': 'South America', 'BO': 'South America',
    'PY': 'South America', 'UY': 'South America', 'GY': 'South America', 'SR': 'South America',
    'GF': 'South America', 'FK': 'South America',
    
    # Africa
    'ZA': 'Africa', 'EG': 'Africa', 'NG': 'Africa', 'ET': 'Africa', 'KE': 'Africa', 'UG': 'Africa',
    'DZ': 'Africa', 'SD': 'Africa', 'MA': 'Africa', 'AO': 'Africa', 'GH': 'Africa', 'MZ': 'Africa',
    'MG': 'Africa', 'CM': 'Africa', 'CI': 'Africa', 'NE': 'Africa', 'BF': 'Africa', 'ML': 'Africa',
    'MW': 'Africa', 'ZM': 'Africa', 'SN': 'Africa', 'SO': 'Africa', 'TD': 'Africa', 'ZW': 'Africa',
    'GN': 'Africa', 'RW': 'Africa', 'BJ': 'Africa', 'TN': 'Africa', 'BI': 'Africa', 'SS': 'Africa',
    'TG': 'Africa', 'SL': 'Africa', 'LY': 'Africa', 'LR': 'Africa', 'MR': 'Africa', 'CF': 'Africa',
    'ER': 'Africa', 'GM': 'Africa', 'GA': 'Africa', 'BW': 'Africa', 'NA': 'Africa', 'MU': 'Africa',
    'SZ': 'Africa', 'GQ': 'Africa', 'DJ': 'Africa', 'KM': 'Africa', 'CV': 'Africa', 'ST': 'Africa',
    'SC': 'Africa', 'LS': 'Africa', 'GW': 'Africa', 'TZ': 'Africa', 'CG': 'Africa', 'CD': 'Africa',
    'EH': 'Africa', 'RE': 'Africa',
    
    # Western Europe (including Central Europe)
    'GB': 'Western Europe', 'FR': 'Western Europe', 'DE': 'Western Europe', 'IT': 'Western Europe', 
    'ES': 'Western Europe', 'NL': 'Western Europe', 'BE': 'Western Europe', 'PT': 'Western Europe',
    'CH': 'Western Europe', 'AT': 'Western Europe', 'IE': 'Western Europe', 'DK': 'Western Europe',
    'SE': 'Western Europe', 'NO': 'Western Europe', 'FI': 'Western Europe', 'IS': 'Western Europe',
    'LU': 'Western Europe', 'MC': 'Western Europe', 'LI': 'Western Europe', 'SM': 'Western Europe',
    'VA': 'Western Europe', 'AD': 'Western Europe', 'MT': 'Western Europe', 'GI': 'Western Europe',
    'JE': 'Western Europe', 'GG': 'Western Europe', 'IM': 'Western Europe', 'FO': 'Western Europe',
    
    # Eastern Europe (including Russia, Balkans, and former Soviet states)
    'RU': 'Eastern Europe', 'UA': 'Eastern Europe', 'PL': 'Eastern Europe', 'RO': 'Eastern Europe',
    'CZ': 'Eastern Europe', 'HU': 'Eastern Europe', 'BY': 'Eastern Europe', 'BG': 'Eastern Europe',
    'SK': 'Eastern Europe', 'RS': 'Eastern Europe', 'HR': 'Eastern Europe', 'BA': 'Eastern Europe',
    'AL': 'Eastern Europe', 'LT': 'Eastern Europe', 'LV': 'Eastern Europe', 'EE': 'Eastern Europe',
    'MD': 'Eastern Europe', 'SI': 'Eastern Europe', 'MK': 'Eastern Europe', 'ME': 'Eastern Europe',
    'XK': 'Eastern Europe', 'GR': 'Eastern Europe',  # Greece could go either way
    
    # Middle East
    'TR': 'Middle East', 'IR': 'Middle East', 'SA': 'Middle East', 'YE': 'Middle East', 'IQ': 'Middle East',
    'SY': 'Middle East', 'AE': 'Middle East', 'IL': 'Middle East', 'JO': 'Middle East', 'LB': 'Middle East',
    'OM': 'Middle East', 'PS': 'Middle East', 'KW': 'Middle East', 'QA': 'Middle East', 'BH': 'Middle East',
    'CY': 'Middle East', 'GE': 'Middle East', 'AM': 'Middle East', 'AZ': 'Middle East',
    
    # India (Indian subcontinent)
    'IN': 'India', 'PK': 'India', 'BD': 'India', 'LK': 'India', 'NP': 'India', 'BT': 'India',
    'MV': 'India', 'AF': 'India',  # Afghanistan often grouped with South Asia
    
    # China (Greater China region)
    'CN': 'China', 'TW': 'China', 'HK': 'China', 'MO': 'China',
    
    # Korea
    'KR': 'Korea', 'KP': 'Korea',
    
    # Japan
    'JP': 'Japan',
    
    # Southeast Asia
    'ID': 'Southeast Asia', 'TH': 'Southeast Asia', 'MY': 'Southeast Asia', 'VN': 'Southeast Asia',
    'PH': 'Southeast Asia', 'SG': 'Southeast Asia', 'MM': 'Southeast Asia', 'KH': 'Southeast Asia',
    'LA': 'Southeast Asia', 'TL': 'Southeast Asia', 'BN': 'Southeast Asia',
    
    # Central Asia (the -stan countries + Mongolia)
    'KZ': 'Central Asia', 'UZ': 'Central Asia', 'TM': 'Central Asia', 'TJ': 'Central Asia',
    'KG': 'Central Asia', 'MN': 'Central Asia',
    
    # Oceania
    'AU': 'Oceania', 'NZ': 'Oceania', 'PG': 'Oceania', 'FJ': 'Oceania', 'SB': 'Oceania',
    'NC': 'Oceania', 'PF': 'Oceania', 'VU': 'Oceania', 'WS': 'Oceania', 'KI': 'Oceania',
    'TO': 'Oceania', 'FM': 'Oceania', 'PW': 'Oceania', 'MH': 'Oceania', 'TV': 'Oceania',
    'NR': 'Oceania', 'GU': 'Oceania', 'MP': 'Oceania', 'AS': 'Oceania', 'CK': 'Oceania',
    'NU': 'Oceania', 'TK': 'Oceania', 'NF': 'Oceania', 'PN': 'Oceania', 'WF': 'Oceania',
    
    # Antarctica
    'AQ': 'Antarctica'
}

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
                       lon_train, lon_test, lat_train, lat_test, n_train_cities, 
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
    X_train_coord = partial_reps_np[:n_train_cities]
    X_test_coord = partial_reps_np[n_train_cities:]
    
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
        dist = haversine(lon_test[i], lat_test[i], lon_test_pred[i], lat_test_pred[i])
        pred_distances_km.append(dist)
    
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
                          test_city_info, step, r2_lon, r2_lat, mean_error):
    """Create a single frame for the world map animation with regions colored"""
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Get regions for test cities
    test_regions = []
    for city in test_city_info:
        country = city['country']
        region = country_to_region.get(country, 'Unknown')
        test_regions.append(region)
    
    # Plot predicted test locations by region
    for region in region_colors.keys():
        # Get indices for this region
        region_mask = [r == region for r in test_regions]
        if sum(region_mask) > 0:
            region_lons_pred = lon_pred[region_mask]
            region_lats_pred = lat_pred[region_mask]
            region_lons_true = lon_true[region_mask]
            region_lats_true = lat_true[region_mask]
            
            # Plot true locations with smaller markers
            ax.scatter(region_lons_true, region_lats_true, 
                      s=15, alpha=0.3, c='gray',
                      edgecolors='none')
            
            # Plot predicted locations
            ax.scatter(region_lons_pred, region_lats_pred, 
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
    
    # Set limits and labels
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    
    # Add title with metrics
    ax.set_title(f'Step {step:,} | Lon R²: {r2_lon:.3f} | Lat R²: {r2_lat:.3f} | Mean Error: {mean_error:.0f} km', 
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
    parser = argparse.ArgumentParser(description='Analyze representation dynamics across checkpoints')
    parser.add_argument('experiment_dir', type=str, help='Path to experiment directory')
    parser.add_argument('cities_csv', type=str, help='Path to cities CSV file')
    parser.add_argument('--layers', type=str, default='3,4', 
                       help='Comma-separated layer indices to extract (e.g., "3,4")')
    
    args = parser.parse_args()
    
    # Parse layer indices
    layer_indices = [int(x.strip()) for x in args.layers.split(',')]
    
    # Setup paths
    experiment_dir = Path(args.experiment_dir)
    config_path = experiment_dir / 'config.yaml'
    checkpoints_dir = experiment_dir / 'checkpoints'
    
    if not config_path.exists():
        print(f"Error: Config file not found at {config_path}")
        sys.exit(1)
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    experiment_name = experiment_dir.name
    
    print("="*60)
    print("Representation Dynamics Analysis")
    print("="*60)
    print(f"Experiment: {experiment_name}")
    print(f"Task type: {config['task_type']}")
    print(f"Model layers: {config['model']['num_hidden_layers']}")
    print(f"Hidden size: {config['model']['hidden_size']}")
    print(f"Extracting from layers: {layer_indices}")
    
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
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load tokenizer
    tokenizer_path = project_root / config['tokenizer_path']
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.padding_side = 'left'  # For generation
    print(f"Tokenizer loaded from {tokenizer_path}")
    
    # Load cities data - EXACTLY LIKE NOTEBOOK
    cities_df = load_cities_csv(args.cities_csv)
    print(f"Loaded {len(cities_df)} cities")
    
    # EXACTLY LIKE THE NOTEBOOK - Select cities and create prompts
    np.random.seed(42)  # Different seed for variety
    n_cities_probe = 5000
    n_train_cities = 3000
    
    # Sample cities without replacement
    sampled_city_indices = np.random.choice(len(cities_df), size=n_cities_probe, replace=False)
    sampled_cities = cities_df.iloc[sampled_city_indices]
    
    # Create partial prompts ending at "_" after "c_"
    partial_prompts = []
    city_info = []
    
    for idx, city in sampled_cities.iterrows():
        # Create partial prompt: "dist(c_XXX,c_" (comma + c + underscore)
        # This fully disambiguates and starts the second city pattern
        prompt = f"<bos>dist(c_{city['row_id']},c_"
        partial_prompts.append(prompt)
        city_info.append({
            'row_id': city['row_id'],
            'name': city['asciiname'],
            'longitude': city['longitude'],
            'latitude': city['latitude'],
            'country': city['country_code'] if 'country_code' in city else 'UNK'
        })
    
    print(f"Created {len(partial_prompts)} partial prompts")
    print(f"Will use {n_train_cities} for training, {n_cities_probe - n_train_cities} for testing")
    
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
    
    # Extract longitude and latitude as targets
    longitudes = np.array([c['longitude'] for c in city_info])
    latitudes = np.array([c['latitude'] for c in city_info])
    
    # Split into train and test
    lon_train = longitudes[:n_train_cities]
    lon_test = longitudes[n_train_cities:]
    lat_train = latitudes[:n_train_cities]
    lat_test = latitudes[n_train_cities:]
    
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
                lon_train, lon_test, lat_train, lat_test,
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
                    'lon_pred': result['lon_test_pred'],
                    'lat_pred': result['lat_test_pred'],
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
    
    # Save results
    output_csv = experiment_dir / 'representation_dynamics.csv'
    results_df.to_csv(output_csv, index=False)
    print(f"\nResults saved to {output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("Summary of Results")
    print("="*60)
    print(results_df[['step', 'lon_test_r2', 'lat_test_r2', 'mean_dist_error_km']])
    
    # Create dynamics plot with Loss, R², and Distance Error
    print("\n" + "="*60)
    print("Generating Dynamics Plot")
    print("="*60)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Training Loss
    ax = axes[0]
    if 'loss' in results_df.columns and results_df['loss'].notna().any():
        ax.plot(results_df['step'], results_df['loss'], 'k-', linewidth=2)
        ax.set_xlabel('Training Step', fontsize=12)
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Training Loss', fontsize=14)
        ax.grid(True, alpha=0.3)
        # Add text with final loss
        final_loss = results_df['loss'].iloc[-1]
        if pd.notna(final_loss):
            ax.text(0.95, 0.95, f'Final: {final_loss:.3f}', 
                   transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    else:
        ax.text(0.5, 0.5, 'Loss data not available', 
               transform=ax.transAxes, ha='center', va='center')
        ax.set_title('Training Loss (Not Available)', fontsize=14)
    
    # Plot 2: R² Scores (Test only, cleaner)
    ax = axes[1]
    ax.plot(results_df['step'], results_df['lon_test_r2'], 'b-', label='Longitude', linewidth=2)
    ax.plot(results_df['step'], results_df['lat_test_r2'], 'r-', label='Latitude', linewidth=2)
    # Add average as a thicker line
    avg_r2 = (results_df['lon_test_r2'] + results_df['lat_test_r2']) / 2
    ax.plot(results_df['step'], avg_r2, 'purple', label='Average', linewidth=2.5, alpha=0.7)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('R² Score', fontsize=12)
    ax.set_title('Test R² Scores', fontsize=14)
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.2, 1.0])
    # Add horizontal line at R²=0 for reference
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    # Add text with final values
    ax.text(0.05, 0.95, f'Final R²:\nLon: {results_df["lon_test_r2"].iloc[-1]:.3f}\nLat: {results_df["lat_test_r2"].iloc[-1]:.3f}', 
           transform=ax.transAxes, ha='left', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Distance Error
    ax = axes[2]
    ax.plot(results_df['step'], results_df['mean_dist_error_km'], 'g-', linewidth=2)
    ax.set_xlabel('Training Step', fontsize=12)
    ax.set_ylabel('Distance Error (km)', fontsize=12)
    ax.set_title('Mean Location Error', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    # Add horizontal lines for reference distances
    reference_distances = [10000, 5000, 2000, 1000, 500]
    for dist in reference_distances:
        if dist >= results_df['mean_dist_error_km'].min() and dist <= results_df['mean_dist_error_km'].max():
            ax.axhline(y=dist, color='gray', linestyle=':', alpha=0.3)
            ax.text(results_df['step'].max(), dist, f' {dist}km', 
                   va='center', ha='left', fontsize=8, alpha=0.5)
    # Add text with final error
    ax.text(0.95, 0.95, f'Final: {results_df["mean_dist_error_km"].iloc[-1]:.0f} km', 
           transform=ax.transAxes, ha='right', va='top',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    layers_str = '_'.join(map(str, layer_indices))
    plt.suptitle(f'Representation Dynamics: {experiment_name} (Layers {layers_str})', fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Save plot
    plot_path = experiment_dir / f'representation_dynamics_layers{layers_str}.png'
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
                pred_data['lon_pred'], pred_data['lat_pred'],
                lon_test, lat_test, test_city_info,
                pred_data['step'], pred_data['lon_r2'], 
                pred_data['lat_r2'], pred_data['mean_error']
            )
            frames.append(fig)
        
        # Save as GIF
        gif_path = experiment_dir / f'world_map_evolution_layers{layers_str}.gif'
        
        # Save frames as individual images and then combine into GIF
        from PIL import Image
        images = []
        
        for i, fig in enumerate(frames):
            # Save figure to temporary file
            temp_path = experiment_dir / f'temp_frame_{i:03d}.png'
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
            temp_path = experiment_dir / f'temp_frame_{i:03d}.png'
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


if __name__ == "__main__":
    main()