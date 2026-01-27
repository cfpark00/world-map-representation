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
project_root = Path('')
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
    'Japan': '#FFD700',            # Gold/Yellow
    'Southeast Asia': '#43A047',   # Light Green
    'Central Asia': '#FFB300',     # Amber
    'Oceania': '#00BCD4',          # Light Cyan
    'Unknown': '#9E9E9E',          # Gray
}


def get_prompt_config(prompt_format, city):
    """Returns prompt string and extraction indices for a given format.

    Args:
        prompt_format: The format type for the prompt
        city: Dictionary containing city information (row_id, etc.)

    Returns:
        Dictionary with:
            - prompt: The formatted prompt string
            - extraction_indices: List of token positions to extract
            - position_names: Names for each extracted position
    """

    if prompt_format == 'dist':
        # Format: "<bos> d i s t ( c _ X X X X , c _"
        dist_str = f"dist(c_{city['row_id']},c_"
        spaced_str = ' '.join(dist_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [-3, -2, -1]  # comma, c, underscore
        position_names = ['comma', 'c', 'underscore']

    elif prompt_format == 'dist_city_and_transition':
        # Format: "<bos> d i s t ( c _ X X X X , c _"
        # Extract full first city and transition
        dist_str = f"dist(c_{city['row_id']},c_"
        spaced_str = ' '.join(dist_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [6, 7, 8, 9, 10, 11, 12, 13, 14]  # c _ i1 i2 i3 i4 , c _
        position_names = ['c1', '_1', 'i1', 'i2', 'i3', 'i4', ',', 'c2', '_2']

    elif prompt_format == 'dist_firstcity_last_and_comma':
        # Format: "<bos> d i s t ( c _ X X X X ,"
        # Extract last digit and comma - no need for "c_" after since causal attention can't see it
        dist_str = f"dist(c_{city['row_id']},"
        spaced_str = ' '.join(dist_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [11, 12]  # i4 (last digit), comma
        position_names = ['last_digit', 'comma']

    elif prompt_format == 'trianglearea_firstcity_last_and_comma':
        # Format: "<bos> t r i a r e a ( c _ X X X X ,"
        # Extract last digit and comma - no need for "c_" after since causal attention can't see it
        # Note: The actual dataset uses "triarea" not "trianglearea"
        triarea_str = f"triarea(c_{city['row_id']},"
        spaced_str = ' '.join(triarea_str)
        prompt = f"<bos> {spaced_str}"
        # "triarea" has 7 chars vs "dist" with 4, so positions shift by 3
        extraction_indices = [14, 15]  # i4 (last digit), comma
        position_names = ['last_digit', 'comma']

    elif prompt_format == 'trianglearea_firstcity_last':
        # Format: "<bos> t r i a r e a ( c _ X X X X"
        # Extract only last digit of first city (no comma)
        # Just the city ID without comma at the end
        # Note: The actual dataset uses "triarea" not "trianglearea"
        city_id_str = str(city['row_id'])
        triarea_str = f"triarea(c_{city_id_str}"  # No comma
        spaced_str = ' '.join(triarea_str)
        prompt = f"<bos> {spaced_str}"
        # "triarea" has 7 chars, ( is 1, c is 1, _ is 1, then city digits
        # Position 14 is the last digit of a 4-digit city ID
        extraction_indices = [14]  # Just i4 (last digit)
        position_names = ['last_digit']

    elif prompt_format == 'crossing_firstcity_last_and_comma':
        # Format: "<bos> c r o s s ( c _ X X X X ,"
        # Extract last digit and comma of first city
        cross_str = f"cross(c_{city['row_id']},"
        spaced_str = ' '.join(cross_str)
        prompt = f"<bos> {spaced_str}"
        # "cross" has 5 chars vs "dist" with 4, so positions shift by 1
        # Token positions: <bos> c r o s s ( c _ X X X X ,
        # Position 0: <bos>
        # Position 1-5: c r o s s
        # Position 6: (
        # Position 7: c
        # Position 8: _
        # Position 9-12: city ID digits (for 4-digit ID)
        # Position 13: ,
        extraction_indices = [12, 13]  # last digit, comma
        position_names = ['last_digit', 'comma']

    elif prompt_format == 'randomwalk_firstcity_last_and_comma':
        # Format: "<bos> r w ( max_dist , chain_len ) = c _ X X X X ,"
        # For random walk, we need to construct a valid prompt
        # Use average values from typical dataset config (50-500 for dist, 5-20 for length)
        max_dist = 275  # Average of typical range 50-500
        chain_len = 12   # Average of typical range 5-20
        rw_str = f"rw({max_dist},{chain_len})=c_{city['row_id']},"
        spaced_str = ' '.join(rw_str)
        prompt = f"<bos> {spaced_str}"

        # The tokenizer splits on spaces, so spaces are NOT tokens!
        # Token positions for: <bos> r w ( 2 7 5 , 1 2 ) = c _ X X X X ,
        # Position 0: <bos>
        # Position 1: r
        # Position 2: w
        # Position 3: (
        # Position 4-6: digits of max_dist (2 7 5)
        # Position 7: ,
        # Position 8-9: digits of chain_len (1 2)
        # Position 10: )
        # Position 11: =
        # Position 12: c
        # Position 13: _
        # Position 14-17: city ID digits (assuming 4-digit ID)
        # Position 18: ,

        # For a 4-digit city ID:
        extraction_indices = [17, 18]  # last digit at pos 17, comma at pos 18
        position_names = ['last_digit', 'comma']

    else:
        raise ValueError(f"Unknown prompt_format: {prompt_format}")

    return {
        'prompt': prompt,
        'extraction_indices': extraction_indices,
        'position_names': position_names
    }


def analyze_checkpoint(checkpoint_path, step, partial_input_ids, partial_attention_mask,
                       x_train, x_test, y_train, y_test, n_train_cities,
                       device, layer_indices, return_predictions=False, method_config=None,
                       return_probe_weights=False, return_representations=False,
                       extraction_indices=None, position_names=None):
    """Analyze a single checkpoint and return R² scores and optionally probe weights and representations

    Args:
        extraction_indices: List of token positions to extract representations from
        position_names: Names for each extracted position (for debugging)
    """
    
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
        # hidden_states includes embedding layer at index 0, so layer N is at index N
        layer_reps.append(outputs.hidden_states[idx])  # index 0 = embeddings, index N = layer N output
    
    # Concatenate layers if multiple
    if len(layer_reps) > 1:
        partial_representations = torch.cat(layer_reps, dim=-1)
    else:
        partial_representations = layer_reps[0]
    
    # Extract representations using provided indices
    n_cities = partial_representations.shape[0]
    n_layers = len(layer_indices)
    hidden_dim = partial_representations.shape[-1] // n_layers if n_layers > 1 else partial_representations.shape[-1]

    # Validate that extraction_indices and position_names are provided
    if extraction_indices is None or position_names is None:
        raise ValueError("extraction_indices and position_names must be provided")

    # Extract tokens at specified indices
    token_positions = []
    for idx in extraction_indices:
        token_positions.append(partial_representations[:, idx, :])

    # Stack to create (n_cities, n_tokens, concatenated_layer_dims)
    token_stack = torch.stack(token_positions, dim=1)

    # Reshape to (n_cities, n_tokens, n_layers, hidden_dim)
    n_tokens = len(extraction_indices)
    partial_reps_reshaped = token_stack.reshape(n_cities, n_tokens, n_layers, hidden_dim)

    # For probe training, concatenate all token positions
    partial_last_token_reps = torch.cat(token_positions, dim=1)
    partial_reps_np = partial_last_token_reps.cpu().numpy()

    print(f"Extracted representations: {partial_reps_reshaped.shape}")
    print(f"  Positions: {', '.join(position_names)}")
    
    # Print shape after extraction
    print(f"Extracted representations shape: {partial_reps_np.shape}")
    
    # Split into train and test
    X_train_coord = partial_reps_np[:n_train_cities]
    X_test_coord = partial_reps_np[n_train_cities:]

    # Calculate mean of training coordinates
    x_train_mean = x_train.mean()
    y_train_mean = y_train.mean()

    # Center the targets (predict deviations from mean)
    x_train_centered = x_train - x_train_mean
    x_test_centered = x_test - x_train_mean
    y_train_centered = y_train - y_train_mean
    y_test_centered = y_test - y_train_mean

    # Create probe based on method configuration
    if method_config is None:
        # Default to Ridge with alpha=10.0
        x_probe = Ridge(alpha=10.0)
        y_probe = Ridge(alpha=10.0)
    else:
        method_name = method_config.get('name', 'ridge')
        
        if method_name == 'linear':
            from sklearn.linear_model import LinearRegression
            x_probe = LinearRegression()
            y_probe = LinearRegression()
        elif method_name == 'lasso':
            from sklearn.linear_model import Lasso
            alpha = method_config.get('alpha', 1.0)
            max_iter = method_config.get('max_iter', 1000)
            tol = method_config.get('tol', 0.0001)
            x_probe = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
            y_probe = Lasso(alpha=alpha, max_iter=max_iter, tol=tol)
        else:  # ridge
            alpha = method_config.get('alpha', 10.0)
            solver = method_config.get('solver', 'auto')
            max_iter = method_config.get('max_iter', None)
            tol = method_config.get('tol', 0.0001)
            x_probe = Ridge(alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)
            y_probe = Ridge(alpha=alpha, solver=solver, max_iter=max_iter, tol=tol)
    
    # Train probes on centered targets
    x_probe.fit(X_train_coord, x_train_centered)
    x_train_pred_centered = x_probe.predict(X_train_coord)
    x_test_pred_centered = x_probe.predict(X_test_coord)

    y_probe.fit(X_train_coord, y_train_centered)
    y_train_pred_centered = y_probe.predict(X_train_coord)
    y_test_pred_centered = y_probe.predict(X_test_coord)

    # Add means back to get final predictions
    x_train_pred = x_train_pred_centered + x_train_mean
    x_test_pred = x_test_pred_centered + x_train_mean
    y_train_pred = y_train_pred_centered + y_train_mean
    y_test_pred = y_test_pred_centered + y_train_mean
    
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
    
    if return_probe_weights:
        # Get probe coefficients (weights) and intercepts
        # Note: These are for centered predictions (deviations from training mean)
        result['x_probe_coef'] = x_probe.coef_  # Shape: (n_features,)
        result['x_probe_intercept'] = x_probe.intercept_
        result['y_probe_coef'] = y_probe.coef_  # Shape: (n_features,)
        result['y_probe_intercept'] = y_probe.intercept_
        result['x_train_mean'] = x_train_mean  # Store the mean for reference
        result['y_train_mean'] = y_train_mean
    
    # Store representations separately to avoid CSV bloat
    representations_data = None
    if return_representations:
        representations_data = {
            'representations': partial_reps_reshaped.cpu().numpy(),  # Shape: (n_cities, n_tokens, n_layers, hidden_dim)
            'representations_flat': partial_reps_np,  # Keep flat version for backward compatibility
            'input_ids': partial_input_ids.cpu().numpy()  # For reference
        }
    
    return result, representations_data


def visualize_probe_weights(x_weights, y_weights, step, layer_indices, hidden_size=128):
    """Create visualization of probe weights for x and y coordinates.
    
    Args:
        x_weights: Probe weights for x coordinate prediction
        y_weights: Probe weights for y coordinate prediction
        step: Training step
        layer_indices: List of layer indices used
        hidden_size: Hidden size of each layer
    
    Returns:
        matplotlib figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    
    # Determine how the weights are structured based on layer indices
    n_layers = len(layer_indices)
    total_features = len(x_weights)
    features_per_token = total_features // 3  # We concatenate 3 tokens
    features_per_layer = features_per_token // n_layers  # Should be hidden_size
    
    # Reshape weights to (n_tokens=3, n_layers, hidden_size)
    x_weights_reshaped = x_weights.reshape(3, n_layers, features_per_layer)
    y_weights_reshaped = y_weights.reshape(3, n_layers, features_per_layer)
    
    # Token labels
    token_labels = ['comma (,)', 'c', 'underscore (_)']
    
    # Plot x coordinate weights
    ax = axes[0]
    im_x = ax.imshow(x_weights_reshaped.reshape(3 * n_layers, features_per_layer).T, 
                     aspect='auto', cmap='RdBu_r', vmin=-np.abs(x_weights).max(), 
                     vmax=np.abs(x_weights).max())
    ax.set_title(f'X Coordinate Probe Weights (Step {step})', fontsize=14)
    ax.set_xlabel('Token and Layer', fontsize=12)
    ax.set_ylabel('Hidden Dimension', fontsize=12)
    
    # Add vertical lines to separate tokens
    for i in range(1, 3):
        ax.axvline(x=i * n_layers - 0.5, color='black', linewidth=2, alpha=0.7)
    
    # Set x-axis labels
    xticks = []
    xticklabels = []
    for t_idx, token in enumerate(token_labels):
        for l_idx, layer in enumerate(layer_indices):
            xticks.append(t_idx * n_layers + l_idx)
            xticklabels.append(f'L{layer}')
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8)
    
    # Add token labels above
    for t_idx, token in enumerate(token_labels):
        ax.text(t_idx * n_layers + n_layers/2 - 0.5, -10, token, 
               ha='center', fontsize=10, fontweight='bold')
    
    plt.colorbar(im_x, ax=ax, label='Weight magnitude')
    
    # Plot y coordinate weights
    ax = axes[1]
    im_y = ax.imshow(y_weights_reshaped.reshape(3 * n_layers, features_per_layer).T, 
                     aspect='auto', cmap='RdBu_r', vmin=-np.abs(y_weights).max(), 
                     vmax=np.abs(y_weights).max())
    ax.set_title(f'Y Coordinate Probe Weights (Step {step})', fontsize=14)
    ax.set_xlabel('Token and Layer', fontsize=12)
    ax.set_ylabel('Hidden Dimension', fontsize=12)
    
    # Add vertical lines to separate tokens
    for i in range(1, 3):
        ax.axvline(x=i * n_layers - 0.5, color='black', linewidth=2, alpha=0.7)
    
    # Set x-axis labels
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels, rotation=45, ha='right', fontsize=8)
    
    # Add token labels above
    for t_idx, token in enumerate(token_labels):
        ax.text(t_idx * n_layers + n_layers/2 - 0.5, -10, token, 
               ha='center', fontsize=10, fontweight='bold')
    
    plt.colorbar(im_y, ax=ax, label='Weight magnitude')
    
    plt.suptitle(f'Linear Probe Weights - Layers {layer_indices}', fontsize=16)
    plt.tight_layout()
    
    return fig


def create_fit_quality_plot(x_train_true, y_train_true, x_train_pred, y_train_pred,
                           x_test_true, y_test_true, x_test_pred, y_test_pred,
                           step, experiment_name):
    """Create a scatter plot showing actual vs predicted coordinates for both train and test sets"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    # Calculate R² scores
    from sklearn.metrics import r2_score
    x_train_r2 = r2_score(x_train_true, x_train_pred)
    x_test_r2 = r2_score(x_test_true, x_test_pred)
    y_train_r2 = r2_score(y_train_true, y_train_pred)
    y_test_r2 = r2_score(y_test_true, y_test_pred)
    
    # X coordinate - Training set
    ax = axes[0, 0]
    ax.scatter(x_train_true, x_train_pred, alpha=0.3, s=10, color='blue')
    ax.plot([x_train_true.min(), x_train_true.max()], 
            [x_train_true.min(), x_train_true.max()], 
            'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('True X Coordinate', fontsize=12)
    ax.set_ylabel('Predicted X Coordinate', fontsize=12)
    ax.set_title(f'X Coordinate - Training Set (R²={x_train_r2:.3f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # X coordinate - Test set
    ax = axes[0, 1]
    ax.scatter(x_test_true, x_test_pred, alpha=0.3, s=10, color='green')
    ax.plot([x_test_true.min(), x_test_true.max()], 
            [x_test_true.min(), x_test_true.max()], 
            'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('True X Coordinate', fontsize=12)
    ax.set_ylabel('Predicted X Coordinate', fontsize=12)
    ax.set_title(f'X Coordinate - Test Set (R²={x_test_r2:.3f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Y coordinate - Training set
    ax = axes[1, 0]
    ax.scatter(y_train_true, y_train_pred, alpha=0.3, s=10, color='blue')
    ax.plot([y_train_true.min(), y_train_true.max()], 
            [y_train_true.min(), y_train_true.max()], 
            'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('True Y Coordinate', fontsize=12)
    ax.set_ylabel('Predicted Y Coordinate', fontsize=12)
    ax.set_title(f'Y Coordinate - Training Set (R²={y_train_r2:.3f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Y coordinate - Test set
    ax = axes[1, 1]
    ax.scatter(y_test_true, y_test_pred, alpha=0.3, s=10, color='green')
    ax.plot([y_test_true.min(), y_test_true.max()], 
            [y_test_true.min(), y_test_true.max()], 
            'r--', linewidth=2, label='Perfect fit')
    ax.set_xlabel('True Y Coordinate', fontsize=12)
    ax.set_ylabel('Predicted Y Coordinate', fontsize=12)
    ax.set_title(f'Y Coordinate - Test Set (R²={y_test_r2:.3f})', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.suptitle(f'Probe Fit Quality - {experiment_name} (Step {step})', fontsize=16, y=1.01)
    plt.tight_layout()
    
    return fig


def create_world_map_frame(x_pred, y_pred, x_true, y_true,
                          test_city_info, step, r2_x, r2_y, mean_error,
                          highlight_pattern=None, highlight_label=None, highlight_color=None,
                          plot_links=False, plot_autolim=False):
    """Create a single frame for the world map animation with regions colored and optional highlighting

    Args:
        plot_links: If True, draw lines connecting true to predicted positions
        plot_autolim: If True, automatically set axis limits based on data with 10% padding
    """
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Get regions for test cities
    test_regions = []
    highlight_mask = []
    for city in test_city_info:
        # Use region directly from city info if available
        if 'region' in city:
            region = city['region']
        else:
            country = city['country']
            # Fail fast if country not in mapping
            if country not in country_to_region:
                raise ValueError(f"Country '{country}' not found in region mapping. City: {city['name']}")
            region = country_to_region[country]
        test_regions.append(region)
        
        # Check if this city should be highlighted
        if highlight_pattern and 'region' in city:
            # Parse highlight pattern (e.g., "region:Atlantis")
            if ':' in highlight_pattern:
                field, value = highlight_pattern.split(':', 1)
                if field == 'region' and city.get('region') == value:
                    highlight_mask.append(True)
                else:
                    highlight_mask.append(False)
            else:
                highlight_mask.append(False)
        else:
            highlight_mask.append(False)
    
    # Get unique regions in test data
    unique_regions = list(set(test_regions))
    
    # Plot predicted test locations by region
    for region in unique_regions:
        # Skip if this region will be highlighted separately
        if highlight_pattern and ':' in highlight_pattern:
            field, value = highlight_pattern.split(':', 1)
            if field == 'region' and region == value:
                continue  # Skip this region, it will be highlighted
        
        # Get indices for this region
        region_mask = [r == region for r in test_regions]
        if sum(region_mask) > 0:
            # Get color for this region (use gray if not in predefined colors)
            region_color = region_colors.get(region, '#808080')  # Default gray
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
                      s=30, alpha=0.7, c=region_color,
                      label=f'{region} ({sum(region_mask)})', 
                      edgecolors='black', linewidth=0.3)
    
    # Plot highlighted cities with special markers (e.g., Atlantis with pink stars)
    if highlight_color and sum(highlight_mask) > 0:
        highlight_indices = np.array(highlight_mask)
        highlight_xs_pred = x_pred[highlight_indices]
        highlight_ys_pred = y_pred[highlight_indices]
        highlight_xs_true = x_true[highlight_indices]
        highlight_ys_true = y_true[highlight_indices]
        
        # Plot true locations with stars
        ax.scatter(highlight_xs_true / 10, highlight_ys_true / 10,
                  s=40, alpha=0.5, c='gray',
                  marker='*', edgecolors='none')
        
        # Plot predicted locations with pink stars
        ax.scatter(highlight_xs_pred / 10, highlight_ys_pred / 10,
                  s=100, alpha=0.9, c=highlight_color,
                  marker='*', label=f'{highlight_label} ({sum(highlight_mask)})',
                  edgecolors='black', linewidth=0.5)
    
    # Draw links between true and predicted positions if requested
    if plot_links:
        # Draw lines for all non-highlighted cities
        for i in range(len(x_pred)):
            if not (highlight_mask and highlight_mask[i]):
                # Draw a thin line from true to predicted position
                ax.plot([x_true[i]/10, x_pred[i]/10],
                       [y_true[i]/10, y_pred[i]/10],
                       'gray', alpha=0.2, linewidth=0.5, zorder=1)

        # Draw highlighted links with different style if applicable
        if highlight_color and sum(highlight_mask) > 0:
            highlight_indices = np.where(np.array(highlight_mask))[0]
            for i in highlight_indices:
                ax.plot([x_true[i]/10, x_pred[i]/10],
                       [y_true[i]/10, y_pred[i]/10],
                       color=highlight_color, alpha=0.4, linewidth=1.0,
                       linestyle='--', zorder=2)

    # Add grid and reference lines
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)  # Equator
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3, linewidth=0.5)  # Prime Meridian
    
    # Calculate mean positions of predicted locations for each region
    region_label_positions = {}
    for region in unique_regions:
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
    
    # Set limits based on plot_autolim setting
    if plot_autolim:
        # Calculate bounds from both true and predicted positions
        all_x = np.concatenate([x_true/10, x_pred/10])
        all_y = np.concatenate([y_true/10, y_pred/10])

        min_x, max_x = all_x.min(), all_x.max()
        min_y, max_y = all_y.min(), all_y.max()

        # Calculate data width and height
        data_width = max_x - min_x
        data_height = max_y - min_y

        # Use the larger dimension to maintain equal aspect ratio
        max_dimension = max(data_width, data_height)

        # Add 10% padding based on the larger dimension
        padding = max_dimension * 0.1

        # Calculate centers
        center_x = (min_x + max_x) / 2
        center_y = (min_y + max_y) / 2

        # Set limits using the same span for both axes (equal aspect ratio)
        half_span = (max_dimension + 2 * padding) / 2
        ax.set_xlim(center_x - half_span, center_x + half_span)
        ax.set_ylim(center_y - half_span, center_y + half_span)
    else:
        # Use default world map limits
        ax.set_xlim(-180, 180)
        ax.set_ylim(-90, 90)
    ax.set_xlabel('X coordinate', fontsize=12)
    ax.set_ylabel('Y coordinate', fontsize=12)

    # Ensure equal aspect ratio for geographic data
    ax.set_aspect('equal', adjustable='box')

    # Add title with metrics
    ax.set_title(f'Step {step:,} | X R²: {r2_x:.3f} | Y R²: {r2_y:.3f} | Mean Error: {mean_error:.2f}',
                fontsize=16, pad=20)

    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left', ncol=2, fontsize=8,
             bbox_to_anchor=(0.02, 0.98), framealpha=0.9)

    # Add tick marks
    if not plot_autolim:
        # Use standard world map ticks
        ax.set_xticks(range(-180, 181, 60))
        ax.set_yticks(range(-90, 91, 30))
    # When using autolim, matplotlib will automatically choose appropriate ticks
    
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
    n_train_cities = config['n_train_cities']
    n_test_cities = config['n_test_cities']
    device = torch.device(config['device'])
    seed = config['seed']
    # Required fields - no defaults
    prompt_format = config['prompt_format']
    probe_train_pattern = config['probe_train']
    probe_test_pattern = config['probe_test']
    save_fits = config.get('save_fits', False)  # Default to False if not specified
    save_repr_ckpts = config.get('save_repr_ckpts', [])  # Checkpoints to save representations
    method_config = config.get('method', None)  # Probe method configuration
    
    # Optional checkpoint parameter - can be "final", a number, or None for all
    checkpoint_param = config.get('checkpoint', None)
    
    # Optional fields with explicit None
    highlight_pattern = config.get('highlight', None)
    highlight_label = config.get('highlight_label', None)
    highlight_color = config.get('highlight_color', None)
    plot_links = config.get('plot_links', False)  # Default to False if not specified
    plot_autolim = config.get('plot_autolim', False)  # Default to False if not specified
    # Also check for plot_autobox as an alias for plot_autolim
    if 'plot_autobox' in config:
        plot_autolim = config.get('plot_autobox', False)
    
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
    print(f"Prompt format: {prompt_format}")
    if 'model' in training_config:
        print(f"Model layers: {training_config['model']['num_hidden_layers']}")
        print(f"Hidden size: {training_config['model']['hidden_size']}")
    print(f"Extracting from layers: {layer_indices}")
    if save_repr_ckpts:
        print(f"Will save representations for checkpoints: {save_repr_ckpts}")
    
    # Print probe method configuration
    if method_config:
        print(f"\nProbe method: {method_config.get('name', 'ridge')}")
        if method_config.get('name') in ['ridge', 'lasso']:
            print(f"  Alpha: {method_config.get('alpha', 10.0 if method_config.get('name') == 'ridge' else 1.0)}")
        if method_config.get('solver'):
            print(f"  Solver: {method_config.get('solver')}")
    else:
        print("\nProbe method: ridge (default)")
        print("  Alpha: 10.0")
    
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
    
    # Step 1: Get all train city candidates
    train_candidates = filter_dataframe_by_pattern(cities_df, probe_train_pattern, column_name='region')
    print(f"Cities matching train pattern '{probe_train_pattern}': {len(train_candidates)}")

    # Step 2: Sample train cities
    if len(train_candidates) < n_train_cities:
        raise ValueError(f"Not enough cities for training! Requested {n_train_cities}, but only {len(train_candidates)} available.")
    train_sample = train_candidates.sample(n=n_train_cities, random_state=seed)

    # Step 3: Get all test city candidates (excluding training cities)
    test_candidates = filter_dataframe_by_pattern(cities_df, probe_test_pattern, column_name='region')
    test_candidates = test_candidates[~test_candidates['city_id'].isin(train_sample['city_id'])]
    print(f"Cities matching test pattern '{probe_test_pattern}' (after removing train): {len(test_candidates)}")

    # Step 4: Sample test cities
    if len(test_candidates) < n_test_cities:
        raise ValueError(f"Not enough cities for testing! Requested {n_test_cities}, but only {len(test_candidates)} available after removing training cities.")
    test_sample = test_candidates.sample(n=n_test_cities, random_state=seed)

    # Combine train and test samples
    sampled_cities = pd.concat([train_sample, test_sample], ignore_index=True)

    print(f"Sampled {n_train_cities} training cities and {n_test_cities} test cities")
    print(f"Total cities for probing: {len(sampled_cities)}")
    
    
    # Create partial prompts using centralized config function
    partial_prompts = []
    city_info = []
    extraction_indices = None
    position_names = None

    for idx, city in sampled_cities.iterrows():
        # Get prompt configuration for this format
        city_dict = {
            'row_id': city['row_id'],
            'asciiname': city['asciiname'],
            'x': city['x'],
            'y': city['y'],
            'country_code': city['country_code'],
            'region': city.get('region', None)
        }

        prompt_config = get_prompt_config(prompt_format, city_dict)
        partial_prompts.append(prompt_config['prompt'])

        # Store extraction indices (same for all cities)
        if extraction_indices is None:
            extraction_indices = prompt_config['extraction_indices']
            position_names = prompt_config['position_names']

        city_info.append({
            'row_id': city['row_id'],
            'name': city['asciiname'],
            'x': city['x'],
            'y': city['y'],
            'country': city['country_code'],
            'region': city.get('region', None)  # Store region if available
        })
    
    print(f"Created {len(partial_prompts)} partial prompts")
    
    # Tokenize partial prompts with LEFT padding - EXACTLY LIKE NOTEBOOK
    tokenized_partial = tokenizer(
        partial_prompts,
        padding=True,
        truncation=True,
        return_tensors='pt',
        add_special_tokens=False  # Don't add BOS - already in prompt
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
    
    # Set analysis directory BEFORE the loop
    analysis_dir = output_dir
    
    results = []
    predictions_for_animation = []
    fit_quality_data = []  # Store data for fit quality plots
    
    for i, (step, checkpoint_path) in enumerate(tqdm(checkpoint_dirs, desc="Processing")):
        print(f"\nStep {step}:")
        # Get predictions for animation - always include all checkpoints
        return_preds = True
        
        # Get predictions for fit quality plots based on save_fits setting
        get_fit_quality = save_fits  # Generate for ALL checkpoints if save_fits is True
        
        # Determine if we should save weights for this checkpoint
        save_weights = save_fits  # Save weights when we save fit quality plots
        
        # Check if we should save representations for this checkpoint
        save_representations = False
        if save_repr_ckpts:
            # -1 means last checkpoint in the list
            is_last_checkpoint = (i == len(checkpoint_dirs) - 1)
            if -1 in save_repr_ckpts and is_last_checkpoint:
                save_representations = True
            # Check for specific step number
            elif step in save_repr_ckpts:
                save_representations = True
        
        result_data = analyze_checkpoint(
            checkpoint_path, step,
            partial_input_ids, partial_attention_mask,
            x_train, x_test, y_train, y_test,
            n_train_cities, device, layer_indices,
            return_predictions=(return_preds or get_fit_quality),
            method_config=method_config,
            return_probe_weights=save_weights,
            return_representations=save_representations,
            extraction_indices=extraction_indices,
            position_names=position_names
        )
        
        # Handle tuple return (result, representations_data)
        if isinstance(result_data, tuple):
            result, representations_data = result_data
        else:
            result = result_data
            representations_data = None
        
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
        
        # Collect fit quality data and weights for selected checkpoints
        if get_fit_quality and 'x_train_pred' in result:
            fit_data_entry = {
                'step': step,
                'x_train_pred': result['x_train_pred'],
                'y_train_pred': result['y_train_pred'],
                'x_test_pred': result['x_test_pred'],
                'y_test_pred': result['y_test_pred']
            }
            
            # Add probe weights if available
            if 'x_probe_coef' in result:
                fit_data_entry['x_probe_coef'] = result['x_probe_coef']
                fit_data_entry['y_probe_coef'] = result['y_probe_coef']
                fit_data_entry['x_probe_intercept'] = result['x_probe_intercept']
                fit_data_entry['y_probe_intercept'] = result['y_probe_intercept']
            
            fit_quality_data.append(fit_data_entry)
        
        # Save representations if requested (separate from CSV)
        if representations_data is not None:
            # Create representations directory structure
            repr_dir = analysis_dir / 'representations' / f'checkpoint-{step}'
            repr_dir.mkdir(parents=True, exist_ok=True)
            
            # Save representations as PyTorch tensor
            repr_path = repr_dir / 'representations.pt'
            repr_shape = representations_data['representations'].shape
            torch.save({
                'representations': torch.from_numpy(representations_data['representations']),
                'representations_flat': torch.from_numpy(representations_data['representations_flat']),
                'input_ids': torch.from_numpy(representations_data['input_ids']),
                'step': step,
                'layers': layer_indices,
            }, repr_path)

            # Save metadata
            metadata = {
                'step': step,
                'checkpoint_path': str(checkpoint_path),
                'layers': layer_indices,
                'n_cities': repr_shape[0],
                'n_tokens': repr_shape[1],
                'n_layers': repr_shape[2],
                'hidden_dim': repr_shape[3],
                'n_train_cities': n_train_cities,
                'n_test_cities': repr_shape[0] - n_train_cities,
                'representation_shape': list(repr_shape),  # (n_cities, n_tokens, n_layers, hidden_dim)
                'representation_flat_dim': representations_data['representations_flat'].shape[1],
                'city_info': city_info,  # Include city information
                'probe_train_pattern': probe_train_pattern,
                'probe_test_pattern': probe_test_pattern,
                'prompt_format': prompt_format,  # Include prompt format
                'x_train_mean': result.get('x_train_mean'),  # Training mean for x
                'y_train_mean': result.get('y_train_mean'),  # Training mean for y
            }
            
            metadata_path = repr_dir / 'metadata.json'
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
            
            print(f"  Saved representations to {repr_dir}/")
        
        print(f"  X R²: {result['x_test_r2']:.3f}, Y R²: {result['y_test_r2']:.3f}")
        print(f"  Mean dist error: {result['mean_dist_error']:.2f}")
    
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
                pred_data['y_r2'], pred_data['mean_error'],
                highlight_pattern, highlight_label, highlight_color,
                plot_links, plot_autolim
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
                final_pred['y_r2'], final_pred['mean_error'],
                highlight_pattern, highlight_label, highlight_color,
                plot_links, plot_autolim
            )
            final_map_path = analysis_dir / 'world_map_final.png'
            final_fig.savefig(final_map_path, dpi=150, bbox_inches='tight')
            plt.close(final_fig)
            print(f"Final world map saved to {final_map_path}")
    
    # Generate fit quality plots and save probe weights
    if fit_quality_data:
        print("\n" + "="*60)
        print("Generating Fit Quality Plots and Probe Weights")
        print("="*60)
        
        # Create fits and weights subdirectories
        fits_dir = analysis_dir / 'fits'
        fits_dir.mkdir(parents=True, exist_ok=True)
        weights_dir = analysis_dir / 'weights'
        weights_dir.mkdir(parents=True, exist_ok=True)
        
        for i, fit_data in enumerate(fit_quality_data):
            step = fit_data['step']
            
            # Create fit quality plot
            fig = create_fit_quality_plot(
                x_train, y_train, fit_data['x_train_pred'], fit_data['y_train_pred'],
                x_test, y_test, fit_data['x_test_pred'], fit_data['y_test_pred'],
                step, experiment_name
            )
            
            # Save plot
            plot_path = fits_dir / f'fit_quality_step{step:05d}.png'
            fig.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            print(f"  Saved fit quality plot for step {step} to {plot_path}")
            
            # Save and visualize probe weights if available
            if 'x_probe_coef' in fit_data:
                # Save weights as numpy arrays
                x_weights_path = weights_dir / f'step{step:05d}_x_weights.npy'
                y_weights_path = weights_dir / f'step{step:05d}_y_weights.npy'
                np.save(x_weights_path, fit_data['x_probe_coef'])
                np.save(y_weights_path, fit_data['y_probe_coef'])
                
                # Also save intercepts and training means
                intercepts_path = weights_dir / f'step{step:05d}_intercepts.npz'
                np.savez(intercepts_path,
                        x_intercept=fit_data['x_probe_intercept'],
                        y_intercept=fit_data['y_probe_intercept'],
                        x_train_mean=fit_data.get('x_train_mean', 0.0),
                        y_train_mean=fit_data.get('y_train_mean', 0.0))
                
                # Create weight visualization
                # Get hidden size from training config or use default
                hidden_size = 128  # Default
                if 'model' in training_config:
                    hidden_size = training_config['model'].get('hidden_size', 128)
                
                weight_fig = visualize_probe_weights(
                    fit_data['x_probe_coef'], 
                    fit_data['y_probe_coef'],
                    step, 
                    layer_indices,
                    hidden_size
                )
                
                # Save weight visualization
                weight_plot_path = weights_dir / f'step{step:05d}_weights.png'
                weight_fig.savefig(weight_plot_path, dpi=150, bbox_inches='tight')
                plt.close(weight_fig)
                print(f"  Saved probe weights to {weights_dir}/step{step:05d}_*.npy and visualization")
    
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