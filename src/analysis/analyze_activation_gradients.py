#!/usr/bin/env python3
"""
Analyze gradients flowing from loss to specific token activations.

This script performs forward pass on complete sequences like:
<bos>dist(c_1234,c_5678)=2332<eos>

And backpropagates gradients from the loss on the output tokens
to specific intermediate activations (e.g., "4" and "," tokens).

Usage:
    python analyze_activation_gradients.py configs/analysis/activation_gradients.yaml
"""

import argparse
import yaml
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import shutil
import json
import pickle
from tqdm import tqdm
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
# Get project root from current file location (src/analysis/script.py -> project root)
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.utils import init_directory, euclidean_distance, filter_dataframe_by_pattern
from transformers import AutoTokenizer, Qwen2ForCausalLM


def create_distance_sequences(cities_df: pd.DataFrame, config: dict) -> List[Dict]:
    """
    Create complete distance task sequences for analysis.

    Args:
        cities_df: DataFrame with city information
        config: Configuration dict with sampling parameters

    Returns:
        List of dictionaries with sequence information
    """
    n_samples = config.get('n_samples', 100)
    seed = config.get('seed', 42)
    probe_train_pattern = config.get('probe_train', 'region:.*')  # Default to all
    probe_test_pattern = config.get('probe_test', 'region:.*')

    np.random.seed(seed)

    # Apply filtering patterns if specified
    train_cities = filter_dataframe_by_pattern(cities_df, probe_train_pattern, column_name='region')
    test_cities = filter_dataframe_by_pattern(cities_df, probe_test_pattern, column_name='region')

    print(f"Train cities after filtering: {len(train_cities)}")
    print(f"Test cities after filtering: {len(test_cities)}")

    # Sample from both train and test sets
    n_train_samples = int(n_samples * 0.7)  # 70% from train
    n_test_samples = n_samples - n_train_samples

    sequences = []

    # Create train sequences
    for i in range(n_train_samples):
        # Sample two different cities
        if len(train_cities) < 2:
            print(f"Warning: Not enough train cities to sample pairs")
            break

        city_indices = np.random.choice(len(train_cities), size=2, replace=False)
        city1 = train_cities.iloc[city_indices[0]]
        city2 = train_cities.iloc[city_indices[1]]

        # Calculate actual Euclidean distance using x,y coordinates
        distance = euclidean_distance(city1['x'], city1['y'], city2['x'], city2['y'])
        distance = int(np.round(distance))  # Round to integer as in dataset creation

        # Create sequence string with 4-digit zero-padded city IDs (space-delimited as expected by model)
        c1_str = str(int(city1['city_id'])).zfill(4)
        c2_str = str(int(city2['city_id'])).zfill(4)
        dist_str = f"dist(c_{c1_str},c_{c2_str})={distance}"
        spaced_str = ' '.join(dist_str)
        sequence = f"<bos> {spaced_str} <eos>"

        # Store sequence info
        sequences.append({
            'sequence': sequence,
            'city1_id': int(city1['city_id']),
            'city2_id': int(city2['city_id']),
            'distance': distance,
            'city1_x': city1['x'],
            'city1_y': city1['y'],
            'city2_x': city2['x'],
            'city2_y': city2['y'],
            'city1_region': city1.get('region', 'Unknown'),
            'city2_region': city2.get('region', 'Unknown'),
            'split': 'train'
        })

    # Create test sequences
    for i in range(n_test_samples):
        if len(test_cities) < 2:
            print(f"Warning: Not enough test cities to sample pairs")
            break

        city_indices = np.random.choice(len(test_cities), size=2, replace=False)
        city1 = test_cities.iloc[city_indices[0]]
        city2 = test_cities.iloc[city_indices[1]]

        # Calculate actual Euclidean distance
        distance = euclidean_distance(city1['x'], city1['y'], city2['x'], city2['y'])
        distance = int(np.round(distance))

        # Create sequence string with 4-digit zero-padded city IDs
        c1_str = str(int(city1['city_id'])).zfill(4)
        c2_str = str(int(city2['city_id'])).zfill(4)
        dist_str = f"dist(c_{c1_str},c_{c2_str})={distance}"
        spaced_str = ' '.join(dist_str)
        sequence = f"<bos> {spaced_str} <eos>"

        sequences.append({
            'sequence': sequence,
            'city1_id': int(city1['city_id']),
            'city2_id': int(city2['city_id']),
            'distance': distance,
            'city1_x': city1['x'],
            'city1_y': city1['y'],
            'city2_x': city2['x'],
            'city2_y': city2['y'],
            'city1_region': city1.get('region', 'Unknown'),
            'city2_region': city2.get('region', 'Unknown'),
            'split': 'test'
        })

    return sequences


def tokenize_and_identify_positions(sequence: str, tokenizer) -> Dict:
    """
    Tokenize sequence and identify key token positions.

    Args:
        sequence: Input sequence string (already space-delimited)
        tokenizer: HuggingFace tokenizer

    Returns:
        Dictionary with tokens, input_ids, and key positions
    """
    # Tokenize (sequence is already space-delimited)
    encoding = tokenizer(sequence, return_tensors='pt', add_special_tokens=False)
    input_ids = encoding['input_ids']

    # Decode tokens to find positions
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0]]

    # Find key positions
    positions = {}

    # Find positions in the sequence
    for i, token in enumerate(tokens):
        # First city digits
        if i > 0 and tokens[i-1] == '_' and token.isdigit():
            # Start of first city ID
            if 'city1_start' not in positions:
                positions['city1_start'] = i
                # Find end of first city ID (look for comma)
                for j in range(i, min(i+10, len(tokens))):
                    if tokens[j] == ',':
                        positions['city1_end'] = j - 1  # Last digit before comma
                        positions['comma'] = j
                        break

        # Second city digits
        if i > 0 and tokens[i-1] == '_' and token.isdigit() and 'comma' in positions and i > positions['comma']:
            # Start of second city ID
            if 'city2_start' not in positions:
                positions['city2_start'] = i
                # Find end of second city ID (look for closing paren)
                for j in range(i, min(i+10, len(tokens))):
                    if tokens[j] == ')':
                        positions['city2_end'] = j - 1  # Last digit before paren
                        positions['close_paren'] = j
                        break

        # Equals sign
        if token == '=':
            positions['equals'] = i

        # EOS token
        if token == '<eos>' or 'eos' in token.lower():
            positions['eos'] = i

    # Output tokens are those after '=' and before '<eos>'
    if 'equals' in positions and 'eos' in positions:
        positions['output_start'] = positions['equals'] + 1
        positions['output_end'] = positions['eos'] - 1

    return {
        'input_ids': input_ids,
        'tokens': tokens,
        'positions': positions,
        'sequence_length': len(tokens)
    }


def analyze_gradients_for_sequence(
    model,
    tokenizer,
    sequence_info: Dict,
    target_layers: List[int],
    target_positions: List[str],
    device: torch.device,
    debug_first: bool = False
) -> Dict:
    """
    Analyze gradients for a single sequence.

    Args:
        model: The language model
        tokenizer: The tokenizer
        sequence_info: Dictionary with sequence information
        target_layers: List of layer indices to analyze
        target_positions: List of position names to track gradients for
        device: Torch device

    Returns:
        Dictionary with gradient analysis results
    """
    # Tokenize sequence
    tokenized = tokenize_and_identify_positions(sequence_info['sequence'], tokenizer)
    input_ids = tokenized['input_ids'].to(device)
    positions = tokenized['positions']

    # Map position names to actual indices
    position_mapping = {
        'city1_last_digit': positions.get('city1_end'),
        'comma': positions.get('comma'),
        'city2_last_digit': positions.get('city2_end'),
        'equals': positions.get('equals')
    }

    # DEBUG: Print what positions we found (first sequence only)
    if debug_first:
        print(f"\nDEBUG - Sequence: {sequence_info['sequence'][:50]}...")
        print(f"DEBUG - Found positions: {positions}")
        print(f"DEBUG - Position mapping: {position_mapping}")
        print(f"DEBUG - First 30 tokens: {tokenized['tokens'][:30]}")
        print(f"DEBUG - Target positions requested: {target_positions}")

    # Forward pass
    model.eval()  # Keep eval mode for consistent behavior (dropout off)

    # Storage for captured gradients - now capture ALL tokens
    captured_gradients = {}
    captured_full_gradients = {}  # Store full gradient tensors for heatmap

    # Define backward hook to capture gradients
    def capture_all_gradients(layer_idx):
        def hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                grad = grad_output[0]
                # Store the full gradient tensor for heatmap
                if len(grad.shape) == 3:  # [batch, seq, hidden]
                    # Store full sequence gradient
                    captured_full_gradients[f"layer_{layer_idx}"] = grad[0].clone().detach()  # [seq, hidden]

                    # Also extract specific positions as before
                    for pos_name, pos_idx in position_mapping.items():
                        if pos_idx is not None and pos_idx < grad.shape[1]:
                            key = f"layer_{layer_idx}_{pos_name}"
                            captured_gradients[key] = grad[0, pos_idx, :].clone().detach()
                            if debug_first:
                                print(f"DEBUG - Captured gradient for {key}: norm={captured_gradients[key].norm().item():.6f}")
        return hook

    # Register backward hooks on target layers
    hooks = []
    for layer_idx in target_layers:
        if layer_idx < len(model.model.layers):
            hook = model.model.layers[layer_idx].register_full_backward_hook(
                capture_all_gradients(layer_idx)
            )
            hooks.append(hook)

    # Forward pass
    outputs = model(input_ids)
    logits = outputs.logits

    # Calculate loss on output tokens (the distance value and <eos>)
    if 'output_start' in positions and 'output_end' in positions:
        # Get positions that predict output tokens
        target_start = positions['output_start']
        target_end = positions['output_end'] + 1  # Include <eos>

        # Loss only on the output portion
        loss_fn = nn.CrossEntropyLoss(reduction='mean')

        # Logits for positions that predict output tokens
        pred_logits = logits[0, target_start-1:target_end-1, :]

        # Target tokens
        targets = input_ids[0, target_start:target_end]

        # Calculate loss
        loss = loss_fn(pred_logits, targets)
    else:
        # Fallback to full sequence loss
        loss_fn = nn.CrossEntropyLoss(reduction='mean')
        pred_logits = logits[0, :-1, :]
        targets = input_ids[0, 1:]
        loss = loss_fn(pred_logits, targets)

    # Backward pass - this will trigger the hooks
    loss.backward()

    # Clean up hooks
    for hook in hooks:
        hook.remove()

    # Extract gradients from captured_gradients
    gradient_results = {
        'loss': loss.item(),
        'gradients': {},
        'full_gradients': {},  # Add full gradient tensors
        'positions': positions,
        'tokens': tokenized['tokens'],
        'distance': sequence_info['distance'],
        'city1_region': sequence_info['city1_region'],
        'city2_region': sequence_info['city2_region'],
        'split': sequence_info['split']
    }

    # Convert captured gradients to the expected format
    for key, grad_tensor in captured_gradients.items():
        # Parse layer index and position name from key
        parts = key.split('_')
        layer_idx = int(parts[1])
        pos_name = '_'.join(parts[2:])

        gradient_results['gradients'][key] = {
            'gradient': grad_tensor.cpu().numpy(),
            'activation': None,  # We don't have activations with this method
            'position': position_mapping.get(pos_name),
            'layer': layer_idx
        }

    # Add full gradient tensors
    for key, grad_tensor in captured_full_gradients.items():
        gradient_results['full_gradients'][key] = grad_tensor.cpu().numpy()

    return gradient_results


def perform_gradient_pca_analysis(gradient_results: List[Dict], key: str, n_components: int = 10) -> Dict:
    """
    Perform PCA analysis on gradients for a specific layer/position combination.

    Args:
        gradient_results: List of gradient analysis results
        key: Specific gradient key (e.g., 'layer_8_comma')
        n_components: Number of PCA components

    Returns:
        Dictionary with PCA results
    """
    # Extract gradients for this key
    gradients = []
    distances = []
    regions = []
    splits = []

    for result in gradient_results:
        if key in result['gradients']:
            gradients.append(result['gradients'][key]['gradient'])
            distances.append(result['distance'])
            regions.append((result['city1_region'], result['city2_region']))
            splits.append(result['split'])

    if not gradients:
        return None

    # Convert to numpy array
    gradients = np.array(gradients)  # Shape: (n_samples, hidden_dim)

    # Standardize gradients
    scaler = StandardScaler()
    gradients_scaled = scaler.fit_transform(gradients)

    # Perform PCA
    n_components_actual = min(n_components, gradients_scaled.shape[0], gradients_scaled.shape[1])
    pca = PCA(n_components=n_components_actual)
    pca_gradients = pca.fit_transform(gradients_scaled)

    return {
        'pca': pca,
        'scaler': scaler,
        'pca_gradients': pca_gradients,
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'distances': np.array(distances),
        'regions': regions,
        'splits': splits,
        'original_shape': gradients.shape
    }


def create_gradient_pca_plots(pca_results: Dict, key: str, output_dir: Path):
    """
    Create PCA visualization plots for gradients.

    Args:
        pca_results: Results from perform_gradient_pca_analysis
        key: Gradient key for title
        output_dir: Directory to save plots
    """
    if pca_results is None:
        return

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Plot 1: PC1 vs PC2 colored by distance
    ax = axes[0, 0]
    scatter = ax.scatter(
        pca_results['pca_gradients'][:, 0],
        pca_results['pca_gradients'][:, 1],
        c=pca_results['distances'],
        cmap='viridis',
        alpha=0.6,
        s=30
    )
    ax.set_xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]:.1%})')
    ax.set_ylabel(f'PC2 ({pca_results["explained_variance_ratio"][1]:.1%})')
    ax.set_title('Gradient PCA: PC1 vs PC2 (colored by distance)')
    plt.colorbar(scatter, ax=ax, label='Distance')
    ax.grid(True, alpha=0.3)

    # Plot 2: PC1 vs PC3 colored by distance (if available)
    ax = axes[0, 1]
    if pca_results['pca_gradients'].shape[1] > 2:
        scatter = ax.scatter(
            pca_results['pca_gradients'][:, 0],
            pca_results['pca_gradients'][:, 2],
            c=pca_results['distances'],
            cmap='viridis',
            alpha=0.6,
            s=30
        )
        ax.set_xlabel(f'PC1 ({pca_results["explained_variance_ratio"][0]:.1%})')
        ax.set_ylabel(f'PC3 ({pca_results["explained_variance_ratio"][2]:.1%})')
        ax.set_title('Gradient PCA: PC1 vs PC3 (colored by distance)')
        plt.colorbar(scatter, ax=ax, label='Distance')
    ax.grid(True, alpha=0.3)

    # Plot 3: Explained variance
    ax = axes[1, 0]
    n_components_to_show = min(10, len(pca_results['explained_variance_ratio']))
    ax.bar(range(1, n_components_to_show + 1),
           pca_results['explained_variance_ratio'][:n_components_to_show])
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance Ratio')
    ax.set_title('PCA Explained Variance')
    ax.set_xticks(range(1, n_components_to_show + 1))
    ax.grid(True, alpha=0.3)

    # Plot 4: Cumulative explained variance
    ax = axes[1, 1]
    cumsum = np.cumsum(pca_results['explained_variance_ratio'])
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=8)
    ax.axhline(y=0.9, color='r', linestyle='--', label='90% variance')
    ax.axhline(y=0.95, color='orange', linestyle='--', label='95% variance')
    ax.set_xlabel('Number of Components')
    ax.set_ylabel('Cumulative Explained Variance')
    ax.set_title('Cumulative Variance Explained')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.suptitle(f'Gradient PCA Analysis: {key}', fontsize=14)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / f'gradient_pca_{key}.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"  Saved PCA plot for {key}")


def create_gradient_norm_heatmap(gradient_results: List[Dict], target_layers: List[int], output_dir: Path):
    """
    Create a heatmap showing gradient norms per token across layers.

    Args:
        gradient_results: List of gradient analysis results
        target_layers: List of layer indices
        output_dir: Directory to save the plot
    """
    # Process multiple sequences and average the gradient norms
    all_heatmaps = []
    all_tokens = []
    max_seq_len = 0

    # First pass: find max sequence length and collect tokens
    for result in gradient_results[:10]:  # Use first 10 sequences for visualization
        if 'full_gradients' not in result:
            continue

        # Store tokens for x-axis labels
        if 'tokens' in result and len(all_tokens) == 0:
            all_tokens = result['tokens']
            print(f"DEBUG - Captured tokens for heatmap: {all_tokens[:25]}")

        for key in result['full_gradients']:
            if result['full_gradients'][key] is not None:
                seq_len = result['full_gradients'][key].shape[0]
                max_seq_len = max(max_seq_len, seq_len)
                break

    if max_seq_len == 0:
        print("No gradient data available for heatmap")
        return

    # Create token labels for x-axis
    # Find closing paren position dynamically
    close_paren_idx = max_seq_len - 1  # Default to end
    if all_tokens:
        for i, token in enumerate(all_tokens):
            if token == ')':
                close_paren_idx = i
                break

    token_labels = []
    if all_tokens:
        for i, token in enumerate(all_tokens[:min(close_paren_idx + 1, max_seq_len)]):
            # Replace digits with X for aggregation (works for any task)
            if token.isdigit():
                token_labels.append('X')
            else:
                token_labels.append(token)
    else:
        # Fallback to position numbers if tokens not available
        token_labels = [str(i) for i in range(min(close_paren_idx + 1, max_seq_len))]

    # Update max_seq_len to stop at close paren
    display_seq_len = len(token_labels)

    # Second pass: create heatmaps with padding
    for result in gradient_results[:10]:
        if 'full_gradients' not in result:
            continue

        # Create heatmap for this sequence with max length
        heatmap = np.zeros((len(target_layers), max_seq_len))

        for i, layer_idx in enumerate(target_layers):
            key = f"layer_{layer_idx}"
            if key in result['full_gradients'] and result['full_gradients'][key] is not None:
                # Calculate gradient norm for each token
                gradient_tensor = result['full_gradients'][key]  # [seq_len, hidden_dim]
                gradient_norms = np.linalg.norm(gradient_tensor, axis=1)  # [seq_len]
                # Fill in the heatmap up to the actual sequence length
                seq_len = len(gradient_norms)
                heatmap[i, :seq_len] = gradient_norms

        all_heatmaps.append(heatmap)

    # Average across sequences
    if not all_heatmaps:
        print("No gradient data available for heatmap")
        return

    avg_heatmap = np.mean(all_heatmaps, axis=0)

    # Truncate to display_seq_len for visualization
    avg_heatmap = avg_heatmap[:, :display_seq_len]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap
    im = ax.imshow(avg_heatmap, aspect='auto', cmap='viridis', interpolation='nearest')

    # Set labels
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Gradient Norm Heatmap (averaged over sequences)', fontsize=14)

    # Set y-tick labels to layer indices
    ax.set_yticks(range(len(target_layers)))
    ax.set_yticklabels([f'Layer {l}' for l in target_layers])

    # Set x-tick labels to actual tokens
    ax.set_xticks(range(len(token_labels)))
    ax.set_xticklabels(token_labels, rotation=0, ha='center', fontsize=10)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Gradient Norm', fontsize=12)

    # Add vertical lines at key structural tokens (dynamically detected)
    for i, token in enumerate(token_labels[:-1]):
        next_token = token_labels[i+1] if i+1 < len(token_labels) else ''
        # Add lines at major transitions
        if token == '(' or token == ',' or token == ')':
            ax.axvline(x=i+0.5, color='white', linestyle=':', alpha=0.5, linewidth=0.5)

    plt.tight_layout()

    # Save the plot
    plot_path = output_dir / 'gradient_norm_heatmap.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved gradient norm heatmap to {plot_path}")

    # Also create individual heatmaps for specific sequences
    fig, axes = plt.subplots(2, 2, figsize=(20, 10))
    axes = axes.flatten()

    for idx, ax in enumerate(axes):
        if idx >= len(all_heatmaps):
            ax.axis('off')
            continue

        # Trim the heatmap to display_seq_len (up to close paren)
        heatmap = all_heatmaps[idx][:, :display_seq_len]

        im = ax.imshow(heatmap, aspect='auto', cmap='viridis', interpolation='nearest')
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Layer')
        ax.set_title(f'Sequence {idx+1}')
        ax.set_yticks(range(len(target_layers)))
        ax.set_yticklabels([f'L{l}' for l in target_layers])
        plt.colorbar(im, ax=ax)

    plt.suptitle('Gradient Norm Heatmaps - Individual Sequences', fontsize=14)
    plt.tight_layout()

    # Save individual sequences plot
    plot_path = output_dir / 'gradient_norm_heatmap_individual.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved individual gradient norm heatmaps to {plot_path}")


def visualize_gradient_statistics(all_gradient_results: List[Dict], output_dir: Path):
    """
    Create comprehensive visualizations of gradient statistics.

    Args:
        all_gradient_results: List of all gradient analysis results
        output_dir: Directory to save plots
    """
    # Aggregate gradient statistics
    gradient_stats = {}

    for result in all_gradient_results:
        for key, grad_info in result['gradients'].items():
            gradient = grad_info['gradient']

            if key not in gradient_stats:
                gradient_stats[key] = {
                    'norms': [],
                    'means': [],
                    'stds': [],
                    'distances': [],
                    'losses': [],
                    'splits': []
                }

            gradient_stats[key]['norms'].append(np.linalg.norm(gradient))
            gradient_stats[key]['means'].append(np.mean(gradient))
            gradient_stats[key]['stds'].append(np.std(gradient))
            gradient_stats[key]['distances'].append(result['distance'])
            gradient_stats[key]['losses'].append(result['loss'])
            gradient_stats[key]['splits'].append(result['split'])

    # Create figure with subplots
    n_keys = len(gradient_stats)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    # Plot gradient norm vs distance for each key
    for i, (key, stats) in enumerate(gradient_stats.items()):
        if i >= 6:
            break

        ax = axes[i]

        # Separate by split
        train_mask = np.array(stats['splits']) == 'train'
        test_mask = ~train_mask

        # Plot train and test separately
        ax.scatter(
            np.array(stats['distances'])[train_mask],
            np.array(stats['norms'])[train_mask],
            alpha=0.5, s=20, label='Train', color='blue'
        )
        ax.scatter(
            np.array(stats['distances'])[test_mask],
            np.array(stats['norms'])[test_mask],
            alpha=0.5, s=20, label='Test', color='red'
        )

        ax.set_xlabel('Distance')
        ax.set_ylabel('Gradient Norm')
        ax.set_title(key.replace('_', ' ').title())
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Add correlation coefficient
        corr = np.corrcoef(stats['distances'], stats['norms'])[0, 1]
        ax.text(0.05, 0.95, f'Corr: {corr:.3f}',
                transform=ax.transAxes, fontsize=10,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    plt.suptitle('Gradient Norm vs Distance', fontsize=14)
    plt.tight_layout()

    # Save plot
    plot_path = output_dir / 'gradient_statistics.png'
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"Saved gradient statistics plot to {plot_path}")


def main():
    parser = argparse.ArgumentParser(description='Analyze activation gradients')
    parser.add_argument('config_path', type=str, help='Path to config YAML')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output')
    parser.add_argument('--debug', action='store_true', help='Debug mode with fewer samples')

    args = parser.parse_args()

    # Load config
    with open(args.config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract config values
    task_type = config.get('task_type', 'distance')

    # Currently only distance task is fully implemented
    if task_type != 'distance':
        raise NotImplementedError(f"Task type '{task_type}' is not yet implemented. Only 'distance' is currently supported.")

    output_dir = Path(config['output_dir'])
    experiment_dir = Path(config['experiment_dir'])
    cities_csv = Path(config.get('cities_csv', 'data/world_cities.csv'))
    n_samples = config.get('n_samples', 100)
    target_layers = config.get('target_layers', [6, 8, 10])
    target_positions = config.get('target_positions', ['city1_last_digit', 'comma'])
    device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
    seed = config.get('seed', 42)
    checkpoint = config.get('checkpoint', 'final')  # Default to final checkpoint

    # Debug mode
    if args.debug:
        n_samples = min(10, n_samples)
        print(f"Debug mode: Using only {n_samples} samples")

    # Initialize output directory
    output_dir = init_directory(output_dir, overwrite=args.overwrite)

    # Copy config to output
    shutil.copy(args.config_path, output_dir / 'config.yaml')

    print("="*60)
    print("Activation Gradient Analysis")
    print("="*60)
    print(f"Experiment: {experiment_dir.name}")
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {n_samples}")
    print(f"Target layers: {target_layers}")
    print(f"Target positions: {target_positions}")
    print(f"Device: {device}")

    # Determine checkpoint path
    checkpoints_dir = experiment_dir / 'checkpoints'
    if checkpoint == 'final':
        checkpoint_path = checkpoints_dir / 'final'
        if not checkpoint_path.exists():
            # Try to find the latest numbered checkpoint
            checkpoint_dirs = sorted([d for d in checkpoints_dir.glob('checkpoint-*')
                                    if d.is_dir() and d.name != 'checkpoint-0'])
            if checkpoint_dirs:
                checkpoint_path = checkpoint_dirs[-1]
            else:
                raise ValueError(f"No final checkpoint found in {experiment_dir}")
    else:
        checkpoint_path = checkpoints_dir / f'checkpoint-{checkpoint}'
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint {checkpoint} not found in {experiment_dir}")

    print(f"\nLoading model from: {checkpoint_path}")

    # Load model and tokenizer
    model = Qwen2ForCausalLM.from_pretrained(checkpoint_path)
    model = model.to(device)

    # IMPORTANT: We need the model in a state where gradients flow!
    # Don't use eval() mode if we want gradients through the model
    # Actually, eval() mode is fine - it just disables dropout, not gradients
    model.eval()

    # But we MUST ensure the model parameters require grad for gradients to flow
    for param in model.parameters():
        param.requires_grad = True

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    print("Model and tokenizer loaded successfully")

    # Load cities data
    cities_df = pd.read_csv(cities_csv)

    # Ensure city_id column exists (it should already be there)
    if 'city_id' not in cities_df.columns:
        raise ValueError(f"city_id column not found in {cities_csv}")

    print(f"Loaded {len(cities_df)} cities")

    # Create sequences for analysis
    print("\nCreating distance sequences...")
    sequences = create_distance_sequences(cities_df, config)
    print(f"Created {len(sequences)} sequences")

    # Analyze gradients for each sequence
    print("\nAnalyzing gradients...")
    gradient_results = []

    for i, seq_info in enumerate(tqdm(sequences, desc="Processing sequences")):
        result = analyze_gradients_for_sequence(
            model, tokenizer, seq_info,
            target_layers, target_positions, device,
            debug_first=(i == 0)  # Debug first sequence
        )
        gradient_results.append(result)

        # Check if we're actually getting gradients
        if i == 0:
            print(f"\nDEBUG - First result gradients keys: {list(result['gradients'].keys())}")
            if not result['gradients']:
                print("ERROR: No gradients collected! Something is wrong with position detection.")
                print(f"Positions found: {result['positions']}")
                print(f"Tokens: {result['tokens'][:30]}")
                raise RuntimeError("No gradients collected - position detection failed")

    print(f"Successfully analyzed {len(gradient_results)} sequences")

    # Save results
    print("\nSaving results...")

    # Create subdirectories
    gradients_dir = output_dir / 'gradients'
    gradients_dir.mkdir(exist_ok=True)

    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(exist_ok=True)

    pca_dir = output_dir / 'pca'
    pca_dir.mkdir(exist_ok=True)

    # Organize gradients by layer and position
    organized_gradients = {}

    for result in gradient_results:
        for key, grad_info in result['gradients'].items():
            if key not in organized_gradients:
                organized_gradients[key] = {
                    'gradients': [],
                    'activations': [],
                    'losses': [],
                    'distances': []
                }

            organized_gradients[key]['gradients'].append(grad_info['gradient'])
            organized_gradients[key]['activations'].append(grad_info['activation'])
            organized_gradients[key]['losses'].append(result['loss'])
            organized_gradients[key]['distances'].append(result['distance'])

    # Save organized gradients
    for key, data in organized_gradients.items():
        np.savez(
            gradients_dir / f'{key}.npz',
            gradients=np.array(data['gradients']),
            activations=np.array(data['activations']),
            losses=np.array(data['losses']),
            distances=np.array(data['distances'])
        )
        print(f"  Saved {key}: shape {np.array(data['gradients']).shape}")

    # Perform PCA analysis on gradients
    print("\nPerforming PCA analysis on gradients...")
    pca_results_all = {}

    for key in organized_gradients.keys():
        print(f"  Analyzing {key}...")
        pca_results = perform_gradient_pca_analysis(gradient_results, key, n_components=20)
        if pca_results:
            pca_results_all[key] = pca_results

            # Save PCA results
            pca_save_path = pca_dir / f'{key}_pca.pkl'
            with open(pca_save_path, 'wb') as f:
                pickle.dump(pca_results, f)

            # Create PCA plots
            create_gradient_pca_plots(pca_results, key, figures_dir)

    # Create comprehensive visualizations
    print("\nCreating visualizations...")
    visualize_gradient_statistics(gradient_results, figures_dir)

    # Create gradient norm heatmap
    print("\nCreating gradient norm heatmap...")
    create_gradient_norm_heatmap(gradient_results, target_layers, figures_dir)

    # Save metadata
    metadata = {
        'n_samples': len(gradient_results),
        'target_layers': target_layers,
        'target_positions': target_positions,
        'checkpoint_path': str(checkpoint_path),
        'keys': list(organized_gradients.keys()),
        'config': config
    }

    with open(output_dir / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)

    # Print summary statistics
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)

    for key, data in organized_gradients.items():
        gradients = np.array(data['gradients'])
        grad_norms = np.linalg.norm(gradients, axis=1)
        distances = np.array(data['distances'])

        # Calculate correlation between gradient norm and distance
        corr = np.corrcoef(grad_norms, distances)[0, 1]

        print(f"\n{key}:")
        print(f"  Mean gradient norm: {np.mean(grad_norms):.6f}")
        print(f"  Std gradient norm: {np.std(grad_norms):.6f}")
        print(f"  Correlation with distance: {corr:.3f}")

    # Summary of PCA results
    if pca_results_all:
        print("\n" + "="*60)
        print("PCA Analysis Summary")
        print("="*60)

        for key, pca_results in pca_results_all.items():
            cumsum = np.cumsum(pca_results['explained_variance_ratio'])
            n_90 = np.argmax(cumsum >= 0.9) + 1
            n_95 = np.argmax(cumsum >= 0.95) + 1

            print(f"\n{key}:")
            print(f"  PC1 variance: {pca_results['explained_variance_ratio'][0]:.1%}")
            print(f"  PC2 variance: {pca_results['explained_variance_ratio'][1]:.1%}")
            print(f"  Components for 90% variance: {n_90}")
            print(f"  Components for 95% variance: {n_95}")

    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()