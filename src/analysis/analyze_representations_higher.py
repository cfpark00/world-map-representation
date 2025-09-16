#!/usr/bin/env python3
"""
Analyze how internal representations evolve during training across all checkpoints.
Tracks R² scores for x/y coordinate prediction from partial prompts.
Generates dynamics plot showing R² scores over training.

Usage:
    python analyze_representations_higher.py <config_path> [--overwrite]
"""

import sys
import os
from pathlib import Path
import re
import argparse
import torch
import numpy as np
import pandas as pd
import yaml
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, Qwen2ForCausalLM
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_absolute_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory, filter_dataframe_by_pattern
import json


def analyze_checkpoint(checkpoint_path, step, partial_input_ids, partial_attention_mask,
                       x_train, x_test, y_train, y_test, n_train_cities,
                       device, layer_indices, method_config=None, prompt_format="dist"):
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
        # hidden_states includes embedding layer at index 0, so layer N is at index N
        layer_reps.append(outputs.hidden_states[idx])  # index 0 = embeddings, index N = layer N output

    # Concatenate layers if multiple
    if len(layer_reps) > 1:
        partial_representations = torch.cat(layer_reps, dim=-1)
    else:
        partial_representations = layer_reps[0]

    # Extract representations based on prompt format
    n_cities = partial_representations.shape[0]
    n_layers = len(layer_indices)
    hidden_dim = partial_representations.shape[-1] // n_layers if n_layers > 1 else partial_representations.shape[-1]

    if prompt_format == "dist_city_and_transition":
        # For dist_city_and_transition, extract 9 positions:
        # <bos> d i s t ( c _ i1 i2 i3 i4 , c _
        # Positions: 6 (c), 7 (_), 8 (i1), 9 (i2), 10 (i3), 11 (i4), 12 (,), 13 (c), 14 (_)

        # Extract the 9 token positions
        token_positions = []
        position_names = []

        # First city representations
        token_positions.append(partial_representations[:, 6, :])   # c
        position_names.append("c1")
        token_positions.append(partial_representations[:, 7, :])   # _
        position_names.append("_1")
        token_positions.append(partial_representations[:, 8, :])   # i1
        position_names.append("i1")
        token_positions.append(partial_representations[:, 9, :])   # i2
        position_names.append("i2")
        token_positions.append(partial_representations[:, 10, :])  # i3
        position_names.append("i3")
        token_positions.append(partial_representations[:, 11, :])  # i4
        position_names.append("i4")

        # Transition
        token_positions.append(partial_representations[:, 12, :])  # ,
        position_names.append(",")

        # Second city start
        token_positions.append(partial_representations[:, 13, :])  # c
        position_names.append("c2")
        token_positions.append(partial_representations[:, 14, :])  # _
        position_names.append("_2")

        # Stack to create (n_cities, 9 tokens, concatenated_layer_dims)
        token_stack = torch.stack(token_positions, dim=1)

        # Reshape to (n_cities, 9 tokens, n_layers, hidden_dim)
        partial_reps_reshaped = token_stack.reshape(n_cities, 9, n_layers, hidden_dim)

        # For probe training, concatenate all 9 positions
        partial_last_token_reps = torch.cat(token_positions, dim=1)
        partial_reps_np = partial_last_token_reps.cpu().numpy()

    elif prompt_format == "dist_city_last_and_comma":
        # For dist_city_last_and_comma, extract 2 positions:
        # <bos> d i s t ( c _ i1 i2 i3 i4 , c _
        # Extract: position 11 (i4 - last digit) and position 12 (comma)

        # Extract the 2 token positions
        token_positions = []
        position_names = []

        # Last digit of first city
        token_positions.append(partial_representations[:, 11, :])  # i4 (last digit)
        position_names.append("last_digit")

        # Comma
        token_positions.append(partial_representations[:, 12, :])  # ,
        position_names.append("comma")

        # Stack to create (n_cities, 2 tokens, concatenated_layer_dims)
        token_stack = torch.stack(token_positions, dim=1)

        # Reshape to (n_cities, 2 tokens, n_layers, hidden_dim)
        partial_reps_reshaped = token_stack.reshape(n_cities, 2, n_layers, hidden_dim)

        # For probe training, concatenate both positions
        partial_last_token_reps = torch.cat(token_positions, dim=1)
        partial_reps_np = partial_last_token_reps.cpu().numpy()

    else:
        # Default behavior for "dist" format - last 3 tokens
        underscore_reps = partial_representations[:, -1, :]  # "_" (last token)
        c_reps = partial_representations[:, -2, :]           # "c"
        comma_reps = partial_representations[:, -3, :]       # ","

        # Concatenate all three representations
        partial_last_token_reps = torch.cat([comma_reps, c_reps, underscore_reps], dim=1)

        # Convert to numpy
        partial_reps_np = partial_last_token_reps.cpu().numpy()

        # Stack the three token representations and reshape
        # Shape: (n_cities, 3 tokens, concatenated_layer_dims)
        token_stack = torch.stack([comma_reps, c_reps, underscore_reps], dim=1)
        # Reshape to (n_cities, 3 tokens, n_layers, hidden_dim)
        partial_reps_reshaped = token_stack.reshape(n_cities, 3, n_layers, hidden_dim)

    # Split into train and test
    X_train_coord = partial_reps_np[:n_train_cities]
    X_test_coord = partial_reps_np[n_train_cities:]

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

    # Train probes
    x_probe.fit(X_train_coord, x_train)
    x_train_pred = x_probe.predict(X_train_coord)
    x_test_pred = x_probe.predict(X_test_coord)

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

    return {
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
    method_config = config.get('method', None)  # Probe method configuration

    # Optional checkpoint parameter - can be "final", a number, or None for all
    checkpoint_param = config.get('checkpoint', None)

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
            else:
                print(f"Error: Final checkpoint not found at {final_path}")
                sys.exit(1)
        else:
            # Numeric checkpoint specified
            checkpoint_path = checkpoints_dir / f'checkpoint-{checkpoint_param}'
            if checkpoint_path.exists():
                checkpoint_dirs = [(int(checkpoint_param), checkpoint_path)]
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

    # Load tokenizer - no fallback
    tokenizer_path = checkpoints_dir / 'checkpoint-0'
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found at {tokenizer_path}")
        sys.exit(1)

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path))
    tokenizer.padding_side = 'left'  # For generation

    # Load cities data
    cities_df = pd.read_csv(cities_csv)

    # Use the scaled coordinates directly (x and y are already in the dataset)
    cities_df['row_id'] = cities_df['city_id']     # Use city_id as row_id

    # Apply filtering based on patterns
    if probe_train_pattern != '.*' or probe_test_pattern != '.*':
        train_cities = filter_dataframe_by_pattern(cities_df, probe_train_pattern, column_name='region')
        test_cities = filter_dataframe_by_pattern(cities_df, probe_test_pattern, column_name='region')

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
        elif prompt_format == 'dist_city_and_transition':
            # For this format, we want to extract from the full first city and transition
            # "<bos> d i s t ( c _ X X X X , c _"
            dist_str = f"dist(c_{city['row_id']},c_"
            spaced_str = ' '.join(dist_str)
            prompt = f"<bos> {spaced_str}"  # Same as dist but we'll extract different positions
        elif prompt_format == 'dist_city_last_and_comma':
            # For this format, same prompt but we'll extract last digit and comma
            # "<bos> d i s t ( c _ X X X X , c _"
            dist_str = f"dist(c_{city['row_id']},c_"
            spaced_str = ' '.join(dist_str)
            prompt = f"<bos> {spaced_str}"  # Same prompt, different extraction
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
            'country': city['country_code'],
            'region': city.get('region', None)  # Store region if available
        })

    # Tokenize partial prompts with LEFT padding - EXACTLY LIKE NOTEBOOK
    tokenized_partial = tokenizer(
        partial_prompts,
        padding=True,
        truncation=True,
        return_tensors='pt'
    )

    partial_input_ids = tokenized_partial['input_ids'].to(device)
    partial_attention_mask = tokenized_partial['attention_mask'].to(device)

    # Extract x and y as targets (scaled values)
    xs = np.array([c['x'] for c in city_info])
    ys = np.array([c['y'] for c in city_info])

    # Split into train and test
    x_train = xs[:n_train_cities]
    x_test = xs[n_train_cities:]
    y_train = ys[:n_train_cities]
    y_test = ys[n_train_cities:]

    # Analyze all checkpoints

    # Set analysis directory
    analysis_dir = output_dir

    results = []

    for i, (step, checkpoint_path) in enumerate(tqdm(checkpoint_dirs, desc="Processing")):

        result = analyze_checkpoint(
            checkpoint_path, step,
            partial_input_ids, partial_attention_mask,
            x_train, x_test, y_train, y_test,
            n_train_cities, device, layer_indices,
            method_config=method_config,
            prompt_format=prompt_format
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

    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('step')

    # Use the output_dir from config
    analysis_dir = output_dir

    # Save results
    output_csv = analysis_dir / 'representation_dynamics.csv'
    results_df.to_csv(output_csv, index=False)

    # Create dynamics plot with vertically arranged subplots

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
    plt.close()


if __name__ == "__main__":
    main()