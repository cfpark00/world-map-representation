#!/usr/bin/env python3
"""
Extract and save representations from models for specified layers and token positions.
Compatible with analyze_representations_higher.py format.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import yaml
import argparse
from transformers import AutoTokenizer, Qwen2ForCausalLM
import pandas as pd
from tqdm import tqdm

# Add parent directory to path
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory

def get_prompt_config(prompt_format, city):
    """Returns prompt string and extraction indices for a given format."""

    # Use city_id which is the 4-digit ID in the cities CSV
    city_id = city['city_id'] if 'city_id' in city else f"{city.name:04d}"

    if prompt_format == 'distance_firstcity_last_and_trans':
        dist_str = f"dist(c_{city_id},"
        spaced_str = ' '.join(dist_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [11, 12]  # last digit, comma
        position_names = ['last_digit', 'trans']

    elif prompt_format == 'trianglearea_firstcity_last_and_trans':
        triarea_str = f"triarea(c_{city_id},"
        spaced_str = ' '.join(triarea_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [14, 15]  # last digit, comma
        position_names = ['last_digit', 'trans']

    elif prompt_format == 'angle_firstcity_last_and_trans':
        angle_str = f"angle(c_{city_id},"
        spaced_str = ' '.join(angle_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [12, 13]  # last digit, comma
        position_names = ['last_digit', 'trans']

    elif prompt_format == 'compass_firstcity_last_and_trans':
        compass_str = f"compass(c_{city_id},"
        spaced_str = ' '.join(compass_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [14, 15]  # last digit, comma
        position_names = ['last_digit', 'trans']

    elif prompt_format == 'inside_firstcity_last_and_trans':
        inside_str = f"inside(c_{city_id},"
        spaced_str = ' '.join(inside_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [13, 14]  # last digit, comma
        position_names = ['last_digit', 'trans']

    elif prompt_format == 'perimeter_firstcity_last_and_trans':
        perim_str = f"perim(c_{city_id},"
        spaced_str = ' '.join(perim_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [12, 13]  # last digit, comma
        position_names = ['last_digit', 'trans']

    elif prompt_format == 'crossing_firstcity_last_and_trans':
        cross_str = f"cross(c_{city_id},"
        spaced_str = ' '.join(cross_str)
        prompt = f"<bos> {spaced_str}"
        extraction_indices = [12, 13]  # last digit, comma
        position_names = ['last_digit', 'trans']

    else:
        raise ValueError(f"Unknown prompt format: {prompt_format}")

    return {
        'prompt': prompt,
        'extraction_indices': extraction_indices,
        'position_names': position_names
    }

def extract_representations(model, tokenizer, cities_df, prompt_format, layer_idx, token_idx=-1, max_samples=5000):
    """
    Extract representations from a specific layer and token position.

    Args:
        model: The loaded model
        tokenizer: The tokenizer
        cities_df: DataFrame with city information
        prompt_format: Format of prompts (e.g., 'distance_firstcity_last_and_trans')
        layer_idx: Which layer to extract (0-based)
        token_idx: Which token position to extract (-1 for concatenation)
        max_samples: Maximum number of samples

    Returns:
        representations: numpy array of shape (n_samples, hidden_dim * n_positions)
        labels: numpy array of labels (x, y coordinates)
    """
    model.eval()
    device = next(model.parameters()).device

    # Sample cities
    n_samples = min(len(cities_df), max_samples)
    sampled_cities = cities_df.sample(n=n_samples, random_state=42)

    all_representations = []
    all_labels = []

    with torch.no_grad():
        for _, city in tqdm(sampled_cities.iterrows(), total=n_samples, desc=f"Extracting layer {layer_idx}"):
            # Get prompt configuration
            prompt_config = get_prompt_config(prompt_format, city)
            prompt = prompt_config['prompt']
            extraction_indices = prompt_config['extraction_indices']

            # Tokenize
            inputs = tokenizer(prompt, return_tensors='pt').to(device)

            # Get hidden states
            outputs = model(**inputs, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of tensors, one per layer

            # Extract from specified layer
            layer_output = hidden_states[layer_idx + 1]  # +1 because 0 is embeddings
            layer_output = layer_output.squeeze(0)  # Remove batch dimension

            # Extract specified token positions
            if token_idx == -1:
                # Concatenate all extraction indices
                reps = []
                for idx in extraction_indices:
                    # Check if index is valid
                    if idx < layer_output.shape[0]:
                        reps.append(layer_output[idx].cpu().numpy())
                    else:
                        # Use last token if index out of bounds
                        reps.append(layer_output[-1].cpu().numpy())
                representation = np.concatenate(reps)
            else:
                # Extract specific token position
                idx = extraction_indices[token_idx]
                if idx < layer_output.shape[0]:
                    representation = layer_output[idx].cpu().numpy()
                else:
                    representation = layer_output[-1].cpu().numpy()

            all_representations.append(representation)
            all_labels.append([city['x'], city['y']])

    return np.array(all_representations), np.array(all_labels)

def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Parse experiment and task info from output_dir
    output_dir = Path(config['output_dir'])
    exp_name = config.get('experiment', 'unknown')
    checkpoint_name = config.get('checkpoint', None)

    # Initialize output directory
    if output_dir.exists() and not overwrite:
        print(f"Output directory {output_dir} exists. Use --overwrite to continue.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Load cities data
    cities_df = pd.read_csv(config['cities_csv'])
    print(f"Loaded {len(cities_df)} cities")

    # Find model checkpoint
    exp_dir = Path(config['experiment_dir'])
    checkpoints_dir = exp_dir / "checkpoints"

    if checkpoint_name:
        checkpoint_path = checkpoints_dir / checkpoint_name
    else:
        # Use last checkpoint
        checkpoints = sorted(checkpoints_dir.glob("checkpoint-*"))
        if not checkpoints:
            print(f"No checkpoints found in {checkpoints_dir}")
            return
        checkpoint_path = checkpoints[-1]
        checkpoint_name = checkpoint_path.name

    print(f"Loading model from {checkpoint_path}")

    # Load model and tokenizer
    model = Qwen2ForCausalLM.from_pretrained(checkpoint_path, torch_dtype=torch.float16)

    # Find tokenizer
    if (checkpoint_path / "tokenizer_config.json").exists():
        tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained("data/tokenizers/default_tokenizer")

    if torch.cuda.is_available():
        model = model.cuda()

    # Extract representations
    representations, labels = extract_representations(
        model, tokenizer, cities_df,
        prompt_format=config['prompt_format'],
        layer_idx=config['layer_index'],
        token_idx=config.get('token_index', -1),
        max_samples=config.get('max_samples', 5000)
    )

    # Save to checkpoint subdirectory
    save_dir = output_dir / checkpoint_name
    save_dir.mkdir(parents=True, exist_ok=True)

    np.save(save_dir / 'representations.npy', representations)
    np.save(save_dir / 'labels.npy', labels)

    # Save metadata
    metadata = {
        'experiment': exp_name,
        'checkpoint': checkpoint_name,
        'prompt_format': config['prompt_format'],
        'layer': config['layer_index'],
        'token_index': config.get('token_index', -1),
        'n_samples': representations.shape[0],
        'representation_dim': representations.shape[1]
    }

    with open(save_dir / 'metadata.yaml', 'w') as f:
        yaml.dump(metadata, f)

    print(f"Saved representations: {representations.shape}")
    print(f"Output directory: {save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)