#!/usr/bin/env python3
"""
Extract layer representations from trained models for dimensionality analysis.
Saves representations as numpy arrays for further analysis.
"""

import sys
import os
from pathlib import Path
import torch
import numpy as np
import yaml
import argparse
from transformers import AutoTokenizer, Qwen2ForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle

# Add parent directory to path
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory, get_dataset, preprocess_config
from datasets import load_from_disk

def extract_representations(model, dataloader, layer_idx, max_samples=5000, device='cuda'):
    """
    Extract representations from a specific layer.

    Args:
        model: The loaded model
        dataloader: DataLoader for the dataset
        layer_idx: Which layer to extract (0-based)
        max_samples: Maximum number of samples to extract
        device: Device to run on

    Returns:
        representations: numpy array of shape (n_samples, hidden_dim)
        labels: numpy array of labels if available
    """
    model.eval()
    model.to(device)

    all_representations = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Extracting layer {layer_idx}")):
            if batch_idx * dataloader.batch_size >= max_samples:
                break

            input_ids = batch['input_ids'].to(device)

            # Get hidden states
            outputs = model(input_ids, output_hidden_states=True)
            hidden_states = outputs.hidden_states  # Tuple of tensors, one per layer

            # Extract representations from specified layer
            # hidden_states[0] is embeddings, [1] is layer 0, etc.
            layer_output = hidden_states[layer_idx + 1]  # +1 because 0 is embeddings

            # Average over sequence length to get single representation per sample
            # Shape: (batch_size, seq_len, hidden_dim) -> (batch_size, hidden_dim)
            representations = layer_output.mean(dim=1)

            all_representations.append(representations.cpu().numpy())

            # Get labels if available
            if 'labels' in batch:
                all_labels.append(batch['labels'].cpu().numpy())

    # Concatenate all batches
    all_representations = np.concatenate(all_representations, axis=0)[:max_samples]

    if all_labels:
        all_labels = np.concatenate(all_labels, axis=0)[:max_samples]
    else:
        all_labels = None

    return all_representations, all_labels

def process_experiment(exp_name, exp_dir, layer=5, max_samples=5000):
    """Process a single experiment to extract representations."""

    print(f"\nProcessing {exp_name}...")

    # Find the final checkpoint
    checkpoints = list(exp_dir.glob("checkpoint-*"))
    if not checkpoints:
        print(f"  No checkpoints found in {exp_dir}")
        return False

    # Get the checkpoint with highest number (last checkpoint)
    checkpoint_nums = []
    for ckpt in checkpoints:
        try:
            num = int(ckpt.name.split('-')[1])
            checkpoint_nums.append((num, ckpt))
        except:
            continue

    if not checkpoint_nums:
        print(f"  No valid checkpoints found")
        return False

    checkpoint_nums.sort()
    final_checkpoint = checkpoint_nums[-1][1]

    print(f"  Using checkpoint: {final_checkpoint.name}")

    # Load model
    try:
        model = Qwen2ForCausalLM.from_pretrained(final_checkpoint, torch_dtype=torch.float16)
        tokenizer = AutoTokenizer.from_pretrained(exp_dir / "tokenizer")
    except Exception as e:
        print(f"  Error loading model: {e}")
        return False

    # Load config to find dataset
    config_path = exp_dir / "config.yaml"
    if not config_path.exists():
        print(f"  Config not found")
        return False

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Load dataset using the same method as training
    dataset_path = Path(config['dataset']['path'])
    if not dataset_path.exists():
        print(f"  Dataset not found: {dataset_path}")
        return False

    try:
        # Load dataset using HuggingFace datasets
        from src.utils import BaseDataset
        dataset_hf = load_from_disk(dataset_path)

        # Get train split
        if 'train' in dataset_hf:
            train_data = dataset_hf['train']
        else:
            train_data = dataset_hf

        # Create BaseDataset instance
        dataset = BaseDataset(train_data, split='train')

        print(f"  Loaded dataset with {len(dataset)} samples")

        if len(dataset) == 0:
            print(f"  Empty dataset")
            return False

    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return False

    # Create dataloader with appropriate collator
    from src.utils import get_collator

    # Create a simple collator that just gets input_ids
    collator = get_collator(config, tokenizer)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=collator)

    # Extract representations
    representations, labels = extract_representations(
        model, dataloader, layer, max_samples=max_samples,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Save representations
    output_dir = exp_dir / "representations" / f"layer_{layer}"
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "representations.npy", representations)
    if labels is not None:
        np.save(output_dir / "labels.npy", labels)

    # Save metadata
    metadata = {
        'experiment': exp_name,
        'checkpoint': final_checkpoint.name,
        'layer': layer,
        'n_samples': representations.shape[0],
        'hidden_dim': representations.shape[1]
    }

    with open(output_dir / "metadata.yaml", 'w') as f:
        yaml.dump(metadata, f)

    print(f"  Saved representations: {representations.shape}")

    return True

def main(config_path, overwrite=False, debug=False):
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize output directory
    output_dir = Path(config['output_dir'])
    if output_dir.exists() and not overwrite:
        print(f"Output directory {output_dir} exists. Use --overwrite to continue.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    with open(output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)

    # Process experiments
    layer = config.get('layer', 5)
    max_samples = config.get('max_samples', 5000)

    results = []

    # Process PT1 experiments
    print("\n=== Processing PT1 experiments ===")
    for i in range(1, 8):
        exp_name = f"pt1-{i}"
        exp_dir = Path(f"data/experiments/{exp_name}")
        if exp_dir.exists():
            success = process_experiment(exp_name, exp_dir, layer, max_samples)
            results.append({'experiment': exp_name, 'success': success})

    # Process PT2 experiments
    print("\n=== Processing PT2 experiments ===")
    for i in range(1, 9):
        exp_name = f"pt2-{i}"
        exp_dir = Path(f"data/experiments/{exp_name}")
        if exp_dir.exists():
            success = process_experiment(exp_name, exp_dir, layer, max_samples)
            results.append({'experiment': exp_name, 'success': success})

    # Save results summary
    with open(output_dir / 'extraction_results.yaml', 'w') as f:
        yaml.dump(results, f)

    print(f"\nExtraction complete. Results saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('config_path', type=str, help='Path to config file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    args = parser.parse_args()

    main(args.config_path, args.overwrite, args.debug)