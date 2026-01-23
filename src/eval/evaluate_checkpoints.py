#!/usr/bin/env python3
"""
Evaluate model checkpoints on test split of dataset.
Uses the same tokenized dataset that was used for training.

Usage:
    python evaluate_checkpoints.py <config.yaml> [--overwrite]
"""

import sys
import os
from pathlib import Path
import yaml
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from transformers import AutoTokenizer, Qwen2ForCausalLM
from datasets import load_from_disk
from typing import List, Dict, Any, Optional
from tqdm import tqdm
import json
import re
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
project_root = Path('/n/home12/cfpark00/WM_1')
sys.path.insert(0, str(project_root))

from src.utils import init_directory, convert_numpy_to_python, save_training_plots
from src.evaluation import evaluate_with_generation
from src.metrics import calculate_metric, get_failure_value, format_metric_for_display


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def get_checkpoint_paths(experiment_dir: Path, checkpoint_spec: Any) -> List[Path]:
    """Get list of checkpoint paths based on specification.

    Args:
        experiment_dir: Path to experiment directory
        checkpoint_spec: "all", "last", or list of checkpoint names/steps

    Returns:
        List of checkpoint paths sorted by step number
    """
    checkpoints_dir = experiment_dir / "checkpoints"

    if not checkpoints_dir.exists():
        raise ValueError(f"Checkpoints directory not found: {checkpoints_dir}")

    if checkpoint_spec == "all":
        # Get all checkpoints
        ckpt_paths = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
    elif checkpoint_spec == "last":
        # Get the checkpoint with highest step number
        all_ckpts = [p for p in checkpoints_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")]
        if not all_ckpts:
            raise ValueError("No checkpoints found")
        # Sort by step number
        ckpt_paths = [max(all_ckpts, key=lambda p: int(p.name.split('-')[1]))]
    elif isinstance(checkpoint_spec, list):
        # Get specific checkpoints
        ckpt_paths = []
        for ckpt in checkpoint_spec:
            if isinstance(ckpt, int):
                ckpt_name = f"checkpoint-{ckpt}"
            else:
                ckpt_name = ckpt
            ckpt_path = checkpoints_dir / ckpt_name
            if ckpt_path.exists():
                ckpt_paths.append(ckpt_path)
            else:
                print(f"Warning: Checkpoint not found: {ckpt_path}")
    else:
        raise ValueError(f"Invalid checkpoint specification: {checkpoint_spec}")

    # Sort by step number
    ckpt_paths.sort(key=lambda p: int(p.name.split('-')[1]))

    return ckpt_paths


def evaluate_checkpoint_with_details(
    model_path: Path,
    tokenizer,
    test_dataset,
    config: dict,
    device: str,
    save_full: bool = False
) -> Dict[str, Any]:
    """Evaluate a single checkpoint on test dataset, optionally saving full details.

    Returns:
        Dictionary with evaluation metrics and optionally detailed results
    """
    # Load model
    print(f"  Loading model from {model_path}")
    model = Qwen2ForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device
    )
    model.eval()

    # Use unified evaluation logic
    if save_full:
        # Get num_samples from config or use all test data
        num_samples = config.get('num_samples', len(test_dataset))
        batch_size = config.get('eval_batch_size', 32)

        # Call unified evaluation with return_details=True
        metrics, detailed_results = evaluate_with_generation(
            model=model,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            device=device,
            num_samples=num_samples,
            batch_size=batch_size,
            config=config,
            return_details=True  # This gets us the detailed results
        )

        # Clean up model
        del model
        torch.cuda.empty_cache()

        return {
            'metrics': convert_numpy_to_python(metrics),
            'detailed_results': detailed_results
        }

    else:
        # Use standard evaluation without detailed results
        num_samples = min(len(test_dataset), 10000)

        metrics = evaluate_with_generation(
            model=model,
            eval_dataset=test_dataset,
            tokenizer=tokenizer,
            device=device,
            num_samples=num_samples,
            batch_size=config.get('eval_batch_size', 32),
            config=config,
            return_details=False  # Just return aggregated metrics
        )

        # Clean up model
        del model
        torch.cuda.empty_cache()

        return {'metrics': convert_numpy_to_python(metrics)}


def plot_results(all_results: Dict[str, Dict], output_dir: Path, config=None):
    """Generate plots for evaluation results using the same function as training."""
    from transformers import TrainerState

    # Create a fake TrainerState with our evaluation results
    state = TrainerState()
    state.log_history = []

    # Convert our results to the format expected by save_training_plots
    for ckpt_name in sorted(all_results.keys(), key=lambda x: int(x.split('-')[1])):
        step = int(ckpt_name.split('-')[1])
        log_entry = {'step': step}
        log_entry.update(all_results[ckpt_name])
        state.log_history.append(log_entry)

    # Use the same plotting function as training - it will create task-specific plots
    save_training_plots(output_dir, state, config)


def main():
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model checkpoints on test dataset')
    parser.add_argument('config', help='Path to configuration YAML file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing output directory')

    args = parser.parse_args()

    config_path = args.config
    config = load_config(config_path)

    # Set random seed
    np.random.seed(config.get('seed', 42))
    torch.manual_seed(config.get('seed', 42))

    # Setup paths
    experiment_dir = Path(config['experiment_dir'])
    output_dir = Path(config['output_dir'])
    dataset_path = Path(config['dataset_path'])

    # Initialize output directory
    init_directory(output_dir, overwrite=args.overwrite)

    # Create subdirectory for evaluation data
    eval_data_dir = output_dir / 'eval_data'
    eval_data_dir.mkdir(parents=True, exist_ok=True)

    # Save config to root output directory
    with open(output_dir / 'eval_config.yaml', 'w') as f:
        yaml.dump(config, f)

    print(f"Evaluating experiment: {experiment_dir}")
    print(f"Output directory: {output_dir}")

    # Load tokenized dataset
    print(f"\nLoading dataset from {dataset_path}")
    dataset = load_from_disk(str(dataset_path))

    # Get the specified split (default to test for backward compatibility)
    split_name = config.get('split', 'test')

    if split_name in dataset:
        eval_dataset = dataset[split_name]
        print(f"  Using {split_name} split: {len(eval_dataset)} samples")
    else:
        raise ValueError(f"Split '{split_name}' not found in dataset. Available splits: {list(dataset.keys())}")

    # Initialize tokenizer from first checkpoint
    print("\nInitializing tokenizer")
    first_ckpt = experiment_dir / "checkpoints" / "checkpoint-0"
    tokenizer = AutoTokenizer.from_pretrained(first_ckpt)

    # Ensure pad token is set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Get checkpoints to evaluate
    checkpoint_paths = get_checkpoint_paths(experiment_dir, config.get('checkpoints', 'all'))
    print(f"\nFound {len(checkpoint_paths)} checkpoints to evaluate")

    # Check if we should save full results
    save_full = config.get('save_full_results', False)

    # Evaluate each checkpoint
    print("\nEvaluating checkpoints")
    all_results = {}
    all_detailed_results = {} if save_full else None

    for ckpt_path in tqdm(checkpoint_paths, desc="Checkpoints"):
        ckpt_name = ckpt_path.name
        print(f"\n  Processing {ckpt_name}")

        result = evaluate_checkpoint_with_details(
            ckpt_path,
            tokenizer,
            eval_dataset,
            config,
            config.get('device', 'cuda'),
            save_full=save_full
        )

        metrics = result['metrics']

        # Convert numpy types to Python native types for JSON serialization
        all_results[ckpt_name] = convert_numpy_to_python(metrics)

        # Store detailed results if available
        if save_full:
            detailed = result.get('detailed_results')
            if detailed:
                all_detailed_results[ckpt_name] = detailed

        # Print summary - metrics are in format: eval_{task}_metric_mean
        for key, value in metrics.items():
            if '_metric_mean' in key:
                print(f"    {key}: {value:.3f}")

    # Save aggregated results to eval_data
    results_path = eval_data_dir / 'evaluation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # Save detailed results to eval_data if available
    if save_full and all_detailed_results:
        for ckpt_name, detailed_results in all_detailed_results.items():
            detailed_path = eval_data_dir / f'detailed_results_{ckpt_name}.jsonl'
            with open(detailed_path, 'w') as f:
                for row in detailed_results:
                    # Convert any remaining numpy types
                    row_clean = convert_numpy_to_python(row)
                    f.write(json.dumps(row_clean) + '\n')
            print(f"  Saved detailed results for {ckpt_name} to {detailed_path}")

    # Generate plots (will create summary/ directory)
    print("\nGenerating plots")
    plot_results(all_results, output_dir, config)

    # Rename summary/ to dynamics/ for clarity
    summary_dir = output_dir / 'summary'
    dynamics_dir = output_dir / 'dynamics'
    if summary_dir.exists():
        summary_dir.rename(dynamics_dir)

    # Print final summary
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)

    # Find best checkpoint for each task based on mean metric
    metric_keys = [k for k in next(iter(all_results.values())).keys() if '_metric_mean' in k]

    for metric_key in metric_keys:
        best_ckpt = None
        best_value = None

        for ckpt_name, results in all_results.items():
            if metric_key in results:
                value = results[metric_key]
                # For distance and most metrics, lower is better (they're errors)
                if best_value is None or value < best_value:
                    best_value = value
                    best_ckpt = ckpt_name

        if best_ckpt:
            task_name = metric_key.replace('eval_', '').replace('_metric_mean', '')
            print(f"{task_name.upper()}: Best = {best_ckpt} (mean = {best_value:.3f})")

    print(f"\nResults saved to:")
    print(f"  Plots: {dynamics_dir}")
    print(f"  Data: {eval_data_dir}")


if __name__ == "__main__":
    main()