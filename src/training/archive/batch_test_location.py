#!/usr/bin/env python3
"""
Batch testing script for location prediction models.
Tests a trained model on a dataset and provides detailed evaluation metrics.

Usage:
    python src/training/batch_test_location.py <checkpoint_path> <dataset_path> [--num_samples 100]
    
Example:
    python src/training/batch_test_location.py experiments/exp1/checkpoints/final outputs/datasets/loc_100kplus_all_42
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse
import json
from collections import defaultdict

# Add root to path for imports
sys.path.append('.')
from src.utils import haversine, parse_location

def load_model_and_tokenizer(checkpoint_path):
    """Load model and tokenizer from checkpoint"""
    print(f"Loading model from {checkpoint_path}")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path)
    
    # Set proper padding for generation
    tokenizer.padding_side = 'left'
    
    return model, tokenizer

def evaluate_batch(model, tokenizer, samples, device, max_new_tokens=20, temperature=0.0):
    """Evaluate a batch of samples and return predictions and metrics"""
    model.eval()
    
    results = []
    
    with torch.no_grad():
        for sample in tqdm(samples, desc="Evaluating samples"):
            prompt = sample['prompt']
            true_completion = sample['completion']
            
            # Tokenize prompt
            inputs = tokenizer(
                prompt,
                return_tensors='pt',
                add_special_tokens=False,
                padding=False,
                truncation=False
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=(temperature > 0),
                temperature=temperature if temperature > 0 else 1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Parse locations
            true_x, true_y = parse_location(true_completion)
            gen_x, gen_y = parse_location(generated)
            
            result = {
                'prompt': prompt,
                'true_completion': true_completion,
                'generated': generated,
                'true_x': true_x,
                'true_y': true_y,
                'gen_x': gen_x,
                'gen_y': gen_y,
                'parse_success': (true_x is not None and gen_x is not None)
            }
            
            # Calculate distance if both parsed successfully
            if result['parse_success']:
                true_lon = true_x / 100.0
                true_lat = true_y / 100.0
                gen_lon = gen_x / 100.0
                gen_lat = gen_y / 100.0
                result['haversine_distance'] = haversine(true_lon, true_lat, gen_lon, gen_lat)
            else:
                result['haversine_distance'] = None
            
            results.append(result)
    
    return results

def analyze_results(results):
    """Analyze evaluation results and compute metrics"""
    # Filter successful parses
    successful = [r for r in results if r['parse_success']]
    failed = [r for r in results if not r['parse_success']]
    
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS")
    print(f"{'='*60}")
    
    print(f"\nTotal samples: {len(results)}")
    print(f"Successfully parsed: {len(successful)} ({100*len(successful)/len(results):.1f}%)")
    print(f"Failed to parse: {len(failed)} ({100*len(failed)/len(results):.1f}%)")
    
    if successful:
        distances = [r['haversine_distance'] for r in successful]
        
        print(f"\nHaversine Distance Statistics (km):")
        print(f"  Mean: {np.mean(distances):.2f}")
        print(f"  Median: {np.median(distances):.2f}")
        print(f"  Std: {np.std(distances):.2f}")
        print(f"  Min: {np.min(distances):.2f}")
        print(f"  Max: {np.max(distances):.2f}")
        print(f"  25th percentile: {np.percentile(distances, 25):.2f}")
        print(f"  75th percentile: {np.percentile(distances, 75):.2f}")
        
        # Accuracy thresholds
        thresholds = [10, 50, 100, 500, 1000, 5000]
        print(f"\nAccuracy at different thresholds:")
        for threshold in thresholds:
            accurate = sum(1 for d in distances if d <= threshold)
            print(f"  Within {threshold:4d} km: {100*accurate/len(distances):5.1f}% ({accurate}/{len(distances)})")
    
    # Show some failed examples
    if failed:
        print(f"\nExample failed generations (first 5):")
        for i, r in enumerate(failed[:5]):
            print(f"\n  {i+1}. Prompt: {r['prompt'][-30:]}")
            print(f"     Expected: {r['true_completion']}")
            print(f"     Generated: {r['generated'][-30:]}")
    
    return successful, failed

def plot_results(successful_results, output_dir):
    """Create visualization plots"""
    if not successful_results:
        print("No successful results to plot")
        return
    
    distances = [r['haversine_distance'] for r in successful_results]
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. Distance histogram
    ax = axes[0, 0]
    ax.hist(distances, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(distances), color='red', linestyle='--', label=f'Mean: {np.mean(distances):.0f} km')
    ax.axvline(np.median(distances), color='green', linestyle='--', label=f'Median: {np.median(distances):.0f} km')
    ax.set_xlabel('Haversine Distance (km)')
    ax.set_ylabel('Count')
    ax.set_title('Distance Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. Log-scale histogram
    ax = axes[0, 1]
    # Filter out zeros for log scale
    distances_nonzero = [d for d in distances if d > 0]
    if distances_nonzero:
        ax.hist(distances_nonzero, bins=50, edgecolor='black', alpha=0.7)
        ax.set_yscale('log')
        ax.set_xlabel('Haversine Distance (km)')
        ax.set_ylabel('Count (log scale)')
        ax.set_title('Distance Distribution (Log Scale)')
        ax.grid(True, alpha=0.3)
    
    # 3. Cumulative distribution
    ax = axes[1, 0]
    sorted_distances = np.sort(distances)
    cumulative = np.arange(1, len(sorted_distances) + 1) / len(sorted_distances) * 100
    ax.plot(sorted_distances, cumulative, linewidth=2)
    ax.set_xlabel('Haversine Distance (km)')
    ax.set_ylabel('Cumulative Percentage (%)')
    ax.set_title('Cumulative Distance Distribution')
    ax.grid(True, alpha=0.3)
    # Add reference lines
    for threshold in [100, 500, 1000]:
        pct = sum(1 for d in distances if d <= threshold) / len(distances) * 100
        ax.axvline(threshold, color='gray', linestyle=':', alpha=0.5)
        ax.text(threshold, pct + 2, f'{threshold}km\n({pct:.0f}%)', ha='center', fontsize=8)
    
    # 4. Scatter plot of true vs generated coordinates
    ax = axes[1, 1]
    true_xs = [r['true_x'] for r in successful_results]
    true_ys = [r['true_y'] for r in successful_results]
    gen_xs = [r['gen_x'] for r in successful_results]
    gen_ys = [r['gen_y'] for r in successful_results]
    
    # Sample for visibility if too many points
    if len(true_xs) > 1000:
        indices = np.random.choice(len(true_xs), 1000, replace=False)
        true_xs = [true_xs[i] for i in indices]
        true_ys = [true_ys[i] for i in indices]
        gen_xs = [gen_xs[i] for i in indices]
        gen_ys = [gen_ys[i] for i in indices]
        sample_distances = [distances[i] for i in indices]
    else:
        sample_distances = distances
    
    scatter = ax.scatter(true_xs, gen_xs, c=sample_distances, cmap='viridis', 
                        alpha=0.5, s=10, vmin=0, vmax=np.percentile(distances, 95))
    ax.plot([min(true_xs), max(true_xs)], [min(true_xs), max(true_xs)], 
            'r--', alpha=0.5, label='Perfect prediction')
    ax.set_xlabel('True X Coordinate')
    ax.set_ylabel('Generated X Coordinate')
    ax.set_title('True vs Generated X Coordinates')
    ax.legend()
    plt.colorbar(scatter, ax=ax, label='Distance (km)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'batch_test_results.png', dpi=150)
    plt.close()
    
    print(f"\nPlots saved to {output_dir / 'batch_test_results.png'}")

def main():
    parser = argparse.ArgumentParser(description='Batch test location prediction model')
    parser.add_argument('checkpoint_path', type=str, help='Path to model checkpoint')
    parser.add_argument('dataset_path', type=str, help='Path to dataset')
    parser.add_argument('--num_samples', type=int, default=None, 
                       help='Number of samples to test (default: all)')
    parser.add_argument('--split', type=str, default='validation',
                       choices=['train', 'validation', 'test'],
                       help='Dataset split to use (default: validation)')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='Generation temperature (0 for greedy, >0 for sampling)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results (default: checkpoint directory)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup paths
    checkpoint_path = Path(args.checkpoint_path)
    dataset_path = Path(args.dataset_path)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        sys.exit(1)
    
    if not dataset_path.exists():
        print(f"Error: Dataset not found at {dataset_path}")
        sys.exit(1)
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_path.parent.parent  # Go up to experiment dir
    output_dir.mkdir(exist_ok=True)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(checkpoint_path)
    model.to(device)
    model.eval()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Load dataset
    print(f"\nLoading dataset from {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # Get the appropriate split
    if args.split in dataset:
        data = dataset[args.split]
    elif 'train' in dataset:
        print(f"Warning: Split '{args.split}' not found, using 'train'")
        data = dataset['train']
    else:
        data = dataset
    
    print(f"Dataset size: {len(data)} samples")
    
    # Sample if requested
    if args.num_samples and args.num_samples < len(data):
        indices = np.random.choice(len(data), args.num_samples, replace=False)
        samples = [data[int(i)] for i in indices]
        print(f"Testing on {len(samples)} randomly sampled examples")
    else:
        samples = [data[i] for i in range(len(data))]
        print(f"Testing on all {len(samples)} examples")
    
    # Run evaluation
    print(f"\nRunning evaluation (temperature={args.temperature})...")
    results = evaluate_batch(model, tokenizer, samples, device, temperature=args.temperature)
    
    # Analyze results
    successful, failed = analyze_results(results)
    
    # Save results
    results_file = output_dir / f'batch_test_results_{args.split}.json'
    with open(results_file, 'w') as f:
        json.dump({
            'checkpoint': str(checkpoint_path),
            'dataset': str(dataset_path),
            'split': args.split,
            'num_samples': len(samples),
            'temperature': args.temperature,
            'successful_count': len(successful),
            'failed_count': len(failed),
            'parse_rate': len(successful) / len(results) if results else 0,
            'mean_distance': np.mean([r['haversine_distance'] for r in successful]) if successful else None,
            'median_distance': np.median([r['haversine_distance'] for r in successful]) if successful else None,
            'std_distance': np.std([r['haversine_distance'] for r in successful]) if successful else None,
        }, f, indent=2)
    print(f"\nResults saved to {results_file}")
    
    # Create plots
    if successful:
        plot_results(successful, output_dir)
    
    # Show some example predictions
    print(f"\n{'='*60}")
    print("EXAMPLE PREDICTIONS")
    print(f"{'='*60}")
    
    # Show best, worst, and median examples
    if successful:
        sorted_by_distance = sorted(successful, key=lambda x: x['haversine_distance'])
        
        examples = [
            ("Best", sorted_by_distance[0]),
            ("Median", sorted_by_distance[len(sorted_by_distance)//2]),
            ("Worst", sorted_by_distance[-1])
        ]
        
        for label, result in examples:
            print(f"\n{label} prediction (distance: {result['haversine_distance']:.2f} km):")
            print(f"  Prompt: {result['prompt']}")
            print(f"  True: {result['true_completion']}")
            print(f"  Generated: {result['generated'][len(result['prompt']):]}")
            print(f"  Coordinates: ({result['true_x']}, {result['true_y']}) -> ({result['gen_x']}, {result['gen_y']})")

if __name__ == "__main__":
    main()