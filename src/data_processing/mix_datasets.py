#!/usr/bin/env python3
"""
Mix multiple datasets by sampling specified number of rows from each.
Useful for creating mixed training datasets to prevent catastrophic forgetting.
"""
import argparse
from pathlib import Path
from datasets import load_from_disk, DatasetDict, concatenate_datasets
import numpy as np
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description='Mix multiple HuggingFace datasets by sampling from each')
    parser.add_argument('output_dir', type=str, help='Output directory for mixed dataset')
    parser.add_argument('--datasets', nargs='+', type=str, required=True,
                       help='List of dataset paths to mix')
    parser.add_argument('--samples', nargs='+', type=int, required=True,
                       help='Number of samples to take from each dataset (same order as --datasets)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for sampling')
    parser.add_argument('--shuffle', action='store_true', help='Shuffle the final mixed dataset')
    
    args = parser.parse_args()
    
    # Validate inputs
    if len(args.datasets) != len(args.samples):
        raise ValueError(f"Number of datasets ({len(args.datasets)}) must match number of samples ({len(args.samples)})")
    
    # Set seed for reproducibility
    np.random.seed(args.seed)
    
    print(f"Mixing {len(args.datasets)} datasets:")
    for dataset_path, n_samples in zip(args.datasets, args.samples):
        print(f"  - {dataset_path}: {n_samples:,} samples")
    
    # Load and sample from each dataset
    mixed_train = []
    mixed_val = []
    mixed_test = []
    
    for dataset_path, n_samples in zip(args.datasets, args.samples):
        print(f"\nLoading {dataset_path}...")
        dataset = load_from_disk(dataset_path)
        
        # Get dataset info
        train_size = len(dataset['train'])
        val_size = len(dataset['validation']) if 'validation' in dataset else 0
        test_size = len(dataset['test']) if 'test' in dataset else 0
        
        print(f"  Original sizes - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")
        
        # Sample from train split
        if n_samples > train_size:
            print(f"  WARNING: Requested {n_samples:,} samples but only {train_size:,} available in train split")
            n_samples = train_size
        
        # Sample indices
        train_indices = np.random.choice(train_size, size=n_samples, replace=False)
        sampled_train = dataset['train'].select(train_indices)
        mixed_train.append(sampled_train)
        
        # For validation and test, take proportional samples
        if 'validation' in dataset and val_size > 0:
            val_ratio = n_samples / train_size
            n_val_samples = min(int(val_size * val_ratio), val_size)
            val_indices = np.random.choice(val_size, size=n_val_samples, replace=False)
            sampled_val = dataset['validation'].select(val_indices)
            mixed_val.append(sampled_val)
            print(f"  Sampled {n_val_samples:,} validation samples")
        
        if 'test' in dataset and test_size > 0:
            test_ratio = n_samples / train_size
            n_test_samples = min(int(test_size * test_ratio), test_size)
            test_indices = np.random.choice(test_size, size=n_test_samples, replace=False)
            sampled_test = dataset['test'].select(test_indices)
            mixed_test.append(sampled_test)
            print(f"  Sampled {n_test_samples:,} test samples")
        
        print(f"  Sampled {n_samples:,} train samples")
    
    # Concatenate all samples
    print("\nConcatenating datasets...")
    final_train = concatenate_datasets(mixed_train)
    
    dataset_dict = {'train': final_train}
    
    if mixed_val:
        final_val = concatenate_datasets(mixed_val)
        dataset_dict['validation'] = final_val
    
    if mixed_test:
        final_test = concatenate_datasets(mixed_test)
        dataset_dict['test'] = final_test
    
    # Shuffle if requested
    if args.shuffle:
        print("Shuffling datasets...")
        for split in dataset_dict:
            dataset_dict[split] = dataset_dict[split].shuffle(seed=args.seed)
    
    # Create DatasetDict
    mixed_dataset = DatasetDict(dataset_dict)
    
    # Save the mixed dataset
    output_path = Path(args.output_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving mixed dataset to {output_path}...")
    mixed_dataset.save_to_disk(str(output_path))
    
    # Print final statistics
    print("\nMixed dataset created successfully!")
    print(f"Output: {output_path}")
    print("\nFinal dataset sizes:")
    for split, data in mixed_dataset.items():
        print(f"  {split}: {len(data):,} samples")
    
    # Show sample examples from train
    print("\nSample train examples:")
    for i in range(min(5, len(mixed_dataset['train']))):
        example = mixed_dataset['train'][i]
        text = example.get('text', example.get('prompt', ''))
        # Truncate long texts
        if len(text) > 100:
            text = text[:97] + "..."
        print(f"  {text}")
    
    print("\nDataset structure:")
    print(mixed_dataset)
    
    print(f"\nTo load this dataset:")
    print(f">>> from datasets import load_from_disk")
    print(f">>> dataset = load_from_disk('{output_path}')")

if __name__ == "__main__":
    main()