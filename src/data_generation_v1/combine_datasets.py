#!/usr/bin/env python3
"""
Combine multiple datasets based on YAML configuration.
Supports full concatenation or sampling from each dataset.
"""
import argparse
import yaml
import numpy as np
import pandas as pd
import shutil
from pathlib import Path
from datasets import load_from_disk, DatasetDict, concatenate_datasets
from tqdm import tqdm
import json
import sys
sys.path.append('.')
from src.utils import init_directory

def load_config(config_path):
    """Load and validate YAML configuration."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required fields
    if 'output_dir' not in config:
        raise ValueError("Config must specify 'output_dir'")
    if 'datasets' not in config:
        raise ValueError("Config must specify 'datasets'")
    
    # Set defaults
    config.setdefault('seed', 42)
    config.setdefault('shuffle', False)
    config.setdefault('mode', 'concat')  # 'concat' or 'sample'
    
    return config

def combine_datasets(config, debug=False):
    """Combine datasets according to configuration."""
    np.random.seed(config['seed'])

    if debug:
        print("[DEBUG MODE] Limiting to 100 samples per split")
    
    mode = config['mode']
    print(f"Combination mode: {mode}")
    print(f"Output directory: {config['output_dir']}")
    
    # Collect datasets for each split
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    total_train = 0
    total_val = 0
    total_test = 0
    
    for dataset_config in config['datasets']:
        dataset_path = dataset_config['path']
        print(f"\nLoading dataset: {dataset_path}")
        
        # Load dataset
        dataset = load_from_disk(dataset_path)
        
        # Get original sizes
        train_size = len(dataset['train']) if 'train' in dataset else 0
        val_size = len(dataset['validation']) if 'validation' in dataset else 0
        test_size = len(dataset['test']) if 'test' in dataset else 0
        
        print(f"  Original sizes - Train: {train_size:,}, Val: {val_size:,}, Test: {test_size:,}")
        
        if mode == 'concat':
            # Full concatenation mode
            if 'train' in dataset:
                train_datasets.append(dataset['train'])
                total_train += train_size
                print(f"  Adding all {train_size:,} train samples")
            
            if 'validation' in dataset:
                val_datasets.append(dataset['validation'])
                total_val += val_size
                print(f"  Adding all {val_size:,} validation samples")
            
            if 'test' in dataset:
                test_datasets.append(dataset['test'])
                total_test += test_size
                print(f"  Adding all {test_size:,} test samples")
                
        elif mode == 'sample':
            # Sampling mode - take specified number of samples
            n_samples = dataset_config.get('n_samples', None)
            ratio = dataset_config.get('ratio', None)
            
            if n_samples is None and ratio is None:
                raise ValueError(f"Dataset {dataset_path} must specify either 'n_samples' or 'ratio' in sample mode")
            
            # Process train split
            if 'train' in dataset:
                if ratio is not None:
                    n_train_samples = int(train_size * ratio)
                else:
                    n_train_samples = min(n_samples, train_size)
                
                if n_train_samples > train_size:
                    print(f"  WARNING: Requested {n_train_samples:,} train samples but only {train_size:,} available")
                    n_train_samples = train_size
                
                if n_train_samples > 0:
                    indices = np.random.choice(train_size, size=n_train_samples, replace=False)
                    sampled_train = dataset['train'].select(indices)
                    train_datasets.append(sampled_train)
                    total_train += n_train_samples
                    print(f"  Sampled {n_train_samples:,} train samples")
            
            # Process validation split (proportional to train sampling)
            if 'validation' in dataset and val_size > 0:
                if ratio is not None:
                    n_val_samples = int(val_size * ratio)
                else:
                    # Proportional to train sampling
                    train_ratio = n_train_samples / train_size if train_size > 0 else 0
                    n_val_samples = int(val_size * train_ratio)
                
                n_val_samples = min(n_val_samples, val_size)
                
                if n_val_samples > 0:
                    indices = np.random.choice(val_size, size=n_val_samples, replace=False)
                    sampled_val = dataset['validation'].select(indices)
                    val_datasets.append(sampled_val)
                    total_val += n_val_samples
                    print(f"  Sampled {n_val_samples:,} validation samples")
            
            # Process test split (proportional to train sampling)
            if 'test' in dataset and test_size > 0:
                if ratio is not None:
                    n_test_samples = int(test_size * ratio)
                else:
                    # Proportional to train sampling
                    train_ratio = n_train_samples / train_size if train_size > 0 else 0
                    n_test_samples = int(test_size * train_ratio)
                
                n_test_samples = min(n_test_samples, test_size)
                
                if n_test_samples > 0:
                    indices = np.random.choice(test_size, size=n_test_samples, replace=False)
                    sampled_test = dataset['test'].select(indices)
                    test_datasets.append(sampled_test)
                    total_test += n_test_samples
                    print(f"  Sampled {n_test_samples:,} test samples")
    
    # Concatenate all collected datasets
    print("\nCombining datasets...")
    
    combined_dict = {}
    
    if train_datasets:
        print(f"  Concatenating {len(train_datasets)} train datasets...")
        combined_train = concatenate_datasets(train_datasets)
        if config['shuffle']:
            print(f"  Shuffling train split...")
            combined_train = combined_train.shuffle(seed=config['seed'])
        combined_dict['train'] = combined_train
    
    if val_datasets:
        print(f"  Concatenating {len(val_datasets)} validation datasets...")
        combined_val = concatenate_datasets(val_datasets)
        if config['shuffle']:
            print(f"  Shuffling validation split...")
            combined_val = combined_val.shuffle(seed=config['seed'])
        combined_dict['validation'] = combined_val
    
    if test_datasets:
        print(f"  Concatenating {len(test_datasets)} test datasets...")
        combined_test = concatenate_datasets(test_datasets)
        if config['shuffle']:
            print(f"  Shuffling test split...")
            combined_test = combined_test.shuffle(seed=config['seed'])
        combined_dict['test'] = combined_test
    
    # Create DatasetDict
    combined_dataset = DatasetDict(combined_dict)

    # Debug mode: limit each split to 100 samples
    if debug:
        debug_dict = {}
        for split_name, split_data in combined_dataset.items():
            if len(split_data) > 100:
                debug_dict[split_name] = split_data.select(range(100))
                print(f"  [DEBUG] Limited {split_name} from {len(split_data):,} to 100")
            else:
                debug_dict[split_name] = split_data
        combined_dataset = DatasetDict(debug_dict)
        total_train = len(combined_dataset.get('train', []))
        total_val = len(combined_dataset.get('validation', []))
        total_test = len(combined_dataset.get('test', []))

    print(f"\nFinal combined dataset sizes:")
    print(f"  Train: {total_train:,}")
    print(f"  Validation: {total_val:,}")
    print(f"  Test: {total_test:,}")
    
    return combined_dataset

def main():
    parser = argparse.ArgumentParser(description='Combine datasets based on YAML configuration')
    parser.add_argument('config', type=str, help='Path to YAML configuration file')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output directory if it exists')
    parser.add_argument('--debug', action='store_true', help='Debug mode: limit to 100 samples per split')

    args = parser.parse_args()
    
    # Load configuration
    print(f"Loading configuration from {args.config}")
    config = load_config(args.config)
    
    # Combine datasets
    combined_dataset = combine_datasets(config, debug=args.debug)
    
    # Initialize output directory
    output_path = init_directory(config['output_dir'], overwrite=args.overwrite)
    
    # Save combined dataset
    print(f"\nSaving combined dataset to {output_path}...")
    combined_dataset.save_to_disk(str(output_path))
    
    # Save metadata
    metadata = {
        'config_file': args.config,
        'config': config,
        'created': pd.Timestamp.now(tz='UTC').isoformat(),
        'mode': config['mode'],
        'n_train': len(combined_dataset.get('train', [])),
        'n_val': len(combined_dataset.get('validation', [])),
        'n_test': len(combined_dataset.get('test', [])),
        'seed': config['seed']
    }
    
    metadata_path = output_path / 'metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Copy config file to output directory
    config_copy_path = output_path / 'combine_config.yaml'
    shutil.copy(args.config, config_copy_path)
    
    print(f"\nSaved files:")
    print(f"  - HuggingFace dataset files")
    print(f"  - metadata.json: Dataset metadata")
    print(f"  - combine_config.yaml: Configuration used")
    
    # Display sample rows
    print("\nSample train rows (first 5):")
    if 'train' in combined_dataset:
        for i in range(min(5, len(combined_dataset['train']))):
            text = combined_dataset['train'][i].get('text', '')
            if len(text) > 100:
                text = text[:97] + "..."
            print(f"  {text}")
    
    print("\nDataset combination complete!")
    print(f"Output: {output_path}")

if __name__ == "__main__":
    main()