#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
sys.path.append('.')  # Add root to path
from src.utils import haversine, load_cities_csv
from datasets import Dataset, DatasetDict
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create distance dataset with train/val/test splits')
parser.add_argument('n_train', type=int, help='Number of training samples')
parser.add_argument('n_val', type=int, help='Number of validation samples')
parser.add_argument('n_test', type=int, help='Number of test samples')
parser.add_argument('output_dir', type=str, help='Output directory for the dataset')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--cities-csv', type=str, default=None, help='Path to cities CSV file')

args = parser.parse_args()

n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
n_tot = n_train + n_val + n_test
output_dir = args.output_dir
seed = args.seed

# Load the cities dataset
print("Loading cities data...")
try:
    df = load_cities_csv(args.cities_csv)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run: python src/data_processing/create_filtered_dataset.py 100000")
    print("Or specify path with --cities-csv")
    sys.exit(1)

n_cities = len(df)

print(f"Number of cities: {n_cities:,}")
print(f"Requested pairs: {n_tot:,} (train: {n_train:,}, val: {n_val:,}, test: {n_test:,})")

# Generate random pairs without replacement (only upper triangle, excluding diagonal)
print(f"\nGenerating random pairs (seed={seed})...")
np.random.seed(seed)

# Create upper triangle indices (excluding diagonal)
# This gives us n_cities * (n_cities - 1) / 2 unique unordered pairs
n_unique_pairs = n_cities * (n_cities - 1) // 2
print(f"Total unique unordered pairs: {n_unique_pairs:,}")
assert n_tot <= n_unique_pairs, f"Requested {n_tot} pairs but only {n_unique_pairs} unique unordered pairs exist"

triu_indices = np.triu_indices(n_cities, k=1)  # k=1 excludes diagonal

# Sample from the unique pairs
selected_pair_indices = np.random.choice(n_unique_pairs, size=n_tot, replace=False)
indices = (triu_indices[0][selected_pair_indices], triu_indices[1][selected_pair_indices])

# Split into train, val, and test
train_i = indices[0][:n_train]
train_j = indices[1][:n_train]
val_i = indices[0][n_train:n_train+n_val]
val_j = indices[1][n_train:n_train+n_val]
test_i = indices[0][n_train+n_val:]
test_j = indices[1][n_train+n_val:]

# Function to create dataset from indices
def create_dataset_dict(indices_i, indices_j, df):
    """Create a dictionary suitable for HuggingFace Dataset"""
    text_list = []
    prompt_list = []
    completion_list = []
    
    for i, j in zip(indices_i, indices_j):
        city1 = df.iloc[i]
        city2 = df.iloc[j]
        
        # Calculate geodesic distance and round to nearest km
        distance = round(haversine(
            city1['longitude'], city1['latitude'],
            city2['longitude'], city2['latitude']
        ))
        
        # Create text format: d(c_XX,c_XX)=XXXX (no zero padding)
        full_text = f"d(c_{int(city1['row_id'])},c_{int(city2['row_id'])})={distance}"
        prompt = f"<bos>d(c_{int(city1['row_id'])},c_{int(city2['row_id'])})="
        completion = f"{distance}<eos>"
        
        text_list.append(full_text)
        prompt_list.append(prompt)
        completion_list.append(completion)
    
    return {
        'text': text_list,
        'prompt': prompt_list,
        'completion': completion_list
    }

# Create train, val, and test datasets
print("\nCreating train dataset...")
train_data = create_dataset_dict(train_i, train_j, df)
train_dataset = Dataset.from_dict(train_data)

print("Creating validation dataset...")
val_data = create_dataset_dict(val_i, val_j, df)
val_dataset = Dataset.from_dict(val_data)

print("Creating test dataset...")
test_data = create_dataset_dict(test_i, test_j, df)
test_dataset = Dataset.from_dict(test_data)

# Combine into DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset,
    'test': test_dataset
})

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Save in HuggingFace format directly to output_dir
print(f"\nSaving HF dataset to {output_dir}...")
dataset_dict.save_to_disk(output_dir)

print(f"\nDataset created successfully!")
print(f"HuggingFace dataset: {output_dir}")
print(f"Train size: {len(train_dataset):,}")
print(f"Val size: {len(val_dataset):,}")
print(f"Test size: {len(test_dataset):,}")

# Print dataset info
print("\nDataset structure:")
print(dataset_dict)
print("\nFeatures:")
print(train_dataset.features)

# Show sample rows
print("\nSample train rows:")
for i in range(min(3, len(train_dataset))):
    print(f"  {train_dataset[i]['text']}")

print("\nSample val rows:")
for i in range(min(3, len(val_dataset))):
    print(f"  {val_dataset[i]['text']}")

print("\nSample test rows:")
for i in range(min(3, len(test_dataset))):
    print(f"  {test_dataset[i]['text']}")

print("\nTo load this dataset:")
print(">>> from datasets import load_from_disk")
print(f">>> dataset = load_from_disk('{output_dir}')")
print(">>> train_data = dataset['train']")
print(">>> val_data = dataset['validation']")
print(">>> test_data = dataset['test']")