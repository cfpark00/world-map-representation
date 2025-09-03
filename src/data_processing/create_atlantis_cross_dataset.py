#!/usr/bin/env python3
"""
Create dataset with Atlantis cross-pairs for distance prediction.
This includes:
1. Inter-Atlantis pairs (Atlantis-Atlantis)
2. Cross pairs (Atlantis-World)
All pairs must include at least one Atlantis city.
"""
import pandas as pd
import numpy as np
import sys
sys.path.append('.')  # Add root to path
from src.utils import haversine, load_cities_csv
from datasets import Dataset, DatasetDict
import os
import argparse
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create Atlantis cross-pairs distance dataset')
parser.add_argument('output_dir', type=str, help='Output directory for the dataset')
parser.add_argument('--atlantis-csv', type=str, 
                   default='outputs/datasets/atlantis_XX0_100_seed42.csv',
                   help='Path to Atlantis CSV file')
parser.add_argument('--cities-csv', type=str, default=None, 
                   help='Path to world cities CSV file')
parser.add_argument('--n_train', type=int, default=100000, 
                   help='Number of training samples (default: 100000)')
parser.add_argument('--n_val', type=int, default=128, 
                   help='Number of validation samples (default: 128)')
parser.add_argument('--n_test', type=int, default=10000, 
                   help='Number of test samples (default: 10000)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--inter_ratio', type=float, default=0.1,
                   help='Ratio of inter-Atlantis pairs (default: 0.1)')
parser.add_argument('--offset', type=int, default=None, 
                   help='Offset for Atlantis city IDs (default: auto-detect from world cities)')

args = parser.parse_args()

n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
n_tot = n_train + n_val + n_test
output_dir = args.output_dir
seed = args.seed
inter_ratio = args.inter_ratio

# Load the world cities dataset
print("Loading world cities data...")
try:
    df_world = load_cities_csv(args.cities_csv)
except FileNotFoundError as e:
    print(f"Error: {e}")
    print("Please run: python src/data_processing/create_filtered_dataset.py 100000")
    print("Or specify path with --cities-csv")
    sys.exit(1)

# Auto-detect offset if not provided
if args.offset is None:
    # World cities count determines the offset
    offset = len(df_world)  # This will be 5075 (since row_id goes from 0 to 5074)
    print(f"Auto-detected offset: {offset} (continuing from world cities)")
else:
    offset = args.offset

# Load the Atlantis dataset
print("Loading Atlantis cities data...")
try:
    df_atlantis = pd.read_csv(args.atlantis_csv)
except FileNotFoundError as e:
    print(f"Error: Could not find Atlantis CSV at {args.atlantis_csv}")
    print("Please run: python src/data_processing/generate_atlantis.py")
    sys.exit(1)

# Apply offset to Atlantis row_ids to continue from world cities
df_atlantis['row_id'] = df_atlantis.index + offset

n_world = len(df_world)
n_atlantis = len(df_atlantis)

print(f"Number of world cities: {n_world:,}")
print(f"Number of Atlantis cities: {n_atlantis:,}")
print(f"Atlantis row ID range: {offset} to {offset + n_atlantis - 1}")
print(f"Requested pairs: {n_tot:,} (train: {n_train:,}, val: {n_val:,}, test: {n_test:,})")
print(f"Inter-Atlantis ratio: {inter_ratio:.1%}")

# Combine both dataframes for easier indexing
# World cities keep their original row_ids (0 to n_world-1)
# Atlantis cities have offset row_ids (offset to offset+n_atlantis-1)
df_combined = pd.concat([df_world, df_atlantis], ignore_index=True)

print(f"\nTotal combined cities: {len(df_combined):,}")

# Calculate number of inter-Atlantis and cross pairs needed
n_inter = int(n_tot * inter_ratio)
n_cross = n_tot - n_inter

print(f"\nTarget pair distribution:")
print(f"  Inter-Atlantis pairs: {n_inter:,} ({n_inter/n_tot:.1%})")
print(f"  Cross pairs (Atlantis-World): {n_cross:,} ({n_cross/n_tot:.1%})")

# Generate pairs
print(f"\nGenerating pairs (seed={seed})...")
np.random.seed(seed)

# 1. Generate inter-Atlantis pairs
n_max_inter = n_atlantis * (n_atlantis - 1) // 2
if n_inter > n_max_inter:
    print(f"Warning: Requested {n_inter} inter-Atlantis pairs but only {n_max_inter} unique pairs exist")
    n_inter = n_max_inter
    n_cross = n_tot - n_inter
    print(f"Adjusted - Inter: {n_inter:,}, Cross: {n_cross:,}")

# Get inter-Atlantis pair indices (in the combined dataframe)
atlantis_indices = np.arange(n_world, n_world + n_atlantis)
triu_indices_inter = np.triu_indices(n_atlantis, k=1)
inter_i = atlantis_indices[triu_indices_inter[0]]
inter_j = atlantis_indices[triu_indices_inter[1]]

# Sample inter-Atlantis pairs
selected_inter = np.random.choice(len(inter_i), size=min(n_inter, len(inter_i)), replace=False)
inter_pairs_i = inter_i[selected_inter]
inter_pairs_j = inter_j[selected_inter]

# 2. Generate cross pairs (Atlantis-World)
# Each Atlantis city can pair with each world city
n_max_cross = n_atlantis * n_world
if n_cross > n_max_cross:
    print(f"Warning: Requested {n_cross} cross pairs but only {n_max_cross} unique pairs exist")
    n_cross = n_max_cross

# Generate all possible cross pairs and sample
atlantis_idx_repeated = np.repeat(atlantis_indices, n_world)
world_idx_tiled = np.tile(np.arange(n_world), n_atlantis)

selected_cross = np.random.choice(len(atlantis_idx_repeated), size=n_cross, replace=False)
cross_pairs_atlantis = atlantis_idx_repeated[selected_cross]
cross_pairs_world = world_idx_tiled[selected_cross]

# Randomly swap positions for each pair (50% chance Atlantis first, 50% World first)
swap_mask = np.random.random(n_cross) < 0.5
cross_pairs_i = np.where(swap_mask, cross_pairs_world, cross_pairs_atlantis)
cross_pairs_j = np.where(swap_mask, cross_pairs_atlantis, cross_pairs_world)

# Combine all pairs
all_pairs_i = np.concatenate([inter_pairs_i, cross_pairs_i])
all_pairs_j = np.concatenate([inter_pairs_j, cross_pairs_j])

# Shuffle the combined pairs
shuffle_idx = np.random.permutation(len(all_pairs_i))
all_pairs_i = all_pairs_i[shuffle_idx]
all_pairs_j = all_pairs_j[shuffle_idx]

# Split into train, val, and test
actual_n_tot = len(all_pairs_i)
n_train_actual = int(actual_n_tot * (n_train / n_tot))
n_val_actual = int(actual_n_tot * (n_val / n_tot))
n_test_actual = actual_n_tot - n_train_actual - n_val_actual

train_i = all_pairs_i[:n_train_actual]
train_j = all_pairs_j[:n_train_actual]
val_i = all_pairs_i[n_train_actual:n_train_actual+n_val_actual]
val_j = all_pairs_j[n_train_actual:n_train_actual+n_val_actual]
test_i = all_pairs_i[n_train_actual+n_val_actual:]
test_j = all_pairs_j[n_train_actual+n_val_actual:]

# Function to create dataset from indices
def create_dataset_dict(indices_i, indices_j, df):
    """Create a dictionary suitable for HuggingFace Dataset using vectorized operations"""
    
    # Get coordinates in degrees for all city pairs at once
    lon1 = df.iloc[indices_i]['longitude'].values
    lat1 = df.iloc[indices_i]['latitude'].values
    lon2 = df.iloc[indices_j]['longitude'].values
    lat2 = df.iloc[indices_j]['latitude'].values
    
    # Use vectorized haversine from utils (handles degrees -> radians internally)
    distances_km = haversine(lon1, lat1, lon2, lat2)
    
    # Round to nearest km
    distances_km = np.round(distances_km).astype(int)
    
    # Get city IDs
    city1_ids = df.iloc[indices_i]['row_id'].values.astype(int)
    city2_ids = df.iloc[indices_j]['row_id'].values.astype(int)
    
    # Create text lists with progress bar
    text_list = []
    prompt_list = []
    completion_list = []
    
    for c1, c2, d in tqdm(zip(city1_ids, city2_ids, distances_km), 
                          total=len(city1_ids), 
                          desc="Formatting samples", 
                          leave=False):
        text_list.append(f"dist(c_{c1},c_{c2})={d}")
        prompt_list.append(f"<bos>dist(c_{c1},c_{c2})=")
        completion_list.append(f"{d}<eos>")
    
    return {
        'text': text_list,
        'prompt': prompt_list,
        'completion': completion_list
    }

# Create train, val, and test datasets
print("\nCreating train dataset...")
train_data = create_dataset_dict(train_i, train_j, df_combined)
train_dataset = Dataset.from_dict(train_data)

print("Creating validation dataset...")
val_data = create_dataset_dict(val_i, val_j, df_combined)
val_dataset = Dataset.from_dict(val_data)

print("Creating test dataset...")
test_data = create_dataset_dict(test_i, test_j, df_combined)
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

# Analyze the actual distribution of pairs
def analyze_pair_types(indices_i, indices_j, n_world, split_name):
    """Analyze the types of pairs in a split"""
    inter_atlantis = 0
    cross_pairs = 0
    
    for i, j in zip(indices_i, indices_j):
        if i >= n_world and j >= n_world:
            inter_atlantis += 1
        else:
            cross_pairs += 1
    
    total = len(indices_i)
    print(f"\n{split_name} pair distribution:")
    print(f"  Inter-Atlantis: {inter_atlantis:,} ({inter_atlantis/total:.1%})")
    print(f"  Cross (Atlantis-World): {cross_pairs:,} ({cross_pairs/total:.1%})")
    return inter_atlantis, cross_pairs

print("\n" + "="*60)
print("Dataset created successfully!")
print("="*60)

print(f"\nHuggingFace dataset: {output_dir}")
print(f"Train size: {len(train_dataset):,}")
print(f"Val size: {len(val_dataset):,}")
print(f"Test size: {len(test_dataset):,}")

# Analyze pair distributions
analyze_pair_types(train_i, train_j, n_world, "Train")
analyze_pair_types(val_i, val_j, n_world, "Validation")
analyze_pair_types(test_i, test_j, n_world, "Test")

# Print dataset info
print("\nDataset structure:")
print(dataset_dict)
print("\nFeatures:")
print(train_dataset.features)

# Show sample rows with type indication
print("\nSample train rows (with pair types):")
for i in range(min(10, len(train_dataset))):
    sample = train_dataset[i]
    text = sample['text']
    # Parse city IDs to determine type
    import re
    match = re.match(r'dist\(c_(\d+),c_(\d+)\)=(\d+)', text)
    if match:
        c1, c2, dist = map(int, match.groups())
        if c1 >= offset and c2 >= offset:
            pair_type = "[Atlantis-Atlantis]"
        elif c1 >= offset or c2 >= offset:
            pair_type = "[Atlantis-World]"
        else:
            pair_type = "[World-World]"  # Should not happen
        print(f"  {text} {pair_type}")

print("\nTo load this dataset:")
print(">>> from datasets import load_from_disk")
print(f">>> dataset = load_from_disk('{output_dir}')")
print(">>> train_data = dataset['train']")
print(">>> val_data = dataset['validation']")
print(">>> test_data = dataset['test']")

print("\nFormat: dist(c_X,c_Y)=Z")
print(f"  - World city IDs: 0 to {n_world-1}")
print(f"  - Atlantis city IDs: {offset} to {offset+n_atlantis-1}")
print("  - Z: haversine distance in km (rounded to nearest integer)")
print("  - All pairs include at least one Atlantis city")