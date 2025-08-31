#!/usr/bin/env python3
import pandas as pd
import numpy as np
from math import radians, floor
from datasets import Dataset, DatasetDict
import os
import argparse
import sys
sys.path.append('.')  # Add root to path
from src.utils import load_cities_csv
from tqdm import tqdm

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create location dataset with train and optional validation split')
parser.add_argument('n_train', type=int, nargs='?', help='Number of training samples')
parser.add_argument('output_dir', type=str, help='Output directory for the dataset')
parser.add_argument('--n_val', type=int, default=0, help='Number of validation samples (default: 0, no validation set)')
parser.add_argument('--all', action='store_true', help='Use all cities (creates one sample per city)')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--cities-csv', type=str, default=None, help='Path to cities CSV file')

args = parser.parse_args()

if not args.all and args.n_train is None:
    parser.error("n_train is required unless --all is specified")

n_train = args.n_train if not args.all else None
n_val = args.n_val
output_dir = args.output_dir
seed = args.seed
use_all = args.all

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

if use_all:
    print(f"Using all cities (setting n_train={n_cities})")
    n_train = n_cities

print(f"Requested samples: train={n_train:,}, val={n_val:,}")

# Generate random city indices without replacement
print(f"\nGenerating random city indices (seed={seed})...")
np.random.seed(seed)

# Sample cities without replacement for both train and val
train_indices = np.random.choice(n_cities, size=n_train, replace=False)
val_indices = np.random.choice(n_cities, size=n_val, replace=False) if n_val > 0 else []

# Function to create dataset from indices
def create_dataset_dict(city_indices, df):
    """Create a dictionary suitable for HuggingFace Dataset"""
    text_list = []
    prompt_list = []
    completion_list = []
    
    for idx in tqdm(city_indices, desc="Formatting samples", leave=False):
        city = df.iloc[idx]
        
        # Convert longitude from degrees to radians and scale by 1000
        # Longitude ranges from -180 to 180 degrees (-pi to pi radians)
        # After conversion to 0 to 2pi range: add pi, then scale
        lon_rad = radians(city['longitude']) + np.pi  # Now 0 to 2pi
        lon_scaled = floor(lon_rad * 1000)  # 0 to ~6283
        
        # Convert latitude from degrees to radians and scale by 1000
        # Latitude ranges from -90 to 90 degrees (-pi/2 to pi/2 radians)
        # After conversion to 0 to pi range: add pi/2, then scale
        lat_rad = radians(city['latitude']) + np.pi/2  # Now 0 to pi
        lat_scaled = floor(lat_rad * 1000)  # 0 to ~3141
        
        # Create text format: loc(c_XX)=XXXX,YYYY (no zero padding)
        city_id = int(city['row_id'])
        full_text = f"loc(c_{city_id})={lon_scaled},{lat_scaled}"
        prompt = f"<bos>loc(c_{city_id})="
        completion = f"{lon_scaled},{lat_scaled}<eos>"
        
        text_list.append(full_text)
        prompt_list.append(prompt)
        completion_list.append(completion)
    
    return {
        'text': text_list,
        'prompt': prompt_list,
        'completion': completion_list
    }

# Create train dataset
print("\nCreating train dataset...")
train_data = create_dataset_dict(train_indices, df)
train_dataset = Dataset.from_dict(train_data)

# Create validation dataset if requested
if n_val > 0:
    print(f"Creating validation dataset...")
    val_data = create_dataset_dict(val_indices, df)
    val_dataset = Dataset.from_dict(val_data)
    
    # Create DatasetDict with train and validation splits
    dataset_dict = DatasetDict({
        'train': train_dataset,
        'validation': val_dataset
    })
else:
    # Create DatasetDict with only train split
    dataset_dict = DatasetDict({
        'train': train_dataset
    })

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Save in HuggingFace format directly to output_dir
print(f"\nSaving HF dataset to {output_dir}...")
dataset_dict.save_to_disk(output_dir)

print(f"\nDataset created successfully!")
print(f"HuggingFace dataset: {output_dir}")
print(f"Train size: {len(train_dataset):,}")
if n_val > 0:
    print(f"Validation size: {len(val_dataset):,}")

# Print dataset info
print("\nDataset structure:")
print(dataset_dict)
print("\nFeatures:")
print(train_dataset.features)

# Show sample rows with explanation
print("\nSample train rows:")
for i in range(min(10, len(train_dataset))):
    sample = train_dataset[i]
    print(f"  {sample['text']}")

if n_val > 0:
    print("\nSample validation rows:")
    for i in range(min(5, len(val_dataset))):
        sample = val_dataset[i]
        print(f"  {sample['text']}")

print("\nTo load this dataset:")
print(">>> from datasets import load_from_disk")
print(f">>> dataset = load_from_disk('{output_dir}')")
print(">>> train_data = dataset['train']")
if n_val > 0:
    print(">>> val_data = dataset['validation']")

print("\nFormat: loc(c_XX)=XXXX,YYYY")
print("  - c_XX: city ID (no zero padding)")
print("  - XXXX: floor(1000 * longitude in radians), range 0-6283")
print("  - YYYY: floor(1000 * latitude in radians), range 0-3141")