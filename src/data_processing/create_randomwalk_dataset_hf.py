#!/usr/bin/env python3
import pandas as pd
import numpy as np
import sys
sys.path.append('.')  # Add root to path
from src.utils import haversine, load_cities_csv
from scipy.spatial import cKDTree
from datasets import Dataset, DatasetDict
import os
import argparse
import random
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create random walk dataset with train/val/test splits')
parser.add_argument('n_train', type=int, help='Number of training samples')
parser.add_argument('n_val', type=int, help='Number of validation samples')
parser.add_argument('n_test', type=int, help='Number of test samples')
parser.add_argument('output_dir', type=str, help='Output directory for the dataset')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')
parser.add_argument('--cities-csv', type=str, default=None, help='Path to cities CSV file')
parser.add_argument('--max-length', type=int, default=32, help='Maximum sequence length (default: 32)')
parser.add_argument('--distance-km', type=float, default=200.0, help='Distance threshold in km (default: 200)')
parser.add_argument('--visualize', type=int, default=0, help='Number of random sequences to visualize (default: 0, no visualization)')

args = parser.parse_args()

n_train = args.n_train
n_val = args.n_val
n_test = args.n_test
n_tot = n_train + n_val + n_test
output_dir = args.output_dir
seed = args.seed
max_length = args.max_length
distance_km = args.distance_km
n_visualize = args.visualize

# Load the 100k cities dataset
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
print(f"Requested sequences: {n_tot:,} (train: {n_train:,}, val: {n_val:,}, test: {n_test:,})")
print(f"Max sequence length: {max_length}")
print(f"Distance threshold: {distance_km} km")

# Build spatial index for efficient neighbor finding
print("\nBuilding spatial index...")
coordinates = df[['latitude', 'longitude']].values
coords_rad = np.radians(coordinates)
tree = cKDTree(coords_rad)

# Convert distance to radians
EARTH_RADIUS_KM = 6371.0
distance_threshold_rad = distance_km / EARTH_RADIUS_KM

# Set random seed
np.random.seed(seed)
random.seed(seed)

def generate_random_walk(start_idx, df, tree, coords_rad, max_len=32):
    """Generate a random walk starting from start_idx"""
    walk = [start_idx]
    current_idx = start_idx
    
    # Continue walk up to max_len-1 additional steps
    for _ in range(max_len - 1):
        # Find neighbors within distance threshold
        point_rad = coords_rad[current_idx]
        neighbor_indices = tree.query_ball_point(point_rad, distance_threshold_rad)
        
        # Remove current city and cities already in walk
        valid_neighbors = [idx for idx in neighbor_indices if idx not in walk]
        
        if not valid_neighbors:
            # No valid neighbors, end the walk
            break
        
        # Randomly select next city
        next_idx = random.choice(valid_neighbors)
        walk.append(next_idx)
        current_idx = next_idx
    
    return walk

def visualize_random_walks(dataset, df, tree, coords_rad, n_samples=10, output_dir='.'):
    """Visualize random walk sequences on a world map"""
    
    print(f"\nVisualizing {n_samples} random sequences...")
    
    # Create figure with larger size
    fig, ax = plt.subplots(figsize=(20, 10))
    
    # Plot all cities as small blue dots
    ax.scatter(df['longitude'], df['latitude'], s=1, c='lightblue', alpha=0.5, zorder=1)
    
    # Randomly select sequences to visualize
    total_sequences = len(dataset)
    sample_indices = np.random.choice(total_sequences, min(n_samples, total_sequences), replace=False)
    
    # Color palette for different sequences
    colors = plt.cm.Set1(np.linspace(0, 1, n_samples))
    
    sequences_info = []
    
    for idx, seq_idx in enumerate(sample_indices):
        text = dataset[int(seq_idx)]['text']
        
        # Parse the sequence
        city_ids_str = text.split('=')[1]
        city_ids = city_ids_str.split(',')
        
        # Get coordinates for each city in the sequence
        lons = []
        lats = []
        for city_id in city_ids:
            # Extract the numeric ID from c_XXXX format
            city_num = int(city_id.split('_')[1])
            # Find the city in the dataframe
            city_row = df[df['row_id'] == city_num].iloc[0]
            lons.append(city_row['longitude'])
            lats.append(city_row['latitude'])
        
        # Plot the sequence as a line with markers
        ax.plot(lons, lats, 'o-', color=colors[idx], linewidth=2, markersize=4, 
                alpha=0.7, label=f'Seq {idx+1} ({len(city_ids)} cities)', zorder=2)
        
        sequences_info.append(f"Sequence {idx+1}: {len(city_ids)} cities")
    
    # Add title and labels
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Random Walk Sequences (Distance Threshold: {int(distance_km)}km, Max Length: {max_length})', 
                 fontsize=14)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Set axis limits to show world map
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    
    # Save the figure
    viz_path = os.path.join(output_dir, 'random_walk_visualization.png')
    plt.tight_layout()
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {viz_path}")
    
    plt.close()

def create_dataset_dict(n_samples, df, tree, coords_rad):
    """Create a dictionary suitable for HuggingFace Dataset"""
    text_list = []
    prompt_list = []
    completion_list = []
    
    for _ in range(n_samples):
        # Randomly choose sequence length between 1 and max_length
        target_length = random.randint(1, max_length)
        
        # Randomly choose starting city
        start_idx = random.randint(0, n_cities - 1)
        
        # Generate random walk
        walk = generate_random_walk(start_idx, df, tree, coords_rad, max_len=target_length)
        
        # Convert to city IDs (no zero padding)
        city_ids = [f"c_{int(df.iloc[idx]['row_id'])}" for idx in walk]
        
        # Create text format: srd_200=c_XX,c_XX,... (no zero padding)
        full_text = f"srd_{int(distance_km)}=" + ",".join(city_ids)
        prompt = f"<bos>srd_{int(distance_km)}="
        completion = ",".join(city_ids) + "<eos>"
        
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
train_data = create_dataset_dict(n_train, df, tree, coords_rad)
train_dataset = Dataset.from_dict(train_data)

print("Creating validation dataset...")
val_data = create_dataset_dict(n_val, df, tree, coords_rad)
val_dataset = Dataset.from_dict(val_data)

print("Creating test dataset...")
test_data = create_dataset_dict(n_test, df, tree, coords_rad)
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

# Show sample rows with length info
print("\nSample train rows:")
for i in range(min(5, len(train_dataset))):
    text = train_dataset[i]['text']
    n_cities_in_seq = len(text.split('=')[1].split(','))
    print(f"  {text[:100]}... ({n_cities_in_seq} cities)")

print("\nSample val rows:")
for i in range(min(5, len(val_dataset))):
    text = val_dataset[i]['text']
    n_cities_in_seq = len(text.split('=')[1].split(','))
    print(f"  {text[:100]}... ({n_cities_in_seq} cities)")

print("\nSample test rows:")
for i in range(min(5, len(test_dataset))):
    text = test_dataset[i]['text']
    n_cities_in_seq = len(text.split('=')[1].split(','))
    print(f"  {text[:100]}... ({n_cities_in_seq} cities)")

# Calculate sequence length statistics
print("\nSequence length statistics:")
for split_name, dataset in [('Train', train_dataset), ('Val', val_dataset), ('Test', test_dataset)]:
    lengths = [len(text.split('=')[1].split(',')) for text in dataset['text']]
    print(f"{split_name}:")
    print(f"  Mean: {np.mean(lengths):.2f}")
    print(f"  Min: {np.min(lengths)}")
    print(f"  Max: {np.max(lengths)}")
    print(f"  Std: {np.std(lengths):.2f}")

print("\nTo load this dataset:")
print(">>> from datasets import load_from_disk")
print(f">>> dataset = load_from_disk('{output_dir}')")
print(">>> train_data = dataset['train']")
print(">>> val_data = dataset['validation']")
print(">>> test_data = dataset['test']")

# Visualize random sequences if requested
if n_visualize > 0:
    # Combine all datasets for visualization sampling
    all_data = {
        'text': train_data['text'] + val_data['text'] + test_data['text'],
        'prompt': train_data['prompt'] + val_data['prompt'] + test_data['prompt'],
        'completion': train_data['completion'] + val_data['completion'] + test_data['completion']
    }
    combined_dataset = Dataset.from_dict(all_data)
    visualize_random_walks(combined_dataset, df, tree, coords_rad, n_samples=n_visualize, output_dir=output_dir)