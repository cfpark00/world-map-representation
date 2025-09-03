#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
import argparse
sys.path.append('.')  # Add root to path

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create world city map')
parser.add_argument('--csv', type=str, default='outputs/datasets/cities_100k_plus_seed42.csv',
                   help='Path to city CSV file (default: outputs/datasets/cities_100k_plus_seed42.csv)')
args = parser.parse_args()

# Read the filtered CSV file directly
print(f"Loading city data from {args.csv}...")
large_cities = pd.read_csv(args.csv)

city_count = len(large_cities)
print(f"Found {city_count:,} cities in dataset")

# Create equirectangular projection plot
fig, ax = plt.subplots(figsize=(20, 12))

# Plot world map outline (simple box for equirectangular)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Plot cities as dots using x,y coordinates
ax.scatter(large_cities['x'], large_cities['y'], 
           s=1, c='red', alpha=0.6, marker='.')

# Add labels and title with padding
ax.set_xlabel('X (Longitude)', fontsize=20, labelpad=10)
ax.set_ylabel('Y (Latitude)', fontsize=20, labelpad=10)
ax.set_title(f'World Cities (n={city_count:,}) - Cartesian Coordinates', fontsize=24, pad=20)

# Add axis ticks with larger font
ax.set_xticks(np.arange(-180, 181, 30))
ax.set_yticks(np.arange(-90, 91, 30))
ax.tick_params(axis='both', labelsize=18)

# Make the plot more map-like
ax.set_aspect('equal')
ax.set_facecolor('#f0f8ff')  # Light blue background for oceans

# Save the plot
output_filename = 'outputs/figures/world_cities_cartesian.png'
# Create output directory if it doesn't exist
import os
os.makedirs('outputs/figures', exist_ok=True)
plt.tight_layout()
plt.savefig(output_filename, dpi=150, bbox_inches='tight')
print(f"Map saved as '{output_filename}'")

# Also show the plot
plt.show()

# Print some statistics
print(f"\nStatistics:")
print(f"Total cities plotted: {len(large_cities)}")
print(f"X range: [{large_cities['x'].min():.2f}, {large_cities['x'].max():.2f}]")
print(f"Y range: [{large_cities['y'].min():.2f}, {large_cities['y'].max():.2f}]")
print(f"Sample cities:")
for i in range(min(5, len(large_cities))):
    row = large_cities.iloc[i]
    print(f"  {row['asciiname']}: ({row['x']:.2f}, {row['y']:.2f}) - ID: {row['city_id']}")