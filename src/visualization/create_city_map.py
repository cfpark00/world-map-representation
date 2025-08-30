#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
sys.path.append('.')  # Add root to path
from src.utils import extract_coordinates

# Get population threshold from command line argument
if len(sys.argv) > 1:
    pop_threshold = int(sys.argv[1])
else:
    pop_threshold = 100000  # Default value

# Read the CSV file
print("Loading city data...")
df = pd.read_csv('/n/home12/cfpark00/WM_1/data/geonames-all-cities-with-a-population-1000.csv', 
                 sep=';', encoding='utf-8-sig')

# Filter cities with population > threshold
print(f"Filtering cities with population > {pop_threshold:,}...")
large_cities = df[df['Population'] > pop_threshold].copy()

# Extract coordinates (last column contains "lat, lon")
print("Extracting coordinates...")
large_cities = extract_coordinates(large_cities)

city_count = len(large_cities)
print(f"Found {city_count:,} cities with population > {pop_threshold:,}")

# Create equirectangular projection plot
fig, ax = plt.subplots(figsize=(20, 10))

# Plot world map outline (simple box for equirectangular)
ax.set_xlim(-180, 180)
ax.set_ylim(-90, 90)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--')

# Plot cities as dots
ax.scatter(large_cities['longitude'], large_cities['latitude'], 
           s=1, c='red', alpha=0.6, marker='.')

# Add labels and title
ax.set_xlabel('Longitude', fontsize=12)
ax.set_ylabel('Latitude', fontsize=12)
ax.set_title(f'World Cities with Population > {pop_threshold:,} (n={city_count:,}) - Equirectangular Projection', fontsize=14)

# Add axis ticks
ax.set_xticks(np.arange(-180, 181, 30))
ax.set_yticks(np.arange(-90, 91, 30))

# Make the plot more map-like
ax.set_aspect('equal')
ax.set_facecolor('#f0f8ff')  # Light blue background for oceans

# Save the plot with threshold in filename
output_filename = f'outputs/figures/world_cities_pop_{pop_threshold}.png'
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
print(f"Population range: {large_cities['Population'].min():,} - {large_cities['Population'].max():,}")
print(f"Top 5 cities by population:")
top_cities = large_cities.nlargest(5, 'Population')[['Name', 'Country name EN', 'Population']]
for idx, row in top_cities.iterrows():
    print(f"  {row['Name']}, {row['Country name EN']}: {row['Population']:,}")