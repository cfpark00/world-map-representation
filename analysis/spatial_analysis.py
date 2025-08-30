import json
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
from pathlib import Path

# Load the 100k dataset
data_path = Path("/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1/data/raw/world_cities_100k.json")
print(f"Loading data from {data_path}")

with open(data_path, 'r') as f:
    cities_data = json.load(f)

# Extract coordinates
coordinates = []
city_names = []
for city in cities_data:
    lat = city['lat']
    lon = city['lng']
    coordinates.append([lat, lon])
    city_names.append(city['city'])

coordinates = np.array(coordinates)
print(f"Loaded {len(coordinates)} cities")

# Convert lat/lon to radians for proper distance calculation
coords_rad = np.radians(coordinates)

# Build KDTree
print("Building KDTree...")
tree = cKDTree(coords_rad)

# Earth radius in km
EARTH_RADIUS_KM = 6371.0

# Convert 200km to radians (approximate)
# Arc length = radius * angle_in_radians
# 200km = 6371km * angle_in_radians
distance_threshold_rad = 200.0 / EARTH_RADIUS_KM

# Sample 10,000 random points
n_samples = 10000
sample_indices = np.random.choice(len(coordinates), n_samples, replace=False)
print(f"\nSampling {n_samples} random cities...")

# Count cities within 200km for each sampled point
counts = []
for idx in sample_indices:
    point_rad = coords_rad[idx]
    
    # Find all points within the threshold distance
    # Using Euclidean distance in radians as approximation
    # For more accuracy, we'd use haversine formula
    neighbors = tree.query_ball_point(point_rad, distance_threshold_rad)
    
    # Subtract 1 to exclude the point itself
    num_neighbors = len(neighbors) - 1
    counts.append(num_neighbors)

counts = np.array(counts)

# Statistics
print("\n=== Results ===")
print(f"Number of cities analyzed: {n_samples}")
print(f"Cities within 200km radius statistics:")
print(f"  Mean: {np.mean(counts):.2f}")
print(f"  Median: {np.median(counts):.0f}")
print(f"  Min: {np.min(counts)}")
print(f"  Max: {np.max(counts)}")
print(f"  Std Dev: {np.std(counts):.2f}")

# Distribution
print(f"\nDistribution:")
for threshold in [0, 5, 10, 20, 50, 100]:
    pct = np.sum(counts >= threshold) / len(counts) * 100
    print(f"  Cities with >= {threshold} neighbors: {pct:.1f}%")

# Create histogram
plt.figure(figsize=(10, 6))
plt.hist(counts, bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Number of cities within 200km')
plt.ylabel('Frequency')
plt.title(f'Distribution of nearby cities (200km radius) for {n_samples} random samples')
plt.axvline(x=np.mean(counts), color='red', linestyle='--', label=f'Mean: {np.mean(counts):.2f}')
plt.axvline(x=np.median(counts), color='green', linestyle='--', label=f'Median: {np.median(counts):.0f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/n/home12/cfpark00/WM_1/analysis/nearby_cities_distribution.png', dpi=150)
print(f"\nHistogram saved to /n/home12/cfpark00/WM_1/analysis/nearby_cities_distribution.png")

# Save results to file
results = {
    'n_samples': n_samples,
    'distance_threshold_km': 200,
    'statistics': {
        'mean': float(np.mean(counts)),
        'median': float(np.median(counts)),
        'min': int(np.min(counts)),
        'max': int(np.max(counts)),
        'std_dev': float(np.std(counts))
    },
    'distribution': {
        f'>={threshold}': float(np.sum(counts >= threshold) / len(counts) * 100)
        for threshold in [0, 5, 10, 20, 50, 100]
    }
}

with open('/n/home12/cfpark00/WM_1/analysis/spatial_analysis_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to /n/home12/cfpark00/WM_1/analysis/spatial_analysis_results.json")