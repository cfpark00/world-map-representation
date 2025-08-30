#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the CSV file
print("Loading city data...")
df = pd.read_csv('/n/home12/cfpark00/WM_1/data/geonames-all-cities-with-a-population-1000.csv', 
                 sep=';', encoding='utf-8-sig')

# Create figure with subplots
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('World City Population Distribution', fontsize=16)

# 1. Linear scale histogram
ax1 = axes[0, 0]
ax1.hist(df['Population'], bins=50, edgecolor='black', alpha=0.7, color='steelblue')
ax1.set_xlabel('Population')
ax1.set_ylabel('Number of Cities')
ax1.set_title('Population Distribution (Linear Scale)')
ax1.grid(True, alpha=0.3)

# 2. Log scale histogram
ax2 = axes[0, 1]
ax2.hist(df['Population'], bins=50, edgecolor='black', alpha=0.7, color='darkgreen')
ax2.set_xlabel('Population (log scale)')
ax2.set_ylabel('Number of Cities')
ax2.set_title('Population Distribution (Log Scale)')
ax2.set_xscale('log')
ax2.grid(True, alpha=0.3)

# 3. Population bins with custom ranges
ax3 = axes[1, 0]
bins = [1000, 5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000, 25000000]
hist_data = pd.cut(df['Population'], bins=bins)
bin_counts = hist_data.value_counts().sort_index()
bin_labels = [f'{bins[i]/1000:.0f}k-{bins[i+1]/1000:.0f}k' if bins[i+1] < 1000000 
              else f'{bins[i]/1000:.0f}k-{bins[i+1]/1000000:.0f}M' if bins[i] < 1000000
              else f'{bins[i]/1000000:.0f}M-{bins[i+1]/1000000:.0f}M' 
              for i in range(len(bins)-1)]
ax3.bar(range(len(bin_counts)), bin_counts.values, edgecolor='black', alpha=0.7, color='coral')
ax3.set_xticks(range(len(bin_counts)))
ax3.set_xticklabels(bin_labels, rotation=45, ha='right')
ax3.set_xlabel('Population Range')
ax3.set_ylabel('Number of Cities')
ax3.set_title('Cities by Population Categories')
ax3.grid(True, alpha=0.3, axis='y')

# 4. Cumulative distribution
ax4 = axes[1, 1]
sorted_pop = np.sort(df['Population'])
cumulative = np.arange(1, len(sorted_pop) + 1) / len(sorted_pop)
ax4.plot(sorted_pop, cumulative, linewidth=2, color='purple')
ax4.set_xlabel('Population')
ax4.set_ylabel('Cumulative Proportion')
ax4.set_title('Cumulative Distribution Function')
ax4.set_xscale('log')
ax4.grid(True, alpha=0.3)

# Add vertical lines for common thresholds
thresholds = [25000, 50000, 100000, 1000000]
threshold_colors = ['green', 'orange', 'red', 'darkred']
for thresh, color in zip(thresholds, threshold_colors):
    ax4.axvline(x=thresh, color=color, linestyle='--', alpha=0.5, 
                label=f'{thresh/1000:.0f}k' if thresh < 1000000 else f'{thresh/1000000:.0f}M')
ax4.legend()

plt.tight_layout()
import os
os.makedirs('outputs/figures', exist_ok=True)
plt.savefig('outputs/figures/city_population_histogram.png', dpi=150, bbox_inches='tight')
print("Histogram saved as 'outputs/figures/city_population_histogram.png'")

# Print statistics
print("\nPopulation Statistics:")
print(f"Total cities: {len(df):,}")
print(f"Mean population: {df['Population'].mean():,.0f}")
print(f"Median population: {df['Population'].median():,.0f}")
print(f"Min population: {df['Population'].min():,}")
print(f"Max population: {df['Population'].max():,}")

print("\nCities by threshold:")
for thresh in [25000, 50000, 100000, 250000, 500000, 1000000]:
    count = len(df[df['Population'] > thresh])
    pct = 100 * count / len(df)
    print(f"  > {thresh:>9,}: {count:>6,} cities ({pct:>5.1f}%)")

plt.show()