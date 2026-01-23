#!/usr/bin/env python3
"""
Generate standalone horizontal colorbar for FTWB2 vs FTWB1 difference plots.
Uses RdBu colormap with center=0, vmin=-0.5, vmax=0.5.
"""

import matplotlib.pyplot as plt
import matplotlib as mpl
from pathlib import Path

# Output directory
OUTPUT_DIR = Path("/n/home12/cfpark00/datadir/WM_1/data/experiments/revision/exp1/plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Create figure with horizontal colorbar
fig = plt.figure(figsize=(8, 1.5))
ax = fig.add_axes([0.05, 0.3, 0.9, 0.5])

# Create colormap matching the heatmap
cmap = plt.cm.RdBu
norm = mpl.colors.TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5)

# Create colorbar
cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='horizontal')

# Remove tick labels for clean look
cb.set_ticks([])
cb.outline.set_visible(False)

# Save
output_path = OUTPUT_DIR / "ftwb2_vs_ftwb1_colorbar_horizontal.png"
plt.savefig(output_path, dpi=150, bbox_inches='tight')
print(f"Saved: {output_path}")

plt.close()
