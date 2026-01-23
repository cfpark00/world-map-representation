# 18:23 - Additional Plots: Cities and Probe Generalization

## Summary
Created new visualization infrastructure for basic city plots and probe generalization analysis. Fixed incorrect FTWB2 task mapping in canonical_experiments.md.

## Tasks Completed

### 1. Created Additional Plots Infrastructure
Created new folders for additional visualization work:
- `scripts/revision/additional_plots/`
- `configs/revision/additional_plots/`
- `data/experiments/revision/additional_plots/`

### 2. Basic Cities Plot
**Script**: `src/scripts/plot_cities_basic.py`
- Plots all 5,175 cities colored by region
- Uses exact same color scheme as plotly PCA visualizations (Plotly + D3 + G10 qualitative colors)
- Styling: thick spines (3px), thick ticks (3px), bold tick labels (16pt)
- Tick labels show actual data scale (×10)
- No axis labels, no legend
- **Output**: `data/experiments/revision/additional_plots/cities_basic/figures/cities_basic.png`

### 3. Probe Atlantis Predictions Plots
**Script**: `src/scripts/plot_probe_atlantis_predictions.py`
- Shows predicted vs true Atlantis locations from linear probe generalization
- All world cities colored by region, Atlantis true (red circles), predicted (black X)
- Same styling as cities_basic plot

Created plots for 9 FTWB2 experiments:
- **WITH distance**: ftwb2-1, ftwb2-4, ftwb2-8, ftwb2-13, ftwb2-15
- **NO distance**: ftwb2-2, ftwb2-3, ftwb2-5, ftwb2-6

### 4. Fixed Canonical Experiments Documentation
**File**: `docs/canonical_experiments.md`

The FTWB2 task mapping was WRONG. Fixed based on actual dataset configs:

| ID | Old (Wrong) | New (Correct) |
|----|-------------|---------------|
| ftwb2-2 | distance + angle | angle + compass |
| ftwb2-3 | distance + compass | inside + perimeter |
| ftwb2-4 | distance + inside | crossing + distance |
| ftwb2-5 | distance + perimeter | trianglearea + angle |
| ftwb2-6 | distance + crossing | compass + inside |
| ftwb2-7 | trianglearea + angle | perimeter + crossing |
| ftwb2-8 | trianglearea + compass | distance + angle |
| ftwb2-9 | trianglearea + inside | trianglearea + compass |
| ftwb2-10 | trianglearea + perimeter | angle + inside |
| ftwb2-11 | trianglearea + crossing | compass + perimeter |
| ftwb2-12 | angle + compass | inside + crossing |
| ftwb2-13 | angle + inside | perimeter + distance |
| ftwb2-14 | angle + perimeter | crossing + trianglearea |
| ftwb2-15 | angle + crossing | distance + compass |
| ftwb2-16 | compass + inside | trianglearea + inside |
| ftwb2-17 | compass + perimeter | angle + perimeter |
| ftwb2-18 | compass + crossing | compass + crossing |
| ftwb2-19 | inside + perimeter | inside + distance |
| ftwb2-20 | inside + crossing | perimeter + trianglearea |
| ftwb2-21 | perimeter + crossing | crossing + angle |

### 5. 3D Nullspace Probe Visualization (Experimental)
**Script**: `src/scripts/plot_probe_3d_nullspace.py`
- Attempted to find a 3rd dimension (orthogonal to X,Y probes) where world cities are flat
- Hypothesis: If Atlantis is well-integrated, it should also be flat in this dimension
- Method: Pick 10 cities each from Boston/NYC, South Africa, North China areas, compute plane normal via SVD
- Results inconclusive - both WITH and WITHOUT distance show similar Atlantis offset

## Files Created/Modified

### New Scripts
- `src/scripts/plot_cities_basic.py`
- `src/scripts/plot_probe_atlantis_predictions.py`
- `src/scripts/plot_probe_3d_nullspace.py`

### New Configs (12 total)
- `configs/revision/additional_plots/cities_basic.yaml`
- `configs/revision/additional_plots/probe_atlantis_ftwb2-{1,2,3,4,5,6,8,13,15}.yaml`
- `configs/revision/additional_plots/probe_3d_nullspace_ftwb2-{2,13}.yaml`

### New Bash Scripts (12 total)
- `scripts/revision/additional_plots/plot_cities_basic.sh`
- `scripts/revision/additional_plots/plot_probe_atlantis_ftwb2-{1,2,3,4,5,6,8,13,15}.sh`
- `scripts/revision/additional_plots/plot_probe_3d_nullspace_ftwb2-{2,13}.sh`

### Modified
- `docs/canonical_experiments.md` - Fixed FTWB2 task mapping

## Output Locations
All outputs in `data/experiments/revision/additional_plots/`:
- `cities_basic/figures/cities_basic.png`
- `probe_atlantis_ftwb2-{N}/figures/probe_atlantis_predictions.png`
- `probe_3d_nullspace_ftwb2-{N}/figures/probe_3d_nullspace.png`
- `probe_3d_nullspace_ftwb2-{N}/figures/probe_x_vs_nullest.png`
