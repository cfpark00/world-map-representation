# 20:15 - Probe Atlantis Plots and Histogram Styling Updates

## Summary
Updated probe atlantis prediction plots with new axis limits and marker styling. Fine-tuned the probe generalization histogram with exp5 line color and thickness.

## Tasks Completed

### 1. Probe Atlantis Predictions Plot Updates
**Script**: `src/scripts/plot_probe_atlantis_predictions.py`

Updated styling:
- **Axis limits**: xlim=[-900, 250], ylim=[50, 625] (stored as /10 for plotting)
- **Dot sizes**: Doubled city dots (6→12), Atlantis markers (40→80)
- **Crosses**: Larger (s=120) and thicker (linewidth=3)
- **Marker swap**: Ground truth now black crosses, predictions now red circles (was reversed)
- **Tick labels**: Updated to match new axis ranges

Replotted for ftwb2-2, ftwb2-5, and ftwb2-13.

### 2. Probe Generalization Histogram with Exp5 Updates
**Script**: `src/scripts/plot_probe_generalization_histogram_with_exp5.py`

Updated exp5 line styling:
- **Color**: Changed through green → purple → pink → #FF2AE0 → green → #2CBF32 → final #38C50D
- **Thickness**: Adjusted from 3 → 6 → 4 (final)
- **Tested dimming baseline**: Tried alpha=0.4/0.5 for orange histogram/line, reverted to original 0.6/0.8

## Files Modified

### Scripts
- `src/scripts/plot_probe_atlantis_predictions.py` - Axis limits, marker sizes, marker swap
- `src/scripts/plot_probe_generalization_histogram_with_exp5.py` - Exp5 line color (#38C50D) and thickness (4)

## Output Locations
- `data/experiments/revision/additional_plots/probe_atlantis_ftwb2-{2,5,13}/figures/probe_atlantis_predictions.png`
- `data/experiments/revision/exp1/plots/probe_generalization_histogram_with_exp5.png`
