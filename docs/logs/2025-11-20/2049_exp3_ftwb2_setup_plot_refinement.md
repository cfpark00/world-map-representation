# Exp3 FTWB2 Setup and Plot Refinement - 2025-11-20 20:49

## Summary
Session focused on understanding revision exp3 infrastructure, setting up additional ftwb2 training configs for the wide model, and refining the all_metrics_combined plot for better publication readiness.

## Activities

### 1. Revision Exp3 Review
- **Purpose**: Model width ablation study - testing wide vs narrow models for fine-tuning
  - Wide: 2× width (hidden=256, intermediate=1024, heads=8), ½ epochs (3)
  - Narrow: ½ width (hidden=64, intermediate=256, heads=2), 2× epochs (12)
- **Current status**: Only wide model track completed (pt1_wide + 7 ftwb1 fine-tunings)
  - Narrow model not yet pretrained
- **Research question**: Does model architecture affect fine-tuning effectiveness with constant compute?

### 2. FTWB2 Dataset Understanding
Analyzed ftwb2 dataset composition:
- **Main training**: 20k + 100k samples from 2 primary tasks (no atlantis + atlantis required)
- **Warmup data**: 256 samples × 7 tasks (all with Atlantis) for catastrophic forgetting prevention
- **Total**: ~240k samples per ftwb2 dataset
- Purpose: Train heavily on 2 tasks while maintaining all 7 task capabilities

### 3. Created Exp3 Wide FTWB2 Configs
Generated training infrastructure for pt1_wide with selected ftwb2 datasets:

**Files created (6 configs + 6 scripts):**
- `configs/revision/exp3/pt1_wide/pt1_wide_ftwb2-{2,4,9,12,13,15}.yaml`
- `scripts/revision/exp3/pt1_wide/pt1_wide_ftwb2-{2,4,9,12,13,15}.sh`

**Configuration**:
- Dataset paths: data/datasets/ftwb2-{2,4,9,12,13,15}
- Base checkpoint: data/experiments/revision/exp3/pt1_wide/checkpoints/final
- Learning rate: 1e-5
- Epochs: 30
- Architecture: hidden=256, intermediate=1024, heads=8, layers=6

### 4. Plot Refinement - all_metrics_combined.png
Location: `/n/home12/cfpark00/datadir/WM_1/scratch/pt1_eval_plot/plot_all_metrics_combined.py`

**Iterative improvements made:**
1. Thickened plot lines (linewidth: 2.5 → 4 → 8)
2. Removed grid
3. Removed top spine
4. Thickened axis lines (2px) and ticks
5. Removed axis labels (for manual addition in Illustrator)
6. Increased tick label size (18 → 22 → 33) and made bold
7. Removed minor ticks (log scale subticks)
8. Added tick label padding (pad=10)
9. **Line style differentiation** (reviewer feedback):
   - Left axis (error metrics): **solid lines** with circles
   - Right axis (accuracy metrics): **dashed lines** with squares
   - Addresses concern about axis assignment clarity without color coding
10. Increased marker sizes (4 → 6 → 9)
11. Enlarged figure (12×8 → 14×9 inches)
12. Set crossing line to highest z-order (10) to prevent occlusion
13. Increased legend font (14 → 16 → 24) and made bold

**Final plot characteristics:**
- Dual y-axis: left (error, log scale), right (accuracy, linear 0-1)
- 7 tasks: distance, trianglearea, angle, perimeter (errors) + crossing, compass, inside (accuracy)
- No labels (user will add in design software)
- Publication-ready styling with clear visual hierarchy

**Final metrics from pt1 model:**
- Error (↓): distance=2.38, trianglearea=1189.93, angle=0.96, perimeter=21.52
- Accuracy (↑): crossing=97.66%, compass=99.10%, inside=97.97%

### 5. Documentation Review
- Reviewed `docs/repo_usage.md` - fail-fast philosophy, orchestration vs implementation
- Reviewed exp1 ftwb1 infrastructure for understanding baseline computation approach

## Files Modified
- `scratch/pt1_eval_plot/plot_all_metrics_combined.py` (multiple iterations)

## Files Created
- 6 training configs: `configs/revision/exp3/pt1_wide/pt1_wide_ftwb2-*.yaml`
- 6 training scripts: `scripts/revision/exp3/pt1_wide/pt1_wide_ftwb2-*.sh`

## Next Steps
1. Train pt1_wide on ftwb2-{2,4,9,12,13,15} datasets (6 runs)
2. Set up evaluation infrastructure for exp3 (112 eval configs needed)
3. Consider training pt1_narrow base model to complete the ablation study
4. Compare wide vs narrow fine-tuning performance once both tracks complete

## Status
- ✅ Exp3 wide model base + ftwb1 training complete
- ✅ Exp3 wide ftwb2 configs created
- ⏳ Exp3 wide ftwb2 training pending
- ⏳ Exp3 evaluation infrastructure not yet created
- ❌ Exp3 narrow model not started
