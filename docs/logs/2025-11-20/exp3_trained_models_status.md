# Exp3 Trained Models Status

**Date**: 2025-11-20
**Issue**: Not all exp3 models have been trained yet

## Training Status

### ✓ Trained Models (9 total)
- `pt1_wide` - Base wide model
- `pt1_wide_ftwb1` - Wide fine-tuned on distance
- `pt1_wide_ftwb2` - Wide fine-tuned on trianglearea
- `pt1_wide_ftwb3` - Wide fine-tuned on angle
- `pt1_wide_ftwb4` - Wide fine-tuned on compass
- `pt1_wide_ftwb5` - Wide fine-tuned on inside
- `pt1_wide_ftwb6` - Wide fine-tuned on perimeter
- `pt1_wide_ftwb7` - Wide fine-tuned on crossing

### ✗ Not Yet Trained (8 total)
- `pt1_narrow` - Base narrow model
- `pt1_narrow_ftwb1` through `pt1_narrow_ftwb7` - All narrow fine-tuned variants

## Available Scripts

### For Trained Models Only

**Representation extraction:**
```bash
# Only pt1_wide and pt1_wide_ftwb{1-7}
bash scripts/revision/exp3/representation_extraction/extract_trained_only.sh
```

**PCA visualization:**
```bash
# Only pt1_wide models (base + ftwb)
bash scripts/revision/exp3/pca_timeline/pca_trained_only_all.sh
```

This runs:
- `pca_base_models_mixed.sh` (pt1_wide only)
- `pca_base_models_raw.sh` (pt1_wide only)
- `pca_wide_ftwb_mixed.sh` (all 7 wide ftwb)
- `pca_wide_ftwb_raw.sh` (all 7 wide ftwb)
- `pca_wide_ftwb_na.sh` (all 7 wide ftwb)

### For All Models (When pt1_narrow is trained)

**Representation extraction:**
```bash
bash scripts/revision/exp3/representation_extraction/extract_all.sh
```

**PCA visualization:**
```bash
bash scripts/revision/exp3/pca_timeline/pca_all.sh
```

## Updated Scripts

The following scripts have been updated to only process pt1_wide (narrow removed):
- `scripts/revision/exp3/representation_extraction/extract_base_models.sh`
- `scripts/revision/exp3/pca_timeline/pca_base_models_mixed.sh`
- `scripts/revision/exp3/pca_timeline/pca_base_models_raw.sh`

## Next Steps

1. **Use trained models now**: Run `extract_trained_only.sh` and `pca_trained_only_all.sh`
2. **After pt1_narrow training completes**:
   - Run narrow-specific scripts:
     - `extract_narrow_ftwb.sh`
     - `pca_narrow_ftwb_mixed.sh`
     - `pca_narrow_ftwb_raw.sh`
     - `pca_narrow_ftwb_na.sh`
   - Or use the master scripts: `extract_all.sh` and `pca_all.sh`

## PCA Types for Wide Models

1. **mixed** (2 configs): pt1_wide base + mixed alignment
2. **raw** (2 configs): pt1_wide base + raw PCA
3. **mixed** (7 configs): pt1_wide_ftwb{1-7} + mixed alignment
4. **raw** (7 configs): pt1_wide_ftwb{1-7} + raw PCA
5. **na** (7 configs): pt1_wide_ftwb{1-7} + no-Atlantis probe

Total available now: 25 PCA configs for trained models
