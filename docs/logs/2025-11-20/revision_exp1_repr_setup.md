# Revision Exp1 Representation Extraction Setup

**Date:** 2025-11-20
**Task:** Set up representation extraction for revision/exp1 experiments

## Summary

Created representation extraction configs and batch scripts for all revision/exp1 models (3 seeds × 22 models = 66 models).

This extracts layer 5 representations from the last checkpoint of each model, using linear probing to analyze learned representations.

## What Was Created

### 1. Representation Extraction Configs (66 total)

**Location:** `configs/revision/exp1/representation_extraction/`

**Structure:**
```
configs/revision/exp1/representation_extraction/
├── seed1/
│   ├── base/
│   │   └── distance_firstcity_last_and_trans_l5.yaml
│   ├── ftwb2-1/
│   │   └── distance_firstcity_last_and_trans_l5.yaml
│   ├── ftwb2-2/
│   │   └── angle_firstcity_last_and_trans_l5.yaml
│   ...
│   └── ftwb2-21/
│       └── angle_firstcity_last_and_trans_l5.yaml
├── seed2/
│   └── [same structure as seed1]
└── seed3/
    └── [same structure as seed1]
```

**Key Config Settings:**
- `layers: [5]` - Extract representations from layer 5
- `save_repr_ckpts: [-2]` - Extract from last checkpoint only
- `perform_pca: true` - Also compute PCA on representations
- `method: {name: "linear"}` - Use linear probing
- `n_train_cities: 3250, n_test_cities: 1250` - Train/test split for probing
- Task selection:
  - **Base models:** Use `distance` task (most common)
  - **FTWB2 models:** Use first trained task from each experiment

### 2. Batch Extraction Scripts (9 total)

**Location:** `scripts/revision/exp1/representation_extraction/`

**Scripts:**
```
extract_seed1_base_ftwb2-1-7.sh    (8 extractions: base + ftwb2 1-7)
extract_seed1_ftwb2-8-14.sh        (7 extractions: ftwb2 8-14)
extract_seed1_ftwb2-15-21.sh       (7 extractions: ftwb2 15-21)

extract_seed2_base_ftwb2-1-7.sh    (8 extractions)
extract_seed2_ftwb2-8-14.sh        (7 extractions)
extract_seed2_ftwb2-15-21.sh       (7 extractions)

extract_seed3_base_ftwb2-1-7.sh    (8 extractions)
extract_seed3_ftwb2-8-14.sh        (7 extractions)
extract_seed3_ftwb2-15-21.sh       (7 extractions)
```

**Total:** 66 representation extractions (66 models × 1 task each)

## Task Assignment for FTWB2 Models

Each ftwb2 model is trained on 2 tasks. We extract representations using the **first task** from the training data:

| FTWB2 # | Trained Tasks | Extraction Task |
|---------|---------------|-----------------|
| 1 | distance, trianglearea | distance |
| 2 | angle, compass | angle |
| 3 | inside, perimeter | inside |
| 4 | crossing, distance | crossing |
| 5 | trianglearea, angle | trianglearea |
| 6 | compass, inside | compass |
| 7 | perimeter, crossing | perimeter |
| 8 | angle, distance | angle |
| 9 | compass, trianglearea | compass |
| 10 | angle, inside | angle |
| 11 | compass, perimeter | compass |
| 12 | crossing, inside | crossing |
| 13 | distance, perimeter | distance |
| 14 | crossing, trianglearea | crossing |
| 15 | compass, distance | compass |
| 16 | inside, trianglearea | inside |
| 17 | angle, perimeter | angle |
| 18 | compass, crossing | compass |
| 19 | distance, inside | distance |
| 20 | perimeter, trianglearea | perimeter |
| 21 | angle, crossing | angle |

## Running Representation Extraction

Each script can be run independently:

```bash
# Run from project root
bash scripts/revision/exp1/representation_extraction/extract_seed1_base_ftwb2-1-7.sh
bash scripts/revision/exp1/representation_extraction/extract_seed1_ftwb2-8-14.sh
bash scripts/revision/exp1/representation_extraction/extract_seed1_ftwb2-15-21.sh
# ... etc
```

**Recommended:** Run scripts in parallel across seeds:
```bash
# In separate terminals or SLURM jobs
bash scripts/revision/exp1/representation_extraction/extract_seed1_base_ftwb2-1-7.sh &
bash scripts/revision/exp1/representation_extraction/extract_seed2_base_ftwb2-1-7.sh &
bash scripts/revision/exp1/representation_extraction/extract_seed3_base_ftwb2-1-7.sh &
```

## Output Structure

After running, representation analysis results will be stored at:

```
data/experiments/revision/exp1/
├── pt1_seed1/
│   └── analysis_higher/
│       └── distance_firstcity_last_and_trans_l5/
│           ├── representations/
│           ├── pca_results/
│           └── probe_results/
├── pt1_seed1_ftwb2-1/
│   └── analysis_higher/
│       └── distance_firstcity_last_and_trans_l5/
│           └── [same structure]
...
└── pt1_seed3_ftwb2-21/
    └── analysis_higher/
        └── angle_firstcity_last_and_trans_l5/
            └── [same structure]
```

## Generation Scripts

Created two utility scripts:

1. **`src/scripts/generate_revision_exp1_repr_configs.py`**
   - Generates all 66 representation extraction config files
   - Automatically assigns extraction task based on training data
   - Can be re-run if configs need to be regenerated

2. **`src/scripts/generate_revision_exp1_repr_scripts.py`**
   - Generates the 9 batch bash scripts
   - Can be re-run if script organization needs to change

## Next Steps

1. **Run representation extraction** using the batch scripts
2. **Analyze extracted representations** for PCA visualization
3. **Compare representations across seeds and models**
4. **Use for CKA analysis** and other representation comparison metrics

## Notes

- Representations are extracted from **layer 5 only** (the deepest layer)
- Only the **last checkpoint** is processed (not intermediate checkpoints)
- Uses `firstcity_last_and_trans` prompt format for all extractions
- PCA is computed automatically during extraction (`perform_pca: true`)
- Linear probing is used to evaluate representation quality
- This follows the same structure as exp3 representation extraction
- Each model has exactly 1 representation extraction config (unlike eval which has 15)

## Comparison with Evaluation

| Feature | Evaluation | Representation Extraction |
|---------|-----------|---------------------------|
| Number of configs per model | 15 (7 atlantis + 7 normal + 1 multi) | 1 (first trained task) |
| Total configs | 990 | 66 |
| Purpose | Measure task performance | Analyze internal representations |
| Output | Metrics (accuracy/error) | Representations + PCA + probes |
| Layers | Final output | Layer 5 hidden states |
