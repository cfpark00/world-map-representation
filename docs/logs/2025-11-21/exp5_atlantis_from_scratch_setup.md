# Exp5: Multi-task Training with Atlantis from Scratch - 2025-11-21

## Summary

Created complete infrastructure for Experiment 5, which tests whether training with Atlantis data from the beginning prevents representational disagreement.

## Research Question

**Does training a 7-task model from scratch with Atlantis included from the beginning prevent the representational disagreement observed when Atlantis is introduced only during evaluation?**

## Hypothesis

The current PT1 model was trained on `multitask_pt1` (7M samples, no Atlantis). When evaluated on Atlantis cities, the model shows poor generalization - suggesting the representations don't naturally extend to OOD data.

**Hypothesis**: If we train from scratch on `multitask_pt1_with_atlantis` (where Atlantis cities are naturally included in the training data), the model should:
1. Learn representations that naturally include Atlantis cities
2. Show NO representational disagreement between train cities and Atlantis cities
3. Perform well on Atlantis evaluation tasks

## Experiment Design

### Dataset Creation

Create 6 new 1M datasets with Atlantis included (distance_1M_with_atlantis already exists):
- `trianglearea_1M_with_atlantis`
- `angle_1M_with_atlantis`
- `compass_1M_with_atlantis`
- `inside_1M_with_atlantis`
- `perimeter_1M_with_atlantis`
- `crossing_1M_with_atlantis`

Then combine all 7 into:
- `multitask_pt1_with_atlantis` (~7M samples)

### Model Architecture

**Baseline architecture** (same as original PT1):
- 128 hidden size, 512 intermediate, 4 heads, 6 layers
- 6 epochs of training
- Seed: 42

### Training Configuration

**Config**: `configs/revision/exp5/pt1_with_atlantis.yaml`
- Dataset: `data/datasets/multitask_pt1_with_atlantis`
- Output: `data/experiments/revision/exp5/pt1_with_atlantis`
- Same hyperparameters as original PT1

## Infrastructure Created

### Dataset Configs (7)

**Individual task datasets** (6 new):
- `configs/data_generation/trianglearea_1M_with_atlantis.yaml`
- `configs/data_generation/angle_1M_with_atlantis.yaml`
- `configs/data_generation/compass_1M_with_atlantis.yaml`
- `configs/data_generation/inside_1M_with_atlantis.yaml`
- `configs/data_generation/perimeter_1M_with_atlantis.yaml`
- `configs/data_generation/crossing_1M_with_atlantis.yaml`

**Combined dataset**:
- `configs/data_generation/combine_multitask_pt1_with_atlantis.yaml`

### Training Config (1)

- `configs/revision/exp5/pt1_with_atlantis.yaml`

### Execution Scripts (2)

- `scripts/revision/exp5/generate_datasets.sh` - Generate all datasets
- `scripts/revision/exp5/train_pt1_with_atlantis.sh` - Train model

## Usage

### Step 1: Generate Datasets

```bash
# Generate all 6 new task datasets and combine them
bash scripts/revision/exp5/generate_datasets.sh
```

**Runtime estimate**: ~2-3 hours for all dataset generation

**Output locations**:
- `data/datasets/trianglearea_1M_with_atlantis/`
- `data/datasets/angle_1M_with_atlantis/`
- `data/datasets/compass_1M_with_atlantis/`
- `data/datasets/inside_1M_with_atlantis/`
- `data/datasets/perimeter_1M_with_atlantis/`
- `data/datasets/crossing_1M_with_atlantis/`
- `data/datasets/multitask_pt1_with_atlantis/`

### Step 2: Train Model

```bash
# Train PT1 model from scratch with Atlantis
bash scripts/revision/exp5/train_pt1_with_atlantis.sh
```

**Runtime estimate**: ~8-10 hours (same as original PT1)

**Output location**: `data/experiments/revision/exp5/pt1_with_atlantis/`

### Step 3: Analysis (TODO)

After training completes, we should:

1. **Extract representations** for both train cities and Atlantis cities
2. **Compute CKA** between train and Atlantis representations
3. **Generate PCA visualizations** to see if Atlantis cities fit naturally in the representation space
4. **Evaluate on all tasks** including Atlantis variants
5. **Compare with original PT1** to quantify the improvement

## Expected Results

### Success Criteria

If the hypothesis is correct, we should observe:

1. **High CKA similarity** between train city representations and Atlantis city representations
2. **Smooth PCA embeddings** where Atlantis cities fall naturally within the geographic structure
3. **Good Atlantis task performance** (close to train performance, not degraded)
4. **No representational disagreement** in 3D PCA plots

### Comparison with Original PT1

| Metric | Original PT1 | PT1 with Atlantis (Expected) |
|--------|--------------|------------------------------|
| Train→Atlantis CKA | Low? | High |
| Atlantis task performance | Poor | Good |
| PCA structure | Atlantis outliers | Atlantis integrated |
| Representational agreement | Disagreement | Agreement |

## Scientific Significance

This experiment tests a fundamental hypothesis about OOD generalization:

**Does poor OOD generalization come from:**
- A) Fundamental limitations of the model architecture? OR
- B) The training distribution not including OOD examples?

If PT1_with_atlantis performs well on Atlantis tasks and shows integrated representations, it suggests that **B) is the primary issue** - models can generalize to "OOD" data if that data is naturally part of the training distribution.

This has implications for:
- How we think about generalization in neural networks
- The importance of diverse training data
- Whether "zero-shot" OOD generalization is realistic

## Files Created

**Configs** (8):
- 6 individual task dataset configs (1M with Atlantis)
- 1 combined dataset config
- 1 training config

**Scripts** (2):
- Dataset generation script
- Training script

**Documentation** (1):
- This file

## Status

- ✅ Infrastructure complete
- ⏳ Dataset generation pending
- ⏳ Training pending
- ⏳ Analysis pending

## Next Steps

1. Run dataset generation (Step 1)
2. Verify datasets created successfully
3. Run training (Step 2)
4. Set up evaluation and analysis infrastructure
5. Compare results with original PT1
6. Write up findings for rebuttal

---

**Date**: 2025-11-21
**Context**: Rebuttal phase - testing fundamental hypothesis about OOD generalization
**Experiment**: Exp5 - PT1 trained from scratch with Atlantis included
