# Normalized Improvement Metric Explanation

**Date:** 2025-11-20
**Context:** Revision Exp1 FTWB2 evaluation heatmaps

## Overview

The normalized improvement metric quantifies how much a model improves over the untrained (atlantis) baseline, relative to the fully-trained (standard) baseline.

**Scale:**
- **0.0** = No improvement (performance equals atlantis baseline)
- **1.0** = Full improvement (performance equals standard baseline)
- **>1.0** = Exceeds standard baseline performance

## Metric Definitions

### Two Types of Tasks

We have two types of tasks based on what "better" means:

1. **Error Metrics** (lower is better): distance, trianglearea, angle, perimeter
2. **Accuracy Metrics** (higher is better): crossing, inside, compass

### Normalization Formulas

#### For Error Metrics (distance, trianglearea, angle, perimeter)

Uses **log-ratio** to account for multiplicative nature of errors:

```
normalized = log(baseline_atlantis / value) / log(baseline_atlantis / baseline_standard)
```

**Why log-ratio?**
- Errors often improve multiplicatively (e.g., 1000 → 100 → 10)
- Log-ratio treats relative improvements consistently
- Example: Improving from 1000→100 gets same credit as 100→10

**Example calculation:**
```
baseline_atlantis = 1000  (untrained on atlantis cities)
baseline_standard = 100   (fully trained on standard cities)
value = 316               (current performance)

normalized = log(1000/316) / log(1000/100)
          = log(3.165) / log(10)
          = 1.154 / 2.303
          = 0.50

Interpretation: 50% of the way from untrained to fully trained
```

#### For Accuracy Metrics (crossing, inside, compass)

Uses **linear** normalization:

```
normalized = (value - baseline_atlantis) / (baseline_standard - baseline_atlantis)
```

**Why linear?**
- Accuracies improve additively (e.g., 0.3 → 0.5 → 0.7)
- Linear scaling is natural for bounded metrics [0, 1]
- Equal absolute improvements get equal credit

**Example calculation:**
```
baseline_atlantis = 0.30  (untrained on atlantis cities)
baseline_standard = 0.90  (fully trained on standard cities)
value = 0.60              (current performance)

normalized = (0.60 - 0.30) / (0.90 - 0.30)
          = 0.30 / 0.60
          = 0.50

Interpretation: 50% of the way from untrained to fully trained
```

## Edge Cases

### When normalized < 0
- Performance is **worse** than atlantis baseline
- Clipped to 0.0 in practice (shouldn't happen with trained models)

### When normalized > 1.0
- Performance **exceeds** standard baseline
- Clipped to 1.5 maximum for visualization
- Indicates exceptional generalization

### Division by zero
- If `baseline_atlantis == baseline_standard`: normalized = 0.0
- Means no room for improvement (already at standard level on atlantis)

## Interpretation Guide

| Normalized Value | Meaning |
|-----------------|---------|
| 0.0 - 0.2 | Minimal improvement, mostly relies on transfer |
| 0.2 - 0.5 | Moderate improvement, partial task learning |
| 0.5 - 0.8 | Good improvement, substantial task learning |
| 0.8 - 1.0 | Excellent improvement, near fully-trained performance |
| > 1.0 | Exceptional, exceeds standard baseline |

## In the Heatmaps

**Trained tasks (marked with 'T'):**
- Expected to be high (0.8-1.0+)
- Shows direct training effectiveness
- Green cells indicate successful training

**Transfer tasks (unmarked):**
- Expected to be moderate (0.3-0.7)
- Shows generalization capability
- Yellow/orange cells indicate partial transfer

## Summary Statistics

For each seed, we report:

1. **Trained tasks mean**: Average normalized performance on tasks the model was explicitly trained on
2. **Transfer tasks mean**: Average normalized performance on tasks the model was NOT trained on

**Typical results:**
- Trained: 0.85-0.95 (models learn trained tasks well)
- Transfer: 0.50-0.65 (moderate generalization to untrained tasks)

## Key Insights from Normalization

1. **Fairness**: Each task contributes equally regardless of raw scale
   - Distance errors (~100-1000) and angle errors (~10-100) both normalized to [0,1]

2. **Interpretability**: 0.5 always means "halfway from untrained to trained"
   - Easy to compare across different task types

3. **Log-ratio for errors**: Captures that going from terrible→bad is as valuable as bad→good
   - Prevents ceiling effects where improvements on good models look small

4. **Linear for accuracy**: Natural for bounded metrics
   - 10% accuracy gain at 30% baseline = 10% gain at 80% baseline

## Code Implementation

See the `normalize_metric()` function in:
- `src/scripts/plot_revision_exp1_ftwb2_heatmaps.py`
- `scratch/cka_to_generalization/plot_multi_task_evaluation.py`
