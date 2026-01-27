# Discovery: Neural Network Learns Numerical ID Patterns Before Geographic Semantics

**Date**: September 20, 2024
**Model**: m1_10M_width=256_depth=10
**Task**: World distance prediction (layer 8 analysis)
**Discovery Location**: Step 17,763 / 236,838 (7.5% through training)

## Summary

During early training, the model discovers and strongly encodes a simple numerical pattern—the number of digits in city IDs—before learning any meaningful geographic relationships. This pattern persists as an independent component even in the final checkpoint's PCA space.

## The Phenomenon

### Initial Discovery (Step 17,763)

When analyzing representations at step 17,763, we observed two distinct clusters in PCA space with extraordinary separation (Cohen's d = 6.48 on PC1). Investigation revealed these clusters correspond exactly to:

- **Small cluster (531 cities)**: All cities with `city_id < 1000` (1-3 digit IDs)
- **Large cluster (4,469 cities)**: All cities with `city_id ≥ 1000` (4 digit IDs)

Further sub-clustering revealed a hierarchy based on digit count:
- **Tiny cluster (3 cities)**: IDs 0-9 (single digit)
- **Small cluster (54 cities)**: IDs 10-99 (two digits)
- **Medium cluster (474 cities)**: IDs 100-999 (three digits)
- **Large cluster (4,469 cities)**: IDs 1000+ (four digits)

### Persistence to Final Checkpoint

**Critical finding**: This ID-based clustering remains visible as an independent principal component in the final checkpoint (step 236,838), despite the model having learned complex geographic relationships. This suggests the model maintains multiple parallel representation schemes:

1. **Numerical ID encoding** (learned early, preserved throughout)
2. **Geographic coordinate encoding** (learned later, becomes dominant)
3. **Regional/semantic encoding** (emerges in middle training)

## Why This Happens

### 1. Prompt Structure
The model sees prompts like:
```
dist(c_0003,c_
dist(c_0465,c_
dist(c_1234,c_
```

The ID appears directly in the input, making digit patterns immediately accessible.

### 2. Learning Dynamics
- **Simplicity bias**: Neural networks learn simple patterns first
- **Digit patterns** are easier to detect than geographic relationships
- **Leading zeros** in formatting make digit count highly salient

### 3. Task Structure
- During early training, the model hasn't learned coordinate mappings
- ID patterns provide a crude but consistent signal for organization
- Acts as a "scaffold" for later learning

## Implications

### For Understanding Neural Network Learning

1. **Staged Learning**: Networks naturally progress from simple numerical patterns to complex semantic understanding
2. **Feature Persistence**: Early-learned features aren't overwritten but remain as latent dimensions
3. **Multiple Representations**: Models can maintain parallel encoding schemes for the same entities

### For Model Interpretability

- Early checkpoints reveal what patterns are "easiest" for the model
- PCA on different checkpoints can trace the evolution of representations
- Simple numerical biases may persist even in well-trained models

### For Training Dynamics

- The model uses ~7.5% of training to discover and encode ID patterns
- Geographic learning begins after this numerical scaffold is established
- Final representations are a superposition of numerical and semantic features

## Visualization Evidence

![City ID Pattern Clustering](../../scratch/city_id_pattern_clusters.png)
*Figure 1: Perfect separation of cities by ID digit count at step 17,763*

Key observations from visualizations:
- PC1 at step 17,763 almost perfectly separates ID < 1000 from ID ≥ 1000
- Silhouette score: 0.534 (indicating strong cluster quality)
- No geographic or regional pattern in early clusters
- Even Atlantis cities are split by their ID values, not their geographic uniqueness

## Technical Details

### Detection Method
```python
# K-means clustering with k=2 on flattened representations
kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
labels = kmeans.fit_predict(representations_flat)

# Perfect correspondence with ID threshold
small_cluster = all cities with city_id < 1000
large_cluster = all cities with city_id >= 1000
accuracy = 100%
```

### Statistical Measures
- **PC1 separation**: Cohen's d = 6.48 (massive effect size)
- **PC2 separation**: Cohen's d = 0.30 (negligible)
- **Variance explained by PC1**: 27.1%
- **Cluster sizes**: 531 (ID<1000) vs 4,469 (ID≥1000)

## Future Research Questions

1. **Universality**: Does this occur in all geographic reasoning models?
2. **Mitigation**: Could randomizing ID assignment prevent this artifact?
3. **Utility**: Does this numerical scaffold actually help or hinder geographic learning?
4. **Persistence**: Why does this pattern remain in the final model instead of being overwritten?

## Conclusion

This discovery reveals how neural networks naturally decompose complex tasks, starting with the simplest available patterns. The persistence of ID-based clustering from 7.5% to 100% of training suggests that models maintain a richer representation space than typically assumed, with both task-relevant (geography) and task-irrelevant (ID numerics) features coexisting in the final learned representation.

---

*This phenomenon was discovered through interactive exploration of checkpoint representations using PCA visualization and clustering analysis. The finding highlights the importance of analyzing intermediate checkpoints to understand learning dynamics.*