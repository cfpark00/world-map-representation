# Blog Post Cleanup and Representation Analysis
Date: 2025-09-01
Time: 21:35

## Summary
Cleaned up the catastrophic forgetting blog post based on user feedback and ran comprehensive representation analysis on fine-tuned models to validate findings about prompt-format-specific knowledge storage.

## Key Accomplishments

### 1. Blog Post Revisions
- **Removed time references**: Eliminated "10 minutes" from title and body text
  - Changed title from "How a Small Transformer Lost Its World Map in 10 Minutes" to just "How a Small Transformer Lost Its World Map"
  - Removed runtime estimates from the narrative
- **Removed AI Safety section**: Deleted the "What This Means for AI Safety and Capabilities" section as not relevant to the core findings
- **Fixed appendix formatting**: Corrected the technical appendix dropdown to match the style of other blog posts
  - Put summary text on same line as opening tag
  - Added proper CSS classes for dark mode compatibility

### 2. Representation Analysis Runs
Ran comprehensive analysis on multiple models to validate catastrophic forgetting findings:

#### Model: rw200_100k_1m_20epochs_pt1 (Fine-tuned from distance model)
- **Distance prompt format** (`dist(c_X,c_`):
  - Initial: R²=(0.945, 0.899), Error=1185 km (inherited from pretrained)
  - Final: R²=(-0.016, -0.101), Error=6290 km
  - Catastrophic forgetting confirmed - representations completely destroyed
  
- **Random walk prompt format** (`walk_200=c_X,c_`):
  - Initial: R²=(-0.276, -0.273), Error=7021 km
  - Final: R²=(0.220, 0.085), Error=5491 km
  - Slight improvement but representations remain poor
  - Key finding: The pretrained model couldn't access its knowledge through the wrong prompt format

#### Model: loc_100k_200epochs_ar (Location task)
- Distance prompt analysis:
  - Throughout training: R² ≈ -0.3 (worse than random)
  - Error: ~7200 km
  - Confirms location task doesn't develop geographic representations

#### Model: loc_100k_200epochs (Location task, different run)
- Distance prompt analysis:
  - Similar results: R² ≈ -0.3 throughout
  - Error: ~7200 km
  - Consistent with other location model

### 3. Key Insights Validated
- **Prompt-format lock-in**: Models store knowledge in format-specific silos
- **Catastrophic forgetting**: Fine-tuning immediately destroys prior representations
- **Task-specific representations**: Location task never develops coordinate representations despite using same city tokens

## Technical Details
- All analyses used layers 3 and 4 for representation extraction
- 5000 city probes (3000 train, 2000 test)
- Ridge regression with alpha=10.0
- Generated dynamics plots and world map animations for each run

## Files Modified
- `/reports/2025-09-01-catastrophic-forgetting-geographic-knowledge/index.mdx`
- `/reports/2025-09-01-catastrophic-forgetting-geographic-knowledge/metadata.json`

## Analysis Outputs Created
- `/outputs/experiments/rw200_100k_1m_20epochs_pt1/analysis/dist_layers3_4_probe5000_train3000/`
- `/outputs/experiments/rw200_100k_1m_20epochs_pt1/analysis/rw200_layers3_4_probe5000_train3000/`
- `/outputs/experiments/loc_100k_200epochs_ar/analysis/dist_layers3_4_probe5000_train3000/`
- `/outputs/experiments/loc_100k_200epochs/analysis/dist_layers3_4_probe5000_train3000/`

## Next Steps
- Further experiments with different learning rates to test if catastrophic forgetting can be mitigated
- Test replay strategies (mixing original task data during fine-tuning)
- Explore LoRA or adapter layers as alternatives to full fine-tuning