# Session Log: 2025-09-01 18:00 - Representation Analysis and Catastrophic Forgetting Discovery

## Summary
Extended the representation analysis script to support multiple prompt formats and task types, ran comprehensive analyses on distance and random walk models, and discovered catastrophic forgetting when fine-tuning overwrites pretrained representations. Created a blog post documenting this finding.

## Major Tasks Completed

### 1. Structure Documentation Update
- Recreated `claude_notes/structure.txt` with a clean, reasonable project structure
- Previous file had auto-generated 3MB of node_modules and data file listings
- New structure focuses on important directories and files with descriptions

### 2. Representation Analysis Script Enhancement

#### Added Prompt Format Support
- Added `--prompt-format` parameter with options: 'dist' and 'rw200'
- Default behavior: auto-selects based on task type
- Enables cross-prompt testing (e.g., testing distance model with random walk prompts)

#### Added Task Type Parameter
- Added `--task-type` parameter to specify/override task type
- Properly extracts eval metrics based on task:
  - Distance tasks: show distance prediction error (km)
  - Random walk tasks: show validity ratio (0-1)

#### Updated Plotting Logic
- Top plot now adaptively shows:
  - Distance prediction error for distance models (log scale)
  - Validity ratio for random walk models (0-1 scale with percentage labels)
  - Location reconstruction error as fallback
- Bottom plot y-axis now auto-adjusts to data range instead of fixed [-0.2, 1.0]

#### Added Test Mode
- Added `--test` flag to verify argument parsing and prompt generation
- Prints resolved parameters and sample prompts without running full analysis

### 3. Comprehensive Model Analysis

#### Distance Model (dist_100k_1M_20epochs)
- With native prompts: R² = 0.956/0.923, location error = 993 km
- With random walk prompts: R² = -0.224/-0.261, complete failure

#### Random Walk Models
- `rw200_100k_1m_20epochs`: 
  - Native: R² = 0.192/0.088, validity = 96.1%
  - Distance prompts: R² = -0.208/-0.258, failure
  
- `rw200_100k_1m_20epochs_pt1` (pretrained on distance):
  - Native: R² = 0.220/0.085, validity = 96.2%
  - Distance prompts: R² = -0.016/-0.101, failure

### 4. Catastrophic Forgetting Discovery

#### Key Finding
- Copied pretrained checkpoint (step 19540 from distance model) as checkpoint-0
- Analysis revealed catastrophic forgetting:
  - Step 0 (pretrained): R² = 0.945/0.899, error = 1,185 km
  - Step 3908 (after fine-tuning): R² = -0.186/-0.153, error = 6,772 km
- Complete destruction of distance representations in just 3,908 steps

### 5. Blog Post Creation
- Wrote concise blog post on catastrophic forgetting finding
- Created in `/reports/catastrophic-forgetting-in-llms/`
- Includes dramatic visualization showing R² collapse from 0.95 to -0.2
- Highlights implications for LLM fine-tuning practices

## Technical Insights

### Prompt-Format Dependency
- Models learn prompt-specific representations
- Same geographic knowledge becomes inaccessible with different prompt formats
- No transfer between `dist(c_X,c_` and `walk_200=c_X,c_` formats

### Task-Specific Representations
- Distance models: Strong geographic representations (R² ~0.95)
- Random walk models: Weak geographic representations (R² ~0.22)
- Task difficulty directly impacts representation quality

### Catastrophic Forgetting
- Fine-tuning can completely destroy pretrained capabilities
- Not gradual degradation but immediate catastrophic loss
- Happens even when tasks share same domain (geography)

## Code Changes

### Modified Files
- `src/analysis/analyze_representations.py`: Major enhancements for prompt formats and task types
- `claude_notes/structure.txt`: Complete rewrite with clean documentation

### New Files
- `reports/catastrophic-forgetting-in-llms/index.mdx`: Blog post
- `reports/catastrophic-forgetting-in-llms/metadata.json`: Post metadata
- `reports/catastrophic-forgetting-in-llms/catastrophic_forgetting.png`: Key figure

## Key Commands Used
```bash
# Analysis with different prompt formats
python src/analysis/analyze_representations.py \
    --exp_dir outputs/experiments/rw200_100k_1m_20epochs_pt1 \
    --cities_csv outputs/datasets/cities_100k_plus_seed42.csv \
    --layers 3 4 \
    --task-type randomwalk \
    --prompt-format dist

# Test mode to verify parameters
python src/analysis/analyze_representations.py \
    --exp_dir ... \
    --test

# Copy pretrained checkpoint for analysis
cp -r outputs/experiments/dist_100k_1M_20epochs/checkpoints/checkpoint-19540 \
      outputs/experiments/rw200_100k_1m_20epochs_pt1/checkpoints/checkpoint-0
```

## Next Steps/Questions
1. Can we develop training methods that preserve representations across tasks?
2. Is there a way to access "lost" knowledge with different prompts?
3. How much of LLMs' limitations are just catastrophically forgotten capabilities?

## Time Spent
Approximately 1 hour 20 minutes (16:40 - 18:00)