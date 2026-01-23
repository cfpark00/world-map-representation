# 16:56 - Raw Metrics CSV Generation for Revision Appendix

## Summary
Created a script to generate raw evaluation metrics CSV for reviewer-requested appendix table. The CSV includes all models across 4 seeds (seed0=original seed42, seed1, seed2, seed3) with 7 tasks × 2 conditions (ID/OOD).

## Tasks Completed

### 1. Reviewed Revision Exp1 Structure
- Read `docs/repo_usage.md` for project conventions
- Explored `scripts/revision/exp1/` directory structure
- Understood the pipeline: Training → Evaluation → Representation Extraction → PCA Timeline → Probe Generalization → Plotting

### 2. Created Raw Metrics Generation Script
- **Created**: `src/scripts/generate_raw_metrics_latex.py`
- **Purpose**: Generate CSV of raw evaluation metrics (mean error) for all models
- **Output**: `data/experiments/revision/exp1/tables/raw_metrics_all_seeds.csv`

### 3. CSV Format
- **Header comment**: `# seed0 corresponds to original seed=42`
- **Columns**: `seed,model,distance_id,distance_ood,trianglearea_id,trianglearea_ood,angle_id,angle_ood,compass_id,compass_ood,inside_id,inside_ood,perimeter_id,perimeter_ood,crossing_id,crossing_ood`
- **Rows**: 116 models (4 seeds × 29 models each)
  - 1 base model per seed
  - 7 ftwb1 models per seed (ftwb1-1 to ftwb1-7)
  - 21 ftwb2 models per seed (ftwb2-1 to ftwb2-21)
- **Values**: 2 significant figures, scientific notation where appropriate (e.g., `9.3e+04`)

### 4. Key Implementation Details
- Reads evaluation results from `evals/{task}/eval_data/evaluation_results.json`
- Uses **last checkpoint** for each model (sorted by checkpoint number)
- Handles both original models (`data/experiments/pt1*`) and revision seeds (`data/experiments/revision/exp1/pt1_seed*`)

## Files Created/Modified
- `src/scripts/generate_raw_metrics_latex.py` - Script to generate metrics CSV
- `data/experiments/revision/exp1/tables/raw_metrics_all_seeds.csv` - Output CSV

## Usage
```bash
uv run python src/scripts/generate_raw_metrics_latex.py
```
