# Revision Experiments Setup - 2025-11-19 21:48

## Summary
Set up complete experiment configurations for paper revision requests, creating 4 major experiment suites (exp1-4) with training, representation extraction, and visualization pipelines.

## Experiments Created

### Exp1: Multi-task PT1 with Additional Seeds
**Goal**: Test robustness of multi-task pretraining (pt1) across different random seeds

**Created**:
- 3 training configs for pt1 with seeds 1, 2, 3 (original was seed 42)
- Model: hidden=128, intermediate=512, heads=4, epochs=6
- Training scripts for all 3 seeds
- Representation extraction configs for distance task (layer 5, ftwb format)
- PCA visualization configs (3 types: default, no_atlantis, raw) for each seed
- Complete pipeline scripts

**Files**:
- 3 training configs + scripts
- 3 extraction configs + scripts
- 9 PCA configs + 3 PCA scripts (3 types × 3 seeds)
- Output: `data/experiments/revision/exp1/pt1_seed{1,2,3}/`

### Exp2: Two-Task (PT2) Training with Additional Seeds
**Goal**: Repeat all 7 two-task combinations with 2 additional seeds for robustness

**PT2 Task Pairs**:
1. pt2-1: distance + trianglearea
2. pt2-2: angle + compass
3. pt2-3: inside + perimeter
4. pt2-4: crossing + distance
5. pt2-5: trianglearea + angle
6. pt2-6: compass + inside
7. pt2-7: perimeter + crossing

**Created**:
- 16 training configs (8 pt2 variants × 2 seeds)
- Model: Same as pt1 but 21 epochs
- Seeds: 1 and 2 (original was 42)
- 16 extraction configs (one per variant+seed)
- 48 PCA configs (16 experiments × 3 PCA types)
- 7 meta scripts (one per pt2-{1-7}) that run full pipeline for both seeds

**Files**:
- 16 training configs + scripts
- 16 extraction configs + scripts
- 48 PCA configs + 16 PCA scripts
- 7 meta scripts: `scripts/revision/exp2/run_pt2-{1-7}_all.sh`
- Output: `data/experiments/revision/exp2/pt2-{1-8}_seed{1,2}/`

### Exp3: Model Width Ablation Study
**Goal**: Test how model width affects fine-tuning by keeping compute roughly constant

**Models**:
- **Wide**: 2× width (hidden=256, intermediate=1024, heads=8), ½ epochs (3)
- **Narrow**: ½ width (hidden=64, intermediate=256, heads=2), 2× epochs (12)
- **Baseline** (for comparison): hidden=128, intermediate=512, heads=4, epochs=6

**Created**:
- 2 pretrain configs (wide, narrow) + scripts
- 14 fine-tuning configs using **ftwb protocol** (7 tasks × 2 model types)
  - ftwb = Fine-Tuning With Warmup and Baseline
  - Uses datasets: ftwb1-{1-7}
  - Learning rate: 1e-5 (1/3 of pretrain)
  - Epochs: 30
  - Loads from pretrained checkpoint
- 2 meta scripts for complete pipelines

**Files**:
- 2 pretrain configs + scripts
- 14 ftwb fine-tune configs + scripts
- 2 meta scripts: `scripts/revision/exp3/run_pt1_{wide,narrow}_all.sh`
- Output: `data/experiments/revision/exp3/pt1_{wide,narrow}/` and `pt1_{wide,narrow}_ftwb{1-7}/`

**Note**: Removed non-ftwb fine-tuning configs after clarification that ftwb is the proper protocol.

### Exp4: Single-Task (PT1-X) with Additional Seeds
**Goal**: Repeat all 7 single-task pretraining experiments with 2 additional seeds

**Tasks** (pt1-{1-7}):
1. distance
2. trianglearea
3. angle
4. compass
5. inside
6. perimeter
7. crossing

**Created**:
- 14 training configs (7 tasks × 2 seeds)
- Seeds: 1 and 2 (original was 42)
- 42 epochs per task
- 7 meta scripts (one per task) running both seeds

**Files**:
- 14 training configs + scripts
- 7 meta scripts: `scripts/revision/exp4/run_pt1-{1-7}_all.sh`
- Output: `data/experiments/revision/exp4/pt1-{1-7}_seed{1,2}/`

## Key Configuration Details

### Seeds
- Original experiments: seed 42
- New experiments: seeds 1, 2 (and 3 for exp1)

### FTWB Fine-tuning Protocol
Discovered that proper fine-tuning uses "ftwb" (Fine-Tuning With Warmup and Baseline):
- Datasets combine main task + Atlantis data + 256 samples of all 7 tasks for warmup
- Lower learning rate: 1e-5 (vs 3e-4 for pretraining)
- Fewer epochs: 30 (vs 42 for single-task pretrain)
- Loads from pretrained checkpoint

### PCA Visualization Types
Three types created for each experiment:
1. **Default (mixed)**: PC1→x, PC2→y, PC3→r0 (linear probed alignment)
2. **No Atlantis (na)**: Same as default but excludes Atlantis from probe training
3. **Raw (pca)**: Pure PCA components without probe alignment

### Representation Extraction
- Layer 5 focus
- Prompt format: `{task}_firstcity_last_and_trans`
- 4500 cities total (3250 train, 1250 test)
- Extracts from last checkpoint (-2)

## Directory Structure Created

```
configs/revision/
├── exp1/pt1_seed/
│   ├── {pt1_seed1,2,3}.yaml
│   ├── extract_representations/
│   └── pca_timeline/
├── exp2/pt2_seed/
│   ├── pt2-{1-8}/
│   ├── extract_representations/
│   └── pca_timeline/
├── exp3/
│   ├── pt1_wide/
│   │   ├── pt1_wide.yaml
│   │   └── pt1_wide_ftwb{1-7}.yaml
│   └── pt1_narrow/
│       ├── pt1_narrow.yaml
│       └── pt1_narrow_ftwb{1-7}.yaml
└── exp4/pt1_single_task_seed/
    └── pt1-{1-7}/

scripts/revision/
├── exp1/ (mirrors exp1 configs)
├── exp2/ (+ meta scripts)
├── exp3/ (+ meta scripts)
└── exp4/ (+ meta scripts)
```

## Total Files Created

- **Configs**: ~120 YAML files
- **Scripts**: ~120 shell scripts
- **Meta scripts**: 16 scripts for running complete pipelines

## Next Steps

These experiments will allow the research to address reviewer concerns about:
1. **Robustness**: Multiple seeds for all key experiments
2. **Model capacity**: Width ablation study (exp3)
3. **Single vs multi-task**: Complete seed coverage for both (exp1, exp4)
4. **Two-task combinations**: Seed coverage for all pt2 variants (exp2)

## Usage Examples

```bash
# Exp1: Run complete pt1 seed1 pipeline
bash scripts/revision/exp1/pt1_seed/pt1_seed1.sh
bash scripts/revision/exp1/pt1_seed/extract_representations/pt1_seed1.sh
bash scripts/revision/exp1/pt1_seed/pca_timeline/pt1_seed1.sh

# Exp2: Run complete pt2-1 pipeline (both seeds)
bash scripts/revision/exp2/run_pt2-1_all.sh

# Exp3: Run wide model complete pipeline
bash scripts/revision/exp3/run_pt1_wide_all.sh

# Exp4: Run pt1-1 (distance) both seeds
bash scripts/revision/exp4/run_pt1-1_all.sh
```
