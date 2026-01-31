# Research Context: Understanding Representation Formation in Geographic Space

**Last Updated**: 2026-01-27 04:50

## Codebase Reorganization (2026-01-23)

Major reorganization from flat structure to track-based organization. See `docs/REORGANIZATION.md` for full details.

### Completed Tracks
- **data_generation_v1**: Data generation + tokenizer creation (COMPLETE)
  - 7 core tasks: distance, trianglearea, angle, compass, inside, perimeter, crossing
  - Structure: `src/data_generation_v1/`, `configs/data_generation_v1/`, `scripts/data_generation_v1/`
  - Output: `data/data_generation_v1/` (cities/, tokenizers/, single_datasets/, derived_datasets/)
  - Config counts: cities(1), tokenizers(1), single_tasks(21), pretraining(18), finetuning(70)
  - Tested: City generation (5,175 cities), tokenizer generation (98 tokens)

### Planned Tracks (TODO)
- **pretraining_v1**: All pretraining (PT1, PT1-X, PT2, PT3, all seeds)
- **finetuning_v1**: All fine-tuning (FT, FTWB, all combos/seeds)
- **cka_v1**: CKA analysis infrastructure

### Naming Convention
- Pretraining: pt7 (all 7 tasks), pt2-{1..21}, pt3-{1..35}
- Fine-tuning: ft1-{1..7}, ft2-{1..21}, ft3-{1..35}
- Fine-tuning with warmup+baseline: ftwb1-{1..7}, ftwb2-{1..21}, ftwb3-{1..35}

---

## Research Goal

This project investigates a fundamental question in AI interpretability: **What conditions determine how representations form during training?** Specifically, we study when neural network representations organize themselves in modular, interpretable ways versus when they become fractured and entangled.

Rather than analyzing pre-trained models, we focus on the causal factors that govern representation formation by using geographic data as a controlled testbed. By training transformer models on synthetic location prediction tasks with world cities, we can systematically study representation formation under different conditions.

## Research Scope

### Primary Research Questions
1. When do representations form in modular vs. fractured ways?
2. How do scaling dynamics (data size, model size) affect representation formation?
3. What impact do different training regimes (single-task, multi-task, fine-tuning) have on representation structure?
4. Can we identify critical hyperparameters that determine representation quality?

### Methodological Approach
- **Domain**: Geographic space (world cities with latitude/longitude coordinates)
- **Tasks**: 7 geometric/spatial reasoning tasks
  1. distance: Predict geodesic distance between city pairs
  2. trianglearea: Compute area of triangle formed by 3 cities
  3. angle: Calculate angle at vertex of 3 cities
  4. compass: Determine bearing/direction between cities
  5. inside: Check if point is inside triangle
  6. perimeter: Calculate perimeter of triangle
  7. crossing: Determine if two line segments intersect

- **Model Architecture**: Small Qwen2.5-inspired transformers
  - Baseline: 128 hidden size, 512 intermediate, 4 heads, 6 layers
  - Custom character-level tokenizer (45 tokens)
  - Task-specific loss masking

- **Training Regimes**:
  - PT1: Multi-task pretraining on all 7 tasks (7M samples, 6 epochs)
  - PT1-X: Single-task pretraining (1M samples per task, 42 epochs)
  - PT2: Two-task combinations (8 variants, 21 epochs)
  - FTWB: Fine-Tuning With warmup and Baseline (checkpoint loading, 1e-5 lr, 30 epochs)

## Current State (2025-11-22)

### Completed Work

#### Phase 1: Initial Experiments
- Established multi-task (PT1) and single-task (PT1-X) pretraining baselines
- Developed two-task combinations (PT2) to study task interactions
- Created fine-tuning protocol (FTWB) with warmup and baseline preservation
- Implemented representation extraction pipeline (layer-wise analysis)
- Built PCA visualization system with 3 types:
  - Default/mixed: Linear probe alignment (PC1→x, PC2→y, PC3→r0)
  - No Atlantis (na): Probe trained without Atlantis data
  - Raw: Pure PCA without probe alignment

#### Phase 2: Analysis Infrastructure
- **CKA v2** (Centered Kernel Alignment, 2025-11-19 clean rewrite):
  - GPU/CPU accelerated centered kernel alignment computation
  - Hierarchical output organization: `{group}/{pair}/layer{X}/`
  - Timeline visualization showing CKA evolution across checkpoints
  - Statistical summaries (final, mean, std, min, max CKA values)
  - Representation alignment across experiments by city IDs
  - Cross-seed similarity analysis (21×21 matrices)
  - Location: `src/analysis/cka_v2/`
- **Legacy CKA** (older implementation):
  - Single-layer pairwise comparisons
  - Location: `src/analysis/analysis_cka_*.py`
- Dimensionality analysis tools
- Training dynamics visualization
- Timeline analysis of representation formation

#### Phase 3: Paper Revision Experiments (2025-11-19)
In response to reviewer feedback requesting robustness checks and ablation studies, created 4 comprehensive experiment suites:

**Exp1: Multi-task PT1 Seed Robustness + FTWB1/FTWB2 Fine-tuning** (Complete infrastructure: 2025-11-20)
- **Base models**: 3 training configs with seeds 1, 2, 3 (original: seed 42)
- **FTWB1 (single-task FT)**: 21 models (3 seeds × 7 tasks) - **Essential for expected generalization baseline**
  - Training infrastructure created 2025-11-20
  - 21 training configs, 315 evaluation configs, 21 representation extraction configs
  - Enables computation of **Actual - Expected generalization** (core paper contribution)
  - Expected generalization = max performance from single-task specialists
  - Actual generalization = multi-task model performance
  - Status: ⏳ Training pending, infrastructure complete
- **FTWB2 (two-task FT)**: 63 models (3 seeds × 21 task combinations)
  - All 21 two-task combinations trained for each seed
  - 990 evaluation configs (66 models × 15 tasks: 7 atlantis + 7 normal + 1 multi)
  - 66 representation extraction configs (layer 5)
  - Status: ✅ Training complete, ✅ Evaluations complete
- **Complete infrastructure** (2025-11-20 18:01):
  - 1,413 total configs (990 eval + 87 repr + 21 training + 315 FTWB1 eval)
  - 26 executable scripts (11 eval + 11 repr + 4 training)
  - 9 generator scripts for automated config/script creation
  - Visualization: Heatmaps showing normalized performance
  - Bug fix: `evaluate_checkpoints.py` KeyError with `save_full_results=False`
- **Visualization styling updates** (2025-11-21 evening):
  - **FTWB1**: Added 'T' markers on diagonal (7×7 matrices)
  - **FTWB2**: Transposed to 21×7, removed colorbar, clean styling
  - **FTWB2 vs FTWB1**: Transposed to 21×7, RdBu colormap for difference plots
  - **Aggregated plots**: All three plot types now have averaged versions across 4 seeds
  - Figure sizes: FTWB1 (10,8), FTWB2/comparison (10,16)
  - Output: 15 publication-ready PNG files in `plots/`
- **Probe Generalization Analysis** (2025-11-21 15:16):
  - **Purpose**: Evaluate how well spatial representations transfer to OOD (Atlantis) cities
  - **Method**: Train linear probe on 4000 non-Atlantis cities to predict x,y coordinates
  - **Test sets**: 87 Atlantis cities (OOD) + 100 held-out non-Atlantis (baseline)
  - **Coverage**: All 84 FTWB2 models (4 seeds × 21 task combinations)
  - **Key Results** (pooled across all seeds):
    - Baseline (non-Atlantis): 18.6 ± 16.2 error (8,400 samples)
    - Atlantis (no distance task): 107.8 ± 104.9 error (5,220 samples)
    - Atlantis (with distance task): 508.3 ± 288.3 error (2,088 samples)
  - **Key Finding**: Distance task severely impairs probe generalization (~5x worse)
  - **Statistical significance**: Mann-Whitney U p ≈ 0 between Atlantis groups
  - **Scripts**: `evaluate_probe_generalization.py`, `plot_probe_generalization_histogram.py`
  - **Output**: `data/experiments/revision/exp1/plots/probe_generalization_histogram.png`
- **Raw Metrics CSV** (2025-11-21 16:56):
  - **Purpose**: Generate raw evaluation metrics for reviewer-requested appendix table
  - **Script**: `generate_raw_metrics_latex.py`
  - **Output**: `data/experiments/revision/exp1/tables/raw_metrics_all_seeds.csv`
  - **Format**: 116 rows (4 seeds × 29 models), 14 metric columns (7 tasks × ID/OOD)
  - **Seeds**: seed0=original(42), seed1, seed2, seed3
- **Evaluation settings**:
  - Temperature 0.0 (deterministic greedy decoding)
  - Last checkpoint only (not all checkpoints like original pt1)
  - Both atlantis (OOD) and normal (ID) tasks evaluated
- **Normalized metrics**:
  - Scale: 0 (atlantis baseline) to 1 (standard baseline)
  - Error metrics: Log-ratio normalization (distance, trianglearea, angle, perimeter)
  - Accuracy metrics: Linear normalization (crossing, inside, compass)
- Output: `data/experiments/revision/exp1/pt1_seed{1,2,3}/` (base models)
- Output: `data/experiments/revision/exp1/pt1_seed{1,2,3}_ftwb1-{1-7}/` (single-task FT)
- Output: `data/experiments/revision/exp1/pt1_seed{1,2,3}_ftwb2-{1-21}/` (two-task FT)

**Exp2: Two-task PT2 and Three-task PT3 Seed Robustness**
- **PT2 (Two-task):** All 8 PT2 task combinations with 2 additional seeds (1, 2)
  - 16 total experiments (8 variants × 2 seeds)
  - Task pairs: distance+trianglearea, angle+compass, inside+perimeter, crossing+distance, trianglearea+angle, compass+inside, perimeter+crossing, (pt2-8)
  - Output: `data/experiments/revision/exp2/pt2-{1-8}_seed{1,2}/`
  - **Representation extraction** (2025-11-20): 49 configs for layers 3,4,5,6
    - PT2 seed1: 21 configs (layers 3,4,6; layer 5 already existed)
    - PT2 seed2: 28 configs (all 4 layers)
- **PT3 (Three-task):** All 8 PT3 task combinations with 2 seeds (1, 2) [Added 2025-11-20]
  - 16 total experiments (8 variants × 2 seeds)
  - Task triples: distance+trianglearea+angle, compass+inside+perimeter, crossing+distance+trianglearea, angle+compass+inside, perimeter+crossing+distance, trianglearea+angle+compass, inside+perimeter+crossing, distance+trianglearea+compass
  - Output: `data/experiments/revision/exp2/pt3-{1-8}_seed{1,2}/`
  - Infrastructure: Training configs/scripts + 8 meta scripts (`run_pt3-X_all.sh`)
  - **Representation extraction** (2025-11-20): 56 configs for layers 3,4,5,6
    - PT3 seed1: 28 configs (all 4 layers)
    - PT3 seed2: 28 configs (all 4 layers)
- **Multi-layer extraction infrastructure** (2025-11-20):
  - 105 total configs for PT2/PT3 seeds
  - 11 bash scripts for selective/complete extraction
  - Master script: `scripts/revision/exp2/extract_pt2_pt3_all_multilayer.sh`
  - Created by: `generate_pt2_pt3_multilayer_repr_configs.py` and `generate_pt2_pt3_multilayer_run_scripts.py`
- **Cross-seed CKA Analysis Infrastructure** (2025-11-21 02:16):
  - **Non-overlapping pairs only**: Models that share NO training tasks
    - PT2: 14 pairs (out of 21 possible)
    - PT3: 7 pairs (out of 21 possible)
  - **Seed combinations**: 3 unique pairs per model pair (orig-1, orig-2, 1-2)
  - **Total calculations**: 252 configs (168 PT2 + 84 PT3)
    - 14 PT2 pairs × 4 layers × 3 seed combos = 168
    - 7 PT3 pairs × 4 layers × 3 seed combos = 84
  - **Key optimizations**:
    - Only unique seed pairs (exploits CKA symmetry, 50% reduction)
    - Auto-detects final checkpoint per experiment (`use_final_only: true`)
    - Automatic overwrite enabled
  - **Execution scripts** (15 total):
    - Layer-specific: `run_pt{2,3}_seed_cka_l{3,4,5,6}.sh` (8 scripts)
    - Masters: `run_pt{2,3}_seed_cka_all_layers.sh`, `run_exp2_seed_cka_all.sh` (3 scripts)
    - Parallel chunks: `run_exp2_seed_cka_chunk{1,2,3,4}.sh` (4 × 63 calcs)
  - **Generator scripts**: 3 total
    - `generate_exp2_seed_cka_configs.py` (config generation)
    - `generate_exp2_seed_cka_run_scripts.py` (layer/master scripts)
    - `generate_exp2_seed_cka_chunked_scripts.py` (4-way parallel split)
  - **Output**: `data/experiments/revision/exp2/cka_analysis/`
  - **Status**: ⏳ Ready to run, infrastructure complete
- **Full 21×21 CKA Matrix** (2025-11-21 12:34):
  - **Coverage**: All 21 PT2 experiments (7 variants × 3 seeds) at layers 3, 4, 6
    - Layer 5 already complete
    - 210 pairs per layer = 630 total CKA computations
  - **Config format fix**: Updated `generate_exp2_pt2_all_pairs_cka_configs.py`
    - Changed from `repr_path` (single file) to `repr_dir` (checkpoint directory)
    - Matches layer 5 format expected by `analyze_cka_pair.py`
    - Uses `analysis_higher/` representation directories
  - **Execution**: 3 parallel chunks + status checker
    - `run_pt2_21x21_l346_chunk1.sh` - Layer 3 (210 jobs)
    - `run_pt2_21x21_l346_chunk2.sh` - Layer 4 (210 jobs)
    - `run_pt2_21x21_l346_chunk3.sh` - Layer 6 (210 jobs)
    - `check_pt2_21x21_l346_status.sh` - Progress monitor
  - **Status**: ⏳ Running (started 2025-11-21 afternoon)
- **Same-Task CKA Analysis** (2025-11-21 12:34):
  - **Research question**: Does multi-task training increase representational alignment even for the same task?
  - **Method**: Compare different seeds of the same task (intra-task similarity)
    - Extracts off-diagonal entries from 3×3 seed blocks (orig, seed1, seed2)
    - 3 unique seed pairs per variant: (orig,seed1), (orig,seed2), (seed1,seed2)
    - PT1: 18 values at layer 5 (6 variants, excludes pt1-7 crossing)
    - PT2: 18 values at layer 5 (6 variants, excludes pt2-7 crossing)
    - PT3: 18 values at layer 5 (6 variants, excludes pt3-7/8 crossing)
  - **Comparison with cross-task**: Also plots inter-task CKA from non-overlapping pairs
  - **Results** (Layer 5):
    - **Same-task CKA**: PT1=0.786, PT2=0.912, PT3=0.906
    - **Cross-task CKA**: PT1=0.655, PT2=0.873, PT3=0.851
    - **Key finding**: Same-task > Cross-task, showing multi-task training increases alignment
  - **Visualization**: Two plots with dimmed points (alpha=0.15)
    - `same_task_cka_trends.png` - Same-task only (solid green line)
    - `same_task_cka_trends_with_cross.png` - Both same-task and cross-task
  - **Scripts**: `collect_same_task_cka_trends.py`, `plot_same_task_cka_trends.py`
  - **Output**: `data/experiments/revision/exp2/cka_trends/`
- **PT2 21×21 Visualization** (2025-11-21):
  - **Purpose**: Visualize all 21×21 PT2 combinations at layer 5
  - **Plots generated**:
    - Full 21×21 matrix with seed labels
    - 7×7 averaged across seeds (without SEM)
    - 7×7 averaged with SEM annotations
  - **Styling**: Matches exp4 (no labels, adaptive text color, gray grid)
  - **Script**: `plot_21x21_pt2_cka_matrix_l5.py`
  - **Output**: `data/experiments/revision/exp2/cka_analysis_all/`

**Exp3: Model Width Ablation** (Evaluation infrastructure complete 2025-11-21)
- **Research question**: Does model architecture (wide vs narrow) affect fine-tuning performance with constant compute?
- **Model variants**:
  - **Wide**: 2× width (256 hidden, 1024 intermediate, 8 heads), ½ epochs (3)
  - **Narrow**: ½ width (64 hidden, 256 intermediate, 2 heads), 2× epochs (12)
  - **Wider**: 4× width (512 hidden, 2048 intermediate, 16 heads), ½ epochs (3) - Created 2025-11-21
- **Training regimes**:
  - Base pretraining on multitask_pt1
  - FTWB1: Fine-tuning on all 7 single tasks (ftwb1-{1-7})
  - FTWB2: Fine-tuning on selected two-task combinations (ftwb2-{2,4,9,12,13,15})
- **Current status** (2025-11-21):
  - ✅ Wide model: All training complete (base + 7 FTWB1 + 6 FTWB2 = 14 models)
  - ✅ Wide model: All evaluations complete (210 evals)
  - ✅ Wide model: All plots generated (3 plots)
  - ✅ Narrow model: Training complete (base + 7 FTWB1 = 8 models)
  - ❌ Narrow model: Evaluations not run
  - ⏳ Wider model: Config created, training pending
- **Evaluation infrastructure** (2025-11-21):
  - 420 evaluation configs (28 models × 15 evals)
  - 16 execution scripts (7 chunks per model type, ~30 evals each)
  - 3 plotting scripts (FTWB1 matrices, FTWB2 matrices, FTWB2-FTWB1 diff)
- **Wide model results** (2025-11-21):
  - FTWB1 (7×7): Trained=0.803±0.146, Transfer=0.491±0.267
  - FTWB2 (7×6): Trained=0.856±0.186, Transfer=0.496±0.274
  - FTWB2-FTWB1 diff: Trained=+0.042, Transfer=-0.056, Overall=-0.028
  - **Finding**: Two-task training performs similarly to single-task specialists
- **Visualization updates** (2025-11-21 17:06):
  - Updated all heatmaps to match exp1 publication-ready styling
  - **Matrix orientation**: Transposed to rows=experiments, cols=tasks
  - **Clean styling**: Removed colorbars (FTWB2/diff), titles, labels, tick marks, cell separators
  - **Font sizes**: 13.2pt for cell annotations (1.2x bigger)
  - **Training markers**: Added 'T' markers indicating which tasks were trained
  - **FTWB2 row ordering**: Distance-containing experiments on top 3 rows: [4, 13, 15, 2, 9, 12]
    - Row 1: exp 4 (crossing, distance)
    - Row 2: exp 13 (distance, perimeter)
    - Row 3: exp 15 (compass, distance)
    - Row 4-6: non-distance experiments (2, 9, 12)
  - Scripts: `plot_revision_exp3_ftwb{1,2}_heatmaps.py`, `plot_revision_exp3_ftwb2_vs_ftwb1.py`
- Output: `data/experiments/revision/exp3/pt1_{wide,narrow,wider}/` and fine-tuned variants

**Exp4: Single-task PT1-X Seed Robustness + Cross-Seed CKA Analysis**
- All 7 single-task experiments with multiple seeds (expanded 2025-11-20)
- **Training:** 21 total experiments base (7 tasks × 3 seeds: 1, 2, 3) + 4 additional pt1-5 seeds (4-7)
- **CRITICAL: PT1-5 (inside task) seed replacement** (2025-11-20):
  - Seed2 training failed for pt1-5 (inside task is brittle)
  - **21×21 matrix uses**: orig (seed42), seed1, **seed3** (NOT seed2)
  - **Future 28×28 uses**: orig, seed1, seed3, **seed4** (NOT seed2 or seed3)
  - Plot scripts updated to show "s3" instead of "s2" for pt1-5
- **21×21 CKA Matrix** (7 tasks × 3 seeds, pt1-5 uses seed3):
  - 21 experiments: 7×(orig + seed1 + seed2/seed3)
  - 231 unique pairs per layer
  - **Layers analyzed**: 3, 4, 5, 6
  - **Path fix (2025-11-20):** 147 configs corrected to point to `data/experiments/pt1-X/` instead of `revision/exp4/pt1-X/`
  - **CKA Analysis Status (2025-11-21 01:40)**:
    - Layer 3: 231/231 complete ✅
    - Layer 4: 231/231 complete ✅
    - Layer 5: 231/231 complete ✅
    - Layer 6: 231/231 complete ✅
  - **Completion Infrastructure (2025-11-21)**:
    - Created `FINAL_run_missing_layer3.sh` (25 final analyses)
    - User completed all remaining layer 3 analyses
    - All 21×21 matrices complete for layers 3, 4, 5, 6
- **PT1-5_seed4 Infrastructure** (2025-11-20):
  - 4 representation extraction configs (layers 3-6)
  - 112 CKA configs (28 pairs × 4 layers)
  - Compares pt1-5_seed4 vs all other experiments
  - Run script: `run_pt1-5_seed4_cka.sh`
- Clean CKA v2 infrastructure (`src/analysis/cka_v2/`):
  - Hierarchical output organization: `{group}/{pair}/layer{X}/`
  - CPU/GPU accelerated computation
  - Timeline visualization of representation similarity
  - Statistical summaries (mean, std, min, max)
- **Seed-specific execution scripts:**
  - `run_seed3_cka.sh` - All seed3 comparisons (prioritizes pt1-5 layer 5 first)
  - `run_pt1-5_seed4_cka.sh` - All pt1-5_seed4 comparisons (2025-11-20)
- **Visualization (Complete Infrastructure 2025-11-21 01:40):**
  - **Plotting scripts created for all layers:**
    - Layer 3: `plot_21x21_cka_matrix_l3.py`
    - Layer 4: `plot_21x21_cka_matrix_l4.py`
    - Layer 5: `plot_21x21_cka_matrix.py`
    - Layer 6: `plot_21x21_cka_matrix_l6.py`
  - **Shell wrappers:** `plot_21x21_cka_l{3,4,5,6}.sh`
  - **Generated plots per layer (6 total):**
    - Full 21×21 CKA matrix (PNG + CSV)
    - 7×7 task-averaged matrix without SEM
    - 7×7 task-averaged matrix with SEM
    - Bar plot comparing intra-task vs inter-task CKA
  - **7×7 matrix styling (2025-11-21)**:
    - Layer 6 updated to match layers 3-5 (added SEM calculation and with_sem plot)
    - Removed all axis labels for cleaner presentation
    - Font sizes: 24pt (without SEM), 18pt (with SEM)
    - Fixed SEM calculation: diagonal n=3 (unique cross-seed pairs), off-diagonal n=9
    - Two versions: with SEM (`mean ± sem`) and without SEM
  - **Bar plot enhancements (2025-11-21 01:40)**:
    - Fixed overlapping bars issue (was using empty category labels)
    - Added proper multi-line category labels with sample sizes
    - Implemented Welch's t-test for statistical significance testing
    - Added significance brackets with labels (ns, *, **, ***)
    - p-values displayed in plot titles
    - Error bars: SEM instead of STD
    - Text placement: on top of bars (better readability)
    - Grid lines and axis labels for clarity
  - **Statistical Results (All layers ns, p>0.05):**
    - Layer 3: Intra=0.338, Inter=0.259, p=0.313 (ns)
    - Layer 4: Intra=0.672, Inter=0.589, p=0.288 (ns)
    - Layer 5: Intra=0.796, Inter=0.698, p=0.122 (ns)
    - Layer 6: Intra=0.797, Inter=0.714, p=0.146 (ns)
  - **CKA-Generalization Correlation** (2025-11-20):
    - Updated to use absolute paths
    - Pearson r=0.405, p=0.0264, R²=0.164 (statistically significant)
    - 30 data points (6 tasks × 5 targets, excluding crossing)
- **PCA Timeline Configs** (2025-11-20):
  - 8 config sets: original, seed1, seed2, seed3 (mixed & raw each)
  - 55 total configs across all seeds
  - **Probe configuration:**
    - `probe_train`: Excludes Atlantis (no effect on probe training)
    - `probe_test`: Includes Atlantis (preserves color scheme)
  - 8 run scripts generated for all combinations
- Task-specific prompt handling: `distance_firstcity_last_and_trans`, `trianglearea_firstcity_last_and_trans`, etc.
- Output: `data/experiments/revision/exp4/pt1-{1-7}_seed{1,2,3}/` + `pt1-5_seed{4-7}/`
- CKA results: `data/experiments/revision/exp4/cka_analysis/`
- **PCA-based CKA and Procrustes Analysis** (2025-11-20 afternoon):
  - **Problem**: CKA on full representations shows similar inter-seed and inter-task variability despite 3D PCA plots showing clear geometric differences
  - **Root cause**: CKA is invariant to invertible linear transforms → can't distinguish different geometric shapes
  - **Solution**: Dual analysis approach
    1. **CKA on First 3 PCs**: Measure subspace similarity (1351 configs)
    2. **Procrustes Distance on First 3 PCs**: Measure geometric shape similarity (1351 configs)
  - **Implementation**:
    - Each experiment gets independent PCA (3 components, fitted on non-Atlantis)
    - CKA: Centered linear kernel, same as full analysis
    - Procrustes: Optimal rotation/scaling alignment, measures residual distance
  - **Expected results**:
    - CKA: Similar values (subspaces have similar dimension)
    - Procrustes: Low intra-task (same shape), high inter-task (different shapes)
  - **Output locations**:
    - `data/experiments/revision/exp4/cka_analysis_first3/`
    - `data/experiments/revision/exp4/procrustes_analysis_first3/`
  - **Scripts**:
    - `run_14x14_l5_first3.sh` for both CKA and Procrustes
    - Visualization with 14×14 and 7×7 matrices plus bar plots
    - Procrustes has both distance and similarity (1-distance) plots
- **Status**: 21×21 matrix complete, pt1-5_seed4 infrastructure ready, PCA-based metrics ready to run

Total created: ~200+ config files, ~150+ shell scripts, 7 meta scripts for training
**Exp4 expansions (2025-11-20)**:
- Seed3 infrastructure (28 repr extraction + 7 PCA + 525 CKA configs)
- Layer 6 analysis (231 additional CKA configs for 21×21)
- PT1-5 seeds 4-7 (4 training configs + scripts)
- Seed-specific CKA run scripts (3 scripts for selective execution)
- **PCA-based metrics** (2025-11-20 afternoon): 1351 CKA configs + 1351 Procrustes configs (first 3 PCs)

### Key Technical Details

#### FTWB Fine-tuning Protocol
Discovered that proper fine-tuning uses special datasets (ftwb1-{1-7}) combining:
- 20k samples of main task (no Atlantis)
- 100k samples of main task (Atlantis required)
- 256 samples each of all 7 tasks for "warmup" to prevent catastrophic forgetting
- Lower learning rate: 1e-5 (vs 3e-4 for pretraining)
- Fewer epochs: 30 (vs 42 for single-task pretrain)
- Loads from pretrained checkpoint

#### Representation Extraction
- **Multi-layer analysis** (2025-11-20): Layers 3, 4, 5, 6 for all 63 trained models
- Layer 5 was initial focus (middle layer for 6-layer models)
- Prompt format: `{task}_firstcity_last_and_trans`
- 4500 cities total (3250 train, 1250 test)
- Extracts from last checkpoint (-2)
- **Complete coverage status** (2025-11-20):
  - PT1-X (21 models): ✓ All have layers 3,4,5,6
  - PT2 original (8 models): ✓ All have layers 3,4,5,6
  - PT3 original (8 models): ✓ All have layers 3,4,5,6
  - PT2 seeds (14 models): Infrastructure ready (105 configs created)
  - PT3 seeds (14 models): Infrastructure ready (105 configs created)

#### Dataset Structure
- Train cities: 3250 cities from real world locations
- Test cities: 1250 held-out cities
- Special location: "Atlantis" (fictitious city) used to test generalization
- City encoding: `c_{id}:(lat,lon)` format

### Main Findings So Far

1. **Multi-task vs Single-task**: Multi-task pretraining (PT1) creates more generalizable representations than single-task pretraining (PT1-X)

2. **Task Interactions**: Two-task combinations (PT2) show interesting patterns - some task pairs complement each other while others interfere

3. **Representation Structure**: PCA visualizations reveal that:
   - Geographic structure emerges naturally in representation space
   - Different layers capture different aspects of spatial relationships
   - Linear probes can align representations to geographic coordinates

4. **Fine-tuning Dynamics**: FTWB protocol successfully transfers knowledge while maintaining baseline performance on other tasks through warmup samples

5. **Training Dynamics**: Representations evolve continuously during training, with modularity increasing over time

6. **Distance Task Impairs OOD Generalization** (2025-11-21): Linear probe generalization to Atlantis (OOD) cities is ~5x worse for models trained WITH distance task (508 error) vs WITHOUT (108 error). This suggests distance task causes representations to overfit to training city relationships, reducing transferability to novel locations.

### Rebuttal Strategy (2025-11-20)

**Addressing Reviewer Concern**: "PCA visualizations show clear task structure but quantitative metrics don't capture this"

**Our Response**:
1. **Problem identified**: CKA alone is too invariant
   - CKA measures subspace similarity, blind to geometric shape
   - Same intrinsic dimension → high CKA, even if shapes differ

2. **Dual metric approach**:
   - **CKA on first 3 PCs**: "Do tasks span similar subspaces?" (dimensional similarity)
   - **Procrustes on first 3 PCs**: "Do tasks have same geometric shape?" (structural similarity)

3. **Expected rebuttal impact**:
   - Low Procrustes intra-task variance → geometric consistency across seeds
   - High Procrustes inter-task distance → distinct geometric structures per task
   - Supports claim: "Task representations have consistent, interpretable geometry"

### Recent Visualization Work

**Exp1 Heatmap Styling** (2025-11-21 evening):
- Updated all FTWB1, FTWB2, and comparison plots for publication
- **Key changes**:
  - Transposed matrices: rows=experiments (trained on), cols=tasks (evaluated on)
  - Removed all non-essential elements: colorbar, titles, axis labels, tick labels
  - Increased font size to 13.2pt for better readability
  - Added 'T' markers to indicate trained tasks
  - Created aggregated versions averaging across all 4 seeds
- **Results documented**:
  - FTWB1 aggregated: Trained=0.838±0.136, Transfer=0.474±0.292
  - FTWB2 aggregated: Trained=0.897±0.142, Transfer=0.601±0.278
  - FTWB2-FTWB1 aggregated: Trained diff=+0.054, Transfer diff=-0.019, Overall=+0.002
- **Key insight**: Two-task training achieves ~max performance of single-task specialists
- All plot scripts now in consistent style: `plot_revision_exp1_ftwb{1,2}_heatmaps.py`, `plot_revision_exp1_ftwb2_vs_ftwb1.py`

**Exp5: PT1 Trained with Atlantis from Scratch** (2025-11-21 18:23):
- **Research question**: Does training with Atlantis from scratch prevent OOD generalization issues?
- **Dataset**: `multitask_pt1_with_atlantis` - identical to `multitask_pt1` but includes Atlantis cities in training
- **Training**: Single seed (42), same architecture as PT1
- **Infrastructure created**:
  - Representation extraction: 4 configs (layers 3,4,5,6), 5 scripts
  - PCA timeline: 3 configs (mixed, na, raw variants for layer 5), 1 script
  - Probe generalization: 1 config, 1 script
  - Histogram plotting: `plot_probe_generalization_histogram_with_exp5.py`
- **Key Result**:
  - **Exp5 Atlantis error: 23.2** (very close to baseline 18.6)
  - Compare to exp1 models NOT trained with Atlantis:
    - Atlantis without distance task: 107.8 ± 104.9
    - Atlantis with distance task: 508.3 ± 288.3
- **Conclusion**: Training with Atlantis from scratch eliminates OOD generalization issues, confirming hypothesis that poor generalization stems from training distribution, not model limitations
- **Status**: Training complete, probe gen complete, histogram generated
- **Output**: `data/experiments/revision/exp5/pt1_with_atlantis/`

**Exp6: Scattered Atlantis Experiment** (2025-11-27):
- **Research question**: Are the observed effects (e.g., distance task impairing OOD generalization) due to Atlantis being clustered in one location?
- **Motivation**: Reviewer concern that Atlantis clustering at (-35, 35) might explain findings
- **Method**: Scatter 100 Atlantis cities uniformly across the world (x: [-180, 180], y: [-90, 90]) instead of Gaussian clustering
- **Code changes**: Modified `src/data_processing/create_city_dataset.py` to support `scattered_atlantis` config key with uniform random distribution
- **Infrastructure created**:
  - **Data generation**: 50 configs (city dataset + 21 task datasets + 28 combined datasets)
  - **Data scripts**: step1 (cities), step2-{1,2,3} (tasks in parallel), step3 (combining)
  - **Training**: PT1 + 7 FTWB1 + 21 FTWB2 configs and scripts
  - **Visualization**: City scatter plot to verify uniform distribution
- **Output locations**:
  - Datasets: `data/experiments/revision/exp6/datasets/`
  - PT1 model: `data/experiments/revision/exp6/pt1/`
  - FTWB models: `data/experiments/revision/exp6/pt1_ftwb{1,2}-*/`
- **Status**: ⏳ Infrastructure complete, data generation in progress

**Additional Plots Infrastructure** (2025-11-21 18:23):
- Created new visualization folders: `scripts/revision/additional_plots/`, `configs/revision/additional_plots/`, `data/experiments/revision/additional_plots/`
- **Basic Cities Plot** (`src/scripts/plot_cities_basic.py`):
  - Scatters all 5,175 cities colored by region
  - Uses exact plotly color scheme (Plotly + D3 + G10 qualitative colors) for matplotlib compatibility
  - Styling: thick spines (3px), thick ticks (3px), bold tick labels (16pt), no axis labels, no legend
  - Output: `data/experiments/revision/additional_plots/cities_basic/figures/cities_basic.png`
- **Probe Atlantis Predictions Plots** (`src/scripts/plot_probe_atlantis_predictions.py`):
  - Shows predicted vs true Atlantis locations from linear probe generalization
  - World cities colored by region, Atlantis true (red circles), predicted (black X)
  - Created for 9 FTWB2 experiments (5 with distance, 4 without distance)
- **3D Nullspace Visualization** (`src/scripts/plot_probe_3d_nullspace.py`, experimental):
  - Dim1: X linear probe, Dim2: Y linear probe, Dim3: direction orthogonal to world cities plane
  - Method: Pick 10 cities each from Boston/NYC, South Africa, North China; compute plane normal via SVD
  - Results inconclusive - both WITH and WITHOUT distance show similar Atlantis offset

**CRITICAL FIX: FTWB2 Task Mapping** (2025-11-21):
- Fixed incorrect task mapping in `docs/canonical_experiments.md`
- Original mapping was systematically wrong; corrected based on actual dataset configs (`data/datasets/ftwb2-X/combine_config.yaml`)
- Key corrections: ftwb2-13 = distance+perimeter (NOT perimeter+crossing), ftwb2-2 = angle+compass (NOT distance+angle)
- Impacts all probe generalization analysis interpretations

**Training Dynamics Visualization** (2025-11-21 02:16):
- Created comprehensive 3-panel vertical plots for all PT1 experiments (PT1-1 through PT1-7)
- **Panel structure**:
  - Top: Training and validation loss curves (log-log scale)
  - Middle: Task-specific metric (left y-axis) and Mean Coordinate R² (right y-axis, dual-color)
  - Bottom: Mean distance error to ground truth coordinates (log-log scale)
- **Key features**:
  - Per-experiment adaptive y-axis ranges and ticks for optimal visualization
  - Thick black spines (3px), large labels (28pt) for publication quality
  - Removed grid lines and x-axis ticks from top/middle panels
  - Thicker plot lines (5px) for visibility
  - Fixed distance error yticks [50, 100, 500] for consistency
- **Data sources**:
  - Loss: `trainer_state.json` from checkpoints
  - Task metrics: `eval_data/*.jsonl` files
  - R² and distance: `representation_dynamics.csv` from analysis_higher
- **Key finding**: PT1-7 (crossing task) failed to learn coordinate representations (R² = -0.075), all others converged successfully (R² 0.61-0.99)
- **Output**: `scratch/formation_dynamics/figures/pt1-{1-7}_vertical.png`

### Pending Analysis

The following experiments are set up but not yet run:
- **Exp4 Layer 3 and Layer 6 CKA calculations** (ready to run, 2025-11-21):
  - Layer 3: 231 calculations split across 3 scripts (~77 each)
  - Layer 6: 21 calculations (all pt1-5_seed3 pairs)
  - Scripts: `run_missing_cka_layer3_part{1,2,3}.sh` + `run_missing_cka_layer6.sh`
  - Expected runtime: Several hours (layer 3), ~30 min (layer 6)
  - Required for: Complete 21×21 CKA matrices across all 4 layers
- **PT2/PT3 multi-layer representation extraction** (ready to run, 2025-11-20):
  - 105 configs for layers 3,4,5,6 across PT2/PT3 seeds
  - Master script: `bash scripts/revision/exp2/extract_pt2_pt3_all_multilayer.sh`
  - Expected runtime: ~12 hours for all extractions
  - Required before: Exp2 seed CKA analysis
- **Exp2 seed CKA analysis** (ready to run, 2025-11-21):
  - 252 calculations for PT2/PT3 non-overlapping pairs
  - Parallel execution: 4 chunks of 63 calculations each
  - Scripts: `bash scripts/revision/exp2/cka_analysis/run_exp2_seed_cka_chunk{1,2,3,4}.sh`
  - Expected runtime: ~4-6 hours per chunk
  - Required for: 1→2→3 task regime analysis with seed robustness
- **Exp4 PCA-based metrics** (ready to run, 2025-11-20):
  - 14×14 CKA analysis on first 3 PCs (original + seed1, layer 5)
  - 14×14 Procrustes analysis on first 3 PCs (original + seed1, layer 5)
  - Visualization scripts ready for both metrics
  - Expected runtime: ~2-3 hours for full analysis
- **CKA trends analysis** (pending PT2/PT3 extraction):
  - Cross-experiment CKA for non-overlapping task pairs
  - Follows pattern from world-representation repo
  - Will show learning dynamics across PT1-X, PT2, PT3
- Statistical analysis of seed robustness
- Width ablation analysis (Exp3) to determine capacity effects
- Comparative analysis of PT1 vs PT1-X vs PT2 vs PT3 representation quality

## Research Direction

### Immediate Next Steps

**Critical Priority** (Exp1 FTWB1):
1. **Train FTWB1 models** (21 training runs, ~several hours)
   - Execute: `bash scripts/revision/exp1/training/train_all_ftwb1_sequential.sh`
   - Essential for computing expected generalization baseline
2. **Evaluate FTWB1 models** (315 evaluations after training)
   - Execute: `bash scripts/revision/exp1/eval/eval_all_ftwb1.sh`
3. **Extract FTWB1 representations** (21 extractions)
   - Execute: `bash scripts/revision/exp1/representation_extraction/extract_all_ftwb1.sh`
4. **Update Exp1 plotting script** to include expected generalization and generate Actual - Expected difference plots

**Rebuttal Preparation** (Exp4):
5. ~~**Run Exp4 missing CKA calculations**~~ ✅ **COMPLETE** (2025-11-21 01:40)
   - All 231 pairs complete for layers 3, 4, 5, 6
   - Full 21×21 CKA matrices available for all 4 layers
6. ~~**Generate layer 3 and 6 visualizations**~~ ✅ **COMPLETE** (2025-11-21 01:40)
   - Created plotting scripts for all layers (3, 4, 5, 6)
   - Fixed bar plot overlap issues
   - Added significance testing (Welch's t-test)
   - Generated all 24 plots (6 per layer × 4 layers)
   - All saved to `data/experiments/revision/exp4/cka_analysis/`
7. **Run PT2/PT3 multi-layer representation extraction** (105 configs, ~12 hours)
   - Execute: `bash scripts/revision/exp2/extract_pt2_pt3_all_multilayer.sh`
   - Verify all extractions complete successfully
8. **Run Exp4 PCA-based metrics** (CKA + Procrustes on first 3 PCs)
9. **Set up CKA trends analysis** for non-overlapping task pairs
10. Generate side-by-side comparison plots for rebuttal
11. Analyze seed robustness: intra-task vs inter-task variance (complete after layer 3/6)
12. ~~Draft rebuttal text explaining complementary metrics~~ ✅ **IN PROGRESS** (2025-11-22)
13. Compare wide vs narrow model performance (Exp3)
14. Synthesize findings for paper revision

### Rebuttal Writing Session (2025-11-22)
**Paper submitted to ICLR 2026, received reviews, now writing rebuttals.**

**Completed Reviewer Responses:**
- **PuHx**: Complete response addressing all 4 weaknesses and 3 questions
  - W1 (small figures), W2 (missing citations), W3 (quantitative integration metric), W4 (figure 8a interpretation)
  - Q1 (LLM implications), Q2 (CKA interventions), Q3 (non-overlapping pairs in CKA)
  - Added citations: Berglund et al. (reversal curse), Lampinen et al. (knowledge updating)

- **8dPS**: Complete response addressing all 5 weaknesses and 3 questions
  - W1 (single seed) - Major response clarifying we never claimed visual=CKA, results replicate across 63 models
  - W2 (city ID assignment), W3 (clustered Atlantis), W4 (2D planar world), W5 (divergent task explanation)
  - Q1 (crossing task), Q2 (linear probe accuracies), Q3 (normalized improvement definition with formulas)
  - Added 7 citations: Bachmann, Engels & Tegmark, Csordás, Hoffmann, Gopalani, Kim, Lester

**Completed Reviewer Responses (Evening Session):**
- **oLe1**: Complete - W1 (introduction meandering), On Surprisingness, Q1 (regularization/training factors), Q2 (model size)
  - Added citations: Pezeshki et al. (2021), Shah et al. (2020) for gradient plateau literature
  - Key points: Introduction revised (removed LLM debate, added Related Work section), representations stabilize early (App. Fig. 16), 2× wider model still shows divergent task pattern (App. Fig. 20)

**Completed Reviewer Response (2025-11-23):**
- **taAU**: Complete - W1-W5, Minor Points, Q1-Q4 all addressed
  - W1: Systematic PRH→MSH reframing with detailed table of all changes
  - W2: Comprehensive literature survey (7 papers reviewed showing gap)
  - W3: Acknowledged Section 3 may be unsurprising, defended Section 4 findings
  - W4: Agreed correlation is weak, pivoted to divergent task finding
  - W5: Defended 7 tasks (enables exhaustive analysis of all combinations)
  - Q1: Tokenization details documented in App. B
  - Q2: Cross-seed consistency (distinct from 8dPS response)
  - Q3: Crossing task failure with citations for loss plateau literature
  - Q4: Linear decoding figure reference clarified
  - Added 16 citations total

**General Rebuttal Section (2025-11-22, 23:00):**
- Complete rewrite of General section with professional structure
- Summarized 272 new models across 6 experiment types
- Listed 6 major paper revisions
- Key findings from multi-seed: core results replicate, multi-task reduces variance, R² improves 0.126→0.188
- Toned down PRH claims in paper contributions list to match body text

**Key Rebuttal Strategies:**
1. Strong defense on single-seed criticism - we never claimed visual similarity = high CKA
2. Honest about not knowing why distance task is divergent
3. Mentioned scattered Atlantis experiment as in-progress
4. Added "ask for score increase" at end of summaries
5. Expressed genuine gratitude for multi-seed suggestion (improved paper)

**Rebuttal Location:** `rebuttal/REBUTTAL.txt`

**Rebuttal Status (2025-11-23 00:18):**
- **All 4 reviewer responses complete**
- **General section complete**
- **Ready for final review and submission**

### Long-term Research Goals
1. **Scaling Laws**: Establish quantitative relationships between data/model scale and representation modularity
2. **Critical Hyperparameters**: Identify key factors determining modular vs fractured representations
3. **Generalization Principles**: Develop guidelines for training processes that encourage modular representations
4. **Extension to Real Data**: Validate findings on more complex, real-world datasets

### Open Questions
1. What is the minimum dataset size for modular representations to form?
2. How does the number of tasks in multi-task training affect representation quality?
3. Can we predict which task pairs will complement vs interfere?
4. What architectural changes promote more modular representations?
5. **How stable are these findings across different random seeds?** ← Exp4 CKA analysis will address this
6. Do same-task different-seed representations show high CKA similarity (robustness)?
7. Do different-task same-seed representations show low CKA similarity (task-specificity)?
8. Does the 21×21 CKA matrix reveal clustering patterns (e.g., similar tasks group together)?

## Related Work

This work builds on several research threads:
- **Fractured Representations**: "Questioning Representational Optimism in Deep Learning: The Fractured Entangled Representation Hypothesis"
- **Algorithmic Dynamics**: "Competition Dynamics Shape Algorithmic Phases of In-Context Learning" (arXiv:2412.01003)
- **Synthetic Data**: Recent work (arXiv:2410.11767) on using synthetic data for representation study

## Project Context

This research is part of a broader investigation into representation formation conditions, using geographic space as an initial testbed. The long-term vision includes:
- Extension to PCFG and HHMM-based synthetic data generation
- Systematic hyperparameter sweeps
- Study of RL and fine-tuning effects on representations
- Development of general principles for encouraging modular representations

See `docs/research_proposal.md` for the complete research agenda.

---

**Note**: This document reflects the current state as of 2025-11-22. Major updates should be documented with date stamps.

---

## Paper Editing Session (2025-11-22)

Extensive editing of `iclr2026_conference.tex` focusing on clarity, terminology, and addressing reviewer feedback.

### Key Changes
- **New terminology**: "best-teacher model" for the heuristic expectation in fine-tuning, "divergent tasks" (bold definition), "hidden spaces" (bold definition)
- **Hypothesis block**: Formatted the fine-tuning hypothesis in italicized quote block matching Multitask Scaling Hypothesis style
- **Section titles**: Shortened Section 3 and 4 titles for single-line fit
- **Citations**: Added McCloskey 1989 for catastrophic forgetting, Kumar 2022 for gradient descent distortion
- **Consistency**: All "Atlantis" now uses `\texttt{Atlantis}`, all "SGD" changed to "gradient descent"
- **Figure captions**: Updated result1-1, result1-2, result2-1, result2-2 with correct panel descriptions, appendix references
- **Result 3 rewrite**: Clearer explanation of linear probe analysis on exemplar models (with/without distance task)
- Paper compiles at 24 pages with some undefined reference warnings for appendix figures to be created

### Appendix Writing Session (2025-11-22, 05:40-07:55)

Focused on writing and improving appendix sections, grounding descriptions in actual code implementations.

#### Appendix C: Experimental Details
- **World Section**: Added discussion of flat 2D manifold vs spherical choice
  - Early experiments used spherical coordinates
  - No canonical geometry for nonlinear models
  - Planar enables clean linear probing
  - Cited Engels et al. and Csordas et al. (nonlinear features), Bronstein et al. (geometric deep learning)
- **Dataset Sizes**: Full fine-tuning composition (100k target task, 20k pretraining replay, 256 per task warmup)
- **Tokenization**: Justified character-level tokenization, cited Bachmann et al. (2025) on pitfalls of next-token prediction
- **City ID Assignment**: Random assignment ensuring no geographic information leakage

#### Appendix D: Analysis Methods
- **Representation Extraction**: Layer extraction info (residual stream, layers 3-6, layer 5 default), added `<bos>` token to example
- **Omitting cities with leading zeros**: Explained why 0 is special (never leading digit in numerical outputs)
- **Linear Probing & PCA**: Grounded in actual code - 3250/1250 train/test split, OLS without regularization, color by geographic region
- **Reconstruction Error**: Measured as absolute Euclidean distance
- **CKA**: Rewrote with actual formula from code, city filter details, layer info

#### Compilation Fixes
- Added `\usepackage{booktabs}` for table formatting
- Defined `\degree` command
- Fixed package clashes, removed duplicate citations

### Extended Related Works Revision (2025-11-22, 16:00-16:40)

Complete revision of Extended Related Works section (Appendix), removing duplicates and writing proper prose for all paragraphs.

#### Changes Made
- **Interpretability & Internal Representations**: Removed duplicates from main text, kept unique neuroscience roots (hubel1962receptive, fukushima1980neocognitron), added olah2017feature (feature visualization), vafa2025foundationmodelfoundusing (replaces 2024 duplicate)
- **Fine-tuning**: Condensed to unique directions not in main text (parameter efficiency, zeroth-order optimization, weight composition, representation adaptation), added zhao2025echochamber, zweiger2025selfadapting
- **Dynamics of Representations** (NEW prose): How representations evolve during ICL (park2024icl, shai2025transformersrepresentbeliefstate, demircan2024sparseautoencoders) and fine-tuning (casademunt2025steering, minder2025overcomingsparsity); SAE temporal limitations (lubana2025priorsintime); VLM unused representations (fu2025hiddenplainsightvlms)
- **Geometric Deep Learning** (NEW prose): Cited foundational equivariant works (bronstein2021, cohen2016groupequivariant, weiler2021generale2equivariant); explicitly stated we study emergent geometry, not geometric inductive biases
- **Loss Plateaus** (NEW prose): Connected to our crossing task failure; transformer-specific studies (hoffmann2024eureka, gopalani2025, singh2024needsrightinductionhead); general optimization (shah2020, pezeshki2021, bachmann2025); kim2025taskdiversity most related (multi-task shortens plateaus - similar to crossing finding)
- **Removed**: "Extended Background" paragraph (content moved or redundant), "Other" paragraph (fu2025 moved to Dynamics)

#### New Citations Added
- shai2025transformersrepresentbeliefstate
- demircan2024sparseautoencodersrevealtemporal
- lubana2025priorstimemissinginductive
- vafa2025foundationmodelfoundusing
- cohen2016groupequivariantconvolutionalnetworks
- weiler2021generale2equivariantsteerablecnns
- olah2017feature
- singh2024needsrightinductionhead

### Current Paper Status
- Paper compiles at 30 pages with no undefined citations
- All appendix sections complete with proper prose
- Extended Related Works cleaned: no duplicates from main text, all paragraphs have prose

### World Map Without Atlantis Figure (2025-11-22, 16:50)
- Created `src/scripts/plot_cities_world_no_atlantis.py` - variant of `plot_cities_basic.py` that excludes Atlantis cities
- Same styling: thick spines, bold tick labels, no legend, no axis labels
- Filters out 100 Atlantis cities, leaving 5,075 world cities
- Output: `data/experiments/revision/additional_plots/world_no_atlantis/figures/world_no_atlantis.png`

### Discussion Section Rewrite (2025-11-22, 18:00)
Major restructuring and rewriting of the Discussion section with extensive new citations and refined narrative.

#### New Discussion Structure (4 paragraphs)
1. **Continual learning and world models**: Motivation from fundamental DNN properties toward general intelligence; world models must adapt consistently when world changes; "cascading updates across different computational tasks"; ICL vs fine-tuning gap with recent approaches (augmented transformers + stateful architectures)
2. **Dynamics of representations**: Literature survey from Rosenblatt/Rumelhart to modern ICL dynamics (Park 2024, Shai 2025) and fine-tuning representations (Wang 2025, Casademunt 2025, Minder 2025); sells our framework as defining "updatable world with consistent expected changes"
3. **Forward and backward modularity**: Key insight that forward modularity ≠ backward modularity; multi-task creates clean representations but "fractured and partial" for adaptation; ICL sidesteps by operating in forward pass only
4. **Limitations**: Benefits first (holistic pipeline analysis, non-trivial phenomenology); correlational findings acknowledged; PRH claims partial (single architecture/modality, no *true* multimodality or cross-architecture convergence)

#### New Citations Added (18 total)
- **Stateful architectures**: hochreiter1997long (LSTM), schlag2021lineartransformerssecretlyfast, behrouz2024titanslearningmemorizetest, yang2025gateddeltanetworksimproving
- **Transformer augmentations**: chen2024generativeadaptercontextualizinglanguage, charakorn2025texttolorainstanttransformeradaption, zweiger2025selfadaptinglanguagemodels
- **Representation dynamics**: wang2025simplemechanisticexplanationsoutofcontext, bigelow2025beliefdynamicsrevealdual
- **Critical learning/plasticity**: achille2019criticallearningperiodsdeep, dohare2024maintainingplasticitydeepcontinual

#### Also Added
- Citation to Brown 2020 for in-context learning
- Citation link from Result 1 (early representation saturation) to critical learning periods literature
- Removed old "Continual world models" paragraph (key vocabulary merged into paragraphs 1 and 3)

### Final Paper Revisions (2025-11-22, 18:30)
Final editing pass on paper, toning down claims and fixing stylistic issues.

#### Key Changes
1. **Title**: Changed to "Origins and Roles of World Representations in Neural Networks" (restored original)
2. **Result 3 title**: Removed "Evidence for the Platonic Hypothesis" → now just "Task diversity aligns representations"
3. **PRH claims toned down**: "directly connects" → "partially connects"; "ideal testbed" → "potential testbed"
4. **Forward/backward modularity**: "often fractured" → "can be fractured"; removed speculative sentences about ICL
5. **Limitations section**: Rewritten to start with "We study world representation formation and adaptation in a controlled synthetic setting with small-scale models..."
6. **Conclusion**: Completely rewritten - leads with framework properties, then multi-task convergence, then divergent tasks finding
7. **Em-dash cleanup**: Reduced from 7 to 3 in main text (kept only purposeful ones at key contrast points)
8. **Geometric Deep Learning appendix**: Added explanation for 2D planar choice - "geometry can be absorbed into task definition"
9. **BibTeX fix**: Removed duplicate `zweiger2025selfadaptinglanguagemodels` entry

#### Final Paper State
- **31 pages**, compiles successfully
- All major claims appropriately calibrated
- Ready for submission
- Zipped to `paper_submission.zip`

### Paper Polish and Anonymization Check (2025-11-22, 20:14)
Final polish pass before submission, focusing on citation style, terminology consistency, and anonymization verification.

#### Changes Made
1. **Citation style fixes** (`\cite` → `\citet`): Fixed 5 instances where citations were sentence subjects
2. **"Related Works" → "Related Work"**: Fixed section title to standard academic convention (singular)
3. **Task naming consistency**: Standardized to `\texttt{triangle area}` in prose, `\texttt{triarea}` only in technical tables
4. **Self-citation reduction** (8 → 4): Removed redundant self-citations, kept only most relevant instances
5. **Discussion section label**: Fixed `\label{app:discussion}` → `\label{sec:discussion}`
6. **Footnote wording**: "We believe" → "We regard" (more assertive)
7. **Quantitative precision**: "roughly 90% of training" → "first ~15% of training" (measured)
8. **Hedging removal**: Removed "It is unclear if this is generally true"
9. **Citation spacing**: Fixed `GeoNames\cite{...}` → `GeoNames~\citep{...}`
10. **Figure reference**: Added `(Fig.~\ref{fig:7taskmodel})` to 7-task model discussion

#### Anonymization Verified
- `\iclrfinalcopy` commented out - author info hidden in PDF
- PDF shows "Anonymous authors" and "Paper under double-blind review"
- All self-citations use third person (no "our prior work", "we previously showed")
- No identifying info in comments or file paths

#### Final State
- Paper compiles at 30 pages
- Created fresh `paper.zip` with all changes
- Ready for anonymous submission

---

## ICML 2026 Pivot (2026-01-27)

**ICLR 2026 rejected.** Pivoting to ICML 2026 submission (deadline: Jan 28 AoE).

### Senior Researcher Advice
- The divergent task observation is the most interesting finding
- Keep it simple - situate in 30-year grounding debate (Harnad 1990, Bender & Koller 2020, LeCun 2022)
- Core message: "Relational data leads to emergent world models, but not all relations are equally informative. Some can weaken world model formation."
- Missing: theory for why distance hurts - if explained, could be best paper material

### Key Changes Made

#### Title Revision
- **Old**: "Origins and Roles of World Representations in Neural Networks"
- **New**: "Convergent World Representations and Divergent Tasks"
- Rationale: Captures core tension between convergence (multi-task) and divergence (distance task)

#### Abstract Rewrite
- Restructured to lead with convergence/divergence findings
- New punchline: "training on multiple relational tasks reliably produces convergent world representations, but some lurking divergent tasks can catastrophically harm new entity integration via fine-tuning"

#### Style
- Removed all Oxford commas throughout paper (preference)

### Editorial Ick Points Added
Comments marked with `%##ICKPOINT##` for future revision:
1. "This work" paragraph doesn't mention divergent tasks
2. Contribution bullet 2 buries divergence story
3. Discussion is dense - needs tightening
4. Conclusion doesn't land the punchline

### Key Speculation: Why Is Distance Divergent?
Added `%## KEY SPECULATION ##` comment block in Discussion:
- **Hypothesis**: Divergence is property of task-architecture pairing (gradient geometry), not learned weights
- The gradient signal from certain tasks inherently routes updates through pathways that bypass shared representations
- This explains why single-task CKA predicts fine-tuning failure even after joint multi-task pretraining
- Left mechanistic understanding to future work

### ICML 2026 Submission Info
- Abstract deadline: January 23, 2026 AoE (passed)
- Full paper deadline: January 28, 2026 AoE
- 8 pages main body (strict limit)
- Paper location: `paper_icml/`

### Paper Polish Session (2026-01-27 07:46)
**Discussion section tightening:**
- Paragraph 1 (Continual learning): ~180 → ~90 words
- Paragraph 2 (Dynamics): ~150 → ~80 words, cut literature descriptions
- Paragraph 3 (Forward/backward): Added italic key insight
- NEW Paragraph 4 (Future work): Gradient geometry hypothesis (~40 words)

**Other changes:**
- Simplified Limitations (diagnostic marker, no mechanism)
- Added Sec 5 closing hypothesis (task-architecture pairing)
- Rewrote Conclusion (~70 words, echoes title's convergent/divergent)
- PRH abbreviation after first main text appearance
- Removed all ick point comments
- Added li2025justintimedistributedtaskrepresentations citation

**Status:** Compiles at 21 pages, pushed to main

### Appendix Restoration Session (2026-01-27 11:10)
**Problem discovered:** ICML appendix was missing significant content from ICLR version.

**Content restored from ICLR:**
- Normalized Improvement equations (error-based and accuracy-based formulas)
- Representation Extraction details (colorbox token example, leading zeros paragraph)
- Linear Probing & PCA paragraphs (probing, PCA, reconstruction error)
- CKA formula and computation details
- All section explanations (Training Dynamics, Representation Dynamics, CKA results, Pretraining Variations)

**Extended Related Work merge:**
- Had duplicate sections at lines 374 and 683
- Merged into single 7-topic section: Internal Representations, Fine-tuning, Multi-task Learning, Synthetic Data, Dynamics of Representations, Geometric Deep Learning, Loss Plateaus
- All citations verified from ICLR bib file (not hallucinated)

**Status:** Compiles at 41 pages, pushed to Overleaf

### Single-Column Appendix Conversion (2026-01-28 04:06)
**Change:** Converted appendix from two-column to single-column format per ICML guidelines.

**Modifications:**
- Added `\onecolumn` after `\appendix`
- Removed 4 `\clearpage` commands
- Changed figure placements from `[!h]`/`[H]` to `[htbp]`
- Fixed `\textdegree` in math mode (now uses `\degree`)

**Status:** Compiles at 24 pages (8 main + 16 appendix), pushed to Overleaf
