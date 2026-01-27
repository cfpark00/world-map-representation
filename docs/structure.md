# Project Structure

## Track-Based Organization (2026-01-23)

The codebase is being reorganized from a flat structure to track-based organization.
See `docs/REORGANIZATION.md` for the full plan and status.

### Completed: data_generation_v1
```
src/data_generation_v1/
├── __init__.py
├── tasks/                    # 7 task implementations
│   ├── angle.py
│   ├── compass.py
│   ├── crossing.py
│   ├── distance.py
│   ├── inside.py
│   ├── perimeter.py
│   └── trianglearea.py
├── create_city_dataset.py
├── combine_datasets.py
├── append_cities_to_dataset.py
├── create_tokenizer.py
└── utils.py                  # Track-specific utilities

configs/data_generation_v1/
├── cities/                   # 1 yaml
│   └── city_dataset_default.yaml
├── tokenizers/               # 1 yaml
│   └── default_tokenizer.yaml
├── single_tasks/             # 21 yamls (7 tasks × 3 variants)
│   ├── {task}_1M_no_atlantis.yaml
│   ├── {task}_1M_with_atlantis.yaml
│   └── {task}_100k_atlantis_required.yaml
└── derived/
    ├── pretraining/          # 18 yamls (pt7, pt2-{1..8}, pt3-{1..8})
    └── finetuning/           # 70 yamls (ft1-7, ft2-21, ft3-7, ftwb1-7, ftwb2-21, ftwb3-7)

scripts/data_generation_v1/
├── cities/
├── tokenizers/
├── single_tasks/
└── derived/

data/data_generation_v1/      # Output (gitignored)
├── cities/                   # City dataset (5,175 cities)
├── tokenizers/               # Tokenizers (98 tokens)
├── single_datasets/          # Individual task datasets
└── derived_datasets/         # Combined/mixed datasets
```

### Planned Tracks
- `pretraining_v1` - All pretraining scripts
- `finetuning_v1` - All fine-tuning scripts
- `cka_v1` - CKA analysis infrastructure

---

## Legacy Structure (pre-2026-01-23)

The structure below documents the codebase before track-based reorganization.
Some paths may be outdated.

```
.
├── configs
│   ├── analysis_3d_plots        # 3D visualization analysis configs
│   ├── analysis_activation_gradients  # Gradient analysis configs for token activations
│   ├── analysis_cka            # CKA analysis configs for model comparisons
│   ├── analysis_cka_l3         # CKA configs for layer 3 (pt1, pt2, pt3)
│   ├── analysis_cka_l4         # CKA configs for layer 4 (pt1, pt2, pt3)
│   ├── analysis_cka_l6         # CKA configs for layer 6 (pt1, pt2, pt3)
│   ├── analysis_cka_pt2        # CKA configs for pt2 experiments
│   ├── analysis_cka_pt3        # CKA configs for pt3 experiments
│   ├── analysis_dimensionality # Intrinsic dimensionality analysis (TwoNN, correlation dim)
│   │   ├── pt1                 # PT1 single-task model configs
│   │   └── pt2                 # PT2 multi-task model configs
│   ├── analysis_kernel         # Kernel matrix computation configs
│   ├── analysis_manifold       # Manifold analysis configs
│   ├── analysis_pca_timeline    # PCA timeline visualization configs
│   ├── analysis_representation  # Representation analysis configs
│   │   └── randomwalk_pretrain_no_atlantis_15ep_llr_pad
│   │       ├── 0-5              # Layer 0-5 configs
│   │       └── regions          # Regional analysis configs
│   ├── analysis_representation_higher  # Stripped representation analysis (no world map)
│   │   ├── seed1                # Seed1 PT1-X representation extraction configs (2025-11-19)
│   │   ├── seed2                # Seed2 PT1-X representation extraction configs (2025-11-19)
│   │   ├── seed3                # Seed3 PT1-X representation extraction configs (2025-11-20)
│   │   └── seed4                # Seed4 PT1-5 representation extraction configs (2025-11-20)
│   ├── analysis_v2              # Clean CKA v2 infrastructure (2025-11-19)
│   │   ├── cka_cross_seed       # Cross-seed CKA configs: 21×21 matrix (924 configs)
│   │   └── representation_extraction  # Original repr extraction configs (superseded)
│   ├── data_generation
│   │   ├── pftset              # Partial fine-tuning dataset combinations (asymmetric)
│   │   └── ftset               # Full fine-tuning dataset combinations
│   ├── extract_representations  # Configs for extracting model representations
│   ├── eval                     # Checkpoint evaluation configs
│   │   ├── m1_10M_ft1          # Fine-tuning 1 (distance only) evaluation configs
│   │   └── m1_10M_ft2          # Fine-tuning 2 (multi-task) evaluation configs
│   ├── revision                 # Revision experiments (2025-11-19, exp1 expanded 2025-11-20)
│   │   ├── exp1                 # PT1 multi-task with seeds 1,2,3 (base + ftwb1 + ftwb2)
│   │   │   ├── training         # Training configs for ftwb1 (21 single-task models)
│   │   │   ├── eval             # Eval configs: 1305 total (base+ftwb1+ftwb2 × 15 tasks)
│   │   │   │   └── seed{1,2,3}  # Per-seed eval configs (base, ftwb1-1..7, ftwb2-1..21)
│   │   │   ├── representation_extraction  # Repr configs: 91 total (orig+seed ftwb1/ftwb2 × layer5, 2025-11-21)
│   │   │   │   ├── pt1_ftwb1-{1-7}        # Original FTWB1 repr configs
│   │   │   │   ├── pt1_seed{1,2,3}_ftwb1-{1-7}  # Seed FTWB1 repr configs
│   │   │   │   └── pt1_seed{1,2,3}_ftwb2-{1-21}  # Seed FTWB2 repr configs
│   │   │   ├── pca_timeline     # PCA timeline configs: 273 total (mixed/na/raw × models, 2025-11-21)
│   │   │   │   ├── pt1_ftwb1-{1-7}        # Original FTWB1 PCA configs
│   │   │   │   ├── pt1_seed{1,2,3}_ftwb1-{1-7}  # Seed FTWB1 PCA configs
│   │   │   │   └── pt1_seed{1,2,3}_ftwb2-{1-21}  # Seed FTWB2 PCA configs
│   │   │   └── probe_generalization  # Probe generalization configs: 84 total (2025-11-21)
│   │   │       ├── pt1_seed1_ftwb2-{1-21}.yaml  # Seed1 FTWB2 probe gen configs
│   │   │       ├── original/pt1_ftwb2-{1-21}.yaml  # Original seed probe gen configs
│   │   │       ├── seed2/pt1_seed2_ftwb2-{1-21}.yaml  # Seed2 probe gen configs
│   │   │       └── seed3/pt1_seed3_ftwb2-{1-21}.yaml  # Seed3 probe gen configs
│   │   ├── exp2                 # Two/Three-task seed robustness
│   │   │   ├── pt2_seed         # PT2 two-task with seeds 1,2 (all 8 variants)
│   │   │   │   ├── training     # PT2 training configs
│   │   │   │   ├── representation_extraction  # PT2 layer 5 extraction configs
│   │   │   │   ├── extract_representations_multilayer  # PT2 layers 3,4,5,6 configs (2025-11-20)
│   │   │   │   └── pca_timeline # PT2 PCA configs
│   │   │   ├── pt2_seed_cka     # PT2 seed CKA configs: 168 configs (14 pairs × 4 layers × 3 seeds, 2025-11-21)
│   │   │   ├── pt3_seed         # PT3 three-task with seeds 1,2 (all 8 variants, 2025-11-20)
│   │   │   │   ├── training     # PT3 training configs
│   │   │   │   ├── representation_extraction  # PT3 layer 5 extraction configs
│   │   │   │   ├── extract_representations_multilayer  # PT3 layers 3,4,5,6 configs (2025-11-20)
│   │   │   │   └── pca_timeline # PT3 PCA configs
│   │   │   └── pt3_seed_cka     # PT3 seed CKA configs: 84 configs (7 pairs × 4 layers × 3 seeds, 2025-11-21)
│   │   ├── exp3                 # Model width ablation (wide/narrow/wider, 2025-11-20, eval added 2025-11-21)
│   │   │   ├── pt1_wide         # Wide model configs: hidden=256, intermediate=1024, heads=8, epochs=3
│   │   │   │   ├── pt1_wide.yaml  # Base pretraining config
│   │   │   │   ├── pt1_wide_ftwb1.yaml  # FTWB1 single-task fine-tuning (7 tasks)
│   │   │   │   └── pt1_wide_ftwb2-{2,4,9,12,13,15}.yaml  # FTWB2 two-task fine-tuning (6 configs)
│   │   │   ├── pt1_narrow       # Narrow model configs: hidden=64, intermediate=256, heads=2, epochs=12
│   │   │   │   ├── pt1_narrow.yaml  # Base pretraining config
│   │   │   │   └── pt1_narrow_ftwb1.yaml  # FTWB1 single-task fine-tuning (7 tasks)
│   │   │   ├── pt1_wider        # Wider model configs: hidden=512, intermediate=2048, heads=16, epochs=3 (2025-11-21)
│   │   │   │   └── pt1_wider.yaml  # Base pretraining config
│   │   │   ├── eval             # Evaluation configs (420 total, 2025-11-21)
│   │   │   │   ├── wide         # Wide model evals: base + ftwb1-{1-7} + ftwb2-{2,4,9,12,13,15}
│   │   │   │   └── narrow       # Narrow model evals: base + ftwb1-{1-7} + ftwb2-{2,4,9,12,13,15}
│   │   │   ├── representation_extraction  # Layer 5 repr extraction for wide/narrow models
│   │   │   └── pca_timeline     # PCA visualization configs (raw/mixed/na variants)
│   │   ├── additional_plots     # Additional visualization configs (2025-11-21)
│   │   │   ├── cities_basic.yaml  # Basic city scatter plot config
│   │   │   ├── probe_atlantis_ftwb2-{1,2,3,4,5,6,8,13,15}.yaml  # Probe prediction plots
│   │   │   └── probe_3d_nullspace_ftwb2-{2,13}.yaml  # 3D nullspace probe visualization
│   │   ├── exp4                 # Single-task PT1-X seed robustness (2025-11-19, expanded 2025-11-20)
│   │   │   ├── pt1_single_task_seed  # PT1-X with seeds 1,2,3 (all 7 tasks) + pt1-5 seeds 4-7
│   │   │   ├── representation_extraction  # Seed1/2/3/4 repr extraction configs (layers 3-6)
│   │   │   ├── pca_timeline     # PCA timeline configs (original/seed1/2/3, mixed & raw)
│   │   │   │   ├── original     # Original experiments (seed 42), mixed probe alignment
│   │   │   │   ├── original_raw # Original experiments, raw PCA (no probe)
│   │   │   │   ├── seed1/seed1_raw  # Seed1 experiments (mixed & raw)
│   │   │   │   ├── seed2/seed2_raw  # Seed2 experiments (mixed & raw, no pt1-5)
│   │   │   │   └── seed3/seed3_raw  # Seed3 experiments (mixed & raw)
│   │   │   ├── cka_cross_seed   # Cross-seed CKA configs: 21×21 + seed4 (1330+ configs, layers 3-6)
│   │   │   ├── cka_cross_seed_first3  # PCA-based CKA (first 3 PCs): 1351 configs (2025-11-20)
│   │   │   └── procrustes_cross_seed_first3  # Procrustes distance (first 3 PCs): 1351 configs (2025-11-20)
│   │   ├── exp5                 # PT1 trained with Atlantis from scratch (2025-11-21)
│   │   │   ├── pt1_with_atlantis.yaml  # Training config (seed 42)
│   │   │   ├── representation_extraction  # Layer 3,4,5,6 repr extraction configs
│   │   │   ├── pca_timeline     # PCA timeline configs (mixed/na/raw)
│   │   │   └── probe_generalization  # Probe generalization config (atlantis.yaml)
│   │   └── exp6                 # Scattered Atlantis experiment (2025-11-27)
│   │       ├── data_generation  # Data generation configs (50 total)
│   │       │   ├── city_dataset_scattered_atlantis.yaml  # 100 cities uniform random
│   │       │   ├── {task}_1M_with_atlantis.yaml  # 7 task datasets with Atlantis
│   │       │   ├── {task}_1M_no_atlantis.yaml    # 7 task datasets without Atlantis
│   │       │   ├── {task}_100k_atlantis_required.yaml  # 7 task datasets Atlantis required
│   │       │   └── ftset        # Combined dataset configs
│   │       │       ├── combine_multitask_pt1.yaml  # 7M PT1 pretraining
│   │       │       ├── combine_ftwb1-{1-7}.yaml    # 7 single-task FT
│   │       │       └── combine_ftwb2-{1-21}.yaml   # 21 two-task FT
│   │       ├── training         # Training configs
│   │       │   ├── train_pt1.yaml  # PT1 7-task pretraining
│   │       │   ├── ftwb1        # 7 FTWB1 single-task configs
│   │       │   └── ftwb2        # 21 FTWB2 two-task configs
│   │       ├── cities_scattered_atlantis_plot.yaml  # Visualization config
│   │       └── scattered_atlantis_region_mapping.json
│   ├── tokenizers
│   ├── training
│   │   ├── pftset              # Partial fine-tuning training configs (pftX-Y)
│   │   └── ftset               # Full fine-tuning training configs
│   ├── visualization
│   ├── atlantis_region_mapping.json
│   └── tasks.json
├── data
│   ├── datasets                 # Generated datasets
│   ├── experiments             # Training outputs
│   ├── figures
│   ├── geographic_mappings
│   ├── tokenizers
│   ├── vis                      # Exported visualizations (2025-11-21)
│   │   ├── group1_pt1/          # PT1-X 3D PCA visualizations (42 files)
│   │   │   └── pt1-{1-7}_seed{0,1,2}_{mixed,raw}.html
│   │   ├── group2/              # PT1 + FTWB 3D PCA visualizations (348 files)
│   │   │   ├── pt1_seed{0,1,2,3}_{mixed,raw,na}.html
│   │   │   ├── pt1_seed{0,1,2,3}_ftwb1-{1-7}_{mixed,raw,na}.html
│   │   │   └── pt1_seed{0,1,2,3}_ftwb2-{1-21}_{mixed,raw,na}.html
│   │   ├── group1_pt1.zip       # Zipped group1_pt1 (121 MB)
│   │   └── group2.zip           # Zipped group2 (1.0 GB)
│   └── geonames-all-cities-with-a-population-1000.csv
├── docs
│   ├── md                       # Technical documentation
│   │   ├── findings             # Documented discoveries and phenomena
│   │   ├── canonical_experiments.md
│   │   ├── cka_v2_infrastructure.md
│   │   ├── cross_seed_cka_21x21.md
│   │   └── cross_seed_cka_setup.md
│   ├── logs
│   │   ├── 2025-08-30
│   │   ├── 2025-08-31
│   │   ├── 2025-09-01
│   │   ├── 2025-09-02
│   │   ├── 2025-09-03
│   │   ├── 2025-09-09
│   │   ├── 2025-09-10
│   │   ├── 2025-09-12
│   │   ├── 2025-09-13
│   │   ├── 2025-09-14
│   │   ├── 2025-09-15
│   │   ├── 2025-09-16
│   │   ├── 2025-09-17
│   │   ├── 2025-09-18
│   │   ├── 2025-09-20
│   │   ├── 2025-09-21
│   │   ├── 2025-09-22
│   │   ├── 2025-09-23
│   │   ├── 2025-09-24
│   │   ├── 2025-09-25
│   │   ├── 2025-11-19
│   │   ├── 2025-11-20
│   │   ├── 2025-11-21
│   │   ├── 2025-11-22
│   │   ├── 2025-11-23
│   │   └── 2025-11-27
│   ├── reports
│   ├── paper_writing
│   ├── closing_tasks.md
│   ├── repo_usage.md
│   ├── research_context.md
│   ├── research_proposal.md
│   ├── start.md
│   └── structure.md
├── old_commits                  # Previous versions for reference
│   ├── aug31_snapshot          # Old files from August 31
│   └── b9c7d24                 # Commit b9c7d24a0e02cebe1d966d591487ef4ffd857e6e
├── scratch                      # Temporary work directory (gitignored, organized 2025-09-21)
│   ├── analysis                 # Analysis scripts and outputs
│   │   ├── atlantis            # Atlantis-specific analysis
│   │   ├── city_id             # City ID pattern analysis
│   │   ├── clusters            # Cluster detection and analysis
│   │   └── gradients           # Gradient analysis
│   ├── formation_dynamics       # Training dynamics visualization (2025-11-21)
│   │   ├── figures              # Generated vertical plots (pt1-1 through pt1-7)
│   │   ├── plot_pt1_all.py      # Main script: 3-panel vertical plots for all PT1 experiments
│   │   ├── plot_pt1-2_vertical.py  # Standalone PT1-2 debugging script
│   │   ├── plot_pt1-3_vertical.py  # Original PT1-3 script
│   │   └── collect_all_metrics.py  # Metrics aggregation (R², distance error)
│   ├── cka_analysis_clean      # Clean CKA analysis outputs (2025-09-24)
│   │   ├── cka_checkpoints.csv # All checkpoint CKA values (12,628 measurements)
│   │   ├── cka_summary.csv     # Summary statistics (308 pair-layer combinations)
│   │   ├── cka_organized.json  # Hierarchical organization with statistics
│   │   └── timelines_non_overlap # CKA timeline plots for non-overlapping pairs
│   ├── cka_to_generalization   # CKA-generalization correlation analysis (2025-09-25)
│   │   ├── plot_cka_generalization_correlation.py # CKA vs transfer, task reciprocity (FT1/FTWB1)
│   │   ├── plot_generalization_heatmap.py # Single-task generalization heatmaps (FT1/FTWB1)
│   │   ├── plot_multi_task_evaluation.py # Multi-task evaluation plots (FT2/FT3/FTWB2/FTWB3)
│   │   ├── experiment_task_mapping.txt # Complete experiment and task documentation
│   │   ├── ft1_cka_vs_generalization_scatter.png # FT1 CKA vs generalization
│   │   ├── ft1_task_reciprocity_scatter.png # FT1 task helping vs benefiting
│   │   ├── ft1_generalization_heatmap.png # FT1 single-task generalization
│   │   ├── ftwb1_cka_vs_generalization_scatter.png # FTWB1 CKA vs generalization
│   │   ├── ftwb1_task_reciprocity_scatter.png # FTWB1 task helping vs benefiting
│   │   ├── ftwb1_generalization_heatmap.png # FTWB1 single-task generalization
│   │   ├── ft2_evaluation_plot.png # FT2 actual vs predicted performance
│   │   ├── ft3_evaluation_plot.png # FT3 actual vs predicted performance
│   │   ├── ftwb2_evaluation_plot.png # FTWB2 actual vs predicted performance
│   │   └── ftwb3_evaluation_plot.png # FTWB3 actual vs predicted performance
│   ├── dimensionality          # Dimensionality analysis (2025-09-24)
│   │   └── test_manifold_metrics.py # Core metrics: TwoNN, correlation dim, local PCA 2D
│   ├── archive                  # Old analyses for reference
│   ├── dataproperties          # Data property analysis
│   ├── eval_plots              # Evaluation comparison plots
│   ├── external                # External test scripts
│   ├── m1_10M_plots            # Model evaluation plots
│   ├── plots                   # Visualization scripts and outputs
│   │   ├── analysis            # Analysis plots
│   │   ├── evaluation          # FT evaluation plots and scripts (updated 2025-09-25)
│   │   │   ├── plot_ft1_heatmap.py    # Single-task fine-tuning heatmaps
│   │   │   ├── plot_ft2_heatmap.py    # 2-task fine-tuning heatmaps (RdBu colormap)
│   │   │   ├── plot_ft3_heatmap.py    # 3-task fine-tuning heatmaps (RdBu colormap)
│   │   │   ├── plot_ftwb1_heatmap.py  # Single-task warmup+bias heatmaps
│   │   │   ├── plot_ftwb2_heatmap.py  # 2-task warmup+bias heatmaps (RdBu colormap)
│   │   │   └── plot_ftwb3_heatmap.py  # 3-task warmup+bias heatmaps (RdBu colormap, created 2025-09-25)
│   │   └── metrics             # Metric visualization
│   ├── tempplots               # Temporary plots
│   ├── testing                 # Test scripts
│   │   ├── circlecount         # Circle count testing
│   │   ├── gradients           # Gradient testing
│   │   ├── integration         # Integration tests
│   │   └── metrics             # Metric testing
│   └── utils                   # Utility functions
├── scripts
│   ├── analysis
│   ├── analysis_representations  # Scripts for running representation analysis
│   ├── data_generation
│   │   ├── merge               # Dataset merging/combining scripts
│   │   ├── multi               # Multi-task dataset scripts
│   │   └── single_tasks        # Individual task dataset scripts
│   ├── eval                    # Checkpoint evaluation scripts
│   ├── meta                    # Meta-scripts for project management
│   │   └── zip                 # HTML collection and zipping scripts
│   ├── revision                # Revision experiment scripts (2025-11-19, expanded 2025-11-20)
│   │   ├── exp1                # PT1 multi-task with seeds (complete infrastructure 2025-11-20)
│   │   │   ├── training        # FTWB1 training scripts (4 scripts)
│   │   │   ├── eval            # Evaluation scripts (11 scripts: batch + master + ftwb1)
│   │   │   ├── representation_extraction  # Repr extraction (11 scripts)
│   │   │   ├── probe_generalization  # Probe generalization scripts (4 run_all scripts, 2025-11-21)
│   │   │   ├── plots           # Plotting scripts for aggregated results
│   │   │   └── README.md       # Quick reference guide
│   │   ├── exp2                # PT2/PT3 two/three-task seed variations (+ meta scripts)
│   │   │   ├── pt2_seed        # PT2 training scripts (8 variants × 2 seeds)
│   │   │   ├── pt3_seed        # PT3 training scripts (8 variants × 2 seeds, 2025-11-20)
│   │   │   ├── representation_extraction  # PT2/PT3 multi-layer repr extraction (2025-11-20)
│   │   │   │   ├── extract_pt2_*_multilayer.sh  # PT2 layer 3,4,5,6 scripts
│   │   │   │   └── extract_pt3_*_multilayer.sh  # PT3 layer 3,4,5,6 scripts
│   │   │   ├── extract_pt2_pt3_all_multilayer.sh  # Master script for all 105 extractions
│   │   │   ├── cka_analysis     # PT2/PT3 seed CKA scripts (15 scripts, 252 calcs, 2025-11-21)
│   │   │   │   ├── run_pt2_seed_cka_l{3,4,5,6}.sh  # PT2 layer-specific scripts (4 files)
│   │   │   │   ├── run_pt3_seed_cka_l{3,4,5,6}.sh  # PT3 layer-specific scripts (4 files)
│   │   │   │   ├── run_pt2_seed_cka_all_layers.sh  # PT2 all layers master
│   │   │   │   ├── run_pt3_seed_cka_all_layers.sh  # PT3 all layers master
│   │   │   │   ├── run_exp2_seed_cka_all.sh        # Combined PT2+PT3 master
│   │   │   │   ├── run_exp2_seed_cka_chunk{1,2,3,4}.sh  # 4 balanced chunks (63 each)
│   │   │   │   └── run_exp2_seed_cka_4chunks_parallel.sh  # Chunk master
│   │   │   ├── run_pt2-X_all.sh  # PT2 meta scripts (8 files)
│   │   │   └── run_pt3-X_all.sh  # PT3 meta scripts (8 files, 2025-11-20)
│   │   ├── exp3                # Model width ablation (wide/narrow/wider, 2025-11-20, eval added 2025-11-21)
│   │   │   ├── pt1_wide        # Wide model scripts: pt1_wide.sh + ftwb{1-7}.sh + ftwb2-{2,4,9,12,13,15}.sh
│   │   │   ├── pt1_narrow      # Narrow model scripts: pt1_narrow.sh + ftwb{1-7}.sh
│   │   │   ├── pt1_wider       # Wider model scripts: pt1_wider.sh (2025-11-21)
│   │   │   ├── eval            # Evaluation scripts (16 chunked + 2 master, 2025-11-21)
│   │   │   │   ├── eval_wide_* # Wide eval scripts (7 chunks of ~30 evals each)
│   │   │   │   ├── eval_narrow_* # Narrow eval scripts (7 chunks of ~30 evals each)
│   │   │   │   ├── eval_wide_all.sh   # Master script for all wide evals
│   │   │   │   └── eval_narrow_all.sh # Master script for all narrow evals
│   │   │   ├── run_pt1_wide_all.sh   # Meta script: pretrain + 7 ftwb fine-tunings
│   │   │   └── run_pt1_narrow_all.sh # Meta script: pretrain + 7 ftwb fine-tunings
│   │   ├── exp4                # PT1-X single-task seed variations (+ meta scripts, expanded 2025-11-20)
│   │   │   ├── pt1_single_task_seed  # Training scripts (all seeds including pt1-5 seeds 4-7)
│   │   │   ├── representation_extraction  # Seed1/2/3 repr extraction scripts
│   │   │   ├── pca_timeline    # Seed1/2/3 PCA timeline scripts
│   │   │   ├── cka_analysis    # Cross-seed CKA scripts (21×21, 28×28, layers 3-6, 2025-11-21)
│   │   │   │   ├── run_missing_cka_layer3_part{1,2,3}.sh  # Layer 3 execution (231 configs, 2025-11-21)
│   │   │   │   └── run_missing_cka_layer6.sh  # Layer 6 pt1-5_seed3 (21 configs, 2025-11-21)
│   │   │   ├── cka_analysis_first3  # PCA-based CKA scripts (14×14, layers 4-5, 2025-11-20)
│   │   │   ├── procrustes_analysis_first3  # Procrustes distance scripts (14×14, layer 5, 2025-11-20)
│   │   │   └── run_pt1-X_all.sh  # Meta scripts (7 files)
│   │   ├── exp5                # PT1 trained with Atlantis from scratch (2025-11-21)
│   │   │   ├── train_pt1_with_atlantis.sh  # Training script
│   │   │   ├── generate_datasets.sh  # Dataset generation (7 tasks with Atlantis)
│   │   │   ├── representation_extraction  # Layer 3,4,5,6 extraction scripts
│   │   │   ├── pca_timeline     # PCA timeline scripts (pca_all.sh)
│   │   │   ├── probe_generalization  # Probe generalization script (run_atlantis.sh)
│   │   │   └── plots            # Plotting scripts
│   │   └── exp6                # Scattered Atlantis experiment (2025-11-27)
│   │       ├── data_generation  # Data generation scripts
│   │       │   ├── step1_gen_cities.sh  # Generate scattered Atlantis cities
│   │       │   ├── step2-{1,2,3}_gen_tasks_*.sh  # Task dataset generation (parallel)
│   │       │   └── step3_combine_all.sh  # Combine datasets
│   │       ├── training         # Training scripts
│   │       │   ├── train_pt1.sh  # PT1 7-task pretraining
│   │       │   ├── train_all_ftwb1.sh  # All 7 FTWB1 trainings
│   │       │   ├── train_ftwb2_part{1,2,3}.sh  # FTWB2 in 3 batches of 7
│   │       │   ├── ftwb1/train_ftwb1-{1-7}.sh  # Individual FTWB1 scripts
│   │       │   └── ftwb2/train_ftwb2-{1-21}.sh  # Individual FTWB2 scripts
│   │       └── plot_cities_scattered_atlantis.sh  # Visualization script
│   ├── tokenizers
│   ├── training
│   ├── utils
│   └── visualization
├── src
│   ├── analysis
│   │   ├── cka_v2               # CKA v2 infrastructure (2025-11-19)
│   │   │   ├── visualization    # CKA visualization scripts (2025-11-21)
│   │   │   │   ├── plot_21x21_cka_matrix.py     # Layer 5 21×21 and 7×7 matrices
│   │   │   │   └── plot_21x21_cka_matrix_l4.py  # Layer 4 21×21 and 7×7 matrices
│   │   │   └── compute_cka.py   # Core CKA computation module
│   │   ├── past                # Archived analysis scripts
│   │   ├── analyze_activation_gradients.py  # Gradient flow analysis from loss to activations
│   │   ├── analyze_representations.py  # Full-featured analysis with all visualizations
│   │   ├── compute_cka_from_representations.py  # CKA computation between model representations
│   │   ├── compute_kernel_matrices.py  # GPU-accelerated kernel matrix computation
│   │   ├── analyze_representations_higher.py  # Minimal version for dynamics plot only
│   │   ├── compute_cka_matrix_direct.py    # Direct CKA matrix computation from kernel matrices
│   │   ├── recreate_pca_plot_from_data.py  # Recreate PCA visualizations from saved data
│   │   ├── visualize_pca_3d.py
│   │   ├── visualize_pca_3d_timeline.py    # PCA timeline animation
│   │   ├── visualize_prediction_error_3d.py
│   │   └── visualize_xy_residual_pca_3d.py  # X/Y directions + residual PCA visualization
│   ├── data_processing
│   │   ├── combine_datasets.py
│   │   ├── create_city_dataset.py
│   │   ├── create_distance_dataset.py
│   │   ├── create_randomwalk_dataset.py    # Random walk sequences
│   │   ├── create_trianglearea_dataset.py  # Triangle area calculation
│   │   ├── create_angle_dataset.py         # Angle at center city
│   │   ├── create_nearest_neighbor_dataset.py  # K nearest neighbors (deterministic for must_include)
│   │   ├── create_crossing_dataset.py      # Line segment intersection (supports must_include)
│   │   ├── create_inside_dataset.py        # Point in convex hull (supports must_include)
│   │   ├── create_center_dataset.py        # Center of mass finder (partial must_include support)
│   │   ├── create_circlecount_dataset.py   # Count cities in radius
│   │   ├── create_compass_dataset.py       # Compass direction (uses data_utils)
│   │   ├── create_perimeter_dataset.py     # Polygon perimeter (supports must_include)
│   │   ├── create_randring_dataset.py      # Random ring sampling (stochastic)
│   │   └── data_utils.py                   # Shared pair generation strategies
│   ├── eval
│   │   └── evaluate_checkpoints.py         # Checkpoint evaluation script
│   ├── scripts                   # Orchestration scripts (entry points)
│   │   ├── generate_revision_exp1_eval_configs.py      # Exp1 eval configs (990, 2025-11-20)
│   │   ├── generate_revision_exp1_eval_scripts.py      # Exp1 eval batch scripts (2025-11-20)
│   │   ├── generate_revision_exp1_repr_configs.py      # Exp1 repr configs (66, 2025-11-20)
│   │   ├── generate_revision_exp1_repr_scripts.py      # Exp1 repr batch scripts (2025-11-20)
│   │   ├── generate_revision_exp1_ftwb1_configs.py     # Exp1 FTWB1 training (21, 2025-11-20)
│   │   ├── generate_revision_exp1_ftwb1_train_scripts.py  # Exp1 FTWB1 train scripts (2025-11-20)
│   │   ├── generate_revision_exp1_ftwb1_eval_configs.py   # Exp1 FTWB1 eval (315, 2025-11-20)
│   │   ├── generate_revision_exp1_ftwb1_repr_configs.py   # Exp1 FTWB1 repr (21, 2025-11-20)
│   │   ├── plot_revision_exp1_ftwb2_heatmaps.py        # Exp1 visualization (2025-11-20)
│   │   ├── generate_pt2_pt3_multilayer_repr_configs.py  # PT2/PT3 multi-layer repr configs (2025-11-20)
│   │   ├── generate_pt2_pt3_multilayer_run_scripts.py   # PT2/PT3 multi-layer run scripts (2025-11-20)
│   │   ├── generate_exp2_seed_cka_configs.py        # Exp2 PT2/PT3 seed CKA configs (252, 2025-11-21)
│   │   ├── generate_exp2_seed_cka_run_scripts.py    # Exp2 seed CKA scripts (11, 2025-11-21)
│   │   ├── generate_exp2_seed_cka_chunked_scripts.py  # Exp2 seed CKA 4-chunk scripts (5, 2025-11-21)
│   │   ├── generate_revision_exp3_eval_configs.py   # Exp3 eval configs (420, 2025-11-21)
│   │   ├── generate_revision_exp3_eval_scripts.py   # Exp3 eval scripts (16 chunked, 2025-11-21)
│   │   ├── plot_revision_exp3_ftwb1_heatmaps.py     # Exp3 FTWB1 7×7 matrices (2025-11-21)
│   │   ├── plot_revision_exp3_ftwb2_heatmaps.py     # Exp3 FTWB2 7×6 matrices (2025-11-21)
│   │   ├── plot_revision_exp3_ftwb2_vs_ftwb1.py     # Exp3 FTWB2-FTWB1 diff plots (2025-11-21)
│   │   ├── generate_missing_cka_configs_21x21.py    # Layer 3 CKA config generator (231, 2025-11-21)
│   │   ├── generate_missing_l6_configs.py           # Layer 6 pt1-5_seed3 configs (21, 2025-11-21)
│   │   ├── evaluate_probe_generalization.py         # Linear probe OOD generalization (2025-11-21)
│   │   ├── plot_probe_generalization_histogram.py   # Histogram of probe errors (2025-11-21)
│   │   ├── plot_probe_generalization_histogram_with_exp5.py  # Histogram + exp5 reference line (2025-11-21)
│   │   ├── generate_raw_metrics_latex.py            # Raw metrics CSV for appendix (2025-11-21)
│   │   ├── zip_pt1_ft2_htmls.py       # Collect PCA timeline HTMLs for pt1_ft2-X experiments
│   │   └── zip_pt1_ftwb2_htmls.py     # Collect PCA timeline HTMLs for pt1_ftwb2-X experiments
│   ├── training
│   │   └── train.py
│   ├── visualization
│   │   └── visualize_cities.py
│   ├── create_tokenizer.py
│   ├── dimensionality.py        # 2D manifold testing (TwoNN, correlation dim, local PCA)
│   ├── evaluation.py            # Unified evaluation module (single source of truth)
│   ├── metrics.py               # Centralized task metric calculations
│   ├── representation_extractor.py
│   ├── utils.py
│   └── utils_backup.py          # Backup of utils.py
├── rebuttal                    # ICLR 2026 rebuttal materials (2025-11-22)
│   ├── INITIAL_PAPER.txt       # Submitted paper text for reference
│   ├── PAST_EXAMPLE.txt        # Successful rebuttal example for style reference
│   └── REBUTTAL.txt            # Draft rebuttal responses to all 4 reviewers
├── anything.ipynb
├── CLAUDE.md
├── pyproject.toml
├── README.md
└── uv.lock
```

## Key Changes

### 2026-01-27
- ICML 2026 submission work in `paper_icml/` (separate git repo synced to Overleaf)
- Updated title: "Convergent World Representations and Divergent Tasks"
- Rewrote abstract to emphasize convergence/divergence findings

### 2026-01-23
- Major reorganization from flat structure to track-based organization
  - Created `data_generation_v1` track (COMPLETE)
  - Moved: `src/data_processing/`, `src/tasks/` → `src/data_generation_v1/`
  - Moved: `configs/data_generation/`, `configs/tokenizers/` → `configs/data_generation_v1/`
  - Moved: `scripts/data_generation/`, `scripts/tokenizers/` → `scripts/data_generation_v1/`
  - Created track docs: `docs/tracks/data_generation_v1/notes.md`
  - Updated all config paths (171 files): `data/datasets/` → `data/data_generation_v1/`
  - Organized configs into subfolders: cities/, tokenizers/, single_tasks/, derived/
  - Removed non-core tasks: randomwalk, randring, center, circlecount, nearest_neighbor
  - Removed pft (partial fine-tuning) configs
  - Replaced hardcoded cluster paths (8,804 files)
  - Tested: City generation (5,175 cities), tokenizer generation (98 tokens)
  - See `docs/REORGANIZATION.md` for full plan and naming conventions

### 2025-11-27
- Created Exp6: Scattered Atlantis experiment infrastructure
  - Tests whether observed effects are due to Atlantis clustering vs uniform distribution
  - Modified src/data_processing/create_city_dataset.py to support uniform random distribution
  - Created 50 data generation configs (city dataset + 21 task datasets + 28 combined datasets)
  - Created 29 data generation scripts (step1/2/3 for sequential execution)
  - Created PT1 training config and script
  - Created 28 FTWB training configs (7 FTWB1 + 21 FTWB2)
  - Created batch training scripts (train_all_ftwb1.sh, train_ftwb2_part{1,2,3}.sh)
  - Generator scripts: generate_exp6_data_configs.py, generate_exp6_data_scripts.py, generate_exp6_ftwb_training_configs.py

### 2025-09-25 (05:43 Session)
- Created HTML collection scripts for PCA visualization review
  - scripts/meta/zip/ directory for organizational scripts
  - src/scripts/zip_pt1_ft2_htmls.py - Collects pt1_ft2-X PCA timeline HTMLs
  - src/scripts/zip_pt1_ftwb2_htmls.py - Collects pt1_ftwb2-X PCA timeline HTMLs
  - Bash wrappers: all_pt1_ft2-X_htmls.sh and all_pt1_ftwb2-X_htmls.sh
  - Implements fail-fast validation (checks all 63 files exist before proceeding)
  - Outputs organized zip files to ~/WM_1/ with clear naming conventions

### 2025-09-25 (Earlier)
- Fixed evaluation heatmap colormap issue (changed from coolwarm to RdBu)
  - Negative values (underperformance) now correctly shown as red
  - Positive values (exceeding expectations) now correctly shown as blue
  - Fixed in plot_ft2_heatmap.py, plot_ft3_heatmap.py, plot_ftwb2_heatmap.py
- Created plot_ftwb3_heatmap.py for 3-task fine-tuning with warmup+bias
  - Shows FTWB3 outperforms FT3 by +12% average (+20% on untrained tasks)
- Created CKA-to-generalization correlation analysis in scratch/cka_to_generalization/
  - Scatter plot showing CKA scores vs FTWB1 transfer performance
  - Excluded crossing task due to training instabilities
  - Found moderate positive correlation (r=0.354, p=0.055)
- Added task reciprocity visualization revealing asymmetric transfer relationships
  - "Net givers" (help others more): perimeter, angle, trianglearea
  - "Net takers" (benefit more): inside, compass, distance

### 2025-09-17 (12:11)
- Fixed critical evaluation bug causing ~130 error discrepancy
  - Removed double-spacing bug in evaluate_checkpoints.py
  - Fixed BOS token handling (was adding duplicate BOS tokens)

### 2025-09-17 (14:37)
- Unified all evaluation logic into src/evaluation.py (single source of truth)
  - Fixed task-specific prompt/completion splitting (rfind vs find for '=')
  - Fixed post-training eval to decode full outputs like training eval
  - Added return_details flag for memory-efficient training vs detailed post-training
- Reorganized evaluation output structure:
  - eval_config.yaml at root
  - dynamics/ for plots (renamed from summary/)
  - eval_data/ for results and detailed data
- Fixed evaluation for tasks with parameter '=' signs (center, circlecount, randring)
  - Post-training eval now matches training eval (~5 vs ~130 → ~5 vs ~6)
- Created visualize_xy_residual_pca_3d.py
  - Novel 3D visualization showing X/Y predictive directions + residual structure
  - Uses proper coordinate centering matching analyze_representations.py
  - Shows only test cities for cleaner visualization
- Fixed add_special_tokens bugs in multiple analysis scripts

### 2025-09-16 (23:43)
- Created src/metrics.py - centralized metric calculation system
  - Eliminated ~270 lines of scattered duplicate metric implementations
  - Each task type's logic now in single TaskMetric class
  - Single source of truth for parsing, calculation, and failure values
- Refactored utils.py to use centralized metrics
  - Replaced massive if-elif chains with clean API calls
  - Maintained exact backward compatibility

### 2025-09-16 (Earlier)
- Added 7 new geometry dataset generation tasks:
  - Nearest neighbor (k-nearest cities)
  - Line crossing detection (segment intersection)
  - Inside convex hull (point-in-polygon test)
  - Center of mass (closest city to centroid)
  - Circle count (cities within radius)
  - Compass direction (8-way directional output)
  - Perimeter (polygon perimeter calculation)
- Standardized dataset sizes based on unique data availability
- Created meta script create_all_geometry_datasets_pad.sh for all 11 tasks
- Fixed center_dataset.py tie-breaking (deterministically selects first)

### 2025-09-15
- Fixed critical evaluation bug for angle/trianglearea tasks (was passing answer in prompt)
- Created analyze_representations_higher.py - minimal version for dynamics plots only
- Added log-scale x-axis to loss plots for better early training visualization
- Removed all DEBUG print statements from analysis scripts
- Added dist_city_last_and_comma prompt format for representation analysis

### 2025-09-14
- Added 3 new dataset types: random walk, triangle area, angle
- Overhauled evaluation system with consistent error metrics
- Created multi-task training infrastructure for 4M combined dataset
- Fixed coordinate scaling issues (10x factor in distance calculations)
- Updated plotting with proper log scales and boundaries

### 2025-09-12
- old_commits/ now properly organized by commit hash
- Loss masking feature added to create_distance_dataset.py
- Fixed representation analysis bug in analyze_representations.py
- Added single checkpoint analysis support
