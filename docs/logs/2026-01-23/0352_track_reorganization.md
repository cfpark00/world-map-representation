# Session Log: Track-Based Reorganization

**Date:** 2026-01-23
**Duration:** ~2 hours

## Summary
Reorganized codebase from flat structure to track-based organization following new `repo_usage.md` template. Completed `data_generation_v1` track, established naming conventions, and documented the full reorganization plan.

## Tasks Completed

### 1. Updated Core Documentation
- Replaced `docs/start.txt` → `docs/start.md` (from research-template)
- Replaced `docs/closing_tasks.md` (from research-template)
- Replaced `docs/repo_usage.md` (from research-template with track system)
- Renamed `docs/structure.txt` → `docs/structure.md` (proper markdown)
- Moved legacy docs to `docs/docs/` subfolder

### 2. Created data_generation_v1 Track
- Moved code: `src/data_processing/`, `src/tasks/` → `src/data_generation_v1/`
- Moved configs: `configs/data_generation/`, `configs/tokenizers/` → `configs/data_generation_v1/`
- Moved scripts: `scripts/data_generation/`, `scripts/tokenizers/` → `scripts/data_generation_v1/`
- Created track docs: `docs/tracks/data_generation_v1/notes.md`

### 3. Updated All Config Paths (171→113 files after cleanup)
- `data/datasets/` → `data/data_generation_v1/`
- Organized output structure:
  - `cities/` - city dataset
  - `tokenizers/` - tokenizer outputs
  - `single_datasets/` - individual task datasets
  - `derived_datasets/` - combined datasets

### 4. Organized Config Subfolders
```
configs/data_generation_v1/
├── cities/              1 yaml
├── tokenizers/          1 yaml
├── single_tasks/       21 yamls (7 tasks × 3 variants)
└── derived/
    ├── pretraining/    18 yamls
    └── finetuning/     70 yamls
```

### 5. Cleanup
- Removed non-core tasks: randomwalk, randring, center, circlecount, nearest_neighbor
- Removed pft (partial fine-tuning) configs entirely
- Removed pacificus and sanity configs
- Removed old m1/m2 configs (referenced deleted tasks)
- Removed old-style naming duplicates (combine_distance_ft1, combine_multitask_ft2-5)

### 6. Fixed Code
- Updated imports: `src.data_processing.data_utils` → `src.data_generation_v1.utils`
- Added `--debug` flags to `create_city_dataset.py` and `combine_datasets.py`
- Fixed hardcoded paths to use relative paths

### 7. Replaced Cluster Paths (8,804 files)
- `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1` → ``
- `/n/home12/cfpark00/WM_1` → ``
- `/n/home12/cfpark00/datadir/WM_1` → ``

### 8. Tested
- ✅ City generation: 5,175 cities to `data/data_generation_v1/cities/`
- ✅ Tokenizer generation: 98 tokens to `data/data_generation_v1/tokenizers/`

### 9. Documented Naming Convention Plan
Created comprehensive rename plan in `docs/REORGANIZATION.md`:
- pt7 (all 7 tasks), pt2-{1..8}, pt3-{1..8}
- ft1-{1..7}, ft2-{1..21}, ft3-{1..7}
- ftwb1-{1..7}, ftwb2-{1..21}, ftwb3-{1..7}
- 88 total files to rename (drop `combine_` prefix)

## Files Created/Modified

**Created:**
- `docs/REORGANIZATION.md` - full reorganization plan
- `docs/tracks/data_generation_v1/notes.md` - track documentation
- `docs/logs/2026-01-23/0352_track_reorganization.md` - this log
- `scratch/path_migration/update_data_gen_paths.py` - path update script

**Modified:**
- `docs/start.md`, `docs/closing_tasks.md`, `docs/repo_usage.md`
- 113 config files in `configs/data_generation_v1/`
- Multiple Python files for import fixes

## Key Decisions
1. **7 core tasks only**: distance, trianglearea, angle, compass, inside, perimeter, crossing
2. **Track structure**: src/, configs/, scripts/, data/, docs/tracks/ per track
3. **Naming convention**: pt{N}-{X} for pretraining, ft{N}-{X}/ftwb{N}-{X} for fine-tuning
4. **Partial coverage OK**: pt2 has 8/21, pt3 has 8/35 due to compute limits

## Next Steps
1. Execute the 88-file rename (drop `combine_` prefix, update output_dir)
2. Create `pretraining_v1` track
3. Create `finetuning_v1` track
4. Create `cka_v1` track
5. Test single task dataset generation
6. Test combined dataset generation
