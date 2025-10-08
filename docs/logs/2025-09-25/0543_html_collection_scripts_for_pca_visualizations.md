# HTML Collection Scripts for PCA Visualizations
*September 25, 2025 - 05:43*

## Summary
Created scripts to collect and zip PCA timeline HTML files from analysis results for systematic visualization review.

## Work Completed

### 1. Project Guidelines Review
- Read `CLAUDE.md` and `docs/repo_usage.md` to understand project structure and conventions
- Key principles: fail-fast error handling, explicit behavior, proper directory organization
- Important constraint: Never run git operations unless explicitly requested

### 2. HTML Collection Script Creation
Created systematic approach for collecting PCA timeline visualization files:

**Files Created:**
- `src/scripts/zip_pt1_ft2_htmls.py` - Python script for pt1_ft2-X experiments
- `scripts/meta/zip/all_pt1_ft2-X_htmls.sh` - Bash wrapper for ft2 collection
- `src/scripts/zip_pt1_ftwb2_htmls.py` - Python script for pt1_ftwb2-X experiments
- `scripts/meta/zip/all_pt1_ftwb2-X_htmls.sh` - Bash wrapper for ftwb2 collection

**Script Features:**
- Validates all required files exist before proceeding (fail-fast approach)
- Collects from `data/experiments/` directory structure
- Covers all X values 1-21 and Y suffixes ["", "_raw", "_na"]
- Creates organized temp folder with clear naming convention
- Outputs zip to `~/WM_1/` with descriptive filenames

### 3. Directory Structure and Validation
- Created `scripts/meta/zip/` directory for organizational scripts
- Path pattern: `data/experiments/pt1_ft[w]2-X/analysis_higher/distance_firstcity_last_and_trans_l5/pca_timeline[Y]/pca_3d_timeline.html`
- Scripts correctly identify missing files (expected behavior - analysis not run yet)

### 4. File Organization
**New Directory Structure:**
```
scripts/meta/zip/
├── all_pt1_ft2-X_htmls.sh
└── all_pt1_ftwb2-X_htmls.sh

src/scripts/
├── zip_pt1_ft2_htmls.py
└── zip_pt1_ftwb2_htmls.py
```

## Technical Implementation

### Error Handling
- Implements fail-fast validation: checks all 63 files exist before proceeding
- Reports missing files with clear error messages
- Exits immediately if any files missing (research integrity principle)

### File Naming Convention
- Input: `pt1_ft[w]2-X/analysis_higher/distance_firstcity_last_and_trans_l5/pca_timeline[Y]/pca_3d_timeline.html`
- Output: `pt1_ft[w]2-X_pca_timeline_[default|raw|na].html` in zip
- Zip files: `all_pt1_ft2_pca_htmls.zip` and `all_pt1_ftwb2_pca_htmls.zip`

### Following Project Conventions
- Python orchestration scripts in `src/scripts/`
- Minimal bash wrappers using `uv run` pattern
- No unnecessary abstraction - straightforward implementation
- Clear, sequential flow in main() functions

## Current Status
- Scripts created and tested (correctly report missing files)
- Ready to execute once PCA analysis generates the required HTML files
- Both ft2 and ftwb2 variants available for comprehensive collection

## Next Steps
- Scripts will be ready to use once PCA timeline analysis is completed
- Will generate organized zip archives for systematic review of visualization results