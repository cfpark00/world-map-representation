# Log: 2025-09-03 03:22 - Unified Distance Dataset Generation Refactoring

## Overview
Major refactoring of distance dataset generation scripts to create a unified, configurable system with proper special token handling and symmetric pair generation.

## Key Accomplishments

### 1. Created Unified Distance Dataset Script
- **File**: `src/data_processing/create_distance_dataset.py`
- Replaced three separate scripts with one unified script
- Takes single CSV input file with YAML-based group definitions
- Supports multiple pair generation strategies:
  - `all_pairs`: Random pairs from all cities
  - `within_groups`: Pairs only within specified groups
  - `between_groups`: Pairs between specific group pairs
  - `mixed`: Mix different pair types with ratios
  - `must_include`: All pairs must include at least one city from specified groups

### 2. Group Definition System
- Groups defined through flexible filtering conditions in YAML:
  - `city_names`: Regex patterns (e.g., `"Atlantis_.*"` or `"^(?!Atlantis_).*"`)
  - `country_codes`: List or regex patterns
  - `regions`: List or regex patterns
  - `bounds`: Geographic coordinate boundaries
  - `city_ids`: Specific city IDs
- Removed complex `include`/`exclude` dictionary syntax in favor of regex patterns

### 3. Created Configuration Files
Created 4 specific YAML configs for distance generation:
- `dist_1M_no_atlantis.yaml`: 1M pairs, NO Atlantis cities
- `dist_1M_with_atlantis.yaml`: 1M pairs, including Atlantis (random)
- `dist_100k_atlantis_required.yaml`: 100k pairs, at least one must be Atlantis
- `dist_20k_no_atlantis.yaml`: 20k pairs, NO Atlantis (for later mixing)

All configs use consistent `n_val: 128` for validation sets.

### 4. Fixed Special Token Handling
- **Critical Fix**: Ensured ALL dataset fields include special tokens:
  - `text`: `<bos>dist(c_X,c_Y)=Z<eos>`
  - `prompt`: `<bos>dist(c_X,c_Y)=`
  - `completion`: `Z<eos>`
- Updated ALL data processing scripts:
  - `create_distance_dataset.py` ✓
  - `create_location_dataset.py` ✓
  - `create_randomwalk_dataset.py` ✓
- Fixed analysis script to use `add_special_tokens=False`
- Verified training scripts NEVER add special tokens (`add_special_tokens=False` everywhere)

### 5. Implemented Symmetric Pair Generation
- Added global random swapping after pair generation
- Ensures P(selecting (i,j)) = P(selecting (j,i)) for all city pairs
- Removed redundant swapping logic from individual strategies
- All pair generation strategies are now fully symmetric

### 6. Enhanced Output Structure
- Script now uses `init_directory()` with environment variable safety checks
- Copies YAML config to output directory as `config.yaml`
- Saves comprehensive `metadata.json` with dataset information
- Added `--overwrite` flag for controlled directory overwriting

### 7. Created Batch Processing Script
- **File**: `scripts/create_distance_datasets.sh`
- Runs all 4 distance dataset configurations in sequence
- Sets safety prefix environment variable
- Follows minimal style of existing scripts

## Files Modified/Created

### New Files:
- `src/data_processing/create_distance_dataset.py` (unified script)
- `configs/dist_1M_no_atlantis.yaml`
- `configs/dist_1M_with_atlantis.yaml`
- `configs/dist_100k_atlantis_required.yaml`
- `configs/dist_20k_no_atlantis.yaml`
- `scripts/create_distance_datasets.sh`

### Modified Files:
- `src/data_processing/create_location_dataset.py` (added special tokens to text field)
- `src/data_processing/create_randomwalk_dataset.py` (added special tokens to text field)
- `src/analysis/analyze_representations.py` (added `add_special_tokens=False`)

### Deleted Files:
- `configs/distance_generation_examples.yaml`
- `configs/distance_all_pairs.yaml`
- `configs/distance_atlantis_cross.yaml`

## Technical Notes

### Special Token Philosophy:
- Datasets fully control special tokens in all fields
- Tokenizers NEVER add special tokens (`add_special_tokens=False`)
- Prevents double `<bos>` tokens and missing token issues
- Consistent across training, evaluation, and analysis scripts

### Dataset Splitting:
- Currently uses simple sequential slicing (first N for train, next M for val, rest for test)
- Applied AFTER random pair generation and symmetric swapping
- No stratification or city-level splitting (potential future improvement)

### Regex Patterns for Groups:
- Simplified from dictionary-based include/exclude to pure regex
- Negative lookahead for exclusions: `"^(?!Atlantis_).*"`
- Exact matches: `"^Africa$"`
- List conversion: `[US, CA]` → `"^(US|CA)$"`

## Verified Working:
- Confirmed training config `train_dist_1M_no_atlantis_20epochs.yaml` is ready to use
- Dataset exists at expected path with proper HuggingFace format
- Tokenizer exists and is properly configured
- All paths and configurations validated

## Next Steps:
- Training can proceed with the prepared datasets
- Consider implementing stratified splits or city-level splitting for better evaluation
- Potential to add dataset mixing functionality using `mix_datasets.py`