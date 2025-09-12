# Development Log - 2025-09-10 15:04
## Topic: Tokenizer Refactoring and Repository Standardization

### Summary
Major refactoring session focused on standardizing the WM_1 repository to follow strict conventions, implementing a new ASCII-based tokenizer system, and ensuring all scripts comply with repository standards.

### Key Accomplishments

#### 1. Repository Standardization
- **Fixed config field naming**: Changed all `exp_dir` to `output_dir` across configs and code
- **Fixed absolute paths**: Converted all absolute paths to relative paths in configs
- **Standardized bash scripts**: Ensured all scripts are minimal wrappers with single commands

#### 2. New Tokenizer System
- **Created HuggingFace-compatible tokenizer**:
  - Implemented `src/create_tokenizer.py` following standard patterns
  - Created `configs/tokenizers/default_tokenizer.yaml` with ASCII characters (94 chars + 4 special tokens = 98 total)
  - Added `scripts/tokenizers/create_tokenizer.sh` 
  - Tokenizer uses space as delimiter (not a token), supports all printable ASCII except space

- **Benefits over old system**:
  - Proper HuggingFace integration
  - Full ASCII support for future flexibility
  - Clean YAML-driven configuration
  - Professional software engineering approach borrowed from RO_1

#### 3. Data Processing Updates
- **Updated `create_distance_dataset.py`**:
  - Now loads HF tokenizer from config path
  - Formats text as space-delimited: `<bos> d i s t ( c _ 1 2 3 4 , c _ 5 6 7 8 ) = 9 0 <eos>`
  - Properly measures token lengths with new tokenizer
  - Added `tokenizer_path` to all distance dataset configs

- **Fixed script patterns**:
  - `create_city_dataset.py`: Now uses `output_dir` and saves config as `config.yaml`
  - All scripts follow pattern: `config_path` as only positional arg, with `--overwrite` and `--debug` flags

#### 4. Training Configuration Updates
- **Updated all 4 training configs**:
  - Changed tokenizer path to `data/tokenizers/default_tokenizer`
  - Updated vocab_size from 44 to 98
  - Added logging configuration section
  
- **Created individual training scripts**:
  - Removed generic `train_base.sh`
  - Created 4 specific scripts for each training config
  - Fixed path to use `src/training/train.py`

- **Updated `train.py`**:
  - Now reads `output_dir` instead of `exp_dir`
  - Supports `logging.logging_steps` and `logging.report_to` from configs
  - Properly uses `init_directory` from utils

### Files Modified/Created

#### Created:
- `/src/create_tokenizer.py`
- `/configs/tokenizers/default_tokenizer.yaml` (renamed from wm1_ascii_tokenizer.yaml)
- `/scripts/tokenizers/create_tokenizer.sh`
- `/scripts/training/train_dist_1M_no_atlantis_15epochs.sh`
- `/scripts/training/train_dist_1M_with_atlantis_15epochs.sh`
- `/scripts/training/ft_atlantis_100k.sh`
- `/scripts/training/ft_atlantis_120k_mixed.sh`
- `/scratch/test_tokenizer_manual.py` (for testing)

#### Modified:
- `/src/data_processing/create_distance_dataset.py` - Refactored to use HF tokenizer
- `/src/data_processing/create_city_dataset.py` - Fixed to use `output_dir`
- `/src/training/train.py` - Updated to use `output_dir` and logging config
- `/src/utils.py` - Changed validation from `exp_dir` to `output_dir`
- All configs in `/configs/data/` - Added `tokenizer_path`
- All configs in `/configs/training/` - Updated paths and vocab size
- `/scripts/data_generation/create_distance_datasets.sh` - Fixed to use proper pattern
- `/scripts/data_generation/merge_ft_datasets.sh` - Fixed config path

#### Removed:
- `/scripts/training/train_base.sh`
- `analyze_pairs` functionality from create_distance_dataset.py

### Technical Details

#### Tokenizer Comparison:
- **Old**: 44 tokens (lowercase only, limited punctuation)
- **New**: 98 tokens (all ASCII printable except space)
- **Format change**: Now uses space-delimited format for HF compatibility
- **Same sequence length** for existing data (both are character-level)

#### Standards Enforced:
1. All configs must have `output_dir` (not `exp_dir`, `output`, etc.)
2. All paths must be relative from project root
3. Bash scripts must be minimal (just `uv run python script.py config.yaml --overwrite`)
4. Config is copied to output as `config.yaml` (not variations)
5. All Python scripts take config as only positional arg

### Next Steps
- Regenerate all datasets with new tokenizer
- Retrain models with updated configs
- Consider implementing src/scripts/ directory for orchestration scripts

### Notes
- Breaking change: Models trained with old tokenizer are incompatible with new one
- All datasets need to be regenerated with new tokenizer
- Training from scratch required due to vocab size change (44 → 98)