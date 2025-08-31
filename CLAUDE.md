# WM_1 Project

## Project Description
This project implements location prediction models using transformer architectures. It trains language models to predict geographic distances and locations between cities using their coordinates encoded as special tokens. The dataset consists of world cities with various population thresholds, and models are trained for tasks like distance prediction, threshold classification, and location generation.

## Python Environment
Use the UV virtual environment located at `.venv/`:
```bash
source .venv/bin/activate  # or use `uv run` for commands
```

## Directory Structure (Rough Overview)
```
WM_1/
├── configs/           # Training configurations (YAML)
├── src/              # Source code
│   ├── data_processing/    # Dataset creation scripts
│   ├── training/           # Model training scripts
│   ├── tokenizer/          # Custom tokenizer setup
│   └── visualization/      # Plotting utilities
├── notebooks/        # Jupyter notebooks for exploration
├── outputs/          # Generated datasets and models
│   ├── datasets/          # Processed city datasets & HF datasets
│   └── tokenizer/         # Custom tokenizer files
├── analysis/         # Analysis scripts
└── claude_notes/     # Documentation including structure.txt (full directory structure)
```

For complete directory structure, see `claude_notes/structure.txt`

## Important Paths
- **Root Directory**: `/n/holylfs06/LABS/finkbeiner_lab/Users/cfpark00/datadir/WM_1`  
  (also symlinked as `/n/home12/cfpark00/WM_1`)
- All scripts should be run from the root directory
- Exception: Notebooks can be run from their own directory

## Development Philosophy
This project follows the **ResearchPy Philosophy** - a research-first development model where:
- Implementation (HOW) lives in `src/utils.py`
- Orchestration (WHAT/WHEN) lives in scripts
- Fail fast with no fallbacks - explicit is better than implicit

See `claude_notes/tips/researchpy_philosophy.md` for detailed examples and principles.

## Running Scripts
Always execute scripts from the project root:
```bash
# From root directory:
python src/training/train.py configs/loc_100k_200epochs.yaml
python src/training/train.py configs/dist_100k_1M_10epochs.yaml --overwrite
```