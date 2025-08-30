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

## Running Scripts
Always execute scripts from the project root:
```bash
# From root directory:
python src/training/train_location.py configs/location_training.yaml
python src/data_processing/create_distance_dataset_hf.py 10000 1000 1000
```