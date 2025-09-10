# WM_1: World Model Experiments

Learning geographic representations through synthetic location prediction tasks.

## Overview

This project investigates how neural networks form representations of geographic space by training small transformer models on synthetic location prediction tasks. Using controllable datasets of world cities, we study when representations form in modular vs. fractured ways, providing insights into fundamental questions about representation formation in neural networks.

## Research Focus

The project addresses a fundamental question in AI interpretability: **What conditions determine how representations form during training?** By using geographic data as a controlled testbed, we can systematically study:

- When representations organize themselves in modular, interpretable ways
- When they become fractured and entangled
- How scaling dynamics affect representation formation
- The impact of different data distributions on learned structures

See [docs/research_proposal.md](docs/research_proposal.md) for the full research agenda.

## Project Structure

```
WM_1/
├── configs/               # Configuration files (YAML)
│   ├── data/             # Dataset generation configs
│   ├── training/         # Model training configs
│   ├── analysis/         # Analysis and evaluation configs
│   └── experiments/      # Historical experiment configs
├── src/                  # Source code
│   ├── data_processing/  # Dataset creation scripts
│   ├── training/         # Model training scripts
│   ├── analysis/         # Analysis and visualization
│   ├── tokenizer/        # Custom tokenizer setup
│   └── utils.py          # Shared utilities
├── scripts/              # Execution scripts
│   ├── data_generation/  # Dataset creation scripts
│   ├── training/         # Training execution scripts
│   ├── analysis/         # Analysis scripts
│   └── utils/            # Utility scripts
├── data/                 # Output directory (gitignored)
├── docs/                 # Documentation
│   ├── logs/             # Development logs
│   └── reports/          # Research reports
└── scratch/              # Temporary workspace (gitignored)
```

## Setup

### Environment

This project uses `uv` for Python package management:

```bash
# Clone the repository
git clone https://github.com/cfpark00/world-map-representation.git
cd WM_1

# Install dependencies
uv sync

# Configure environment
cp .env.example .env
# Edit .env to set DATA_DIR path
```

### Running Experiments

All experiments are run through scripts in the `scripts/` directory:

```bash
# Generate datasets
bash scripts/data_generation/create_dataset.sh

# Train models
bash scripts/training/train_base.sh

# Run analysis
bash scripts/analysis/run_analysis.sh
```

Or directly with Python:

```bash
# Using uv run
uv run python src/training/train.py configs/training/train_dist_1M_no_atlantis_5epochs.yaml

# Or with activated environment
source .venv/bin/activate
python src/training/train.py configs/training/train_dist_1M_no_atlantis_5epochs.yaml
```

## Key Components

### Custom Tokenizer

The project uses a character-level tokenizer optimized for geographic tasks:
- 45 tokens total: special tokens, grammar symbols, lowercase letters, digits
- Efficient encoding of city IDs and coordinates
- See `src/tokenizer/` for implementation

### Task Types

1. **Distance Prediction**: Predict geodesic distance between city pairs
   - Format: `dist(c_123,c_456)=789`
   
2. **Location Prediction**: Predict city coordinates from ID
   - Format: `c_123:(45.5,-122.6)`
   
3. **Random Walk**: Generate sequential paths through nearby cities
   - Format: `WALK:c_123->c_456->c_789`

### Model Architecture

- Small transformer models (configurable size)
- Trained with task-specific loss masking
- Multi-task support with automatic collator selection
- Checkpointing and evaluation during training

## Development Workflow

1. **Create/modify configs** in `configs/` subdirectories
2. **Run experiments** via scripts in `scripts/`
3. **Results appear** in paths specified by config `output_dir`
4. **Analysis notebooks** in `notebooks/` for exploration

## Contributing

This project follows the research template conventions:
- Implementation (HOW) in `src/` modules
- Orchestration (WHAT/WHEN) in scripts
- All configs must specify `output_dir`
- Fail fast with explicit errors (no silent fallbacks)

See [CLAUDE.md](CLAUDE.md) and [docs/repo_usage.md](docs/repo_usage.md) for detailed development guidelines.

## Citation

If you use this code for research, please cite:

```bibtex
@software{wm1_2024,
  title = {WM1: World Model Experiments},
  author = {Park, Chan},
  year = {2024},
  url = {https://github.com/cfpark00/world-map-representation}
}
```

## License

MIT License - see LICENSE file for details.