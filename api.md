# WM_1 API Documentation

The WM_1 project trains transformer language models to predict geographic distances and locations between world cities using coordinate tokens.

## Data Processing

### Generate Filtered City Dataset
```bash
python src/data_processing/generate_filtered_dataset.py [pop_threshold] [--seed SEED]
```
- `pop_threshold`: Minimum population threshold (default: 50000)
- `--seed`: Random seed (default: 42)

Creates filtered CSV of cities above population threshold in `outputs/datasets/`

### Create Distance Dataset
```bash
python src/data_processing/create_distance_dataset.py OUTPUT_DIR \
    [--n_train N] [--n_val N] [--n_test N] \
    [--seed SEED] [--cities-csv PATH]
```
- `OUTPUT_DIR`: Output directory for the dataset
- `--n_train`: Number of training samples (default: 100000)
- `--n_val`: Number of validation samples (default: 128)
- `--n_test`: Number of test samples (default: 10000)
- `--seed`: Random seed (default: 42)
- `--cities-csv`: Path to cities CSV file

Creates distance prediction dataset with format: `dist(c_123,c_456)=78.9km`

### Create Location Dataset
```bash
python src/data_processing/create_location_dataset.py [N_TRAIN] OUTPUT_DIR \
    [--n_val N] [--all] [--seed SEED] [--cities-csv PATH]
```
- `N_TRAIN`: Number of training samples
- `OUTPUT_DIR`: Output directory for the dataset
- `--n_val`: Number of validation samples (default: 0)
- `--all`: Use all cities (one sample per city)
- `--seed`: Random seed (default: 42)
- `--cities-csv`: Path to cities CSV file

Creates location prediction dataset with format: `loc(c_123)=45.67,-123.45`

### Create Random Walk Dataset
```bash
python src/data_processing/create_randomwalk_dataset.py OUTPUT_DIR \
    [--n_train N] [--n_val N] [--n_test N] \
    [--seed SEED] [--cities-csv PATH] \
    [--max-length N] [--distance-km KM] [--visualize N]
```
- `OUTPUT_DIR`: Output directory for the dataset
- `--n_train`: Number of training samples (default: 10000)
- `--n_val`: Number of validation samples (default: 128)
- `--n_test`: Number of test samples (default: 1000)
- `--seed`: Random seed (default: 42)
- `--cities-csv`: Path to cities CSV file
- `--max-length`: Maximum sequence length (default: 32)
- `--distance-km`: Distance threshold in km (default: 200)
- `--visualize`: Number of sequences to visualize (default: 10)

Creates random walk sequences of nearby cities within distance threshold.

## Tokenizer

### Create HuggingFace Tokenizer
```bash
python src/tokenizer/create_hf_tokenizer.py [--save-path PATH]
```
- `--save-path`: Output path for tokenizer (default: `outputs/tokenizer/wm1_tokenizer`)

Creates custom tokenizer with city tokens (c_0 to c_99999) and special tokens.

## Training

### Train Model
```bash
python src/training/train.py CONFIG_PATH [--overwrite]
```
- `CONFIG_PATH`: Path to training config YAML file
- `--overwrite`: Overwrite existing experiment directory

Trains model using HuggingFace Trainer with configuration from YAML file.

Example config structure:
```yaml
experiment_name: dist_100k_1M_20epochs
task_type: distance  # or location, randomwalk
dataset_path: outputs/datasets/distance_100k_1M
tokenizer_path: outputs/tokenizer/wm1_tokenizer
model:
  num_hidden_layers: 6
  hidden_size: 256
  num_attention_heads: 8
  intermediate_size: 1024
training:
  num_train_epochs: 20
  per_device_train_batch_size: 128
  learning_rate: 1e-3
  warmup_ratio: 0.05
```

## Analysis

### Analyze Representations
```bash
python src/analysis/analyze_representations.py \
    --exp_dir EXP_DIR --cities_csv CSV_PATH \
    [--layers L1 L2 ...] [--n_probe_cities N] \
    [--n_train_cities N] [--device DEVICE]
```
- `--exp_dir`: Path to experiment directory with checkpoints
- `--cities_csv`: Path to cities CSV file
- `--layers`: Layer indices to extract (default: 3 4)
- `--n_probe_cities`: Number of cities to probe (default: 5000)
- `--n_train_cities`: Number for training probes (default: 3000)
- `--device`: Device to use (default: cuda if available)

Analyzes how internal representations evolve during training. Outputs:
- `analysis/representation_dynamics.csv`: R² scores per checkpoint
- `analysis/representation_dynamics_layers*.png`: R² and loss plots
- `analysis/world_map_evolution_layers*.gif`: Animated prediction map

## Visualization

### Create City Map
```bash
python src/visualization/create_city_map.py [POP_THRESHOLD]
```
- `POP_THRESHOLD`: Minimum population (default: 100000)

Creates world map visualization of cities above population threshold.

### Create Population Histogram
```bash
python src/visualization/create_population_histogram.py [POP_THRESHOLD]
```
- `POP_THRESHOLD`: Minimum population (default: 100000)

Creates histogram of city population distribution.

## Usage Examples

```bash
# 1. Generate city dataset
python src/data_processing/generate_filtered_dataset.py 100000

# 2. Create tokenizer
python src/tokenizer/create_hf_tokenizer.py

# 3. Create training dataset
python src/data_processing/create_distance_dataset.py outputs/datasets/dist_100k_1M \
    --n_train 100000 --n_val 128 --n_test 10000

# 4. Train model
python src/training/train.py configs/dist_100k_1M_20epochs.yaml

# 5. Analyze representations
python src/analysis/analyze_representations.py \
    --exp_dir outputs/experiments/dist_100k_1M_20epochs \
    --cities_csv outputs/datasets/cities_100k_plus_seed42.csv \
    --layers 3 4
```