#!/bin/bash
# Fix incomplete layer 3 data for pt2 experiments

set -e

BASE_DIR=""
cd $BASE_DIR

echo "Creating representation extraction configs for pt2 layer 3..."

# Task mappings for pt2
declare -A TASKS
TASKS[1]="distance"
TASKS[2]="angle"
TASKS[3]="inside"
TASKS[4]="crossing"
TASKS[5]="trianglearea"
TASKS[6]="compass"
TASKS[7]="perimeter"
TASKS[8]="distance"

# Create configs for layer 3
for i in {1..8}; do
    TASK=${TASKS[$i]}
    CONFIG_DIR="configs/analysis_representation_higher/ftset/pt2-${i}"
    mkdir -p $CONFIG_DIR

    # Create layer 3 config based on layer 5
    cat > ${CONFIG_DIR}/${TASK}_firstcity_last_and_trans_l3.yaml << EOF
# Analyze representations for pt2-${i} layer 3
# Task: ${TASK}

cities_csv: data/datasets/cities/cities.csv
device: cuda
experiment_dir: data/experiments/pt2-${i}
layers: [3]
method:
  name: linear
n_test_cities: 1250
n_train_cities: 3250
output_dir: data/experiments/pt2-${i}/analysis_higher/${TASK}_firstcity_last_and_trans_l3
perform_pca: true
probe_test: 'region:.* && city_id:^[1-9][0-9]{3,}$'
probe_train: 'region:.* && city_id:^[1-9][0-9]{3,}$'
prompt_format: ${TASK}_firstcity_last_and_trans
save_repr_ckpts: null  # Save all checkpoints
seed: 42
EOF
    echo "Created config for pt2-${i} layer 3"
done

echo ""
echo "Running representation extraction for layer 3..."
for i in {1..8}; do
    TASK=${TASKS[$i]}
    CONFIG="configs/analysis_representation_higher/ftset/pt2-${i}/${TASK}_firstcity_last_and_trans_l3.yaml"
    echo "Extracting representations for pt2-${i} layer 3..."
    uv run python src/analysis/analyze_representations_higher.py $CONFIG --overwrite
done

echo ""
echo "Re-running CKA computation for pt2 layer 3..."
bash scripts/analysis/cka/compute_cka_pt2_l3.sh

echo ""
echo "Re-collecting all CKA data..."
uv run python scripts/analysis/collect_all_cka_data.py

echo ""
echo "Re-generating plots..."
uv run python scripts/analysis/plot_cka_timelines_non_overlapping.py

echo "Done! Check the updated plots in scratch/cka_timelines_non_overlap/"