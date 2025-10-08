#!/bin/bash
# Create layer 3 configs for all pt2 experiments

set -e

BASE_DIR="/n/home12/cfpark00/WM_1"
cd $BASE_DIR

echo "Creating layer 3 configs for pt2 experiments..."

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

# Create configs for layer 3 for each pt2 experiment
for i in {1..8}; do
    TASK=${TASKS[$i]}
    CONFIG_DIR="configs/analysis_representation_higher/ftset/pt2-${i}"

    # Create layer 3 config
    cat > ${CONFIG_DIR}/${TASK}_firstcity_last_and_trans_l3.yaml << EOF
cities_csv: data/datasets/cities/cities.csv
device: cuda
experiment_dir: data/experiments/pt2-${i}
layers:
- 3
method:
  name: linear
n_test_cities: 1250
n_train_cities: 3250
output_dir: /n/home12/cfpark00/WM_1/data/experiments/pt2-${i}/analysis_higher/${TASK}_firstcity_last_and_trans_l3
perform_pca: true
probe_test: region:.* && city_id:^[1-9][0-9]{3,}$
probe_train: region:.* && city_id:^[1-9][0-9]{3,}$
prompt_format: ${TASK}_firstcity_last_and_trans
save_repr_ckpts:
- -2
seed: 42
EOF
    echo "Created: pt2-${i}/${TASK}_firstcity_last_and_trans_l3.yaml"
done

echo ""
echo "Configs created! Now run representation extraction with:"
echo ""
echo "for i in {1..8}; do"
echo "  bash scripts/analysis_representations_higher/ftset/run_pt2-\${i}_l3.sh"
echo "done"