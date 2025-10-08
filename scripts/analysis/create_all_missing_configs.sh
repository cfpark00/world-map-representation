#!/bin/bash
# Create all missing layer configs for pt2 and pt3

set -e

BASE_DIR="/n/home12/cfpark00/WM_1"
cd $BASE_DIR

# Task mappings for pt2
declare -A PT2_TASKS
PT2_TASKS[1]="distance"
PT2_TASKS[2]="angle"
PT2_TASKS[3]="inside"
PT2_TASKS[4]="crossing"
PT2_TASKS[5]="trianglearea"
PT2_TASKS[6]="compass"
PT2_TASKS[7]="perimeter"
PT2_TASKS[8]="distance"

# Task mappings for pt3
declare -A PT3_TASKS
PT3_TASKS[1]="distance"
PT3_TASKS[2]="compass"
PT3_TASKS[3]="crossing"
PT3_TASKS[4]="angle"
PT3_TASKS[5]="perimeter"
PT3_TASKS[6]="trianglearea"
PT3_TASKS[7]="inside"
PT3_TASKS[8]="distance"

echo "Creating missing configs..."
echo ""

# PT2 - Layer 3, 4, 6
for layer in 3 4 6; do
    echo "Creating PT2 Layer $layer configs..."
    for i in {1..8}; do
        TASK=${PT2_TASKS[$i]}
        CONFIG_DIR="configs/analysis_representation_higher/ftset/pt2-${i}"
        CONFIG_FILE="${CONFIG_DIR}/${TASK}_firstcity_last_and_trans_l${layer}.yaml"

        # Only create if doesn't exist
        if [ ! -f "$CONFIG_FILE" ]; then
            cat > $CONFIG_FILE << EOF
cities_csv: data/datasets/cities/cities.csv
device: cuda
experiment_dir: data/experiments/pt2-${i}
layers:
- ${layer}
method:
  name: linear
n_test_cities: 1250
n_train_cities: 3250
output_dir: /n/home12/cfpark00/WM_1/data/experiments/pt2-${i}/analysis_higher/${TASK}_firstcity_last_and_trans_l${layer}
perform_pca: true
probe_test: region:.* && city_id:^[1-9][0-9]{3,}$
probe_train: region:.* && city_id:^[1-9][0-9]{3,}$
prompt_format: ${TASK}_firstcity_last_and_trans
save_repr_ckpts:
- -2
seed: 42
EOF
            echo "  Created: pt2-${i}/${TASK}_firstcity_last_and_trans_l${layer}.yaml"
        fi
    done
done

echo ""

# PT3 - Layer 3, 4, 6
for layer in 3 4 6; do
    echo "Creating PT3 Layer $layer configs..."
    for i in {1..8}; do
        TASK=${PT3_TASKS[$i]}
        CONFIG_DIR="configs/analysis_representation_higher/ftset/pt3-${i}"
        mkdir -p $CONFIG_DIR
        CONFIG_FILE="${CONFIG_DIR}/${TASK}_firstcity_last_and_trans_l${layer}.yaml"

        # Only create if doesn't exist
        if [ ! -f "$CONFIG_FILE" ]; then
            cat > $CONFIG_FILE << EOF
cities_csv: data/datasets/cities/cities.csv
device: cuda
experiment_dir: data/experiments/pt3-${i}
layers:
- ${layer}
method:
  name: linear
n_test_cities: 1250
n_train_cities: 3250
output_dir: /n/home12/cfpark00/WM_1/data/experiments/pt3-${i}/analysis_higher/${TASK}_firstcity_last_and_trans_l${layer}
perform_pca: true
probe_test: region:.* && city_id:^[1-9][0-9]{3,}$
probe_train: region:.* && city_id:^[1-9][0-9]{3,}$
prompt_format: ${TASK}_firstcity_last_and_trans
save_repr_ckpts:
- -2
seed: 42
EOF
            echo "  Created: pt3-${i}/${TASK}_firstcity_last_and_trans_l${layer}.yaml"
        fi
    done
done

echo ""
echo "All configs created!"