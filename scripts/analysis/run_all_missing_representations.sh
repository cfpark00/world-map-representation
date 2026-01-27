#!/bin/bash
# Run representation extraction for all missing representations

set -e

BASE_DIR=""
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

echo "Extracting missing representations..."
echo ""

# PT2 - All layers (3, 4, 6) that need extraction
for layer in 3 4 6; do
    echo "======================================"
    echo "PT2 Layer $layer"
    echo "======================================"
    for i in {1..8}; do
        TASK=${PT2_TASKS[$i]}
        OUTPUT_DIR="data/experiments/pt2-${i}/analysis_higher/${TASK}_firstcity_last_and_trans_l${layer}/representations"

        # Check if representations already exist
        if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null | grep checkpoint | wc -l)" -gt 30 ]; then
            echo "  pt2-${i} L${layer}: Already has representations, skipping..."
        else
            CONFIG="configs/analysis_representation_higher/ftset/pt2-${i}/${TASK}_firstcity_last_and_trans_l${layer}.yaml"
            echo "  Extracting pt2-${i} L${layer} (task: ${TASK})..."
            uv run python src/analysis/analyze_representations_higher.py $CONFIG --overwrite
        fi
    done
    echo ""
done

# PT3 - All layers (3, 4, 5, 6)
for layer in 3 4 5 6; do
    echo "======================================"
    echo "PT3 Layer $layer"
    echo "======================================"
    for i in {1..8}; do
        TASK=${PT3_TASKS[$i]}
        OUTPUT_DIR="data/experiments/pt3-${i}/analysis_higher/${TASK}_firstcity_last_and_trans_l${layer}/representations"

        # Check if representations already exist
        if [ -d "$OUTPUT_DIR" ] && [ "$(ls -A $OUTPUT_DIR 2>/dev/null | grep checkpoint | wc -l)" -gt 30 ]; then
            echo "  pt3-${i} L${layer}: Already has representations, skipping..."
        else
            CONFIG="configs/analysis_representation_higher/ftset/pt3-${i}/${TASK}_firstcity_last_and_trans_l${layer}.yaml"
            if [ -f "$CONFIG" ]; then
                echo "  Extracting pt3-${i} L${layer} (task: ${TASK})..."
                uv run python src/analysis/analyze_representations_higher.py $CONFIG --overwrite
            else
                echo "  pt3-${i} L${layer}: Config not found, creating..."
                # Create the config if it doesn't exist
                mkdir -p configs/analysis_representation_higher/ftset/pt3-${i}
                cat > $CONFIG << EOF
cities_csv: data/datasets/cities/cities.csv
device: cuda
experiment_dir: data/experiments/pt3-${i}
layers:
- ${layer}
method:
  name: linear
n_test_cities: 1250
n_train_cities: 3250
output_dir: /data/experiments/pt3-${i}/analysis_higher/${TASK}_firstcity_last_and_trans_l${layer}
perform_pca: true
probe_test: region:.* && city_id:^[1-9][0-9]{3,}$
probe_train: region:.* && city_id:^[1-9][0-9]{3,}$
prompt_format: ${TASK}_firstcity_last_and_trans
save_repr_ckpts:
- -2
seed: 42
EOF
                echo "  Config created, now extracting..."
                uv run python src/analysis/analyze_representations_higher.py $CONFIG --overwrite
            fi
        fi
    done
    echo ""
done

echo "All representations extracted!"
echo ""
echo "Now you can run CKA computations:"
echo "  bash scripts/analysis/cka/compute_cka_pt2_l3.sh"
echo "  bash scripts/analysis/cka/compute_cka_pt2_l4.sh"
echo "  bash scripts/analysis/cka/compute_cka_pt2_l6.sh"
echo "  bash scripts/analysis/cka/compute_cka_pt3_l3.sh"
echo "  bash scripts/analysis/cka/compute_cka_pt3_l4.sh"
echo "  bash scripts/analysis/cka/compute_cka_pt3_l6.sh"