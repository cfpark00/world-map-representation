#!/bin/bash
# Run representation extraction for pt2 layer 3

set -e

BASE_DIR="/n/home12/cfpark00/WM_1"
cd $BASE_DIR

# First create the configs
echo "Creating configs..."
bash scripts/analysis/create_pt2_layer3_configs.sh

# Task mappings
declare -A TASKS
TASKS[1]="distance"
TASKS[2]="angle"
TASKS[3]="inside"
TASKS[4]="crossing"
TASKS[5]="trianglearea"
TASKS[6]="compass"
TASKS[7]="perimeter"
TASKS[8]="distance"

echo ""
echo "Running representation extraction for all pt2 layer 3..."
echo ""

for i in {1..8}; do
    TASK=${TASKS[$i]}
    CONFIG="configs/analysis_representation_higher/ftset/pt2-${i}/${TASK}_firstcity_last_and_trans_l3.yaml"

    echo "======================================"
    echo "Extracting pt2-${i} layer 3 (task: ${TASK})..."
    echo "======================================"

    uv run python src/analysis/analyze_representations_higher.py $CONFIG --overwrite

    echo "Done with pt2-${i}"
    echo ""
done

echo "All pt2 layer 3 representations extracted!"
echo ""
echo "Now you can run CKA computation with:"
echo "bash scripts/analysis/cka/compute_cka_pt2_l3.sh"