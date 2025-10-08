#!/bin/bash
echo "Running dimensionality analysis across PT1 and PT2 models"
echo "========================================================="
for i in 1 2 3; do
    exp_name="pt1-$i"
    if [ -d "data/experiments/$exp_name" ]; then
        echo "Processing $exp_name..."

        task=$([ $i -eq 1 ] && echo "distance" || [ $i -eq 2 ] && echo "trianglearea" || echo "angle")

        config_file="configs/analysis_dimensionality/pt1/${exp_name}_${task}_l5.yaml"
        extract_config="configs/extract_representations/${exp_name}_${task}_l5.yaml"

        cat > $extract_config << EOF
output_dir: data/experiments/$exp_name/analysis_higher/${task}_firstcity_last_and_trans_l5/representations
experiment: $exp_name
experiment_dir: data/experiments/$exp_name
prompt_format: ${task}_firstcity_last_and_trans
cities_csv: data/datasets/cities/cities.csv
layer_index: 5
token_index: -1
max_samples: 5000
EOF

        cat > $config_file << EOF
output_dir: data/experiments/$exp_name/analysis_dimensionality/${task}_l5
representations_base_path: data/experiments/$exp_name/analysis_higher/${task}_firstcity_last_and_trans_l5/representations
max_samples: 5000
twonn_k: 20
mle_k_max: 20
lpe_neighbors: 50
EOF

        uv run python src/scripts/extract_and_save_representations.py $extract_config --overwrite
        uv run python src/scripts/analyze_dimensionality.py $config_file --overwrite
        echo ""
    fi
done
for i in 1 2 3; do
    exp_name="pt2-$i"
    if [ -d "data/experiments/$exp_name" ]; then
        echo "Processing $exp_name..."

        task=$([ $i -eq 1 ] && echo "distance" || [ $i -eq 2 ] && echo "angle" || echo "inside")

        config_file="configs/analysis_dimensionality/pt2/${exp_name}_${task}_l5.yaml"
        extract_config="configs/extract_representations/${exp_name}_${task}_l5.yaml"

        cat > $extract_config << EOF
output_dir: data/experiments/$exp_name/analysis_higher/${task}_firstcity_last_and_trans_l5/representations
experiment: $exp_name
experiment_dir: data/experiments/$exp_name
prompt_format: ${task}_firstcity_last_and_trans
cities_csv: data/datasets/cities/cities.csv
layer_index: 5
token_index: -1
max_samples: 5000
EOF

        cat > $config_file << EOF
output_dir: data/experiments/$exp_name/analysis_dimensionality/${task}_l5
representations_base_path: data/experiments/$exp_name/analysis_higher/${task}_firstcity_last_and_trans_l5/representations
max_samples: 5000
twonn_k: 20
mle_k_max: 20
lpe_neighbors: 50
EOF

        uv run python src/scripts/extract_and_save_representations.py $extract_config --overwrite
        uv run python src/scripts/analyze_dimensionality.py $config_file --overwrite
        echo ""
    fi
done