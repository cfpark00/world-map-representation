#!/bin/bash
cd 
for exp in pt2-1 pt2-2 pt2-3 pt2-4 pt2-5 pt2-6 pt2-7 pt2-8; do
    task=$(grep prompt_format configs/analysis_representation_higher/ftset/${exp}/*.yaml | sed 's/.*: //' | sed 's/_firstcity.*//')
    echo "Processing $exp with task: $task"
    config="configs/analysis_representation_higher/ftset/${exp}/${task}_firstcity_last_and_trans_l5.yaml"
    uv run python src/analysis/analyze_representations_higher.py $config --overwrite
done
for exp in pt3-1 pt3-2 pt3-3 pt3-4 pt3-5 pt3-6 pt3-7 pt3-8; do
    task=$(grep prompt_format configs/analysis_representation_higher/ftset/${exp}/*.yaml | sed 's/.*: //' | sed 's/_firstcity.*//')
    echo "Processing $exp with task: $task"
    config="configs/analysis_representation_higher/ftset/${exp}/${task}_firstcity_last_and_trans_l5.yaml"
    uv run python src/analysis/analyze_representations_higher.py $config --overwrite
done
echo "All analyses completed!"