#!/bin/bash
cd /n/home12/cfpark00/WM_1
for exp in pt1_ft2-1 pt1_ft2-2 pt1_ft2-4 pt1_ft2-6; do
    echo "Processing $exp..."
    repr_dir="data/experiments/${exp}/analysis_higher/distance_firstcity_last_and_trans_l5/representations"
    if [ ! -d "$repr_dir" ] || [ $(ls -d $repr_dir/checkpoint-* 2>/dev/null | wc -l) -eq 0 ]; then
        echo "  Generating representations for $exp..."
        uv run python src/analysis/analyze_representations_higher.py configs/analysis_representation_higher/ftset/${exp}/distance_firstcity_last_and_trans_l5.yaml --overwrite
    else
        echo "  Representations already exist for $exp ($(ls -d $repr_dir/checkpoint-* | wc -l) checkpoints)"
    fi
    echo "  Running temporal PCA visualization for $exp..."
    uv run python src/analysis/visualize_pca_3d_timeline.py configs/analysis_pca_timeline/ftset/${exp}/distance_firstcity_last_and_trans_l5_temporal_atlantis.yaml --overwrite
    echo "  Completed $exp"
    echo ""
done
echo "All experiments completed!"