#!/bin/bash
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_ft2-1.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_ft2-2.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_ft2-3.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_ft2-4.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_ft2-5.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_ft2-6.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/combine_ft2-7.yaml --overwrite

