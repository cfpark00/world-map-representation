#!/bin/bash
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft1-1.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft1-2.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft1-3.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft1-4.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft1-5.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft1-6.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft1-7.yaml --overwrite


