#!/bin/bash
#uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft3-1.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft3-2.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft3-3.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft3-4.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft3-5.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft3-6.yaml --overwrite
uv run python src/data_processing/combine_datasets.py configs/data_generation/ftset/combine_ft3-7.yaml --overwrite

