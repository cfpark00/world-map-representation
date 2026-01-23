#!/usr/bin/env python3
"""
Generate PCA timeline configs for the 4 PT1 base models.
Creates 3 configs per model: mixed, raw, na (no atlantis).
Total: 4 models × 3 types = 12 configs
"""

from pathlib import Path

BASE_EXPERIMENTS = Path('data/experiments')
CONFIG_DIR = Path('configs/revision/exp1_pt1_pca')
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# PT1 base models
PT1_MODELS = [
    ('pt1', BASE_EXPERIMENTS / 'pt1'),
    ('pt1_seed1', BASE_EXPERIMENTS / 'revision/exp1/pt1_seed1'),
    ('pt1_seed2', BASE_EXPERIMENTS / 'revision/exp1/pt1_seed2'),
    ('pt1_seed3', BASE_EXPERIMENTS / 'revision/exp1/pt1_seed3'),
]

# Config templates
MIXED_TEMPLATE = """cities_csv: data/datasets/cities/cities.csv
layer_index: -1
marker_size: 3
n_components: 3
output_dir: {output_dir}
probe_test: region:.* && city_id:^[1-9][0-9]{{3,}}$
probe_train: region:.* && city_id:^[1-9][0-9]{{3,}}$
representations_base_path: {repr_path}
token_index: -1
train_frac: 0.6
axis_mapping:
  type: mixed
  1: x
  2: y
  3: r0
"""

RAW_TEMPLATE = """cities_csv: data/datasets/cities/cities.csv
layer_index: -1
marker_size: 3
n_components: 3
output_dir: {output_dir}
probe_test: region:.* && city_id:^[1-9][0-9]{{3,}}$
probe_train: region:.* && city_id:^[1-9][0-9]{{3,}}$
representations_base_path: {repr_path}
token_index: -1
train_frac: 0.6
axis_mapping:
  type: pca
  1: 0
  2: 1
  3: 2
"""

NA_TEMPLATE = """cities_csv: data/datasets/cities/cities.csv
layer_index: -1
marker_size: 3
n_components: 3
output_dir: {output_dir}
probe_test: region:.* && city_id:^[1-9][0-9]{{3,}}$
probe_train: region:^(?!Atlantis$).* && city_id:^[1-9][0-9]{{3,}}$
representations_base_path: {repr_path}
token_index: -1
train_frac: 0.6
axis_mapping:
  type: mixed
  1: x
  2: y
  3: r0
"""

def main():
    configs_created = []

    for model_name, exp_dir in PT1_MODELS:
        # Find the L5 representations directory
        l5_dir = exp_dir / 'analysis_higher' / 'distance_firstcity_last_and_trans_l5'
        repr_path = l5_dir / 'representations'

        if not repr_path.exists():
            print(f"WARNING: No representations for {model_name} at {repr_path}")
            continue

        # Create 3 configs for each model
        for pca_type, template in [('mixed', MIXED_TEMPLATE), ('raw', RAW_TEMPLATE), ('na', NA_TEMPLATE)]:
            output_dir = l5_dir / f'pca_timeline{"" if pca_type == "mixed" else "_" + pca_type}'

            config_content = template.format(
                output_dir=output_dir.resolve(),
                repr_path=repr_path.resolve(),
            )

            config_path = CONFIG_DIR / f'{model_name}_{pca_type}.yaml'
            config_path.write_text(config_content)
            configs_created.append(config_path)
            print(f"Created: {config_path}")

    print(f"\nTotal configs created: {len(configs_created)}")
    return configs_created

if __name__ == '__main__':
    main()
