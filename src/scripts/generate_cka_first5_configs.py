"""
Generate CKA config files for first-5-PCs analysis.

Based on existing configs in configs/revision/exp4/cka_cross_seed/,
but updates output paths and adds PCA parameters.
"""
import yaml
from pathlib import Path

def main():
    # Base directories
    source_config_base = Path('configs/revision/exp4/cka_cross_seed')
    target_config_base = Path('configs/revision/exp4/cka_cross_seed_first3')

    # Remove target directory if exists and recreate
    if target_config_base.exists():
        import shutil
        shutil.rmtree(target_config_base)
    target_config_base.mkdir(parents=True, exist_ok=True)

    # Find all layer config files
    layer_configs = list(source_config_base.glob('*/layer*.yaml'))

    print(f"Found {len(layer_configs)} config files to process")

    configs_created = 0

    for source_config in layer_configs:
        # Load source config
        with open(source_config, 'r') as f:
            config = yaml.safe_load(f)

        # Extract comparison directory name
        comparison_dir = source_config.parent.name
        layer_file = source_config.name

        # Create target directory
        target_dir = target_config_base / comparison_dir
        target_dir.mkdir(parents=True, exist_ok=True)

        # Update config for PCA analysis
        # Add PCA parameters
        config['n_pca_components'] = 3
        config['pca_train_filter'] = 'region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$'

        # Update output directory: replace 'cka_analysis' with 'cka_analysis_first3'
        old_output = config['output_dir']
        new_output = old_output.replace('/cka_analysis/', '/cka_analysis_first3/')
        config['output_dir'] = new_output

        # Save new config
        target_config_path = target_dir / layer_file
        with open(target_config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, sort_keys=False)

        configs_created += 1

    print(f"\nCreated {configs_created} config files in {target_config_base}")
    print(f"\nConfig structure:")
    print(f"  - Added 'n_pca_components: 3'")
    print(f"  - Added 'pca_train_filter' to exclude Atlantis from PCA training")
    print(f"  - Updated output_dir: cka_analysis -> cka_analysis_first3")


if __name__ == '__main__':
    main()
