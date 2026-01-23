#!/usr/bin/env python3
"""Generate FTWB1 and FTWB2 training configs and scripts for Exp6."""
from pathlib import Path

CONFIG_BASE = Path('configs/revision/exp6/training')
SCRIPT_BASE = Path('scripts/revision/exp6/training')
DATA_BASE = 'data/experiments/revision/exp6'

FTWB_TEMPLATE = '''output_dir: "{output_dir}"

dataset:
  path: "{dataset_path}"
  max_sequence_length: 256

tokenizer_path: "data/tokenizers/default_tokenizer"

model:
  vocab_size: 98
  hidden_size: 128
  num_hidden_layers: 6
  num_attention_heads: 4
  intermediate_size: 512
  init_scale: 0.1
  ckpt: {ckpt_path}

training:
  batch_size: 128
  eval_batch_size: 64
  num_epochs: 30
  optimizer: "adamw"
  learning_rate: 1e-5
  weight_decay: 0.01
  scheduler: "linear_with_warmup"
  warmup_steps: 50
  seed: 42

checkpointing:
  save_strategy: "steps"
  save_steps: 0.025
  eval_strategy: "steps"
  eval_steps: 0.005

logging:
  logging_steps: 10
  report_to: "none"
'''

SCRIPT_TEMPLATE = '''#!/bin/bash
uv run python src/training/train.py {config_path} --overwrite
'''


def main():
    # Create FTWB1 configs and scripts (7)
    print("Creating FTWB1 configs and scripts...")
    for i in range(1, 8):
        config_path = CONFIG_BASE / 'ftwb1' / f'pt1_ftwb1-{i}.yaml'
        script_path = SCRIPT_BASE / 'ftwb1' / f'train_ftwb1-{i}.sh'

        config_content = FTWB_TEMPLATE.format(
            output_dir=f'{DATA_BASE}/pt1_ftwb1-{i}',
            dataset_path=f'{DATA_BASE}/datasets/ftwb1-{i}',
            ckpt_path=f'{DATA_BASE}/pt1/checkpoints/final'
        )

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)

        script_content = SCRIPT_TEMPLATE.format(config_path=config_path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        print(f"  Created: {config_path}")
        print(f"  Created: {script_path}")

    # Create FTWB2 configs and scripts (21)
    print("\nCreating FTWB2 configs and scripts...")
    for i in range(1, 22):
        config_path = CONFIG_BASE / 'ftwb2' / f'pt1_ftwb2-{i}.yaml'
        script_path = SCRIPT_BASE / 'ftwb2' / f'train_ftwb2-{i}.sh'

        config_content = FTWB_TEMPLATE.format(
            output_dir=f'{DATA_BASE}/pt1_ftwb2-{i}',
            dataset_path=f'{DATA_BASE}/datasets/ftwb2-{i}',
            ckpt_path=f'{DATA_BASE}/pt1/checkpoints/final'
        )

        config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(config_path, 'w') as f:
            f.write(config_content)

        script_content = SCRIPT_TEMPLATE.format(config_path=config_path)
        script_path.parent.mkdir(parents=True, exist_ok=True)
        with open(script_path, 'w') as f:
            f.write(script_content)
        script_path.chmod(0o755)

        print(f"  Created: {config_path}")
        print(f"  Created: {script_path}")

    # Create batch scripts
    print("\nCreating batch scripts...")

    # All FTWB1
    batch_ftwb1 = SCRIPT_BASE / 'train_all_ftwb1.sh'
    with open(batch_ftwb1, 'w') as f:
        f.write('#!/bin/bash\n')
        for i in range(1, 8):
            f.write(f'uv run python src/training/train.py {CONFIG_BASE}/ftwb1/pt1_ftwb1-{i}.yaml --overwrite\n')
    batch_ftwb1.chmod(0o755)
    print(f"  Created: {batch_ftwb1}")

    # All FTWB2
    batch_ftwb2 = SCRIPT_BASE / 'train_all_ftwb2.sh'
    with open(batch_ftwb2, 'w') as f:
        f.write('#!/bin/bash\n')
        for i in range(1, 22):
            f.write(f'uv run python src/training/train.py {CONFIG_BASE}/ftwb2/pt1_ftwb2-{i}.yaml --overwrite\n')
    batch_ftwb2.chmod(0o755)
    print(f"  Created: {batch_ftwb2}")

    print(f"\n{'='*60}")
    print(f"Total created:")
    print(f"  - 7 FTWB1 configs in {CONFIG_BASE}/ftwb1/")
    print(f"  - 7 FTWB1 scripts in {SCRIPT_BASE}/ftwb1/")
    print(f"  - 21 FTWB2 configs in {CONFIG_BASE}/ftwb2/")
    print(f"  - 21 FTWB2 scripts in {SCRIPT_BASE}/ftwb2/")
    print(f"  - 2 batch scripts (train_all_ftwb1.sh, train_all_ftwb2.sh)")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
