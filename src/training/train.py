#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
from transformers import (
    Trainer,
    TrainingArguments
)
import json
import argparse
from dotenv import load_dotenv

sys.path.append('.')  # Add root to path
from src.utils import (
    convert_numpy_to_python,
    save_training_plots,
    preprocess_config,
    GenerationEvalCallback,
    init_experiment_directory,
    get_model,
    get_dataset
)

def main():
    # Load environment variables
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train WM1 models with HuggingFace Trainer')
    parser.add_argument('config_path', type=str, help='Path to training config YAML file')
    parser.add_argument('--overwrite', action='store_true', 
                       help='Overwrite existing experiment directory')
    args = parser.parse_args()
    
    config_path = args.config_path
    overwrite = args.overwrite
    
    if not Path(config_path).exists():
        print(f"Error: Config file {config_path} does not exist!")
        sys.exit(1)
    
    # Load and preprocess config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate and preprocess config (handles all type conversions and validation)
    config = preprocess_config(config)
    task_type = config['task_type']
    
    # Initialize experiment directory with safety checks
    exp_dir = init_experiment_directory(config['exp_dir'], overwrite)
    
    # Save config copy to exp_dir
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Set seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets
    train_dataset, eval_dataset, tokenizer = get_dataset(config)
    
    # Initialize model (tokenizer attached as model.tokenizer)
    model = get_model(config)
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=str(exp_dir / 'checkpoints'),
        num_train_epochs=config['training']['num_epochs'],
        per_device_train_batch_size=config['training']['batch_size'],
        per_device_eval_batch_size=config['training']['eval_batch_size'],
        warmup_steps=config['training']['warmup_steps'],
        weight_decay=config['training']['weight_decay'],
        logging_dir=str(exp_dir / 'logs'),
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=config['checkpointing']['eval_steps'],  # HF handles fractional values
        save_strategy="steps",
        save_steps=config['checkpointing']['save_steps'],  # HF handles fractional values
        save_total_limit=None,  # Keep ALL checkpoints
        load_best_model_at_end=False,  # Don't mess with loading "best" model
        metric_for_best_model=None,  # Not using best model selection
        greater_is_better=False,  # Lower is better
        seed=config['training']['seed'],
        data_seed=config['training']['seed'],
        fp16=torch.cuda.is_available(),  # Use FP16 if GPU available
        dataloader_num_workers=2,
        remove_unused_columns=False,
        lr_scheduler_type="linear" if config['training']['scheduler'] == "linear_with_warmup" else config['training']['scheduler'],
        learning_rate=config['training']['learning_rate'],
    )
    
    # Initialize trainer with callback for generation evaluation and plot updates
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,  # Use processing_class instead of tokenizer (deprecated)
        callbacks=[GenerationEvalCallback(exp_dir, tokenizer, eval_dataset, device, task_type, config)],  # Generation eval + plots
    )
    
    # Train
    print("\nStarting training...")
    train_result = trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(exp_dir / 'checkpoints' / 'final'))
    
    # Save training metrics
    with open(exp_dir / 'train_results.json', 'w') as f:
        json.dump(convert_numpy_to_python(train_result.metrics), f, indent=2)
    
    # Final evaluation
    print("\nRunning final evaluation...")
    eval_metrics = trainer.evaluate()
    
    with open(exp_dir / 'eval_results.json', 'w') as f:
        json.dump(convert_numpy_to_python(eval_metrics), f, indent=2)
    
    print("\nFinal evaluation metrics:")
    for key, value in eval_metrics.items():
        if 'metric' in key or 'valid' in key:
            print(f"  {key}: {value:.2f}")
    
    # Save final plot
    save_training_plots(exp_dir, trainer.state, task_type)
    print(f"Final training summary plot saved to {exp_dir / 'summary.png'}")
    
    # Print final statistics
    if 'eval_metric_mean' in eval_metrics:
        metric_name = "distance" if task_type == 'location' else "error"
        print(f"\nFinal {metric_name} statistics:")
        print(f"  Mean: {eval_metrics['eval_metric_mean']:.2f} km")
        print(f"  Median: {eval_metrics['eval_metric_median']:.2f} km")
        print(f"  Std: {eval_metrics['eval_metric_std']:.2f} km")
        print(f"  Min: {eval_metrics['eval_metric_min']:.2f} km")
        print(f"  Max: {eval_metrics['eval_metric_max']:.2f} km")
    
    print(f"\nTraining completed! Results saved to {exp_dir}")


if __name__ == "__main__":
    main()