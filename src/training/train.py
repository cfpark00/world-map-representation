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
from collections import Counter

sys.path.append('.')  # Add root to path
from src.utils import (
    convert_numpy_to_python,
    save_training_plots,
    preprocess_config,
    GenerationEvalCallback,
    init_directory,
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
    
    # Initialize experiment directory with safety checks
    exp_dir = init_directory(config['exp_dir'], overwrite)
    
    # Create checkpoints subdirectory
    (exp_dir / 'checkpoints').mkdir(exist_ok=False)
    
    # Save config copy to exp_dir
    with open(exp_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f)
    
    # Set seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load datasets and collator
    train_dataset, eval_dataset, tokenizer, collator = get_dataset(config)
    
    # Detect task types in dataset (support multi-task)
    sample_size = min(100, len(train_dataset))
    task_counts = Counter(train_dataset[i].get('task_type', 'unknown') for i in range(sample_size))
    task_types = list(task_counts.keys())
    
    if len(task_types) == 1:
        print(f"Single task type detected: {task_types[0]} ({task_counts[task_types[0]]} samples)")
    else:
        print(f"Multi-task dataset detected: {dict(task_counts)}")
        print(f"Task types: {task_types}")
    
    # For backward compatibility, use predominant task type as primary
    primary_task_type = task_counts.most_common(1)[0][0]
    
    # Initialize model (tokenizer attached as model.tokenizer)
    model = get_model(config)
    
    # Save checkpoint-0 (initial model state, either random or pretrained)
    checkpoint_0_path = exp_dir / 'checkpoints' / 'checkpoint-0'
    checkpoint_0_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(checkpoint_0_path))
    tokenizer.save_pretrained(str(checkpoint_0_path))
    print(f"Saved initial model state to {checkpoint_0_path}")
    
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
        data_collator=collator,  # Use task-specific collator
        processing_class=tokenizer,  # Use processing_class instead of tokenizer (deprecated)
        callbacks=[GenerationEvalCallback(exp_dir, tokenizer, eval_dataset, device, task_types, config)],  # Generation eval + plots
    )
    
    # Always evaluate initial model (step 0) - whether from checkpoint or random init
    print("\nEvaluating initial model (step 0)...")
    initial_metrics = trainer.evaluate()
    
    # Run generation-based evaluation for step 0
    from src.utils import evaluate_with_generation
    print("Performing generation-based evaluation for checkpoint-0...")
    gen_metrics = evaluate_with_generation(
        model, eval_dataset, tokenizer, device, 
        primary_task_type, num_samples=64, batch_size=16, config=config
    )
    
    # Combine standard and generation metrics
    if gen_metrics:
        initial_metrics.update(convert_numpy_to_python(gen_metrics))
    
    # Add initial metrics to trainer's log history for plotting
    # This ensures checkpoint-0 appears in plots
    initial_log_entry = {
        'step': 0,
        'epoch': 0.0,
        'eval_loss': initial_metrics.get('eval_loss', float('inf')),
    }
    initial_log_entry.update(convert_numpy_to_python(gen_metrics) if gen_metrics else {})
    trainer.state.log_history.insert(0, initial_log_entry)
    
    # Save initial evaluation metrics
    initial_metrics_path = exp_dir / 'checkpoints' / 'checkpoint-0'
    initial_metrics_path.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    with open(initial_metrics_path / 'eval_results.json', 'w') as f:
        json.dump(convert_numpy_to_python(initial_metrics), f, indent=2)
    
    # Also save the model to checkpoint-0 for consistency
    trainer.save_model(str(initial_metrics_path))
    
    print(f"Initial metrics saved to {initial_metrics_path}")
    print(f"Initial eval loss: {initial_metrics.get('eval_loss', 'N/A'):.4f}")
    
    if gen_metrics:
        if primary_task_type == 'location':
            print(f"Initial avg haversine distance: {gen_metrics['eval_metric_mean']:.2f} km")
        elif primary_task_type == 'distance':
            print(f"Initial avg absolute error: {gen_metrics['eval_metric_mean']:.2f} km")
        else:  # randomwalk
            print(f"Initial avg walk validity: {gen_metrics['eval_metric_mean']:.3f}")
    
    if config['model'].get('ckpt'):
        print("(Loaded from checkpoint)")
    else:
        print("(Random initialization - chance level)")
    
    # Save initial plot with checkpoint-0 data
    save_training_plots(exp_dir, trainer.state, primary_task_type)
    print(f"Initial plots saved to {exp_dir / 'summary/'}")
    
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
    save_training_plots(exp_dir, trainer.state, primary_task_type)
    print(f"Final training plots saved to {exp_dir / 'summary/'}")
    
    # Print final statistics
    if 'eval_metric_mean' in eval_metrics:
        metric_name = "distance" if primary_task_type == 'location' else "error"
        print(f"\nFinal {metric_name} statistics:")
        print(f"  Mean: {eval_metrics['eval_metric_mean']:.2f} km")
        print(f"  Median: {eval_metrics['eval_metric_median']:.2f} km")
        print(f"  Std: {eval_metrics['eval_metric_std']:.2f} km")
        print(f"  Min: {eval_metrics['eval_metric_min']:.2f} km")
        print(f"  Max: {eval_metrics['eval_metric_max']:.2f} km")
    
    print(f"\nTraining completed! Results saved to {exp_dir}")


if __name__ == "__main__":
    main()