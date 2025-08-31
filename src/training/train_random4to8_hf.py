#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    get_linear_schedule_with_warmup,
    TrainerCallback,
    default_data_collator
)
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
import json
import argparse
from dotenv import load_dotenv
from datasets import load_from_disk
from typing import Dict
import torch.nn as nn

sys.path.append('.')  # Add root to path
from src.utils import BaseDataset

# Load environment variables
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train random 4-to-4 mapping model with HuggingFace Trainer')
parser.add_argument('config_path', type=str, help='Path to training config YAML file')
parser.add_argument('--overwrite', action='store_true', 
                   help='Overwrite existing experiment directory')
args = parser.parse_args()

config_path = args.config_path
overwrite = args.overwrite

if not Path(config_path).exists():
    print(f"Error: Config file {config_path} does not exist!")
    sys.exit(1)

# Load config
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Convert all numeric values from strings if needed
config['training']['learning_rate'] = float(config['training']['learning_rate'])
config['training']['weight_decay'] = float(config['training']['weight_decay'])
config['training']['batch_size'] = int(config['training']['batch_size'])
config['training']['eval_batch_size'] = int(config['training'].get('eval_batch_size', config['training']['batch_size']))
config['training']['num_epochs'] = int(config['training']['num_epochs'])
config['training']['warmup_steps'] = int(config['training']['warmup_steps'])
config['training']['seed'] = int(config['training']['seed'])
config['training']['loss_mask_type'] = config['training'].get('loss_mask_type', None)
config['training']['scheduler'] = config['training'].get('scheduler', 'linear_with_warmup')
config['checkpointing']['save_steps'] = float(config['checkpointing']['save_steps'])
config['checkpointing']['eval_steps'] = float(config['checkpointing']['eval_steps'])
config['dataset']['max_sequence_length'] = int(config['dataset']['max_sequence_length'])
config['model']['vocab_size'] = int(config['model']['vocab_size'])
config['model']['hidden_size'] = int(config['model']['hidden_size'])
config['model']['num_hidden_layers'] = int(config['model']['num_hidden_layers'])
config['model']['num_attention_heads'] = int(config['model']['num_attention_heads'])
config['model']['intermediate_size'] = int(config['model']['intermediate_size'])
config['model']['init_scale'] = float(config['model'].get('init_scale', 0.02))

# Check if experiment directory exists
exp_dir = Path(config['exp_dir'])

if exp_dir.exists():
    if overwrite:
        # Get the EXP_DIR_PREFIX from environment
        exp_dir_prefix = os.environ.get('EXP_DIR_PREFIX')
        
        if not exp_dir_prefix:
            print(f"Error: EXP_DIR_PREFIX not set in .env file!")
            print("Cannot use --overwrite without EXP_DIR_PREFIX for safety.")
            sys.exit(1)
        
        # Get absolute path of exp_dir
        exp_dir_absolute = exp_dir.resolve()
        
        # Check if the absolute path starts with EXP_DIR_PREFIX
        if not str(exp_dir_absolute).startswith(exp_dir_prefix):
            print(f"Error: Cannot overwrite {exp_dir_absolute}")
            print(f"Directory must start with EXP_DIR_PREFIX: {exp_dir_prefix}")
            print("This safety check prevents accidental deletion of important directories.")
            sys.exit(1)
        
        # Safe to remove
        print(f"Removing existing experiment directory: {exp_dir_absolute}")
        import shutil
        shutil.rmtree(exp_dir_absolute)
        print("Directory removed successfully.")
    else:
        print(f"Error: Experiment directory {exp_dir} already exists!")
        print("Use --overwrite to remove it, or choose a different exp_dir in the config.")
        sys.exit(1)

# Create experiment directories
exp_dir.mkdir(parents=True, exist_ok=False)
(exp_dir / 'checkpoints').mkdir()
print(f"Created experiment directory: {exp_dir.resolve()}")

# Save config copy to exp_dir
with open(exp_dir / 'config.yaml', 'w') as f:
    yaml.dump(config, f)


def compute_metrics(eval_preds: EvalPrediction) -> Dict[str, float]:
    """
    Compute metrics during evaluation.
    Since we need generation for exact match, this just returns empty.
    The actual generation metrics are computed in GenerationEvalCallback.
    """
    # Loss is computed automatically by trainer
    # Exact match requires generation, handled in callback
    return {}


def parse_random4to4(text):
    """Parse a random 4-to-4 mapping like 'loc(c_1234)=5678'.
    Returns the 4-digit string or None if invalid."""
    if '=' in text:
        parts = text.split('=')
        if len(parts) >= 2:
            # Get the right side (output)
            right_side = parts[1].replace('<eos>', '').strip()
            # Return the right side (may not be exactly 4 digits if generation is wrong)
            return right_side
    return None


def count_matching_digits(true_str, gen_str):
    """Count how many digits match in the same position.
    For comma-separated format like '1234,5678', counts digit matches only (not comma).
    Returns a value from 0 to 8 (4 digits + 4 digits)."""
    if true_str is None or gen_str is None:
        return 0
    
    matches = 0
    for i in range(min(len(true_str), len(gen_str))):
        # Skip the comma at position 4
        if i == 4:
            continue
        if i < len(true_str) and i < len(gen_str):
            if true_str[i] == gen_str[i]:
                matches += 1
    return matches


class GenerationEvalCallback(TrainerCallback):
    """Callback to perform generation-based evaluation and update plots."""
    
    def __init__(self, exp_dir, tokenizer, eval_dataset, device):
        self.exp_dir = exp_dir
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.device = device
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called after standard evaluation - add generation metrics."""
        if model is None:
            return control
            
        # Perform generation-based evaluation
        print("\nPerforming generation-based evaluation...")
        num_samples = min(64, len(self.eval_dataset))  # Use all validation samples if less than 64
        gen_metrics = evaluate_with_generation(
            model, self.eval_dataset, self.tokenizer, self.device, 
            num_samples=num_samples, batch_size=16
        )
        
        # Add metrics to the log
        if state.log_history and gen_metrics:
            state.log_history[-1].update(gen_metrics)
        
        # Also add to metrics dict if provided (for best model tracking)
        if metrics is not None and gen_metrics:
            metrics.update(gen_metrics)
        
        # Save plots after each evaluation
        save_training_plots(None, self.exp_dir, state)
        
        # Print generation metrics
        if gen_metrics:
            print(f"\n[Evaluation Metrics]")
            print(f"  Avg digit matches: {gen_metrics.get('eval_digit_matches_mean', 0):.2f}/8 (±{gen_metrics.get('eval_digit_matches_std', 0):.2f})")
            print(f"  Min digit matches: {gen_metrics.get('eval_digit_matches_min', 0)}")
            print(f"  Max digit matches: {gen_metrics.get('eval_digit_matches_max', 0)}")
            print(f"  Median digit matches: {gen_metrics.get('eval_digit_matches_median', 0):.1f}")
            print(f"  Valid generations: {gen_metrics.get('eval_valid_count', 0)}/{num_samples} ({gen_metrics.get('eval_valid_ratio', 0)*100:.1f}%)")
        
        return control


def evaluate_with_generation(model, eval_dataset, tokenizer, device, num_samples=128, batch_size=16):
    """Evaluate model with generation and digit matching calculation."""
    model.eval()
    
    # Sample a subset for evaluation
    eval_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
    
    digit_matches = []
    all_generated_texts = []
    all_true_texts = []
    
    # Process in batches
    num_batches = (len(eval_indices) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(eval_indices))
            batch_indices = eval_indices[batch_start:batch_end]
            
            # Get prompts and true completions
            batch_prompts = []
            batch_true_completions = []
            
            for idx in batch_indices:
                idx = int(idx)
                raw_item = eval_dataset.dataset[idx]
                batch_prompts.append(raw_item['prompt'])
                batch_true_completions.append(raw_item['completion'])
            
            # Tokenize with LEFT padding for generation
            tokenizer.padding_side = 'left'
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                add_special_tokens=False,
                padding=True,
                truncation=False
            )
            tokenizer.padding_side = 'right'  # Reset to right for training
            
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Generate
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode
            generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Print first few examples from first batch
            if batch_idx == 0:
                num_to_show = min(3, len(generated_batch))
                print(f"\n[Validation examples]")
                for ex_idx in range(num_to_show):
                    print(f"  Example {ex_idx + 1}:")
                    print(f"    Prompt: {batch_prompts[ex_idx]}")
                    print(f"    Expected: {batch_true_completions[ex_idx]}")
                    print(f"    Generated: {generated_batch[ex_idx]}")
            
            # Calculate digit matches
            for i, (prompt, true_completion, generated) in enumerate(zip(batch_prompts, batch_true_completions, generated_batch)):
                # Extract the outputs
                # true_completion is like "7838,1234<eos>" - remove <eos>
                true_output = true_completion.replace('<eos>', '').strip()
                # generated is like "loc(c_9123)=7838,1234" - parse normally
                gen_output = parse_random4to4(generated)
                
                all_generated_texts.append(generated)
                all_true_texts.append(prompt + true_completion)
                
                if true_output and gen_output is not None:
                    matches = count_matching_digits(true_output, gen_output)
                    digit_matches.append(matches)
    
    # Calculate metrics (convert numpy types to Python native types)
    if digit_matches:
        return {
            'eval_digit_matches_mean': float(np.mean(digit_matches)),
            'eval_digit_matches_median': float(np.median(digit_matches)),
            'eval_digit_matches_std': float(np.std(digit_matches)),
            'eval_digit_matches_min': int(np.min(digit_matches)),
            'eval_digit_matches_max': int(np.max(digit_matches)),
            'eval_valid_count': int(len(digit_matches)),
            'eval_valid_ratio': float(len(digit_matches) / num_samples)
        }
    else:
        return {
            'eval_digit_matches_mean': 0.0,
            'eval_valid_count': 0,
            'eval_valid_ratio': 0.0
        }


def save_training_plots(trainer, exp_dir, state=None):
    """Save training plots matching original format."""
    # Use provided state or trainer's state
    if state is None:
        state = trainer.state
    
    # Extract metrics from log history
    train_losses = []
    eval_losses = []
    eval_digit_matches = []
    eval_steps = []
    train_steps = []
    
    for entry in state.log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_losses.append(entry['loss'])
            train_steps.append(entry.get('step', len(train_losses)))
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            eval_steps.append(entry.get('step', len(eval_losses)))
        if 'eval_digit_matches_mean' in entry:
            eval_digit_matches.append(entry['eval_digit_matches_mean'])
    
    # Create plots matching original format
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(train_steps, train_losses, label='Train Loss', alpha=0.7)
    if eval_losses:
        plt.plot(eval_steps, eval_losses, label='Eval Loss', marker='o', markersize=3)
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Digit matches plot (analogous to haversine distance)
    plt.subplot(1, 2, 2)
    if eval_digit_matches:
        plt.plot(eval_steps[:len(eval_digit_matches)], eval_digit_matches, 
                label='Avg Digit Matches', color='green', marker='o', markersize=3)
        plt.xlabel('Steps')
        plt.ylabel('Number of Matching Digits')
        plt.title('Digit Matches on Validation Set')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 8.2])
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'training_plots.png', dpi=100)
    plt.close()
    
    # Also save metrics to JSON (convert numpy types to Python types)
    metrics_data = {
        'train_losses': [float(x) for x in train_losses],
        'train_steps': [int(x) if isinstance(x, (np.integer, np.int64)) else x for x in train_steps],
        'eval_losses': [float(x) for x in eval_losses],
        'eval_steps': [int(x) if isinstance(x, (np.integer, np.int64)) else x for x in eval_steps],
        'eval_digit_matches': [float(x) for x in eval_digit_matches],
    }
    
    with open(exp_dir / 'metrics.json', 'w') as f:
        json.dump(metrics_data, f)


def init_weights(module):
    """Initialize weights with custom scale."""
    init_scale = config['model']['init_scale']
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=init_scale)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=init_scale)


def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load HuggingFace tokenizer
    tokenizer_path = config.get('tokenizer_path', 'outputs/tokenizer/wm1_tokenizer')
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Set padding side for training (right padding for training, will switch to left for generation)
    tokenizer.padding_side = 'right'
    print(f"Tokenizer padding side set to: {tokenizer.padding_side} (for training, will use left for generation)")
    
    # Load dataset
    print(f"Loading dataset from {config['dataset']['path']}")
    dataset = load_from_disk(config['dataset']['path'])
    
    # Prepare datasets
    if 'validation' in dataset:
        train_dataset = BaseDataset(
            dataset['train'], 
            tokenizer, 
            config['dataset']['max_sequence_length'],
            split='train',
            loss_mask_type=config['training']['loss_mask_type']
        )
        eval_dataset = BaseDataset(
            dataset['validation'],
            tokenizer,
            config['dataset']['max_sequence_length'],
            split='validation',
            loss_mask_type=config['training']['loss_mask_type']
        )
        print(f"Using train split with {len(dataset['train'])} samples")
        print(f"Using validation split with {len(dataset['validation'])} samples")
    else:
        print("No validation split found, creating one from train data")
        if 'train' in dataset:
            train_data = dataset['train']
        else:
            train_data = dataset
        
        dataset_size = len(train_data)
        eval_size = min(128, dataset_size // 10)
        
        train_dataset = BaseDataset(
            train_data.select(range(eval_size, dataset_size)),
            tokenizer,
            config['dataset']['max_sequence_length'],
            loss_mask_type=config['training']['loss_mask_type']
        )
        eval_dataset = BaseDataset(
            train_data.select(range(eval_size)),
            tokenizer,
            config['dataset']['max_sequence_length'],
            loss_mask_type=config['training']['loss_mask_type']
        )
        print(f"Using {dataset_size - eval_size} samples for training")
        print(f"Using {eval_size} samples for validation")
    
    # Initialize model
    print("Initializing model...")
    model_config = Qwen2Config(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_key_value_heads=config['model']['num_attention_heads'],
        intermediate_size=config['model']['intermediate_size'],
        max_position_embeddings=config['dataset']['max_sequence_length'],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    model = Qwen2ForCausalLM(model_config)
    
    # Apply custom weight initialization
    init_scale = config['model']['init_scale']
    print(f"Applying weight initialization with scale={init_scale}")
    model.apply(init_weights)
    
    print(f"\nTraining configuration:")
    print(f"  Epochs: {config['training']['num_epochs']}")
    print(f"  Batch size: {config['training']['batch_size']}")
    print(f"  Save steps: {config['checkpointing']['save_steps']} {'(fraction)' if config['checkpointing']['save_steps'] < 1 else 'steps'}")
    print(f"  Eval steps: {config['checkpointing']['eval_steps']} {'(fraction)' if config['checkpointing']['eval_steps'] < 1 else 'steps'}")
    print(f"  Warmup steps: {config['training']['warmup_steps']}")
    print(f"  Scheduler: {config['training']['scheduler']}")
    
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
        # Disable defaults that we'll override
        lr_scheduler_type="constant",  # We'll set custom scheduler
        learning_rate=config['training']['learning_rate'],
    )
    
    # Use default data collator - BaseDataset already returns proper format
    data_collator = default_data_collator  # Just batches the tensors
    
    # Custom optimizer and scheduler setup
    def get_optimizer_and_scheduler(trainer):
        """Setup optimizer and scheduler based on config."""
        # Optimizer
        optimizer = torch.optim.AdamW(
            trainer.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay']
        )
        
        # Scheduler - only support linear_with_warmup
        scheduler_type = config['training']['scheduler']
        warmup_steps = config['training']['warmup_steps']
        
        assert scheduler_type == "linear_with_warmup", f"Only 'linear_with_warmup' scheduler is supported, got: {scheduler_type}"
        
        # Get total steps from trainer's computed value
        total_training_steps = trainer.args.max_steps if trainer.args.max_steps > 0 else len(trainer.get_train_dataloader()) * trainer.args.num_train_epochs
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=int(total_training_steps)
        )
        
        return optimizer, scheduler
    
    # Initialize trainer with callback for generation evaluation and plot updates
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[
            GenerationEvalCallback(exp_dir, tokenizer, eval_dataset, device)
        ]
    )
    
    # Override optimizer and scheduler
    trainer.optimizer, trainer.lr_scheduler = get_optimizer_and_scheduler(trainer)
    
    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    trainer.train()
    
    # Save final model
    print("\nSaving final model...")
    trainer.save_model(str(exp_dir / 'final_model'))
    tokenizer.save_pretrained(str(exp_dir / 'final_model'))
    
    # Save final plots
    save_training_plots(trainer, exp_dir)
    
    print(f"\nTraining complete! Results saved to {exp_dir}")
    
    # Print final metrics
    if trainer.state.log_history:
        last_eval = None
        for entry in reversed(trainer.state.log_history):
            if 'eval_loss' in entry:
                last_eval = entry
                break
        
        if last_eval:
            print("\nFinal evaluation metrics:")
            print(f"  Eval loss: {last_eval.get('eval_loss', 'N/A'):.4f}")
            if 'eval_digit_matches_mean' in last_eval:
                print(f"  Avg digit matches: {last_eval['eval_digit_matches_mean']:.2f}/8")
            if 'eval_valid_ratio' in last_eval:
                print(f"  Valid generation ratio: {last_eval['eval_valid_ratio']*100:.1f}%")


if __name__ == "__main__":
    main()