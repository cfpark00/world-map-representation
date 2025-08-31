#!/usr/bin/env python3
import os
import sys
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
from datetime import datetime
import shutil
import argparse
from dotenv import load_dotenv
sys.path.append('.')  # Add root to path
from src.utils import haversine, parse_location, BaseDataset
from datasets import load_from_disk

# Load environment variables from .env file
load_dotenv()

# Parse command line arguments
parser = argparse.ArgumentParser(description='Train location prediction model')
parser.add_argument('config_path', type=str, help='Path to training config YAML file')
parser.add_argument('--overwrite', action='store_true', 
                   help='Overwrite existing experiment directory (only if path starts with EXP_DIR_PREFIX)')
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
config['checkpointing']['save_steps'] = float(config['checkpointing']['save_steps'])
config['checkpointing']['eval_steps'] = float(config['checkpointing']['eval_steps'])
config['dataset']['max_sequence_length'] = int(config['dataset']['max_sequence_length'])
config['model']['vocab_size'] = int(config['model']['vocab_size'])
config['model']['hidden_size'] = int(config['model']['hidden_size'])
config['model']['num_hidden_layers'] = int(config['model']['num_hidden_layers'])
config['model']['num_attention_heads'] = int(config['model']['num_attention_heads'])
config['model']['intermediate_size'] = int(config['model']['intermediate_size'])
config['model']['init_scale'] = float(config['model'].get('init_scale', 0.02))  # Default to 0.02 if not specified

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

# Load HuggingFace tokenizer
tokenizer_path = config.get('tokenizer_path', 'outputs/tokenizer/wm1_tokenizer')
print(f"Loading tokenizer from {tokenizer_path}")
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# CRITICAL: Set padding side for generation
# For generation, we MUST use left padding so the model generates from the actual tokens
# not from padding tokens on the right
tokenizer.padding_side = 'left'
print(f"Tokenizer padding side set to: {tokenizer.padding_side} (critical for generation)")

# Use BaseDataset from utils
LocationDataset = BaseDataset



# Training functions
def evaluate_with_generation(model, eval_dataset, tokenizer, device, num_samples=128, batch_size=16):
    """Evaluate model with generation and distance calculation using batch processing"""
    model.eval()
    
    # Sample a subset for evaluation
    eval_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
    
    distances = []
    all_generated_texts = []
    all_true_texts = []
    
    # Process in batches
    num_batches = (len(eval_indices) + batch_size - 1) // batch_size
    
    with torch.no_grad():
        for batch_idx in range(num_batches):
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(eval_indices))
            batch_indices = eval_indices[batch_start:batch_end]
            
            # Collect prompts and completions for this batch
            batch_prompts = []
            batch_true_completions = []
            
            for idx in batch_indices:
                idx = int(idx)
                raw_item = eval_dataset.dataset[idx]
                batch_prompts.append(raw_item['prompt'])
                batch_true_completions.append(raw_item['completion'])
            
            # Tokenize batch with LEFT padding for generation
            # CRITICAL: Left padding ensures generation starts from actual tokens
            inputs = tokenizer(
                batch_prompts,
                return_tensors='pt',
                add_special_tokens=False,
                padding=True,  # Pad to longest in batch
                truncation=False
            )
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # Generate completions for entire batch
            max_new_tokens = 20  # Enough for coordinates
            outputs = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy decoding
                temperature=1.0,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
            
            # Decode all generated sequences
            generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            
            # Print first few examples from first batch
            if batch_idx == 0:
                num_to_show = min(4, len(generated_batch))
                print(f"\n[First {num_to_show} validation examples from batch]")
                for ex_idx in range(num_to_show):
                    print(f"\n  Example {ex_idx + 1}:")
                    print(f"    Prompt: {batch_prompts[ex_idx]}")
                    print(f"    Expected: {batch_true_completions[ex_idx]}")
                    print(f"    Generated: {generated_batch[ex_idx]}")
                    # Find where the completion starts (after the '=' sign)
                    if '=' in generated_batch[ex_idx]:
                        completion_start = generated_batch[ex_idx].index('=') + 1
                        print(f"    Completion only: {generated_batch[ex_idx][completion_start:]}")
                    else:
                        print(f"    Completion only: [no '=' found in generation]")
                    if ex_idx == 0:  # Show token IDs only for first one
                        print(f"    Token IDs: {outputs[ex_idx].tolist()}")
            
            # Process each generation in the batch
            for i, (prompt, true_completion, generated) in enumerate(zip(batch_prompts, batch_true_completions, generated_batch)):
                # Parse locations
                true_x, true_y = parse_location(true_completion)
                gen_x, gen_y = parse_location(generated)
                
                # Always append texts for debugging
                all_generated_texts.append(generated)
                all_true_texts.append(prompt + true_completion)
                
                if true_x is not None and gen_x is not None:
                    # Calculate distance
                    # Dataset encoding (from create_location_dataset_hf.py):
                    # x = floor(1000 * (longitude_radians + pi))  -> range 0-6283
                    # y = floor(1000 * (latitude_radians + pi/2)) -> range 0-3141
                    # To convert back to degrees:
                    # longitude = (x/1000 - pi) * 180/pi
                    # latitude = (y/1000 - pi/2) * 180/pi
                    
                    import math
                    true_lon = math.degrees(true_x / 1000.0 - math.pi)  # Convert back to degrees
                    true_lat = math.degrees(true_y / 1000.0 - math.pi/2)  # Convert back to degrees
                    gen_lon = math.degrees(gen_x / 1000.0 - math.pi)
                    gen_lat = math.degrees(gen_y / 1000.0 - math.pi/2)
                    
                    dist = haversine(true_lon, true_lat, gen_lon, gen_lat)
                    distances.append(dist)
    
    return distances, all_generated_texts, all_true_texts

def save_training_plots(exp_dir, train_losses, eval_losses, eval_steps_list, 
                        eval_haversine_distances, eval_haversine_steps):
    """Save training plots to summary.png"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, alpha=0.3, label='Train Loss (per batch)')
    if eval_losses:
        plt.plot(eval_steps_list, eval_losses, 'r-', label='Eval Loss', marker='o', markersize=4)
    plt.xlabel('Step')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if eval_haversine_distances:
        plt.plot(eval_haversine_steps, eval_haversine_distances, 'b-', marker='o', markersize=4)
        plt.xlabel('Step')
        plt.ylabel('Average Haversine Distance (km, log scale)')
        plt.yscale('log')
        plt.title('Evaluation Haversine Distance\n(Lower is better, 20000km = parse failed)')
        plt.grid(True, alpha=0.3)
        # Add horizontal line for reference
        plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100km reference')
        plt.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='1000km reference')
        plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='10000km (poor)')
        plt.axhline(y=20000, color='red', linestyle=':', alpha=0.7, label='20000km (parse failed)')
        plt.ylim(bottom=10, top=30000)  # Set reasonable y-axis limits
        plt.legend()
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'summary.png', dpi=150)
    plt.close()

def evaluate(model, eval_loader, device):
    """Standard evaluation for loss"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    
    with torch.no_grad():
        for batch in tqdm(eval_loader, desc="Evaluating", leave=False):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Count non-padding tokens for accurate loss averaging
            non_pad_tokens = (labels != -100).sum().item()
            total_loss += loss.item() * non_pad_tokens
            total_tokens += non_pad_tokens
    
    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    
    return avg_loss

def main():
    # Set seed for reproducibility
    torch.manual_seed(config['training']['seed'])
    np.random.seed(config['training']['seed'])
    
    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"Loading dataset from {config['dataset']['path']}")
    dataset = load_from_disk(config['dataset']['path'])
    
    # Check if dataset has validation split
    if 'validation' in dataset:
        # Use train and validation splits directly
        train_dataset = LocationDataset(
            dataset['train'], 
            tokenizer, 
            config['dataset']['max_sequence_length'],
            split='train',
            loss_mask_type=config['training']['loss_mask_type']
        )
        eval_dataset = LocationDataset(
            dataset['validation'],
            tokenizer,
            config['dataset']['max_sequence_length'],
            split='validation',
            loss_mask_type=config['training']['loss_mask_type']
        )
        print(f"Using train split with {len(dataset['train'])} samples")
        print(f"Using validation split with {len(dataset['validation'])} samples")
    else:
        # No validation split - create one from train
        print("No validation split found, creating one from train data")
        if 'train' in dataset:
            train_data = dataset['train']
        else:
            train_data = dataset
        
        dataset_size = len(train_data)
        eval_size = min(128, dataset_size // 10)  # Use 10% or max 128 for eval
        
        train_dataset = LocationDataset(
            train_data.select(range(eval_size, dataset_size)),
            tokenizer,
            config['dataset']['max_sequence_length'],
            loss_mask_type=config['training']['loss_mask_type']
        )
        eval_dataset = LocationDataset(
            train_data.select(range(eval_size)),
            tokenizer,
            config['dataset']['max_sequence_length'],
            loss_mask_type=config['training']['loss_mask_type']
        )
        print(f"Using {dataset_size - eval_size} samples for training")
        print(f"Using {eval_size} samples for validation")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=config['training']['eval_batch_size'],
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    # Initialize model
    print("Initializing model...")
    model_config = Qwen2Config(
        vocab_size=config['model']['vocab_size'],
        hidden_size=config['model']['hidden_size'],
        num_hidden_layers=config['model']['num_hidden_layers'],
        num_attention_heads=config['model']['num_attention_heads'],
        num_key_value_heads=config['model']['num_attention_heads'],  # Set to same as num_attention_heads for MHA
        intermediate_size=config['model']['intermediate_size'],
        max_position_embeddings=config['dataset']['max_sequence_length'],
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    model = Qwen2ForCausalLM(model_config)
    
    # Apply custom weight initialization if init_scale is specified
    init_scale = config['model']['init_scale']
    if init_scale != 0.02:  # Only print if not using default
        print(f"Applying custom weight initialization with scale={init_scale}")
    
    def init_weights(module):
        """Initialize weights with custom scale."""
        if isinstance(module, nn.Linear):
            # Initialize linear layers with normal distribution
            module.weight.data.normal_(mean=0.0, std=init_scale)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            # Initialize embeddings with normal distribution
            module.weight.data.normal_(mean=0.0, std=init_scale)
    
    # Apply initialization to all modules
    model.apply(init_weights)
    
    model.to(device)
    
    # Initialize optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    total_steps = len(train_loader) * config['training']['num_epochs']
    
    # Convert save/eval steps from ratio to actual steps if < 1
    save_steps = config['checkpointing']['save_steps']
    if save_steps < 1:
        save_steps = int(save_steps * total_steps)
    save_steps = max(1, save_steps)
    
    eval_steps = config['checkpointing']['eval_steps']
    if eval_steps < 1:
        eval_steps = int(eval_steps * total_steps)
    eval_steps = max(1, eval_steps)
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6
    )
    
    # Training loop
    print(f"\nStarting training for {config['training']['num_epochs']} epochs")
    print(f"Total training steps: {total_steps}")
    print(f"Save every {save_steps} steps, evaluate every {eval_steps} steps")
    
    global_step = 0
    train_losses = []
    eval_losses = []
    eval_steps_list = []
    eval_haversine_distances = []  # Track average haversine distances
    eval_haversine_steps = []  # Track when we evaluated haversine
    
    for epoch in range(config['training']['num_epochs']):
        model.train()
        epoch_loss = 0
        epoch_tokens = 0
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Forward pass
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Track loss
            non_pad_tokens = (labels != -100).sum().item()
            epoch_loss += loss.item() * non_pad_tokens
            epoch_tokens += non_pad_tokens
            train_losses.append(loss.item())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{scheduler.get_last_lr()[0]:.2e}"
            })
            
            global_step += 1
            
            # Evaluation
            if global_step % eval_steps == 0 or global_step == total_steps:
                eval_loss = evaluate(model, eval_loader, device)
                eval_losses.append(eval_loss)
                eval_steps_list.append(global_step)
                
                # Evaluate with generation to get haversine distances
                print(f"\n[Step {global_step}] Evaluating with generation...")
                distances, gen_texts, true_texts = evaluate_with_generation(
                    model, eval_dataset, tokenizer, device, num_samples=64
                )
                
                if distances:
                    avg_distance = np.mean(distances)
                    std_distance = np.std(distances)
                    eval_haversine_distances.append(avg_distance)
                    eval_haversine_steps.append(global_step)
                    print(f"[Step {global_step}] Avg Haversine Distance: {avg_distance:.2f} km (±{std_distance:.2f})")
                    print(f"[Step {global_step}] Eval Loss: {eval_loss:.4f}")
                    print(f"[Step {global_step}] Valid generations: {len(distances)}/64")
                    
                    # Print a few examples
                    if len(gen_texts) > 0:
                        print("\nSample generations:")
                        for i in range(min(3, len(gen_texts))):
                            print(f"  True: {true_texts[i]}")
                            print(f"  Gen:  {gen_texts[i]}")
                            if i < len(distances):
                                print(f"  Distance: {distances[i]:.2f} km\n")
                else:
                    # Still track that we tried but got no valid results
                    # Use Earth's maximum distance (~20,000 km) as placeholder for failed parses
                    eval_haversine_distances.append(20000.0)  # Half Earth's circumference
                    eval_haversine_steps.append(global_step)
                    print(f"[Step {global_step}] No valid generations for distance calculation (0/64 parsed)")
                    print(f"[Step {global_step}] Eval Loss: {eval_loss:.4f}")
                    # Show what the model is generating
                    if gen_texts:
                        print(f"[Step {global_step}] Example failed generations:")
                        for i in range(min(2, len(gen_texts))):
                            print(f"  True: {true_texts[i] if true_texts else 'N/A'}")
                            print(f"  Generated: {gen_texts[i] if gen_texts else 'N/A'}")
                
                # Save updated plots after each evaluation
                save_training_plots(exp_dir, train_losses, eval_losses, eval_steps_list,
                                  eval_haversine_distances, eval_haversine_steps)
                
                model.train()
            
            # Checkpointing
            if config['checkpointing']['save_strategy'] == 'steps' and global_step % save_steps == 0:
                checkpoint_dir = exp_dir / 'checkpoints' / f'step_{global_step}'
                checkpoint_dir.mkdir(exist_ok=True)
                
                model.save_pretrained(checkpoint_dir)
                tokenizer.save_pretrained(checkpoint_dir)
                
                # Save optimizer and scheduler states
                torch.save({
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'global_step': global_step,
                    'epoch': epoch,
                }, checkpoint_dir / 'training_state.pt')
                
                print(f"Saved checkpoint at step {global_step}")
        
        # End of epoch
        avg_epoch_loss = epoch_loss / epoch_tokens if epoch_tokens > 0 else float('inf')
        print(f"\nEpoch {epoch+1} completed. Average loss: {avg_epoch_loss:.4f}")
        
        # Save epoch checkpoint
        if config['checkpointing']['save_strategy'] == 'epoch':
            checkpoint_dir = exp_dir / 'checkpoints' / f'epoch_{epoch+1}'
            checkpoint_dir.mkdir(exist_ok=True)
            
            model.save_pretrained(checkpoint_dir)
            tokenizer.save_pretrained(checkpoint_dir)
            
            torch.save({
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'global_step': global_step,
                'epoch': epoch,
            }, checkpoint_dir / 'training_state.pt')
    
    # Final evaluation
    print("\nRunning final evaluation...")
    final_eval_loss = evaluate(model, eval_loader, device)
    print(f"Final Eval Loss: {final_eval_loss:.4f}")
    
    # Final generation-based evaluation
    print("\nRunning final generation-based evaluation...")
    final_distances, final_gen_texts, final_true_texts = evaluate_with_generation(
        model, eval_dataset, tokenizer, device, num_samples=128
    )
    
    if final_distances:
        final_avg_distance = np.mean(final_distances)
        final_std_distance = np.std(final_distances)
        print(f"Final Avg Haversine Distance: {final_avg_distance:.2f} km (±{final_std_distance:.2f})")
        print(f"Min distance: {np.min(final_distances):.2f} km")
        print(f"Max distance: {np.max(final_distances):.2f} km")
        print(f"Median distance: {np.median(final_distances):.2f} km")
    
    # Save final model
    final_checkpoint_dir = exp_dir / 'checkpoints' / 'final'
    final_checkpoint_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_checkpoint_dir)
    tokenizer.save_pretrained(final_checkpoint_dir)
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'eval_losses': eval_losses,
        'eval_steps': eval_steps_list,
        'eval_haversine_distances': eval_haversine_distances,
        'eval_haversine_steps': eval_haversine_steps,
        'final_eval_loss': final_eval_loss,
        'final_haversine_distance': final_avg_distance if final_distances else None,
        'final_haversine_std': final_std_distance if final_distances else None,
        'config': config
    }
    
    with open(exp_dir / 'training_history.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    # Save final training plots
    save_training_plots(exp_dir, train_losses, eval_losses, eval_steps_list,
                       eval_haversine_distances, eval_haversine_steps)
    print(f"Final training summary plot saved to {exp_dir / 'summary.png'}")
    
    # Also create a histogram of final distances if available
    if final_distances:
        plt.figure(figsize=(8, 6))
        plt.hist(final_distances, bins=30, edgecolor='black', alpha=0.7)
        plt.xlabel('Haversine Distance (km)')
        plt.ylabel('Count')
        plt.title(f'Final Generation Distance Distribution\nMean: {final_avg_distance:.2f} km, Std: {final_std_distance:.2f} km')
        plt.axvline(x=final_avg_distance, color='r', linestyle='--', label=f'Mean: {final_avg_distance:.2f} km')
        plt.axvline(x=np.median(final_distances), color='g', linestyle='--', label=f'Median: {np.median(final_distances):.2f} km')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(exp_dir / 'final_distance_distribution.png', dpi=150)
        plt.close()
    
    print(f"\nTraining completed! Results saved to {exp_dir}")
    print(f"Final model saved to {final_checkpoint_dir}")

if __name__ == "__main__":
    main()