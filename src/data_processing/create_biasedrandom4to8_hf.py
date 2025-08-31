#!/usr/bin/env python3
import numpy as np
from datasets import Dataset, DatasetDict
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create biased random 4-to-8 digit mapping dataset where 50% have duplicate outputs')
parser.add_argument('n_train', type=int, help='Number of training samples')
parser.add_argument('n_val', type=int, help='Number of validation samples (subset of train)')
parser.add_argument('output_dir', type=str, help='Output directory for the dataset')
parser.add_argument('--seed', type=int, default=42, help='Random seed (default: 42)')

args = parser.parse_args()

n_train = args.n_train
n_val = args.n_val
output_dir = args.output_dir
seed = args.seed

if n_val > n_train:
    raise ValueError(f"n_val ({n_val}) cannot be larger than n_train ({n_train})")

print(f"Creating BIASED random 4-to-8 digit mapping dataset")
print(f"n_train: {n_train}, n_val: {n_val}, seed: {seed}")
print(f"50% of samples will share the same output with different inputs!")

# Set random seed
np.random.seed(seed)

# Generate unique random 4-digit inputs (left side)
# We need n_train unique 4-digit numbers
print(f"\nGenerating {n_train} unique random mappings...")

# Generate unique 4-digit inputs (from 0000 to 9999)
all_possible_inputs = np.arange(10000)
if n_train > 10000:
    raise ValueError(f"n_train ({n_train}) cannot be larger than 10000 (max unique 4-digit numbers)")

# Sample n_train unique inputs
train_inputs = np.random.choice(all_possible_inputs, size=n_train, replace=False)

# Generate outputs with 50% bias
# First half: unique random outputs
n_unique = n_train // 2
train_outputs1 = np.zeros(n_train, dtype=int)
train_outputs2 = np.zeros(n_train, dtype=int)

# First half gets unique random outputs
train_outputs1[:n_unique] = np.random.randint(0, 10000, size=n_unique)
train_outputs2[:n_unique] = np.random.randint(0, 10000, size=n_unique)

# Second half: pick random outputs from the first half (creates duplicates)
# This ensures different inputs map to the same outputs
for i in range(n_unique, n_train):
    # Pick a random index from the first half
    source_idx = np.random.randint(0, n_unique)
    # Copy its output
    train_outputs1[i] = train_outputs1[source_idx]
    train_outputs2[i] = train_outputs2[source_idx]

# Shuffle everything together so biased samples are mixed in
shuffle_indices = np.random.permutation(n_train)
train_inputs = train_inputs[shuffle_indices]
train_outputs1 = train_outputs1[shuffle_indices]
train_outputs2 = train_outputs2[shuffle_indices]

print(f"\nCreated mappings:")
print(f"  - {n_unique} unique outputs")
print(f"  - {n_train - n_unique} duplicate outputs (different inputs, same output)")

# Count actual unique outputs to verify
unique_outputs = set()
for o1, o2 in zip(train_outputs1, train_outputs2):
    unique_outputs.add((o1, o2))
print(f"  - Actual unique output pairs: {len(unique_outputs)}")

# Create the dataset
def create_dataset_dict(inputs, outputs1, outputs2):
    """Create a dictionary suitable for HuggingFace Dataset"""
    text_list = []
    prompt_list = []
    completion_list = []
    
    for inp, out1, out2 in zip(inputs, outputs1, outputs2):
        # Format as 4-digit strings with zero padding
        inp_str = f"{inp:04d}"
        out1_str = f"{out1:04d}"
        out2_str = f"{out2:04d}"
        
        # Create text format: loc(c_abcd)=efgh,ijkl (exactly like location format)
        full_text = f"loc(c_{inp_str})={out1_str},{out2_str}"
        prompt = f"<bos>loc(c_{inp_str})="
        completion = f"{out1_str},{out2_str}<eos>"
        
        text_list.append(full_text)
        prompt_list.append(prompt)
        completion_list.append(completion)
    
    return {
        'text': text_list,
        'prompt': prompt_list,
        'completion': completion_list
    }

# Create train dataset
print("\nCreating train dataset...")
train_data = create_dataset_dict(train_inputs, train_outputs1, train_outputs2)
train_dataset = Dataset.from_dict(train_data)

# Create validation dataset as a subset of train
print(f"Creating validation dataset (subset of train)...")
# Select n_val random indices from the training set
val_indices = np.random.choice(n_train, size=n_val, replace=False)
val_inputs = train_inputs[val_indices]
val_outputs1 = train_outputs1[val_indices]
val_outputs2 = train_outputs2[val_indices]

val_data = create_dataset_dict(val_inputs, val_outputs1, val_outputs2)
val_dataset = Dataset.from_dict(val_data)

# Create DatasetDict
dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': val_dataset
})

# Create output directory if needed
os.makedirs(output_dir, exist_ok=True)

# Save in HuggingFace format
print(f"\nSaving HF dataset to {output_dir}...")
dataset_dict.save_to_disk(output_dir)

print(f"\nDataset created successfully!")
print(f"Train size: {len(train_dataset):,}")
print(f"Validation size: {len(val_dataset):,}")

# Print dataset info
print("\nDataset structure:")
print(dataset_dict)
print("\nFeatures:")
print(train_dataset.features)

# Show sample rows with duplicate detection
print("\nSample train rows (checking for duplicates):")
output_counts = {}
for i in range(min(20, len(train_dataset))):
    sample = train_dataset[i]
    text = sample['text']
    # Extract output
    output = text.split('=')[1] if '=' in text else ''
    
    if output not in output_counts:
        output_counts[output] = []
    output_counts[output].append(text.split('=')[0] if '=' in text else '')
    
    if i < 10:
        print(f"  {text}")

print("\nDuplicate outputs in first 20 samples:")
for output, inputs in output_counts.items():
    if len(inputs) > 1:
        print(f"  Output {output} appears {len(inputs)} times with inputs: {inputs}")

print("\nSample validation rows (these are from the training set!):")
for i in range(min(5, len(val_dataset))):
    sample = val_dataset[i]
    print(f"  {sample['text']}")

# Check overlap to confirm
val_texts = set(val_dataset['text'])
train_texts = set(train_dataset['text'])
overlap = val_texts.intersection(train_texts)
print(f"\nValidation-Train overlap: {len(overlap)}/{len(val_texts)} samples")
print("(Should be 100% since validation is a subset of train)")

print("\nTo load this dataset:")
print(">>> from datasets import load_from_disk")
print(f">>> dataset = load_from_disk('{output_dir}')")
print(">>> train_data = dataset['train']")
print(">>> val_data = dataset['validation']")