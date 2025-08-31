#!/usr/bin/env python3
import numpy as np
from datasets import Dataset, DatasetDict
import os
import argparse

# Parse command line arguments
parser = argparse.ArgumentParser(description='Create random 4-to-4 digit mapping dataset')
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

print(f"Creating random 4-to-4 digit mapping dataset")
print(f"n_train: {n_train}, n_val: {n_val}, seed: {seed}")

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

# For each input, generate TWO random 4-digit outputs (like longitude,latitude)
train_outputs1 = np.random.randint(0, 10000, size=n_train)
train_outputs2 = np.random.randint(0, 10000, size=n_train)

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
print("Creating train dataset...")
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

# Show sample rows
print("\nSample train rows:")
for i in range(min(10, len(train_dataset))):
    sample = train_dataset[i]
    print(f"  {sample['text']} | prompt: {sample['prompt']} | completion: {sample['completion']}")

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