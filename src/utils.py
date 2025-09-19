#!/usr/bin/env python3
"""Utility functions for research projects."""

import os
import shutil
import sys
import random
from pathlib import Path
from typing import Union
from dotenv import load_dotenv


def init_directory(directory: Union[str, Path], overwrite: bool = False) -> Path:
    """
    Initialize a directory with safety checks for overwriting.
    
    This is a generic tool for safely creating/overwriting directories. It uses the
    DATA_DIR environment variable to specify a safe prefix - only directories 
    under this prefix can be overwritten. This prevents accidental deletion of 
    important system directories.
    
    Args:
        directory: Path to directory (str or Path object)
        overwrite: Whether to overwrite existing directory
    
    Returns:
        Path object of the created directory
    
    Raises:
        SystemExit: If directory exists without overwrite, or safety checks fail
    """
    load_dotenv()
    
    directory = Path(directory)
    
    if directory.exists():
        if overwrite:
            # Get DATA_DIR from environment (loaded from .env)
            safe_prefix = os.environ.get('DATA_DIR')
            
            if not safe_prefix:
                print(f"Error: DATA_DIR not set in .env!")
                print(f"Cannot use --overwrite without DATA_DIR for safety.")
                print("Set DATA_DIR in .env file to specify where overwriting is allowed.")
                sys.exit(1)
            
            # Convert safe_prefix to absolute path for comparison
            safe_prefix = Path(safe_prefix).resolve()
            
            # Get absolute path of directory
            dir_absolute = directory.resolve()
            
            # Check if the absolute path starts with safe prefix
            if not str(dir_absolute).startswith(str(safe_prefix)):
                print(f"Error: Cannot overwrite {dir_absolute}")
                print(f"Directory must start with DATA_DIR: {safe_prefix}")
                print("This safety check prevents accidental deletion of important directories.")
                sys.exit(1)
            
            # Safe to remove
            print(f"Removing existing directory: {dir_absolute}")
            shutil.rmtree(dir_absolute)
            print("Directory removed successfully.")
        else:
            print(f"Error: Directory {directory} already exists!")
            print("Use --overwrite to remove it, or choose a different path.")
            sys.exit(1)
    
    # Create directory
    directory.mkdir(parents=True, exist_ok=False)
    print(f"Created directory: {directory.resolve()}")
    return directory


def seed_all(seed: int):
    """
    Set seeds for all random number generators to ensure reproducibility.

    Args:
        seed: Random seed value
    """
    # Python built-in random module
    random.seed(seed)

    # NumPy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)

    # CUDA (all GPUs)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Make CuDNN deterministic (may impact performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set environment variable for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)

    print(f"Set all random seeds to {seed}")


# ============================================================================
# Other reusable utilities for the research
# ============================================================================
# Add stateless utility functions below that are expected to be used
# repetitively throughout the research project


#!/usr/bin/env python3
"""Utility functions for the WM_1 project."""

import os
import re
import math
import shutil
import sys
import json
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_from_disk
from scipy.spatial import cKDTree
from transformers import TrainerCallback, AutoTokenizer, DataCollatorForLanguageModeling
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

def euclidean_distance(x1, y1, x2, y2):
    """
    Calculate Euclidean distance between points in 2D space.
    
    Args:
        x1, y1: Coordinates of first point(s)
        x2, y2: Coordinates of second point(s)
        Can be scalars or arrays of same shape
    
    Returns:
        Distance(s) as float
    """
    # Convert to numpy arrays to handle both scalar and vector inputs
    x1 = np.asarray(x1)
    y1 = np.asarray(y1)
    x2 = np.asarray(x2)
    y2 = np.asarray(y2)
    
    # Simple Euclidean distance
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    
    # Return scalar if inputs were scalar
    if x1.shape == ():
        return float(distance)
    return distance




def extract_coordinates(df, coord_column='Coordinates'):
    """
    Extract x and y from a coordinate string column.
    
    Args:
        df: DataFrame with coordinate column
        coord_column: Name of column containing coordinates as "y,x"
    
    Returns:
        DataFrame with added 'x' and 'y' columns
    """
    coords = df[coord_column].str.split(',', expand=True)
    df['y'] = coords[0].astype(float)
    df['x'] = coords[1].astype(float)
    return df



def parse_distance(text):
    """
    Parse distance from text like 'dist(c_1234,c_5678)=901' or just '901'.
    Handles both compact and space-delimited formats.
    
    Args:
        text: String containing distance information
    
    Returns:
        Distance value as int or None if not found
    """
    # Remove all spaces to handle space-delimited tokenizer output
    text = text.replace(' ', '')
    
    # Try to find the pattern after =
    match = re.search(r'=(\d+)', text)
    if match:
        return int(match.group(1))
    # Try just a number
    match = re.search(r'^(\d+)', text)
    if match:
        return int(match.group(1))
    return None


def parse_walk_transitions(text):
    """
    Parse transitions from random walk text like 'rw(200,5)=c_123,c_456,c_789'.
    Returns list of (city1_id, city2_id) tuples for each transition.
    Handles both compact and space-delimited formats.
    
    Args:
        text: String containing walk information
    
    Returns:
        List of (city1_id, city2_id) tuples, or empty list if none found
    """
    # Remove all spaces to handle space-delimited tokenizer output
    text = text.replace(' ', '')
    
    # Find everything after =
    match = re.search(r'=(.+)', text)
    if not match:
        return []
    
    sequence = match.group(1)
    
    # Find all complete city IDs (c_\d+)
    city_matches = list(re.finditer(r'c_(\d+)', sequence))
    
    if len(city_matches) < 2:
        return []  # Need at least 2 cities for a transition
    
    # Extract transitions
    transitions = []
    for i in range(len(city_matches) - 1):
        try:
            city1_id = int(city_matches[i].group(1))
            city2_id = int(city_matches[i + 1].group(1))
            transitions.append((city1_id, city2_id))
        except (ValueError, AttributeError):
            continue
    
    return transitions


def validate_transitions(transitions, cities_df, distance_threshold_km):
    """
    Validate transitions in a walk based on distance constraints.

    Args:
        transitions: List of (city1_id, city2_id) tuples
        cities_df: DataFrame with city data (must have 'city_id', 'x', 'y')
        distance_threshold_km: Maximum distance between consecutive cities

    Returns:
        Tuple of (num_valid_transitions, total_transitions)
    """
    if not transitions:
        return 0, 0

    valid_transitions = 0
    total_transitions = len(transitions)

    for city1_id, city2_id in transitions:
        # Find cities in dataframe using city_id column
        try:
            city1 = cities_df[cities_df['city_id'] == city1_id].iloc[0]
            city2 = cities_df[cities_df['city_id'] == city2_id].iloc[0]
        except (IndexError, KeyError):
            # City ID not found in dataset - transition is invalid
            continue

        # Calculate distance
        distance = euclidean_distance(
            np.array([city1['x']]),
            np.array([city1['y']]),
            np.array([city2['x']]),
            np.array([city2['y']])
        ).item()

        if distance <= distance_threshold_km:
            valid_transitions += 1

    return valid_transitions, total_transitions


class BaseDataset(Dataset):
    """
    Base dataset class that only handles text data.
    Loss masking is now handled by task-specific collators.
    """
    
    def __init__(self, dataset_or_path, split='train'):
        """
        Initialize dataset.
        
        Args:
            dataset_or_path: Path to HuggingFace dataset or dataset object
            split: Dataset split to use ('train', 'val', 'test')
        """
        if isinstance(dataset_or_path, str):
            full_dataset = load_from_disk(dataset_or_path)
            # Handle single split dataset
            if 'train' in full_dataset:
                self.dataset = full_dataset[split] if split in full_dataset else full_dataset['train']
            else:
                # If it's just a single dataset without splits, use it directly
                self.dataset = full_dataset
        else:
            self.dataset = dataset_or_path
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]


class DistanceCollator(DataCollatorForLanguageModeling):
    """
    Collator for distance tasks.
    Text format: <bos>dist(c_X,c_Y)=DISTANCE<eos>
    Only computes loss on the distance value (after '=').
    """
    def __init__(self, tokenizer, max_length=32, mlm=False):
        super().__init__(tokenizer, mlm=mlm)
        self.max_length = max_length
    
    def __call__(self, examples):
        # Extract text from examples
        texts = [ex['text'] for ex in examples]
        
        # Tokenize all texts
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Create labels
        labels = batch['input_ids'].clone()
        
        # Mask everything before '=' for loss computation
        for i, text in enumerate(texts):
            # Find the position of '=' in the original text
            eq_pos = text.find('=')
            if eq_pos != -1:
                # Tokenize up to '=' to find where answer starts
                prompt_text = text[:eq_pos+1]  # Include '='
                prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
                prompt_len = len(prompt_tokens)
                
                # Mask prompt tokens in labels
                for j in range(min(prompt_len, self.max_length)):
                    labels[i, j] = -100
        
        # Mask padding tokens
        labels[batch['attention_mask'] == 0] = -100
        
        batch['labels'] = labels
        return batch




class RandomWalkCollator(DataCollatorForLanguageModeling):
    """
    Collator for random walk tasks.
    Text format: <bos> r w ( max , len ) = c _ X X X X , c _ Y Y Y Y , ... <eos>
    Computes loss on the city sequence (everything after '=').
    """
    def __init__(self, tokenizer, max_length=256, mlm=False):
        super().__init__(tokenizer, mlm=mlm)
        self.max_length = max_length

    def __call__(self, examples):
        # Extract text from examples
        texts = [ex['text'] for ex in examples]

        # Tokenize all texts
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )

        # Create labels
        labels = batch['input_ids'].clone()

        # Mask everything before and including '=' for loss computation
        for i, text in enumerate(texts):
            # Find the position of '=' in the original text
            eq_pos = text.find('=')
            if eq_pos == -1:
                # This should never happen with properly formatted data
                raise ValueError(f"FATAL: Cannot find '=' in random walk text: {text[:100]}...")

            # Tokenize up to and including '=' to find where city sequence starts
            prompt_text = text[:eq_pos+1]  # Include '='
            prompt_tokens = self.tokenizer(prompt_text, add_special_tokens=False)['input_ids']
            prompt_len = len(prompt_tokens)

            # Mask prompt tokens in labels (everything before and including '=')
            for j in range(min(prompt_len, self.max_length)):
                labels[i, j] = -100

        # Mask padding tokens
        labels[batch['attention_mask'] == 0] = -100

        batch['labels'] = labels
        return batch


class FullSequenceCollator(DataCollatorForLanguageModeling):
    """
    Default collator that computes loss on the entire sequence.
    Used as fallback or for tasks that need full sequence loss.
    """
    def __init__(self, tokenizer, max_length=32, mlm=False):
        super().__init__(tokenizer, mlm=mlm)
        self.max_length = max_length
    
    def __call__(self, examples):
        # Extract text from examples
        texts = [ex['text'] for ex in examples]
        
        # Tokenize all texts
        batch = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Labels are just input_ids with padding masked
        labels = batch['input_ids'].clone()
        labels[batch['attention_mask'] == 0] = -100
        
        batch['labels'] = labels
        return batch


class MultiTaskCollator:
    """
    Collator that handles mixed task types in a single batch.
    Each example must have 'text' and 'task_type' fields.
    Optionally uses 'loss_mask' field if present and use_loss_mask is True.
    """
    def __init__(self, tokenizer, max_length=32, use_loss_mask=False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_loss_mask = use_loss_mask
        # Initialize task-specific collators
        self.distance_collator = DistanceCollator(tokenizer, max_length)
        self.randomwalk_collator = RandomWalkCollator(tokenizer, max(max_length, 256))
        self.full_collator = FullSequenceCollator(tokenizer, max_length)
    
    def __call__(self, examples):
        # Check if we should use loss_mask from dataset
        if self.use_loss_mask and 'loss_mask' in examples[0]:
            # USE DATASET'S LOSS MASK - only compute loss where mask is '1'
            texts = [ex['text'] for ex in examples]
            encoding = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # Apply loss masks to create labels
            labels = encoding['input_ids'].clone()

            for i, ex in enumerate(examples):
                if 'loss_mask' in ex:
                    mask_str = ex['loss_mask']
                    # Apply mask: set labels to -100 where mask is '0'
                    for j, mask_char in enumerate(mask_str):
                        if j < labels[i].size(0):  # Ensure we don't go out of bounds
                            if mask_char == '0':
                                labels[i, j] = -100
                    # Also mask padding tokens
                    labels[i, encoding['attention_mask'][i] == 0] = -100

            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': labels
            }
        else:
            # DEFAULT: FULL SEQUENCE LOSS (standard next token prediction on ALL tokens)
            texts = [ex['text'] for ex in examples]
            encoding = self.tokenizer(
                texts,
                return_tensors='pt',
                padding=True,
                truncation=True,
                max_length=self.max_length
            )

            # Simple next token prediction: labels = input_ids
            labels = encoding['input_ids'].clone()

            # Only mask padding tokens (standard practice)
            labels[encoding['attention_mask'] == 0] = -100

            return {
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': labels
            }


def get_collator(config, tokenizer):
    """
    Get the multi-task collator that handles all task types.
    Task type comes from each dataset example, not from config.
    
    Args:
        config: Configuration dictionary
        tokenizer: Tokenizer to use
    
    Returns:
        MultiTaskCollator instance
    """
    max_length = config['dataset']['max_sequence_length']
    # Check if training config specifies to use loss_mask
    use_loss_mask = config.get('training', {}).get('use_loss_mask', False)
    return MultiTaskCollator(tokenizer, max_length, use_loss_mask)


def convert_numpy_to_python(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_python(item) for item in obj]
    return obj


def init_weights(module, init_scale=0.02):
    """Initialize weights with custom scale."""
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=init_scale)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=init_scale)


def evaluate_with_generation(model, eval_dataset, tokenizer, device, num_samples=128, batch_size=16, config=None):
    """Evaluate model with generation and task-specific metric calculation. Automatically detects task types from dataset."""
    # Import metrics module
    from src.metrics import calculate_metric, get_failure_value, format_metric_for_display

    model.eval()
    
    # Sample a subset for evaluation
    eval_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
    
    # Group samples by task type
    task_samples = {}
    for idx in eval_indices:
        idx = int(idx)
        # Handle both BaseDataset (has .dataset) and raw dataset
        if hasattr(eval_dataset, 'dataset'):
            raw_item = eval_dataset.dataset[idx]
        else:
            raw_item = eval_dataset[idx]
        
        # FAIL FAST: Require task_type field
        if 'task_type' not in raw_item:
            raise ValueError(f"FATAL: Dataset item at index {idx} missing required 'task_type' field: {raw_item}")

        task_type = raw_item['task_type']
        if task_type not in task_samples:
            task_samples[task_type] = []
        task_samples[task_type].append((idx, raw_item))
    
    print(f"Task distribution in eval batch: {[(t, len(samples)) for t, samples in task_samples.items()]}")
    
    # Load cities data for tasks that need it
    cities_df = None
    tasks_needing_cities = {'randomwalk', 'randring', 'center'}
    tasks_that_need_cities = tasks_needing_cities.intersection(task_samples.keys())

    if tasks_that_need_cities:
        # Check for cities_csv in eval.<task>.cities_csv structure
        cities_csv = None

        # Try to find cities_csv in any of the tasks that need it
        for task in tasks_that_need_cities:
            if config and 'eval' in config and task in config['eval'] and 'cities_csv' in config['eval'][task]:
                cities_csv = config['eval'][task]['cities_csv']
                break

        if not cities_csv:
            # Require explicit config - no hardcoded default
            raise ValueError(
                f"FATAL: Tasks {tasks_that_need_cities} require cities CSV path. Add to config:\n"
                "eval:\n"
                f"  {list(tasks_that_need_cities)[0]}:\n"
                "    cities_csv: 'data/datasets/cities/cities.csv'"
            )

        print(f"Loading cities data for {tasks_that_need_cities} evaluation from: {cities_csv}")

        # Check if file exists
        import os
        if not os.path.exists(cities_csv):
            raise ValueError(f"FATAL: Cities CSV not found at {cities_csv}. Tasks {tasks_that_need_cities} need city coordinates for validation.")

        cities_df = pd.read_csv(cities_csv)
    
    # Results storage
    all_task_metrics = {}
    
    # Process each task type separately
    for task_type, samples in task_samples.items():
        if not samples:
            continue
            
        print(f"\nEvaluating {len(samples)} {task_type} samples...")
        
        # Extract data for this task
        task_indices = [idx for idx, _ in samples]
        task_items = [item for _, item in samples]
        
        task_metrics = []
        
        # Process in batches
        num_batches = (len(samples) + batch_size - 1) // batch_size
        
        with torch.no_grad():
            for batch_idx in range(num_batches):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(samples))
                batch_items = task_items[batch_start:batch_end]
                
                # Get prompts and true completions
                batch_prompts = []
                batch_true_completions = []
                
                for raw_item in batch_items:
                    # Handle different dataset formats
                    if 'prompt' in raw_item and 'completion' in raw_item:
                        # Old format with separate prompt/completion
                        prompt = raw_item['prompt']
                        completion = raw_item['completion']
                    elif 'text' in raw_item:
                        # New format with combined text - split into prompt/completion
                        text = raw_item['text']
                        if task_type == 'distance':
                            # Split at '=' for distance tasks: <bos>dist(c_123,c_456)=789<eos>
                            eq_pos = text.find('=')
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in distance task text: {text[:100]}...")
                            prompt = text[:eq_pos+1]  # Include '='
                            completion = text[eq_pos+1:]  # Everything after '='
                        elif task_type == 'randomwalk':
                            # Split at '=' for randomwalk tasks: <bos>rw(max,len)=city_list<eos>
                            eq_pos = text.find('=')
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in randomwalk task text: {text[:100]}...")

                            # For evaluation: include first city in prompt to make task harder
                            # This prevents model from choosing an easy starting city
                            full_completion = text[eq_pos+1:]  # Everything after '='

                            # Find the first comma (end of first city)
                            # Format is like: c _ 1 2 3 4 , c _ 5 6 7 8 , ...
                            first_comma = full_completion.find(',')
                            if first_comma != -1:
                                # Include first city and comma in prompt
                                prompt = text[:eq_pos+1] + full_completion[:first_comma+1]  # Include '=' and first city with comma
                                completion = full_completion[first_comma+1:]  # Rest of the cities
                            else:
                                # Only one city or malformed - fall back to original split
                                prompt = text[:eq_pos+1]  # Include '='
                                completion = full_completion
                        elif task_type in ['trianglearea', 'angle', 'perimeter', 'nearest_neighbor']:
                            # Split at '=' for simple format tasks
                            eq_pos = text.find('=')
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in {task_type} task text: {text[:100]}...")
                            prompt = text[:eq_pos+1]  # Include '='
                            completion = text[eq_pos+1:]  # Everything after '='
                        elif task_type in ['compass', 'crossing', 'inside']:
                            # Split at '=' for boolean/direction tasks
                            eq_pos = text.find('=')
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in {task_type} task text: {text[:100]}...")
                            prompt = text[:eq_pos+1]  # Include '='
                            completion = text[eq_pos+1:]  # Everything after '='
                        elif task_type in ['center', 'circlecount', 'randring']:
                            # These have multiple '=' signs, split at the LAST one
                            # center: center(cities;in=TRUE)=result
                            # circlecount: circlecount(c_123,r=456)=789
                            # randring: randring(c_123,r=50,R=200,n=4)=cities
                            eq_pos = text.rfind('=')  # Find LAST '='
                            if eq_pos == -1:
                                raise ValueError(f"FATAL: Cannot find '=' in {task_type} task text: {text[:100]}...")
                            prompt = text[:eq_pos+1]  # Include '='
                            completion = text[eq_pos+1:]  # Everything after last '='
                        else:
                            # Unknown task type - fail immediately
                            raise ValueError(f"FATAL: Unknown task type '{task_type}' - add explicit handling for this task type")
                    else:
                        raise ValueError(f"Dataset item missing both 'prompt'/'completion' and 'text' fields: {raw_item}")
                    
                    # For random walk, don't modify prompt - let model generate from scratch
                    batch_prompts.append(prompt)
                        
                    batch_true_completions.append(completion)
                
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
                
                # Generate with task-appropriate max tokens
                if task_type == 'randomwalk':
                    max_new_tokens = 224  # Allow for long walks with up to 20 cities with padding
                elif task_type in ['trianglearea', 'angle']:
                    max_new_tokens = 30   # Area/angle can be large numbers
                else:
                    max_new_tokens = 20   # Default for other tasks
                
                outputs = model.generate(
                    input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )
                
                # Decode
                generated_batch = tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Print first few examples from first batch for each task
                if batch_idx == 0:
                    num_to_show = min(2, len(generated_batch))
                    print(f"[{task_type.upper()} examples]")
                    for ex_idx in range(num_to_show):
                        print(f"  Example {ex_idx + 1}:")
                        print(f"    Prompt: {batch_prompts[ex_idx]}")
                        print(f"    Expected: {batch_true_completions[ex_idx]}")
                        print(f"    Generated: {generated_batch[ex_idx]}")

                        # Calculate metric using centralized system
                        metric_kwargs = {}
                        if task_type in ['randomwalk', 'randring', 'center'] and cities_df is not None:
                            metric_kwargs['cities_df'] = cities_df

                        metric_value = calculate_metric(
                            task_type,
                            batch_prompts[ex_idx],
                            batch_true_completions[ex_idx],
                            generated_batch[ex_idx],
                            **metric_kwargs
                        )

                        # Format based on task type
                        formatted_value = format_metric_for_display(task_type, metric_value)

                        # Special formatting for display
                        if task_type in ['distance', 'center']:
                            print(f"    Metric: {formatted_value}")
                        elif task_type in ['randomwalk', 'randring']:
                            print(f"    Metric (combined score): {formatted_value}")
                        elif task_type == 'nearest_neighbor':
                            print(f"    Metric (Jaccard): {formatted_value}")
                        elif task_type in ['compass', 'crossing', 'inside']:
                            print(f"    Metric (accuracy): {formatted_value}")
                        else:
                            print(f"    Metric: {formatted_value}")
                
                # Calculate task-specific metrics using centralized system
                for prompt, true_completion, generated in zip(batch_prompts, batch_true_completions, generated_batch):
                    # Prepare kwargs based on task requirements
                    metric_kwargs = {}
                    if task_type in ['randomwalk', 'randring', 'center'] and cities_df is not None:
                        metric_kwargs['cities_df'] = cities_df

                    # Calculate metric using centralized system
                    try:
                        metric_value = calculate_metric(
                            task_type,
                            prompt,
                            true_completion,
                            generated,
                            **metric_kwargs
                        )
                        task_metrics.append(metric_value)
                    except ValueError as e:
                        # Unknown task type - this maintains the original fail-fast behavior
                        if "No metric implementation" in str(e):
                            raise ValueError(f"FATAL: No metric implementation for task type '{task_type}'. Implement the metric calculation for this task type.")
                        else:
                            raise

        # Calculate metrics for this task
        if task_metrics:
            all_task_metrics[task_type] = {
                f'eval_{task_type}_metric_mean': np.mean(task_metrics),
                f'eval_{task_type}_metric_median': np.median(task_metrics),
                f'eval_{task_type}_metric_std': np.std(task_metrics),
                f'eval_{task_type}_metric_min': np.min(task_metrics),
                f'eval_{task_type}_metric_max': np.max(task_metrics),
                f'eval_{task_type}_valid_count': len(task_metrics),
                f'eval_{task_type}_valid_ratio': len(task_metrics) / len(samples)
            }
        else:
            # Use centralized failure value from metrics module
            fail_value = get_failure_value(task_type)

            all_task_metrics[task_type] = {
                f'eval_{task_type}_metric_mean': fail_value,
                f'eval_{task_type}_metric_median': fail_value,
                f'eval_{task_type}_metric_std': 0.0,
                f'eval_{task_type}_metric_min': fail_value,
                f'eval_{task_type}_metric_max': fail_value,
                f'eval_{task_type}_valid_count': 0,
                f'eval_{task_type}_valid_ratio': 0.0
            }
    
    # Flatten all task-specific metrics into single dict
    flat_metrics = {}
    for task_type, task_metrics_dict in all_task_metrics.items():
        flat_metrics.update(task_metrics_dict)
    
    # For backward compatibility, if single task, also add legacy metric names
    if len(all_task_metrics) == 1:
        single_task = list(all_task_metrics.keys())[0]
        task_metrics_dict = all_task_metrics[single_task]
        for key, value in task_metrics_dict.items():
            # Convert eval_taskname_metric_mean to eval_metric_mean etc
            legacy_key = key.replace(f'_{single_task}', '')
            flat_metrics[legacy_key] = value
    
    return flat_metrics


def preprocess_config(config):
    """
    Validate and preprocess training config.
    
    Args:
        config: Raw config dictionary from YAML
    
    Returns:
        Processed config with validated fields and correct types
    
    Raises:
        SystemExit if validation fails
    """
    import sys
    
    # Define required fields and their types
    field_specs = {
        # Experiment configuration
        'output_dir': ('str', None, "Output directory path"),
        'tokenizer_path': ('str', None, "Path to tokenizer"),
        
        # Dataset configuration
        'dataset.path': ('str', None, "Dataset path"),
        'dataset.max_sequence_length': ('int', None, "Maximum sequence length"),
        
        # Model configuration
        'model.vocab_size': ('int', None, "Model vocabulary size"),
        'model.hidden_size': ('int', None, "Model hidden size"),
        'model.num_hidden_layers': ('int', None, "Number of hidden layers"),
        'model.num_attention_heads': ('int', None, "Number of attention heads"),
        'model.intermediate_size': ('int', None, "Intermediate layer size"),
        'model.init_scale': ('float', None, "Weight initialization scale"),
        
        # Training configuration
        'training.learning_rate': ('float', None, "Learning rate"),
        'training.weight_decay': ('float', None, "Weight decay"),
        'training.batch_size': ('int', None, "Training batch size"),
        'training.eval_batch_size': ('int', None, "Evaluation batch size"),
        'training.num_epochs': ('int', None, "Number of epochs"),
        'training.warmup_steps': ('int', None, "Warmup steps"),
        'training.seed': ('int', None, "Random seed"),
        'training.scheduler': ('str', ['linear_with_warmup'], "Learning rate scheduler"),
        
        # Checkpointing configuration
        'checkpointing.save_steps': ('float', None, "Save checkpoint frequency"),
        'checkpointing.eval_steps': ('float', None, "Evaluation frequency"),
    }
    
    # Validate and convert each field
    for field_path, (field_type, allowed_values, description) in field_specs.items():
        # Navigate to the field
        parts = field_path.split('.')
        value = config
        parent = None
        last_part = None
        
        for part in parts:
            if part not in value:
                print(f"Error: Missing required config field '{field_path}': {description}")
                sys.exit(1)
            parent = value
            last_part = part
            value = value[part]
        
        # Convert type
        if field_type == 'int':
            parent[last_part] = int(value)
        elif field_type == 'float':
            parent[last_part] = float(value)
        elif field_type == 'str':
            parent[last_part] = str(value) if value is not None else value
        
        # Validate allowed values
        if allowed_values is not None:
            if parent[last_part] not in allowed_values:
                print(f"Error: Invalid value for '{field_path}': {parent[last_part]}")
                print(f"  Allowed values: {allowed_values}")
                sys.exit(1)
    
    # Validate eval config if present (for task-specific evaluation settings)
    if 'eval' in config and 'randomwalk' in config.get('eval', {}):
        if 'cities_csv' in config['eval']['randomwalk']:
            cities_csv = config['eval']['randomwalk']['cities_csv']
            if not os.path.exists(cities_csv):
                print(f"Error: Cities CSV not found: {cities_csv}")
                sys.exit(1)
            print(f"Randomwalk evaluation configured with cities: {cities_csv}")

    # Check for deprecated randomwalk config structure
    if 'randomwalk' in config:
        print(f"WARNING: Top-level 'randomwalk' config is deprecated. Use 'eval.randomwalk' instead.")
        if 'distance_km' in config['randomwalk']:
            print(f"WARNING: randomwalk.distance_km is IGNORED. Distance threshold is parsed from each sample's prompt.")
    
    return config


def get_model(config):
    """
    Initialize model from config.
    
    Args:
        config: Preprocessed configuration dictionary
    
    Returns:
        model: Initialized model with tokenizer attached as model.tokenizer
    """
    # Load tokenizer
    tokenizer_path = config['tokenizer_path']
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Set padding side for training (will switch to left for generation in callbacks)
    tokenizer.padding_side = 'right'
    print(f"Tokenizer padding side set to: {tokenizer.padding_side} (for training, will use left for generation)")
    
    # Check if we're loading from a checkpoint
    checkpoint_path = config['model'].get('ckpt', None)
    
    if checkpoint_path:
        # Load model from checkpoint
        print(f"Loading model from checkpoint: {checkpoint_path}")
        
        # Check if checkpoint exists
        if not os.path.exists(checkpoint_path):
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            sys.exit(1)
        
        # Load the model directly from checkpoint
        # This preserves the architecture and weights
        model = Qwen2ForCausalLM.from_pretrained(checkpoint_path)
        print(f"Model loaded from checkpoint successfully")
    else:
        # Initialize model from scratch
        print("Initializing model from scratch...")
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
        
        # Create model
        model = Qwen2ForCausalLM(model_config)
        
        # Apply custom weight initialization
        init_scale = config['model']['init_scale']
        print(f"Applying weight initialization with scale={init_scale}")
        model.apply(lambda m: init_weights(m, init_scale))
    
    # Attach tokenizer to model as per convention
    model.tokenizer = tokenizer
    
    return model


def get_dataset(config):
    """
    Load and prepare datasets from config.
    
    Args:
        config: Preprocessed configuration dictionary
    
    Returns:
        tuple: (train_dataset, eval_dataset, tokenizer, collator)
    """
    # Load tokenizer (needed for dataset preparation)
    tokenizer_path = config['tokenizer_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = 'right'
    
    # Load dataset
    print(f"Loading dataset from {config['dataset']['path']}")
    dataset = load_from_disk(config['dataset']['path'])
    
    # Prepare datasets (simplified - no tokenization needed here)
    if 'validation' in dataset:
        train_dataset = BaseDataset(dataset['train'], split='train')
        eval_dataset = BaseDataset(dataset['validation'], split='validation')
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
            train_data.select(range(eval_size, dataset_size))
        )
        eval_dataset = BaseDataset(
            train_data.select(range(eval_size))
        )
        print(f"Using {dataset_size - eval_size} samples for training")
        print(f"Using {eval_size} samples for validation")
    
    # Get multi-task collator
    collator = get_collator(config, tokenizer)
    print(f"Using {collator.__class__.__name__} (task types determined per example)")
    
    return train_dataset, eval_dataset, tokenizer, collator

class GenerationEvalCallback(TrainerCallback):
    """
    Callback to perform generation-based evaluation and update plots during training.
    Can be used with HuggingFace Trainer as a callback.
    """

    def __init__(self, exp_dir, tokenizer, eval_dataset, device, config=None):
        self.exp_dir = exp_dir
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.device = device
        self.config = config
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called after standard evaluation - add generation metrics."""
        if model is None:
            return control
            
        # Import from unified evaluation module
        from src.evaluation import evaluate_with_generation

        # Perform generation-based evaluation
        print("\nPerforming generation-based evaluation...")
        num_samples = 64
        gen_metrics = evaluate_with_generation(
            model, self.eval_dataset, self.tokenizer, self.device,
            num_samples=num_samples, batch_size=16, config=self.config, return_details=False
        )
        
        # Add metrics to the log (convert numpy types to Python native)
        if state.log_history and gen_metrics:
            state.log_history[-1].update(convert_numpy_to_python(gen_metrics))
        
        # Also add to metrics dict if provided (for best model tracking)
        if metrics is not None and gen_metrics:
            metrics.update(convert_numpy_to_python(gen_metrics))
        
        # Save plots after each evaluation
        save_training_plots(self.exp_dir, state, None)
        
        # Print generation metrics
        if gen_metrics:
            print(f"\n[Evaluation Metrics]")

            # Extract task types from metrics keys
            task_types = set()
            for key in gen_metrics.keys():
                if '_metric_mean' in key and key.startswith('eval_'):
                    parts = key.split('_')
                    if len(parts) >= 3:
                        task_types.add(parts[1])

            # Print task-specific metrics
            for task_type in sorted(task_types):
                task_mean_key = f'eval_{task_type}_metric_mean'
                task_std_key = f'eval_{task_type}_metric_std'
                task_min_key = f'eval_{task_type}_metric_min'
                task_max_key = f'eval_{task_type}_metric_max'
                task_median_key = f'eval_{task_type}_metric_median'
                task_count_key = f'eval_{task_type}_valid_count'
                task_ratio_key = f'eval_{task_type}_valid_ratio'
                
                if task_mean_key in gen_metrics:
                    print(f"\n{task_type.upper()} TASK:")
                    if task_type == 'distance':
                        print(f"  Avg Absolute Error: {gen_metrics[task_mean_key]:.2f} units (±{gen_metrics[task_std_key]:.2f})")
                        print(f"  Min: {gen_metrics[task_min_key]:.2f} units")
                        print(f"  Max: {gen_metrics[task_max_key]:.2f} units")  
                        print(f"  Median: {gen_metrics[task_median_key]:.2f} units")
                        print(f"  Avg Walk Validity: {gen_metrics[task_mean_key]:.3f} (±{gen_metrics[task_std_key]:.3f})")
                        print(f"  Min: {gen_metrics[task_min_key]:.3f}")
                        print(f"  Max: {gen_metrics[task_max_key]:.3f}")  
                        print(f"  Median: {gen_metrics[task_median_key]:.3f}")
                    else:
                        print(f"  Metric Mean: {gen_metrics[task_mean_key]:.2f} (±{gen_metrics[task_std_key]:.2f})")
                        print(f"  Min: {gen_metrics[task_min_key]:.2f}")
                        print(f"  Max: {gen_metrics[task_max_key]:.2f}")  
                        print(f"  Median: {gen_metrics[task_median_key]:.2f}")
                    
                    print(f"  Valid generations: {gen_metrics[task_count_key]}/{int(gen_metrics[task_count_key] / gen_metrics[task_ratio_key]) if gen_metrics[task_ratio_key] > 0 else 0} ({gen_metrics[task_ratio_key]*100:.1f}%)")
            
            # For backward compatibility, also print legacy metrics if present
            if 'eval_metric_mean' in gen_metrics:
                # Try to infer task type from other metrics
                primary_task = sorted(task_types)[0] if task_types else 'unknown'
                print(f"\nOVERALL (primary task: {primary_task.upper()}):")
                if primary_task == 'distance':
                    print(f"  Avg Absolute Error: {gen_metrics['eval_metric_mean']:.2f} units (±{gen_metrics['eval_metric_std']:.2f})")
                elif primary_task == 'randomwalk':
                    print(f"  Avg Walk Validity: {gen_metrics['eval_metric_mean']:.3f} (±{gen_metrics['eval_metric_std']:.3f})")
                else:
                    print(f"  Metric Mean: {gen_metrics['eval_metric_mean']:.2f} (±{gen_metrics['eval_metric_std']:.2f})")
                print(f"  Valid generations: {gen_metrics['eval_valid_count']}/{num_samples} ({gen_metrics['eval_valid_ratio']*100:.1f}%)")
        
        return control



def filter_dataframe_by_pattern(df, pattern, column_name='name'):
    """
    Filter DataFrame by regex pattern.
    
    Args:
        df: DataFrame to filter
        pattern: Regex pattern or special syntax like 'column:pattern'
        column_name: Default column to match against
    
    Returns:
        Filtered DataFrame
    """
    if not pattern or pattern == '.*':
        return df
    
    # Check for column-specific syntax
    if ':' in pattern and pattern.count(':') == 1:
        col_prefix, col_pattern = pattern.split(':', 1)
        if col_prefix in df.columns:
            return df[df[col_prefix].str.match(col_pattern, na=False)]
    
    # Default column matching
    if column_name in df.columns:
        return df[df[column_name].str.match(pattern, na=False)]
    else:
        raise ValueError(f"Column '{column_name}' not found in DataFrame")




def save_training_plots(exp_dir, state, config=None):
    """Save training plots in summary/ folder - one for loss, one per task type."""
    # Create summary directory
    summary_dir = Path(exp_dir) / 'summary'
    summary_dir.mkdir(exist_ok=True)

    # Get plot settings from config
    use_log_scale = True  # Default to log scale
    if config and isinstance(config, dict):
        use_log_scale = config.get('plot_log_scale', True)
    
    # Extract all metrics from log history
    train_losses = []
    eval_losses = []
    eval_steps = []
    train_steps = []
    task_metrics = {}  # Will store metrics per task type
    
    # First, check if checkpoint-0 exists and add it to the beginning
    # (only relevant when called from training, not from evaluate_checkpoints)
    checkpoint_0_path = Path(exp_dir) / 'checkpoints' / 'checkpoint-0' / 'eval_results.json'
    loaded_checkpoint_0 = False
    if checkpoint_0_path.exists():
        try:
            with open(checkpoint_0_path, 'r') as f:
                ckpt0_data = json.load(f)
            if 'eval_loss' in ckpt0_data:
                eval_losses.append(ckpt0_data['eval_loss'])
                eval_steps.append(0)
                loaded_checkpoint_0 = True

            # Extract task-specific metrics from checkpoint-0
            for key, value in ckpt0_data.items():
                if '_metric_mean' in key and key.startswith('eval_'):
                    # Extract task type from key like eval_distance_metric_mean
                    parts = key.split('_')
                    if len(parts) >= 3:
                        task_type = parts[1]  # distance, randomwalk, etc
                        if task_type not in task_metrics:
                            task_metrics[task_type] = {'steps': [], 'values': []}
                        task_metrics[task_type]['steps'].append(0)
                        task_metrics[task_type]['values'].append(value)
        except:
            pass  # If we can't read it, just continue without it

    # Then extract metrics from log history (training steps)
    for entry in state.log_history:
        step = entry.get('step', 0)

        if 'loss' in entry and 'eval_loss' not in entry:
            train_losses.append(entry['loss'])
            train_steps.append(step)

        # Only skip step 0 if we already loaded it from file
        if 'eval_loss' in entry and (step != 0 or not loaded_checkpoint_0):
            eval_losses.append(entry['eval_loss'])
            eval_steps.append(step)

        # Extract task-specific metrics
        for key, value in entry.items():
            if '_metric_mean' in key and key.startswith('eval_') and (step != 0 or not loaded_checkpoint_0):
                # Extract task type from key like eval_distance_metric_mean
                parts = key.split('_')
                if len(parts) >= 3:
                    task_type = parts[1]  # distance, randomwalk, etc
                    if task_type not in task_metrics:
                        task_metrics[task_type] = {'steps': [], 'values': []}
                    task_metrics[task_type]['steps'].append(step)
                    task_metrics[task_type]['values'].append(value)
    
    # 1. Create loss plot
    plt.figure(figsize=(10, 6))
    if train_losses:
        plt.plot(train_steps, train_losses, alpha=0.3, label='Train Loss (per batch)')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, 'r-', label='Eval Loss', marker='o', markersize=4)
    if use_log_scale:
        plt.xlabel('Step (log scale)')
        plt.ylabel('Loss (log scale)')
        plt.xscale('log')
        plt.yscale('log')
    else:
        plt.xlabel('Step')
        plt.ylabel('Loss')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(summary_dir / 'loss.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # 2. Create individual task plots
    for task_type, data in task_metrics.items():
        if not data['values']:
            continue
            
        plt.figure(figsize=(10, 6))
        # Filter data based on scale type
        if use_log_scale:
            # Filter out step 0 and values below 100 for log scale
            plot_steps = [s for s in data['steps'] if s >= 100]
            plot_values = [v for s, v in zip(data['steps'], data['values']) if s >= 100]
        else:
            # Use all data for linear scale
            plot_steps = data['steps']
            plot_values = data['values']

        if plot_steps:  # Only plot if we have data
            plt.plot(plot_steps, plot_values, 'b-', marker='o', markersize=4)
        if use_log_scale:
            plt.xlabel('Step (log scale)')
            plt.xscale('log')
            plt.xlim(left=100)  # Start x-axis at 100
        else:
            plt.xlabel('Step')
        
        if task_type == 'distance':
            plt.ylabel('Average Absolute Error (km, log scale)')
            plt.title('Distance Task: Evaluation Absolute Distance Error\n(Lower is better, 4025 km = parse failed)')
            plt.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10 km (excellent)')
            plt.axhline(y=50, color='lightgreen', linestyle='--', alpha=0.5, label='50 km (good)')
            plt.axhline(y=100, color='yellow', linestyle='--', alpha=0.5, label='100 km (okay)')
            plt.axhline(y=500, color='orange', linestyle='--', alpha=0.5, label='500 km (poor)')
            plt.axhline(y=4025, color='red', linestyle=':', alpha=0.7, label='4025 km (format error)')
            plt.yscale('log')
            plt.ylim(bottom=1, top=4427)  # 1.1 * 4025
        elif task_type == 'randomwalk':
            plt.ylabel('Combined Score (1=perfect, 0=failure)')
            plt.title('Random Walk Task: Combined Score\n(Higher is better, 1 = perfect, 0 = complete failure)')
            plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='0.9 (excellent)')
            plt.axhline(y=0.75, color='lightgreen', linestyle='--', alpha=0.5, label='0.75 (good)')
            plt.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='0.5 (medium)')
            plt.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='0.25 (poor)')
            plt.axhline(y=0.0, color='red', linestyle=':', alpha=0.7, label='0.0 (complete failure)')
            plt.ylim(bottom=-0.05, top=1.05)
        elif task_type == 'trianglearea':
            plt.ylabel('Average Absolute Error (square units, log scale)')
            plt.title('Triangle Area Task: Evaluation Error\n(Lower is better, 3.24M = parse failed)')
            plt.axhline(y=100, color='green', linestyle='--', alpha=0.5, label='100 (excellent)')
            plt.axhline(y=1000, color='lightgreen', linestyle='--', alpha=0.5, label='1000 (good)')
            plt.axhline(y=10000, color='yellow', linestyle='--', alpha=0.5, label='10k (okay)')
            plt.axhline(y=100000, color='orange', linestyle='--', alpha=0.5, label='100k (poor)')
            plt.axhline(y=3240000, color='red', linestyle=':', alpha=0.7, label='3.24M (format error)')
            plt.yscale('log')
            plt.ylim(bottom=10, top=3564000)  # 1.1 * 3240000
        elif task_type == 'angle':
            plt.ylabel('Average Absolute Error (degrees, log scale)')
            plt.title('Angle Task: Evaluation Error\n(Lower is better, 0° = perfect, 180° = format error)')
            plt.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10° (excellent)')
            plt.axhline(y=30, color='lightgreen', linestyle='--', alpha=0.5, label='30° (good)')
            plt.axhline(y=60, color='yellow', linestyle='--', alpha=0.5, label='60° (okay)')
            plt.axhline(y=90, color='orange', linestyle='--', alpha=0.5, label='90° (poor)')
            plt.axhline(y=180, color='red', linestyle=':', alpha=0.7, label='180° (format error)')
            plt.yscale('log')
            plt.ylim(bottom=1, top=198)  # 1.1 * 180
        elif task_type == 'perimeter':
            plt.ylabel('Average Absolute Error (units, log scale)')
            plt.title('Perimeter Task: Evaluation Error\n(Lower is better, 20000 = parse failed)')
            plt.axhline(y=50, color='green', linestyle='--', alpha=0.5, label='50 (excellent)')
            plt.axhline(y=200, color='lightgreen', linestyle='--', alpha=0.5, label='200 (good)')
            plt.axhline(y=1000, color='yellow', linestyle='--', alpha=0.5, label='1000 (okay)')
            plt.axhline(y=5000, color='orange', linestyle='--', alpha=0.5, label='5000 (poor)')
            plt.axhline(y=20000, color='red', linestyle=':', alpha=0.7, label='20000 (format error)')
            plt.yscale('log')
            plt.ylim(bottom=10, top=22000)
        elif task_type == 'nearest_neighbor':
            plt.ylabel('Average Jaccard Similarity')
            plt.title('Nearest Neighbor Task: Jaccard Similarity\n(Higher is better, 1.0 = perfect, 0.0 = failure)')
            plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='0.9 (excellent)')
            plt.axhline(y=0.7, color='lightgreen', linestyle='--', alpha=0.5, label='0.7 (good)')
            plt.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='0.5 (okay)')
            plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='0.3 (poor)')
            plt.axhline(y=0.0, color='red', linestyle=':', alpha=0.7, label='0.0 (failure)')
            plt.ylim(bottom=-0.05, top=1.05)
        elif task_type == 'center':
            plt.ylabel('Average Distance Error (units, log scale)')
            plt.title('Center Task: Distance Error to True Center\n(Lower is better, 4025 = parse failed)')
            plt.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10 (excellent)')
            plt.axhline(y=50, color='lightgreen', linestyle='--', alpha=0.5, label='50 (good)')
            plt.axhline(y=200, color='yellow', linestyle='--', alpha=0.5, label='200 (okay)')
            plt.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='1000 (poor)')
            plt.axhline(y=4025, color='red', linestyle=':', alpha=0.7, label='4025 (format error)')
            plt.yscale('log')
            plt.ylim(bottom=1, top=4427)
        elif task_type == 'circlecount':
            plt.ylabel('Average Absolute Error (count, log scale)')
            plt.title('Circle Count Task: Count Error\n(Lower is better, 1000 = parse failed)')
            plt.axhline(y=1, color='green', linestyle='--', alpha=0.5, label='1 (excellent)')
            plt.axhline(y=5, color='lightgreen', linestyle='--', alpha=0.5, label='5 (good)')
            plt.axhline(y=20, color='yellow', linestyle='--', alpha=0.5, label='20 (okay)')
            plt.axhline(y=100, color='orange', linestyle='--', alpha=0.5, label='100 (poor)')
            plt.axhline(y=1000, color='red', linestyle=':', alpha=0.7, label='1000 (format error)')
            plt.yscale('log')
            plt.ylim(bottom=0.5, top=1100)
        elif task_type == 'randring':
            plt.ylabel('Combined Score (1=perfect, 0=failure)')
            plt.title('Random Ring Task: Validity × Length Penalty\n(Higher is better, 1.0 = perfect, 0.0 = failure)')
            plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='0.9 (excellent)')
            plt.axhline(y=0.7, color='lightgreen', linestyle='--', alpha=0.5, label='0.7 (good)')
            plt.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='0.5 (okay)')
            plt.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='0.2 (poor)')
            plt.axhline(y=0.0, color='red', linestyle=':', alpha=0.7, label='0.0 (failure)')
            plt.ylim(bottom=-0.05, top=1.05)
        elif task_type in ['compass', 'crossing', 'inside']:
            plt.ylabel('Average Accuracy')
            plt.title(f'{task_type.title()} Task: Binary Accuracy\n(Higher is better, 1.0 = perfect, 0.0 = failure)')
            plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='0.9 (excellent)')
            plt.axhline(y=0.7, color='lightgreen', linestyle='--', alpha=0.5, label='0.7 (good)')
            plt.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='0.5 (random)')
            plt.axhline(y=0.3, color='orange', linestyle='--', alpha=0.5, label='0.3 (poor)')
            plt.axhline(y=0.0, color='red', linestyle=':', alpha=0.7, label='0.0 (failure)')
            plt.ylim(bottom=-0.05, top=1.05)
        else:
            # Unknown task type - generic plot
            plt.ylabel('Metric Value')
            plt.title(f'{task_type.title()} Task: Evaluation Metrics')
        
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(summary_dir / f'{task_type}.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"Plots saved to {summary_dir}/: loss.png + {[f'{t}.png' for t in task_metrics.keys()]}")


def add_pause_to_gif(input_gif_path, output_gif_path=None, final_frame_duration=3000):
    """
    Modify a GIF to add a pause on the final frame.
    
    Args:
        input_gif_path: Path to the input GIF
        output_gif_path: Path for output GIF (if None, overwrites input)
        final_frame_duration: Duration for final frame in milliseconds (default 3000ms)
    """
    from PIL import Image
    from pathlib import Path
    
    input_path = Path(input_gif_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Input GIF not found: {input_path}")
    
    if output_gif_path is None:
        output_gif_path = input_path
    else:
        output_gif_path = Path(output_gif_path)
    
    # Open the GIF and extract all frames
    img = Image.open(input_path)
    frames = []
    durations = []
    
    try:
        while True:
            # Get frame duration (default to 500ms if not specified)
            duration = img.info.get('duration', 500)
            durations.append(duration)
            
            # Copy and save the frame
            frames.append(img.copy())
            img.seek(img.tell() + 1)
    except EOFError:
        pass  # End of GIF
    
    if not frames:
        raise ValueError("No frames found in GIF")
    
    # Modify the duration of the last frame
    durations[-1] = final_frame_duration
    
    print(f"Found {len(frames)} frames")
    print(f"Original durations: {durations[:5]}... (showing first 5)")
    print(f"Modified last frame duration: {durations[-1]}ms")
    
    # Save the modified GIF
    frames[0].save(
        output_gif_path,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0
    )
    
    print(f"Saved modified GIF to: {output_gif_path}")