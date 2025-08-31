#!/usr/bin/env python3
"""Utility functions for the WM_1 project."""

import os
import re
import math
import shutil
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from datasets import load_from_disk
from scipy.spatial import cKDTree


def init_experiment_directory(exp_dir, overwrite=False, exp_dir_prefix=None):
    """
    Initialize experiment directory with safety checks.
    
    Args:
        exp_dir: Path to experiment directory (str or Path object)
        overwrite: Whether to overwrite existing directory
        exp_dir_prefix: Required prefix for overwrite safety (from EXP_DIR_PREFIX env var)
    
    Returns:
        Path object of the created directory
    
    Raises:
        SystemExit: If directory exists without overwrite, or safety checks fail
    """
    exp_dir = Path(exp_dir)
    
    if exp_dir.exists():
        if overwrite:
            # Get the EXP_DIR_PREFIX for safety check
            if exp_dir_prefix is None:
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
    
    return exp_dir


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    
    Args:
        lon1, lat1: Longitude and latitude of first point(s) in degrees
        lon2, lat2: Longitude and latitude of second point(s) in degrees
        Can be scalars or arrays of same shape
    
    Returns:
        Distance(s) in kilometers
    """
    # Convert to numpy arrays to handle both scalar and vector inputs
    lon1 = np.asarray(lon1)
    lat1 = np.asarray(lat1)
    lon2 = np.asarray(lon2)
    lat2 = np.asarray(lat2)
    
    # Convert degrees to radians
    lon1_rad = np.radians(lon1)
    lat1_rad = np.radians(lat1)
    lon2_rad = np.radians(lon2)
    lat2_rad = np.radians(lat2)
    
    # Haversine formula (fully vectorized)
    dlon = lon2_rad - lon1_rad
    dlat = lat2_rad - lat1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    # Convert to kilometers
    r = 6371  # Radius of earth in kilometers
    distance_km = c * r
    
    # Return scalar if inputs were scalar
    if lon1.shape == ():
        return float(distance_km)
    return distance_km


def load_cities_csv(cities_csv_path=None, default_path='outputs/datasets/cities_100k_plus_seed42.csv'):
    """
    Load cities CSV with fallback to default path.
    
    Args:
        cities_csv_path: Optional path to cities CSV file
        default_path: Default path if cities_csv_path is None
    
    Returns:
        pandas DataFrame with city data
    """
    if cities_csv_path:
        if not os.path.exists(cities_csv_path):
            raise FileNotFoundError(f"Cities CSV not found: {cities_csv_path}")
        df = pd.read_csv(cities_csv_path)
    else:
        if not os.path.exists(default_path):
            raise FileNotFoundError(
                f"Default cities CSV not found: {default_path}\n"
                f"Please run: python src/data_processing/generate_filtered_dataset.py 100000"
            )
        df = pd.read_csv(default_path)
    
    return df


def extract_coordinates(df, coord_column='Coordinates'):
    """
    Extract latitude and longitude from a coordinate string column.
    
    Args:
        df: DataFrame with coordinate column
        coord_column: Name of column containing coordinates as "lat,lon"
    
    Returns:
        DataFrame with added 'latitude' and 'longitude' columns
    """
    coords = df[coord_column].str.split(',', expand=True)
    df['latitude'] = coords[0].astype(float)
    df['longitude'] = coords[1].astype(float)
    return df


def parse_location(text):
    """
    Parse location from text like 'loc(c_1234)=567,890' or '567,890'.
    
    Args:
        text: String containing location information
    
    Returns:
        Tuple of (x, y) coordinates or (None, None) if not found
    """
    # Try to find the pattern after =
    match = re.search(r'=(-?\d+),(-?\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    # Try just numbers with comma
    match = re.search(r'(-?\d+),(-?\d+)', text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None, None


def parse_distance(text):
    """
    Parse distance from text like 'dist(c_1234,c_5678)=901' or just '901'.
    
    Args:
        text: String containing distance information
    
    Returns:
        Distance value as int or None if not found
    """
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
    Parse transitions from random walk text like 'walk_200=c_123,c_456,c_789'.
    Returns list of (city1_id, city2_id) tuples for each transition.
    
    Args:
        text: String containing walk information
    
    Returns:
        List of (city1_id, city2_id) tuples, or empty list if none found
    """
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
        cities_df: DataFrame with city data (must have 'row_id', 'latitude', 'longitude')
        distance_threshold_km: Maximum distance between consecutive cities
    
    Returns:
        Tuple of (num_valid_transitions, total_transitions)
    """
    if not transitions:
        return 0, 0
    
    valid_transitions = 0
    total_transitions = len(transitions)
    
    for city1_id, city2_id in transitions:
        # Find cities in dataframe
        try:
            city1 = cities_df[cities_df['row_id'] == city1_id].iloc[0]
            city2 = cities_df[cities_df['row_id'] == city2_id].iloc[0]
        except (IndexError, KeyError):
            # City ID not found in dataset - transition is invalid
            continue
        
        # Calculate distance
        distance_km = haversine(
            city1['longitude'], city1['latitude'],
            city2['longitude'], city2['latitude']
        )
        
        if distance_km <= distance_threshold_km:
            valid_transitions += 1
    
    return valid_transitions, total_transitions


class BaseDataset(Dataset):
    """
    Base dataset class for location-based tasks with common functionality.
    """
    
    def __init__(self, dataset_or_path, tokenizer, max_length, split='train', loss_mask_type=None):
        """
        Initialize dataset.
        
        Args:
            dataset_or_path: Path to HuggingFace dataset or dataset object
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            split: Dataset split to use ('train', 'val', 'test')
            loss_mask_type: Type of loss masking ('answer_only' or None)
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
        
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.loss_mask_type = loss_mask_type
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        # Tokenize prompt and completion separately
        prompt_tokens = self.tokenizer(item['prompt'], add_special_tokens=False)['input_ids']
        completion_tokens = self.tokenizer(item['completion'], add_special_tokens=False)['input_ids']
        
        # Combine into full sequence
        input_ids = prompt_tokens + completion_tokens
        
        # Truncate if too long
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Pad to max_length
        padding_length = self.max_length - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
        
        # Create labels (shift input_ids by 1 for next-token prediction)
        labels = input_ids.copy()
        
        # Apply loss masking if specified
        if self.loss_mask_type == 'answer_only':
            # Only compute loss on the completion part
            prompt_len = len(prompt_tokens)
            for i in range(min(prompt_len, len(labels))):
                labels[i] = -100  # -100 is ignored in CrossEntropyLoss
        
        # Mask padding tokens in labels
        for i in range(len(labels)):
            if attention_mask[i] == 0:
                labels[i] = -100
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long)
        }


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


def evaluate_with_generation(model, eval_dataset, tokenizer, device, task_type, num_samples=128, batch_size=16, config=None):
    """Evaluate model with generation and task-specific metric calculation."""
    model.eval()
    
    # Sample a subset for evaluation
    eval_indices = np.random.choice(len(eval_dataset), min(num_samples, len(eval_dataset)), replace=False)
    
    metrics = []
    all_generated_texts = []
    all_true_texts = []
    
    # For randomwalk evaluation, load cities data
    cities_df = None
    distance_threshold_km = None
    
    if task_type == 'randomwalk':
        if config is None or 'randomwalk' not in config:
            raise ValueError("FATAL: randomwalk evaluation requires config with randomwalk section")
        
        cities_csv = config['randomwalk']['cities_csv']
        distance_threshold_km = config['randomwalk']['distance_km']
        print(f"Loading cities data for randomwalk evaluation from: {cities_csv}")
        
        cities_df = pd.read_csv(cities_csv)
    
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
                # Handle both BaseDataset (has .dataset) and raw dataset
                if hasattr(eval_dataset, 'dataset'):
                    raw_item = eval_dataset.dataset[idx]
                else:
                    raw_item = eval_dataset[idx]
                
                # For random walk with greedy decoding, add random starting city to avoid identical generations
                if task_type == 'randomwalk' and cities_df is not None:
                    # Sample a random city and append to prompt
                    random_city = cities_df.sample(n=1).iloc[0]
                    base_prompt = raw_item['prompt']
                    # Add the city ID to the prompt (e.g., "<bos>walk_200=" -> "<bos>walk_200=c_123,")
                    modified_prompt = f"{base_prompt}c_{random_city['row_id']},"
                    batch_prompts.append(modified_prompt)
                else:
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
            
            # Generate with task-appropriate max tokens
            if task_type == 'randomwalk':
                max_new_tokens = 100  # Allow for long walks
            else:
                max_new_tokens = 20   # Original for location/distance
            
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
            
            # Print first few examples from first batch
            if batch_idx == 0:
                num_to_show = min(3, len(generated_batch))
                print(f"\n[Validation examples]")
                for ex_idx in range(num_to_show):
                    print(f"  Example {ex_idx + 1}:")
                    print(f"    Prompt: {batch_prompts[ex_idx]}")
                    print(f"    Expected: {batch_true_completions[ex_idx]}")
                    print(f"    Generated: {generated_batch[ex_idx]}")
            
            # Calculate task-specific metrics
            for i, (prompt, true_completion, generated) in enumerate(zip(batch_prompts, batch_true_completions, generated_batch)):
                all_generated_texts.append(generated)
                all_true_texts.append(prompt + true_completion)
                
                if task_type == 'location':
                    true_x, true_y = parse_location(true_completion)
                    gen_x, gen_y = parse_location(generated)
                    
                    if true_x is not None and gen_x is not None:
                        true_lon = math.degrees(true_x / 1000.0 - math.pi)
                        true_lat = math.degrees(true_y / 1000.0 - math.pi/2)
                        gen_lon = math.degrees(gen_x / 1000.0 - math.pi)
                        gen_lat = math.degrees(gen_y / 1000.0 - math.pi/2)
                        
                        dist = haversine(true_lon, true_lat, gen_lon, gen_lat)
                        metrics.append(dist)
                
                elif task_type == 'distance':
                    true_dist = parse_distance(true_completion)
                    gen_dist = parse_distance(generated)
                    
                    if true_dist is not None and gen_dist is not None:
                        abs_error = abs(true_dist - gen_dist)
                        metrics.append(abs_error)
                
                else:  # randomwalk
                    # No fallback check - cities_df MUST be loaded for randomwalk
                    transitions = parse_walk_transitions(generated)
                    
                    if transitions:
                        valid_trans, total_trans = validate_transitions(
                            transitions, cities_df, distance_threshold_km
                        )
                        # Use validity ratio as metric (1.0 = all transitions valid, 0.0 = none valid)
                        validity_ratio = valid_trans / total_trans if total_trans > 0 else 0.0
                        metrics.append(validity_ratio)
                    else:
                        # No transitions found gets 0 validity
                        metrics.append(0.0)
    
    # Calculate metrics
    if metrics:
        return {
            'eval_metric_mean': np.mean(metrics),
            'eval_metric_median': np.median(metrics),
            'eval_metric_std': np.std(metrics),
            'eval_metric_min': np.min(metrics),
            'eval_metric_max': np.max(metrics),
            'eval_valid_count': len(metrics),
            'eval_valid_ratio': len(metrics) / num_samples
        }
    else:
        # Use different failure values for different tasks
        if task_type == 'location':
            fail_value = 20000.0
        elif task_type == 'distance':
            fail_value = 100000.0
        else:  # randomwalk
            fail_value = 0.0  # 0% validity for failed randomwalks
        return {
            'eval_metric_mean': fail_value,
            'eval_metric_median': fail_value,
            'eval_metric_std': 0.0,
            'eval_metric_min': fail_value,
            'eval_metric_max': fail_value,
            'eval_valid_count': 0,
            'eval_valid_ratio': 0.0
        }


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
        # Task configuration
        'task_type': ('str', ['location', 'distance', 'randomwalk'], "Task type must be 'location', 'distance', or 'randomwalk'"),
        'exp_dir': ('str', None, "Experiment directory path"),
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
        'training.loss_mask_type': ('str', ['answer_only', 'full', None], "Loss mask type"),
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
    
    # Additional validation
    task_type = config['task_type']
    print(f"Task type: {task_type}")
    
    # Validate randomwalk-specific fields if needed
    if task_type == 'randomwalk':
        if 'randomwalk' not in config:
            print("Error: randomwalk task requires 'randomwalk' section in config")
            sys.exit(1)
        
        # Check required randomwalk fields
        required_rw_fields = ['cities_csv', 'distance_km']
        for field in required_rw_fields:
            if field not in config['randomwalk']:
                print(f"Error: Missing randomwalk.{field} in config")
                sys.exit(1)
        
        # Check cities CSV exists
        cities_csv = config['randomwalk']['cities_csv']
        if not os.path.exists(cities_csv):
            print(f"Error: Cities CSV not found: {cities_csv}")
            sys.exit(1)
        
        print(f"Randomwalk evaluation config:")
        print(f"  Cities CSV: {cities_csv}")
        print(f"  Distance threshold: {config['randomwalk']['distance_km']} km")
    
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
        tuple: (train_dataset, eval_dataset, tokenizer)
    """
    # Load tokenizer (needed for dataset preparation)
    tokenizer_path = config['tokenizer_path']
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.padding_side = 'right'
    
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
    
    return train_dataset, eval_dataset, tokenizer


from transformers import TrainerCallback, AutoTokenizer
from transformers.models.qwen2 import Qwen2Config, Qwen2ForCausalLM

class GenerationEvalCallback(TrainerCallback):
    """
    Callback to perform generation-based evaluation and update plots during training.
    Can be used with HuggingFace Trainer as a callback.
    """
    
    def __init__(self, exp_dir, tokenizer, eval_dataset, device, task_type, config=None):
        self.exp_dir = exp_dir
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
        self.device = device
        self.task_type = task_type
        self.config = config
    
    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        """Called after standard evaluation - add generation metrics."""
        if model is None:
            return control
            
        # Perform generation-based evaluation
        print("\nPerforming generation-based evaluation...")
        num_samples = 64
        gen_metrics = evaluate_with_generation(
            model, self.eval_dataset, self.tokenizer, self.device, 
            self.task_type, num_samples=num_samples, batch_size=16, config=self.config
        )
        
        # Add metrics to the log (convert numpy types to Python native)
        if state.log_history and gen_metrics:
            state.log_history[-1].update(convert_numpy_to_python(gen_metrics))
        
        # Also add to metrics dict if provided (for best model tracking)
        if metrics is not None and gen_metrics:
            metrics.update(convert_numpy_to_python(gen_metrics))
        
        # Save plots after each evaluation
        save_training_plots(self.exp_dir, state, self.task_type)
        
        # Print generation metrics
        if gen_metrics:
            print(f"\n[Evaluation Metrics]")
            if self.task_type == 'location':
                print(f"  Avg Haversine Distance: {gen_metrics['eval_metric_mean']:.2f} km (±{gen_metrics['eval_metric_std']:.2f})")
                print(f"  Min: {gen_metrics['eval_metric_min']:.2f} km")
                print(f"  Max: {gen_metrics['eval_metric_max']:.2f} km")  
                print(f"  Median: {gen_metrics['eval_metric_median']:.2f} km")
            elif self.task_type == 'distance':
                print(f"  Avg Absolute Error: {gen_metrics['eval_metric_mean']:.2f} km (±{gen_metrics['eval_metric_std']:.2f})")
                print(f"  Min: {gen_metrics['eval_metric_min']:.2f} km")
                print(f"  Max: {gen_metrics['eval_metric_max']:.2f} km")  
                print(f"  Median: {gen_metrics['eval_metric_median']:.2f} km")
            else:  # randomwalk
                print(f"  Avg Walk Validity: {gen_metrics['eval_metric_mean']:.3f} (±{gen_metrics['eval_metric_std']:.3f})")
                print(f"  Min: {gen_metrics['eval_metric_min']:.3f}")
                print(f"  Max: {gen_metrics['eval_metric_max']:.3f}")  
                print(f"  Median: {gen_metrics['eval_metric_median']:.3f}")
            print(f"  Valid generations: {gen_metrics['eval_valid_count']}/{num_samples} ({gen_metrics['eval_valid_ratio']*100:.1f}%)")
        
        return control


def save_training_plots(exp_dir, state, task_type='location'):
    """Save training plots for either location or distance tasks."""
    # Extract metrics from log history
    train_losses = []
    eval_losses = []
    eval_metrics = []
    eval_steps = []
    train_steps = []
    
    for entry in state.log_history:
        if 'loss' in entry and 'eval_loss' not in entry:
            train_losses.append(entry['loss'])
            train_steps.append(entry.get('step', len(train_losses)))
        if 'eval_loss' in entry:
            eval_losses.append(entry['eval_loss'])
            eval_steps.append(entry.get('step', len(eval_losses)))
        if 'eval_metric_mean' in entry:
            eval_metrics.append(entry['eval_metric_mean'])
    
    # Create plots matching original format
    plt.figure(figsize=(12, 5))
    
    # Loss plot
    plt.subplot(1, 2, 1)
    if train_losses:
        plt.plot(train_steps, train_losses, alpha=0.3, label='Train Loss (per batch)')
    if eval_losses:
        plt.plot(eval_steps, eval_losses, 'r-', label='Eval Loss', marker='o', markersize=4)
    plt.xlabel('Step')
    plt.ylabel('Loss (log scale)')
    plt.yscale('log')
    plt.title('Training and Evaluation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Task-specific metric plot
    plt.subplot(1, 2, 2)
    if eval_metrics:
        plt.plot(eval_steps[:len(eval_metrics)], eval_metrics, 'b-', marker='o', markersize=4)
        plt.xlabel('Step')
        
        if task_type == 'location':
            plt.ylabel('Average Haversine Distance (km, log scale)')
            plt.title('Evaluation Haversine Distance\n(Lower is better, 20000km = parse failed)')
            plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100km reference')
            plt.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='1000km reference')
            plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='10000km (poor)')
            plt.axhline(y=20000, color='red', linestyle=':', alpha=0.7, label='20000km (parse failed)')
            plt.ylim(bottom=10, top=30000)
            plt.yscale('log')
        elif task_type == 'distance':
            plt.ylabel('Average Absolute Error (km, log scale)')
            plt.title('Evaluation Absolute Distance Error\n(Lower is better, 100000km = parse failed)')
            plt.axhline(y=10, color='green', linestyle='--', alpha=0.5, label='10km reference')
            plt.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='100km reference')
            plt.axhline(y=1000, color='orange', linestyle='--', alpha=0.5, label='1000km reference')
            plt.axhline(y=10000, color='gray', linestyle='--', alpha=0.5, label='10000km (poor)')
            plt.axhline(y=100000, color='red', linestyle=':', alpha=0.7, label='100000km (parse failed)')
            plt.ylim(bottom=1, top=200000)
            plt.yscale('log')
        else:  # randomwalk
            plt.ylabel('Average Walk Validity (linear scale)')
            plt.title('Evaluation Walk Validity\n(Higher is better, 1.0 = perfect validity)')
            plt.axhline(y=0.1, color='red', linestyle=':', alpha=0.7, label='0.1 (poor)')
            plt.axhline(y=0.25, color='orange', linestyle='--', alpha=0.5, label='0.25 (low)')
            plt.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='0.5 (medium)')
            plt.axhline(y=0.75, color='lightgreen', linestyle='--', alpha=0.5, label='0.75 (good)')
            plt.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, label='0.9 (excellent)')
            plt.ylim(bottom=0, top=1.1)
        
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(exp_dir / 'summary.png', dpi=150)
    plt.close()