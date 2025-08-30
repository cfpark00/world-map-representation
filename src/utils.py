#!/usr/bin/env python3
"""Utility functions for the WM_1 project."""

import os
import re
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_from_disk
from sklearn.metrics.pairwise import haversine_distances


def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points 
    on the earth (specified in decimal degrees)
    Returns distance in kilometers
    """
    # Convert degrees to radians and create coordinate arrays
    # scikit-learn expects [latitude, longitude] in radians
    coords = np.array([[np.radians(lat1), np.radians(lon1)],
                       [np.radians(lat2), np.radians(lon2)]])
    
    # Calculate haversine distance (returns radians)
    dist_radians = haversine_distances(coords)[0, 1]
    
    # Convert to kilometers
    r = 6371  # Radius of earth in kilometers
    return dist_radians * r


def load_cities_csv(cities_csv_path=None, default_path='outputs/datasets/cities_100k_plus.csv'):
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
                f"Please run: python src/data_processing/create_filtered_dataset.py"
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