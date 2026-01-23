"""
Load and align representations for CKA analysis.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import re


def load_all_checkpoints(repr_dir: Path) -> Tuple[Dict[int, np.ndarray], Dict[int, pd.DataFrame]]:
    """
    Load all checkpoint representations from a directory.

    Args:
        repr_dir: Path to representations directory containing checkpoint-* subdirectories

    Returns:
        Tuple of (representations, metadata):
            - representations: Dict mapping checkpoint step -> representation array
            - metadata: Dict mapping checkpoint step -> DataFrame with city info
    """
    import torch
    import json

    repr_dir = Path(repr_dir)

    if not repr_dir.exists():
        raise ValueError(f"Representation directory does not exist: {repr_dir}")

    representations = {}
    metadata = {}

    # Find all checkpoint directories
    checkpoint_dirs = sorted([d for d in repr_dir.glob('checkpoint-*') if d.is_dir()])

    if len(checkpoint_dirs) == 0:
        raise ValueError(f"No checkpoint directories found in {repr_dir}")

    for ckpt_dir in checkpoint_dirs:
        # Extract step number from directory name: checkpoint-12345 -> 12345
        match = re.search(r'checkpoint-(\d+)', ckpt_dir.name)
        if not match:
            continue

        step = int(match.group(1))

        # Load metadata.json
        metadata_file = ckpt_dir / 'metadata.json'
        if not metadata_file.exists():
            continue

        with open(metadata_file, 'r') as f:
            meta = json.load(f)

        # Load representations.pt
        repr_file = ckpt_dir / 'representations.pt'
        if not repr_file.exists():
            continue

        # Load representation data (dict with 'representations_flat' key)
        repr_data = torch.load(repr_file, map_location='cpu')

        # Get flattened representations
        if 'representations_flat' in repr_data:
            repr_flat = repr_data['representations_flat'].numpy()
        elif 'representations' in repr_data:
            # Fallback: flatten manually if only non-flat version exists
            repr_tensor = repr_data['representations']
            n_cities = repr_tensor.shape[0]
            repr_flat = repr_tensor.reshape(n_cities, -1).numpy()
        else:
            raise ValueError(f"No representations found in {repr_file}")

        representations[step] = repr_flat

        # Extract city info from metadata
        city_info = meta['city_info']
        city_ids = [str(c.get('city_id', c.get('row_id', i))) for i, c in enumerate(city_info)]
        city_names = [c['name'] for c in city_info]
        regions = [c.get('region', c.get('country', 'Unknown')) for c in city_info]

        meta_dict = {
            'city_id': city_ids,
            'city_name': city_names,
            'region': regions,
        }
        metadata[step] = pd.DataFrame(meta_dict)

    return representations, metadata


def align_representations(repr1: np.ndarray, meta1: pd.DataFrame,
                         repr2: np.ndarray, meta2: pd.DataFrame,
                         city_filter: Optional[str] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Align two representation matrices to have matching cities in the same order.

    Args:
        repr1: First representation matrix (n1 x d1)
        meta1: Metadata DataFrame for first representations
        repr2: Second representation matrix (n2 x d2)
        meta2: Metadata DataFrame for second representations
        city_filter: Optional filter string (e.g., "region:^(?!Atlantis).* && city_id:^[1-9][0-9]{3,}$")

    Returns:
        Tuple of (aligned_repr1, aligned_repr2, common_city_ids)
    """
    # Apply city filter if provided
    if city_filter:
        meta1 = apply_city_filter(meta1, city_filter)
        meta2 = apply_city_filter(meta2, city_filter)

        # Filter representations to match filtered metadata
        repr1 = repr1[meta1.index.values]
        repr2 = repr2[meta2.index.values]

        # Reset indices after filtering
        meta1 = meta1.reset_index(drop=True)
        meta2 = meta2.reset_index(drop=True)

    # Find common cities
    cities1 = set(meta1['city_id'].tolist())
    cities2 = set(meta2['city_id'].tolist())
    common_cities = sorted(cities1 & cities2)

    if len(common_cities) == 0:
        raise ValueError("No common cities found between representations")

    # Create mapping from city_id to index
    city_to_idx1 = {city_id: idx for idx, city_id in enumerate(meta1['city_id'])}
    city_to_idx2 = {city_id: idx for idx, city_id in enumerate(meta2['city_id'])}

    # Extract aligned representations
    indices1 = [city_to_idx1[city_id] for city_id in common_cities]
    indices2 = [city_to_idx2[city_id] for city_id in common_cities]

    aligned_repr1 = repr1[indices1]
    aligned_repr2 = repr2[indices2]

    return aligned_repr1, aligned_repr2, common_cities


def apply_city_filter(metadata: pd.DataFrame, filter_str: str) -> pd.DataFrame:
    """
    Apply city filter to metadata DataFrame.

    Filter format: "region:REGEX && city_id:REGEX"

    Args:
        metadata: DataFrame with columns ['city_id', 'city_name', 'region']
        filter_str: Filter string with regex patterns

    Returns:
        Filtered DataFrame
    """
    # Parse filter string
    filters = {}
    for part in filter_str.split('&&'):
        part = part.strip()
        if ':' in part:
            field, pattern = part.split(':', 1)
            filters[field.strip()] = pattern.strip()

    # Apply filters
    mask = pd.Series([True] * len(metadata))

    if 'region' in filters:
        mask &= metadata['region'].str.match(filters['region'])

    if 'city_id' in filters:
        mask &= metadata['city_id'].astype(str).str.match(filters['city_id'])

    return metadata[mask]
