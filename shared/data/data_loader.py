"""
Common Data Loading Utilities
==============================

Provides unified data loading functions for .npz format files
used across all LX_project sub-projects.
"""

import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Union, Optional


def load_npz_data(filepath: Union[str, Path]) -> Dict[str, np.ndarray]:
    """
    Load data from .npz file.
    
    Args:
        filepath: Path to .npz file
        
    Returns:
        Dictionary containing arrays from the .npz file
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    data = np.load(filepath)
    return {key: data[key] for key in data.files}


def validate_case_ids(input_ids: np.ndarray, label_ids: np.ndarray) -> bool:
    """
    Validate that case IDs match between input and label files.
    
    Args:
        input_ids: Case IDs from input file
        label_ids: Case IDs from label file
        
    Returns:
        True if IDs match, raises AssertionError otherwise
    """
    assert np.array_equal(input_ids, label_ids), (
        f"Case ID mismatch: input has {input_ids[:5]}... vs labels has {label_ids[:5]}..."
    )
    return True


def load_params_file(params_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load parameters from .csv or .npz file and return as DataFrame.
    
    Args:
        params_path: Path to parameters file (.csv or .npz)
        
    Returns:
        DataFrame with case parameters indexed by case_id
    """
    params_path = Path(params_path)
    
    if params_path.suffix == '.csv':
        df = pd.read_csv(params_path)
    elif params_path.suffix == '.npz':
        data = np.load(params_path)
        df = pd.DataFrame({key: data[key] for key in data.files})
    else:
        raise ValueError(f"Parameter file must be .csv or .npz format, got: {params_path.suffix}")
    
    return df


def split_train_test(case_ids: Union[List, np.ndarray], 
                     train_ratio: float = 0.86,
                     shuffle: bool = True,
                     random_seed: int = 42) -> Tuple[List, List]:
    """
    Split case IDs into training and test sets.
    
    Args:
        case_ids: List or array of case IDs
        train_ratio: Ratio of training data (default: 0.86)
        shuffle: Whether to shuffle before splitting
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_case_ids, test_case_ids)
    """
    case_ids = list(case_ids)
    
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(case_ids)
    
    num_train = int(len(case_ids) * train_ratio)
    train_ids = case_ids[:num_train]
    test_ids = case_ids[num_train:]
    
    return train_ids, test_ids


def save_npz_data(filepath: Union[str, Path], **arrays) -> None:
    """
    Save arrays to .npz file.
    
    Args:
        filepath: Output path for .npz file
        **arrays: Named arrays to save
    """
    filepath = Path(filepath)
    os.makedirs(filepath.parent, exist_ok=True)
    np.savez(filepath, **arrays)
