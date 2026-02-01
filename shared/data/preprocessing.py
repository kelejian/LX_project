"""
Common Preprocessing Utilities
===============================

Provides unified preprocessing and normalization functions
for data used across all LX_project sub-projects.
"""

import numpy as np
from sklearn.preprocessing import MinMaxScaler, MaxAbsScaler, StandardScaler
from typing import Union, Tuple, Optional, Dict
import joblib
from pathlib import Path


def create_scaler(method: str = 'minmax'):
    """
    Create a scaler instance based on the specified method.
    
    Args:
        method: Scaling method ('minmax', 'maxabs', or 'standard')
        
    Returns:
        Scaler instance
    """
    if method == 'minmax':
        return MinMaxScaler()
    elif method == 'maxabs':
        return MaxAbsScaler()
    elif method == 'standard':
        return StandardScaler()
    else:
        raise ValueError(f"Unknown scaler method: {method}. Use 'minmax', 'maxabs', or 'standard'.")


def normalize_waveform_data(waveforms: np.ndarray, 
                            scaler: Optional[object] = None,
                            method: str = 'minmax',
                            fit: bool = True) -> Tuple[np.ndarray, object]:
    """
    Normalize waveform data.
    
    Args:
        waveforms: Input waveforms array, shape (N, channels, time_steps)
        scaler: Pre-fitted scaler (if None, creates new one)
        method: Scaling method ('minmax', 'maxabs', 'standard')
        fit: Whether to fit the scaler (True for training, False for test/val)
        
    Returns:
        Tuple of (normalized_waveforms, scaler)
    """
    original_shape = waveforms.shape
    
    # Reshape to (N * channels, time_steps) for normalization
    N, channels, time_steps = original_shape
    waveforms_reshaped = waveforms.reshape(-1, time_steps)
    
    if scaler is None:
        scaler = create_scaler(method)
    
    if fit:
        normalized = scaler.fit_transform(waveforms_reshaped)
    else:
        normalized = scaler.transform(waveforms_reshaped)
    
    # Reshape back to original shape
    normalized = normalized.reshape(original_shape)
    
    return normalized, scaler


def normalize_features(features: np.ndarray,
                       scaler: Optional[object] = None,
                       method: str = 'minmax',
                       fit: bool = True) -> Tuple[np.ndarray, object]:
    """
    Normalize feature data.
    
    Args:
        features: Input features array, shape (N, feature_dim)
        scaler: Pre-fitted scaler (if None, creates new one)
        method: Scaling method ('minmax', 'maxabs', 'standard')
        fit: Whether to fit the scaler (True for training, False for test/val)
        
    Returns:
        Tuple of (normalized_features, scaler)
    """
    if scaler is None:
        scaler = create_scaler(method)
    
    if fit:
        normalized = scaler.fit_transform(features)
    else:
        normalized = scaler.transform(features)
    
    return normalized, scaler


def save_preprocessors(filepath: Union[str, Path], **preprocessors) -> None:
    """
    Save preprocessing objects (scalers, encoders, etc.) to file.
    
    Args:
        filepath: Output path for preprocessor file
        **preprocessors: Named preprocessing objects to save
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessors, filepath)


def load_preprocessors(filepath: Union[str, Path]) -> Dict:
    """
    Load preprocessing objects from file.
    
    Args:
        filepath: Path to preprocessor file
        
    Returns:
        Dictionary of preprocessing objects
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Preprocessor file not found: {filepath}")
    
    return joblib.load(filepath)


def clip_values(data: np.ndarray, min_val: float, max_val: float) -> np.ndarray:
    """
    Clip array values to specified range.
    
    Args:
        data: Input array
        min_val: Minimum value
        max_val: Maximum value
        
    Returns:
        Clipped array
    """
    return np.clip(data, min_val, max_val)
