"""
Shared Data Handling Module
===========================

Contains utilities for:
- Data loading from .npz files
- Data preprocessing and normalization
- Common data structures
"""

from .data_loader import (
    load_npz_data,
    validate_case_ids,
    load_params_file,
    split_train_test,
    save_npz_data,
)

from .preprocessing import (
    create_scaler,
    normalize_waveform_data,
    normalize_features,
    save_preprocessors,
    load_preprocessors,
    clip_values,
)

__all__ = [
    # Data loading
    'load_npz_data',
    'validate_case_ids',
    'load_params_file',
    'split_train_test',
    'save_npz_data',
    # Preprocessing
    'create_scaler',
    'normalize_waveform_data',
    'normalize_features',
    'save_preprocessors',
    'load_preprocessors',
    'clip_values',
]
