"""
Example Usage: Shared Data Module
==================================

This script demonstrates how to use the shared data module
for loading and preprocessing data across LX_project sub-projects.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path to import shared modules
sys.path.insert(0, str(Path(__file__).parent))

from shared.data import (
    load_npz_data,
    validate_case_ids,
    load_params_file,
    split_train_test,
    save_npz_data,
    normalize_waveform_data,
    normalize_features,
    save_preprocessors,
    load_preprocessors,
)

def example_data_loading():
    """Example: Loading and validating data"""
    print("=" * 60)
    print("Example 1: Data Loading and Validation")
    print("=" * 60)
    
    # Create sample data for demonstration
    sample_case_ids = np.array([1, 2, 3, 4, 5])
    sample_params = np.random.randn(5, 3)
    sample_waveforms = np.random.randn(5, 3, 150)
    
    # Save sample data
    save_npz_data(
        '/tmp/sample_data.npz',
        case_ids=sample_case_ids,
        params=sample_params,
        waveforms=sample_waveforms
    )
    print("✓ Sample data created and saved")
    
    # Load data
    data = load_npz_data('/tmp/sample_data.npz')
    print(f"✓ Data loaded: {list(data.keys())}")
    print(f"  - case_ids shape: {data['case_ids'].shape}")
    print(f"  - params shape: {data['params'].shape}")
    print(f"  - waveforms shape: {data['waveforms'].shape}")
    
    # Validate case IDs
    validate_case_ids(data['case_ids'], data['case_ids'])
    print("✓ Case IDs validated successfully")
    print()


def example_data_splitting():
    """Example: Splitting data into train/test sets"""
    print("=" * 60)
    print("Example 2: Train/Test Split")
    print("=" * 60)
    
    case_ids = np.arange(1, 101)  # 100 cases
    
    train_ids, test_ids = split_train_test(
        case_ids,
        train_ratio=0.8,
        shuffle=True,
        random_seed=42
    )
    
    print(f"✓ Total cases: {len(case_ids)}")
    print(f"✓ Training cases: {len(train_ids)} ({len(train_ids)/len(case_ids)*100:.1f}%)")
    print(f"✓ Test cases: {len(test_ids)} ({len(test_ids)/len(case_ids)*100:.1f}%)")
    print(f"  First 5 train IDs: {train_ids[:5]}")
    print(f"  First 5 test IDs: {test_ids[:5]}")
    print()


def example_normalization():
    """Example: Normalizing waveforms and features"""
    print("=" * 60)
    print("Example 3: Data Normalization")
    print("=" * 60)
    
    # Create sample waveforms (N=10, channels=3, time_steps=150)
    waveforms = np.random.randn(10, 3, 150) * 100 + 50
    print(f"Original waveforms shape: {waveforms.shape}")
    print(f"Original range: [{waveforms.min():.2f}, {waveforms.max():.2f}]")
    
    # Normalize waveforms
    normalized_waveforms, scaler = normalize_waveform_data(
        waveforms,
        method='minmax',
        fit=True
    )
    print(f"✓ Normalized waveforms shape: {normalized_waveforms.shape}")
    print(f"  Normalized range: [{normalized_waveforms.min():.2f}, {normalized_waveforms.max():.2f}]")
    
    # Create sample features (N=10, feature_dim=5)
    features = np.random.randn(10, 5) * 50 + 100
    print(f"\nOriginal features shape: {features.shape}")
    print(f"Original range: [{features.min():.2f}, {features.max():.2f}]")
    
    # Normalize features
    normalized_features, feature_scaler = normalize_features(
        features,
        method='minmax',
        fit=True
    )
    print(f"✓ Normalized features shape: {normalized_features.shape}")
    print(f"  Normalized range: [{normalized_features.min():.2f}, {normalized_features.max():.2f}]")
    
    # Save preprocessors
    save_preprocessors(
        '/tmp/preprocessors.joblib',
        waveform_scaler=scaler,
        feature_scaler=feature_scaler
    )
    print("✓ Preprocessors saved to /tmp/preprocessors.joblib")
    
    # Load preprocessors
    loaded_preprocessors = load_preprocessors('/tmp/preprocessors.joblib')
    print(f"✓ Preprocessors loaded: {list(loaded_preprocessors.keys())}")
    print()


def example_full_pipeline():
    """Example: Complete data processing pipeline"""
    print("=" * 60)
    print("Example 4: Complete Data Pipeline")
    print("=" * 60)
    
    # 1. Create sample raw data
    n_samples = 50
    case_ids = np.arange(1, n_samples + 1)
    params = np.random.randn(n_samples, 3) * 10 + 50
    waveforms = np.random.randn(n_samples, 3, 150) * 100
    
    print(f"Step 1: Created {n_samples} samples")
    
    # 2. Split into train/test
    train_ids, test_ids = split_train_test(case_ids, train_ratio=0.8)
    train_mask = np.isin(case_ids, train_ids)
    test_mask = np.isin(case_ids, test_ids)
    
    train_waveforms = waveforms[train_mask]
    test_waveforms = waveforms[test_mask]
    train_params = params[train_mask]
    test_params = params[test_mask]
    
    print(f"Step 2: Split data (train={len(train_ids)}, test={len(test_ids)})")
    
    # 3. Normalize training data (fit scalers)
    norm_train_waveforms, wave_scaler = normalize_waveform_data(
        train_waveforms, method='minmax', fit=True
    )
    norm_train_params, param_scaler = normalize_features(
        train_params, method='minmax', fit=True
    )
    
    print(f"Step 3: Normalized training data")
    print(f"  Waveforms: {norm_train_waveforms.shape}, range: [{norm_train_waveforms.min():.3f}, {norm_train_waveforms.max():.3f}]")
    print(f"  Params: {norm_train_params.shape}, range: [{norm_train_params.min():.3f}, {norm_train_params.max():.3f}]")
    
    # 4. Normalize test data (use fitted scalers)
    norm_test_waveforms, _ = normalize_waveform_data(
        test_waveforms, scaler=wave_scaler, fit=False
    )
    norm_test_params, _ = normalize_features(
        test_params, scaler=param_scaler, fit=False
    )
    
    print(f"Step 4: Normalized test data (using training scalers)")
    print(f"  Waveforms: {norm_test_waveforms.shape}, range: [{norm_test_waveforms.min():.3f}, {norm_test_waveforms.max():.3f}]")
    print(f"  Params: {norm_test_params.shape}, range: [{norm_test_params.min():.3f}, {norm_test_params.max():.3f}]")
    
    # 5. Save preprocessors
    save_preprocessors(
        '/tmp/pipeline_preprocessors.joblib',
        waveform_scaler=wave_scaler,
        param_scaler=param_scaler
    )
    
    # 6. Save processed data
    save_npz_data(
        '/tmp/processed_train.npz',
        case_ids=case_ids[train_mask],
        waveforms=norm_train_waveforms,
        params=norm_train_params
    )
    save_npz_data(
        '/tmp/processed_test.npz',
        case_ids=case_ids[test_mask],
        waveforms=norm_test_waveforms,
        params=norm_test_params
    )
    
    print(f"Step 5: Saved preprocessors and processed data")
    print("✓ Pipeline complete!")
    print()


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("LX_project Shared Data Module - Usage Examples")
    print("=" * 60 + "\n")
    
    example_data_loading()
    example_data_splitting()
    example_normalization()
    example_full_pipeline()
    
    print("=" * 60)
    print("All examples completed successfully! ✓")
    print("=" * 60)
