# API Reference: Shared Modules

## Overview

This document provides detailed API documentation for the shared modules in LX_project.

## Table of Contents

- [shared.data](#shareddata)
  - [Data Loading](#data-loading)
  - [Preprocessing](#preprocessing)
- [shared.utils](#sharedutils)
  - [Injury Metrics](#injury-metrics)
  - [Random Seed](#random-seed)

---

## shared.data

### Data Loading

#### `load_npz_data(filepath)`

Load data from .npz file.

**Parameters:**
- `filepath` (str or Path): Path to .npz file

**Returns:**
- `dict`: Dictionary containing arrays from the .npz file

**Example:**
```python
from shared.data import load_npz_data

data = load_npz_data('packaged_data.npz')
print(data.keys())  # ['case_ids', 'params', 'waveforms']
```

---

#### `validate_case_ids(input_ids, label_ids)`

Validate that case IDs match between input and label files.

**Parameters:**
- `input_ids` (np.ndarray): Case IDs from input file
- `label_ids` (np.ndarray): Case IDs from label file

**Returns:**
- `bool`: True if IDs match

**Raises:**
- `AssertionError`: If IDs don't match

**Example:**
```python
from shared.data import validate_case_ids

validate_case_ids(input_data['case_ids'], label_data['case_ids'])
```

---

#### `load_params_file(params_path)`

Load parameters from .csv or .npz file and return as DataFrame.

**Parameters:**
- `params_path` (str or Path): Path to parameters file (.csv or .npz)

**Returns:**
- `pd.DataFrame`: DataFrame with case parameters

**Example:**
```python
from shared.data import load_params_file

params_df = load_params_file('parameters.csv')
```

---

#### `split_train_test(case_ids, train_ratio=0.86, shuffle=True, random_seed=42)`

Split case IDs into training and test sets.

**Parameters:**
- `case_ids` (list or np.ndarray): List or array of case IDs
- `train_ratio` (float, default=0.86): Ratio of training data
- `shuffle` (bool, default=True): Whether to shuffle before splitting
- `random_seed` (int, default=42): Random seed for reproducibility

**Returns:**
- `tuple`: (train_case_ids, test_case_ids)

**Example:**
```python
from shared.data import split_train_test

train_ids, test_ids = split_train_test(
    case_ids=[1, 2, 3, 4, 5],
    train_ratio=0.8,
    shuffle=True
)
```

---

#### `save_npz_data(filepath, **arrays)`

Save arrays to .npz file.

**Parameters:**
- `filepath` (str or Path): Output path for .npz file
- `**arrays`: Named arrays to save

**Example:**
```python
from shared.data import save_npz_data

save_npz_data(
    'output.npz',
    case_ids=case_ids,
    params=params,
    waveforms=waveforms
)
```

---

### Preprocessing

#### `create_scaler(method='minmax')`

Create a scaler instance based on the specified method.

**Parameters:**
- `method` (str, default='minmax'): Scaling method ('minmax', 'maxabs', or 'standard')

**Returns:**
- Scaler instance from sklearn.preprocessing

**Example:**
```python
from shared.data import create_scaler

scaler = create_scaler(method='minmax')
```

---

#### `normalize_waveform_data(waveforms, scaler=None, method='minmax', fit=True)`

Normalize waveform data.

**Parameters:**
- `waveforms` (np.ndarray): Input waveforms array, shape (N, channels, time_steps)
- `scaler` (object, optional): Pre-fitted scaler (if None, creates new one)
- `method` (str, default='minmax'): Scaling method ('minmax', 'maxabs', 'standard')
- `fit` (bool, default=True): Whether to fit the scaler (True for training, False for test/val)

**Returns:**
- `tuple`: (normalized_waveforms, scaler)

**Example:**
```python
from shared.data import normalize_waveform_data

# Training data - fit scaler
train_norm, scaler = normalize_waveform_data(
    train_waveforms,
    method='minmax',
    fit=True
)

# Test data - use fitted scaler
test_norm, _ = normalize_waveform_data(
    test_waveforms,
    scaler=scaler,
    fit=False
)
```

---

#### `normalize_features(features, scaler=None, method='minmax', fit=True)`

Normalize feature data.

**Parameters:**
- `features` (np.ndarray): Input features array, shape (N, feature_dim)
- `scaler` (object, optional): Pre-fitted scaler (if None, creates new one)
- `method` (str, default='minmax'): Scaling method ('minmax', 'maxabs', 'standard')
- `fit` (bool, default=True): Whether to fit the scaler (True for training, False for test/val)

**Returns:**
- `tuple`: (normalized_features, scaler)

**Example:**
```python
from shared.data import normalize_features

features_norm, scaler = normalize_features(
    features,
    method='standard',
    fit=True
)
```

---

#### `save_preprocessors(filepath, **preprocessors)`

Save preprocessing objects (scalers, encoders, etc.) to file.

**Parameters:**
- `filepath` (str or Path): Output path for preprocessor file
- `**preprocessors`: Named preprocessing objects to save

**Example:**
```python
from shared.data import save_preprocessors

save_preprocessors(
    'preprocessors.joblib',
    waveform_scaler=wave_scaler,
    feature_scaler=feat_scaler
)
```

---

#### `load_preprocessors(filepath)`

Load preprocessing objects from file.

**Parameters:**
- `filepath` (str or Path): Path to preprocessor file

**Returns:**
- `dict`: Dictionary of preprocessing objects

**Example:**
```python
from shared.data import load_preprocessors

preprocessors = load_preprocessors('preprocessors.joblib')
wave_scaler = preprocessors['waveform_scaler']
```

---

#### `clip_values(data, min_val, max_val)`

Clip array values to specified range.

**Parameters:**
- `data` (np.ndarray): Input array
- `min_val` (float): Minimum value
- `max_val` (float): Maximum value

**Returns:**
- `np.ndarray`: Clipped array

**Example:**
```python
from shared.data import clip_values

clipped = clip_values(data, min_val=0, max_val=100)
```

---

## shared.utils

### Injury Metrics

#### `AIS_cal_head(HIC15, prob_thresholds=[0.01, 0.05, 0.1, 0.2, 0.3])`

Calculate head AIS level from HIC15 value.

Uses formula: P_head(AIS3+) = Φ((ln(HIC15) - 7.45231) / 0.73998)
where Φ is the cumulative normal distribution function.

**Parameters:**
- `HIC15` (float or np.ndarray): HIC15 value(s)
- `prob_thresholds` (list, optional): Probability thresholds for AIS level classification

**Returns:**
- `int or np.ndarray`: AIS level(s), same shape as input

**AIS Interpretation:**
- 0: No injury
- 1: Minor
- 2: Moderate
- 3: Serious
- 4: Severe
- 5: Critical

**Example:**
```python
from shared.utils import AIS_cal_head

# Single value
ais = AIS_cal_head(850.0)  # Returns: 3

# Batch calculation
hic_values = np.array([300, 700, 1000, 1500])
ais_levels = AIS_cal_head(hic_values)  # Returns: array([1, 3, 4, 5])
```

---

#### `AIS_cal_chest(Dmax, OT, prob_thresholds=[0.02, 0.06, 0.15, 0.25, 0.4])`

Calculate chest AIS level from Dmax (chest deflection).

Uses formula: P_chest_defl(AIS3+) = 1 / (1 + e^(10.5456 - 1.568 * ChestDefl^0.4612))

**Parameters:**
- `Dmax` (float or np.ndarray): Maximum chest deflection in mm
- `OT` (int or np.ndarray): Occupant type
  - 1: 5th percentile female
  - 2: 50th percentile male
  - 3: 95th percentile male
- `prob_thresholds` (list, optional): Probability thresholds for AIS level classification

**Returns:**
- `int or np.ndarray`: AIS level(s), same shape as input

**Example:**
```python
from shared.utils import AIS_cal_chest

# Single value
ais = AIS_cal_chest(Dmax=45.0, OT=2)  # Returns: 3

# Batch calculation
dmax_values = np.array([30, 40, 50, 60])
ot_values = np.array([2, 2, 2, 2])
ais_levels = AIS_cal_chest(dmax_values, ot_values)
```

---

#### `AIS_cal_neck(Nij, prob_thresholds=[0.06, 0.1, 0.2, 0.3, 0.4])`

Calculate neck AIS level from Nij (neck injury criterion).

Uses formula: P_neck_Nij(AIS3+) = 1 / (1 + e^(3.2269 - 1.9688 * Nij))

**Parameters:**
- `Nij` (float or np.ndarray): Nij value(s)
- `prob_thresholds` (list, optional): Probability thresholds for AIS level classification

**Returns:**
- `int or np.ndarray`: AIS level(s), same shape as input

**Example:**
```python
from shared.utils import AIS_cal_neck

# Single value
ais = AIS_cal_neck(0.8)  # Returns: 2

# Batch calculation
nij_values = np.array([0.3, 0.7, 1.0, 1.5])
ais_levels = AIS_cal_neck(nij_values)
```

---

### Random Seed

#### `set_random_seed(seed=123)`

Set global random seed for reproducibility.

Sets seeds for:
- NumPy
- Python's random module
- PyTorch (if installed)
- CUDA (if PyTorch and CUDA available)

**Parameters:**
- `seed` (int, default=123): Random seed value

**Returns:**
- None

**Example:**
```python
from shared.utils import set_random_seed, GLOBAL_SEED

# Use default seed
set_random_seed()

# Use custom seed
set_random_seed(42)

# Use global constant
set_random_seed(GLOBAL_SEED)
```

---

#### `GLOBAL_SEED`

Global constant for the default random seed value.

**Value:** 123

**Example:**
```python
from shared.utils import GLOBAL_SEED

print(GLOBAL_SEED)  # 123
```

---

## Complete Usage Example

```python
"""
Complete workflow using shared modules
"""
import numpy as np
from shared.data import (
    load_npz_data,
    split_train_test,
    normalize_waveform_data,
    normalize_features,
    save_preprocessors,
    save_npz_data
)
from shared.utils import (
    AIS_cal_head,
    AIS_cal_chest,
    AIS_cal_neck,
    set_random_seed
)

# Set random seed for reproducibility
set_random_seed(42)

# Load data
data = load_npz_data('raw_data.npz')
case_ids = data['case_ids']
waveforms = data['waveforms']
params = data['params']

# Split data
train_ids, test_ids = split_train_test(case_ids, train_ratio=0.8)

# Create masks
train_mask = np.isin(case_ids, train_ids)
test_mask = np.isin(case_ids, test_ids)

# Normalize training data
train_wave_norm, wave_scaler = normalize_waveform_data(
    waveforms[train_mask],
    method='minmax',
    fit=True
)
train_params_norm, param_scaler = normalize_features(
    params[train_mask],
    method='minmax',
    fit=True
)

# Normalize test data
test_wave_norm, _ = normalize_waveform_data(
    waveforms[test_mask],
    scaler=wave_scaler,
    fit=False
)
test_params_norm, _ = normalize_features(
    params[test_mask],
    scaler=param_scaler,
    fit=False
)

# Save preprocessors
save_preprocessors(
    'preprocessors.joblib',
    waveform_scaler=wave_scaler,
    param_scaler=param_scaler
)

# Save processed data
save_npz_data(
    'train_processed.npz',
    case_ids=case_ids[train_mask],
    waveforms=train_wave_norm,
    params=train_params_norm
)

# Calculate injury metrics (example)
hic_values = np.array([700, 1000, 1500])
dmax_values = np.array([40, 50, 60])
nij_values = np.array([0.8, 1.0, 1.5])
ot = 2  # 50th male

head_ais = AIS_cal_head(hic_values)
chest_ais = AIS_cal_chest(dmax_values, ot)
neck_ais = AIS_cal_neck(nij_values)

print(f"Head AIS: {head_ais}")
print(f"Chest AIS: {chest_ais}")
print(f"Neck AIS: {neck_ais}")
```

---

## Version History

- **v1.0.0** (2026-02-01): Initial release
  - Data loading and preprocessing utilities
  - Injury metrics (AIS calculations)
  - Random seed management

---

## Support

For questions or issues with the shared modules, please refer to:
- [Main README](README.md)
- [Migration Guide](MIGRATION_GUIDE.md)
- [Usage Examples](examples_data_module.py) and [examples_utils_module.py](examples_utils_module.py)
