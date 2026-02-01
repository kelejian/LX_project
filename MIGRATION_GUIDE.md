# Migration Guide: Transitioning to Unified LX_project Architecture

## Overview

This guide helps you migrate your existing code to use the new unified LX_project architecture with shared modules.

## What Changed?

### New Shared Modules

The restructured architecture introduces a `shared/` directory containing common utilities:

```
shared/
├── data/                    # Data handling utilities
│   ├── data_loader.py      # NPZ file loading, case ID validation
│   └── preprocessing.py    # Normalization, scaling
├── utils/                   # Common utilities
│   ├── injury_metrics.py   # AIS calculations (moved from InjuryPredict)
│   └── random_seed.py      # Random seed management
└── config/                  # Shared configuration
```

## Migration Steps

### Step 1: Update Imports

#### For InjuryPredict Project

**Old imports:**
```python
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from utils.set_random_seed import set_random_seed, GLOBAL_SEED
```

**New imports (Option 1 - Use shared modules):**
```python
from shared.utils import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from shared.utils import set_random_seed, GLOBAL_SEED
```

**New imports (Option 2 - Keep local imports):**
```python
# No changes needed - local utils still work
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck
from utils.set_random_seed import set_random_seed, GLOBAL_SEED
```

#### For PulsePredict Project

**For data preparation:**
```python
# Old
from utils.data_prepare import package_pulse_data

# New - can use shared utilities
from shared.data import load_npz_data, save_npz_data, split_train_test
from shared.data import normalize_waveform_data, save_preprocessors
```

### Step 2: Update File Paths

When working from sub-project directories, you may need to adjust import paths:

```python
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Now you can import shared modules
from shared.data import load_npz_data
from shared.utils import AIS_cal_head
```

### Step 3: Backward Compatibility

**Good news!** The original local utilities in each sub-project remain unchanged. You can:
- **Continue using existing code** without any modifications
- **Gradually migrate** to shared modules as needed
- **Mix both approaches** during the transition period

## Common Migration Patterns

### Pattern 1: Data Loading and Preprocessing

**Before:**
```python
# Custom data loading in each project
data = np.load('data.npz')
case_ids = data['case_ids']
waveforms = data['waveforms']

# Custom normalization
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
normalized = scaler.fit_transform(waveforms.reshape(-1, waveforms.shape[-1]))
```

**After:**
```python
from shared.data import load_npz_data, normalize_waveform_data

# Unified data loading
data = load_npz_data('data.npz')

# Unified normalization
normalized, scaler = normalize_waveform_data(
    data['waveforms'], 
    method='minmax', 
    fit=True
)
```

### Pattern 2: AIS Calculations

**Before (in InjuryPredict):**
```python
from utils.AIS_cal import AIS_cal_head, AIS_cal_chest, AIS_cal_neck

head_ais = AIS_cal_head(hic_values)
chest_ais = AIS_cal_chest(dmax_values, ot_values)
neck_ais = AIS_cal_neck(nij_values)
```

**After (can be used in any project):**
```python
from shared.utils import AIS_cal_head, AIS_cal_chest, AIS_cal_neck

# Same API, now available across all projects
head_ais = AIS_cal_head(hic_values)
chest_ais = AIS_cal_chest(dmax_values, ot_values)
neck_ais = AIS_cal_neck(nij_values)
```

### Pattern 3: Random Seed Management

**Before:**
```python
from utils.set_random_seed import set_random_seed, GLOBAL_SEED
set_random_seed(GLOBAL_SEED)
```

**After:**
```python
from shared.utils import set_random_seed, GLOBAL_SEED
set_random_seed(GLOBAL_SEED)
# Now works with or without PyTorch installed!
```

## Project-Specific Migration

### InjuryPredict Migration

Files that may benefit from using shared modules:
- `train.py` - Can use `shared.utils.set_random_seed`
- `eval_model.py` - Can use shared AIS functions
- `utils/data_package.py` - Can use shared data loading utilities
- `utils/dataset_prepare.py` - Can use shared preprocessing utilities

**Minimal changes approach:**
Keep existing code as-is. The shared modules provide additional options but don't require changes.

**Gradual adoption approach:**
1. Update imports in new code to use shared modules
2. Refactor existing code over time
3. Eventually deprecate local copies if desired

### PulsePredict Migration

Files that may benefit:
- `utils/data_prepare.py` - Can use `shared.data` functions
- Data normalization code - Can use `shared.data.preprocessing`

**Note:** PulsePredict has a well-structured template, so migration is optional.

### ARS_optim Migration

The ARS_optim project can now easily:
- Use shared injury metrics for evaluation
- Use shared data loading utilities
- Leverage common preprocessing functions

## Benefits of Migration

1. **Code Reuse**: Write data processing code once, use everywhere
2. **Consistency**: Same data handling across all projects
3. **Maintainability**: Bug fixes in shared code benefit all projects
4. **Flexibility**: Easy to add new projects that leverage existing utilities
5. **Documentation**: Centralized documentation for common functions

## Testing After Migration

### Unit Tests
```python
# Test shared modules
python examples_data_module.py
python examples_utils_module.py
```

### Integration Tests
```python
# Test in InjuryPredict
cd InjuryPredict
python -c "from shared.utils import AIS_cal_head; print(AIS_cal_head(700))"

# Test in PulsePredict
cd PulsePredict
python -c "from shared.data import load_npz_data; print('Import successful')"
```

### Run Existing Tests
Make sure your existing tests still pass:
```bash
# InjuryPredict
cd InjuryPredict
python train.py  # Should work without changes

# PulsePredict
cd PulsePredict
python train.py -c config.json  # Should work without changes
```

## Troubleshooting

### Import Error: "No module named 'shared'"

**Solution 1:** Add parent directory to Python path
```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
```

**Solution 2:** Set PYTHONPATH environment variable
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/LX_project"
```

**Solution 3:** Install as editable package
```bash
cd /path/to/LX_project
pip install -e .
```

### Import Error: "No module named 'torch'"

The shared `random_seed` module now gracefully handles missing PyTorch. It will:
- Use PyTorch if available
- Fall back to numpy/random if PyTorch is not installed

No action needed!

## Best Practices

1. **Start Small**: Begin by using shared modules in new code
2. **Test Thoroughly**: Validate that shared modules work for your use case
3. **Keep Backups**: Original local utilities remain available
4. **Report Issues**: If you find bugs in shared modules, report them
5. **Contribute Back**: If you add useful utilities, consider contributing to shared/

## FAQ

**Q: Do I have to migrate immediately?**
A: No! The original code continues to work. Migration is optional and can be gradual.

**Q: Can I still use local utilities?**
A: Yes! Local utilities in each sub-project remain unchanged.

**Q: What if I need project-specific functionality?**
A: Keep it in your project's local `utils/`. Shared modules are for truly common functionality.

**Q: Will this affect my trained models?**
A: No! Model files and training logic are unchanged. Only helper utilities are shared.

**Q: How do I contribute new shared utilities?**
A: Add them to the appropriate `shared/` subdirectory and update the `__init__.py` file.

## Getting Help

- Check the [main README.md](README.md) for architecture overview
- Review [examples_data_module.py](examples_data_module.py) for data utilities
- Review [examples_utils_module.py](examples_utils_module.py) for injury metrics
- Consult sub-project README files for project-specific guidance

## Summary

The unified architecture provides:
✅ Shared utilities for common tasks
✅ Backward compatibility with existing code
✅ Gradual migration path
✅ Better code reuse and maintainability

**Remember:** Migration is optional! Your existing code continues to work as-is.
