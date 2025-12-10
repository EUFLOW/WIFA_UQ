# Global Calibration

Global calibration finds a **single set of parameters** that works best across all flow conditions. This is the simplest calibration approach and is recommended as a starting point.

## Overview

Global calibration answers the question: "What single parameter values minimize overall model bias?"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Global Calibration                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Model Error Database                                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  bias[sample=0, case] = [0.05, 0.03, -0.02, 0.08, ...]              │    │
│  │  bias[sample=1, case] = [0.02, 0.01, -0.01, 0.04, ...]  ← best?    │    │
│  │  bias[sample=2, case] = [0.07, 0.05, -0.03, 0.10, ...]              │    │
│  │  ...                                                                │    │
│  │  bias[sample=99, case] = [0.04, 0.02, -0.02, 0.06, ...]             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Find sample s* that minimizes Σ|bias[s, case]|                     │    │
│  │                                                                     │    │
│  │  Result: s* = 1  →  k_b* = 0.042, ss_alpha* = 0.91                  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Output: Single optimal parameter set for ALL conditions                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## When to Use Global Calibration

**Recommended when:**

- You have limited reference data (< 50-100 flow cases)
- Atmospheric conditions are relatively homogeneous
- Deployment simplicity is important (single parameter set)
- As a baseline before trying local calibration
- You want interpretable, physics-based calibration

**Consider local calibration instead when:**

- You have abundant data (> 100 flow cases)
- Conditions vary significantly (different stability regimes, seasons)
- Global calibration leaves systematic patterns in residuals

## Available Calibrators

| Calibrator | Description | Use Case |
|------------|-------------|----------|
| `MinBiasCalibrator` | Finds parameters minimizing total absolute bias | Default choice |
| `DefaultParams` | Uses literature/default parameter values | Baseline comparison |

---

## MinBiasCalibrator

The default and recommended global calibrator. Searches through all parameter samples to find the one with minimum total absolute bias.

### Algorithm

```python
# For each sample s in the database:
total_bias[s] = Σ |bias[s, case]| for all cases

# Select the sample with minimum total bias:
s* = argmin(total_bias)

# Extract parameters at that sample:
best_params = {k_b: database.k_b[s*], ss_alpha: database.ss_alpha[s*], ...}
```

### Configuration

```yaml
error_prediction:
  calibrator: MinBiasCalibrator
  # No additional parameters needed
```

### Example Output

```
Calibration Results:
  Best sample index: 42
  Best parameters:
    k_b: 0.0423
    ss_alpha: 0.912
  Total absolute bias at optimum: 1.23
```

### API Usage

```python
from wifa_uq.postprocessing.calibration import MinBiasCalibrator
import xarray as xr

# Load database
database = xr.load_dataset("results_stacked_hh.nc")

# Initialize and fit calibrator
calibrator = MinBiasCalibrator(database)
calibrator.fit()

# Access results
print(f"Best sample index: {calibrator.best_idx_}")
print(f"Best parameters: {calibrator.best_params_}")
# Output: {'k_b': 0.0423, 'ss_alpha': 0.912}
```

### Properties

After calling `fit()`:

| Property | Type | Description |
|----------|------|-------------|
| `best_idx_` | int | Index of optimal sample in database |
| `best_params_` | dict | Parameter values at optimal sample |
| `swept_params` | list | Names of swept parameters |

### How It Works with Cross-Validation

In cross-validation, `MinBiasCalibrator` is fit on the training fold only:

```python
# Inside run_cross_validation:
for train_idx, test_idx in cv_splits:
    dataset_train = database.isel(case_index=train_idx)
    dataset_test = database.isel(case_index=test_idx)

    # Calibrator sees only training data
    calibrator = MinBiasCalibrator(dataset_train)
    calibrator.fit()

    # Apply calibrated parameters to test data
    test_bias = dataset_test["model_bias_cap"].sel(sample=calibrator.best_idx_)
```

This ensures proper out-of-sample evaluation.

---

## DefaultParams

Uses the default parameter values specified in the database metadata. Useful as a baseline to quantify the improvement from calibration.

### Algorithm

```python
# Get default values from database metadata
defaults = database.attrs["param_defaults"]  # {"k_b": 0.04, "ss_alpha": 0.875}

# Find sample closest to these defaults
distances = Σ (param_values - default_values)² for each sample
s* = argmin(distances)
```

### Configuration

```yaml
error_prediction:
  calibrator: DefaultParams
```

### API Usage

```python
from wifa_uq.postprocessing.calibration import DefaultParams
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

calibrator = DefaultParams(database)
calibrator.fit()

print(f"Default sample index: {calibrator.best_idx_}")
print(f"Parameters at default: {calibrator.best_params_}")
```

### Use Cases

1. **Baseline comparison**: Compare MinBiasCalibrator improvement over defaults

   ```python
   # Default calibration
   default_cal = DefaultParams(database)
   default_cal.fit()
   default_rmse = compute_rmse(database, default_cal.best_idx_)

   # Optimized calibration
   minbias_cal = MinBiasCalibrator(database)
   minbias_cal.fit()
   optimized_rmse = compute_rmse(database, minbias_cal.best_idx_)

   improvement = (default_rmse - optimized_rmse) / default_rmse * 100
   print(f"Calibration improved RMSE by {improvement:.1f}%")
   ```

2. **Sanity check**: Verify that calibration actually helps

3. **Publication baseline**: Report results with both default and calibrated parameters

---

## Comparing Calibrators

### Workflow Configuration

Run both calibrators and compare:

```yaml
# config_default.yaml
error_prediction:
  calibrator: DefaultParams
  # ... other settings

# config_minbias.yaml
error_prediction:
  calibrator: MinBiasCalibrator
  # ... other settings
```

### Programmatic Comparison

```python
from wifa_uq.postprocessing.calibration import MinBiasCalibrator, DefaultParams
import numpy as np

database = xr.load_dataset("results_stacked_hh.nc")

# Fit both calibrators
default_cal = DefaultParams(database)
default_cal.fit()

minbias_cal = MinBiasCalibrator(database)
minbias_cal.fit()

# Compare total absolute bias
def total_abs_bias(db, sample_idx):
    return np.abs(db["model_bias_cap"].sel(sample=sample_idx)).sum().values

default_bias = total_abs_bias(database, default_cal.best_idx_)
minbias_bias = total_abs_bias(database, minbias_cal.best_idx_)

print(f"Default params total |bias|: {default_bias:.4f}")
print(f"MinBias params total |bias|: {minbias_bias:.4f}")
print(f"Reduction: {(1 - minbias_bias/default_bias)*100:.1f}%")
```

---

## Understanding Results

### What "Best" Means

MinBiasCalibrator minimizes **total absolute bias**, which:

- Treats all flow cases equally
- Penalizes both positive and negative errors
- May compromise on some conditions to improve overall performance

### Examining Bias Distribution

After calibration, inspect the residual bias distribution:

```python
import matplotlib.pyplot as plt

calibrator = MinBiasCalibrator(database)
calibrator.fit()

# Get bias at calibrated parameters
calibrated_bias = database["model_bias_cap"].sel(sample=calibrator.best_idx_)

# Plot distribution
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.hist(calibrated_bias.values, bins=30, edgecolor='black')
plt.xlabel("Bias (normalized)")
plt.ylabel("Count")
plt.title("Bias Distribution After Calibration")
plt.axvline(0, color='red', linestyle='--', label='Zero bias')
plt.legend()

plt.subplot(1, 2, 2)
# Check for systematic patterns
abl_heights = database["ABL_height"].isel(sample=0).values
plt.scatter(abl_heights, calibrated_bias.values, alpha=0.5)
plt.xlabel("ABL Height (m)")
plt.ylabel("Residual Bias")
plt.title("Bias vs ABL Height")
plt.axhline(0, color='red', linestyle='--')

plt.tight_layout()
plt.savefig("calibration_diagnostics.png")
```

### Signs You May Need Local Calibration

If after global calibration you observe:

- **Systematic patterns** in residuals vs. features (e.g., bias correlates with ABL height)
- **Bimodal bias distribution** suggesting different optimal parameters for different conditions
- **Poor performance on specific subsets** (e.g., stable vs. unstable conditions)

Then consider [Local Calibration](local_calibration.md).

---

## Integration with Bias Prediction

Global calibration is the first stage of a two-stage pipeline:

```
Stage 1: Global Calibration
    Find θ* = {k_b*, ss_alpha*} minimizing total |bias|

Stage 2: Bias Prediction (ML)
    Learn residual_bias = f(ABL_height, wind_veer, ...)
    at the calibrated parameters

Final Output:
    corrected_power = model(θ*) - predicted_residual_bias
```

The bias predictor operates on the residual bias **after** calibration, learning patterns that calibration alone cannot capture.

---

## Best Practices

### 1. Always Compare to Baseline

```yaml
# Run with DefaultParams first to establish baseline
error_prediction:
  calibrator: DefaultParams
```

### 2. Check Parameter Values are Physical

After calibration, verify parameters are within reasonable ranges:

```python
calibrator = MinBiasCalibrator(database)
calibrator.fit()

# k_b should typically be 0.01-0.07 for wake expansion
assert 0.01 <= calibrator.best_params_["k_b"] <= 0.07, "k_b outside expected range"
```

### 3. Examine Edge Cases

If the optimal sample is at the boundary of the parameter range, consider expanding the range:

```python
k_b_range = [database.k_b.min().values, database.k_b.max().values]
optimal_k_b = calibrator.best_params_["k_b"]

if optimal_k_b == k_b_range[0] or optimal_k_b == k_b_range[1]:
    print("WARNING: Optimal k_b is at boundary. Consider expanding param_config range.")
```

### 4. Use Sufficient Samples

With too few samples, you may miss the true optimum:

| n_samples | Coverage Quality |
|-----------|------------------|
| 20-50 | Testing only |
| 100 | Standard |
| 200+ | High-fidelity |

---

## Troubleshooting

### "All samples have similar bias"

**Cause:** Parameters may not significantly affect bias for your dataset.

**Solutions:**
- Check that swept parameters are actually used by the wake model
- Verify parameter ranges are wide enough to see differences
- Examine if reference data has high noise masking parameter effects

### "Optimal parameters seem unrealistic"

**Cause:** The optimization is fitting noise or artifacts in the data.

**Solutions:**
- Increase n_samples for better parameter space coverage
- Check reference data quality
- Consider if the wake model physics applies to your conditions

### "Cross-validation shows high variance"

**Cause:** Optimal parameters vary significantly across CV folds.

**Solutions:**
- This suggests condition-dependent optimal parameters → try local calibration
- Ensure sufficient data in each fold
- Check for outliers in reference data

---

## See Also

- [Local Calibration](local_calibration.md) — Condition-dependent parameter prediction
- [Calibration Theory](../../concepts/calibration_theory.md) — Mathematical foundations
- [Configuration Reference](../configuration.md) — Full YAML options
- [Cross-Validation](../error_prediction/cross_validation.md) — Validation strategies
