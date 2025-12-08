# Local Calibration

Local calibration predicts **optimal parameters for each flow case** based on atmospheric conditions. Instead of a single parameter set, an ML model learns how optimal parameters vary with features like ABL height, wind veer, and stability.

## Overview

Local calibration answers the question: "What parameters work best for *this specific* atmospheric condition?"

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          Local Calibration                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Training Phase:                                                            │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  For each flow case, find optimal sample:                           │    │
│  │                                                                     │    │
│  │  Case 1: ABL=500m, veer=0.01  →  optimal k_b=0.035, α=0.88         │    │
│  │  Case 2: ABL=800m, veer=0.02  →  optimal k_b=0.052, α=0.92         │    │
│  │  Case 3: ABL=300m, veer=0.00  →  optimal k_b=0.028, α=0.85         │    │
│  │  ...                                                                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                              │                                              │
│                              ▼                                              │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Train ML model:                                                    │    │
│  │                                                                     │    │
│  │  optimal_params = f(ABL_height, wind_veer, lapse_rate, ...)        │    │
│  │                                                                     │    │
│  │  Using: Ridge, RandomForest, XGBoost, etc.                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Prediction Phase:                                                          │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  New case: ABL=650m, veer=0.015                                     │    │
│  │                     │                                               │    │
│  │                     ▼                                               │    │
│  │  ML model predicts: k_b=0.044, α=0.90                               │    │
│  │                     │                                               │    │
│  │                     ▼                                               │    │
│  │  Find closest sample in database → use that for bias prediction    │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## When to Use Local Calibration

**Recommended when:**

- Atmospheric conditions vary significantly across your dataset
- Global calibration leaves systematic patterns in residuals
- You have sufficient data to train a reliable parameter predictor (> 100 cases recommended)
- Physical reasoning suggests parameters should vary with conditions

**Stay with global calibration when:**

- Limited reference data (< 50-100 cases)
- Conditions are relatively homogeneous
- Deployment simplicity is critical
- You're establishing a baseline

## How It Works

### Step 1: Find Per-Case Optimal Parameters

For each flow case in the training set, identify which parameter sample minimizes bias:

```python
for case_idx in range(n_cases):
    # Get bias across all samples for this case
    bias_values = database["model_bias_cap"].isel(case_index=case_idx)

    # Find sample with minimum |bias|
    best_sample_idx = argmin(|bias_values|)

    # Record optimal parameters for this case
    optimal_k_b[case_idx] = database.k_b[best_sample_idx]
    optimal_ss_alpha[case_idx] = database.ss_alpha[best_sample_idx]
```

### Step 2: Train Parameter Predictor

Train an ML model to predict optimal parameters from atmospheric features:

```python
# Features (same for all samples - they're physical conditions)
X = database[["ABL_height", "wind_veer", "lapse_rate"]].isel(sample=0)

# Targets (optimal parameter values found in step 1)
y = [optimal_k_b, optimal_ss_alpha]

# Train regressor
regressor.fit(X, y)
```

### Step 3: Predict for New Cases

For test/new cases, predict optimal parameters and find the closest sample:

```python
# Predict optimal parameters
predicted_params = regressor.predict(X_new)

# Find database sample closest to predicted values
closest_sample_idx = find_closest_sample(database, predicted_params)

# Use that sample's bias for prediction
test_bias = database["model_bias_cap"].sel(sample=closest_sample_idx)
```

## Configuration

```yaml
error_prediction:
  calibrator: LocalParameterPredictor

  # ML regressor for parameter prediction
  local_regressor: Ridge           # Options: Linear, Ridge, Lasso, ElasticNet, RandomForest, XGB
  local_regressor_params:
    alpha: 1.0                     # Regularization strength

  # Features used for parameter prediction (same as bias prediction)
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
```

## Available Regressors

| Regressor | `local_regressor` | Best For | Key Parameters |
|-----------|-------------------|----------|----------------|
| Linear Regression | `Linear` | Baseline, interpretability | None |
| Ridge Regression | `Ridge` | Collinear features, default choice | `alpha` |
| Lasso Regression | `Lasso` | Feature selection | `alpha` |
| ElasticNet | `ElasticNet` | Mixed L1/L2 regularization | `alpha`, `l1_ratio` |
| Random Forest | `RandomForest` | Non-linear relationships | `n_estimators`, `max_depth` |
| XGBoost | `XGB` | Complex patterns | `max_depth`, `learning_rate` |

### Regressor Configuration Examples

**Ridge (recommended default):**
```yaml
local_regressor: Ridge
local_regressor_params:
  alpha: 1.0
```

**Random Forest:**
```yaml
local_regressor: RandomForest
local_regressor_params:
  n_estimators: 100
  max_depth: 5
  random_state: 42
```

**XGBoost:**
```yaml
local_regressor: XGB
local_regressor_params:
  max_depth: 3
  n_estimators: 100
  learning_rate: 0.1
```

**ElasticNet:**
```yaml
local_regressor: ElasticNet
local_regressor_params:
  alpha: 0.5
  l1_ratio: 0.5
```

## API Usage

### Basic Usage

```python
from wifa_uq.postprocessing.calibration import LocalParameterPredictor
import xarray as xr
import pandas as pd

# Load database
database = xr.load_dataset("results_stacked_hh.nc")

# Initialize with features
calibrator = LocalParameterPredictor(
    database,
    feature_names=["ABL_height", "wind_veer", "lapse_rate"],
    regressor_name="Ridge",
    regressor_params={"alpha": 1.0}
)

# Fit the parameter predictor
calibrator.fit()

# Get optimal sample indices for training data
optimal_indices = calibrator.get_optimal_indices()
print(f"Optimal indices shape: {optimal_indices.shape}")  # (n_cases,)

# Predict optimal parameters for new data
new_features = pd.DataFrame({
    "ABL_height": [500, 700, 900],
    "wind_veer": [0.01, 0.02, 0.015],
    "lapse_rate": [0.003, 0.005, 0.004]
})
predicted_params = calibrator.predict(new_features)
print(predicted_params)
#        k_b  ss_alpha
# 0   0.038     0.89
# 1   0.048     0.91
# 2   0.043     0.90
```

### Properties After Fitting

| Property | Type | Description |
|----------|------|-------------|
| `optimal_indices_` | ndarray | Per-case optimal sample indices |
| `optimal_params_` | dict | Per-case optimal parameter values |
| `swept_params` | list | Names of swept parameters |
| `regressor` | estimator | Fitted ML regressor |
| `is_fitted` | bool | Whether fit() has been called |

### Methods

| Method | Description |
|--------|-------------|
| `fit()` | Train the parameter predictor |
| `predict(X)` | Predict optimal parameters for new features |
| `get_optimal_indices()` | Get per-case optimal sample indices |

## Diagnostics

### Parameter Prediction Quality Plot

When using local calibration with cross-validation, WIFA-UQ automatically generates a diagnostic plot showing how well parameters are predicted:

```
local_parameter_prediction.png
```

This plot shows predicted vs. actual optimal parameters for each swept parameter, with R² scores indicating prediction quality.

**Interpreting the plot:**

- **High R² (> 0.7)**: Parameter predictor captures the relationship well
- **Low R² (< 0.3)**: Parameters may not vary systematically with features, or features are insufficient
- **Points along 1:1 line**: Good predictions
- **Systematic offset**: Bias in parameter prediction

### Manual Diagnostics

```python
import matplotlib.pyplot as plt
import numpy as np

# After fitting
calibrator = LocalParameterPredictor(database, feature_names=[...])
calibrator.fit()

# Get training data
X_train = database.isel(sample=0).to_dataframe().reset_index()[calibrator.feature_names]
optimal_k_b = calibrator.optimal_params_["k_b"]

# Visualize relationship
plt.figure(figsize=(12, 4))

for i, feature in enumerate(calibrator.feature_names):
    plt.subplot(1, len(calibrator.feature_names), i+1)
    plt.scatter(X_train[feature], optimal_k_b, alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel("Optimal k_b")
    plt.title(f"k_b vs {feature}")

plt.tight_layout()
plt.savefig("parameter_relationships.png")
```

## Comparison: Global vs Local

### Running Both

```python
from wifa_uq.postprocessing.calibration import MinBiasCalibrator, LocalParameterPredictor
from sklearn.model_selection import KFold
import numpy as np

database = xr.load_dataset("results_stacked_hh.nc")
features = ["ABL_height", "wind_veer", "lapse_rate"]

cv = KFold(n_splits=5, shuffle=True, random_state=42)

global_rmse = []
local_rmse = []

for train_idx, test_idx in cv.split(database.case_index):
    train_data = database.isel(case_index=train_idx)
    test_data = database.isel(case_index=test_idx)

    # Global calibration
    global_cal = MinBiasCalibrator(train_data)
    global_cal.fit()
    global_bias = test_data["model_bias_cap"].sel(sample=global_cal.best_idx_)
    global_rmse.append(np.sqrt(np.mean(global_bias.values**2)))

    # Local calibration
    local_cal = LocalParameterPredictor(train_data, feature_names=features)
    local_cal.fit()

    # Predict optimal params for test cases
    X_test = test_data.isel(sample=0).to_dataframe().reset_index()[features]
    pred_params = local_cal.predict(X_test)

    # Find closest samples
    local_bias = []
    for i, row in pred_params.iterrows():
        closest_sample = find_closest_sample(test_data, row)
        local_bias.append(test_data["model_bias_cap"].isel(case_index=i, sample=closest_sample).values)
    local_rmse.append(np.sqrt(np.mean(np.array(local_bias)**2)))

print(f"Global RMSE: {np.mean(global_rmse):.4f} ± {np.std(global_rmse):.4f}")
print(f"Local RMSE:  {np.mean(local_rmse):.4f} ± {np.std(local_rmse):.4f}")
```

### Expected Results

| Scenario | Global RMSE | Local RMSE | Recommendation |
|----------|-------------|------------|----------------|
| Homogeneous conditions | 0.045 | 0.044 | Use global (simpler) |
| Varying stability | 0.055 | 0.038 | Use local |
| Limited data (n<50) | 0.050 | 0.065 | Use global (local overfits) |

## Best Practices

### 1. Start with Global Calibration

Always establish a global baseline first:

```yaml
# First run
error_prediction:
  calibrator: MinBiasCalibrator

# Then compare with
error_prediction:
  calibrator: LocalParameterPredictor
```

### 2. Use Regularization

Local calibration can overfit with limited data. Start with regularized models:

```yaml
# Good default
local_regressor: Ridge
local_regressor_params:
  alpha: 1.0

# If underfitting, reduce regularization
local_regressor_params:
  alpha: 0.1
```

### 3. Choose Appropriate Features

Features should have physical connections to wake model parameters:

| Feature | Affects | Physical Reasoning |
|---------|---------|-------------------|
| ABL height | Wake expansion (k_b) | Deeper boundary layers allow more wake spreading |
| Wind veer | Wake deflection | Directional shear affects wake trajectory |
| Turbulence intensity | Wake recovery | Higher TI → faster wake recovery → different k_b |
| Stability (lapse rate) | Overall wake behavior | Stable conditions suppress mixing |

### 4. Check for Sufficient Variation

Parameters can only be predicted if they actually vary with features:

```python
# Check variation in optimal parameters
optimal_k_b = calibrator.optimal_params_["k_b"]
print(f"k_b range: {optimal_k_b.min():.3f} - {optimal_k_b.max():.3f}")
print(f"k_b std: {optimal_k_b.std():.4f}")

# If std is very small, parameters don't vary much with conditions
# → Global calibration may be sufficient
```

### 5. Validate with Cross-Validation

Always use cross-validation to assess generalization:

```yaml
cross_validation:
  splitting_mode: kfold_shuffled
  n_splits: 5
```

Compare CV metrics between global and local calibration.

## Troubleshooting

### "Local calibration is worse than global"

**Causes:**
- Insufficient training data → overfitting
- Features don't predict optimal parameters well
- Too flexible regressor (e.g., deep Random Forest)

**Solutions:**
- Increase regularization (`alpha` for Ridge)
- Use simpler regressor (Ridge instead of RandomForest)
- Add more relevant features
- Increase training data if possible

### "Parameter predictions are constant"

**Causes:**
- Features don't vary enough in your dataset
- Optimal parameters don't actually depend on features
- Regressor underfitting

**Solutions:**
- Check feature variance: `database["ABL_height"].std()`
- Examine relationships manually (scatter plots)
- Try less regularization or more flexible model

### "Predicted parameters outside valid range"

**Causes:**
- Extrapolation beyond training data range
- Regressor predicting unrealistic values

**Solutions:**
- The pipeline uses the *closest sample* in the database, so extreme predictions are automatically bounded
- For deployment, consider adding explicit parameter bounds

### "High variance across CV folds"

**Causes:**
- Insufficient data in each fold
- Unstable parameter-feature relationships
- Noisy reference data

**Solutions:**
- Use fewer CV splits (e.g., 3 instead of 5)
- Increase regularization
- Consider global calibration for more stable results

## Integration with Bias Prediction

Local calibration feeds into the bias prediction pipeline:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  Stage 1: Local Calibration                                                 │
│    For each case: θ*(case) = LocalParameterPredictor(features)              │
│                                                                             │
│  Stage 2: Bias Extraction                                                   │
│    For each case: residual_bias = database["model_bias_cap"][θ*(case)]      │
│                                                                             │
│  Stage 3: Bias Prediction (ML)                                              │
│    Learn: residual_bias = BiasPredictor(features)                           │
│                                                                             │
│  Final: corrected_power = model(θ*(features)) - predicted_residual_bias     │
└─────────────────────────────────────────────────────────────────────────────┘
```

The key insight: local calibration already reduces bias by using condition-appropriate parameters. The bias predictor then learns the remaining residual patterns.

## See Also

- [Global Calibration](global_calibration.md) — Simpler single-parameter approach
- [Calibration Theory](../../concepts/calibration_theory.md) — Mathematical foundations
- [Cross-Validation](../error_prediction/cross_validation.md) — Validation strategies
- [Configuration Reference](../configuration.md) — Full YAML options
