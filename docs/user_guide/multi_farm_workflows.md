# Multi-Farm Workflows

WIFA-UQ supports training and validating models across multiple wind farms simultaneously. This enables building generalizable bias predictors that work on new, unseen farms.

## Overview

Multi-farm workflows address a key challenge: **Can a model trained on farms A, B, and C predict bias at farm D?**

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       Multi-Farm Workflow                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Farm A          Farm B          Farm C          Farm D (new)               │
│  ┌─────┐        ┌─────┐        ┌─────┐          ┌─────┐                     │
│  │█ █ █│        │█ █  │        │█ █ █│          │█ █ █│                     │
│  │█ █ █│        │█ █ █│        │█ █ █│          │█ █ █│                     │
│  │█ █ █│        │█ █ █│        │█ █  │          │█ █ █│                     │
│  └─────┘        └─────┘        └─────┘          └─────┘                     │
│     │              │              │                 │                       │
│     └──────────────┼──────────────┘                 │                       │
│                    │                                │                       │
│                    ▼                                ▼                       │
│            ┌──────────────┐                 ┌──────────────┐                 │
│            │   Training   │                 │  Prediction  │                 │
│            │  (A + B + C) │────────────────►│   (Farm D)   │                 │
│            └──────────────┘                 └──────────────┘                 │
│                                                                             │
│  Key Questions:                                                             │
│  • How well does the model generalize to unseen farms?                      │
│  • Which features transfer across farms?                                    │
│  • How much farm-specific vs. universal bias exists?                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## When to Use Multi-Farm Workflows

**Use multi-farm when:**

- You have reference data from multiple wind farms
- Goal is a **generalizable** model (not farm-specific)
- You want to predict bias at **new farms** without retraining
- You need **robust validation** via Leave-One-Group-Out CV

**Use single-farm when:**

- You only have data from one farm
- Building a **site-specific** bias correction
- Insufficient data per farm for meaningful splitting

## Configuration

### Basic Multi-Farm Setup

```yaml
# Multi-farm configuration with explicit farm definitions
farms:
  - name: alpha
    system_config: /path/to/alpha/system.yaml
    reference_power: /path/to/alpha/reference_power.nc      # Optional
    reference_resource: /path/to/alpha/reference_resource.nc # Optional
    wind_farm_layout: /path/to/alpha/layout.yaml            # Optional

  - name: beta
    system_config: /path/to/beta/system.yaml

  - name: gamma
    system_config: /path/to/gamma/system.yaml

# Cross-validation with Leave-One-Group-Out
cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm              # Groups by farm name

# Error prediction settings (same for all farms)
error_prediction:
  calibrator: MinBiasCalibrator
  regressor: XGB
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
```

### Path Inference

Each farm can use smart path inference from its `system_config`:

```yaml
farms:
  - name: alpha
    system_config: /path/to/alpha/system.yaml
    # Other paths inferred from windIO !include chain

  - name: beta
    system_config: /path/to/beta/system.yaml
    reference_power: /custom/path/beta_power.nc  # Override inference
```

### Minimal Configuration

If all farms follow the same directory structure:

```yaml
farms:
  - name: alpha
    system_config: /data/farms/alpha/system.yaml
  - name: beta
    system_config: /data/farms/beta/system.yaml
  - name: gamma
    system_config: /data/farms/gamma/system.yaml

cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm
```

---

## Data Stacking

Multi-farm data is stacked into a single dataset with farm identifiers.

### Stacked Dataset Structure

```
Dimensions:
  - case_index: N_total (sum of all farm cases)
  - sample: n_samples (parameter samples)

Coordinates:
  - case_index: [0, 1, 2, ..., N_total-1]
  - wind_farm: ['alpha', 'alpha', ..., 'beta', 'beta', ..., 'gamma', ...]
  - sample: [0, 1, ..., n_samples-1]

Data Variables:
  - model_bias_cap: (sample, case_index)
  - ABL_height: (sample, case_index)
  - wind_veer: (sample, case_index)
  - ... other features
```

### Viewing Stacked Data

```python
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

# Check farm distribution
farm_counts = database.groupby("wind_farm").count()
print("Cases per farm:")
for farm in database.wind_farm.values:
    n_cases = (database.wind_farm == farm).sum().values
    print(f"  {farm}: {n_cases}")

# Access data for specific farm
alpha_data = database.where(database.wind_farm == "alpha", drop=True)
print(f"Alpha dataset shape: {alpha_data.dims}")
```

### Manual Stacking (if needed)

```python
import xarray as xr
import numpy as np

# Load individual farm databases
db_alpha = xr.load_dataset("alpha/results_stacked_hh.nc")
db_beta = xr.load_dataset("beta/results_stacked_hh.nc")
db_gamma = xr.load_dataset("gamma/results_stacked_hh.nc")

# Add farm identifiers
db_alpha = db_alpha.assign_coords(wind_farm=("case_index", ["alpha"] * db_alpha.dims["case_index"]))
db_beta = db_beta.assign_coords(wind_farm=("case_index", ["beta"] * db_beta.dims["case_index"]))
db_gamma = db_gamma.assign_coords(wind_farm=("case_index", ["gamma"] * db_gamma.dims["case_index"]))

# Concatenate along case_index
combined = xr.concat([db_alpha, db_beta, db_gamma], dim="case_index")

# Reset case_index to be sequential
combined = combined.assign_coords(case_index=np.arange(combined.dims["case_index"]))

combined.to_netcdf("results_stacked_hh_combined.nc")
```

---

## Leave-One-Group-Out Cross-Validation

LOGO CV is essential for multi-farm validation. It tests whether the model generalizes to **entirely unseen farms**.

### How LOGO Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Leave-One-Group-Out CV                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Fold 1:  Train on [Beta, Gamma]     Test on [Alpha]                        │
│           ████████████████████████   ░░░░░░░░░░░                            │
│                                                                             │
│  Fold 2:  Train on [Alpha, Gamma]    Test on [Beta]                         │
│           ████████████████████████   ░░░░░░░░░░░                            │
│                                                                             │
│  Fold 3:  Train on [Alpha, Beta]     Test on [Gamma]                        │
│           ████████████████████████   ░░░░░░░░░░░                            │
│                                                                             │
│  Result: Average metrics across folds                                       │
│          → Estimate of performance on NEW farms                             │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Configuration

```yaml
cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm
```

### Why LOGO is Critical

Without LOGO, standard K-Fold **mixes cases from the same farm** in train and test:

```
❌ K-Fold (data leakage):
   Train: [A1, A3, A5, B1, B3, C1, C3, ...]
   Test:  [A2, A4, B2, B4, C2, C4, ...]

   Problem: Model sees farm A training data, then "predicts" farm A test data
   → Overly optimistic metrics due to farm-specific patterns leaking through

✓ LOGO (proper generalization test):
   Train: [All of B, All of C]
   Test:  [All of A]

   Correct: Model must generalize to farm A without ANY farm A exposure
   → Realistic estimate of new-farm performance
```

### Interpreting LOGO Results

```python
from wifa_uq.postprocessing.cross_validation import run_cross_validation

results = run_cross_validation(
    database=database,
    cv_config={"splitting_mode": "LeaveOneGroupOut", "group_key": "wind_farm"},
    ...
)

print("LOGO Cross-Validation Results:")
print(f"  Mean RMSE: {results['mean_rmse']:.4f}")
print(f"  Std RMSE:  {results['std_rmse']:.4f}")
print(f"  Mean R²:   {results['mean_r2']:.3f}")

print("\nPer-Farm Results (when held out):")
for fold in results['fold_results']:
    farm = fold['test_farm']
    print(f"  {farm}: RMSE={fold['rmse']:.4f}, R²={fold['r2']:.3f}")
```

**Example output:**
```
LOGO Cross-Validation Results:
  Mean RMSE: 0.0312
  Std RMSE:  0.0089
  Mean R²:   0.724

Per-Farm Results (when held out):
  alpha: RMSE=0.0245, R²=0.812
  beta:  RMSE=0.0298, R²=0.756
  gamma: RMSE=0.0394, R²=0.605
```

**Interpretation:**
- Overall RMSE ≈ 0.031 on unseen farms
- Farm gamma is harder to predict (lower R²)
- Variance across farms (std=0.009) indicates some farm heterogeneity

### Comparing LOGO vs K-Fold

Always compare to understand the **generalization gap**:

```python
# K-Fold (with leakage)
kfold_results = run_cross_validation(
    database=database,
    cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 5},
    ...
)

# LOGO (proper generalization)
logo_results = run_cross_validation(
    database=database,
    cv_config={"splitting_mode": "LeaveOneGroupOut", "group_key": "wind_farm"},
    ...
)

print(f"K-Fold RMSE:  {kfold_results['mean_rmse']:.4f}")
print(f"LOGO RMSE:    {logo_results['mean_rmse']:.4f}")
print(f"Generalization gap: {(logo_results['mean_rmse'] - kfold_results['mean_rmse']) / kfold_results['mean_rmse'] * 100:.1f}%")
```

**Typical results:**
```
K-Fold RMSE:  0.0198
LOGO RMSE:    0.0312
Generalization gap: 57.6%
```

A large gap indicates significant **farm-specific patterns** that don't transfer.

---

## Generalization Strategies

When LOGO performance is poor, several strategies can help.

### Strategy 1: Focus on Universal Features

Some features transfer better than others:

```yaml
# Features that typically generalize well
error_prediction:
  features:
    - ABL_height           # Atmospheric, not farm-specific
    - wind_veer            # Atmospheric
    - lapse_rate           # Atmospheric
    - turbulence_intensity # Atmospheric

# Features that may be farm-specific (use cautiously)
#   - Farm_Length          # Depends on layout
#   - Blockage_Ratio       # Depends on layout
#   - wind_direction       # Site-specific wind rose
```

**Analysis approach:**

```python
# Compare feature importance across folds
from wifa_uq.postprocessing.sensitivity import SHAPAnalyzer

fold_importances = []
for fold in results['fold_results']:
    predictor = fold['predictor']
    shap_analyzer = SHAPAnalyzer(predictor)
    importance = shap_analyzer.get_feature_importance()
    fold_importances.append(importance)

# Features with consistent importance across folds generalize better
import pandas as pd
importance_df = pd.DataFrame(fold_importances)
print("Feature importance consistency:")
print(importance_df.std() / importance_df.mean())  # CV coefficient
```

### Strategy 2: Regularization

Increase regularization to prevent overfitting to farm-specific patterns:

```yaml
error_prediction:
  regressor: XGB
  regressor_params:
    max_depth: 2              # Shallower trees
    n_estimators: 50          # Fewer trees
    min_child_weight: 5       # More regularization
    reg_alpha: 0.1            # L1 regularization
    reg_lambda: 1.0           # L2 regularization
```

Or use inherently regularized models:

```yaml
error_prediction:
  regressor: Linear
  regressor_params:
    linear_type: ridge
    alpha: 10.0               # Strong regularization
```

### Strategy 3: Feature Normalization by Farm

Normalize features relative to farm-specific baselines:

```python
# Normalize features to zero mean per farm
for feature in ["ABL_height", "wind_veer", "lapse_rate"]:
    farm_means = database.groupby("wind_farm")[feature].mean()
    database[f"{feature}_normalized"] = database.groupby("wind_farm")[feature] - farm_means

# Use normalized features
features:
  - ABL_height_normalized
  - wind_veer_normalized
  - lapse_rate_normalized
```

### Strategy 4: Two-Stage Modeling

Separate farm-specific calibration from universal bias prediction:

```
Stage 1: Per-farm calibration
  → Each farm gets its own optimal parameters (k_b, ss_alpha)

Stage 2: Universal bias prediction
  → Train bias predictor on residuals from Stage 1
  → Use only atmospheric features (no layout features)
```

```python
# Stage 1: Calibrate each farm separately
farm_calibrations = {}
for farm in database.wind_farm.unique():
    farm_data = database.where(database.wind_farm == farm, drop=True)
    calibrator = MinBiasCalibrator(farm_data)
    calibrator.fit()
    farm_calibrations[farm] = calibrator.best_params_

# Stage 2: Train universal predictor on calibrated residuals
# ... (extract residuals at farm-specific calibrated parameters)
# ... (train predictor using only atmospheric features)
```

### Strategy 5: Domain Adaptation

If one farm is very different, consider domain adaptation techniques:

```python
# Identify the problematic farm
for fold in results['fold_results']:
    if fold['rmse'] > 1.5 * results['mean_rmse']:
        print(f"Farm {fold['test_farm']} is an outlier")

# Options:
# 1. Exclude it from training (if data quality issue)
# 2. Weight training samples (down-weight dissimilar farms)
# 3. Use transfer learning (fine-tune on target farm)
```

---

## API Reference

### MultiDatabase Generator

```python
from wifa_uq.database_generation import MultiDatabaseGenerator

generator = MultiDatabaseGenerator(
    farm_configs=[
        {"name": "alpha", "system_config": "/path/to/alpha/system.yaml"},
        {"name": "beta", "system_config": "/path/to/beta/system.yaml"},
        {"name": "gamma", "system_config": "/path/to/gamma/system.yaml"},
    ],
    nsamples=100,
    param_config={
        "Bastankhah_params.k_b": [0.01, 0.07],
        "SelfSimilar_params.ss_alpha": [0.75, 1.0],
    },
    output_path="results_stacked_hh.nc"
)

# Generate databases for all farms and stack
generator.run()

# Access the stacked dataset
database = generator.get_stacked_database()
```

### Cross-Validation with Groups

```python
from wifa_uq.postprocessing.cross_validation import run_cross_validation
from sklearn.model_selection import LeaveOneGroupOut

database = xr.load_dataset("results_stacked_hh.nc")

# Extract groups
groups = database.wind_farm.values

# Run LOGO CV
logo_cv = LeaveOneGroupOut()

results = run_cross_validation(
    database=database,
    cv_splitter=logo_cv,
    groups=groups,
    calibrator_class=MinBiasCalibrator,
    predictor_class=BiasPredictor,
    predictor_kwargs={"regressor_name": "XGB", "feature_names": features}
)
```

### Per-Farm Analysis

```python
def analyze_per_farm(database, results):
    """Analyze results broken down by farm."""

    summary = {}
    for farm in database.wind_farm.unique():
        farm_mask = database.wind_farm == farm
        farm_predictions = results['all_predictions'][farm_mask]
        farm_actuals = results['all_actuals'][farm_mask]

        rmse = np.sqrt(np.mean((farm_predictions - farm_actuals)**2))
        mae = np.mean(np.abs(farm_predictions - farm_actuals))

        summary[farm] = {"rmse": rmse, "mae": mae, "n_cases": farm_mask.sum()}

    return pd.DataFrame(summary).T

# Usage
farm_summary = analyze_per_farm(database, results)
print(farm_summary)
```

---

## Best Practices

### 1. Always Use LOGO for Multi-Farm

```yaml
# Correct
cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm

# Incorrect (data leakage)
cross_validation:
  splitting_mode: kfold_shuffled
```

### 2. Report Both LOGO and K-Fold Metrics

```
Results:
  K-Fold RMSE (within-farm): 0.020
  LOGO RMSE (cross-farm):    0.031
  Generalization gap:        55%
```

### 3. Check Per-Farm Performance

```python
# Identify farms that are hard to predict
for fold in results['fold_results']:
    if fold['rmse'] > 1.5 * results['mean_rmse']:
        print(f"Warning: Farm {fold['test_farm']} has high error")
        print(f"  Consider: different conditions, data quality, layout effects")
```

### 4. Validate Feature Consistency

```python
# Check that features have similar distributions across farms
for feature in features:
    print(f"\n{feature}:")
    for farm in database.wind_farm.unique():
        farm_data = database[feature].where(database.wind_farm == farm, drop=True)
        print(f"  {farm}: mean={farm_data.mean():.3f}, std={farm_data.std():.3f}")
```

### 5. Consider Minimum Farms

With only 2 farms, LOGO tests generalization to 1 held-out farm based on training on 1 farm — very limited. Aim for **3+ farms** for meaningful LOGO validation.

| # Farms | LOGO Folds | Training Data per Fold | Recommendation |
|---------|------------|------------------------|----------------|
| 2 | 2 | 50% | Marginal; report limitations |
| 3 | 3 | 67% | Acceptable minimum |
| 5 | 5 | 80% | Good |
| 10+ | 10+ | 90%+ | Ideal |

---

## Troubleshooting

### "LOGO gives much worse results than K-Fold"

**This is expected!** LOGO tests true generalization. The gap quantifies farm-specific vs. universal patterns.

**Actions:**
- Report both metrics honestly
- Apply generalization strategies (above)
- Consider if your use case requires cross-farm generalization

### "One farm dominates errors"

**Causes:**
- Different turbine types
- Unique terrain/conditions
- Data quality issues
- Insufficient similar training farms

**Solutions:**
```python
# Analyze the problematic farm
problem_farm = "gamma"
farm_data = database.where(database.wind_farm == problem_farm, drop=True)

# Check feature distributions
print(farm_data[features].to_dataframe().describe())

# Compare to other farms
# ... look for unusual patterns
```

### "Not enough data per farm for LOGO"

**Problem:** Each farm has very few cases, so training folds are too small.

**Solutions:**
- Combine similar farms into groups
- Use bootstrap aggregating within LOGO
- Report uncertainty estimates
- Collect more data

### "Features have different scales across farms"

**Problem:** ABL_height at offshore farm A ranges 500-2000m, at onshore farm B ranges 200-800m.

**Solutions:**
```python
# Option 1: Standardize globally
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Option 2: Standardize per-farm
X_scaled = database.groupby("wind_farm").apply(
    lambda x: (x - x.mean()) / x.std()
)
```

---

## Example: Complete Multi-Farm Pipeline

```yaml
# multi_farm_config.yaml

farms:
  - name: north_sea_alpha
    system_config: /data/farms/north_sea_alpha/system.yaml
  - name: north_sea_beta
    system_config: /data/farms/north_sea_beta/system.yaml
  - name: baltic_gamma
    system_config: /data/farms/baltic_gamma/system.yaml
  - name: atlantic_delta
    system_config: /data/farms/atlantic_delta/system.yaml

database_gen:
  nsamples: 100
  param_config:
    Bastankhah_params.k_b: [0.01, 0.07]
    Bastankhah_params.ceps: [0.15, 0.30]
    SelfSimilar_params.ss_alpha: [0.75, 1.0]

cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm

error_prediction:
  calibrator: MinBiasCalibrator
  regressor: XGB
  regressor_params:
    max_depth: 3
    n_estimators: 100
    min_child_weight: 3
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity

sensitivity_analysis:
  enabled: true

output:
  save_per_farm_metrics: true
  save_generalization_gap: true
```

```bash
# Run the pipeline
wifa-uq run multi_farm_config.yaml
```

**Expected outputs:**
```
results/
├── results_stacked_hh.nc           # Combined database
├── cv_metrics.yaml                 # Overall CV results
├── cv_metrics_per_farm.yaml        # Per-farm breakdown
├── generalization_analysis.yaml    # K-Fold vs LOGO comparison
├── shap_summary.png                # Feature importance
└── predictions/
    ├── fold_0_alpha.nc             # Predictions when alpha held out
    ├── fold_1_beta.nc
    ├── fold_2_gamma.nc
    └── fold_3_delta.nc
```

---

## See Also

- [Cross-Validation](error_prediction/cross_validation.md) — Detailed CV strategies
- [Configuration Reference](configuration.md) — Full YAML options
- [Database Generation](database_generation.md) — Single and multi-farm databases
- [Global Calibration](calibration/global_calibration.md) — Calibration in multi-farm context
