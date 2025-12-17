# Cross-Validation Strategies

Cross-validation (CV) ensures your bias predictor generalizes to unseen data. WIFA-UQ supports multiple CV strategies suited to different data structures and validation goals.

## Overview

Cross-validation estimates how well your model will perform on new data by systematically holding out portions of your dataset for testing.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Cross-Validation Process                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Full Dataset                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Case 1 │ Case 2 │ Case 3 │ Case 4 │ Case 5 │ ... │ Case N │        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Fold 1: ████████████████████████████████████████  │░░░░░░░░│              │
│  Fold 2: ████████████████████████████  │░░░░░░░░│  ██████████              │
│  Fold 3: ████████████████  │░░░░░░░░│  ██████████████████████              │
│  Fold 4: ████████  │░░░░░░░░│  ██████████████████████████████              │
│  Fold 5: │░░░░░░░░│  ████████████████████████████████████████              │
│                                                                             │
│          ████ = Training data    ░░░░ = Test data                          │
│                                                                             │
│  Metrics computed on each test fold, then averaged                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Available Strategies

| Strategy | `splitting_mode` | Best For | Preserves |
|----------|------------------|----------|-----------|
| K-Fold (ordered) | `kfold` | Time series, sequential data | Temporal ordering |
| K-Fold (shuffled) | `kfold_shuffled` | i.i.d. data, general use | Nothing (random splits) |
| Leave-One-Group-Out | `LeaveOneGroupOut` | Multi-farm, grouped data | Group integrity |
| Leave-One-Out | `LeaveOneOut` | Small datasets | Maximum test coverage |

## Quick Start

```yaml
cross_validation:
  splitting_mode: kfold_shuffled    # CV strategy
  n_splits: 5                       # Number of folds
  random_state: 42                  # Reproducibility
```

---

## K-Fold Cross-Validation

Divides data into K equal folds; each fold serves as test set once.

### K-Fold Ordered (`kfold`)

Preserves the original data order when creating folds. Use when temporal structure matters.

```yaml
cross_validation:
  splitting_mode: kfold
  n_splits: 5
```

**Visualization:**

```
Original order: [Jan, Feb, Mar, Apr, May, Jun, Jul, Aug, Sep, Oct, Nov, Dec]

Fold 1: Train=[Mar-Dec]           Test=[Jan, Feb]
Fold 2: Train=[Jan-Feb, May-Dec]  Test=[Mar, Apr]
Fold 3: Train=[Jan-Apr, Jul-Dec]  Test=[May, Jun]
Fold 4: Train=[Jan-Jun, Sep-Dec]  Test=[Jul, Aug]
Fold 5: Train=[Jan-Aug, Nov-Dec]  Test=[Sep, Oct]
Fold 6: Train=[Jan-Oct]           Test=[Nov, Dec]
```

**When to use:**
- Data has temporal structure (seasons, campaigns)
- You want adjacent cases in the same fold
- Testing generalization to different time periods

### K-Fold Shuffled (`kfold_shuffled`)

Randomly shuffles data before splitting. The default choice for most applications.

```yaml
cross_validation:
  splitting_mode: kfold_shuffled
  n_splits: 5
  random_state: 42                  # For reproducibility
```

**Visualization:**

```
Original: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Shuffled: [7, 2, 9, 4, 1, 8, 3, 10, 6, 5]

Fold 1: Train=[9,4,1,8,3,10,6,5]  Test=[7, 2]
Fold 2: Train=[7,2,1,8,3,10,6,5]  Test=[9, 4]
...
```

**When to use:**
- Data points are independent (i.i.d.)
- No specific grouping structure
- General-purpose validation
- Most common choice

### Choosing n_splits

| Dataset Size | Recommended `n_splits` | Notes |
|--------------|------------------------|-------|
| < 50 cases | 3-5 | Larger test folds for reliable estimates |
| 50-200 cases | 5-10 | Standard choice |
| > 200 cases | 5-10 | More folds give similar results |

**Trade-offs:**

- **More splits**: More training data per fold, but smaller test sets (higher variance in metrics)
- **Fewer splits**: Larger test sets (more stable metrics), but less training data

---

## Leave-One-Group-Out (LOGO)

Holds out entire groups (e.g., wind farms) for testing. Essential when you have grouped data and want to test generalization to new groups.

### Configuration

```yaml
cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm              # Column defining groups
```

**Visualization:**

```
Dataset with 3 wind farms:

┌─────────────────────────────────────────────────────────────────────────────┐
│  Farm A: [A1, A2, A3, ..., A50]                                             │
│  Farm B: [B1, B2, B3, ..., B40]                                             │
│  Farm C: [C1, C2, C3, ..., C60]                                             │
└─────────────────────────────────────────────────────────────────────────────┘

Fold 1: Train=[Farm B + Farm C]    Test=[Farm A]   (150 cases total)
Fold 2: Train=[Farm A + Farm C]    Test=[Farm B]
Fold 3: Train=[Farm A + Farm B]    Test=[Farm C]
```

### When to Use LOGO

**Required when:**
- Training on multiple wind farms
- Cases within a farm are correlated (same turbines, similar terrain)
- Goal is predicting performance at **new** farms

**Why it matters:**

Without LOGO, cases from the same farm appear in both train and test sets, leading to **data leakage** and overly optimistic metrics:

```
❌ KFold with multi-farm data:
   Train: [A1, A3, B1, B3, C1, C3, ...]
   Test:  [A2, A4, B2, B4, C2, C4, ...]

   Problem: Model learns farm-specific patterns from A1, A3 and
            "predicts" A2, A4 using that leaked information.

✓ LOGO with multi-farm data:
   Train: [All of Farm B, All of Farm C]
   Test:  [All of Farm A]

   Correct: Model must generalize to Farm A without seeing ANY Farm A data.
```

### Multi-Farm Configuration

Complete multi-farm setup with LOGO:

```yaml
# Multi-farm configuration
farms:
  - name: farm_alpha
    system_config: /path/to/alpha/system.yaml
  - name: farm_beta
    system_config: /path/to/beta/system.yaml
  - name: farm_gamma
    system_config: /path/to/gamma/system.yaml

cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm
```

### Group Key Selection

The `group_key` must exist as a coordinate in your stacked dataset:

```python
import xarray as xr

dataset = xr.load_dataset("results_stacked_hh.nc")
print(dataset.coords)
# Coordinates:
#   case_index    (case_index) int64 ...
#   wind_farm     (case_index) <U10 'alpha' 'alpha' ... 'gamma'  ← group_key
#   sample        (sample) int64 ...
```

Common group keys:
- `wind_farm`: Different wind farm sites
- `campaign`: Different measurement campaigns
- `season`: Seasonal grouping (if cases per season are sufficient)

---

## Leave-One-Out (LOO)

Extreme case where each sample is its own test fold. Maximizes training data but computationally expensive.

### Configuration

```yaml
cross_validation:
  splitting_mode: LeaveOneOut
```

**Visualization:**

```
Dataset: [1, 2, 3, 4, 5]

Fold 1: Train=[2,3,4,5]  Test=[1]
Fold 2: Train=[1,3,4,5]  Test=[2]
Fold 3: Train=[1,2,4,5]  Test=[3]
Fold 4: Train=[1,2,3,5]  Test=[4]
Fold 5: Train=[1,2,3,4]  Test=[5]
```

### When to Use

- **Very small datasets** (< 30 cases)
- Need maximum training data per fold
- Computational cost is acceptable (N model fits)

### Limitations

- **Computationally expensive**: N folds for N samples
- **High variance**: Single-sample test sets are noisy
- **Not recommended** for large datasets

---

## Metrics Computed

Cross-validation computes these metrics on each test fold:

| Metric | Description | Ideal Value |
|--------|-------------|-------------|
| RMSE | Root mean squared error | Lower is better |
| MAE | Mean absolute error | Lower is better |
| R² | Coefficient of determination | Higher is better (max 1.0) |
| Bias | Mean error (systematic offset) | Close to 0 |

### Output Format

```yaml
# cv_metrics.yaml
mean_rmse: 0.0234
std_rmse: 0.0045
mean_mae: 0.0189
std_mae: 0.0038
mean_r2: 0.847
std_r2: 0.052
per_fold_metrics:
  fold_0:
    rmse: 0.0212
    mae: 0.0178
    r2: 0.891
  fold_1:
    rmse: 0.0256
    mae: 0.0201
    r2: 0.823
  # ...
```

### Interpreting Results

**Good generalization:**
- Low `std_rmse` relative to `mean_rmse` (< 20%)
- Consistent R² across folds
- No single fold dramatically worse than others

**Signs of problems:**
- High variance across folds → insufficient data or data heterogeneity
- One fold much worse → possible outliers or distinct subpopulation
- Training metrics much better than CV metrics → overfitting

---

## Comparison of Strategies

### By Data Structure

| Data Structure | Recommended Strategy |
|----------------|---------------------|
| Single farm, i.i.d. cases | `kfold_shuffled` |
| Single farm, temporal data | `kfold` |
| Multiple farms | `LeaveOneGroupOut` |
| Very small dataset (< 30) | `LeaveOneOut` |
| Seasonal patterns to preserve | `kfold` or custom groups |

### By Validation Goal

| Goal | Recommended Strategy |
|------|---------------------|
| General performance estimate | `kfold_shuffled` |
| Generalization to new farms | `LeaveOneGroupOut` |
| Generalization to new time periods | `kfold` (ordered) |
| Maximum training data | `LeaveOneOut` |

### Computational Cost

| Strategy | Number of Folds | Relative Cost |
|----------|-----------------|---------------|
| `kfold_shuffled` (k=5) | 5 | Low |
| `kfold` (k=10) | 10 | Low-Medium |
| `LeaveOneGroupOut` | # of groups | Medium |
| `LeaveOneOut` | # of samples | High |

---

## API Usage

### Running Cross-Validation

```python
from wifa_uq.postprocessing.cross_validation import run_cross_validation
from wifa_uq.postprocessing.calibration import MinBiasCalibrator
from wifa_uq.postprocessing.error_prediction import BiasPredictor
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

# Configure CV
cv_config = {
    "splitting_mode": "kfold_shuffled",
    "n_splits": 5,
    "random_state": 42
}

# Run cross-validation
results = run_cross_validation(
    database=database,
    calibrator_class=MinBiasCalibrator,
    predictor_class=BiasPredictor,
    predictor_kwargs={"regressor_name": "XGB", "feature_names": ["ABL_height", "wind_veer"]},
    cv_config=cv_config
)

print(f"Mean RMSE: {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
```

### Custom Cross-Validation Splitter

```python
from sklearn.model_selection import StratifiedKFold
import numpy as np

# Create custom splitter (e.g., stratified by stability class)
stability_labels = np.digitize(
    database["lapse_rate"].isel(sample=0).values,
    bins=[-0.005, 0, 0.005]  # unstable, neutral, stable
)

custom_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Use in cross-validation
results = run_cross_validation(
    database=database,
    calibrator_class=MinBiasCalibrator,
    predictor_class=BiasPredictor,
    predictor_kwargs={...},
    cv_splitter=custom_cv,
    stratify_labels=stability_labels
)
```

### Accessing Per-Fold Results

```python
# Per-fold predictions and actuals
for fold_idx, fold_result in enumerate(results["fold_results"]):
    print(f"\nFold {fold_idx}:")
    print(f"  Test indices: {fold_result['test_indices'][:5]}...")
    print(f"  RMSE: {fold_result['rmse']:.4f}")
    print(f"  Predictions shape: {fold_result['predictions'].shape}")
```

---

## Best Practices

### 1. Match Strategy to Data Structure

```yaml
# Single farm
cross_validation:
  splitting_mode: kfold_shuffled
  n_splits: 5

# Multi-farm
cross_validation:
  splitting_mode: LeaveOneGroupOut
  group_key: wind_farm
```

### 2. Use Sufficient Folds

Too few folds → high variance in estimates:

```python
# Compare different n_splits
for n_splits in [3, 5, 10]:
    results = run_cross_validation(..., cv_config={"n_splits": n_splits})
    print(f"n_splits={n_splits}: RMSE = {results['mean_rmse']:.4f} ± {results['std_rmse']:.4f}")
```

### 3. Set Random State for Reproducibility

```yaml
cross_validation:
  splitting_mode: kfold_shuffled
  n_splits: 5
  random_state: 42    # Always set this!
```

### 4. Check for Leakage in Grouped Data

If you have grouped data but use K-Fold, verify metrics aren't artificially inflated:

```python
# Compare KFold vs LOGO
kfold_results = run_cross_validation(..., cv_config={"splitting_mode": "kfold_shuffled"})
logo_results = run_cross_validation(..., cv_config={"splitting_mode": "LeaveOneGroupOut", "group_key": "wind_farm"})

print(f"KFold RMSE:  {kfold_results['mean_rmse']:.4f}")
print(f"LOGO RMSE:   {logo_results['mean_rmse']:.4f}")

# If LOGO RMSE >> KFold RMSE, there was likely leakage
```

### 5. Examine Fold-Level Variation

High variation suggests data heterogeneity:

```python
import numpy as np

rmse_values = [fold['rmse'] for fold in results['fold_results']]
cv_coefficient = np.std(rmse_values) / np.mean(rmse_values)

if cv_coefficient > 0.3:
    print("WARNING: High CV coefficient suggests heterogeneous data")
    print("Consider: more data, different features, or stratified CV")
```

---

## Troubleshooting

### "LOGO gives much worse metrics than KFold"

**This is expected and correct!** LOGO tests true generalization to unseen groups. The gap indicates how much the model relies on group-specific patterns.

**Solutions:**
- This represents realistic performance on new farms
- Focus on features that generalize across farms
- Consider farm-specific calibration as a separate step

### "High variance across folds"

**Causes:**
- Insufficient data per fold
- Heterogeneous data (different conditions across folds)
- Outliers affecting individual folds

**Solutions:**
- Reduce number of splits
- Use stratified splitting to balance conditions
- Identify and examine problematic folds
- Check for outliers in the data

### "One fold is much worse than others"

**Causes:**
- That fold contains unusual conditions
- Outliers concentrated in that fold
- Data quality issues in that subset

**Solutions:**
```python
# Identify the bad fold
worst_fold_idx = np.argmax([f['rmse'] for f in results['fold_results']])
worst_fold = results['fold_results'][worst_fold_idx]

# Examine test cases in that fold
test_indices = worst_fold['test_indices']
test_features = database.isel(case_index=test_indices, sample=0)

# Check for anomalies
print(test_features[["ABL_height", "wind_veer"]].to_dataframe().describe())
```

### "Not enough data for LOGO"

**Problem:** With few groups (e.g., 2 farms), each fold uses only 1 farm for training.

**Solutions:**
- Combine with nested CV if possible
- Use repeated LOGO with bootstrapping
- Consider KFold but acknowledge limitations
- Acquire more farm data if possible

---

## Advanced Topics

### Nested Cross-Validation

For hyperparameter tuning without leakage:

```
Outer loop: Model evaluation (e.g., 5-fold)
  Inner loop: Hyperparameter selection (e.g., 3-fold)
```

```python
from sklearn.model_selection import GridSearchCV, cross_val_score

# Outer CV for evaluation
outer_cv = KFold(n_splits=5, shuffle=True, random_state=42)

# Inner CV for hyperparameter tuning (wrapped in GridSearchCV)
inner_cv = KFold(n_splits=3, shuffle=True, random_state=42)

param_grid = {"max_depth": [2, 3, 4], "n_estimators": [50, 100, 200]}

# This gives unbiased performance estimate with tuned hyperparameters
nested_scores = cross_val_score(
    GridSearchCV(XGBRegressor(), param_grid, cv=inner_cv),
    X, y, cv=outer_cv, scoring="neg_root_mean_squared_error"
)
```

### Time-Series Cross-Validation

For strictly temporal data where future data shouldn't be used:

```python
from sklearn.model_selection import TimeSeriesSplit

# Only uses past data to predict future
ts_cv = TimeSeriesSplit(n_splits=5)

for train_idx, test_idx in ts_cv.split(X):
    print(f"Train: {train_idx.min()}-{train_idx.max()}, Test: {test_idx.min()}-{test_idx.max()}")
# Train: 0-19, Test: 20-29
# Train: 0-29, Test: 30-39
# ...
```

---

## See Also

- [ML Models](ml_models.md) — Models used in cross-validation
- [Global Calibration](../calibration/global_calibration.md) — Calibration within CV
- [Local Calibration](../calibration/local_calibration.md) — Parameter prediction within CV
- [Configuration Reference](../configuration.md) — Full YAML options
