# ML Models for Bias Prediction

WIFA-UQ supports multiple machine learning models for predicting residual bias after calibration. Each model offers different trade-offs between accuracy, interpretability, and uncertainty quantification.

## Overview

The bias predictor learns to predict residual model error as a function of atmospheric features:

```
residual_bias = f(ABL_height, wind_veer, lapse_rate, turbulence_intensity, ...)
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Bias Prediction Pipeline                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Input Features              ML Model                  Output               │
│  ┌─────────────────┐        ┌─────────────────┐       ┌─────────────────┐   │
│  │ ABL_height      │        │                 │       │ predicted_bias  │   │
│  │ wind_veer       │   ──►  │  XGB / PCE /    │  ──►  │                 │   │
│  │ lapse_rate      │        │  SIR / Linear   │       │ (± uncertainty) │   │
│  │ TI              │        │                 │       │                 │   │
│  │ ...             │        └─────────────────┘       └─────────────────┘   │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Model Comparison

| Model | `regressor` | Interpretability | UQ Method | Best For |
|-------|-------------|------------------|-----------|----------|
| XGBoost | `XGB` | SHAP values | Ensemble variance | Best accuracy, feature importance |
| PCE | `PCE` | Sobol indices | Analytical | Smooth responses, global sensitivity |
| SIR Polynomial | `SIRPolynomial` | Coefficients | Analytical | Dimension reduction, interpretability |
| Ridge | `Linear` | Coefficients | Analytical | Baseline, linear relationships |
| Lasso | `Linear` | Sparse coefficients | Analytical | Feature selection |
| ElasticNet | `Linear` | Sparse coefficients | Analytical | Mixed regularization |

## Quick Start

```yaml
error_prediction:
  calibrator: MinBiasCalibrator
  regressor: XGB                    # Model choice
  regressor_params:
    max_depth: 3
    n_estimators: 100
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
```

---

## XGBoost (XGB)

Gradient boosted decision trees. Generally provides the best predictive accuracy and robust feature importance via SHAP values.

### When to Use

- **Best accuracy** is the priority
- You want **SHAP-based feature importance**
- Non-linear relationships exist between features and bias
- You have sufficient training data (> 50-100 cases)

### Configuration

```yaml
error_prediction:
  regressor: XGB
  regressor_params:
    max_depth: 3              # Tree depth (default: 3)
    n_estimators: 100         # Number of trees (default: 100)
    learning_rate: 0.1        # Step size shrinkage (default: 0.1)
    min_child_weight: 1       # Minimum sum of instance weight in child
    subsample: 1.0            # Fraction of samples per tree
    colsample_bytree: 1.0     # Fraction of features per tree
    random_state: 42          # Reproducibility
```

### Parameter Tuning Guide

| Parameter | Low Value Effect | High Value Effect | Recommended |
|-----------|------------------|-------------------|-------------|
| `max_depth` | Underfitting | Overfitting | 2-5 |
| `n_estimators` | Underfitting | Slower, diminishing returns | 50-200 |
| `learning_rate` | Needs more trees | Overfitting risk | 0.05-0.3 |
| `min_child_weight` | More complex trees | More regularization | 1-10 |

### SHAP Feature Importance

XGBoost automatically generates SHAP (SHapley Additive exPlanations) analysis:

```
Output files:
  shap_summary.png           # Beeswarm plot of feature impacts
  shap_feature_importance.png # Bar chart of mean |SHAP|
```

**Interpreting SHAP plots:**

- **Beeswarm plot**: Each point is a prediction; x-axis shows impact on output; color shows feature value
- **Feature importance**: Mean absolute SHAP value per feature

### API Usage

```python
from wifa_uq.postprocessing.error_prediction import BiasPredictor
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

predictor = BiasPredictor(
    database,
    regressor_name="XGB",
    regressor_params={"max_depth": 3, "n_estimators": 100},
    feature_names=["ABL_height", "wind_veer", "lapse_rate"]
)

predictor.fit(calibrated_sample_indices)

# Predict bias
predictions = predictor.predict(X_test)

# Get SHAP values
shap_values = predictor.get_shap_values()
```

### Strengths & Limitations

**Strengths:**
- Best out-of-the-box accuracy
- Handles non-linear relationships
- Robust to outliers
- SHAP provides trustworthy feature importance

**Limitations:**
- Less interpretable than linear models
- Can overfit with small datasets
- No analytical uncertainty (uses ensemble variance)

---

## Polynomial Chaos Expansion (PCE)

Represents the bias function as a polynomial expansion in the input features. Provides analytical sensitivity indices (Sobol) and smooth predictions.

### When to Use

- You need **Sobol sensitivity indices** for variance decomposition
- The response is expected to be **smooth and polynomial-like**
- You want **analytical uncertainty quantification**
- Interpretability of polynomial coefficients is valuable

### Configuration

```yaml
error_prediction:
  regressor: PCE
  regressor_params:
    degree: 3                 # Maximum polynomial degree (default: 3)
    q_norm: 0.75              # Hyperbolic truncation parameter (default: 0.75)
    fit_type: LeastSquares    # Fitting method: LeastSquares or LARS
```

### Parameters Explained

| Parameter | Description | Typical Range |
|-----------|-------------|---------------|
| `degree` | Maximum total polynomial degree | 2-5 |
| `q_norm` | Controls interaction term truncation (lower = sparser) | 0.5-1.0 |
| `fit_type` | `LeastSquares` (dense) or `LARS` (sparse) | - |

**Polynomial basis:**

For features $(x_1, x_2)$ with `degree=2`:
$$f(x) = c_0 + c_1 x_1 + c_2 x_2 + c_{11} x_1^2 + c_{12} x_1 x_2 + c_{22} x_2^2$$

### Sobol Sensitivity Indices

PCE automatically computes Sobol indices for global sensitivity analysis:

| Index | Meaning |
|-------|---------|
| First-order ($S_i$) | Variance from feature $i$ alone |
| Total-order ($S_{Ti}$) | Variance from $i$ including all interactions |

```
Output files:
  sobol_indices.csv          # Numerical values
  sobol_barplot.png          # Visualization
```

### API Usage

```python
from wifa_uq.postprocessing.error_prediction import BiasPredictor

predictor = BiasPredictor(
    database,
    regressor_name="PCE",
    regressor_params={"degree": 3, "q_norm": 0.75},
    feature_names=["ABL_height", "wind_veer", "lapse_rate"]
)

predictor.fit(calibrated_sample_indices)

# Get Sobol indices
sobol_first, sobol_total = predictor.get_sobol_indices()
print("First-order Sobol indices:")
for feat, val in zip(predictor.feature_names, sobol_first):
    print(f"  {feat}: {val:.3f}")
```

### Strengths & Limitations

**Strengths:**
- Analytical Sobol indices for sensitivity analysis
- Smooth, continuous predictions
- Well-suited for UQ workflows
- Interpretable polynomial coefficients

**Limitations:**
- Assumes polynomial response structure
- Can struggle with discontinuities or sharp transitions
- Number of terms grows with degree and features
- Requires careful degree selection

---

## Sliced Inverse Regression Polynomial (SIRPolynomial)

Combines Sliced Inverse Regression (SIR) for dimension reduction with polynomial regression. Finds low-dimensional projections of features that best explain the response.

### When to Use

- You have **many features** and suspect lower-dimensional structure
- You want **dimension reduction** combined with prediction
- Interpretability of the **projection directions** is valuable
- Linear combinations of features drive the response

### Configuration

```yaml
error_prediction:
  regressor: SIRPolynomial
  regressor_params:
    n_directions: 2           # Number of SIR directions to keep (default: 2)
    n_slices: 10              # Number of slices for SIR (default: 10)
    degree: 2                 # Polynomial degree in reduced space (default: 2)
```

### How It Works

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         SIR Polynomial Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Original Features (p dims)        SIR Directions (d dims)     Polynomial   │
│  ┌─────────────────────┐          ┌─────────────────────┐     ┌─────────┐   │
│  │ ABL_height          │          │ z₁ = w₁·X           │     │         │   │
│  │ wind_veer           │   SIR    │ z₂ = w₂·X           │ ──► │ ŷ=P(z)  │   │
│  │ lapse_rate          │   ───►   │                     │     │         │   │
│  │ TI                  │          │ (d << p)            │     └─────────┘   │
│  │ ...                 │          └─────────────────────┘                   │
│  └─────────────────────┘                                                    │
│       (p=10)                            (d=2)                               │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

1. **SIR** finds directions $w_i$ such that $z_i = w_i \cdot X$ captures response variation
2. **Polynomial** fits $\hat{y} = P(z_1, z_2, ...)$ in the reduced space

### Parameters Explained

| Parameter | Description | Guidance |
|-----------|-------------|----------|
| `n_directions` | Dimensionality of reduced space | Start with 1-2, increase if underfitting |
| `n_slices` | Slices for estimating inverse regression | 5-20, more for larger datasets |
| `degree` | Polynomial degree in reduced space | 1-3 |

### Interpreting Directions

The SIR directions reveal which feature combinations matter:

```python
predictor = BiasPredictor(
    database,
    regressor_name="SIRPolynomial",
    regressor_params={"n_directions": 2},
    feature_names=features
)
predictor.fit(calibrated_indices)

# Get SIR directions
directions = predictor.get_sir_directions()
print("SIR Direction 1:")
for feat, weight in zip(features, directions[0]):
    print(f"  {feat}: {weight:.3f}")
```

Example output:
```
SIR Direction 1:
  ABL_height: 0.72
  wind_veer: 0.45
  lapse_rate: 0.12
  TI: 0.52
```

This indicates the first direction is dominated by ABL_height and TI.

### Strengths & Limitations

**Strengths:**
- Effective dimension reduction
- Interpretable projection directions
- Handles many features gracefully
- Can reveal unexpected feature combinations

**Limitations:**
- Assumes linear projections capture response structure
- Sensitive to `n_slices` parameter
- Less common, fewer reference implementations
- May miss complex non-linear interactions

---

## Linear Regressors

Simple and interpretable models for baseline comparisons and when linear relationships are expected.

### Available Variants

| Variant | `regressor_params.linear_type` | Regularization | Best For |
|---------|-------------------------------|----------------|----------|
| Ridge | `ridge` (default) | L2 (shrinkage) | Collinear features, default |
| Lasso | `lasso` | L1 (sparsity) | Feature selection |
| ElasticNet | `elasticnet` | L1 + L2 | Mixed benefits |
| OLS | `ols` | None | Simple baseline |

### Configuration

**Ridge (default):**
```yaml
error_prediction:
  regressor: Linear
  regressor_params:
    linear_type: ridge
    alpha: 1.0                # Regularization strength
```

**Lasso:**
```yaml
error_prediction:
  regressor: Linear
  regressor_params:
    linear_type: lasso
    alpha: 0.1
```

**ElasticNet:**
```yaml
error_prediction:
  regressor: Linear
  regressor_params:
    linear_type: elasticnet
    alpha: 0.5
    l1_ratio: 0.5            # Balance between L1 and L2
```

### Interpreting Coefficients

Linear models provide direct coefficient interpretation:

```python
predictor = BiasPredictor(
    database,
    regressor_name="Linear",
    regressor_params={"linear_type": "ridge", "alpha": 1.0},
    feature_names=features
)
predictor.fit(calibrated_indices)

# Get coefficients
coefs = predictor.get_coefficients()
print("Feature coefficients:")
for feat, coef in zip(features, coefs):
    print(f"  {feat}: {coef:.4f}")
```

**Interpretation:** A coefficient of 0.002 for ABL_height means a 1-unit increase in ABL_height increases predicted bias by 0.002 (normalized units).

### When to Use Each

| Scenario | Recommended |
|----------|-------------|
| Baseline comparison | Ridge or OLS |
| Many correlated features | Ridge |
| Want automatic feature selection | Lasso |
| Uncertain about regularization type | ElasticNet |
| Very few features, lots of data | OLS |

### Strengths & Limitations

**Strengths:**
- Highly interpretable coefficients
- Fast training and prediction
- Analytical uncertainty estimates
- Good baseline for comparison

**Limitations:**
- Cannot capture non-linear relationships
- May underfit complex responses
- Feature engineering required for interactions

---

## Model Selection Guide

### Decision Tree

```
                        Start Here
                             │
                             ▼
              ┌─────────────────────────────┐
              │  Need Sobol sensitivity     │
              │  indices?                   │
              └─────────────────────────────┘
                      │           │
                     Yes          No
                      │           │
                      ▼           ▼
                    PCE    ┌─────────────────────────────┐
                           │  Many features (>5) with    │
                           │  suspected low-dim structure?│
                           └─────────────────────────────┘
                                  │           │
                                 Yes          No
                                  │           │
                                  ▼           ▼
                           SIRPolynomial  ┌─────────────────────────────┐
                                          │  Need best accuracy or     │
                                          │  SHAP importance?          │
                                          └─────────────────────────────┘
                                                  │           │
                                                 Yes          No
                                                  │           │
                                                  ▼           ▼
                                                XGB      ┌─────────────────────┐
                                                         │  Simple baseline or │
                                                         │  linear expected?   │
                                                         └─────────────────────┘
                                                                  │
                                                                  ▼
                                                               Linear
```

### Comparison by Criterion

| Criterion | Best Choice | Second Choice |
|-----------|-------------|---------------|
| Predictive accuracy | XGB | PCE |
| Interpretability | Linear | SIRPolynomial |
| Sensitivity analysis | PCE (Sobol) | XGB (SHAP) |
| Small dataset (<50) | Linear | PCE (low degree) |
| Many features | SIRPolynomial | XGB |
| Production deployment | XGB or Linear | PCE |

---

## Common Configuration Patterns

### High-Accuracy Setup

```yaml
error_prediction:
  regressor: XGB
  regressor_params:
    max_depth: 4
    n_estimators: 200
    learning_rate: 0.05
    min_child_weight: 3
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
    - wind_speed
```

### Sensitivity Analysis Setup

```yaml
error_prediction:
  regressor: PCE
  regressor_params:
    degree: 3
    q_norm: 0.75
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity

sensitivity_analysis:
  output_sobol: true
```

### Interpretable Baseline

```yaml
error_prediction:
  regressor: Linear
  regressor_params:
    linear_type: ridge
    alpha: 1.0
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
```

### Dimension Reduction Setup

```yaml
error_prediction:
  regressor: SIRPolynomial
  regressor_params:
    n_directions: 2
    n_slices: 10
    degree: 2
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
    - capping_inversion_strength
    - capping_inversion_thickness
    - wind_speed
    - wind_direction
```

---

## Troubleshooting

### "XGBoost is overfitting"

**Symptoms:** Training RMSE much lower than test RMSE.

**Solutions:**
- Reduce `max_depth` (try 2-3)
- Reduce `n_estimators`
- Increase `min_child_weight`
- Add regularization: `reg_alpha`, `reg_lambda`

### "PCE predictions are unstable"

**Symptoms:** Large prediction variance, coefficients vary across CV folds.

**Solutions:**
- Reduce `degree`
- Increase `q_norm` closer to 1.0
- Use `fit_type: LARS` for sparse solution
- Ensure features are scaled

### "Linear model has high bias"

**Symptoms:** Both training and test errors are high.

**Solutions:**
- Add polynomial features manually
- Add interaction terms
- Switch to XGB or PCE for non-linear capture
- Check if features are properly normalized

### "SIRPolynomial gives poor results"

**Symptoms:** Predictions don't track actual values.

**Solutions:**
- Increase `n_directions`
- Adjust `n_slices` (try 5-20)
- Increase `degree` in reduced space
- Verify sufficient data for dimension reduction

---

## See Also

- [Feature Engineering](feature_engineering.md) — Available features and custom features
- [Cross-Validation](cross_validation.md) — Validation strategies
- [Global Calibration](../calibration/global_calibration.md) — Calibration before bias prediction
- [Configuration Reference](../configuration.md) — Full YAML options
