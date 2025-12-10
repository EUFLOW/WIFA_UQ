# Calibration Theory

Calibration is the process of adjusting model parameters to improve agreement with reference data. WIFA-UQ supports multiple calibration strategies, each with different trade-offs.

## The Calibration Problem

Given:
- A wake model M with parameters θ = (k_b, α, ...)
- Reference data y_ref (from LES, SCADA, measurements)
- A set of flow cases i = 1, ..., N

Find parameters θ* that minimize some measure of discrepancy:

```
θ* = argmin_θ L(M(θ), y_ref)
```

Where L is a loss function (MSE, MAE, etc.).

## Why Calibrate?

Default parameter values come from:
- Literature averages across many sites
- Idealized experiments (wind tunnels, single-wake LES)
- Theoretical derivations with simplifying assumptions

These defaults may not match your specific:
- Site conditions (terrain, climate)
- Turbine type (rotor size, hub height)
- Atmospheric regime (offshore vs onshore, stable vs unstable)

Calibration adapts the model to your context.

## Global Calibration

**Definition**: Find a single set of parameters that works best across all conditions.

```
θ*_global = argmin_θ Σᵢ |bias(θ, caseᵢ)|
```

### MinBiasCalibrator

WIFA-UQ's default global calibrator finds the parameter sample with minimum total absolute bias:

```python
from wifa_uq.postprocessing.calibration import MinBiasCalibrator

calibrator = MinBiasCalibrator(database)
calibrator.fit()

print(f"Best sample index: {calibrator.best_idx_}")
print(f"Best parameters: {calibrator.best_params_}")
# e.g., {'k_b': 0.042, 'ss_alpha': 0.91}
```

**Algorithm**:
1. For each parameter sample s in the database
2. Sum absolute bias across all cases: `total_bias[s] = Σᵢ |bias[s, i]|`
3. Select s* = argmin total_bias

### DefaultParams

Uses literature default values without optimization:

```python
from wifa_uq.postprocessing.calibration import DefaultParams

calibrator = DefaultParams(database)
calibrator.fit()
# Returns sample closest to default values from metadata
```

Useful as a baseline for comparison.

### Pros and Cons of Global Calibration

| Pros | Cons |
|------|------|
| Simple and interpretable | Compromises across conditions |
| Robust with limited data | Can't capture condition-dependent behavior |
| Single parameter set to deploy | May underfit complex patterns |
| Fast to compute | Optimal for "average" conditions only |

### When to Use Global Calibration

- Limited reference data (< 50 cases)
- Relatively homogeneous conditions
- Deployment simplicity is important
- As a baseline before trying local calibration

## Local Calibration

**Definition**: Optimal parameters vary as a function of operating conditions.

```
θ*(x) = f(ABL_height, wind_veer, TI, ...)
```

For each flow case, find the best parameters for that specific condition.

### LocalParameterPredictor

WIFA-UQ's local calibrator trains an ML model to predict optimal parameters:

```python
from wifa_uq.postprocessing.calibration import LocalParameterPredictor

calibrator = LocalParameterPredictor(
    database,
    feature_names=['ABL_height', 'wind_veer', 'lapse_rate'],
    regressor_name='Ridge',  # or 'RandomForest', 'XGB', etc.
    regressor_params={'alpha': 1.0}
)
calibrator.fit()

# Predict optimal parameters for new conditions
new_features = pd.DataFrame({'ABL_height': [500], 'wind_veer': [0.005], ...})
optimal_params = calibrator.predict(new_features)
```

**Algorithm**:
1. For each case i in the training set, find the sample s*_i with minimum |bias|
2. Extract the optimal parameter values at s*_i for each case
3. Train: θ*(features) using regression (Ridge, RF, XGBoost, etc.)

### Available Regressors

| Regressor | Config Name | Best For |
|-----------|-------------|----------|
| Ridge Regression | `Ridge` | Smooth, regularized relationships |
| Linear Regression | `Linear` | Simple baselines |
| Lasso | `Lasso` | Feature selection |
| ElasticNet | `ElasticNet` | Mixed L1/L2 regularization |
| Random Forest | `RandomForest` | Non-linear, interactions |
| XGBoost | `XGB` | Complex patterns, larger datasets |

### Pros and Cons of Local Calibration

| Pros | Cons |
|------|------|
| Adapts to conditions | Requires more reference data |
| Can capture complex patterns | Risk of overfitting |
| Often better test performance | More complex to deploy |
| Exploits physical relationships | Harder to interpret |

### When to Use Local Calibration

- Sufficient reference data (> 100 cases recommended)
- Heterogeneous conditions in your dataset
- Clear physical reasons for parameter variation
- Willing to accept added complexity

## Comparing Calibration Strategies

WIFA-UQ makes it easy to compare strategies via cross-validation:

```yaml
# Global calibration config
error_prediction:
  calibrator: MinBiasCalibrator
  calibration_mode: global  # (inferred automatically)

# Local calibration config
error_prediction:
  calibrator: LocalParameterPredictor
  calibration_mode: local
  local_regressor: "Ridge"
  local_regressor_params:
    alpha: 1.0
```

Run both and compare CV metrics:

| Calibration | RMSE | R² |
|-------------|------|-----|
| Global (MinBias) | 0.045 | 0.72 |
| Local (Ridge) | 0.032 | 0.85 |

## The Calibration-Prediction Pipeline

In WIFA-UQ, calibration is the first stage of a two-stage pipeline:

```
┌─────────────────────────────────────────────────────────────┐
│                                                             │
│   Stage 1: Calibration                                      │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  Global: θ* = single best parameters                │   │
│   │  Local:  θ*(x) = ML_model(features)                 │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│   Stage 2: Bias Prediction                                  │
│   ┌─────────────────────────────────────────────────────┐   │
│   │  residual_bias = ML_model(features)                 │   │
│   │  Even after calibration, some bias remains          │   │
│   └─────────────────────────────────────────────────────┘   │
│                           │                                 │
│                           ▼                                 │
│   Corrected output = Model(θ*) - residual_bias             │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

The calibration stage reduces systematic bias; the prediction stage learns any remaining patterns.

## Mathematical Framework

### Global Calibration as Optimization

The global calibration problem:

```
θ*_global = argmin_θ Σᵢ L(yᵢ(θ), yᵢ_ref)
```

With the database approach, we discretize θ-space and search:

```
s* = argmin_{s=1,...,S} Σᵢ |bias[s, i]|
θ*_global = θ[s*]
```

This is efficient because the database pre-computes model outputs for all samples.

### Local Calibration as Supervised Learning

The local calibration problem:

```
For each case i: θ*ᵢ = argmin_θ |bias(θ, caseᵢ)|
Then learn: θ*(x) ≈ Σⱼ wⱼ φⱼ(x)  (linear) or tree ensemble, etc.
```

Training data: (xᵢ, θ*ᵢ) pairs where xᵢ are features for case i.

### Connection to Transfer Learning

Local calibration can be viewed as:
- **Domain adaptation**: Parameters adapt to different atmospheric "domains"
- **Meta-learning**: Learning how to calibrate given conditions
- **Conditional modeling**: P(θ|x) instead of point estimate θ*

## Bayesian Calibration

WIFA-UQ also supports Bayesian calibration via UMBRA:

```python
from wifa_uq.postprocessing.bayesian_calibration import BayesianCalibrationWrapper

calibrator = BayesianCalibrationWrapper(
    database,
    system_yaml='path/to/system.yaml',
    param_ranges={'k_b': [0.01, 0.07], 'ss_alpha': [0.75, 1.0]}
)
calibrator.fit()

# Get posterior samples for uncertainty quantification
posterior = calibrator.get_posterior_samples()
```

Bayesian calibration provides:
- **Posterior distribution** over parameters (not just point estimate)
- **Uncertainty quantification** in calibrated values
- **Principled handling** of limited data

See [Uncertainty Quantification](uncertainty_quantification.md) for details.

## Best Practices

### Start Simple

1. Begin with `MinBiasCalibrator` (global)
2. Evaluate cross-validation performance
3. If insufficient, try `LocalParameterPredictor`
4. Compare metrics to quantify improvement

### Feature Selection for Local Calibration

Choose features that:
- Have physical connection to wake behavior
- Vary meaningfully in your dataset
- Are available at prediction time

Good features: ABL_height, wind_veer, stability indicators
Risky features: Turbine-specific measurements (may not generalize)

### Regularization

For local calibration with limited data:
- Use Ridge or Lasso instead of unregularized Linear
- Start with higher regularization (alpha=1.0) and reduce if underfitting
- Random Forest with limited trees (n_estimators=50-100) for small datasets

### Validation Strategy

- **Single farm**: Use K-Fold cross-validation
- **Multiple farms**: Use Leave-One-Group-Out to test generalization

## Summary

| Strategy | Class | Mode | Best For |
|----------|-------|------|----------|
| Minimum bias | `MinBiasCalibrator` | Global | Default choice, limited data |
| Default values | `DefaultParams` | Global | Baseline comparison |
| Feature-based | `LocalParameterPredictor` | Local | Rich data, heterogeneous conditions |
| Bayesian | `BayesianCalibrationWrapper` | Global | Uncertainty quantification |

## Further Reading

- [Model Bias](model_bias.md) — What bias is and why it matters
- [Uncertainty Quantification](uncertainty_quantification.md) — Probabilistic approaches
- [Cross-Validation](../user_guide/error_prediction/cross_validation.md) — Validation strategies
