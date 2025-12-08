# Uncertainty Quantification

Beyond correcting bias, WIFA-UQ provides tools for quantifying and propagating uncertainty through the modeling chain. This page covers the theoretical foundations of PCE, Bayesian methods, and sensitivity analysis.

## Why Quantify Uncertainty?

Even after calibration and bias correction, predictions remain uncertain due to:

1. **Parameter uncertainty** — We don't know the "true" values of k_b, α, etc.
2. **Model structural uncertainty** — The model itself is an approximation
3. **Input uncertainty** — Atmospheric conditions have measurement/forecast errors
4. **Extrapolation uncertainty** — New conditions may differ from training data

Uncertainty quantification (UQ) provides:
- **Confidence intervals** on predictions
- **Risk assessment** for decision-making
- **Sensitivity ranking** of input factors
- **Robust optimization** accounting for variability

## Polynomial Chaos Expansion (PCE)

PCE represents a model output as a polynomial expansion over uncertain inputs:

```
Y(ξ) = Σₐ yₐ Ψₐ(ξ)
```

Where:
- ξ = (ξ₁, ..., ξₙ) are standardized uncertain inputs
- Ψₐ(ξ) are orthogonal polynomial basis functions
- yₐ are expansion coefficients
- α is a multi-index controlling polynomial degree

### Intuition

Instead of running thousands of Monte Carlo samples, PCE:
1. Fits a polynomial surrogate from a modest number of model runs
2. Samples the polynomial cheaply for uncertainty propagation
3. Extracts variance contributions analytically (Sobol indices)

### Implementation in WIFA-UQ

```python
from wifa_uq.postprocessing.error_predictor import PCERegressor

pce = PCERegressor(
    degree=5,           # Maximum polynomial degree
    marginals='kernel', # Fit marginal distributions from data
    copula='independent', # Assume independent inputs
    q=0.5,              # Hyperbolic truncation parameter
    max_features=5,     # Safety limit on input dimension
    allow_high_dim=False
)

pce.fit(X_train, y_train)
y_pred = pce.predict(X_test)
```

### PCE Parameters Explained

| Parameter | Description | Typical Values |
|-----------|-------------|----------------|
| `degree` | Maximum polynomial degree | 3-7 (higher = more flexible, more data needed) |
| `marginals` | How to model input distributions | `'kernel'` (data-driven), `'uniform'`, `'normal'` |
| `copula` | Dependency structure between inputs | `'independent'`, `'normal'` (Gaussian copula) |
| `q` | Hyperbolic truncation (sparsity) | 0.5-1.0 (lower = sparser basis) |

### When PCE Works Well

- **Low-to-moderate dimension** (< 10 inputs recommended)
- **Smooth relationships** (polynomial-like)
- **Sufficient samples** (typically 2-5× the number of basis terms)

### Limitations

- Curse of dimensionality for many inputs
- Struggles with discontinuities or highly non-linear responses
- Requires careful basis selection

## Sobol Sensitivity Indices

Sobol indices decompose output variance into contributions from each input:

```
Var(Y) = Σᵢ Vᵢ + Σᵢ<ⱼ Vᵢⱼ + ... + V₁₂...ₙ
```

### First-Order Index (Sᵢ)

The fraction of variance due to input i alone:

```
Sᵢ = Vᵢ / Var(Y) = Var(E[Y|Xᵢ]) / Var(Y)
```

**Interpretation**: If S_ABL_height = 0.4, then 40% of bias variance is explained by ABL height variation alone.

### Total-Order Index (STᵢ)

The fraction of variance involving input i (including interactions):

```
STᵢ = 1 - Var(E[Y|X₋ᵢ]) / Var(Y)
```

**Interpretation**: If ST_ABL_height = 0.55, then ABL height is involved in 55% of variance (including its interactions with other variables).

### Computing Sobol Indices from PCE

With a fitted PCE, Sobol indices are computed analytically from the coefficients:

```python
from wifa_uq.postprocessing.PCE_tool.pce_utils import run_pce_sensitivity

results = run_pce_sensitivity(
    X=features,
    y=observations,
    feature_names=['ABL_height', 'wind_veer', 'lapse_rate'],
    pce_config={'degree': 5, 'marginals': 'kernel'},
    output_dir=Path('results/')
)

print(results['sobol_first'])   # First-order indices
print(results['sobol_total'])   # Total-order indices
```

### Interpreting Sobol Plots

WIFA-UQ generates bar charts showing:
- **Blue bars**: First-order indices (main effects)
- **Orange bars**: Total-order indices (including interactions)

```
Feature        S1      ST
─────────────────────────
ABL_height    0.40    0.55
wind_veer     0.25    0.30
lapse_rate    0.10    0.15
blockage      0.05    0.10
```

Large gap between ST and S1 indicates important interactions.

## Bayesian Calibration

Bayesian methods treat parameters as random variables with prior distributions updated by data.

### Bayes' Theorem

```
P(θ|data) ∝ P(data|θ) × P(θ)
```

- **P(θ)**: Prior distribution (what we believe before seeing data)
- **P(data|θ)**: Likelihood (how well parameters explain observations)
- **P(θ|data)**: Posterior distribution (updated beliefs)

### Advantages of Bayesian Approach

1. **Full posterior distribution** — Not just a point estimate
2. **Natural uncertainty quantification** — Spread of posterior reflects uncertainty
3. **Principled handling of limited data** — Prior regularizes inference
4. **Propagation to predictions** — Sample parameters → sample outputs

### UMBRA Integration

WIFA-UQ uses UMBRA for Bayesian calibration:

```python
from wifa_uq.postprocessing.bayesian_calibration import BayesianCalibrationWrapper

calibrator = BayesianCalibrationWrapper(
    database,
    system_yaml='path/to/system.yaml',
    param_ranges={
        'k_b': [0.01, 0.07],
        'ss_alpha': [0.75, 1.0]
    }
)
calibrator.fit()

# Posterior samples for UQ
posterior_samples = calibrator.get_posterior_samples()

# Point estimate (median)
print(calibrator.best_params_)
```

### Posterior Predictive Distribution

To get prediction uncertainty:

```python
# For each posterior sample, run the model
predictions = []
for theta in posterior_samples:
    pred = model(theta, new_conditions)
    predictions.append(pred)

# Prediction interval
lower, upper = np.percentile(predictions, [5, 95])
```

### Computational Considerations

Bayesian inference typically requires:
- **MCMC sampling** (expensive) or **variational inference** (approximate)
- Many model evaluations → consider surrogate models
- Careful convergence diagnostics

UMBRA uses TMCMC (Transitional Markov Chain Monte Carlo) for efficient sampling.

## SHAP Values for ML Interpretability

When using tree-based models (XGBoost), SHAP provides local explanations:

```
SHAP value for feature j, instance i = contribution of feature j to prediction i
```

### Global Feature Importance

Average absolute SHAP values across instances:

```python
# Run automatically in WIFA-UQ cross-validation
# Outputs: bias_prediction_shap.png, bias_prediction_shap_importance.png
```

### Beeswarm Plot Interpretation

The SHAP beeswarm plot shows:
- **X-axis**: SHAP value (impact on prediction)
- **Y-axis**: Features (ranked by importance)
- **Color**: Feature value (red = high, blue = low)
- **Each point**: One instance

**Example interpretation**:
- High ABL_height (red) → negative SHAP → reduces predicted bias
- Low ABL_height (blue) → positive SHAP → increases predicted bias

### SHAP vs Sobol

| Aspect | SHAP | Sobol |
|--------|------|-------|
| **Model** | Any (esp. trees) | PCE surrogate |
| **Type** | Local (per instance) | Global (variance-based) |
| **Interactions** | Via interaction values | Via ST - S1 |
| **Output** | Feature contributions | Variance fractions |

Use SHAP for XGBoost, Sobol for PCE models.

## SIR-Based Sensitivity

Sliced Inverse Regression (SIR) finds directions in feature space that best explain response variation.

### Concept

SIR identifies a linear combination β'X that captures the relationship between inputs and output:

```
Y ≈ f(β'X) where β is a direction vector
```

The coefficients β indicate feature importance in that direction.

### Implementation

```python
from wifa_uq.postprocessing.error_predictor import SIRPolynomialRegressor

model = SIRPolynomialRegressor(n_directions=1, degree=2)
model.fit(X, y)

importance = model.get_feature_importance(feature_names)
```

### Interpretation

Larger absolute coefficients in β → more important feature for dimension reduction.

WIFA-UQ generates:
- `observation_sensitivity_sir.png` — Bar chart of SIR direction coefficients
- `observation_sensitivity_sir_shadow.png` — Scatter of projected data vs response

## Uncertainty Propagation Workflow

### For PCE-Based Analysis

```yaml
error_prediction:
  model: "PCE"
  model_params:
    degree: 5
    marginals: "kernel"

sensitivity_analysis:
  run_observation_sensitivity: true
  method: "pce_sobol"
  pce_config:
    degree: 5
    marginals: "kernel"
```

### For XGBoost + SHAP

```yaml
error_prediction:
  model: "XGB"

sensitivity_analysis:
  run_bias_sensitivity: true
  # SHAP is automatic for tree models
```

### For Bayesian UQ

```yaml
error_prediction:
  calibrator: BayesianCalibration
  # Requires system_yaml and param_ranges
```

## Practical Recommendations

### Choosing a UQ Method

| Situation | Recommended Approach |
|-----------|---------------------|
| Few inputs (< 5), smooth response | PCE + Sobol |
| Many inputs, complex patterns | XGBoost + SHAP |
| Need full parameter distribution | Bayesian (UMBRA) |
| Quick feature ranking | SIR |
| Limited data | Start with simpler methods |

### Sanity Checks

1. **Sum of S1 indices** should be close to 1 (if independent inputs)
2. **ST ≥ S1** always (total includes interactions)
3. **SHAP values sum to prediction - baseline** (by construction)
4. **Posterior should narrow** with more data

### Common Pitfalls

- **Overfitting PCE** — Use cross-validation to select degree
- **Ignoring interactions** — Check ST vs S1 gap
- **Extrapolation** — UQ trained on limited conditions may not transfer
- **Computational cost** — High-degree PCE or Bayesian can be slow

## Summary

| Method | Output | Best For |
|--------|--------|----------|
| **PCE + Sobol** | Variance decomposition | Global sensitivity, few inputs |
| **SHAP** | Per-instance explanations | Tree models, many inputs |
| **SIR** | Dimension reduction | Feature ranking, visualization |
| **Bayesian** | Posterior distribution | Full UQ, limited data |

## Further Reading

- [Model Bias](model_bias.md) — What we're trying to quantify uncertainty for
- [Calibration Theory](calibration_theory.md) — How parameters are estimated
- [PCE Tool README](https://github.com/EUFLOW/WIFA-UQ/tree/main/wifa_uq/postprocessing/PCE_tool) — Standalone PCE analysis
- OpenTURNS documentation for PCE details
- SHAP documentation for interpretation guidelines
