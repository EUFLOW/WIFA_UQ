# Model Bias in Wake Modeling

This page explains what model bias is, why it matters for wind farm performance prediction, and how WIFA-UQ addresses it.

## What is Model Bias?

**Model bias** is the systematic difference between a model's predictions and reality. Unlike random noise, bias is predictable and often depends on operating conditions.

For wake models:

```
bias = P_model - P_reference
```

Where:
- `P_model` is the power predicted by a wake model (PyWake, FOXES, etc.)
- `P_reference` is the "true" power (from LES, SCADA, or measurements)

In WIFA-UQ, we normalize by rated power for comparability across turbines and farms:

```
normalized_bias = (P_model - P_reference) / P_rated
```

## Why Wake Models Have Bias

Engineering wake models make simplifying assumptions that introduce systematic errors:

### 1. Wake Deficit Representation

Most models use analytical functions (Gaussian, Jensen, etc.) to describe the velocity deficit:

```
ΔU/U∞ = f(x, r; k, σ, ...)
```

These parameterizations capture average behavior but miss:
- Meandering and unsteady effects
- Complex turbine interactions
- Near-wake transitions

### 2. Atmospheric Simplifications

Models typically assume:
- Neutral stratification (or simple stability corrections)
- Logarithmic inflow profiles
- Homogeneous turbulence

Reality includes:
- Strong stratification effects on wake recovery
- Low-level jets and complex shear profiles
- Heterogeneous turbulence from terrain and thermal effects

### 3. Uncertain Parameters

Key parameters like wake expansion rate (k_b) and turbulence intensity (TI) are:
- Derived from limited measurements
- Site-specific and condition-dependent
- Often set to literature defaults

### 4. Blockage Effects

Large wind farms create:
- Upstream induction (global blockage)
- Local speedup/slowdown between turbines
- Farm-scale momentum extraction

These effects are challenging to model accurately.

## How Bias Manifests

Bias varies systematically with physical conditions:

### Atmospheric Dependence

| Condition | Typical Bias Pattern |
|-----------|---------------------|
| **Stable atmosphere** | Models often underpredict wake losses (wakes persist longer) |
| **Convective atmosphere** | Models may overpredict wake losses (faster mixing) |
| **High wind veer** | Direction changes cause wake-turbine misalignment |
| **Low ABL height** | Wake interactions with capping inversion |

### Layout Dependence

| Layout Feature | Bias Effect |
|----------------|-------------|
| **Deep arrays** | Cumulative wake interactions amplify errors |
| **Tight spacing** | Near-wake effects harder to model |
| **Irregular layouts** | Superposition models struggle with complex interactions |

### Parameter Sensitivity

Small changes in model parameters can shift bias:

```
Δbias/Δk_b ≈ 0.5-2.0  (per 0.01 change in k_b)
```

This sensitivity is why calibration is valuable.

## The Cost of Ignoring Bias

Uncorrected bias leads to:

### Financial Impact

- **Energy yield estimates**: 5-10% bias → millions in financing errors
- **O&M planning**: Wrong load predictions → suboptimal maintenance
- **Curtailment strategies**: Incorrect wake predictions → lost revenue

### Technical Impact

- **Array optimization**: Biased models → suboptimal layouts
- **Control strategies**: Wake steering based on wrong models
- **Lifetime estimation**: Incorrect load calculations

## WIFA-UQ's Approach to Bias

WIFA-UQ addresses bias through a multi-step strategy:

### Step 1: Characterize the Bias Landscape

Generate a database spanning:
- Multiple parameter samples (k_b, α, TI corrections, etc.)
- Multiple flow cases (different atmospheric conditions)
- Multiple metrics (farm-average power, per-turbine, etc.)

This reveals how bias depends on parameters and conditions.

### Step 2: Calibrate Parameters

Find parameter settings that reduce systematic bias:

**Global calibration**: Single best parameter set
```python
k_b* = argmin_k Σ |bias(k, case)|
```

**Local calibration**: Condition-dependent parameters
```python
k_b*(case) = f(ABL_height, wind_veer, ...)
```

### Step 3: Learn Residual Bias

Even after calibration, residual bias remains. Learn it as a function of features:

```python
residual_bias = ML_model(ABL_height, wind_veer, lapse_rate, blockage_ratio, ...)
```

### Step 4: Apply Correction

The corrected prediction is:

```python
P_corrected = P_model(calibrated_params) - predicted_residual_bias
```

## Bias vs. Uncertainty

It's important to distinguish:

| Concept | Definition | Treatment |
|---------|------------|-----------|
| **Bias** | Systematic, predictable error | Calibration + ML correction |
| **Uncertainty** | Random variability, epistemic gaps | Probabilistic methods (PCE, Bayesian) |

WIFA-UQ addresses both:
- **Bias correction** reduces the mean error
- **Uncertainty quantification** characterizes the remaining spread

## Measuring Success

WIFA-UQ evaluates bias correction via cross-validation:

### Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **RMSE** | √(Σ(pred-true)²/n) | Overall prediction error |
| **R²** | 1 - SS_res/SS_tot | Variance explained |
| **MAE** | Σ\|pred-true\|/n | Average absolute error |

### Visualization

The standard diagnostic plot shows three panels:

1. **ML Model Performance**: Predicted vs true bias (should align with 1:1)
2. **Uncorrected Model**: Raw model vs reference (shows original bias)
3. **Corrected Model**: After bias correction (should be tighter around 1:1)

## Example: Bias Reduction

A typical WIFA-UQ workflow might achieve:

| Stage | Farm-Average RMSE |
|-------|-------------------|
| Uncalibrated model | 8-12% of rated |
| After calibration | 5-8% of rated |
| After ML correction | 2-4% of rated |

The exact improvement depends on:
- Quality and quantity of reference data
- Richness of physical features
- Consistency of bias patterns

## Key Takeaways

1. **Bias is systematic** — It follows patterns that can be learned
2. **Bias depends on conditions** — Atmospheric state, layout, and parameters all matter
3. **Calibration helps but isn't enough** — Residual bias remains
4. **ML can capture complex patterns** — Features like ABL height and wind veer are predictive
5. **Cross-validation is essential** — Ensure corrections generalize to new conditions

## Further Reading

- [Calibration Theory](calibration_theory.md) — Global vs local calibration strategies
- [Uncertainty Quantification](uncertainty_quantification.md) — Probabilistic methods for remaining uncertainty
- [Feature Engineering](../user_guide/error_prediction/feature_engineering.md) — What features predict bias
