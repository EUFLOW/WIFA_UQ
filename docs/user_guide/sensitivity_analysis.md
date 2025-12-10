# Sensitivity Analysis

Sensitivity analysis reveals which features most strongly influence model bias predictions. WIFA-UQ provides three complementary approaches: SHAP values, Sobol indices, and SIR directions.

## Overview

Different sensitivity methods answer different questions:

| Method | Question Answered | Model Required |
|--------|-------------------|----------------|
| **SHAP** | How does each feature affect *this specific* prediction? | XGBoost |
| **Sobol** | What fraction of output *variance* comes from each feature? | PCE |
| **SIR Directions** | What *linear combinations* of features drive the response? | SIRPolynomial |

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    Sensitivity Analysis Overview                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  SHAP (Local + Global)                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  For each prediction:                                               │    │
│  │  ŷ = baseline + φ(ABL_height) + φ(wind_veer) + φ(lapse_rate) + ...  │    │
│  │                                                                     │    │
│  │  "This prediction is high because ABL_height pushed it up by 0.02"  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  Sobol (Global Variance Decomposition)                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Var(ŷ) = V_ABL + V_veer + V_lapse + V_interactions                 │    │
│  │                                                                     │    │
│  │  "45% of bias variance is explained by ABL_height alone"            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  SIR Directions (Dimension Reduction)                                       │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  z₁ = 0.7·ABL + 0.5·veer + 0.1·lapse  (most important direction)    │    │
│  │  z₂ = 0.2·ABL - 0.3·veer + 0.8·lapse  (second direction)            │    │
│  │                                                                     │    │
│  │  "The combination of high ABL and high veer drives predictions"     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```yaml
error_prediction:
  regressor: XGB                    # For SHAP
  # regressor: PCE                  # For Sobol
  # regressor: SIRPolynomial        # For SIR directions
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity

sensitivity_analysis:
  enabled: true
  output_dir: sensitivity_results/
```

---

## SHAP Analysis

**SH**apley **A**dditive ex**P**lanations decompose each prediction into feature contributions based on game-theoretic principles.

### How SHAP Works

For each prediction, SHAP computes how much each feature contributed:

```
Prediction = E[ŷ] + Σ SHAP_value(feature_i)

Example:
  Base prediction (mean): 0.02
  + ABL_height contribution: +0.015
  + wind_veer contribution: -0.008
  + lapse_rate contribution: +0.003
  + turbulence_intensity contribution: +0.005
  ─────────────────────────────────────
  Final prediction: 0.035
```

### Configuration

```yaml
error_prediction:
  regressor: XGB
  regressor_params:
    max_depth: 3
    n_estimators: 100
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity

sensitivity_analysis:
  enabled: true
  shap:
    output_summary_plot: true
    output_dependence_plots: true
    output_values_csv: true
```

### Output Files

| File | Description |
|------|-------------|
| `shap_summary.png` | Beeswarm plot showing all SHAP values |
| `shap_feature_importance.png` | Bar chart of mean |SHAP| per feature |
| `shap_dependence_*.png` | Dependence plots for each feature |
| `shap_values.csv` | Raw SHAP values for all predictions |

### Interpreting SHAP Plots

**Beeswarm Plot (`shap_summary.png`):**

```
           ◄─── Negative impact    Positive impact ───►

ABL_height    ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●
              (many high values push prediction up)

wind_veer     ●●●●●●●●●●●●●●●●●●●●●●
              (moderate impact, both directions)

lapse_rate    ●●●●●●●●●●●●●●
              (smaller impact)

TI            ●●●●●●●●
              (least important)

Color: Blue = low feature value, Red = high feature value
```

**Reading the beeswarm plot:**
- Each dot is one prediction
- X-axis: SHAP value (impact on prediction)
- Color: Feature value (red = high, blue = low)
- Vertical spread: Distribution of impacts

**Feature Importance Bar Chart (`shap_feature_importance.png`):**

Shows mean |SHAP value| per feature — overall importance regardless of direction.

**Dependence Plots (`shap_dependence_*.png`):**

Show how SHAP values change with feature values:

```
SHAP value
    │      ●  ●
    │    ●  ●● ●●●
    │   ●●●●●●●●●●●
    │  ●●●●●●●●●
0 ──┼──●●●●──────────── ABL_height
    │ ●●●
    │●●
    │
```

Color often shows interaction with another feature.

### API Usage

```python
from wifa_uq.postprocessing.error_prediction import BiasPredictor
from wifa_uq.postprocessing.sensitivity import SHAPAnalyzer
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

# Train predictor
predictor = BiasPredictor(
    database,
    regressor_name="XGB",
    feature_names=["ABL_height", "wind_veer", "lapse_rate", "turbulence_intensity"]
)
predictor.fit(calibrated_sample_indices)

# Run SHAP analysis
shap_analyzer = SHAPAnalyzer(predictor)
shap_values = shap_analyzer.compute_shap_values()

# Get feature importance ranking
importance = shap_analyzer.get_feature_importance()
print("Feature importance (mean |SHAP|):")
for feat, imp in importance.items():
    print(f"  {feat}: {imp:.4f}")

# Generate plots
shap_analyzer.plot_summary(save_path="shap_summary.png")
shap_analyzer.plot_dependence("ABL_height", save_path="shap_dep_abl.png")
```

### SHAP Best Practices

| Practice | Rationale |
|----------|-----------|
| Use sufficient data | SHAP needs enough samples for reliable estimates |
| Check for interactions | Color in dependence plots reveals interactions |
| Compare with domain knowledge | Validate that important features make physical sense |
| Use for debugging | Unexpected importance may indicate data issues |

---

## Sobol Sensitivity Indices

Sobol indices decompose output variance into contributions from each feature and their interactions. Available when using PCE (Polynomial Chaos Expansion).

### How Sobol Works

Total variance is partitioned:

```
Var(Y) = Σ Vᵢ + Σ Vᵢⱼ + Σ Vᵢⱼₖ + ...

Where:
  Vᵢ   = Variance from feature i alone (first-order)
  Vᵢⱼ  = Additional variance from interaction of i and j
  Vᵢⱼₖ = Three-way interaction variance
  ...
```

**First-order index** (Sᵢ): Fraction of variance from feature i alone
```
Sᵢ = Vᵢ / Var(Y)
```

**Total-order index** (Sᵀᵢ): Fraction of variance involving feature i (including all interactions)
```
Sᵀᵢ = (Vᵢ + Σⱼ Vᵢⱼ + Σⱼₖ Vᵢⱼₖ + ...) / Var(Y)
```

### Configuration

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
  enabled: true
  sobol:
    output_indices_csv: true
    output_bar_plot: true
    output_pie_chart: true
```

### Output Files

| File | Description |
|------|-------------|
| `sobol_indices.csv` | First-order and total indices for each feature |
| `sobol_barplot.png` | Grouped bar chart comparing S₁ and Sᵀ |
| `sobol_pie.png` | Pie chart of first-order contributions |

### Interpreting Sobol Indices

**Example output (`sobol_indices.csv`):**

| Feature | S1 (First-order) | ST (Total) |
|---------|------------------|------------|
| ABL_height | 0.45 | 0.52 |
| wind_veer | 0.25 | 0.31 |
| lapse_rate | 0.15 | 0.20 |
| turbulence_intensity | 0.08 | 0.12 |

**Interpretation:**
- ABL_height alone explains 45% of variance (S1=0.45)
- ABL_height including interactions explains 52% (ST=0.52)
- The gap (0.52 - 0.45 = 0.07) indicates interaction effects
- Sum of S1 ≤ 1.0; sum of ST can exceed 1.0 due to shared interaction variance

**Key insights:**
- If S1 ≈ ST: Feature acts mostly independently
- If ST >> S1: Feature has strong interactions with others
- If Σ S1 << 1: Significant interaction effects exist

### API Usage

```python
from wifa_uq.postprocessing.error_prediction import BiasPredictor
from wifa_uq.postprocessing.sensitivity import SobolAnalyzer
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

# Train PCE predictor
predictor = BiasPredictor(
    database,
    regressor_name="PCE",
    regressor_params={"degree": 3},
    feature_names=["ABL_height", "wind_veer", "lapse_rate", "turbulence_intensity"]
)
predictor.fit(calibrated_sample_indices)

# Compute Sobol indices (analytical from PCE coefficients)
sobol_analyzer = SobolAnalyzer(predictor)
S1, ST = sobol_analyzer.compute_indices()

print("Sobol Sensitivity Indices:")
print("-" * 50)
print(f"{'Feature':<25} {'S1':>10} {'ST':>10}")
print("-" * 50)
for feat, s1, st in zip(predictor.feature_names, S1, ST):
    print(f"{feat:<25} {s1:>10.3f} {st:>10.3f}")

# Check for interaction effects
total_s1 = sum(S1)
print(f"\nSum of S1: {total_s1:.3f}")
print(f"Interaction contribution: {1 - total_s1:.3f} ({(1-total_s1)*100:.1f}%)")

# Generate plots
sobol_analyzer.plot_barplot(save_path="sobol_barplot.png")
```

### Sobol Best Practices

| Practice | Rationale |
|----------|-----------|
| Check Σ S1 | If << 1, interactions are important |
| Compare S1 vs ST | Large gaps indicate interaction-driven features |
| Use adequate PCE degree | Too low may miss non-linear effects |
| Validate with physical intuition | Important features should make physical sense |

---

## SIR Directions

Sliced Inverse Regression (SIR) finds linear combinations of features that best explain the response. Unlike SHAP (local) or Sobol (variance), SIR reveals the **effective dimensionality** of the feature space.

### How SIR Works

SIR discovers that the response Y depends on features X primarily through a few linear combinations:

```
Y ≈ f(w₁ᵀX, w₂ᵀX, ...)

Where:
  w₁ = [0.7, 0.5, 0.1, 0.2]  → First direction (most important)
  w₂ = [0.2, -0.3, 0.8, 0.1] → Second direction

Interpretation:
  Direction 1 = 0.7·ABL + 0.5·veer + 0.1·lapse + 0.2·TI
  "High ABL combined with high veer" is the primary driver
```

### Configuration

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

sensitivity_analysis:
  enabled: true
  sir:
    output_directions_csv: true
    output_projection_plot: true
    output_loadings_plot: true
```

### Output Files

| File | Description |
|------|-------------|
| `sir_directions.csv` | Direction vectors (feature loadings) |
| `sir_projections.png` | Data projected onto SIR directions |
| `sir_loadings.png` | Bar chart of loadings per direction |
| `sir_eigenvalues.png` | Eigenvalue spectrum (direction importance) |

### Interpreting SIR Directions

**Example output (`sir_directions.csv`):**

| Feature | Direction 1 | Direction 2 |
|---------|-------------|-------------|
| ABL_height | 0.72 | 0.18 |
| wind_veer | 0.48 | -0.35 |
| lapse_rate | 0.12 | 0.82 |
| turbulence_intensity | 0.45 | 0.21 |

**Interpretation:**
- **Direction 1** is dominated by ABL_height (0.72) and wind_veer (0.48)
  - "Cases with high ABL and high veer behave similarly"
- **Direction 2** is dominated by lapse_rate (0.82)
  - "Stability (lapse_rate) provides independent information"

**Projection plot insight:**
```
Direction 2
    │           ●
    │        ●  ●●●  (stable cases)
    │      ●●●●●●●
    │    ●●●●●●●●●
────┼──●●●●●●●●●●●●●────── Direction 1
    │  ●●●●●●●●●
    │    ●●●●●  (unstable cases)
    │      ●
    │

Color = bias value → reveals structure in reduced space
```

### API Usage

```python
from wifa_uq.postprocessing.error_prediction import BiasPredictor
from wifa_uq.postprocessing.sensitivity import SIRAnalyzer
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

# Train SIRPolynomial predictor
predictor = BiasPredictor(
    database,
    regressor_name="SIRPolynomial",
    regressor_params={"n_directions": 2, "n_slices": 10},
    feature_names=["ABL_height", "wind_veer", "lapse_rate", "turbulence_intensity"]
)
predictor.fit(calibrated_sample_indices)

# Analyze SIR directions
sir_analyzer = SIRAnalyzer(predictor)
directions = sir_analyzer.get_directions()
eigenvalues = sir_analyzer.get_eigenvalues()

print("SIR Directions (feature loadings):")
print("-" * 60)
for i, direction in enumerate(directions):
    print(f"\nDirection {i+1} (eigenvalue: {eigenvalues[i]:.3f}):")
    for feat, loading in zip(predictor.feature_names, direction):
        bar = "█" * int(abs(loading) * 20)
        sign = "+" if loading >= 0 else "-"
        print(f"  {feat:<20} {sign}{abs(loading):.3f} {bar}")

# Plot projections
sir_analyzer.plot_projections(save_path="sir_projections.png")
```

### SIR Best Practices

| Practice | Rationale |
|----------|-----------|
| Start with n_directions=2 | Visualizable; increase if needed |
| Check eigenvalue spectrum | Large gap suggests correct dimensionality |
| Interpret loadings physically | Directions should make physical sense |
| Compare projection plot colors | Structured patterns confirm useful reduction |

---

## Comparing Methods

### When to Use Each

| Scenario | Recommended Method |
|----------|-------------------|
| Explain individual predictions | SHAP |
| Quantify variance contributions | Sobol |
| Find effective dimensionality | SIR |
| Feature selection | SHAP or Sobol (high importance = keep) |
| Understand interactions | Sobol (S1 vs ST gap) or SHAP dependence |
| Communicate to stakeholders | SHAP (intuitive plots) |

### Running Multiple Analyses

You can run multiple sensitivity methods by using different regressors:

```python
# SHAP analysis with XGBoost
predictor_xgb = BiasPredictor(database, regressor_name="XGB", feature_names=features)
predictor_xgb.fit(calibrated_indices)
shap_results = SHAPAnalyzer(predictor_xgb).compute_shap_values()

# Sobol analysis with PCE
predictor_pce = BiasPredictor(database, regressor_name="PCE", feature_names=features)
predictor_pce.fit(calibrated_indices)
sobol_S1, sobol_ST = SobolAnalyzer(predictor_pce).compute_indices()

# SIR analysis
predictor_sir = BiasPredictor(database, regressor_name="SIRPolynomial", feature_names=features)
predictor_sir.fit(calibrated_indices)
sir_directions = SIRAnalyzer(predictor_sir).get_directions()
```

### Cross-Method Validation

Consistent results across methods increase confidence:

```python
import pandas as pd
import numpy as np

# Aggregate importance rankings
rankings = pd.DataFrame({
    "SHAP": np.argsort(-shap_importance),
    "Sobol_S1": np.argsort(-sobol_S1),
    "Sobol_ST": np.argsort(-sobol_ST),
    "SIR_Dir1": np.argsort(-np.abs(sir_directions[0]))
}, index=features)

print("Feature importance rankings (1=most important):")
print(rankings + 1)  # Convert to 1-indexed

# Check consistency
if (rankings.std(axis=1) < 1).all():
    print("\n✓ Rankings are consistent across methods")
else:
    print("\n⚠ Rankings differ — examine why")
```

---

## Sensitivity Analysis Workflow

### Complete Analysis Pipeline

```yaml
# config_sensitivity.yaml
paths:
  database: results_stacked_hh.nc
  output_dir: sensitivity_analysis/

error_prediction:
  calibrator: MinBiasCalibrator
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
    - wind_speed

# Run with each regressor for comprehensive analysis
sensitivity_analysis:
  enabled: true

  # XGBoost for SHAP
  run_shap: true
  shap_regressor: XGB
  shap_params:
    max_depth: 3
    n_estimators: 100

  # PCE for Sobol
  run_sobol: true
  sobol_regressor: PCE
  sobol_params:
    degree: 3
    q_norm: 0.75

  # SIR for dimension reduction
  run_sir: true
  sir_params:
    n_directions: 2
    n_slices: 10
```

### Interpreting Combined Results

**Example summary:**

```
Sensitivity Analysis Summary
════════════════════════════════════════════════════════════════

SHAP Feature Importance (mean |SHAP|):
  1. ABL_height:          0.0234
  2. turbulence_intensity: 0.0156
  3. wind_veer:           0.0098
  4. lapse_rate:          0.0067
  5. wind_speed:          0.0045

Sobol Indices:
                          S1        ST      Interactions
  ABL_height:           0.42      0.51         0.09
  turbulence_intensity: 0.28      0.35         0.07
  wind_veer:            0.12      0.18         0.06
  lapse_rate:           0.08      0.14         0.06
  wind_speed:           0.05      0.09         0.04

  Sum of S1: 0.95 → Low interaction effects overall

SIR Directions:
  Direction 1: 0.68·ABL + 0.52·TI + 0.32·veer + ...
    → "ABL and turbulence dominate"
  Direction 2: 0.75·lapse + 0.41·veer - 0.28·ABL + ...
    → "Stability provides independent information"

Conclusion:
  • ABL_height is the dominant driver across all methods
  • turbulence_intensity is consistently second
  • Low interaction effects (sum S1 ≈ 0.95)
  • Two effective dimensions capture most variation
```

---

## Troubleshooting

### "SHAP values are all near zero"

**Causes:**
- Model has poor predictive power
- Features don't relate to target
- Target has very low variance

**Solutions:**
- Check model R² (should be reasonable)
- Verify features are correctly loaded
- Examine target distribution

### "Sobol indices don't sum to 1"

**Note:** First-order indices (S1) should sum to ≤ 1. Total indices (ST) can sum to > 1 due to shared interaction variance.

**If S1 sum >> 1:**
- PCE may be poorly fitted
- Try lower polynomial degree
- Check for numerical issues

### "SIR directions are hard to interpret"

**Causes:**
- Features are on different scales
- Too many directions requested
- Weak relationship between features and target

**Solutions:**
- Standardize features before analysis
- Reduce n_directions
- Check eigenvalue spectrum for natural cutoff

### "Different methods give different rankings"

**This can be legitimate!** Methods measure different things:
- SHAP: Prediction contribution
- Sobol: Variance contribution
- SIR: Projection importance

**When to investigate:**
- Rankings differ by > 2 positions
- Most important feature differs
- Physical intuition suggests one is wrong

---

## See Also

- [ML Models](error_prediction/ml_models.md) — Regressor options for sensitivity analysis
- [Feature Engineering](error_prediction/feature_engineering.md) — Features available for analysis
- [Configuration Reference](configuration.md) — Full YAML options
- [Cross-Validation](error_prediction/cross_validation.md) — Validating sensitivity results
