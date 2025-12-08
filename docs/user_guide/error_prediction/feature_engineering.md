# Feature Engineering

Features are the atmospheric and layout quantities used to predict model bias. Selecting the right features is crucial for accurate bias prediction and meaningful sensitivity analysis.

## Overview

Features connect physical conditions to model errors:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Feature → Bias Relationship                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Physical Conditions          Features              Model Bias              │
│  ┌─────────────────────┐     ┌─────────────────┐    ┌─────────────────┐     │
│  │ Atmospheric profile │     │ ABL_height      │    │                 │     │
│  │ Wind measurements   │ ──► │ wind_veer       │ ──►│ bias = f(X)     │     │
│  │ Turbulence data     │     │ lapse_rate      │    │                 │     │
│  │ Farm geometry       │     │ TI, Farm_Length │    │                 │     │
│  └─────────────────────┘     └─────────────────┘    └─────────────────┘     │
│                                                                             │
│  Good features:                                                             │
│  • Capture physics that the wake model misses                               │
│  • Vary across your dataset                                                 │
│  • Are reliably measurable/computable                                       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Quick Start

```yaml
error_prediction:
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
```

## Feature Categories

WIFA-UQ features fall into three categories:

| Category | Source | Examples |
|----------|--------|----------|
| **Atmospheric** | Derived from profiles | ABL_height, wind_veer, lapse_rate |
| **Flow** | Direct measurements | wind_speed, wind_direction, TI |
| **Layout** | Farm geometry (wind-direction dependent) | Farm_Length, Blockage_Ratio |

---

## Atmospheric Features

Derived from vertical atmospheric profiles during preprocessing. See [Preprocessing](../preprocessing.md) for computation details.

### ABL_height

**Atmospheric Boundary Layer height** — The depth of the turbulent layer in contact with the surface.

| Property | Value |
|----------|-------|
| Units | meters |
| Typical range | 200 – 2000 m |
| Source | Height where wind speed reaches 99% of maximum |

**Physical relevance:**
- Shallow ABL constrains wake expansion
- Deep ABL allows wakes to recover faster
- Affects vertical momentum mixing

**When important:**
- Offshore environments with varying marine boundary layers
- Complex terrain with thermal effects
- Seasonal stability variations

```yaml
features:
  - ABL_height
```

### wind_veer

**Wind direction change with height** — Rate of directional shear (dθ/dz).

| Property | Value |
|----------|-------|
| Units | degrees/meter |
| Typical range | -0.01 to +0.05 deg/m |
| Source | Linear fit to wind direction profile |

**Physical relevance:**
- Causes wake deflection with height
- Skews wake shape
- Important for tall turbines and deep arrays

**When important:**
- Locations with strong Ekman spirals
- Stable atmospheric conditions
- Turbines with large rotor diameters

```yaml
features:
  - wind_veer
```

### lapse_rate

**Potential temperature gradient** — Stability indicator from capping inversion fitting.

| Property | Value |
|----------|-------|
| Units | K/m |
| Typical range | -0.01 to +0.01 K/m |
| Interpretation | Negative = unstable, Zero = neutral, Positive = stable |

**Physical relevance:**
- Controls vertical mixing intensity
- Stable conditions suppress wake recovery
- Unstable conditions enhance turbulent mixing

**When important:**
- Sites with strong diurnal stability cycles
- Offshore with sea-breeze effects
- Any site where stability varies significantly

```yaml
features:
  - lapse_rate
```

### turbulence_intensity

**Turbulence Intensity (TI)** — Ratio of velocity fluctuations to mean wind speed.

| Property | Value |
|----------|-------|
| Units | dimensionless (often expressed as %) |
| Typical range | 0.02 – 0.20 (2% – 20%) |
| Source | √(2k/3) / U, where k is turbulent kinetic energy |

**Physical relevance:**
- High TI accelerates wake recovery
- Low TI allows wakes to persist longer
- Directly affects wake-induced losses

**When important:**
- Complex terrain with orographic turbulence
- Sites with variable surface roughness
- Comparing onshore vs offshore performance

```yaml
features:
  - turbulence_intensity
```

### capping_inversion_strength

**Strength of the capping inversion** — Temperature jump at ABL top.

| Property | Value |
|----------|-------|
| Units | K (Kelvin) |
| Typical range | 0 – 10 K |
| Source | Fitted from potential temperature profile |

**Physical relevance:**
- Strong inversions trap turbulence below
- Affects entrainment at ABL top
- Can limit vertical wake expansion

```yaml
features:
  - capping_inversion_strength
```

### capping_inversion_thickness

**Thickness of the capping inversion layer**.

| Property | Value |
|----------|-------|
| Units | meters |
| Typical range | 50 – 500 m |
| Source | Fitted from potential temperature profile |

```yaml
features:
  - capping_inversion_thickness
```

---

## Flow Features

Direct measurements or simple derivations from wind data.

### wind_speed

**Hub-height wind speed**.

| Property | Value |
|----------|-------|
| Units | m/s |
| Typical range | 3 – 25 m/s |
| Source | Reference resource data at hub height |

**Physical relevance:**
- Wake deficit is wind-speed dependent
- Thrust coefficient varies with wind speed
- Power curve non-linearity

```yaml
features:
  - wind_speed
```

### wind_direction

**Hub-height wind direction**.

| Property | Value |
|----------|-------|
| Units | degrees (meteorological convention) |
| Typical range | 0 – 360° |
| Source | Reference resource data |

**Physical relevance:**
- Determines which turbines are waked
- Affects layout features (Farm_Length, etc.)
- May correlate with stability patterns

**Note:** Often used as a categorical or cyclic feature due to wraparound at 0°/360°.

```yaml
features:
  - wind_direction
```

---

## Layout Features

Computed from farm geometry relative to wind direction. These features change as wind direction changes, capturing how the farm "sees" the flow.

### Farm_Length

**Extent of the farm in the wind direction**.

| Property | Value |
|----------|-------|
| Units | rotor diameters |
| Typical range | 5 – 100 D |
| Source | Computed from turbine positions and wind direction |

**Physical relevance:**
- Longer farms accumulate more wake losses
- Deep arrays have more wake interactions
- Affects global blockage effects

```yaml
features:
  - Farm_Length
```

### Farm_Width

**Extent of the farm perpendicular to wind direction**.

| Property | Value |
|----------|-------|
| Units | rotor diameters |
| Typical range | 5 – 50 D |

**Physical relevance:**
- Wide farms affect more of the flow field
- Lateral wake merging effects

```yaml
features:
  - Farm_Width
```

### Blockage_Ratio

**Fraction of incoming flow blocked by upstream turbines**.

| Property | Value |
|----------|-------|
| Units | dimensionless |
| Typical range | 0 – 0.9 |
| Source | Geometric calculation from turbine positions |

**Physical relevance:**
- High blockage = significant global blockage effects
- Affects upstream flow deceleration
- Important for dense arrays

```yaml
features:
  - Blockage_Ratio
```

### Blocking_Distance

**Normalized distance to blocking turbines**.

| Property | Value |
|----------|-------|
| Units | dimensionless (normalized) |
| Typical range | 0 – 1 |

**Physical relevance:**
- Closer blockers have stronger effects
- Used with Blockage_Ratio for blockage modeling

```yaml
features:
  - Blocking_Distance
```

---

## Feature Selection Guide

### Recommended Starting Set

For most applications, start with these core features:

```yaml
error_prediction:
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
```

### Extended Set for Complex Sites

Add layout and flow features for complex scenarios:

```yaml
error_prediction:
  features:
    # Atmospheric
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity
    - capping_inversion_strength
    # Flow
    - wind_speed
    - wind_direction
    # Layout
    - Farm_Length
    - Farm_Width
    - Blockage_Ratio
```

### Selection Criteria

| Criterion | Guidance |
|-----------|----------|
| **Physical relevance** | Feature should relate to wake physics the model may miss |
| **Variance in dataset** | Features with no variation can't help prediction |
| **Measurement quality** | Noisy or uncertain features may hurt more than help |
| **Correlation with target** | Check scatter plots of feature vs. bias |
| **Multicollinearity** | Highly correlated features add redundancy |

### Checking Feature Variance

```python
import xarray as xr

database = xr.load_dataset("results_stacked_hh.nc")

# Check variance of each feature
for feature in ["ABL_height", "wind_veer", "lapse_rate", "turbulence_intensity"]:
    values = database[feature].isel(sample=0).values
    print(f"{feature}:")
    print(f"  Mean: {values.mean():.4f}")
    print(f"  Std:  {values.std():.4f}")
    print(f"  Range: [{values.min():.4f}, {values.max():.4f}]")
    print()
```

### Checking Feature-Bias Correlation

```python
import matplotlib.pyplot as plt
import numpy as np

# Get bias at calibrated sample (e.g., sample 0 for defaults)
bias = database["model_bias_cap"].isel(sample=0).values

features = ["ABL_height", "wind_veer", "lapse_rate", "turbulence_intensity"]

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
for ax, feature in zip(axes.flat, features):
    x = database[feature].isel(sample=0).values
    ax.scatter(x, bias, alpha=0.5)
    ax.set_xlabel(feature)
    ax.set_ylabel("Bias")

    # Add correlation coefficient
    corr = np.corrcoef(x, bias)[0, 1]
    ax.set_title(f"r = {corr:.3f}")

plt.tight_layout()
plt.savefig("feature_bias_correlations.png")
```

---

## Adding Custom Features

You can extend WIFA-UQ with custom features computed from your data.

### Method 1: Pre-compute in Preprocessing

Add custom features during the preprocessing step:

```python
from wifa_uq.preprocessing import PreprocessingInputs
import xarray as xr

# Standard preprocessing
preprocessing = PreprocessingInputs(
    ref_resource_path="reference_resource.nc",
    output_path="processed_resource.nc",
    steps=["recalculate_params"]
)
preprocessing.run()

# Load and add custom feature
processed = xr.load_dataset("processed_resource.nc")

# Example: Richardson number (stability parameter)
g = 9.81  # gravity
theta_ref = 300  # reference potential temperature

# Assuming we have the necessary variables
Ri = (g / theta_ref) * processed["lapse_rate"] / (processed["wind_shear"]**2 + 1e-10)
processed["richardson_number"] = Ri

# Save with custom feature
processed.to_netcdf("processed_resource_custom.nc")
```

Then use in configuration:

```yaml
paths:
  processed_resource: processed_resource_custom.nc

error_prediction:
  features:
    - ABL_height
    - wind_veer
    - richardson_number    # Custom feature
```

### Method 2: Add After Database Generation

Add features to the stacked database:

```python
import xarray as xr
import numpy as np

# Load database
database = xr.load_dataset("results_stacked_hh.nc")

# Compute custom feature
# Example: Stability class (categorical encoded as numeric)
lapse_rate = database["lapse_rate"].isel(sample=0)
stability_class = xr.where(lapse_rate < -0.003, 0,    # unstable
                  xr.where(lapse_rate > 0.003, 2,     # stable
                           1))                         # neutral

# Add to dataset (broadcast across samples)
database["stability_class"] = stability_class.broadcast_like(database["ABL_height"])

# Save
database.to_netcdf("results_stacked_hh_custom.nc")
```

### Method 3: Feature Transformations

Apply transformations to existing features:

```python
import numpy as np

# Log transform (for right-skewed features)
database["log_ABL_height"] = np.log(database["ABL_height"])

# Polynomial features
database["ABL_height_squared"] = database["ABL_height"] ** 2

# Interaction terms
database["ABL_TI_interaction"] = database["ABL_height"] * database["turbulence_intensity"]

# Cyclic encoding for wind direction
database["wind_dir_sin"] = np.sin(np.radians(database["wind_direction"]))
database["wind_dir_cos"] = np.cos(np.radians(database["wind_direction"]))
```

### Custom Feature Best Practices

| Practice | Rationale |
|----------|-----------|
| **Document units and derivation** | Reproducibility and interpretation |
| **Check for NaN/Inf** | Custom calculations may produce invalid values |
| **Normalize appropriately** | Match scale of other features |
| **Verify physical meaning** | Features should relate to wake physics |
| **Test impact on predictions** | Compare CV metrics with/without feature |

---

## Feature Preprocessing

### Handling Missing Values

Features may have missing values for some flow cases:

```python
# Check for missing values
for feature in features:
    n_missing = np.isnan(database[feature].isel(sample=0).values).sum()
    if n_missing > 0:
        print(f"{feature}: {n_missing} missing values")
```

**Strategies:**

1. **Drop cases with missing features** (default behavior)
2. **Impute with mean/median**
3. **Use models that handle NaN** (e.g., XGBoost)

```python
# Imputation example
from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="median")
X_imputed = imputer.fit_transform(X)
```

### Feature Scaling

Some models benefit from scaled features:

| Model | Scaling Needed? |
|-------|-----------------|
| XGBoost | No (tree-based) |
| Linear/Ridge/Lasso | Yes |
| PCE | Yes |
| SIRPolynomial | Yes |

```yaml
error_prediction:
  regressor: Linear
  scale_features: true    # Enable automatic scaling
```

Or manually:

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

### Handling Cyclic Features

Wind direction wraps around at 0°/360°. Use cyclic encoding:

```python
# Convert wind direction to cyclic features
wind_dir_rad = np.radians(database["wind_direction"])
database["wind_dir_sin"] = np.sin(wind_dir_rad)
database["wind_dir_cos"] = np.cos(wind_dir_rad)

# Use these instead of raw wind_direction
features:
  - wind_dir_sin
  - wind_dir_cos
```

---

## Feature Importance Analysis

After training, examine which features matter most.

### XGBoost SHAP Analysis

```yaml
error_prediction:
  regressor: XGB
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity

sensitivity_analysis:
  output_shap: true
```

Output files:
- `shap_summary.png`: Beeswarm plot
- `shap_feature_importance.png`: Bar chart

### PCE Sobol Indices

```yaml
error_prediction:
  regressor: PCE
  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - turbulence_intensity

sensitivity_analysis:
  output_sobol: true
```

Output files:
- `sobol_indices.csv`: First-order and total indices
- `sobol_barplot.png`: Visualization

### Linear Coefficients

```python
predictor = BiasPredictor(database, regressor_name="Linear", feature_names=features)
predictor.fit(calibrated_indices)

# Get standardized coefficients for fair comparison
coefs = predictor.get_coefficients()
feature_stds = X.std(axis=0)
standardized_coefs = coefs * feature_stds

print("Standardized coefficients (larger = more important):")
for feat, coef in sorted(zip(features, np.abs(standardized_coefs)), key=lambda x: -x[1]):
    print(f"  {feat}: {coef:.4f}")
```

---

## Troubleshooting

### "Feature not found in dataset"

**Cause:** Feature name doesn't match dataset variable name.

**Solution:**
```python
# List available variables
print(database.data_vars)

# Check exact spelling
"ABL_height" in database.data_vars  # True or False?
```

### "Feature has no variance"

**Cause:** Feature is constant across all cases.

**Solution:**
- Remove the feature (it can't help prediction)
- Check if preprocessing failed for that variable
- Verify data source has variation

### "High feature correlation causes unstable coefficients"

**Cause:** Multicollinearity between features.

**Solution:**
- Use regularized models (Ridge, Lasso)
- Remove redundant features
- Use PCE or XGB which handle correlation better

```python
# Check correlation matrix
import pandas as pd

X_df = pd.DataFrame(X, columns=features)
corr_matrix = X_df.corr()
print(corr_matrix)

# Flag high correlations
high_corr = np.where(np.abs(corr_matrix) > 0.8)
for i, j in zip(*high_corr):
    if i < j:
        print(f"High correlation: {features[i]} - {features[j]}: {corr_matrix.iloc[i, j]:.3f}")
```

### "Custom feature produces NaN"

**Cause:** Division by zero, log of negative, or missing inputs.

**Solution:**
```python
# Add safety checks
epsilon = 1e-10
safe_ratio = numerator / (denominator + epsilon)

# Or handle explicitly
custom_feature = xr.where(condition, computed_value, np.nan)
```

---

## Feature Reference Table

| Feature | Category | Units | Typical Range | Requires |
|---------|----------|-------|---------------|----------|
| `ABL_height` | Atmospheric | m | 200-2000 | wind_speed profile |
| `wind_veer` | Atmospheric | deg/m | -0.01 to 0.05 | wind_direction profile |
| `lapse_rate` | Atmospheric | K/m | -0.01 to 0.01 | potential_temperature profile |
| `turbulence_intensity` | Atmospheric | - | 0.02-0.20 | TKE (k) + wind_speed |
| `capping_inversion_strength` | Atmospheric | K | 0-10 | potential_temperature profile |
| `capping_inversion_thickness` | Atmospheric | m | 50-500 | potential_temperature profile |
| `wind_speed` | Flow | m/s | 3-25 | Direct measurement |
| `wind_direction` | Flow | deg | 0-360 | Direct measurement |
| `Farm_Length` | Layout | D | 5-100 | Turbine positions + wind_dir |
| `Farm_Width` | Layout | D | 5-50 | Turbine positions + wind_dir |
| `Blockage_Ratio` | Layout | - | 0-0.9 | Turbine positions + wind_dir |
| `Blocking_Distance` | Layout | - | 0-1 | Turbine positions + wind_dir |

---

## See Also

- [Preprocessing](../preprocessing.md) — How atmospheric features are derived
- [ML Models](ml_models.md) — How features are used in prediction
- [Sensitivity Analysis](../sensitivity_analysis.md) — Feature importance methods
- [Configuration Reference](../configuration.md) — Full YAML options
