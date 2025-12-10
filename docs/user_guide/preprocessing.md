# Preprocessing

Preprocessing transforms raw atmospheric data into derived quantities that are useful for bias prediction. This page explains the preprocessing pipeline, available steps, and when to use them.

## Overview

The preprocessing stage operates on raw NetCDF resource files (atmospheric profiles) and produces an enriched dataset with derived quantities like ABL height, wind veer, and lapse rate.

```
Raw Resource (resource.nc)              Processed Resource
┌─────────────────────────┐             ┌─────────────────────────┐
│ wind_speed(time, height)│             │ wind_speed(time, height)│
│ wind_direction(...)     │   ──────►   │ wind_direction(...)     │
│ potential_temperature   │             │ potential_temperature   │
│ k (TKE)                 │             │ k (TKE)                 │
└─────────────────────────┘             │ + ABL_height(time)      │
                                        │ + wind_veer(time, height│
                                        │ + lapse_rate(time)      │
                                        │ + turbulence_intensity  │
                                        └─────────────────────────┘
```

## Configuration

Enable preprocessing in your YAML config:

```yaml
preprocessing:
  run: true
  steps:
    - recalculate_params    # Calculate derived quantities
```

If `run: false` or no steps are specified, the raw resource file is used directly.

## Available Steps

### `recalculate_params`

The primary preprocessing step. Calculates multiple derived quantities from vertical profiles.

**What it computes:**

| Variable | Computation Method | Required Inputs |
|----------|-------------------|-----------------|
| `ABL_height` | Height where wind speed reaches 99% of max | `wind_speed`, `height` |
| `wind_veer` | dθ/dz (wind direction gradient) | `wind_direction`, `height` |
| `turbulence_intensity` | √(2k/3) / U | `k`, `wind_speed` |
| `lapse_rate` | dΘ/dz from capping inversion fitting | `potential_temperature`, `height` |
| `capping_inversion_strength` | Temperature jump at ABL top | `potential_temperature`, `height` |
| `capping_inversion_thickness` | Depth of capping inversion | `potential_temperature`, `height` |

**Example:**

```yaml
preprocessing:
  run: true
  steps:
    - recalculate_params
```

## Derived Quantities Explained

### ABL Height

**Definition:** The height at which wind speed first reaches 99% of the maximum velocity in the profile.

**Physical meaning:** Captures the top of the atmospheric boundary layer, including cases with low-level jets where the maximum wind speed occurs below the free atmosphere.

**Algorithm:**
```
1. Find max_wind_speed = max(U(z)) for each time step
2. Compute threshold = 0.99 × max_wind_speed
3. Find lowest height where U ≥ threshold
```

**Fallback:** If temperature data is available but velocity-based calculation fails, the temperature-based ABL height from capping inversion fitting is used.

**Typical range:** 200–2000 m depending on stability and time of day.

### Wind Veer

**Definition:** The rate of change of wind direction with height (dθ/dz).

**Physical meaning:** Describes Ekman spiral effects and can indicate the presence of baroclinicity or complex atmospheric structures.

**Algorithm:**
```
1. Convert wind direction to radians
2. Unwrap angles to handle 0°/360° crossing
3. Compute gradient: veer = d(direction)/d(height)
4. Convert back to degrees per meter
```

**Units:** degrees per meter (deg/m)

**Typical range:** -0.01 to +0.05 deg/m (positive = veering, clockwise with height in Northern Hemisphere)

### Lapse Rate

**Definition:** The vertical gradient of potential temperature (dΘ/dz).

**Physical meaning:** Indicates atmospheric stability. Positive values indicate stable stratification; near-zero indicates neutral conditions.

**Algorithm:** Uses the `ci_fitting` function from the WAYVE library to fit the capping inversion structure and extract the lapse rate below the ABL top.

**Units:** K/m (Kelvin per meter)

**Typical range:** -0.01 to +0.01 K/m
- Stable: +0.003 to +0.01 K/m
- Neutral: ≈ 0 K/m
- Unstable: negative values

### Turbulence Intensity

**Definition:** Ratio of turbulent velocity fluctuations to mean wind speed.

**Physical meaning:** Describes the intensity of turbulent mixing, which affects wake recovery rates.

**Algorithm:**
```
TI = √(2k/3) / U

where:
  k = turbulent kinetic energy (m²/s²)
  U = mean wind speed (m/s)
```

**Units:** dimensionless (often expressed as percentage)

**Typical range:** 0.02–0.20 (2%–20%)

### Capping Inversion Strength

**Definition:** The temperature jump across the capping inversion at the ABL top.

**Physical meaning:** Stronger inversions indicate more stable conditions that suppress vertical mixing.

**Units:** K (Kelvin)

**Typical range:** 0–10 K

### Capping Inversion Thickness

**Definition:** The vertical extent of the temperature inversion layer.

**Physical meaning:** Thin inversions create sharper ABL tops; thick inversions indicate gradual transitions.

**Units:** m (meters)

**Typical range:** 50–500 m

## Input Requirements

For preprocessing to work correctly, your input NetCDF file should contain:

### Required Variables

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `wind_speed` | (time, height) | m/s | Horizontal wind speed profile |
| `height` | (height,) | m | Vertical coordinate (AGL) |

### Recommended Variables

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `wind_direction` | (time, height) | degrees | Wind direction profile |
| `potential_temperature` | (time, height) | K | Potential temperature profile |
| `k` | (time, height) | m²/s² | Turbulent kinetic energy |

### Optional Variables

| Variable | Dimensions | Units | Description |
|----------|------------|-------|-------------|
| `LMO` | (time,) | m | Monin-Obukhov length (stability) |

If `LMO` is not present, neutral stability (LMO = 10¹⁰ m) is assumed.

## Example NetCDF Structure

A typical input resource file:

```
Dimensions:
  time: 100
  height: 50

Variables:
  height (height): [0, 10, 20, 30, ..., 500]  # meters AGL
  wind_speed (time, height): float32
  wind_direction (time, height): float32
  potential_temperature (time, height): float32
  k (time, height): float32  # TKE
```

## Preprocessing API

For programmatic use:

```python
from pathlib import Path
from wifa_uq.preprocessing.preprocessing import PreprocessingInputs

# Initialize preprocessor
preprocessor = PreprocessingInputs(
    ref_resource_path=Path("resource.nc"),
    output_path=Path("processed_resource.nc"),
    steps=["recalculate_params"]
)

# Run the pipeline
output_path = preprocessor.run_pipeline()
print(f"Processed data saved to: {output_path}")
```

## Handling Missing Data

The preprocessing pipeline handles missing inputs gracefully:

| Missing Variable | Behavior |
|------------------|----------|
| `k` | TI calculation skipped |
| `wind_direction` | Wind veer calculation skipped |
| `potential_temperature` | Temperature-based calculations skipped |
| `height` | Error (required for any derived quantity) |

Warnings are printed for skipped calculations:

```
Skipping TI recalculation: 'k' or 'wind_speed' not found.
Skipping wind veer calculation: Missing 'wind_direction'
```

## When to Use Preprocessing

### Always Use When:

- Your resource file only contains raw vertical profiles
- You need ABL height, wind veer, or lapse rate as ML features
- Reference data comes from LES or mesoscale simulations

### Skip Preprocessing When:

- Your resource file already contains all required derived quantities
- You're using pre-processed data from a previous run
- You only need features available in the raw data

### Configuration Example: Skip Preprocessing

```yaml
preprocessing:
  run: false

# Use raw resource directly
paths:
  reference_resource: already_processed_data.nc
```

Or use a previously processed file:

```yaml
preprocessing:
  run: false

paths:
  # Point to existing processed file
  reference_resource: previous_run/processed_physical_inputs.nc
```

## Troubleshooting

### "ValueError: ci_fitting failed"

**Cause:** The capping inversion fitting algorithm couldn't converge, often due to unusual temperature profiles.

**Solution:**
1. Check that `potential_temperature` increases with height (stable atmosphere)
2. Verify height coordinates are reasonable (0–1000+ m)
3. The preprocessor will still output other variables; only lapse_rate and capping inversion metrics will be missing

### "Wind veer shows unrealistic values"

**Cause:** Wind direction data crossing 0°/360° boundary wasn't properly unwrapped.

**Solution:** This should be handled automatically. If issues persist, check that wind_direction is in degrees (0–360), not radians.

### "ABL_height shows NaN values"

**Cause:** Wind speed profile is monotonic (always increasing or decreasing with height).

**Solution:** For profiles without a clear velocity maximum, ABL_height defaults to the maximum height in the profile. Consider using temperature-based ABL height instead.

### "TI values seem too high/low"

**Cause:** TKE (`k`) might be in unexpected units or represent something different.

**Solution:** Verify TKE units are m²/s². Some models output k in different forms.

## Best Practices

1. **Validate input data** before preprocessing:
   ```python
   import xarray as xr
   ds = xr.load_dataset("resource.nc")
   print(ds)  # Check dimensions and variables
   ```

2. **Inspect output** after preprocessing:
   ```python
   ds_out = xr.load_dataset("processed_physical_inputs.nc")
   print(ds_out["ABL_height"].min(), ds_out["ABL_height"].max())
   ```

3. **Use consistent height grids** across datasets when combining multiple farms

4. **Document preprocessing choices** in your workflow description for reproducibility

## Integration with Database Generation

Preprocessed variables become available as features in the model error database:

```yaml
# These features are available after preprocessing
error_prediction:
  features:
    - ABL_height            # From recalculate_params
    - wind_veer             # From recalculate_params
    - lapse_rate            # From recalculate_params
    - turbulence_intensity  # From recalculate_params
    - Blockage_Ratio        # Added during database_gen
    - Blocking_Distance     # Added during database_gen
```

The database generator interpolates height-dependent variables (like `wind_veer`, `turbulence_intensity`) to hub height automatically.

## See Also

- [Configuration Reference](configuration.md) — Full YAML options
- [Database Generation](database_generation.md) — Next step in the pipeline
- [Feature Engineering](error_prediction/feature_engineering.md) — Using derived features
