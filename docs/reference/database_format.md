# Database Format Reference

Documentation of the NetCDF output structure produced by WIFA-UQ database generation.

## Overview

WIFA-UQ produces NetCDF4 files containing model bias data across parameter samples and flow cases. The primary output is a "stacked" dataset that combines multiple wind farms and/or flow cases into a single file.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Database Structure                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Dimensions:                                                                │
│    sample (100)      ← Parameter samples from Latin Hypercube               │
│    case_index (500)  ← Stacked flow cases (farm × time)                     │
│                                                                             │
│  Coordinates:                                                               │
│    sample      [0, 1, 2, ..., 99]                                           │
│    case_index  [0, 1, 2, ..., 499]                                          │
│    k_b         (sample) [0.012, 0.034, ...]   ← Swept parameter values      │
│    ss_alpha    (sample) [0.78, 0.92, ...]                                   │
│    wind_farm   (case_index) ['alpha', 'alpha', ..., 'beta', ...]            │
│                                                                             │
│  Data Variables:                                                            │
│    model_bias_cap  (sample, case_index)  ← Primary output: normalized bias  │
│    ABL_height      (sample, case_index)  ← Features (constant across sample)│
│    wind_veer       (sample, case_index)                                     │
│    lapse_rate      (sample, case_index)                                     │
│    ...                                                                      │
│                                                                             │
│  Attributes:                                                                │
│    swept_params: ['k_b', 'ss_alpha']                                        │
│    param_defaults: {'k_b': 0.04, 'ss_alpha': 0.875}                         │
│    rated_power: 15000.0                                                     │
│    creation_date: '2024-01-15T10:30:00'                                     │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## File Types

| File | Description | Typical Size |
|------|-------------|--------------|
| `results_stacked_hh.nc` | Primary stacked database (hub-height) | 10-100 MB |
| `results_stacked_turbine.nc` | Per-turbine results (optional) | 100 MB - 1 GB |
| `processed_resource.nc` | Preprocessed atmospheric data | 1-50 MB |

---

## Dimensions

### sample

Parameter sample index from Latin Hypercube Sampling.

| Property | Value |
|----------|-------|
| Size | `nsamples` from config (default: 100) |
| Type | int64 |
| Range | 0 to nsamples-1 |

**Note:** Sample 0 always contains the default parameter values.

### case_index

Flattened index across all flow cases. For multi-farm setups, this stacks all farms.

| Property | Value |
|----------|-------|
| Size | Total flow cases across all farms |
| Type | int64 |
| Range | 0 to total_cases-1 |

**Stacking order:**
```
Single farm:  case_index = flow_case_index
Multi-farm:   case_index = farm_offset + flow_case_index
              where farm_offset = sum(cases in previous farms)
```

---

## Coordinates

### Parameter Coordinates

Each swept parameter becomes a coordinate indexed by `sample`:

```python
# Access parameter values
database.k_b.values          # Array of k_b values for each sample
database.ss_alpha.values     # Array of ss_alpha values for each sample

# Get parameters for specific sample
sample_42_params = {
    'k_b': database.k_b.sel(sample=42).values,
    'ss_alpha': database.ss_alpha.sel(sample=42).values
}
```

| Coordinate | Dimension | Type | Description |
|------------|-----------|------|-------------|
| `k_b` | (sample,) | float64 | Wake expansion coefficient |
| `ceps` | (sample,) | float64 | Bastankhah epsilon parameter |
| `ss_alpha` | (sample,) | float64 | Self-similarity alpha |
| *others* | (sample,) | float64 | Additional swept parameters |

### Identifier Coordinates

```python
# Wind farm identifier (multi-farm only)
database.wind_farm.values    # ['alpha', 'alpha', ..., 'beta', 'beta', ...]

# Original indices (optional)
database.original_case_idx.values  # Original case index within each farm
```

| Coordinate | Dimension | Type | Description |
|------------|-----------|------|-------------|
| `wind_farm` | (case_index,) | string | Farm identifier |
| `original_case_idx` | (case_index,) | int64 | Case index within original farm |

---

## Data Variables

### Primary Output

#### model_bias_cap

Normalized model bias (primary output for calibration and prediction).

```python
# Definition
model_bias_cap = (P_model - P_reference) / P_rated
```

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | dimensionless (fraction of rated power) |
| Typical range | -0.2 to +0.2 |

**Interpretation:**
- Positive: Model overpredicts power
- Negative: Model underpredicts power
- Value of 0.05 = 5% of rated power bias

```python
# Access bias data
bias = database["model_bias_cap"]

# Bias at sample 0 (default params) for all cases
default_bias = bias.sel(sample=0)

# Bias for specific case across all samples
case_100_bias = bias.isel(case_index=100)

# Mean bias across cases for each sample
mean_bias_per_sample = bias.mean(dim="case_index")
```

### Feature Variables

Features are stored with dimensions `(sample, case_index)` but are typically constant across samples (physical conditions don't change with model parameters).

#### ABL_height

Atmospheric Boundary Layer height.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | meters |
| Typical range | 200 - 2000 m |

```python
# Features are constant across samples, so use sample=0
abl_heights = database["ABL_height"].isel(sample=0)
```

#### wind_veer

Wind direction change with height.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | degrees/meter |
| Typical range | -0.01 to +0.05 deg/m |

#### lapse_rate

Potential temperature gradient (stability indicator).

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | K/m |
| Typical range | -0.01 to +0.01 K/m |

#### turbulence_intensity

Turbulence intensity at hub height.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | dimensionless |
| Typical range | 0.02 - 0.20 |

#### wind_speed

Hub-height wind speed.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | m/s |
| Typical range | 3 - 25 m/s |

#### wind_direction

Hub-height wind direction.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | degrees (meteorological) |
| Range | 0 - 360° |

### Layout Features

Wind-direction-dependent farm geometry features.

#### Farm_Length

Farm extent in wind direction.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | rotor diameters |
| Typical range | 5 - 100 D |

#### Farm_Width

Farm extent perpendicular to wind.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | rotor diameters |
| Typical range | 5 - 50 D |

#### Blockage_Ratio

Fraction of flow blocked by upstream turbines.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | dimensionless |
| Range | 0 - 0.9 |

#### Blocking_Distance

Normalized distance to blocking turbines.

| Property | Value |
|----------|-------|
| Dimensions | (sample, case_index) |
| Type | float64 |
| Units | dimensionless |
| Range | 0 - 1 |

### Optional Variables

These may be present depending on configuration:

| Variable | Dimensions | Description |
|----------|------------|-------------|
| `P_model` | (sample, case_index) | Raw model power (kW) |
| `P_reference` | (case_index,) | Reference power (kW) |
| `capping_inversion_strength` | (sample, case_index) | Inversion strength (K) |
| `capping_inversion_thickness` | (sample, case_index) | Inversion thickness (m) |

---

## Attributes

Global attributes stored in the NetCDF file.

### Essential Attributes

```python
database.attrs["swept_params"]      # List of swept parameter names
database.attrs["param_defaults"]    # Dict of default parameter values
database.attrs["rated_power"]       # Rated power in kW
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `swept_params` | list | Names of swept parameters |
| `param_defaults` | dict | Default values for each parameter |
| `rated_power` | float | Turbine/farm rated power (kW) |

### Metadata Attributes

```python
database.attrs["creation_date"]     # ISO timestamp
database.attrs["wifa_uq_version"]   # Software version
database.attrs["config_hash"]       # Hash of configuration used
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `creation_date` | string | ISO 8601 creation timestamp |
| `wifa_uq_version` | string | WIFA-UQ version |
| `config_hash` | string | MD5 hash of configuration |
| `pywake_version` | string | PyWake version used |

### Farm Attributes (Multi-farm)

```python
database.attrs["farm_names"]        # List of farm names
database.attrs["farm_case_counts"]  # Cases per farm
```

| Attribute | Type | Description |
|-----------|------|-------------|
| `farm_names` | list | Names of included farms |
| `farm_case_counts` | dict | Number of cases per farm |

---

## Accessing Data

### Loading the Database

```python
import xarray as xr

# Load full database
database = xr.load_dataset("results_stacked_hh.nc")

# Lazy loading (for large files)
database = xr.open_dataset("results_stacked_hh.nc")

# View structure
print(database)
```

### Common Access Patterns

```python
# Get bias at default parameters
default_bias = database["model_bias_cap"].sel(sample=0)

# Get bias for specific parameter values
# (find closest sample)
target_k_b = 0.045
closest_sample = abs(database.k_b - target_k_b).argmin().values
bias_at_target = database["model_bias_cap"].isel(sample=closest_sample)

# Get all features for ML
features = ["ABL_height", "wind_veer", "lapse_rate", "turbulence_intensity"]
X = database[features].isel(sample=0).to_dataframe().reset_index()

# Filter by farm (multi-farm)
alpha_data = database.where(database.wind_farm == "alpha", drop=True)

# Get parameter values for best sample
best_sample = 42
best_params = {
    param: database[param].sel(sample=best_sample).values
    for param in database.attrs["swept_params"]
}
```

### Converting to DataFrame

```python
import pandas as pd

# Convert features to DataFrame
df = database.isel(sample=0).to_dataframe().reset_index()
print(df.columns)
# Index(['case_index', 'ABL_height', 'wind_veer', 'lapse_rate', ...])

# Pivot for sample × case matrix
bias_matrix = database["model_bias_cap"].to_dataframe().unstack("case_index")
```

---

## File Inspection

### Command Line

```bash
# View structure with ncdump
ncdump -h results_stacked_hh.nc

# View with xarray (Python)
python -c "import xarray as xr; print(xr.open_dataset('results_stacked_hh.nc'))"
```

### Python Inspection

```python
import xarray as xr

db = xr.open_dataset("results_stacked_hh.nc")

# Dimensions
print("Dimensions:", dict(db.dims))
# Dimensions: {'sample': 100, 'case_index': 500}

# Coordinates
print("Coordinates:", list(db.coords))
# Coordinates: ['sample', 'case_index', 'k_b', 'ss_alpha', 'wind_farm']

# Variables
print("Variables:", list(db.data_vars))
# Variables: ['model_bias_cap', 'ABL_height', 'wind_veer', ...]

# Attributes
print("Attributes:", dict(db.attrs))
# Attributes: {'swept_params': ['k_b', 'ss_alpha'], 'param_defaults': {...}, ...}

# Variable details
print(db["model_bias_cap"])
# <xarray.DataArray 'model_bias_cap' (sample: 100, case_index: 500)>
# ...
```

---

## Preprocessed Resource Format

The preprocessed resource file (`processed_resource.nc`) contains derived atmospheric quantities.

### Structure

```
Dimensions:
  time (or flow_case): N_cases
  height: N_heights (for profile data)

Variables:
  ABL_height      (time,)           - Boundary layer height
  wind_veer       (time,)           - Wind veer rate
  lapse_rate      (time,)           - Temperature lapse rate
  turbulence_intensity (time,)      - Hub-height TI
  wind_speed      (time, height)    - Wind speed profiles
  wind_direction  (time, height)    - Direction profiles
  potential_temperature (time, height) - Temperature profiles
```

### Access Example

```python
resource = xr.open_dataset("processed_resource.nc")

# Get derived quantities
abl_heights = resource["ABL_height"].values
wind_veer = resource["wind_veer"].values

# Get profiles for specific case
case_0_wind = resource["wind_speed"].isel(time=0)
heights = resource["height"].values
```

---

## Memory Considerations

### Estimating File Size

```
Size ≈ n_samples × n_cases × n_variables × 8 bytes

Example:
  100 samples × 500 cases × 10 variables × 8 bytes = 4 MB (uncompressed)

With compression: typically 2-4x smaller
```

### Working with Large Files

```python
# Use lazy loading
database = xr.open_dataset("large_file.nc", chunks={"case_index": 100})

# Process in chunks
for chunk in database["model_bias_cap"].groupby_bins("case_index", bins=10):
    process(chunk)

# Select before loading
subset = database.sel(sample=slice(0, 10)).load()
```

---

## Validation

### Check Database Integrity

```python
def validate_database(path):
    """Validate WIFA-UQ database structure."""
    db = xr.open_dataset(path)

    errors = []

    # Check required dimensions
    for dim in ["sample", "case_index"]:
        if dim not in db.dims:
            errors.append(f"Missing dimension: {dim}")

    # Check required variables
    if "model_bias_cap" not in db.data_vars:
        errors.append("Missing variable: model_bias_cap")

    # Check required attributes
    for attr in ["swept_params", "param_defaults", "rated_power"]:
        if attr not in db.attrs:
            errors.append(f"Missing attribute: {attr}")

    # Check parameter coordinates exist
    for param in db.attrs.get("swept_params", []):
        if param not in db.coords:
            errors.append(f"Missing parameter coordinate: {param}")

    # Check for NaN values
    nan_count = db["model_bias_cap"].isnull().sum().values
    if nan_count > 0:
        errors.append(f"Found {nan_count} NaN values in model_bias_cap")

    # Check bias range
    bias_min = db["model_bias_cap"].min().values
    bias_max = db["model_bias_cap"].max().values
    if bias_min < -1 or bias_max > 1:
        errors.append(f"Bias range [{bias_min:.2f}, {bias_max:.2f}] seems unrealistic")

    if errors:
        print("Validation errors:")
        for e in errors:
            print(f"  - {e}")
        return False

    print("Database validation passed!")
    return True

# Usage
validate_database("results_stacked_hh.nc")
```

---

## See Also

- [Configuration Schema](config_schema.md) — YAML options that control output
- [Database Generation](../user_guide/database_generation.md) — How databases are created
