# Database Generation

Database generation creates a structured dataset exploring how model bias varies with uncertain parameters and atmospheric conditions. This is the core data that enables calibration and bias prediction.

## Overview

The database generator:
1. Samples uncertain wake model parameters (Latin hypercube sampling)
2. Runs the wake model (PyWake) for each parameter sample
3. Computes bias relative to reference data
4. Adds physical and layout-dependent features
5. Produces a stacked NetCDF dataset ready for ML

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         Database Generation                                 │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐   ┌─────────────────┐   ┌──────────────────────────────┐  │
│  │   Parameter  │   │   Wake Model    │   │      Bias Calculation        │  │
│  │   Samples    │──►│   (PyWake)      │──►│  bias = (model - ref) / P_r  │  │
│  │   k_b, α,... │   │   100× runs     │   │                              │  │
│  └──────────────┘   └─────────────────┘   └──────────────────────────────┘  │
│                                                      │                      │
│                                                      ▼                      │
│  ┌──────────────────────────────────────────────────────────────────────┐   │
│  │                    Model Error Database                              │   │
│  │  Dimensions: [sample × case_index]                                   │   │
│  │  Variables: model_bias_cap, pw_power_cap, ref_power_cap              │   │
│  │  Features: ABL_height, wind_veer, Blockage_Ratio, Farm_Length, ...   │   │
│  │  Coords: k_b, ss_alpha (swept parameters)                            │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Configuration

```yaml
database_gen:
  run: true
  flow_model: pywake           # Wake model to use
  n_samples: 100               # Number of parameter samples
  param_config:                # Parameters to sweep
    attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b:
      range: [0.01, 0.07]
      default: 0.04
      short_name: k_b
    attributes.analysis.blockage_model.ss_alpha:
      range: [0.75, 1.0]
      default: 0.875
      short_name: ss_alpha
```

## Parameter Configuration

### Parameter Path Syntax

Parameters are specified using dot-separated paths that match the windIO system YAML structure:

```yaml
# In your system.yaml:
attributes:
  analysis:
    wind_deficit_model:
      name: Bastankhah2014
      wake_expansion_coefficient:
        k_a: 0.04
        k_b: 0.0    # ← This is swept

# Path in config:
attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b
```

### Configuration Formats

**Short format** (range only):
```yaml
param_config:
  attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b: [0.01, 0.07]
```

The short name is inferred from the last component of the path (`k_b`).

**Full format** (recommended):
```yaml
param_config:
  attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b:
    range: [0.01, 0.07]    # Sampling bounds [min, max]
    default: 0.04          # First sample uses this value
    short_name: k_b        # Coordinate name in output database
```

### Available Parameters

#### Bastankhah Gaussian Wake Model

| Parameter | Path | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| k_b | `...wake_expansion_coefficient.k_b` | [0.01, 0.07] | 0.04 | Wake expansion (TI-dependent term) |
| k_a | `...wake_expansion_coefficient.k_a` | [0.01, 0.1] | 0.04 | Wake expansion (ambient term) |
| ceps | `...ceps` | [0.15, 0.3] | 0.2154 | Epsilon coefficient for wake deficit |

#### Self-Similarity Blockage Model

| Parameter | Path | Range | Default | Description |
|-----------|------|-------|---------|-------------|
| ss_alpha | `...blockage_model.ss_alpha` | [0.75, 1.0] | 0.875 | Induction zone decay parameter |

#### Custom Parameters

Any parameter accessible via windIO path notation can be swept:

```yaml
param_config:
  attributes.analysis.your_model.your_parameter:
    range: [min_value, max_value]
    default: nominal_value
    short_name: display_name
```

## Sampling Strategy

### Latin Hypercube Sampling

Parameters are sampled using Latin hypercube sampling (LHS) to ensure good coverage of the parameter space:

```python
# For n_samples = 100 and 2 parameters:
# Each parameter range is divided into 100 strata
# Each stratum is sampled exactly once
# Result: uniform coverage with no clustering
```

### First Sample = Default

The first sample (index 0) always uses the default parameter values. This provides a baseline for comparison:

```python
# Sample 0: k_b = 0.04 (default), ss_alpha = 0.875 (default)
# Samples 1-99: Random LHS samples within ranges
```

### Reproducibility

A fixed random seed ensures reproducible sampling:

```python
# In run_parameter_sweep:
seed = 1  # Fixed seed for reproducibility
```

## Bias Calculation

### Definition

Model bias is computed as the normalized difference between model output and reference:

```
bias = (P_model - P_reference) / P_rated
```

Where:
- `P_model` = PyWake power output (farm average)
- `P_reference` = Reference power (LES, SCADA, etc.)
- `P_rated` = Turbine rated power

### Why Normalize by Rated Power?

Normalizing by rated power (rather than reference power or model power):
- Provides a consistent scale across operating conditions
- Avoids division by near-zero values at low wind speeds
- Enables direct comparison across turbines and farms
- Bias of 0.05 = 5% of rated power error

### Farm-Level Aggregation

Bias is computed at the farm level (average across turbines):

```python
# Per-turbine bias
turbine_bias = pw_power[turbine, time] - ref_power[turbine, time]

# Farm-average bias (normalized)
farm_bias = mean(turbine_bias, axis=turbines) / rated_power
```

## Layout Features

The database generator adds layout-dependent features that vary with wind direction.

### Blockage Ratio

**Definition:** Fraction of rotor area blocked by upstream turbines.

**Physical meaning:** Higher blockage = more wake interference expected.

**Algorithm:**
1. Discretize each rotor disk into a grid of points
2. For each point, check if any upstream turbine's wake intersects it
3. Blockage ratio = fraction of blocked points

**Range:** 0 (front-row turbine) to ~0.9 (deeply embedded)

### Blocking Distance

**Definition:** Normalized distance to the nearest blocking turbine.

**Physical meaning:** Closer blockers = stronger wake effects.

**Algorithm:**
1. For each blocked point, record distance to blocking turbine
2. Unblocked points get L∞ = 20D (maximum distance)
3. Average across all rotor points, normalize by L∞

**Range:** 0 (very close blocker) to 1 (unblocked or far)

### Farm Length

**Definition:** Farm extent in the wind direction, normalized by rotor diameter.

**Physical meaning:** Longer farms = more potential for wake accumulation.

**Algorithm:**
1. Project all turbine positions onto wind direction vector
2. Farm length = max projection - min projection
3. Normalize by rotor diameter D

**Units:** Rotor diameters (D)

### Farm Width

**Definition:** Farm extent perpendicular to wind direction.

**Physical meaning:** Wider farms = more lateral wake interactions.

**Algorithm:**
1. Project positions onto cross-wind vector
2. Farm width = max projection - min projection
3. Normalize by rotor diameter D

**Units:** Rotor diameters (D)

## Output Format

### NetCDF Structure

The output database is a stacked xarray Dataset:

```
Dimensions:
  sample: 100              # Parameter samples
  case_index: N            # Stacked (wind_farm × flow_case)

Data Variables:
  model_bias_cap (sample, case_index)    # Normalized bias
  pw_power_cap (sample, case_index)      # PyWake power / rated
  ref_power_cap (sample, case_index)     # Reference power / rated
  ABL_height (case_index)                # From preprocessing
  wind_veer (case_index)                 # From preprocessing
  lapse_rate (case_index)                # From preprocessing
  Blockage_Ratio (case_index)            # Layout feature
  Blocking_Distance (case_index)         # Layout feature
  Farm_Length (case_index)               # Layout feature
  Farm_Width (case_index)                # Layout feature
  turb_rated_power (wind_farm)           # Turbine rated power

Coordinates:
  sample: [0, 1, 2, ..., 99]
  case_index: [0, 1, 2, ..., N-1]
  k_b (sample): [0.04, 0.023, 0.067, ...]     # Swept parameter values
  ss_alpha (sample): [0.875, 0.82, 0.95, ...] # Swept parameter values
  wind_farm (case_index): ["FarmName", ...]   # Farm identifier

Attributes:
  swept_params: ["k_b", "ss_alpha"]
  param_paths: ["attributes.analysis...k_b", "...ss_alpha"]
  param_defaults: '{"k_b": 0.04, "ss_alpha": 0.875}'
```

### Loading the Database

```python
import xarray as xr

db = xr.load_dataset("results_stacked_hh.nc")

# Access bias for sample 0 (default parameters)
default_bias = db["model_bias_cap"].isel(sample=0)

# Get parameter values for each sample
k_b_values = db.coords["k_b"].values

# Get all features for ML
features = db[["ABL_height", "wind_veer", "Blockage_Ratio"]].isel(sample=0)
```

## API Usage

### Single-Farm Generation

```python
from pathlib import Path
from wifa_uq.model_error_database.database_gen import DatabaseGenerator

generator = DatabaseGenerator(
    nsamples=100,
    param_config={
        "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": {
            "range": [0.01, 0.07],
            "default": 0.04,
            "short_name": "k_b"
        }
    },
    system_yaml_path=Path("wind_energy_system.yaml"),
    ref_power_path=Path("reference_power.nc"),
    processed_resource_path=Path("processed_physical_inputs.nc"),
    wf_layout_path=Path("wind_farm.yaml"),
    output_db_path=Path("results/database.nc"),
    model="pywake"
)

database = generator.generate_database()
```

### Multi-Farm Generation

```python
from wifa_uq.model_error_database.multi_farm_gen import generate_multi_farm_database

farm_configs = [
    {"name": "Farm1", "system_config": Path("farm1/system.yaml")},
    {"name": "Farm2", "system_config": Path("farm2/system.yaml")},
]

database = generate_multi_farm_database(
    farm_configs=farm_configs,
    param_config={...},
    n_samples=100,
    output_dir=Path("multi_farm_results/"),
    run_preprocessing=True,
    preprocessing_steps=["recalculate_params"],
)
```

## Rated Power Inference

The generator needs turbine rated power for bias normalization. It searches in order:

1. **Explicit `rated_power` key** (recommended):
   ```yaml
   turbines:
     performance:
       rated_power: 15000000  # Watts
   ```

2. **Maximum of power curve**:
   ```yaml
   turbines:
     performance:
       power_curve:
         power_values: [0, 1e6, 5e6, 15e6, 15e6, 0]
   ```

3. **Parse from turbine name** (last resort):
   ```yaml
   turbines:
     name: "IEA 15MW Offshore Reference"  # Extracts "15" × 1e6
   ```

If all methods fail, an error is raised with guidance on adding rated power.

## Performance Considerations

### Execution Time

Database generation involves running PyWake `n_samples` times:

| n_samples | Turbines | Flow Cases | Approximate Time |
|-----------|----------|------------|------------------|
| 50 | 10 | 100 | ~2-5 minutes |
| 100 | 10 | 100 | ~5-10 minutes |
| 100 | 100 | 500 | ~30-60 minutes |

### Memory Usage

The database size scales as:
```
size ≈ n_samples × n_cases × n_variables × 8 bytes
```

For 100 samples, 1000 cases, 10 variables: ~8 MB

### Recommendations

1. **Start small** with n_samples=20 for initial testing
2. **Increase samples** to 100-200 for production runs
3. **Use preprocessing caching** to avoid re-running preprocessing
4. **Consider parallelization** for large multi-farm studies (future feature)

## Skipping Database Generation

If you have an existing database, skip generation:

```yaml
database_gen:
  run: false

paths:
  database_file: existing_database.nc  # Will be loaded from output_dir
```

## Troubleshooting

### "Could not find or infer 'rated_power'"

Add rated_power to your turbine definition. See [Rated Power Inference](#rated-power-inference) and [Metadata Note](../appendix/metadata_note.md).

### "Mismatch in 'time' dimension"

The reference power and resource files must have the same number of time steps:

```python
import xarray as xr
power = xr.load_dataset("reference_power.nc")
resource = xr.load_dataset("resource.nc")
print(f"Power: {len(power.time)}, Resource: {len(resource.time)}")
```

### "Feature not found in dataset"

Ensure preprocessing was run with the correct steps:

```yaml
preprocessing:
  run: true
  steps: [recalculate_params]  # Creates ABL_height, wind_veer, etc.
```

### PyWake simulation errors

Check your windIO system YAML for:
- Valid wake model configuration
- Correct turbine Ct curve
- Reasonable wind speed ranges

## See Also

- [Configuration Reference](configuration.md) — Full YAML options
- [Preprocessing](preprocessing.md) — Preparing input data
- [Swept Parameters Reference](../reference/swept_parameters.md) — All available parameters
- [Database Format Reference](../reference/database_format.md) — NetCDF schema details
- [windIO Integration](../concepts/windio_integration.md) — Data format standards
