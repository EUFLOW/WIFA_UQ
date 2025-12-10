# windIO Integration

WIFA-UQ uses the [windIO](https://github.com/IEAWindSystems/windIO) standard for defining wind energy systems. This page explains the data formats, how WIFA-UQ automatically discovers files through the windIO structure, and how to set up your own datasets.

## What is windIO?

windIO is an open standard for representing wind energy data in YAML and NetCDF formats. It provides schemas for:

- **Wind energy systems** — Complete plant definitions
- **Wind farms** — Turbine layouts and specifications
- **Energy resources** — Atmospheric conditions and profiles
- **Simulation outputs** — Power, loads, and other results

The standard ensures interoperability across tools in the wind energy ecosystem.

## Minimal Configuration

WIFA-UQ only requires a path to the windIO system configuration file. All other paths are automatically inferred from the `!include` directives in the windIO structure:

```yaml
# Minimal WIFA-UQ config
paths:
  system_config: data/KUL_LES/wind_energy_system/system_pywake.yaml
  output_dir: results/

preprocessing:
  run: true

database_gen:
  run: true
  # ...
```

That's it. WIFA-UQ will automatically discover:
- The wind farm layout
- The energy resource (atmospheric profiles)
- The reference power data (simulation outputs)

## How Path Inference Works

WIFA-UQ parses the windIO `!include` chain to find all referenced files:

```
system_pywake.yaml
  ├── site: !include ../site/energy_site.yaml
  │     └── energy_resource: !include ../plant_energy_resource/energy_resource.yaml
  │           └── wind_resource: !include resource.nc  ← reference_resource
  │
  ├── wind_farm: !include ../plant_wind_farm/wind_farm.yaml  ← wind_farm_layout
  │
  └── simulation_outputs: !include ../observed_output/simulation_outputs.yaml
        └── turbine_data: !include observedPowerKUL.nc  ← reference_power
```

The inference follows these paths automatically, so you don't need to specify them manually.

## Optional Explicit Paths

You can still override any path if needed:

```yaml
paths:
  system_config: data/KUL_LES/wind_energy_system/system_pywake.yaml

  # Optional overrides (uncomment to use explicit paths)
  # reference_power: data/KUL_LES/observed_output/observedPowerKUL.nc
  # reference_resource: data/KUL_LES/plant_energy_resource/originalData.nc
  # wind_farm_layout: data/KUL_LES/plant_wind_farm/wind_farm.yaml

  output_dir: results/
```

Explicit paths take precedence over inferred paths.

## The `!include` Directive

windIO uses YAML's `!include` directive to compose files from multiple sources:

```yaml
# In system.yaml
wind_farm: !include plant_wind_farm/wind_farm.yaml
```

This directive loads the referenced file inline. The windIO library handles both YAML and NetCDF includes:

```yaml
# YAML include
site: !include site/energy_site.yaml

# NetCDF include (loaded as xarray Dataset converted to dict)
wind_resource: !include resource.nc
```

Paths in `!include` are relative to the file containing them.

## windIO File Structure

A typical windIO-compliant dataset:

```
my_dataset/
├── wind_energy_system/
│   └── system.yaml              # Top-level (only file you need to reference)
├── site/
│   └── energy_site.yaml         # Site boundaries and resource reference
├── plant_wind_farm/
│   └── wind_farm.yaml           # Turbine layout and specs
├── plant_energy_resource/
│   ├── energy_resource.yaml     # Resource wrapper
│   └── resource.nc              # Atmospheric profiles
└── observed_output/
    ├── simulation_outputs.yaml  # Output wrapper
    └── turbine_data.nc          # Reference power
```

## Wind Energy System YAML

The top-level system file ties everything together:

```yaml
# system.yaml
name: MyWindFarm

# Include sub-components (WIFA-UQ follows these automatically)
site: !include ../site/energy_site.yaml
wind_farm: !include ../plant_wind_farm/wind_farm.yaml
simulation_outputs: !include ../observed_output/simulation_outputs.yaml

# Analysis settings (used by wake models)
attributes:
  analysis:
    wind_deficit_model:
      name: Bastankhah2014
      wake_expansion_coefficient:
        k_a: 0.04
        k_b: 0.0
    blockage_model:
      name: SelfSimilarityDeficit2020
      ss_alpha: 0.875
```

The `simulation_outputs` field links to the reference power data that WIFA-UQ uses for bias calculation.

## Wind Farm Definition

Defines turbine layout and specifications:

```yaml
# wind_farm.yaml
name: OffshoreWindFarm

layouts:
  - coordinates:
      x: [0, 1000, 2000, 3000]
      y: [0, 0, 0, 0]

turbines:
  name: "IEA 15MW Offshore Reference"
  hub_height: 150.0
  rotor_diameter: 240.0

  performance:
    rated_power: 15000000  # Watts
    rated_wind_speed: 10.59
    cutin_wind_speed: 3.0
    cutout_wind_speed: 25.0

    Ct_curve:
      Ct_values: [0.0, 0.792, 0.792, 0.405, 0.0]
      Ct_wind_speeds: [0.0, 3.0, 10.59, 15.0, 25.0]
```

### Required Fields

| Field | Description | Used For |
|-------|-------------|----------|
| `layouts.coordinates.x` | X positions (m) | Layout features, blockage |
| `layouts.coordinates.y` | Y positions (m) | Layout features, blockage |
| `turbines.hub_height` | Hub height (m) | Interpolating profiles |
| `turbines.rotor_diameter` | Rotor diameter (m) | Normalization, blockage |
| `turbines.performance.rated_power` | Rated power (W) | Normalizing bias |

### Rated Power Inference

WIFA-UQ needs rated power to normalize bias. It searches in order:

1. `turbines.performance.rated_power` — Explicit value (preferred)
2. `max(turbines.performance.power_curve.power_values)` — From power curve
3. Parse from `turbines.name` — e.g., "15MW" in name → 15e6 W

**Best practice**: Always include `rated_power` explicitly.

## Energy Resource (Atmospheric Data)

The resource chain typically looks like:

```yaml
# energy_site.yaml
name: MySite
energy_resource: !include ../plant_energy_resource/energy_resource.yaml
boundaries:
  # ...
```

```yaml
# energy_resource.yaml
name: AtmosphericConditions
wind_resource: !include resource.nc
```

The actual atmospheric data is in NetCDF:

```
# resource.nc structure
Dimensions:
  time: 100        # Number of flow cases
  height: 50       # Vertical levels

Variables:
  wind_speed(time, height)           # m/s
  wind_direction(time, height)       # degrees
  potential_temperature(time, height) # K
  k(time, height)                    # TKE (m²/s²), optional

Coordinates:
  height: [0, 10, 20, ..., 500]  # meters AGL
  time: [0, 1, 2, ..., 99]       # flow case indices
```

### Required Variables

| Variable | Units | Used For |
|----------|-------|----------|
| `wind_speed` | m/s | Wake model input, TI calculation |
| `wind_direction` | degrees | Wake model input, layout features |

### Optional Variables

| Variable | Units | Used For |
|----------|-------|----------|
| `potential_temperature` | K | Lapse rate, ABL height |
| `k` (TKE) | m²/s² | Turbulence intensity |
| `LMO` | m | Monin-Obukhov length (stability) |
| `ABL_height` | m | If pre-calculated |

### Derived Variables

WIFA-UQ preprocessing computes these from raw profiles:

| Derived | From | Method |
|---------|------|--------|
| `ABL_height` | velocity profile | Height where U = 0.99 × U_max |
| `wind_veer` | direction profile | dθ/dz |
| `lapse_rate` | temperature profile | dΘ/dz from capping inversion fitting |
| `turbulence_intensity` | k, wind_speed | √(2k/3) / U |

## Simulation Outputs (Reference Power)

The simulation outputs file links to the reference power:

```yaml
# simulation_outputs.yaml
turbine_data: !include observedPowerKUL.nc
```

The NetCDF contains power data:

```
# turbine_data.nc
Dimensions:
  turbine: 9       # Number of turbines
  time: 100        # Must match resource time dimension

Variables:
  power(turbine, time)  # Watts

Coordinates:
  turbine: [0, 1, 2, ..., 8]
  time: [0, 1, 2, ..., 99]
```

**Important**: The `time` dimension must align with the resource file.

## Creating Your Own Dataset

### Step 1: Prepare NetCDF Files

```python
import xarray as xr
import numpy as np

# Create resource file
n_cases = 100
n_heights = 50
heights = np.linspace(0, 500, n_heights)

resource = xr.Dataset({
    'wind_speed': (['time', 'height'], np.random.uniform(5, 15, (n_cases, n_heights))),
    'wind_direction': (['time', 'height'], np.random.uniform(250, 290, (n_cases, n_heights))),
    'potential_temperature': (['time', 'height'], 280 + 0.005 * heights),
}, coords={'time': np.arange(n_cases), 'height': heights})

resource.to_netcdf('resource.nc')

# Create reference power file
n_turbines = 9
ref_power = xr.Dataset({
    'power': (['turbine', 'time'], np.random.uniform(1e6, 5e6, (n_turbines, n_cases)))
}, coords={'turbine': np.arange(n_turbines), 'time': np.arange(n_cases)})

ref_power.to_netcdf('turbine_data.nc')
```

### Step 2: Create windIO YAML Structure

```yaml
# wind_farm.yaml
name: MyFarm
layouts:
  - coordinates:
      x: [0, 500, 1000, 0, 500, 1000, 0, 500, 1000]
      y: [0, 0, 0, 500, 500, 500, 1000, 1000, 1000]
turbines:
  name: "Generic 5MW"
  hub_height: 90.0
  rotor_diameter: 126.0
  performance:
    rated_power: 5000000
    Ct_curve:
      Ct_values: [0.0, 0.8, 0.8, 0.4, 0.0]
      Ct_wind_speeds: [0.0, 4.0, 10.0, 15.0, 25.0]
```

```yaml
# energy_resource.yaml
name: MyResource
wind_resource: !include resource.nc
```

```yaml
# simulation_outputs.yaml
turbine_data: !include turbine_data.nc
```

```yaml
# system.yaml (the only file you reference in WIFA-UQ config)
name: MySystem
site: !include energy_site.yaml
wind_farm: !include wind_farm.yaml
simulation_outputs: !include simulation_outputs.yaml

attributes:
  analysis:
    wind_deficit_model:
      name: Bastankhah2014
      wake_expansion_coefficient:
        k_a: 0.04
        k_b: 0.0
    blockage_model:
      name: SelfSimilarityDeficit2020
      ss_alpha: 0.875
```

### Step 3: Create WIFA-UQ Config

```yaml
# my_workflow.yaml
paths:
  system_config: my_dataset/system.yaml  # Only this is required!
  output_dir: my_dataset/results

preprocessing:
  run: true
  steps: [recalculate_params]

database_gen:
  run: true
  flow_model: pywake
  n_samples: 100
  param_config:
    attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b:
      range: [0.01, 0.07]
      default: 0.04
      short_name: "k_b"
```

## Multi-Farm Configuration

For multi-farm workflows, each farm only needs a system config:

```yaml
paths:
  farms:
    - name: "Farm_A"
      system_config: data/farm_a/system.yaml
    - name: "Farm_B"
      system_config: data/farm_b/system.yaml
  output_dir: results/multi_farm/
```

Paths are inferred independently for each farm.

## Common Issues and Solutions

### "Could not find or infer 'rated_power'"

Add `rated_power` to your turbine's performance section:

```yaml
turbines:
  performance:
    rated_power: 5000000  # Add this line
```

### "time dimension mismatch"

Ensure your resource.nc and turbine_data.nc have the same number of time steps:

```python
resource = xr.load_dataset('resource.nc')
power = xr.load_dataset('turbine_data.nc')
assert len(resource.time) == len(power.time), "Time dimensions must match!"
```

### Path inference fails

If automatic inference doesn't find your files, check:

1. **Include chain is complete** — Each file must have valid `!include` directives
2. **Paths are relative** — `!include` paths are relative to the containing file
3. **Files exist** — All referenced files must be present

You can always fall back to explicit paths:

```yaml
paths:
  system_config: my_dataset/system.yaml
  reference_power: my_dataset/power.nc        # Explicit override
  reference_resource: my_dataset/resource.nc  # Explicit override
```

### "!include file not found"

Check that paths in `!include` are relative to the YAML file containing them:

```yaml
# If this file is in dataset/wind_energy_system/system.yaml
# and wind_farm.yaml is in dataset/plant_wind_farm/
wind_farm: !include ../plant_wind_farm/wind_farm.yaml  # Correct (relative)
wind_farm: !include dataset/plant_wind_farm/wind_farm.yaml  # Wrong (absolute)
```

## Resources

- [windIO GitHub](https://github.com/IEAWindSystems/windIO) — Schema definitions and examples
- [windIO Documentation](https://windio.readthedocs.io/) — Detailed format specifications
- [WIFA-UQ Examples](https://github.com/EUFLOW/WIFA-UQ/tree/main/examples/data) — Working datasets

## Summary

| What You Provide | What WIFA-UQ Infers |
|------------------|---------------------|
| `system_config` path | wind_farm_layout, reference_resource, reference_power |
| windIO-compliant structure | All file locations via `!include` chain |
| Optional explicit paths | Override any inferred path |

The key insight: **structure your data following windIO conventions, point WIFA-UQ at the system file, and everything else is automatic.**
