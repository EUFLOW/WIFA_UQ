# Rated Power Inference and Data Format Documentation

## Overview

The `DatabaseGenerator` class needs to know the rated power of turbines in the wind farm
to normalize bias calculations. This document explains how rated power is determined and
how to structure your input files.

## Rated Power Inference

The `_infer_rated_power()` method attempts to find the rated power using **three strategies**,
in order of preference:

### Strategy 1: Explicit `rated_power` Key (Recommended)

The cleanest approach is to specify `rated_power` in your turbine's `performance` section:

```yaml
# wind_farm.yaml
turbines:
  name: "IEA 15MW Offshore Reference"
  hub_height: 150.0
  rotor_diameter: 240.0
  performance:
    rated_power: 15000000  # 15 MW in Watts
    Ct_curve:
      Ct_values: [0.0, 0.8, 0.8, 0.4, 0.0]
      Ct_wind_speeds: [0.0, 4.0, 10.0, 15.0, 25.0]
```

This follows the [windIO turbine schema](https://github.com/EUFLOW/windIO) exactly.

### Strategy 2: Power Curve Maximum

If `rated_power` is not specified but a power curve is provided, the maximum value
from `power_curve.power_values` is used:

```yaml
turbines:
  name: "Generic 10MW Turbine"
  hub_height: 119.0
  rotor_diameter: 178.0
  performance:
    power_curve:
      power_values: [0.0, 1.0e6, 5.0e6, 10.0e6, 10.0e6, 0.0]
      power_wind_speeds: [0.0, 4.0, 8.0, 12.0, 20.0, 25.0]
    Ct_curve:
      Ct_values: [0.0, 0.8, 0.75, 0.4, 0.0]
      Ct_wind_speeds: [0.0, 4.0, 10.0, 15.0, 25.0]
```

In this case, rated power would be inferred as `10.0e6` (10 MW).

### Strategy 3: Turbine Name Parsing (Fallback)

As a last resort, the code parses the turbine `name` field looking for patterns
like "15MW", "22 MW", "10.5mw", etc.:

```yaml
turbines:
  name: "IEA 22MW Offshore Reference"  # Will parse "22" and multiply by 1e6
  hub_height: 170.0
  rotor_diameter: 282.0
  # Note: No performance data needed if name contains MW rating
```

**Warning:** This is fragile and should only be used for quick prototyping.

## Where Turbine Data is Searched

The code looks for turbine data in two locations:

1. **Wind Farm File** (`wind_farm.yaml`): `turbines` at the top level
2. **System File** (`system.yaml`): `wind_farm.turbines` nested inside

```yaml
# system.yaml structure
name: MySystem
wind_farm:
  turbines:  # <-- Also checked here
    name: "..."
    performance:
      rated_power: ...
```

## Optional: Override with `rated_power` Parameter

For cases where inferring rated power is problematic, you can add `rated_power`
as an optional parameter to `DatabaseGenerator`:

```python
# Future enhancement
gen = DatabaseGenerator(
    nsamples=100,
    param_config={...},
    system_yaml_path=...,
    rated_power=15e6,  # Direct override, skip inference
    ...
)
```

**Note:** This feature is not yet implemented but is the recommended approach
for complex setups.

## Legacy `meta.yaml` Files

Some older examples use a `meta.yaml` file containing metadata like:

```yaml
# meta.yaml (legacy format)
rated_power: 15000000
system: wind_energy_system/system.yaml
ref_power: observed_output/power.nc
ref_resource: plant_energy_resource/resource.nc
```

This approach is **deprecated** in favor of windIO-compliant files. The current
workflow expects all necessary information to be derivable from the windIO files
themselves.

## Troubleshooting

If you see this error:

```
ValueError: Could not find or infer 'rated_power'.
Tried:
  1. 'rated_power' key in '...turbines.performance'.
  2. 'max(power_curve.power_values)' in '...turbines.performance'.
  3. Parsing 'XMW' from '...turbines.name' field.
```

**Solutions:**

1. Add `performance.rated_power` to your turbine definition (recommended)
2. Add a power curve with realistic values
3. Include "XMW" in your turbine name (e.g., "My 10MW Turbine")
4. Check that your `wind_farm.yaml` file is properly formatted

## Example: Complete windIO-compliant Setup

```yaml
# wind_farm.yaml
name: OffshoreWindFarm
layouts:
  - coordinates:
      x: [0, 1000, 2000, 3000]
      y: [0, 0, 0, 0]
turbines:
  name: "IEA 15MW Offshore Reference Turbine"
  hub_height: 150.0
  rotor_diameter: 240.0
  performance:
    rated_power: 15000000  # <-- Best practice: always include this
    rated_wind_speed: 10.59
    cutin_wind_speed: 3.0
    cutout_wind_speed: 25.0
    Ct_curve:
      Ct_values: [0.0, 0.792, 0.792, 0.405, 0.0]
      Ct_wind_speeds: [0.0, 3.0, 10.59, 15.0, 25.0]
```

This format ensures:
- Clear, unambiguous rated power specification
- Full windIO schema compliance
- Compatibility with all WIFA-UQ features
