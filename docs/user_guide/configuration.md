# Configuration Reference

WIFA-UQ workflows are driven by YAML configuration files. This page provides a complete reference for all available options.

## Overview

A configuration file controls the entire workflow:

```yaml
description: "Human-readable description of this workflow"

paths:
  # Input and output file locations

preprocessing:
  # Data preparation options

database_gen:
  # Model error database generation

error_prediction:
  # ML-based bias prediction and cross-validation

sensitivity_analysis:
  # Feature importance analysis
```

## Minimal Configuration

WIFA-UQ uses smart path inference from windIO structures. The minimal configuration requires only:

```yaml
paths:
  system_config: path/to/wind_energy_system.yaml
  output_dir: results/

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
      short_name: k_b

error_prediction:
  run: true
  features: [ABL_height, wind_veer, lapse_rate]
  model: XGB
  calibrator: MinBiasCalibrator
  bias_predictor: BiasPredictor
  cross_validation:
    splitting_mode: kfold_shuffled
    n_splits: 5
```

Other paths (reference_power, reference_resource, wind_farm_layout) are automatically inferred from the windIO `!include` chain.

## Section Reference

### `paths`

Specifies input data and output locations. All paths are relative to the config file's directory.

| Key | Required | Description |
|-----|----------|-------------|
| `system_config` | **Yes** | Path to windIO system YAML (wind_energy_system.yaml) |
| `reference_power` | No* | NetCDF with observed/LES power data |
| `reference_resource` | No* | NetCDF with atmospheric profiles |
| `wind_farm_layout` | No* | YAML with turbine positions and specs |
| `output_dir` | No | Output directory (default: `wifa_uq_results/`) |
| `processed_resource_file` | No | Preprocessed resource filename (default: `processed_physical_inputs.nc`) |
| `database_file` | No | Database filename (default: `results_stacked_hh.nc`) |

*These paths are automatically inferred from the windIO system config if not specified.

#### Path Inference

WIFA-UQ follows the windIO `!include` chain to discover files:

```
system.yaml
  ├── site: !include energy_site.yaml
  │     └── energy_resource: !include resource.nc  → reference_resource
  ├── wind_farm: !include wind_farm.yaml           → wind_farm_layout
  └── simulation_outputs: !include outputs.yaml
        └── turbine_data: !include power.nc        → reference_power
```

#### Explicit Path Override

You can always specify paths explicitly to override inference:

```yaml
paths:
  system_config: system.yaml
  reference_power: custom/path/to/power.nc      # Overrides inferred path
  reference_resource: custom/path/to/resource.nc
  wind_farm_layout: custom/path/to/layout.yaml
  output_dir: my_results/
```

---

### `preprocessing`

Controls data preparation before database generation.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `run` | bool | `false` | Whether to run preprocessing |
| `steps` | list | `[]` | Preprocessing steps to apply |

#### Available Steps

| Step | Description |
|------|-------------|
| `recalculate_params` | Calculate derived quantities from vertical profiles |

```yaml
preprocessing:
  run: true
  steps:
    - recalculate_params
```

See [Preprocessing](preprocessing.md) for detailed documentation on each step.

---

### `database_gen`

Controls the model error database generation via parameter sweeps.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `run` | bool | `false` | Whether to generate the database |
| `flow_model` | string | `"pywake"` | Wake model to use (`"pywake"`) |
| `n_samples` | int | `100` | Number of parameter samples |
| `param_config` | dict | — | Parameters to sweep (see below) |

#### `param_config` Format

Parameters can be specified in two formats:

**Short format** (range only):
```yaml
param_config:
  attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b: [0.01, 0.07]
```

**Full format** (with metadata):
```yaml
param_config:
  attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b:
    range: [0.01, 0.07]    # [min, max] sampling bounds
    default: 0.04          # Default value (first sample uses this)
    short_name: k_b        # Name used in database coordinates
```

#### Common Swept Parameters

| Parameter Path | Short Name | Typical Range | Description |
|----------------|------------|---------------|-------------|
| `attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b` | `k_b` | [0.01, 0.07] | Wake expansion coefficient |
| `attributes.analysis.wind_deficit_model.ceps` | `ceps` | [0.15, 0.3] | Bastankhah epsilon coefficient |
| `attributes.analysis.blockage_model.ss_alpha` | `ss_alpha` | [0.75, 1.0] | Self-similarity blockage alpha |

See [Database Generation](database_generation.md) for complete parameter reference.

---

### `error_prediction`

Configures the ML-based bias prediction pipeline.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `run` | bool | `false` | Whether to run error prediction |
| `features` | list | — | Feature names for ML model (required) |
| `model` | string | `"XGB"` | ML model type |
| `model_params` | dict | `{}` | Model-specific hyperparameters |
| `calibrator` | string | — | Calibrator class name (required) |
| `local_regressor` | string | `"Ridge"` | Regressor for local calibration |
| `local_regressor_params` | dict | `{}` | Local regressor hyperparameters |
| `bias_predictor` | string | `"BiasPredictor"` | Predictor class name |
| `cross_validation` | dict | — | CV configuration (see below) |

#### Available Features

Features from preprocessing:

| Feature | Units | Description |
|---------|-------|-------------|
| `ABL_height` | m | Atmospheric boundary layer height |
| `wind_veer` | deg/m | Wind direction change with height |
| `lapse_rate` | K/m | Potential temperature gradient |
| `turbulence_intensity` | — | TI from TKE profile |
| `wind_speed` | m/s | Wind speed at hub height |
| `wind_direction` | deg | Wind direction at hub height |

Features from database generation:

| Feature | Units | Description |
|---------|-------|-------------|
| `Blockage_Ratio` | — | Fraction of rotor blocked [0-1] |
| `Blocking_Distance` | — | Normalized distance to blockers [0-1] |
| `Farm_Length` | D | Farm extent in wind direction |
| `Farm_Width` | D | Farm extent perpendicular to wind |

#### Available Models

| Model | `model` value | Description |
|-------|---------------|-------------|
| XGBoost | `"XGB"` | Gradient boosting (default, uses SHAP) |
| SIR + Polynomial | `"SIRPolynomial"` | Dimension reduction + polynomial |
| PCE | `"PCE"` | Polynomial Chaos Expansion |
| Linear | `"Linear"` | OLS/Ridge/Lasso/ElasticNet |

**XGBoost parameters:**
```yaml
model: XGB
model_params:
  max_depth: 4
  n_estimators: 200
  learning_rate: 0.1
  random_state: 42
```

**PCE parameters:**
```yaml
model: PCE
model_params:
  degree: 5              # Polynomial degree
  marginals: kernel      # "kernel", "uniform", "normal"
  copula: independent    # "independent" or "normal"
  q: 0.5                 # Hyperbolic truncation parameter
  max_features: 5        # Safety limit on input dimension
  allow_high_dim: false  # Allow > max_features inputs
```

**SIR+Polynomial parameters:**
```yaml
model: SIRPolynomial
model_params:
  n_directions: 1   # Number of SIR directions
  degree: 2         # Polynomial degree
```

**Linear parameters:**
```yaml
model: Linear
model_params:
  method: ridge     # "ols", "ridge", "lasso", "elasticnet"
  alpha: 1.0        # Regularization strength
  l1_ratio: 0.5     # ElasticNet mixing (only for elasticnet)
```

#### Available Calibrators

| Calibrator | Mode | Description |
|------------|------|-------------|
| `MinBiasCalibrator` | Global | Single parameter set minimizing total bias |
| `DefaultParams` | Global | Use default parameter values |
| `LocalParameterPredictor` | Local | ML-predicted params per flow case |
| `BayesianCalibration` | Global | Bayesian inference (requires UMBRA) |

**For local calibration**, specify the regressor:
```yaml
calibrator: LocalParameterPredictor
local_regressor: Ridge          # Linear, Ridge, Lasso, ElasticNet, RandomForest, XGB
local_regressor_params:
  alpha: 1.0
```

#### Cross-Validation Configuration

```yaml
cross_validation:
  run: true
  splitting_mode: kfold_shuffled    # or LeaveOneGroupOut
  n_splits: 5                       # For KFold only
  metrics:
    - rmse
    - r2
    - mae
```

**For multi-farm LeaveOneGroupOut:**
```yaml
cross_validation:
  splitting_mode: LeaveOneGroupOut
  groups:
    Offshore:
      - Farm1
      - Farm2
    Onshore:
      - Farm3
      - Farm4
```

Group names must match the `name` field in your farms list (multi-farm config) or the `wind_farm` coordinate in your database.

---

### `sensitivity_analysis`

Controls feature importance and sensitivity analysis.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `run_observation_sensitivity` | bool | `false` | Run SA on observations |
| `run_bias_sensitivity` | bool | `false` | Run SA on bias predictions |
| `method` | string | `"auto"` | SA method (`"auto"`, `"shap"`, `"sir"`, `"pce_sobol"`) |
| `pce_config` | dict | `{}` | PCE config for Sobol indices |

```yaml
sensitivity_analysis:
  run_observation_sensitivity: true
  run_bias_sensitivity: true
  method: auto              # Uses SHAP for XGB, SIR for SIRPolynomial
  pce_config:               # Only for method: pce_sobol
    degree: 5
    marginals: kernel
    copula: independent
    q: 0.5
```

**Method selection:**
- `auto`: SHAP for tree models, SIR directions for SIR models, Sobol for PCE
- `shap`: Force SHAP TreeExplainer (requires tree model)
- `sir`: Force SIR direction coefficients
- `pce_sobol`: Force PCE-based Sobol indices

---

## Multi-Farm Configuration

For workflows spanning multiple wind farms, use the `farms` key:

```yaml
paths:
  output_dir: results/multi_farm/
  database_file: combined_database.nc

farms:
  - name: Farm1                                    # Required: unique identifier
    system_config: data/farm1/wind_energy_system.yaml  # Required
    # Optional explicit paths (otherwise inferred):
    # reference_power: data/farm1/power.nc
    # reference_resource: data/farm1/resource.nc
    # wind_farm_layout: data/farm1/wind_farm.yaml

  - name: Farm2
    system_config: data/farm2/wind_energy_system.yaml

  - name: Farm3
    system_config: data/farm3/wind_energy_system.yaml

preprocessing:
  run: true
  steps: [recalculate_params]

database_gen:
  run: true
  flow_model: pywake
  n_samples: 100
  param_config:
    # Shared across all farms
    attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b:
      range: [0.01, 0.07]
      default: 0.04
      short_name: k_b

error_prediction:
  run: true
  features: [ABL_height, wind_veer, lapse_rate]
  model: XGB
  calibrator: MinBiasCalibrator
  bias_predictor: BiasPredictor
  cross_validation:
    splitting_mode: LeaveOneGroupOut
    groups:
      Group1:
        - Farm1
        - Farm2
      Group2:
        - Farm3
```

Each farm requires:
- `name`: Unique identifier (used in CV grouping)
- `system_config`: Path to windIO system YAML

Other paths are auto-inferred per farm using the same logic as single-farm mode.

---

## Complete Example

Here's a fully-specified configuration showing all available options:

```yaml
description: "Complete WIFA-UQ configuration example"

paths:
  # Required
  system_config: wind_energy_system/system_pywake.yaml

  # Optional (auto-inferred if omitted)
  reference_power: observed_output/observedPower.nc
  reference_resource: plant_energy_resource/originalData.nc
  wind_farm_layout: plant_wind_farm/wind_farm.yaml

  # Output paths
  output_dir: wifa_uq_results/
  processed_resource_file: processed_physical_inputs.nc
  database_file: results_stacked_hh.nc

preprocessing:
  run: true
  steps:
    - recalculate_params

database_gen:
  run: true
  flow_model: pywake
  n_samples: 100
  param_config:
    attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b:
      range: [0.01, 0.07]
      default: 0.04
      short_name: k_b
    attributes.analysis.wind_deficit_model.ceps:
      range: [0.15, 0.3]
      default: 0.2154
      short_name: ceps
    attributes.analysis.blockage_model.ss_alpha:
      range: [0.75, 1.0]
      default: 0.875
      short_name: ss_alpha

error_prediction:
  run: true

  features:
    - ABL_height
    - wind_veer
    - lapse_rate
    - Blockage_Ratio
    - Blocking_Distance
    - Farm_Length
    - Farm_Width

  model: XGB
  model_params:
    max_depth: 4
    n_estimators: 200
    learning_rate: 0.1
    random_state: 42

  calibrator: LocalParameterPredictor
  local_regressor: Ridge
  local_regressor_params:
    alpha: 1.0

  bias_predictor: BiasPredictor

  cross_validation:
    run: true
    splitting_mode: kfold_shuffled
    n_splits: 5
    metrics:
      - rmse
      - r2
      - mae

sensitivity_analysis:
  run_observation_sensitivity: true
  run_bias_sensitivity: true
  method: auto
  pce_config:
    degree: 5
    marginals: kernel
    copula: independent
    q: 0.5
```

---

## Workflow Execution

Configurations are executed via the `run.py` script:

```bash
cd examples
python run.py my_config.yaml
```

Or programmatically:

```python
from wifa_uq.workflow import run_workflow
from pathlib import Path

cv_results, y_preds, y_tests = run_workflow(Path("my_config.yaml"))
```

---

## Output Files

After a successful run, the output directory contains:

| File | Description |
|------|-------------|
| `processed_physical_inputs.nc` | Preprocessed atmospheric data |
| `results_stacked_hh.nc` | Model error database |
| `cv_results.csv` | Cross-validation metrics per fold |
| `predictions.npz` | Raw predictions array |
| `correction_results.png` | Before/after correction scatter plots |
| `bias_prediction_shap.png` | SHAP beeswarm plot (XGBoost) |
| `bias_prediction_shap_importance.png` | SHAP bar chart (XGBoost) |
| `bias_prediction_sir_importance.png` | SIR importance (SIRPolynomial) |
| `pce_sobol_indices.png` | Sobol indices (PCE) |
| `local_parameter_prediction.png` | Parameter prediction quality (local calibration) |

For multi-farm workflows, additional plots are generated:
- `cv_fold_metrics.png`
- `cv_fold_heatmap.png`
- `cv_predictions_by_fold.png`
- `cv_generalization_summary.png`

---

## See Also

- [Preprocessing](preprocessing.md) — Details on preprocessing steps
- [Database Generation](database_generation.md) — Parameter sweep mechanics
- [Multi-Farm Workflows](multi_farm_workflows.md) — Cross-farm studies
- [windIO Integration](../concepts/windio_integration.md) — Data format standards
