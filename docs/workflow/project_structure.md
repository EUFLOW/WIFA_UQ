# Project Structure

This page explains the organization of the WIFA-UQ repository, helping you navigate the codebase and understand where to find (or add) specific functionality.

## Repository Overview

```
WIFA-UQ/
├── wifa_uq/                    # Main Python package
│   ├── preprocessing/          # Data preprocessing utilities
│   ├── model_error_database/   # Database generation and parameter sweeps
│   ├── postprocessing/         # Calibration, error prediction, UQ
│   └── workflow.py             # Main workflow orchestrator
│
├── examples/                   # Example workflows and data
│   ├── run.py                  # CLI entry point
│   ├── *.yaml                  # Configuration files
│   └── data/                   # Example datasets
│
├── tests/                      # Test suite
│   ├── unit/                   # Unit tests
│   └── conftest.py             # Pytest fixtures
│
├── docs/                       # Documentation (you are here)
│
├── pyproject.toml              # Project metadata and dependencies
├── README.md                   # Repository README
└── LICENSE                     # MIT License
```

## Core Package: `wifa_uq/`

The main package is organized into three processing stages plus a workflow orchestrator.

### `wifa_uq/preprocessing/`

Handles preparation of atmospheric input data before database generation.

```
preprocessing/
└── preprocessing.py    # PreprocessingInputs class
```

**Key Class: `PreprocessingInputs`**

- Loads raw NetCDF resource files
- Recalculates derived quantities:
  - ABL height (from velocity profile or temperature)
  - Wind veer (directional shear)
  - Lapse rate (temperature gradient)
  - Turbulence intensity (from TKE)
- Outputs processed NetCDF ready for database generation

### `wifa_uq/model_error_database/`

Generates databases of model bias by running parameter sweeps.

```
model_error_database/
├── __init__.py
├── database_gen.py        # DatabaseGenerator class
├── run_pywake_sweep.py    # PyWake parameter sweep
├── multi_farm_gen.py      # Multi-farm database generation
└── utils.py               # Layout feature calculations
```

**Key Classes:**

| Class | Purpose |
|-------|---------|
| `DatabaseGenerator` | Orchestrates single-farm database generation |
| `MultiFarmDatabaseGenerator` | Combines databases from multiple farms |

**Key Functions:**

| Function | Purpose |
|----------|---------|
| `run_parameter_sweep()` | Run PyWake across parameter samples |
| `create_parameter_samples()` | Generate Latin hypercube samples |
| `blockage_metrics()` | Calculate per-turbine blockage ratio/distance |
| `farm_length_width()` | Calculate farm dimensions in wind coordinates |

### `wifa_uq/postprocessing/`

Contains calibration, error prediction, and uncertainty quantification tools.

```
postprocessing/
├── __init__.py
├── postprocesser.py           # Base class (ABC)
│
├── calibration/
│   ├── __init__.py
│   └── basic_calibration.py   # Calibrator classes
│
├── error_predictor/
│   └── error_predictor.py     # ML models and CV pipeline
│
├── bayesian_calibration/
│   ├── __init__.py
│   └── bayesian_calibration.py # UMBRA-based Bayesian inference
│
└── PCE_tool/
    ├── pce_utils.py           # PCE construction and Sobol analysis
    ├── main.py                # Standalone PCE analysis script
    ├── config.yaml            # PCE-specific config
    └── README.md              # PCE tool documentation
```

**Calibration Classes:**

| Class | Mode | Description |
|-------|------|-------------|
| `MinBiasCalibrator` | Global | Find parameter set with minimum total bias |
| `DefaultParams` | Global | Use default parameter values |
| `LocalParameterPredictor` | Local | Predict optimal params per flow case |
| `BayesianCalibrationWrapper` | Global | Bayesian inference with UMBRA |

**ML Regressors:**

| Class | Description |
|-------|-------------|
| `PCERegressor` | Polynomial Chaos Expansion (OpenTURNS) |
| `SIRPolynomialRegressor` | Sliced Inverse Regression + polynomial |
| `LinearRegressor` | OLS/Ridge/Lasso/ElasticNet wrapper |
| (XGBoost) | Used via sklearn Pipeline with StandardScaler |

**Pipeline Classes:**

| Class | Purpose |
|-------|---------|
| `BiasPredictor` | Wrapper for ML pipeline fit/predict |
| `MainPipeline` | Combines calibrator + bias predictor |

### `wifa_uq/workflow.py`

The main orchestrator that ties everything together.

**Key Function: `run_workflow(config_path)`**

1. Loads and parses YAML configuration
2. Detects single-farm vs multi-farm mode
3. Runs preprocessing (if enabled)
4. Generates/loads error database
5. Runs cross-validated error prediction
6. Generates sensitivity analysis plots
7. Saves results

## Examples Directory: `examples/`

Contains runnable examples with data and configurations.

### Entry Point: `run.py`

Simple script that loads a YAML config and calls `run_workflow()`:

```python
from wifa_uq.workflow import run_workflow

config_path = Path(sys.argv[1])
run_workflow(config_path)
```

**Usage:**
```bash
python examples/run.py examples/kul_les_example.yaml
```

### Configuration Files

| File | Description |
|------|-------------|
| `kul_les_example.yaml` | KUL LES data with PCE + local calibration |
| `kul_single_farm_xgb_example.yaml` | KUL LES with XGBoost + all features |
| `edf_single_farm_example.yaml` | EDF Horns Rev dataset |
| `multi_farm_example.yaml` | Multi-farm workflow with LOGO CV |
| `err_prediction_example.yaml` | Legacy config format |

### Data Directory: `examples/data/`

Contains windIO-compliant datasets for testing and demonstration.

```
data/
├── KUL_LES/                          # KU Leuven LES simulations
│   ├── wind_energy_system/
│   │   └── system_pywake.yaml        # windIO system definition
│   ├── plant_wind_farm/
│   │   └── FLOW_UQ_vnv_toy_study_wind_farm.yaml
│   ├── plant_energy_resource/
│   │   └── originalData.nc           # Atmospheric profiles
│   └── observed_output/
│       └── observedPowerKUL.nc       # Reference power data
│
├── EDF_datasets/                     # EDF LES datasets
│   ├── HR1/, HR2/, HR3/              # Horns Rev configurations
│   ├── NYSTED1/, NYSTED2/            # Nysted configurations
│   └── VirtWF_*/                     # Virtual wind farms
│
└── ...
```

**Dataset Structure (windIO format):**

Each dataset typically contains:
- `wind_energy_system.yaml` — System definition with analysis settings
- `wind_farm.yaml` — Turbine layout and specifications
- `energy_resource.yaml` or `*.nc` — Atmospheric conditions
- `turbine_data.nc` — Reference power output

## Tests Directory: `tests/`

Pytest-based test suite with fixtures and unit tests.

```
tests/
├── conftest.py                    # Shared fixtures
└── unit/
    ├── test_calibration_basic.py  # Calibrator tests
    ├── test_database_gen.py       # DatabaseGenerator tests
    ├── test_error_predictor_*.py  # Error prediction tests
    ├── test_multi_farm_gen.py     # Multi-farm tests
    ├── test_preprocessing.py      # Preprocessing tests
    └── test_run_pywake_sweep.py   # Parameter sweep tests
```

### Running Tests

```bash
# With pixi
pixi run test

# With pytest directly
pytest tests/unit -v

# With coverage
pixi run test-cov
```

### Key Fixtures (in `conftest.py`)

| Fixture | Description |
|---------|-------------|
| `tiny_bias_db` | Synthetic xarray Dataset mimicking database output |
| `windio_turbine_dict` | windIO-compliant turbine definition |
| `windio_system_dict` | windIO-compliant system definition |
| `pywake_param_config` | Standard parameter config for sweeps |
| `multi_farm_configs` | Config list for multi-farm tests |

## Configuration Files

### `pyproject.toml`

Defines project metadata, dependencies, and tooling:

```toml
[project]
name = "wifa_uq"
version = "0.1"
dependencies = [
    "py_wake>=2.6.5",
    "foxes>=1.2.3",
    "wayve @ git+https://...",
    "wifa @ git+https://...",
    ...
]

[tool.pixi.tasks]
test = "pytest tests/unit -v"
lint = "pycodestyle wifa_uq --max-line-length=120"
```

### YAML Workflow Configs

All workflow configurations follow this structure:

```yaml
description: "..."

paths:
  system_config: ...
  reference_power: ...
  reference_resource: ...
  output_dir: ...

preprocessing:
  run: true/false
  steps: [...]

database_gen:
  run: true/false
  flow_model: pywake
  n_samples: 100
  param_config: {...}

error_prediction:
  run: true/false
  features: [...]
  model: "XGB" / "PCE" / "SIRPolynomial"
  calibrator: MinBiasCalibrator / LocalParameterPredictor
  cross_validation: {...}

sensitivity_analysis:
  run_observation_sensitivity: true/false
  run_bias_sensitivity: true/false
```

## Module Dependency Graph

```
workflow.py
    │
    ├── preprocessing/preprocessing.py
    │       └── (xarray, numpy, scipy, wayve)
    │
    ├── model_error_database/
    │       ├── database_gen.py
    │       │       └── run_pywake_sweep.py
    │       │               └── (wifa.pywake_api, windIO)
    │       ├── multi_farm_gen.py
    │       │       └── database_gen.py
    │       └── utils.py
    │               └── (numpy, scipy)
    │
    └── postprocessing/
            ├── calibration/basic_calibration.py
            │       └── (sklearn, xgboost)
            │
            ├── error_predictor/error_predictor.py
            │       ├── (sklearn, xgboost, shap)
            │       └── PCE_tool/pce_utils.py
            │               └── (openturns)
            │
            └── bayesian_calibration/
                    └── (umbra)
```

## Adding New Functionality

### Adding a New Calibrator

1. Create a class in `wifa_uq/postprocessing/calibration/`
2. Implement `fit()` method that sets `best_idx_` and `best_params_`
3. Register in `wifa_uq/workflow.py` CLASS_MAP and CALIBRATION_MODES
4. Add tests in `tests/unit/test_calibration_basic.py`

### Adding a New ML Regressor

1. Create sklearn-compatible class in `error_predictor.py`
2. Add to `build_predictor_pipeline()` in `workflow.py`
3. Handle sensitivity analysis (SHAP for trees, custom for others)

### Adding a New Physical Feature

1. Calculate in `preprocessing.py` (if derived from profiles)
2. Or calculate in `database_gen.py` (if layout-dependent)
3. Add to test fixtures in `conftest.py`
4. Document in feature list
