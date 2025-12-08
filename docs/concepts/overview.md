# Architecture Overview

WIFA-UQ is designed as a modular pipeline for reducing and quantifying uncertainty in wind farm wake models. This page provides a high-level view of how the components fit together.

## The Problem

Engineering wake models (PyWake, FOXES, WAYVE) trade physical fidelity for computational speed. They use simplified representations of turbulence, wake expansion, and atmospheric effects. This introduces systematic errors—**model bias**—that varies with operating conditions.

WIFA-UQ addresses this by:
1. Characterizing how bias depends on uncertain model parameters
2. Finding optimal parameter settings (calibration)
3. Learning to predict residual bias from physical features
4. Quantifying the uncertainty in predictions

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           WIFA-UQ Pipeline                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │  Reference Data │  LES simulations, SCADA measurements, or field data    │
│  │  (Truth)        │  Provides ground truth for bias calculation            │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐                                                        │
│  │  Preprocessing  │  Calculate derived quantities from raw profiles:       │
│  │                 │  • ABL height    • Wind veer    • Lapse rate           │
│  │                 │  • Turbulence intensity                                │
│  └────────┬────────┘                                                        │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐                                 │
│  │  Wake Model     │    │  Parameter      │                                 │
│  │  (PyWake/FOXES) │◄───│  Samples        │  Latin hypercube sampling of    │
│  │                 │    │  (k_b, α, ...)  │  uncertain parameters           │
│  └────────┬────────┘    └─────────────────┘                                 │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                                │
│  │         Model Error Database            │                                │
│  │  ┌─────────────────────────────────┐    │                                │
│  │  │ bias[sample, case] =            │    │  For each parameter sample     │
│  │  │   (model - reference) / rated   │    │  and flow case, store the      │
│  │  └─────────────────────────────────┘    │  normalized bias               │
│  │  + Physical features (ABL_height, ...)  │                                │
│  │  + Layout features (blockage, ...)      │                                │
│  └────────┬────────────────────────────────┘                                │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                                │
│  │           Calibration                   │                                │
│  │  ┌─────────────┐  ┌─────────────────┐   │                                │
│  │  │   Global    │  │     Local       │   │                                │
│  │  │ Single best │  │ Optimal params  │   │                                │
│  │  │ param set   │  │ = f(features)   │   │                                │
│  │  └─────────────┘  └─────────────────┘   │                                │
│  └────────┬────────────────────────────────┘                                │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                                │
│  │         Bias Prediction                 │                                │
│  │  ┌─────────────────────────────────┐    │                                │
│  │  │ residual_bias = ML(features)    │    │  XGBoost, PCE, SIR, or         │
│  │  │                                 │    │  linear models                 │
│  │  └─────────────────────────────────┘    │                                │
│  └────────┬────────────────────────────────┘                                │
│           │                                                                 │
│           ▼                                                                 │
│  ┌─────────────────────────────────────────┐                                │
│  │         Corrected Output                │                                │
│  │  power_corrected = model(calibrated)    │                                │
│  │                    - predicted_bias     │                                │
│  └─────────────────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Component Responsibilities

### Preprocessing

**Module:** `wifa_uq.preprocessing.PreprocessingInputs`

Transforms raw atmospheric profiles into features suitable for bias prediction:

| Input | Output | Method |
|-------|--------|--------|
| Velocity profile | ABL height | Height where U reaches 99% of max |
| Wind direction profile | Wind veer | dθ/dz (directional shear) |
| Temperature profile | Lapse rate | dΘ/dz from capping inversion fitting |
| TKE profile | Turbulence intensity | √(2k/3) / U |

### Database Generation

**Module:** `wifa_uq.model_error_database.DatabaseGenerator`

Creates a structured dataset exploring the parameter-bias relationship:

1. **Sample parameters** — Generate N samples of uncertain parameters (k_b, ss_alpha, etc.)
2. **Run simulations** — Execute wake model for each sample across all flow cases
3. **Compute bias** — Calculate normalized difference from reference
4. **Add features** — Attach physical and layout features to each case

The output is an xarray Dataset with dimensions `[sample, case_index]`.

### Calibration

**Module:** `wifa_uq.postprocessing.calibration`

Finds optimal parameter settings:

| Strategy | Class | Description |
|----------|-------|-------------|
| **Global** | `MinBiasCalibrator` | Single parameter set minimizing total absolute bias |
| **Global** | `DefaultParams` | Use literature/default values |
| **Local** | `LocalParameterPredictor` | ML model predicting optimal params per case |
| **Bayesian** | `BayesianCalibrationWrapper` | Posterior distribution via MCMC |

### Bias Prediction

**Module:** `wifa_uq.postprocessing.error_predictor`

Learns the residual bias after calibration:

```
bias = f(ABL_height, wind_veer, lapse_rate, blockage_ratio, ...)
```

Available models:
- **XGBoost** — Gradient boosting with SHAP interpretability
- **PCE** — Polynomial Chaos Expansion with Sobol indices
- **SIR+Polynomial** — Dimension reduction + polynomial regression
- **Linear** — Ridge/Lasso/ElasticNet regularized regression

### Cross-Validation

Evaluates generalization using:
- **K-Fold** — Random splits for single-farm studies
- **Leave-One-Group-Out** — Hold out entire farms for multi-farm studies

## Data Flow

```
                    ┌──────────────────┐
                    │  wind_farm.yaml  │  Turbine positions, specs
                    └────────┬─────────┘
                             │
                             ▼
┌──────────────┐    ┌──────────────────┐    ┌──────────────────┐
│ resource.nc  │───►│   Preprocessing  │───►│ processed.nc     │
│ (raw profiles)│    └──────────────────┘    │ (+ derived vars) │
└──────────────┘                             └────────┬─────────┘
                                                      │
                    ┌──────────────────┐              │
                    │  system.yaml     │              │
                    │  (wake model     │              │
                    │   settings)      │              │
                    └────────┬─────────┘              │
                             │                        │
                             ▼                        ▼
┌──────────────┐    ┌──────────────────────────────────────────┐
│ ref_power.nc │───►│           DatabaseGenerator              │
│ (truth data) │    │  • Parameter sweep                       │
└──────────────┘    │  • Bias calculation                      │
                    │  • Feature attachment                    │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │        results_stacked_hh.nc             │
                    │  Dims: [sample, case_index]              │
                    │  Vars: model_bias_cap, pw_power_cap,     │
                    │        ref_power_cap, ABL_height, ...    │
                    │  Coords: k_b, ss_alpha, wind_farm        │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │     Calibration + Bias Prediction        │
                    │     (Cross-validated)                    │
                    └────────────────────┬─────────────────────┘
                                         │
                                         ▼
                    ┌──────────────────────────────────────────┐
                    │              Outputs                     │
                    │  • cv_results.csv (metrics)              │
                    │  • predictions.npz (raw predictions)     │
                    │  • *.png (diagnostic plots)              │
                    └──────────────────────────────────────────┘
```

## Extensibility Points

WIFA-UQ is designed for extension at several levels:

### Adding New Wake Models

Currently supports PyWake via the WIFA framework. To add a new model:
1. Implement a runner in `model_error_database/`
2. Update `DatabaseGenerator` to dispatch based on `flow_model` config

### Adding New Calibrators

1. Create a class with `fit()` method setting `best_idx_` and `best_params_`
2. Register in `workflow.py` CLASS_MAP and CALIBRATION_MODES

### Adding New ML Models

1. Create sklearn-compatible estimator with `fit()` and `predict()`
2. Add to `build_predictor_pipeline()` in `workflow.py`
3. Implement sensitivity analysis hooks if needed

### Adding New Features

**Physical features** (from atmospheric data):
- Calculate in `preprocessing.py`

**Layout features** (from turbine positions):
- Calculate in `database_gen.py` or `utils.py`

## Configuration-Driven Workflow

All components are configured via YAML:

```yaml
preprocessing:
  run: true
  steps: [recalculate_params]

database_gen:
  run: true
  flow_model: pywake
  n_samples: 100
  param_config: {...}

error_prediction:
  run: true
  model: "XGB"
  calibrator: LocalParameterPredictor
  features: [ABL_height, wind_veer, ...]
```

The `workflow.py` orchestrator interprets this configuration and executes the appropriate pipeline.

## Related Pages

- [Model Bias](model_bias.md) — Why bias correction matters
- [Calibration Theory](calibration_theory.md) — Global vs local approaches
- [Uncertainty Quantification](uncertainty_quantification.md) — PCE, Bayesian methods
- [windIO Integration](windio_integration.md) — Data format standards
