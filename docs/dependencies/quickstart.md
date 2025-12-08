# Quickstart

This guide will get you running your first WIFA-UQ workflow in about 5 minutes. By the end, you'll have:

- Run a complete calibration and bias prediction pipeline
- Generated cross-validation metrics
- Produced sensitivity analysis plots

## Prerequisites

Make sure you've completed the [installation](../getting_started/installation.md) steps. You should be able to run:

```bash
python -c "import wifa_uq; print('Ready!')"
```

## The Workflow Runner

WIFA-UQ uses a simple command-line interface. The entry point is `examples/run.py`, which takes a YAML configuration file as input:

```bash
python examples/run.py <path-to-config.yaml>
```

## Step 1: Explore the Example Data

The `examples/` directory contains everything you need:

```
examples/
├── run.py                           # Workflow entry point
├── kul_les_example.yaml             # PCE-based workflow config
├── kul_single_farm_xgb_example.yaml # XGBoost workflow config
├── edf_single_farm_example.yaml     # EDF dataset config
├── multi_farm_example.yaml          # Multi-farm config
└── data/
    ├── KUL_LES/                     # KU Leuven LES dataset
    ├── EDF_datasets/                # EDF LES datasets
    └── ...
```

## Step 2: Run Your First Workflow

Let's start with the KUL LES dataset using PCE-based bias prediction:

```bash
cd examples
python run.py kul_les_example.yaml
```

You'll see output like:

```
--- Starting WIFA-UQ Workflow ---
Using config file: kul_les_example.yaml
Preprocessor initialized for originalData.nc.
Applying steps: ['recalculate_params']
  Running 'recalculate_params'...
    Calculating wind veer...
    Running 'ci_fitting' for thermal parameters...
Preprocessing complete.
--- Running Database Generation ---
Case: KUL_LES, 9 turbines, Rated Power: 3.6 MW, Hub Height: 90.0 m
Parameter sweep complete. Processing physical inputs...
...
--- Cross-Validation Results (mean) ---
rmse    0.023456
r2      0.876543
mae     0.018234
--- Workflow complete ---
```

## Step 3: Examine the Results

After the workflow completes, check the output directory:

```bash
ls data/KUL_LES/wifa_uq_results/
```

You'll find:

| File | Description |
|------|-------------|
| `processed_physical_inputs.nc` | Preprocessed atmospheric data |
| `results_stacked_hh.nc` | Model error database |
| `cv_results.csv` | Cross-validation metrics per fold |
| `predictions.npz` | Raw predictions for analysis |
| `correction_results.png` | Scatter plots of model correction |
| `bias_prediction_*.png` | Sensitivity analysis plots |

## Step 4: View the Results

Open `correction_results.png` to see three panels:

1. **ML Model Performance** — Predicted vs true bias (should cluster around 1:1 line)
2. **Uncorrected Model** — Raw PyWake power vs reference
3. **Corrected Model** — PyWake power after bias correction (should be tighter)

The sensitivity plots show which features most influence the bias:

- `bias_prediction_shap.png` — SHAP beeswarm plot (for XGBoost)
- `bias_prediction_sir_importance.png` — SIR direction coefficients
- `pce_sobol_indices.png` — PCE-based Sobol indices

## Understanding the Config File

Let's look at the key sections of `kul_les_example.yaml`:

```yaml
# Input/output paths (relative to the YAML file)
paths:
  system_config: data/KUL_LES/wind_energy_system/system_pywake.yaml
  reference_power: data/KUL_LES/observed_output/observedPowerKUL.nc
  reference_resource: data/KUL_LES/plant_energy_resource/originalData.nc
  wind_farm_layout: data/KUL_LES/plant_wind_farm/FLOW_UQ_vnv_toy_study_wind_farm.yaml
  output_dir: data/KUL_LES/wifa_uq_results

# Preprocessing: recalculate derived quantities
preprocessing:
  run: true
  steps: [recalculate_params]

# Database generation: sweep uncertain parameters
database_gen:
  run: true
  flow_model: pywake
  n_samples: 100
  param_config:
    attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b:
      range: [0.01, 0.07]
      default: 0.04
      short_name: "k_b"

# Error prediction: ML model and cross-validation
error_prediction:
  run: true
  features: [ABL_height, wind_veer, lapse_rate]
  model: "PCE"
  calibrator: LocalParameterPredictor
  cross_validation:
    splitting_mode: kfold_shuffled
    n_splits: 6
```

## Try Different Configurations

### XGBoost Instead of PCE

Run the XGBoost-based workflow:

```bash
python run.py kul_single_farm_xgb_example.yaml
```

This uses:
- XGBoost gradient boosting for bias prediction
- SHAP values for sensitivity analysis
- More physical features (blockage metrics, farm geometry)

### EDF Datasets

Try a different wind farm dataset:

```bash
python run.py edf_single_farm_example.yaml
```

The EDF datasets include various virtual and real wind farm configurations.

## Common Workflow Patterns

### Skip Preprocessing (Use Existing Data)

If you've already preprocessed the data:

```yaml
preprocessing:
  run: false
```

### Skip Database Generation (Use Existing Database)

If you've already generated the error database:

```yaml
database_gen:
  run: false
```

### Change the Number of Parameter Samples

More samples = better coverage but slower:

```yaml
database_gen:
  n_samples: 200  # Default is 100
```

### Change Cross-Validation Strategy

For leave-one-group-out CV (useful with multiple farms):

```yaml
cross_validation:
  splitting_mode: LeaveOneGroupOut
  groups:
    Group1: [Farm1, Farm2]
    Group2: [Farm3, Farm4]
```

## What's Happening Under the Hood?

1. **Preprocessing** (`PreprocessingInputs`)
   - Loads raw atmospheric data
   - Calculates derived quantities: ABL height, wind veer, lapse rate, TI

2. **Database Generation** (`DatabaseGenerator`)
   - Samples uncertain wake model parameters (k_b, ss_alpha, etc.)
   - Runs PyWake for each sample
   - Computes bias = (model - reference) / rated_power
   - Adds layout features (blockage ratio, farm dimensions)

3. **Calibration** (`MinBiasCalibrator` or `LocalParameterPredictor`)
   - Global: Find single best parameter set
   - Local: Predict optimal parameters as f(atmospheric state)

4. **Bias Prediction** (`BiasPredictor` with XGB/SIR/PCE)
   - Train ML model: bias = f(features)
   - Cross-validate to assess generalization

5. **Sensitivity Analysis**
   - SHAP/SIR/Sobol to identify important features

## Next Steps

- [Project Structure](../workflow/project_structure.md) — Understand the codebase organization
- [Configuration Reference](../user_guide/configuration.md) — All YAML options explained
- [Tutorials](../tutorials/index.md) — Step-by-step guides for specific use cases

## Troubleshooting

### "FileNotFoundError: System YAML not found"

Check that paths in your config are relative to the YAML file location, not the current directory.

### "ValueError: Could not find or infer 'rated_power'"

Your wind farm definition is missing turbine power information. Add `performance.rated_power` to your turbine YAML. See [Metadata Note](../appendix/metadata_note.md).

### Workflow is Very Slow

- Reduce `n_samples` in `database_gen`
- Ensure you're not re-running preprocessing/database_gen unnecessarily
- Check that numba JIT compilation is working (first run is slower)
