# FLOW META API

## Description

This repository is created in the context of Task 4.3 of the FLOW project designed to develop an open-source Python framework integrating tools to perform calibration and bias correction with uncertainty quantification.

## Useful links

* [WIFA public repository](https://github.com/EUFLOW/WIFA)
* [WIFA's online documentation](https://eu-flow.pages.windenergy.dtu.dk/wp4/FLOW_API/)
* [WindIO repository](https://github.com/EUFLOW/windIO)


## Folder breakdown
The wifa_uq folder contains the various modules which can be imported to carry out various tasks such as wifa_uq/preprocessing, (scripts related to preprocessing raw data prior to using model error database), wifa_uq/model_error_database (scripts related to generating the database of model biases), wifa_uq/postprocessing (scripts related to various applications of wifa UQ (e.g., PCE, bayesian calibration, error predictor) ).

The examples folder contains implementations of bayesian calibration and model calibration & bias prediction using wifa_uq. The data used in the examples is contained here also

### model_error_database
- run_pywake_sweep.py contains a function to run the pywake api for a range of defined parameters (for a given flow case).

- database_gen.py runs the pywake sweep function for all cases in a given dataset, appends other features and returns the database in netCDF form.

- utils.py contains functions for calculating additional features in the model bias database (i.e., blockage metrics etc.)

## Multi-Farm Workflows

WIFA-UQ supports generating combined databases from multiple wind farms for
cross-validation studies.

### Configuration

Use the `farms` key in your config to specify multiple wind farms:
```yaml
paths:
  output_dir: combined_results/
  database_file: results_stacked_hh.nc

farms:
  - name: Farm1
    system_config: path/to/farm1/system.yaml
    reference_power: path/to/farm1/power.nc
    reference_resource: path/to/farm1/resource.nc
    wind_farm_layout: path/to/farm1/layout.yaml
  - name: Farm2
    system_config: path/to/farm2/system.yaml
    # ...

database_gen:
  run: true
  # ... same as single-farm config

error_prediction:
  cross_validation:
    splitting_mode: LeaveOneGroupOut
    groups:
      Group1:
        - Farm1
        - Farm2
      Group2:
        - Farm3
```

### How It Works

1. Each farm is processed independently (preprocessing + database generation)
2. Individual farm databases are combined into a single dataset
3. The `wind_farm` coordinate preserves farm identity
4. Cross-validation can use `LeaveOneGroupOut` to test generalization across farms

### preprocessing
the preprocessing.py scripts carries out several steps such as interpolating the height dimension such that all cases are the same, and recalculating atmospheric input parameters from the vertical profile of potential temperature. These are based on the requirements of the datasets used so far. Further steps may be required for future datasets.

### postprocessing
contains scripts related to various applications of wifa UQ (e.g., PCE, bayesian calibration, error predictor).

## Implementation
### examples/torque_workflow
a config .yaml file is created to describe the workflow for a given case.

the main.py script will interpret the config script and execute the workflow using the wifa_uq modules

## Architecture Diagram
```
┌─────────────────────────────────────────────────────────────────┐
│                        workflow.py                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Preprocessing │───▶│  Database    │───▶│ Error Prediction│   │
│  │              │    │  Generation  │    │                  │   │
│  └──────────────┘    └──────────────┘    └────────┬─────────┘   │
│                                                    │            │
│                      ┌─────────────────────────────┼────────┐   │
│                      │                             ▼        │   │
│                      │  ┌─────────────┐   ┌──────────────┐  │   │
│                      │  │ Calibrator  │   │ BiasPredictor│  │   │
│                      │  │ (Global/    │   │ (XGB/SIR)    │  │   │
│                      │  │  Local)     │   └──────────────┘  │   │
│                      │  └─────────────┘                     │   │
│                      │         MainPipeline                 │   │
│                      └──────────────────────────────────────┘   │
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Sensitivity Analysis                        │   │
│  │  ┌────────┐  ┌─────────────────┐  ┌─────────────────┐    │   │
│  │  │  SHAP  │  │ SIR Directions  │  │  PCE Sobol      │    │   │
│  │  └────────┘  └─────────────────┘  └─────────────────┘    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
