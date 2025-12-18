# WIFA-UQ

[![Coverage](https://EUFLOW.github.io/WIFA_UQ/coverage.svg)](https://EUFLOW.github.io/WIFA_UQ/coverage/)
[![Documentation](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://EUFLOW.github.io/WIFA_UQ/)

## Description


Uncertainty quantification for wind farm wake modeling.

This repository is designed to develop an open-source Python framework integrating tools to perform calibration and bias correction with uncertainty quantification. It is built on top of WIFA, which uses WindIO as an input data standard.

## Features

- **Model Error Databases** — Sample uncertain wake model parameters and compare against reference data (LES or SCADA)
- **Flexible Calibration** — Global (single best parameters) or local (condition-dependent) calibration strategies
- **ML Bias Prediction** — XGBoost, PCE, SIR, and linear models to predict residual bias from atmospheric features
- **Sensitivity Analysis** — SHAP values, Sobol indices, and SIR directions for feature importance
- **Multi-Farm Support** — Combine data from multiple wind farms with Leave-One-Group-Out cross-validation
- **windIO Integration** — Uses the [windIO](https://github.com/EUFLOW/windIO) standard for wind energy system definitions


## Documentation

Full documentation is available at **[EUFLOW.github.io/WIFA_UQ](https://EUFLOW.github.io/WIFA_UQ/)**

- [Installation Guide](https://EUFLOW.github.io/WIFA_UQ/getting_started/installation/)
- [Quickstart Tutorial](https://EUFLOW.github.io/WIFA_UQ/dependencies/quickstart/)
- [Configuration Reference](https://EUFLOW.github.io/WIFA_UQ/user_guide/configuration/)
- [API Concepts](https://EUFLOW.github.io/WIFA_UQ/concepts/overview/)



### Useful links

* [WIFA public repository](https://github.com/EUFLOW/WIFA)
* [WIFA's online documentation](https://EUFLOW.github.io/WIFA_UQ/)
* [WindIO repository](https://github.com/EUFLOW/windIO)



## Quick Start

```
pixi install
pixi run python examples/run.py examples/kul_les_example.yaml
```

## Architecture Diagrams
```

┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Reference Data          Wake Model (e.g., PyWake/foxes/WAYVE) │
│   (LES / SCADA)          with uncertain params                  │
│        │                        │                               │
│        │    ┌───────────────────┘                               │
│        ▼    ▼                                                   │
│   ┌──────────────┐                                              │
│   │  Calibration │ → Find params that minimize bias             │
│   └──────────────┘                                              │
│           │                                                     │
│           ▼                                                     │
│   ┌──────────────┐                                              │
│   │ Bias         │ → Learn: bias = f(ABL_height, wind_veer,...) │
│   │ Prediction   │                                              │
│   └──────────────┘                                              │
│           │                                                     │
│           ▼                                                     │
│   Corrected Output = PyWake(calibrated) - predicted_bias        │
│                                                                 │
│   Result: Lower error on held-out test cases                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────────┐   │
│  │ Preprocessing│───▶│  Database    │───▶│ Error Prediction │   │
│  │              │    │  Generation  │    │                  │   │
│  └──────────────┘    └──────────────┘    └─────────┬────────┘   │
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
