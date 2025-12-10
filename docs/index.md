# WIFA-UQ

**Uncertainty Quantification for Wind Farm Wake Modeling**

WIFA-UQ is an open-source Python framework for calibration, bias correction, and uncertainty quantification in wind farm simulations. Developed as part of the FLOW project (Task 4.3), it provides tools to systematically reduce and quantify model errors in wake modeling.

## What WIFA-UQ Does

Wind farm wake models (like PyWake, FOXES, and WAYVE) have inherent uncertainties due to simplified physics and uncertain input parameters. WIFA-UQ addresses this by:

1. **Generating Model Error Databases** — Systematically sampling uncertain parameters and comparing model outputs to reference data (LES simulations or SCADA measurements)

2. **Calibrating Model Parameters** — Finding optimal parameter values that minimize bias, either globally across all conditions or locally as a function of atmospheric state

3. **Predicting Residual Bias** — Training machine learning models to predict remaining bias as a function of physical features (ABL height, wind veer, turbulence intensity, etc.)

4. **Quantifying Uncertainty** — Using techniques like Polynomial Chaos Expansion (PCE) and Bayesian inference to propagate uncertainty through the modeling chain

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   Reference Data          Wake Model (e.g., PyWake/FOXES/WAYVE) │
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
│   Corrected Output = Model(calibrated) - predicted_bias         │
│                                                                 │
│   Result: Lower error on held-out test cases                    │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

- **windIO Integration** — Uses the [windIO](https://github.com/EUFLOW/windIO) standard for wind energy system definitions
- **Multiple Calibration Strategies** — Global (MinBiasCalibrator), local (LocalParameterPredictor), and Bayesian approaches
- **Flexible ML Models** — XGBoost, SIR+Polynomial, PCE, and linear regressors for bias prediction
- **Sensitivity Analysis** — SHAP values, SIR directions, and PCE-based Sobol indices
- **Multi-Farm Support** — Combine data from multiple wind farms with Leave-One-Group-Out cross-validation
- **Reproducible Workflows** — YAML-based configuration for experiment tracking

## Quick Example

```bash
# Run a complete workflow
python examples/run.py examples/kul_les_example.yaml
```

This executes:
1. Preprocessing of atmospheric data
2. Database generation via PyWake parameter sweeps
3. Cross-validated error prediction with sensitivity analysis

## Getting Started

- [Installation](getting_started/installation.md) — Set up your environment with pixi or pip
- [Quickstart](dependencies/quickstart.md) — Run your first workflow in 5 minutes
- [Project Structure](workflow/project_structure.md) — Understand the repository layout

## Links

- **Repository**: [github.com/EUFLOW/WIFA-UQ](https://github.com/EUFLOW/WIFA-UQ)
- **WIFA Framework**: [github.com/EUFLOW/WIFA](https://github.com/EUFLOW/WIFA)
- **windIO Standard**: [github.com/EUFLOW/windIO](https://github.com/EUFLOW/windIO)
- **WIFA Documentation**: [eu-flow.pages.windenergy.dtu.dk/wp4/FLOW_API](https://eu-flow.pages.windenergy.dtu.dk/wp4/FLOW_API)

## Citation

If you use WIFA-UQ in your research, please cite:

```bibtex
@software{wifa_uq,
  title = {WIFA-UQ: Uncertainty Quantification for Wind Farm Wake Modeling},
  author = {Quick, Julian and Mouradi, Rem-Sophia and Schulte, Jonas and
            Devesse, Koen and Aerts, Frederik and Mathieu, Antoine},
  year = {2024},
  url = {https://github.com/EUFLOW/WIFA-UQ}
}
```

## License

WIFA-UQ is released under the MIT License. See [LICENSE](https://github.com/EUFLOW/WIFA-UQ/blob/main/LICENSE) for details.
