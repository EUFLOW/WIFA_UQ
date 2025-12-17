# PCE Training and Validation Tool

This repository provides a workflow to build, evaluate, and visualize **Polynomial Chaos Expansion (PCE) metamodels** for wind model bias emulation.
It supports optional configuration parameters, flexible plotting, and reproducible random sampling.

---

##  Repository structure

- config.yaml # Main configuration file (user editable)
- main.py # Entry point of the workflow
- pce_utils.py # Utilities: PCE routines, plotting, metrics
- plots/ # (Created automatically) folder for saved plots
- README.md # Documentation

##  Outputs

Depending on configuration, the tool can produce:

- **Scatter plots**
  Observed vs. PCE predictions for randomly selected time indices.

- **Distribution comparisons**
  PDFs comparing observed and PCE biases for randomly selected time indices.

- **Metric plots**
  Time-series of evaluation metrics:
  - RMSE
  - R²
  - Wasserstein distance
  - KS statistic
  - KL divergence
