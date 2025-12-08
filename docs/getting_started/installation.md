# Installation

WIFA-UQ supports two installation methods: **pixi** (recommended) for reproducible environments, and **pip** for standard Python installations.

## Prerequisites

- **Python 3.8–3.13** (Python 3.12 recommended)
- **Git** for cloning the repository and installing dependencies from Git URLs
- **C compiler** for building some dependencies (gcc on Linux, Xcode on macOS)

## Method 1: Using Pixi (Recommended)

[Pixi](https://pixi.sh) is a fast, reproducible package manager that handles both conda and pip dependencies. It's the recommended way to install WIFA-UQ for consistent environments across machines.

### Step 1: Install Pixi

```bash
# Linux/macOS
curl -fsSL https://pixi.sh/install.sh | bash

# Or with Homebrew
brew install pixi
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/EUFLOW/WIFA-UQ.git
cd WIFA-UQ
```

### Step 3: Create the Environment

```bash
# Create the default environment
pixi install

# Or create the development environment (includes Jupyter, plotting tools)
pixi install -e dev
```

### Step 4: Activate and Use

```bash
# Run commands in the pixi environment
pixi run python examples/run.py examples/kul_les_example.yaml

# Or activate a shell
pixi shell
python examples/run.py examples/kul_les_example.yaml
```

### Available Environments

| Environment | Command | Includes |
|-------------|---------|----------|
| `default` | `pixi install` | Core dependencies |
| `dev` | `pixi install -e dev` | + Jupyter, cartopy, ncplot |
| `test` | `pixi install -e test` | + pytest, coverage |

### Pixi Tasks

WIFA-UQ defines several convenience tasks:

```bash
# Run unit tests
pixi run test

# Run tests with coverage
pixi run test-cov

# Run linter
pixi run lint
```

## Method 2: Using Pip

For standard pip installations, we recommend using a virtual environment.

### Step 1: Create a Virtual Environment

```bash
python -m venv wifa-uq-env
source wifa-uq-env/bin/activate  # Linux/macOS
# or: wifa-uq-env\Scripts\activate  # Windows
```

### Step 2: Clone and Install

```bash
git clone https://github.com/EUFLOW/WIFA-UQ.git
cd WIFA-UQ

# Install in editable mode
pip install -e .

# Or with test dependencies
pip install -e ".[test]"

# Or with development dependencies
pip install -e ".[dev]"
```

### Note on Git Dependencies

WIFA-UQ depends on several packages installed from Git repositories:

- **wayve** — KU Leuven wake modeling library
- **wifa** — WIFA framework (PyWake/FOXES/WAYVE APIs)
- **umbra** — Bayesian calibration tools
- **sliced** — Sliced Inverse Regression

These are automatically installed via pip from the URLs specified in `pyproject.toml`. If you encounter issues, ensure Git is available in your PATH.

## Verifying Installation

After installation, verify everything works:

```bash
# Check the package imports correctly
python -c "import wifa_uq; print('WIFA-UQ installed successfully!')"

# Run a quick test
python -c "from wifa_uq.workflow import run_workflow; print('Workflow module loaded!')"
```

## Core Dependencies

WIFA-UQ relies on several key packages:

| Package | Purpose |
|---------|---------|
| `py_wake` | PyWake wind farm simulation |
| `foxes` | FOXES wake modeling framework |
| `xarray` | NetCDF data handling |
| `scikit-learn` | ML pipeline infrastructure |
| `xgboost` | Gradient boosting models |
| `shap` | Explainable ML (SHAP values) |
| `openturns` | Polynomial Chaos Expansion |
| `numba` | JIT compilation for performance |

## Platform Notes

### Linux (Recommended)

Full support. All features work out of the box.

### macOS (Apple Silicon)

Supported via pixi with `osx-arm64` platform. Some packages may require Rosetta 2 for x86 emulation.

### Windows

Limited testing. We recommend using WSL2 (Windows Subsystem for Linux) for the best experience.

## Troubleshooting

### ImportError: No module named 'sliced'

The `sliced` package is required for SIR-based regression. Install it manually if needed:

```bash
pip install git+https://github.com/kilojoules/sliced@ee892bdbde4caf66e8afe5ce7bf48367d0ef6273
```

### UMBRA Installation Issues

UMBRA (for Bayesian calibration) requires a working C++ compiler. On Linux:

```bash
sudo apt-get install build-essential
```

On macOS, install Xcode command line tools:

```bash
xcode-select --install
```

### HDF5/NetCDF Errors

If you encounter HDF5 or NetCDF4 errors, ensure the system libraries are installed:

```bash
# Ubuntu/Debian
sudo apt-get install libhdf5-dev libnetcdf-dev

# macOS (via Homebrew)
brew install hdf5 netcdf
```

With pixi, these are handled automatically via conda-forge.

### Memory Issues with Large Databases

For large parameter sweeps (>1000 samples), you may need to increase available memory or process data in chunks. Consider:

- Reducing `n_samples` in your config
- Using a machine with more RAM
- Processing farms individually before combining

## Next Steps

- Continue to [Quickstart](../dependencies/quickstart.md) to run your first workflow
- See [Project Structure](../workflow/project_structure.md) to understand the codebase
