import numpy as np
from xarray import Dataset
from pathlib import Path
from xarray import open_dataset
import foxes
import foxes.variables as FV
import foxes.constants as FC

datasets_dir = Path("../model_error_database/EDF_datasets/")
cases_dirs = list(datasets_dir.glob("*"))
ref_height = 100.0
k_values = np.linspace(0.01, 0.1, 10)


def calc_error(cfd_results, pop_results):
    # RMSE:
    P_cfd = cfd_results["power"].mean(dim="time").values
    P_fxs = pop_results[FV.P].mean(dim="state").values
    # mse = np.sqrt(np.mean((P_cfd[None] - P_fxs) ** 2, axis=(1, 2)))
    mse = np.sqrt(np.mean((P_cfd[None] - P_fxs) ** 2, axis=1))
    return mse


n_cases = len(cases_dirs)
all_results_coords = {
    "case": [case_dir.name for case_dir in cases_dirs],
}
all_results_data = {
    "wake_k": (("scan_index",), k_values),
    "error": (
        (
            "case",
            "scan_index",
        ),
        np.full((n_cases, len(k_values)), np.nan),
    ),
}

with foxes.Engine.new("process", verbosity=0):
    for ci, case_dir in enumerate(cases_dirs):
        print(f"\nENTERING CASE {ci}/{n_cases}: {case_dir}")

        physical_inputs = open_dataset(case_dir / "updated_physical_inputs.nc")
        physical_inputs = physical_inputs.interp(height=ref_height).mean(dim="time")
        for v, d in physical_inputs.data_vars.items():
            if ci == 0:
                all_results_data[v] = (("case",), np.full(n_cases, np.nan))
            if v in all_results_data:
                all_results_data[v][1][ci] = d.values

        tdfiles = list(case_dir.glob("turbine_data*.nc"))
        assert len(tdfiles) == 1, f"Expected one turbine data file, found {tdfiles}"
        cfd_results = open_dataset(tdfiles[0])
        sysfiles = list(case_dir.glob("*system*.yaml"))
        assert len(sysfiles) == 1, f"Expected one system file, found {sysfiles}"
        wio_dict = foxes.input.yaml.windio.windio_file2dict(sysfiles[0])
        n_turbines = foxes.input.yaml.windio.read_n_turbines(wio_dict)

        n_pop = len(k_values)
        pop_data = np.zeros((n_pop, n_turbines))
        pop_data[:] = k_values[:, None]
        pop_data = Dataset(
            {
                FV.K: (("index", FC.TURBINE), pop_data),
            }
        )

        idict, algo, odir = foxes.input.yaml.windio.read_windio_dict(
            wio_dict,
            population_params=dict(data_source=pop_data, verbosity=0),
            verbosity=0,
        )
        for wmodel in algo.wake_models.values():
            if hasattr(wmodel, "wake_k"):
                wmodel.wake_k = foxes.core.WakeK()

        print("Running foxes")
        farm_results = algo.calc_farm()
        print("Done running foxes")

        pop_results = algo.population_model.farm2pop_results(algo, farm_results)
        k = pop_results[FV.K].values[:, 0, 0]

        P_mean = (
            (pop_results[FV.P] * pop_results[FV.WEIGHT])
            .sum(dim=FC.STATE)
            .sum(dim=FC.TURBINE)
        )
        error = calc_error(cfd_results, pop_results)
        all_results_data["error"][1][ci, :] = error
        print(f"Error: {error}")

print(Dataset(all_results_data, coords=all_results_coords))

quit()
"""

# Generate sample data
np.random.seed(0)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()

# Add noise to the data
y += 0.1 * np.random.randn(80)

# Define the kernel (RBF kernel)
kernel = 1.0 * RBF(length_scale=1.0)

# Create a Gaussian Process Regressor with the defined kernel
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=0)

# Fit the Gaussian Process model to the training data
gp.fit(X_train, y_train)

# Make predictions on the test data
y_pred, sigma = gp.predict(X_test, return_std=True)

# Visualize the results
x = np.linspace(0, 5, 1000)[:, np.newaxis]
y_mean, y_cov = gp.predict(x, return_cov=True)

plt.figure(figsize=(10, 5))
plt.scatter(X_train, y_train, c='r', label='Training Data')
plt.plot(x, y_mean, 'k', lw=2, zorder=9, label='Predicted Mean')
plt.fill_between(x[:, 0], y_mean - 1.96 * np.sqrt(np.diag(y_cov)), y_mean + 1.96 *
                 np.sqrt(np.diag(y_cov)), alpha=0.2, color='k', label='95% Confidence Interval')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.show()
"""
