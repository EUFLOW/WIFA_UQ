import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path

error_method = "rmse"
vrs = ["best_k", "best_k_error"]
lbl = ["best_k", "errorP"]
rdir = Path("results") / error_method
odir = rdir / "figures" / "evaluate"
cases_file = rdir / "calibration_results.nc"

for v, l in zip(vrs, lbl):

    ds = xr.open_dataset(cases_file)

    # Check for 'best_k' variable
    if v not in ds.variables:
        raise ValueError(f"Variable '{v}' not found in the NetCDF file.")

    best_k = ds[v].values

    # Plot best_k vs every other variable
    dr = odir / v
    dr.mkdir(parents=True, exist_ok=True)
    for var in ds.variables:
        if var == v:
            continue
        data = ds[var].values
        # Flatten if needed
        if data.shape != best_k.shape:
            try:
                data = data.flatten()
            except Exception:
                print(f"Skipping variable {var} due to shape mismatch.")
                continue
        fpath = dr / f'eval_{l}_vs_{var}.png'
        print(f"Writing file {fpath}")
        plt.figure()
        plt.scatter(data, best_k, alpha=0.7)
        if var == "best_k_error":
            plt.xlabel("Power Error (RMSE/capacity)")
        else:
            plt.xlabel(var)
        plt.ylabel(l)
        plt.title(f'{l} vs {var}')
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(fpath)
        plt.close()
