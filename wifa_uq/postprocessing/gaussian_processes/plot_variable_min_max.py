from pathlib import Path
import xarray as xr
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to NetCDF file
nc_path = os.path.join(
    os.path.dirname(__file__),
    'results', 'rmse', 'calibration_results.nc'
)
output_dir = Path(os.path.dirname(__file__)) / 'results' / 'rmse' / 'figures'
excl = ["best_k", "best_k_error", "best_k_farm_P", "time_index"]

# Load dataset
with xr.open_dataset(nc_path) as ds:
    var_names = [v for v in ds.variables if v not in excl]
    min_vals = []
    max_vals = []
    valid_vars = []
    scale_factors = []
    vrs = []
    for var in var_names:
        arr = ds[var].values.flatten()
        # Only plot numeric variables
        if arr.dtype.kind not in 'fi':
            continue
        vrs.append(var)
        if var == "k":
            var = "RANS k"
        arr = arr[~np.isnan(arr)]
        if arr.size == 0:
            continue
        mn = np.min(arr)
        mx = np.max(arr)
        print("VAR",var,mn,mx)
        # Find power of 10 to scale into [-10, 10]
        power = 1
        while abs(mx) / (10**power) > 10:
            power += 1
        while abs(mx) / (10**power) < 1:
            power -= 1
        scale = 10**power
        min_vals.append(mn/scale)
        max_vals.append(mx/scale)
        scale_factors.append(scale)
        valid_vars.append(var)

fpath = output_dir / 'variable_min_max_barplot.png'
output_dir.mkdir(parents=True, exist_ok=True)
print(f"Saving plot to {fpath}")
plt.figure(figsize=(8, max(4, len(valid_vars) * 0.4)))
for i, (mn, mx) in enumerate(zip(min_vals, max_vals)):
    plt.barh(i, mx - mn, left=mn, color='skyblue', edgecolor='k')
    # Add dots for each data entry
    arr = ds[vrs[i]].values.flatten()
    arr = arr[~np.isnan(arr)]
    scale = scale_factors[i]
    xvals = arr / scale
    # Random y jitter within the bar
    yvals = i + (np.arange(182)/181 - 0.5) * 0.8  # 0.6 for some spread
    plt.plot(xvals, yvals, 'k.', alpha=0.2, markersize=7)
def sci_notation(s):
    return f"{s:.0e}"
plt.yticks(range(len(valid_vars)), [
    f"{v} ({sci_notation(s)})" for v, s in zip(valid_vars, scale_factors)
])
plt.xlabel('Scaled Value')
plt.title('Min/Max Ranges of Variables (dots are case indices)')
plt.tight_layout()
plt.savefig(fpath)

