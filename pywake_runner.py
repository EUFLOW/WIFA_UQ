import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from wifa.pywake_api import run_pywake
from sklearn.linear_model import LinearRegression
from windIO.utils.yml_utils import validate_yaml, load_yaml
from scipy.interpolate import griddata

kk = 0.04

dat = load_yaml('./1WT_simulations/windIO_1WT/wind_energy_system/system.yaml')
run_pywake(dat, output_dir='k_%.2f' % kk)
flow_field = xr.load_dataset(f'k_{kk:.2f}/FarmFlow.nc')

reference_flow_field = xr.load_dataset('1WT_simulations/result_code_saturne_1WT_LIGHT/single_time_flow_field.nc')
#reference_flow_field = reference_flow_field.where(reference_flow_field.x > 0, drop=True)
#reference_flow_field = reference_flow_field.where(reference_flow_field.x < 5000, drop=True)
#reference_flow_field = reference_flow_field.where(reference_flow_field.y < 500, drop=True)
#reference_flow_field = reference_flow_field.where(reference_flow_field.y > -500, drop=True)

x_grid = x_list = flow_field.x
y_grid = y_list = flow_field.y
x_grid, y_grid = np.meshgrid(x_list, y_list)

# First try reshaping the input arrays to do all time steps at once
x_ref, y_ref = np.meshgrid(reference_flow_field.x, reference_flow_field.y)
points = np.column_stack((x_ref.ravel(), y_ref.ravel()))

# Reshape wind speed data to be (n_points, n_times)
wind_speed_reshaped = reference_flow_field.wind_speed.values[:, 0, :, :].transpose(0, 2, 1).reshape(len(reference_flow_field.time), -1).T

# Do interpolation for all time steps at once
speed_interp = griddata(
    points,
    wind_speed_reshaped,
    (x_grid, y_grid),
    method='cubic'
)

# Reshape result back to original dimensions
speed_interp = speed_interp.reshape(len(y_list), len(x_list), len(reference_flow_field.time))

interpolated_reference_data = xr.Dataset(
    data_vars={
        'speed': (['y', 'x', 'time'], speed_interp)
    },
    coords={
        'x': x_list,
        'y': y_list,
        'time': reference_flow_field.time
    }
)

wind_diffs = []
kk_values = np.arange(0.01, 0.3, 0.01)
#kk_values = [.05, .1]
for kk in kk_values:
    dat['attributes']['analysis']['wind_deficit_model']['wake_expansion_coefficient']['k_b'] = kk
    run_pywake(dat, output_dir='k_%.2f' % kk)

    flow_field = xr.load_dataset(f'k_{kk:.2f}/FarmFlow.nc')

    wind_diff = flow_field.wind_speed - interpolated_reference_data.speed
    wind_diffs.append(np.sqrt(((wind_diff ** 2).sum(['x', 'y']).isel(z=0))))
    plt.contourf(flow_field.x, flow_field.y, flow_field.isel(time=0, z=0).wind_speed, 100)
    plt.title('k=%.2f' % kk)
    plt.savefig('figs/k_%.2f.png' % kk)
    plt.clf()

refdat = xr.load_dataset('1WT_simulations/windIO_1WT/plant_energy_resource/1WT_calibration_data_IEA15MW.nc')

	
# Add all relevant variables from refdat as coordinates to final_diffs
variables_to_add = ['z0', 'ABL_height', 'wind_speed', 'wind_direction', 'LMO', 
                   'lapse_rate', 'capping_inversion_strength', 
                   'capping_inversion_thickness', 'turbulence_intensity']

# If you want to replace the integer time coordinates with float coordinates:

final_diffs = xr.concat(wind_diffs, dim=pd.Index(kk_values, name='kk'))
best_ks = kk_values[final_diffs.argmin('kk').values]
finaldat = refdat.assign_coords({'optimal_k': ('time', best_ks)})


for var in variables_to_add:
    final_diffs = final_diffs.assign_coords({var: ('time', refdat[var].values)})

final_diffs = final_diffs.assign_coords(time=refdat.time)


for tt in range(finaldat.time.size):

    this_k = float(finaldat.optimal_k[tt].values)

    flow_field = xr.load_dataset(f'k_{this_k:.2f}/FarmFlow.nc').isel(z=0, time=tt)

    fig, ax = plt.subplots(3, figsize=(5, 10))
    ax[0].set_title('time=%i, best k is %.2f' % (tt, this_k))
    ax[0].contourf(reference_flow_field.x, reference_flow_field.y, reference_flow_field.wind_speed.isel(time=tt).values[0].T, 100)
    ax[1].contourf(flow_field.x, flow_field.y, flow_field.wind_speed, 100)
    ax[2].contourf(flow_field.x, flow_field.y, reference_flow_field.wind_speed.isel(time=tt).values[0].T - flow_field.wind_speed, 100)
    plt.savefig('figs/time_%i' % tt)
    plt.clf()
    plt.close()




features = ['z0', 'ABL_height', 'wind_speed', 'wind_direction', 'LMO', 
            'lapse_rate', 'capping_inversion_strength', 
            'capping_inversion_thickness', 'turbulence_intensity']

# Convert to pandas DataFrame for easier manipulation
df = finaldat.to_dataframe()

# Separate features and target
X = df[features]
Y = df['optimal_k']

# Create and fit the linear regression model
model = LinearRegression()
model.fit(X, Y)

X_proj = X.dot(model.coef_)

fig, ax = plt.subplots(2, 1, figsize=(5, 10))
ax[0].scatter(X_proj, Y)
ax[0].set_xlabel('Projected Dimension')
ax[0].set_ylabel('Optimal k')
ax[1].bar(range(len(features)), (model.coef_))
ax[1].set_xticks(range(len(features)))
ax[1].axhline(0, c='k', ls='--')
ax[1].set_xticklabels(features, rotation=45)
ax[1].set_ylabel('Value of Projection')
plt.tight_layout()
plt.savefig('projection')
plt.clf()
