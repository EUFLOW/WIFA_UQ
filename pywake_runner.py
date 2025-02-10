import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from wifa.pywake_api import run_pywake
from windIO.utils.yml_utils import validate_yaml, load_yaml
from scipy.interpolate import griddata

kk = 0.04

dat = load_yaml('./windIO_1WT/wind_energy_system/system.yaml')
run_pywake(dat, output_dir='k_%.2f' % kk)
flow_field = xr.load_dataset(f'k_{kk:.2f}/FarmFlow.nc')

reference_flow_field = xr.load_dataset('result_code_saturne_1WT_LIGHT/single_time_flow_field.nc')
#reference_flow_field = reference_flow_field.where(reference_flow_field.x > 0, drop=True)
#reference_flow_field = reference_flow_field.where(reference_flow_field.x < 5000, drop=True)
#reference_flow_field = reference_flow_field.where(reference_flow_field.y < 500, drop=True)
#reference_flow_field = reference_flow_field.where(reference_flow_field.y > -500, drop=True)

x_grid = x_list = flow_field.x
y_grid = y_list = flow_field.y
x_grid, y_grid = np.meshgrid(x_list, y_list)
speed_interp = griddata((reference_flow_field.x, reference_flow_field.y), reference_flow_field.wind_speed, (x_grid, y_grid), method='cubic')

interpolated_reference_data = xr.Dataset(
        data_vars={
            'speed': (['y', 'x', 'time'], speed_interp[:, :, 0, :])
        },
        coords={
            'x': x_list,
            'y': y_list,
            'time': reference_flow_field.time
        }
    )

wind_diffs = []
kk_values = [.05, .1]
for kk in kk_values:
    dat['attributes']['analysis']['wind_deficit_model']['wake_expansion_coefficient']['k_b'] = kk
    run_pywake(dat, output_dir='k_%.2f' % kk)

    flow_field = xr.load_dataset(f'k_{kk:.2f}/FarmFlow.nc')

    wind_diff = flow_field.effective_wind_speed - interpolated_reference_data.speed
    wind_diffs.append(wind_diff)

final_diffs = xr.concat(wind_diffs, dim=pd.Index(kk_values, name='kk'))

refdat = xr.load_dataset('windIO_1WT/plant_energy_resource/1WT_calibration_data_IEA15MW.nc')

	
# Add all relevant variables from refdat as coordinates to final_diffs
variables_to_add = ['z0', 'ABL_height', 'wind_speed', 'wind_direction', 'LMO', 
                   'lapse_rate', 'capping_inversion_strength', 
                   'capping_inversion_thickness', 'TI']

for var in variables_to_add:
    final_diffs = final_diffs.assign_coords({var: ('time', refdat[var].values)})

# If you want to replace the integer time coordinates with float coordinates:
final_diffs = final_diffs.assign_coords(time=refdat.time)
