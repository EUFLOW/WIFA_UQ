import numpy as np
import xarray as xr
from typing import Dict, List, Union, Tuple
from windIO.utils.yml_utils import validate_yaml, load_yaml
from wifa.pywake_api import run_pywake
import matplotlib.pyplot as plt
import shutil


def set_nested_dict_value(d: dict, path: List[str], value: float) -> None:
    """Set value in nested dictionary using a path list."""
    current = d
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value

def create_parameter_samples(param_config: Dict[str, Tuple[float, float]], n_samples: int, seed: int = None) -> Dict[str, np.ndarray]:
    """
    Create samples for multiple parameters based on their ranges.
    
    Args:
        param_config: Dictionary mapping parameter paths to (min, max) tuples
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        Dictionary mapping parameter paths to arrays of samples
    """
    if seed is not None:
        np.random.seed(seed)
    
    return {
        param: np.random.uniform(min_val, max_val, n_samples)
        for param, (min_val, max_val) in param_config.items()
    }

# Example usage:
param_config = {
    "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": (0.01, 0.3),
    "attributes.analysis.blockage_model.ss_alpha": (0.01, 0.3)
}

def run_parameter_sweep(dat: dict, param_config: Dict[str, Tuple[float, float]], n_samples: int = 30, seed: int = None) -> List[xr.Dataset]:
    """
    Run parameter sweep across multiple parameters.
    
    Args:
        dat: Base configuration dictionary
        param_config: Dictionary mapping parameter paths to (min, max) tuples
        n_samples: Number of samples to generate
        seed: Random seed for reproducibility
    
    Returns:
        List of output datasets
    """
    # Generate samples for all parameters
    samples = create_parameter_samples(param_config, n_samples, seed)
    
    wind_diffs = []
    param_values = {}
    
    for i in range(n_samples):
        # Update all parameters for this sample
        for param_path, param_samples in samples.items():
            # Convert string path to list of keys
            path = param_path.split('.')
            set_nested_dict_value(dat, path, param_samples[i])
        
        # Run simulation
        run_pywake(dat, output_dir=f'sample_{i}')
        
        # Process results (your existing processing code)
        flow_field = xr.load_dataset(f'sample_{i}/FarmFlow.nc')
        wind_diff = flow_field.wind_speed - reference_flow_field.wind_speed
        wind_diffs.append(np.sqrt(((wind_diff ** 2).sum(['x', 'y']).isel(z=0))))
        shutil.rmtree(f'sample_{i}')
        
        # Plotting (if desired)
        plt.contourf(flow_field.x, flow_field.y, flow_field.isel(time=0, z=0).wind_speed, 100)
        plt.title(f'Sample {i}')
        plt.savefig(f'figs/sample_{i}.png')
        plt.clf()
    
    # Create final dataset
    combined_wind_diffs = xr.concat(wind_diffs, dim='sample')
    
    # Add parameter values to dataset
    merged_data = xr.Dataset(
        data_vars={'wind_diff': combined_wind_diffs},
        coords={
            param_path.split('.')[-1]: xr.DataArray(
                param_samples,
                dims=['sample'],
                coords={'sample': np.arange(len(param_samples))}
            )
            for param_path, param_samples in samples.items()
        }
    )
    
    # Add reference data
    refdat = xr.load_dataset('1WT_simulations/windIO_1WT/plant_energy_resource/1WT_calibration_data_IEA15MW.nc')
    for var in refdat.data_vars:
        merged_data[var] = refdat[var]
    
    return merged_data

# Example usage:
param_config = {
    "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": (0.01, 0.3),
    "attributes.analysis.blockage_model.ss_alpha": (0.01, 0.3)
}

dat = load_yaml('./1WT_simulations/windIO_1WT/wind_energy_system/system.yaml')
reference_flow_field = xr.load_dataset('1WT_simulations/result_code_saturne_1WT_LIGHT/single_time_flow_field.nc')

results = run_parameter_sweep(dat, param_config, n_samples=100, seed=3)

plt.scatter(results.ss_alpha, results.k_b, c=results.wind_diff.min('time'))
cbar = plt.colorbar()
plt.xlabel('ss_slpha')
plt.ylabel('k')
cbar.set_label('L2 Velocity Norm (m/s)')
plt.savefig('example_database')
plt.clf()

results.to_netcdf('results.nc')
