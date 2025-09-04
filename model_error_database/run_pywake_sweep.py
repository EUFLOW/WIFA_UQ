import numpy as np
import xarray as xr
from typing import Dict, List, Union, Tuple
from windIO.yaml import load_yaml 
import time
from pathlib import Path
from wifa.pywake_api import run_pywake
import os


def set_nested_dict_value(d: dict, path: List[str], value: float) -> None:
    """Set value in nested dictionary using a path list."""
    current = d
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value
  
def create_parameter_samples(param_config: Dict[str, Tuple[float, float]], n_samples: int, 
                             seed: int = None, manual_first_sample: Dict[str, float] = None) -> Dict[str, np.ndarray]:
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

    samples={
        param: np.random.uniform(min_val, max_val, n_samples)
        for param, (min_val, max_val) in param_config.items()
    }
    if manual_first_sample:
        # overwrite the first element with given defaults
        for param, val in manual_first_sample.items():
            samples[param][0] = val

    return samples

def run_parameter_sweep(turb_rated_power,dat: dict, param_config: Dict[str, Tuple[float, float]], reference_power: dict, 
              reference_physical_inputs: dict,n_samples: int = 100, seed: int = None, output_dir='cases/default/pywake_sampling/') -> List[xr.Dataset]:
    """
    run the pywake api for a range of ṕarameter samples
    compare reference power to pywake power
    calculate the RMSE over the entire farm
    normalize based on rated power
    return the power errors for each sample as a netcdf

    Args:
        turb_rated_power: rated power of a single turbine in the park (for normalizing power errors)
        dat: windIO system dat file
        param_config: specifying which parameters to sample from and ranges of values
        reference_power: xarray with the power values from the reference simulation
        reference_physical_inputs: xarray with physical inputs to the reference simulations 
        n_samples: number of parameter samples
        seed: random seed for generating parameter samples 

    """

    # Generate samples for all parameters
    # Specifying the first sample (for comparison to default parameters)
    default = {
    "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": 0.04,
    "attributes.analysis.blockage_model.ss_alpha": 0.875
    }

    samples = create_parameter_samples(param_config, n_samples, seed, manual_first_sample=default)
    n_flow_cases=reference_power.power.shape[1]

    bias_cap=np.zeros((n_samples, n_flow_cases), dtype=np.float64)
    pw=np.zeros((n_samples, n_flow_cases), dtype=np.float64)
    ref=np.zeros((n_samples, n_flow_cases), dtype=np.float64)

    # Run the first sample with specific (default) samples

    #

    for i in range(n_samples):  
        # Update all parameters for this sample
        for param_path, param_samples in samples.items():
            # Convert string path to list of keys
            path = param_path.split('.')
            set_nested_dict_value(dat, path, param_samples[i])
        
        sample_dir = f'{output_dir}/sample_{i}'
        
        # Run simulation
        run_pywake(dat, output_dir=sample_dir)
  
        # Process results (in terms of power)
        pw_power = xr.open_dataset("results/turbine_data.nc").power.values

        ref_power = reference_power.power.values  
            # workaround for some cases
        # if ref_power.shape == (pw_power.shape[1], pw_power.shape[0]):
        #     ref_power = ref_power.T

        model_err=pw_power-ref_power
        # # in case there are nan reference values (relevant for scada)
        # masked_model_err = np.ma.masked_array(model_err, mask=np.isnan(ref_power))

        """.
        In the context of bias-correction, dividing by the model value makes more sense
        If bias = 10%, model overpredicts by 10% (of the model value)
        Therefore, a multiplication by 0.9 would correct the bias.

        In the context of forecasting accuracy... it seems more intuitive to divide by the reference value?
        If bias = 10%, model output is 10% higher than the actual value
        """

        # calculating farm level bias
        model_bias_cap=np.nanmean(model_err, axis=0)/turb_rated_power

        # Fill pre-allocated arrays for all samples
        bias_cap[i, :] = model_bias_cap
        # pywake power (farm average)
        pw[i, :] = np.nanmean(pw_power, axis=0)/turb_rated_power
        # reference power (farm average)
        ref[i, :] = np.nanmean(ref_power, axis=0)/turb_rated_power

    # Convert to xarray.DataArray
    flow_case_coords = np.arange(n_flow_cases, dtype=np.float64)
    sample_coords = np.arange(n_samples, dtype=np.float64)

    bias_cap = xr.DataArray(
        bias_cap,
        dims=['sample', 'flow_case'],
        coords={'sample': sample_coords, 'flow_case': flow_case_coords}
    )
    pw = xr.DataArray(
        pw,
        dims=['sample', 'flow_case'],
        coords={'sample': sample_coords, 'flow_case': flow_case_coords}
    )
    ref = xr.DataArray(
        ref,
        dims=['sample', 'flow_case'],
        coords={'sample': sample_coords, 'flow_case': flow_case_coords}
    )

    # Add parameter values to dataset
    merged_data = xr.Dataset(
        data_vars={'model_bias_cap': bias_cap, 'pw_power_cap': pw, 'ref_power_cap': ref},
        coords={
            param_path.split('.')[-1]: xr.DataArray(
                param_samples,
                dims=['sample'],
                coords={'sample': sample_coords}
            )
            for param_path, param_samples in samples.items()
        }
    )
    return merged_data

if __name__ == "__main__":

    # Example usage:
    param_config = {
        "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": (0.01, 0.3),
        "attributes.analysis.blockage_model.ss_alpha": (0.75, 1.25)
    }

    # navigating to a file containing metadata required to run pywake api
    case="HR1"
    meta_file=f"EDF_datasets/{case}/meta.yaml"
    meta=load_yaml(Path(meta_file))

    print(f"metadata for flow case: {meta}")
    
    dat = load_yaml(Path(f"EDF_datasets/{case}/{meta['system']}"))
    reference_physical_inputs = xr.load_dataset(f"EDF_datasets/{case}/{meta['ref_resource']}")
    turb_rated_power=meta['rated_power']
    reference_power=xr.load_dataset(f"EDF_datasets/{case}/{meta['ref_power']}")
    output_dir=f"EDF_datasets/{case}/pywake_samples"

    start = time.time()
    print(f"Output directory to save results: {output_dir}")
    results = run_parameter_sweep(turb_rated_power,dat,param_config,reference_power,reference_physical_inputs, n_samples=10, seed=3,output_dir=Path(output_dir))
    print("Time taken for parameter sweep:", time.time() - start)
    results.to_netcdf('results.nc')
