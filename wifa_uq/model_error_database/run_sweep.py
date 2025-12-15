import numpy as np
import xarray as xr
from typing import Dict, List
from windIO.yaml import load_yaml
import time
from pathlib import Path
from wifa import run_pywake, run_foxes
import json
import argparse


def set_nested_dict_value(d: dict, path: List[str], value: float) -> None:
    """Set value in nested dictionary using a path list."""
    current = d
    for key in path[:-1]:
        current = current[key]
    current[path[-1]] = value


def create_parameter_samples(
    param_config: Dict[str, dict], n_samples: int, seed: int = None
) -> Dict[str, np.ndarray]:
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

    samples = {}
    for param_path, config in param_config.items():
        min_val, max_val = config["range"]
        samples[param_path] = np.random.uniform(min_val, max_val, n_samples)

        # First sample is always the default (for baseline comparison)
        if "default" in config:
            samples[param_path][0] = config["default"]

    return samples


def run_parameter_sweep(
    run_func: callable,
    turb_rated_power,
    dat: dict,
    param_config: Dict[str, dict],
    reference_power: dict,
    reference_physical_inputs: dict,
    n_samples: int = 100,
    seed: int = None,
    output_dir="cases/default/sampling/",
    run_func_kwargs={},
) -> List[xr.Dataset]:
    """
    run the wifa api for a range of parameter samples
    compare reference power to engineering wake model power
    calculate the RMSE over the entire farm
    normalize based on rated power
    return the power errors for each sample as a netcdf

    Args:
        run_func: callable to run the simulation (run_foxes or run_pywake)
        turb_rated_power: rated power of a single turbine in the park (for normalizing power errors)
        dat: windIO system dat file
        param_config: specifying which parameters to sample from and ranges of values
        reference_power: xarray with the power values from the reference simulation
        reference_physical_inputs: xarray with physical inputs to the reference simulations
        n_samples: number of parameter samples
        seed: random seed for generating parameter samples
        run_func_kwargs: additional keyword arguments to pass to the run_func

    """

    samples = create_parameter_samples(param_config, n_samples, seed)
    n_flow_cases = reference_power.time.size
    sample_coords = np.arange(n_samples, dtype=np.float64)
    flow_case_coords = np.arange(n_flow_cases, dtype=np.float64)

    bias_cap = np.zeros((n_samples, n_flow_cases), dtype=np.float64)
    pw = np.zeros((n_samples, n_flow_cases), dtype=np.float64)
    ref = np.zeros((n_samples, n_flow_cases), dtype=np.float64)

    # Run the first sample with specific (default) samples

    for i in range(n_samples):
        # Update all parameters for this sample
        for param_path, param_samples in samples.items():
            # Convert string path to list of keys
            path = param_path.split(".")
            set_nested_dict_value(dat, path, param_samples[i])

        sample_dir = output_dir / f"sample_{i}"

        # Run simulation
        run_func(dat, output_dir=sample_dir, **run_func_kwargs)

        # Process results (in terms of power)
        pw_power = xr.open_dataset(sample_dir / "turbine_data.nc").power.values.T

        ref_power = reference_power.power.values
        # workaround for some cases
        if ref_power.shape == (pw_power.shape[1], pw_power.shape[0]):
            ref_power = ref_power.T

        model_err = pw_power - ref_power
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
        model_bias_cap = np.nanmean(model_err, axis=0) / turb_rated_power

        # Fill pre-allocated arrays for all samples
        bias_cap[i, :] = model_bias_cap
        # pywake or foxes power (farm average)
        pw[i, :] = np.nanmean(pw_power, axis=0) / turb_rated_power
        # reference power (farm average)
        ref[i, :] = np.nanmean(ref_power, axis=0) / turb_rated_power

    # Build parameter coordinates: one coordinate per swept parameter
    param_coords = {}
    for param_path, param_samples in samples.items():
        cfg = param_config.get(param_path, {})
        short_name = cfg.get("short_name", param_path.split(".")[-1])

        param_coords[short_name] = xr.DataArray(
            param_samples, dims=["sample"], coords={"sample": sample_coords}
        )

    # Build dataset directly from NumPy arrays (bias_cap, pw, ref)
    merged_data = xr.Dataset(
        data_vars={
            "model_bias_cap": (("sample", "flow_case"), bias_cap),
            "pw_power_cap": (("sample", "flow_case"), pw),
            "ref_power_cap": (("sample", "flow_case"), ref),
        },
        coords={
            "sample": sample_coords,
            "flow_case": flow_case_coords,
            **param_coords,
        },
    )

    # Store metadata
    merged_data.attrs["swept_params"] = [
        param_config[p]["short_name"] for p in param_config.keys()
    ]
    merged_data.attrs["param_paths"] = list(param_config.keys())
    merged_data.attrs["param_defaults"] = json.dumps(
        {
            param_config[p]["short_name"]: param_config[p].get("default")
            for p in param_config.keys()
        }
    )

    return merged_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "tool",
        help="The simulation tool, either 'foxes' or 'pywake'",
    )
    parser.add_argument(
        "-e",
        "--example",
        help="The sub folder name within examples/data",
        default="EDF_datasets",
    )
    parser.add_argument(
        "-c",
        "--case",
        help="The case name within the example folder",
        default="HR1",
    )
    parser.add_argument(
        "-o",
        "--out_name",
        help="The name of the samples output folder within the case folder",
        default="samples",
    )
    args = parser.parse_args()

    if args.tool == "foxes":
        run_func = run_foxes
        run_func_kwargs = {"verbosity": 0}
    elif args.tool == "pywake":
        run_func = run_pywake
        run_func_kwargs = {}
    else:
        raise ValueError(
            "Invalid simulation tool specified. Choose either 'foxes' or 'pywake'."
        )

    # Example usage:
    param_config = {
        "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": {
            "range": [0.01, 0.03],
            "default": 0.04,
            "short_name": "k_b",
        },
        "attributes.analysis.blockage_model.ss_alpha": {
            "range": [0.75, 1.25],
            "default": 0.875,
            "short_name": "ss_alpha",
        },
    }

    # navigating to a file containing metadata required to run wifa api
    case = args.case
    base_dir = Path(__file__).parent.parent.parent
    edf_dir = base_dir / "examples" / "data" / args.example
    case_dir = edf_dir / case
    meta_file = case_dir / "meta.yaml"
    meta = load_yaml(Path(meta_file))

    print(f"metadata for flow case: {meta}")
    dat = load_yaml(case_dir / f"{meta['system']}")
    reference_physical_inputs = xr.load_dataset(case_dir / f"{meta['ref_resource']}")
    turb_rated_power = meta["rated_power"]
    reference_power = xr.load_dataset(case_dir / f"{meta['ref_power']}")
    output_dir = case_dir / args.out_name

    start = time.time()
    print(f"Output directory to save results: {output_dir}")
    results = run_parameter_sweep(
        run_func,
        turb_rated_power,
        dat,
        param_config,
        reference_power,
        reference_physical_inputs,
        n_samples=10,
        seed=3,
        output_dir=output_dir,
        run_func_kwargs=run_func_kwargs,
    )
    print("Time taken for parameter sweep:", time.time() - start)
    results.to_netcdf("results.nc")
