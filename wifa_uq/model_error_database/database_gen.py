import xarray as xr
import numpy as np
import time
import re  # Import regular expression library
from pathlib import Path
from scipy.interpolate import interp1d
from windIO.yaml import load_yaml
from wifa import run_pywake, run_foxes
from wifa_uq.model_error_database.run_sweep import run_parameter_sweep
from wifa_uq.model_error_database.utils import (
    calc_boundary_area,
    blockage_metrics,
    farm_length_width,
)


class DatabaseGenerator:
    def __init__(
        self,
        nsamples: int,
        param_config: dict,
        system_yaml_path: Path,
        ref_power_path: Path,
        processed_resource_path: Path,
        wf_layout_path: Path,
        output_db_path: Path,
        model="pywake",
    ):
        """
        Initializes the DatabaseGenerator.

        Args:
            nsamples (int): Number of parameter samples to run.
            param_config (dict): Dictionary of parameters to sample.
            system_yaml_path (Path): Path to the windIO system YAML file.
            ref_power_path (Path): Path to the reference power NetCDF file (the "truth" data).
            processed_resource_path (Path): Path to the *preprocessed* physical inputs NetCDF.
            wf_layout_path (Path): Path to the wind_farm YAML (for layout utils).
            output_db_path (Path): Full path to save the final stacked NetCDF database.
            model (str, optional): Model to use. Defaults to "pywake".
        """
        self.nsamples = nsamples
        self.param_config = self._normalize_param_config(param_config)
        self.model = model
        self.system_yaml_path = Path(system_yaml_path)
        self.ref_power_path = Path(ref_power_path)
        self.processed_resource_path = Path(processed_resource_path)
        self.wf_layout_path = Path(wf_layout_path)
        self.output_db_path = Path(output_db_path)

        # Ensure output directory exists
        self.output_db_path.parent.mkdir(parents=True, exist_ok=True)

        # Validate input paths
        if not self.system_yaml_path.exists():
            raise FileNotFoundError(f"System YAML not found: {self.system_yaml_path}")
        if not self.ref_power_path.exists():
            raise FileNotFoundError(
                f"Reference power file not found: {self.ref_power_path}"
            )
        if not self.processed_resource_path.exists():
            raise FileNotFoundError(
                f"Processed resource file not found: {self.processed_resource_path}"
            )
        if not self.wf_layout_path.exists():
            raise FileNotFoundError(
                f"Wind farm layout file not found: {self.wf_layout_path}"
            )

    def _normalize_param_config(self, param_config: dict) -> dict:
        """
        Normalize param_config to full format.
        Handles both simple [min, max] and full {range, default, short_name} formats.
        """
        normalized = {}
        for path, config in param_config.items():
            if isinstance(config, list):
                # Simple format: [min, max]
                short_name = path.split(".")[-1]  # Use last part of path
                normalized[path] = {
                    "range": config,
                    "default": None,
                    "short_name": short_name,
                }
            elif isinstance(config, dict):
                # Full format
                if "short_name" not in config:
                    config["short_name"] = path.split(".")[-1]
                normalized[path] = config
            else:
                raise ValueError(f"Invalid param_config format for {path}")

        return normalized

    def _infer_rated_power(self, wf_dat: dict, system_dat: dict) -> float:
        """
        Attempts to find the rated power using multiple strategies, in order.
        """
        # The 'turbines' dict could be at the top level of wf_dat
        # or inside system_dat['wind_farm']
        turbine_data_sources = [
            wf_dat.get("turbines"),
            system_dat.get("wind_farm", {}).get("turbines"),
        ]

        for turbine_data in turbine_data_sources:
            if not turbine_data:
                continue

            # --- Strategy 1: Check for explicit 'rated_power' key ---
            try:
                power = turbine_data["performance"]["rated_power"]
                if power:
                    print(f"Found 'rated_power' key: {power} W")
                    return float(power)
            except (KeyError, TypeError):
                pass  # Not found, try next strategy

            # --- Strategy 2: Get max from 'power_curve' ---
            try:
                power_values = turbine_data["performance"]["power_curve"][
                    "power_values"
                ]
                if power_values:
                    power = max(power_values)
                    print(f"Found 'power_curve'. Max power: {power} W")
                    return float(power)
            except (KeyError, TypeError):
                pass  # Not found, try next strategy

            # --- Strategy 3: Parse the turbine 'name' ---
            try:
                name = turbine_data["name"]
                # Regex to find numbers (int or float) followed by "MW" (case-insensitive)
                match = re.search(r"(\d+(\.\d+)?)\s*MW", name, re.IGNORECASE)
                if match:
                    power_mw = float(match.group(1))
                    power_w = power_mw * 1_000_000
                    print(
                        f"Inferred rated power from turbine name '{name}': {power_w} W"
                    )
                    return power_w
            except (KeyError, TypeError):
                pass  # Not found, try next strategy

        # All strategies failed
        raise ValueError(
            "Could not find or infer 'rated_power'.\n"
            "Tried: \n"
            "  1. 'rated_power' key in '...turbines.performance'.\n"
            "  2. 'max(power_curve.power_values)' in '...turbines.performance'.\n"
            "  3. Parsing 'XMW' from '...turbines.name' field.\n"
            "Please add one of these to your windIO turbine file."
        )

    def generate_database(self) -> xr.Dataset:
        """
        Runs the full database generation pipeline for the single specified case.
        """
        starttime = time.time()

        # --- 1. Load all data from explicit paths ---
        print(f"Loading system config: {self.system_yaml_path.name}")
        dat = load_yaml(self.system_yaml_path)
        print(f"Loading reference power: {self.ref_power_path.name}")
        reference_power = xr.load_dataset(self.ref_power_path)
        print(f"Loading processed resource: {self.processed_resource_path.name}")
        reference_physical_inputs = xr.load_dataset(self.processed_resource_path)
        print(f"Loading wind farm layout: {self.wf_layout_path.name}")
        wf_dat = load_yaml(self.wf_layout_path)

        # --- 2. Infer metadata (replaces meta.yaml) ---

        # --- NEW ROBUST INFERENCE ---
        turb_rated_power = self._infer_rated_power(wf_dat, dat)

        # Get other metadata from the loaded files
        nt = len(wf_dat["layouts"][0]["coordinates"]["x"])
        hh = wf_dat["turbines"]["hub_height"]
        d = wf_dat["turbines"]["rotor_diameter"]
        case_name = wf_dat.get("name", self.system_yaml_path.stem)

        print(
            f"Case: {case_name}, {nt} turbines, Rated Power: {turb_rated_power / 1e6:.1f} MW, Hub Height: {hh} m"
        )

        # --- 3. Run parameter sweep ---
        output_dir = self.output_db_path.parent / "samples"
        output_dir.mkdir(exist_ok=True)

        if self.model == "pywake":
            result = run_parameter_sweep(
                run_pywake,
                turb_rated_power,
                dat,
                self.param_config,
                reference_power,
                None,  # reference_physical_inputs is not actually used by run_pywake_sweep
                n_samples=self.nsamples,
                seed=1,
                output_dir=output_dir,
            )
        elif self.model == "foxes":
            result = run_parameter_sweep(
                run_foxes,
                turb_rated_power,
                dat,
                self.param_config,
                reference_power,
                None,  # reference_physical_inputs is not actually used by run_pywake_sweep
                n_samples=self.nsamples,
                seed=1,
                output_dir=output_dir,
                run_func_kwargs={"verbosity": 0},
            )
        else:
            raise NotImplementedError(f"Model '{self.model}' not implemented yet.")

        print("Parameter sweep complete. Processing physical inputs...")

        # --- 4. Process and add physical inputs ---
        phys_inputs = reference_physical_inputs.copy()

        # Rename 'time' to 'flow_case' to match run_pywake_sweep output
        if "time" in phys_inputs.dims:
            # Check if flow_case dimension already exists from run_pywake_sweep
            n_flow_cases = len(result.flow_case)
            if len(phys_inputs.time) != n_flow_cases:
                raise ValueError(
                    f"Mismatch in 'time' dimension of resource ({len(phys_inputs.time)}) "
                    f"and 'flow_case' dimension of simulation ({n_flow_cases})."
                )

            # Use the coordinates from the simulation result, but data from resource
            phys_inputs = phys_inputs.rename({"time": "flow_case"})
            phys_inputs = phys_inputs.assign_coords(flow_case=result.flow_case)

        # Interpolate to hub height if 'height' dimension exists
        if "height" in phys_inputs.dims:
            print(f"Interpolating physical inputs to hub height ({hh} m)...")
            heights = phys_inputs.height.values

            interp_ds = xr.Dataset(coords=phys_inputs.coords)
            for var, da in phys_inputs.data_vars.items():
                if "height" in da.dims:
                    # Create 1D interpolation function for each flow case
                    f_interp = interp1d(
                        heights,
                        da,
                        axis=da.dims.index("height"),
                        fill_value="extrapolate",
                    )
                    # Create new DataArray with interpolated values
                    new_dims = [dim for dim in da.dims if dim != "height"]
                    interp_ds[var] = (new_dims, f_interp(hh))
                else:
                    # Keep non-height-dependent variables
                    interp_ds[var] = da

            phys_inputs = interp_ds
        else:
            print("Physical inputs have no 'height' dim, assuming hub-height or 0D.")

        # Add all physical inputs to the results dataset
        for var in phys_inputs.data_vars:
            if var in result.coords:
                continue  # Don't overwrite coords
            result[var] = phys_inputs[var]

        # --- 5. Add farm-level features ---
        print("Adding farm-level features...")
        result = result.expand_dims(dim={"wind_farm": [case_name]})
        result["turb_rated_power"] = xr.DataArray(
            [turb_rated_power], dims=["wind_farm"]
        )
        result["nt"] = xr.DataArray([nt], dims=["wind_farm"])

        x = wf_dat["layouts"][0]["coordinates"]["x"]
        y = wf_dat["layouts"][0]["coordinates"]["y"]
        density = calc_boundary_area(x, y, show=False) / nt
        result["farm_density"] = xr.DataArray([density], dims=["wind_farm"])

        # --- 6. Flatten and add layout features ---
        print("Stacking dataset...")
        stacked = result.stack(case_index=("wind_farm", "flow_case"))
        stacked = stacked.dropna(dim="case_index", how="all", subset=["model_bias_cap"])
        stacked = stacked.reset_index("case_index")  # Makes case_index a variable

        print("Adding layout-dependent features (Blockage, etc.)...")
        BR_farms, BD_farms, lengths, widths = [], [], [], []

        xy = np.column_stack((x, y))
        wind_dirs = stacked.wind_direction.values  # Get all wind directions at once

        for wd in wind_dirs:
            # L_inf_factor=20.0, grid_res=151
            BR, BD, BR_farm, BD_farm = blockage_metrics(
                xy, wd, d, grid_res=51, plot=False
            )
            length, width = farm_length_width(x, y, wd, d, plot=False)

            BR_farms.append(BR_farm)
            BD_farms.append(BD_farm)
            lengths.append(length)
            widths.append(width)

        stacked["Blockage_Ratio"] = xr.DataArray(BR_farms, dims=["case_index"])
        stacked["Blocking_Distance"] = xr.DataArray(BD_farms, dims=["case_index"])
        stacked["Farm_Length"] = xr.DataArray(lengths, dims=["case_index"])
        stacked["Farm_Width"] = xr.DataArray(widths, dims=["case_index"])

        # --- 7. Save the final database ---
        print(f"Saving final database to: {self.output_db_path}")
        self.output_db_path.parent.mkdir(parents=True, exist_ok=True)
        stacked.to_netcdf(self.output_db_path)

        tottime = round(time.time() - starttime, 3)
        print(f"Database generation complete. Total time: {tottime} seconds")

        return stacked


# This check prevents this code from running when imported
if __name__ == "__main__":
    # You can add a simple test harness here for debugging this file directly
    print("This script is a module and is intended to be imported by 'workflow.py'.")
    print("To run a workflow, please use 'run.py' in the root directory.")
