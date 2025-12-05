# ./wifa_uq/preprocessing/preprocessing.py

# %%
import xarray as xr
import numpy as np
import shutil
from windIO.yaml import load_yaml
from matplotlib.ticker import ScalarFormatter
from wifa.wayve_api import ci_fitting
import matplotlib.pyplot as plt
from pathlib import Path

# --- Helper Functions (pasted from our previous steps) ---


def _calculate_abl_from_velocity(ds: xr.Dataset) -> xr.DataArray:
    """
    Calculates ABL height from the velocity profile.

    Definition: Height where wind speed first reaches 99% of the
    maximum wind speed in the profile (freestream or LLJ peak).
    """
    print("    Calculating ABL_height from velocity profile (99% of max)...")

    # Get the maximum wind speed for each time step (flow_case)
    max_wind_speed = ds["wind_speed"].max(dim="height")

    # Calculate the 99% threshold
    threshold = 0.99 * max_wind_speed

    # Find all heights where wind speed is >= 99% of max
    above_threshold = ds["wind_speed"] >= threshold

    # Create a DataArray of heights, broadcasted to match the wind_speed array
    height_da = ds["height"] * xr.ones_like(ds["wind_speed"])

    # Where wind speed is below threshold, set height to infinity
    valid_heights = height_da.where(above_threshold, np.inf)

    # Find the minimum height (the first to cross the threshold) for each time step
    abl_height = valid_heights.min(dim="height")

    # Handle cases with no valid height (all inf) by setting to max height
    abl_height = abl_height.where(abl_height != np.inf, ds["height"].max())

    return abl_height


def _calculate_veer(ds: xr.Dataset) -> xr.DataArray:
    """Calculates wind veer (d(WD)/dz) from the wind direction profile."""
    print("    Calculating wind veer...")

    # Ensure wind_direction and height exist
    if "wind_direction" not in ds or "height" not in ds:
        raise ValueError("Missing 'wind_direction' or 'height' for veer calculation.")

    # 1. Convert degrees to radians
    wd_rad = np.deg2rad(ds["wind_direction"])

    # 2. Unwrap angles along the height dimension to handle 0/360 crossing
    unwrapped_rad = xr.apply_ufunc(
        np.unwrap,
        wd_rad,
        input_core_dims=[["height"]],
        output_core_dims=[["height"]],
        dask="parallelized",
        output_dtypes=[wd_rad.dtype],
    )

    # 3. Calculate the gradient (d(rad)/dz)
    height_coords = ds["height"].values

    veer_rad_per_m = xr.apply_ufunc(
        np.gradient,
        unwrapped_rad,
        height_coords,
        input_core_dims=[["height"], ["height"]],
        output_core_dims=[["height"]],
        kwargs={"axis": -1},  # Tell numpy.gradient to use the last (core) dimension
        dask="parallelized",
        output_dtypes=[unwrapped_rad.dtype],
    )

    # 4. Convert back to degrees/meter for interpretability
    veer_deg_per_m = np.rad2deg(veer_rad_per_m)

    veer_deg_per_m.attrs["long_name"] = "Wind Veer"
    veer_deg_per_m.attrs["units"] = "deg/m"

    return veer_deg_per_m


# --- Main Class ---


class PreprocessingInputs:
    def __init__(
        self, ref_resource_path: Path, output_path: Path, steps: list[str] = None
    ):
        """
        Initializes the preprocessor for a SINGLE reference resource file.

        Args:
            ref_resource_path (Path): Path to the original NetCDF resource file.
            output_path (Path): Path to write the processed NetCDF file.
            steps (list[str], optional): Steps to run: 'update_heights', 'recalculate_params'.
        """
        self.ref_resource_path = Path(ref_resource_path)
        self.output_path = Path(output_path)
        self.steps = steps if steps is not None else []

        if not self.ref_resource_path.exists():
            raise FileNotFoundError(
                f"Input resource file not found: {self.ref_resource_path}"
            )

        print(f"Preprocessor initialized for {self.ref_resource_path.name}.")
        if self.steps:
            print(f"Applying steps: {self.steps}")
        else:
            print("No preprocessing steps will be applied.")

    def run_pipeline(self) -> Path:
        """
        Runs the configured preprocessing pipeline.
        Returns the path to the processed file.
        """
        if not self.steps:
            print(
                f"No steps to run. Copying {self.ref_resource_path.name} to {self.output_path.name}"
            )
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.ref_resource_path, self.output_path)
            return self.output_path

        current_input_path = self.ref_resource_path

        if "update_heights" in self.steps:
            print(
                "  WARNING: Skipping 'update_heights'. This step is not fully supported "
                "in this refactor as it requires a target height grid."
            )

        if "recalculate_params" in self.steps:
            print("  Running 'recalculate_params'...")
            self._recalculate_params(current_input_path, self.output_path)

        if not self.output_path.exists():
            shutil.copy(self.ref_resource_path, self.output_path)

        print(f"Preprocessing complete. Output at: {self.output_path}")
        return self.output_path

    def _recalculate_params(self, input_path: Path, output_path: Path):
        """
        Recalculate parameters from vertical profiles.
        Reads from input_path and writes to output_path.
        """
        try:
            with xr.load_dataset(input_path) as ref_inputs:
                ref_inputs_modified = ref_inputs.copy()

                # --- Update TI ---
                if "k" in ref_inputs and "wind_speed" in ref_inputs:
                    ref_inputs_modified["turbulence_intensity"] = (
                        np.sqrt((2 / 3) * ref_inputs["k"]) / ref_inputs["wind_speed"]
                    )
                else:
                    print(
                        "    Skipping TI recalculation: 'k' or 'wind_speed' not found."
                    )

                # --- Calculate Wind Veer ---
                try:
                    ref_inputs_modified["wind_veer"] = _calculate_veer(
                        ref_inputs_modified
                    )
                except Exception as e:
                    print(f"    Skipping wind veer calculation: {e}")

                # --- Try for velocity based ABL ---
                if (
                    "wind_speed" in ref_inputs_modified
                    and "height" in ref_inputs_modified
                ):
                    try:
                        ref_inputs_modified["ABL_height"] = (
                            _calculate_abl_from_velocity(ref_inputs_modified)
                        )
                    except Exception as e:
                        print(f"    Velocity-based ABL calculation failed: {e}")

                # --- Run temperature-based calculations (Lapse Rate, etc.) ---
                print("    Checking for temperature-based parameters...")
                if (
                    "height" in ref_inputs_modified
                    and "potential_temperature" in ref_inputs_modified
                ):
                    non_height_dims = [
                        d
                        for d in ref_inputs_modified["potential_temperature"].dims
                        if d != "height"
                    ]
                    non_height_shape = [
                        s
                        for d, s in ref_inputs_modified[
                            "potential_temperature"
                        ].sizes.items()
                        if d != "height"
                    ]

                    if "LMO" not in ref_inputs_modified:
                        print(
                            "    'LMO' not found. Assuming neutral stability (LMO=1e10)."
                        )
                        ref_inputs_modified["LMO"] = xr.DataArray(
                            np.full(non_height_shape, 1e10), dims=non_height_dims
                        )

                    if "ABL_height" in ref_inputs_modified:
                        abl_guess = ref_inputs_modified["ABL_height"]
                    else:
                        print(
                            "    'ABL_height' (initial guess) not found. Assuming 1000m."
                        )
                        abl_guess = xr.DataArray(
                            np.full(non_height_shape, 1000.0), dims=non_height_dims
                        )

                    print(
                        "    Running 'ci_fitting' for thermal parameters (lapse_rate, etc.)..."
                    )
                    H_temp, dthdz, dth, inv_thickness, unknown1, unknown2 = (
                        xr.apply_ufunc(
                            ci_fitting,  # wifa.wayve_api.ci_fitting
                            ref_inputs_modified["height"],
                            ref_inputs_modified["potential_temperature"],
                            ref_inputs_modified["LMO"],  # Use LMO (or default)
                            abl_guess,  # Use abl_guess
                            input_core_dims=[["height"], ["height"], [], []],
                            output_core_dims=[[], [], [], [], [], []],  # <-- UPDATED
                            vectorize=True,
                            dask="allowed",
                            output_dtypes=[
                                float,
                                float,
                                float,
                                float,
                                float,
                                float,
                            ],  # <-- UPDATED
                        )
                    )

                    # ALWAYS save the thermal properties
                    ref_inputs_modified["lapse_rate"] = dthdz
                    ref_inputs_modified["capping_inversion_strength"] = dth
                    ref_inputs_modified["capping_inversion_thickness"] = inv_thickness

                    if "ABL_height" not in ref_inputs_modified:
                        print("    Using temperature-based ABL_height as fallback.")
                        ref_inputs_modified["ABL_height"] = H_temp
                    # ---------------------------------
                else:
                    print(
                        "    Skipping temperature-based calculations: 'height' or 'potential_temperature' not found."
                    )

                if "lapse_rate" not in ref_inputs_modified:
                    print("    WARNING: Could not calculate lapse_rate.")

                output_path.parent.mkdir(parents=True, exist_ok=True)
                ref_inputs_modified.to_netcdf(output_path)

        except Exception as e:
            # This outer try/except block will catch the error from ci_fitting
            print(
                f"    FATAL Error during 'recalculate_params': {type(e).__name__}: {e}"
            )
            if not output_path.exists() and input_path != output_path:
                shutil.copy(input_path, output_path)
            raise e  # Re-raise it to stop the main workflow

    # ... (rest of the class, e.g., compare_physical_inputs, batch_update_params) ...
    # ... (the __main__ block at the end) ...

    def compare_physical_inputs(
        self,
        case_name: str,
        TI_height_idx: int,
        base_dir="EDF_datasets",
        updated_file="updated_physical_inputs.nc",
    ):
        """
        Compare original and modified physical inputs for a given case.

        Parameters:
        - case_name: name of the case (used for path construction)
        - TI_height_idx: index of height at which to extract TI
        - base_dir: base directory containing cases
        - updated_file: name of the updated netcdf file (relative to case directory)
        """
        meta_file = f"{self.base_dir}/{case_name}/meta.yaml"
        meta = load_yaml(Path(meta_file))

        # Load datasets
        original_path = f"{self.base_dir}/{case_name}/{meta['ref_resource']}"
        updated_path = f"{self.base_dir}/{case_name}/{updated_file}"

        ds_orig = xr.load_dataset(original_path)
        ds_mod = xr.load_dataset(updated_path)

        fig, axs = plt.subplots(5, 1, figsize=(8, 15), sharex=True)

        # Capping inversion strength [K]
        axs[0].plot(
            ds_orig.capping_inversion_strength.values, label="Old", linestyle="--"
        )
        axs[0].plot(ds_mod.capping_inversion_strength.values, label="New")
        axs[0].set_title("Capping Inversion Strength")
        axs[0].set_ylabel(r"$\Delta \theta$ [K]")
        axs[0].legend()

        # Capping inversion thickness [m]
        axs[1].plot(
            ds_orig.capping_inversion_thickness.values, label="Old", linestyle="--"
        )
        axs[1].plot(ds_mod.capping_inversion_thickness.values, label="New")
        axs[1].set_title("Capping Inversion Thickness")
        axs[1].set_ylabel("Thickness [m]")
        axs[1].legend()

        # ABL Height [m]
        axs[2].plot(ds_orig.ABL_height.values, label="Old", linestyle="--")
        axs[2].plot(ds_mod.ABL_height.values, label="New")
        axs[2].set_title("ABL Height")
        axs[2].set_ylabel("Height [m]")
        axs[2].legend()

        # Lapse Rate [K/m]
        axs[3].plot(ds_orig.lapse_rate.values, label="Old", linestyle="--")
        axs[3].plot(ds_mod.lapse_rate.values, label="New")
        axs[3].set_title("Lapse Rate")
        axs[3].set_ylabel(r"$\partial \theta / \partial z$ [K/m]")
        axs[3].legend()

        formatter = ScalarFormatter(useMathText=True)
        formatter.set_powerlimits(
            (-2, 2)
        )  # scientific notation outside range 10^-2 to 10^2
        axs[3].yaxis.set_major_formatter(formatter)

        # Turbulence Intensity (unitless)
        axs[4].plot(
            ds_orig.turbulence_intensity.values[:, TI_height_idx],
            label=f"Old (height idx {TI_height_idx})",
            linestyle="--",
        )
        axs[4].plot(
            ds_mod.turbulence_intensity.values[:, TI_height_idx],
            label=f"New (height idx {TI_height_idx})",
        )
        axs[4].set_title("Turbulence Intensity")
        axs[4].set_ylabel("TI [-]")
        axs[4].set_xlabel("Time index")
        axs[4].legend()

        plt.tight_layout()
        plt.show()

    def batch_update_params(self):
        meta_file = self.base_dir / self.case_names[0] / "meta.yaml"
        meta = load_yaml(meta_file)
        ref_inputs = xr.load_dataset(
            self.base_dir / self.case_names[0] / meta["ref_resource"]
        )
        new_heights = ref_inputs.height

        for case in self.case_names:
            self.update_heights(case, new_heights)
            self.recalculate_params(case)


if __name__ == "__main__":
    case_names = [
        "HR1",
        "HR2",
        "HR3",
        "NYSTED1",
        "NYSTED2",
        "VirtWF_ABL_IEA10",
        "VirtWF_ABL_IEA15_ali_DX5_DY5",
        "VirtWF_ABL_IEA15_stag_DX5_DY5",  # this has 998 heights originally, interpolating later
        "VirtWF_ABL_IEA15_stag_DX5_DY7p5",  # this has 998 heights originally, interpolating later
        "VirtWF_ABL_IEA15_stag_DX7p5_DY5",  # this has 998 heights originally, interpolating later
        "VirtWF_ABL_IEA22",
    ]
    base_dir = "EDF_datasets"
    preprocessor = PreprocessingInputs(base_dir=base_dir, case_names=case_names)

    # plotting for a single case
    preprocessor.compare_physical_inputs(case_name=case_names[5], TI_height_idx=100)
