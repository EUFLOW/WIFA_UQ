# %%
# %%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
from windIO.yaml import load_yaml 
from run_pywake_sweep import *
from pathlib import Path
from scipy.interpolate import interp1d
import time
from utils import calc_boundary_area
from utils import blockage_metrics,farm_length_width


class DatabaseGenerator:
    def __init__(self,nsamples,param_config,base_folder,case_names,model="pywake"):
        self.nsamples=nsamples
        self.param_config=param_config
        self.model=model
        self.base_folder=base_folder
        self.case_names=case_names

    def generate_database(self):
        all_case_results = []
        starttime = time.time()

        for case in self.case_names:
            case_folder = self.base_folder / case

            # --- Load metadata and reference datasets ---
            meta = load_yaml(case_folder / "meta.yaml")
            dat = load_yaml(case_folder / meta["system"])
            reference_power = xr.load_dataset(case_folder / meta["ref_power"])
            turb_rated_power = meta["rated_power"]
            nt = meta["nt"]
            output_dir = case_folder / "pywake_samples"
            reference_physical_inputs = xr.load_dataset(case_folder / "updated_physical_inputs.nc")

            # --- Run parameter sweep ---
            if self.model == "pywake":
                result = run_parameter_sweep(
                    turb_rated_power,
                    dat,
                    self.param_config,
                    reference_power,
                    reference_physical_inputs,
                    n_samples=self.nsamples,
                    seed=1,
                    output_dir=output_dir,
                )
            else:
                raise NotImplementedError(f"Model '{self.model}' not implemented yet.")

            # adding physical inputs to dataset (optionally interpolating to hub height)
            hh=dat['wind_farm']['turbines']['hub_height'] 

            z0=reference_physical_inputs.z0.values
            LMO=reference_physical_inputs.LMO.values
            ABL_height=reference_physical_inputs.ABL_height.values
            capping_inversion_strength=reference_physical_inputs.capping_inversion_strength.values
            capping_inversion_thickness=reference_physical_inputs.capping_inversion_thickness.values
            ws=reference_physical_inputs.wind_speed.values
            wd=reference_physical_inputs.wind_direction.values
            ti=reference_physical_inputs.turbulence_intensity.values
            theta=reference_physical_inputs.potential_temperature.values
            epsilon=reference_physical_inputs.epsilon.values
            k=reference_physical_inputs.k.values
            lapse_rate=reference_physical_inputs.lapse_rate.values

            x=dat['wind_farm']['layouts'][0]['coordinates']['x']
            y=dat['wind_farm']['layouts'][0]['coordinates']['y']
            density=calc_boundary_area(x, y,show=False)/(nt) # wf area in m2 divided by nt

            # store the reference physical inputs as hub height values in another dataset
            if "height" in reference_physical_inputs.dims:
                print("interpolating to hub height")
                heights=reference_physical_inputs.height.values
                ws = interp1d(heights, ws, axis=1, fill_value="extrapolate")(hh)
                wd = interp1d(heights, wd, axis=1, fill_value="extrapolate")(hh)
                ti = np.maximum(interp1d(heights, ti, axis=1, fill_value="extrapolate")(hh),2e-2,)
                theta=interp1d(heights, theta, axis=1, fill_value="extrapolate")(hh)
                epsilon=interp1d(heights, epsilon, axis=1, fill_value="extrapolate")(hh)
                k=interp1d(heights, k, axis=1, fill_value="extrapolate")(hh)
                print('data interpolated')

            else:
                print('no interpolation needed')

            # Add the hub height inputs back into the results dataset
            result["wind_speed"] = xr.DataArray(ws, dims=["flow_case"])
            result["wind_direction"] = xr.DataArray(wd, dims=["flow_case"])
            result["turbulence_intensity"] = xr.DataArray(ti, dims=["flow_case"])
            result["potential_temperature"] = xr.DataArray(theta, dims=["flow_case"])
            result["epsilon"] = xr.DataArray(epsilon, dims=["flow_case"])
            result["k"] = xr.DataArray(k, dims=["flow_case"])
            result["lapse_rate"] = xr.DataArray(lapse_rate, dims=["flow_case"])
            result["z0"] = xr.DataArray(z0, dims=["flow_case"])
            result["LMO"] = xr.DataArray(LMO, dims=["flow_case"])
            result["ABL_height"] = xr.DataArray(ABL_height, dims=["flow_case"])
            result["capping_inversion_strength"] = xr.DataArray(capping_inversion_strength, dims=["flow_case"])
            result["capping_inversion_thickness"] = xr.DataArray(capping_inversion_thickness, dims=["flow_case"])

            # adding some WF specific variables
            # these variables are a function of wind farm
            result = result.expand_dims(dim={"wind_farm": [case]})

            result["turb_rated_power"] = xr.DataArray(
            [turb_rated_power], 
            dims=["wind_farm"])

            result["nt"] = xr.DataArray(
            [nt], 
            dims=["wind_farm"])

            wf_dat=load_yaml(Path(f"{self.base_folder}/{case}/{meta['wf_dat']}"))
            x=wf_dat['layouts'][0]['coordinates']['x']
            y=wf_dat['layouts'][0]['coordinates']['y']
            density=calc_boundary_area(x, y,show=False)/(nt) # wf area in m2 divided by nt

            result["farm_density"] = xr.DataArray(
            [density], 
            dims=["wind_farm"])

            all_case_results.append(result)

        # %%
        # combining datasets from different wind farms, whilst including an identifier for the wind farm simulation used
        # since there are different numbers of flow cases for each wind farm, and the dimensions need to be the same size, we will have some nan values
        # therefore the different flow cases for each wind farm are also stacked together if that is preferred

        case_datasets=[]
        for case, result in zip(self.case_names, all_case_results):
            case_datasets.append(result)
        combined = xr.concat(case_datasets, dim='wind_farm')

        # Flattenning case and index into one dimension 
        stacked = combined.stack(case_index=('wind_farm', 'flow_case'))  # shape: [sample, case_index]
        stacked = stacked.dropna(dim='case_index', subset=['model_bias_cap'])
        stacked = stacked.reset_index('case_index')
        # stacked.to_netcdf('results_stacked_hh.nc')
        # combined.to_netcdf('results_combined.nc')

        tottime=round(time.time() - starttime,3)
        print(f"Total time taken: {tottime} seconds")

        print("Adding Layout Features")

        #%% Post Processing
        # Adding some layout specific features
        case_i=stacked.case_index.values

        BR_farms=[]
        BD_farms=[]
        lengths=[]
        widths=[]
        for i in case_i:
            wind_farm=stacked.wind_farm.values[i]
            meta_file=f"{self.base_folder}/{wind_farm}/meta.yaml"
            meta=load_yaml(Path(meta_file))
            dat = load_yaml(Path(f"{self.base_folder}/{wind_farm}/{meta['system']}"))
        
            x=dat['wind_farm']['layouts'][0]['coordinates']['x']
            y=dat['wind_farm']['layouts'][0]['coordinates']['y']
            d=dat['wind_farm']['turbines']['rotor_diameter']

            xy = np.column_stack((x, y))

            wind_dir=stacked.wind_direction.values[i]

            BR, BD, BR_farm, BD_farm = blockage_metrics(xy, wind_dir, d)
            length, width=farm_length_width(x,y,wind_dir,d)

            BR_farms.append(BR_farm)
            BD_farms.append(BD_farm)
            lengths.append(length)
            widths.append(width)

        stacked["Blockage_Ratio"] = xr.DataArray(BR_farms, dims=["case_index"])
        stacked["Blocking_Distance"] = xr.DataArray(BD_farms, dims=["case_index"])
        stacked["Farm_Length"] = xr.DataArray(lengths, dims=["case_index"])
        stacked["Farm_Width"] = xr.DataArray(widths, dims=["case_index"])

        stacked.to_netcdf('results_stacked_hh.nc')

        return stacked


if __name__ == "__main__":
    base_dir = Path("EDF_datasets")


    # Identifiers for the different wind farm simulations on windlab
    case_names=[
        "HR1",   
        "HR2",     
        "HR3",
        "NYSTED1",   
        "NYSTED2",
        "VirtWF_ABL_IEA10", 
        "VirtWF_ABL_IEA15_ali_DX5_DY5",   
        "VirtWF_ABL_IEA15_stag_DX5_DY5",    
        "VirtWF_ABL_IEA15_stag_DX5_DY7p5",  
        "VirtWF_ABL_IEA15_stag_DX7p5_DY5",  
        "VirtWF_ABL_IEA22"
    ]

    # defining ranges for the parameter samples
    param_config = {
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": (0.01, 0.07),
            "attributes.analysis.blockage_model.ss_alpha": (0.75, 1.0)
        }
    
    nsamples=10  # First sample by definition will be default, then 100 additional random samples
    db_generator = DatabaseGenerator(nsamples, param_config, base_dir, case_names, model="pywake")
    db_generator.generate_database()


