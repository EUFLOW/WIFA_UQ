import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xarray import Dataset
from shutil import rmtree
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.model_selection import train_test_split
from pathlib import Path
from xarray import open_dataset
import foxes
import foxes.variables as FV
import foxes.constants as FC

error_method = "rmse"
datasets_dir = Path("../model_error_database/EDF_datasets/")
cases_dirs = sorted(list(datasets_dir.glob("*")))
results_dir = Path("results") / error_method
resfig_dir = results_dir / "figures"
calfig_dir = resfig_dir / "calibration"
results_file = results_dir / "calibration_results.nc"
k_values = np.linspace(0.01, 0.2, 191)
rotor = "level9"
verbosity = 4

engine = "process"
chunksize_s = 200
chunksize_p = 4000
n_procs = None

print("Scanning k values:\n", k_values, "\n")

if resfig_dir.is_dir():
    rmtree(resfig_dir)
calfig_dir.mkdir(exist_ok=True, parents=True)

def calc_error(cfd_results, pop_results, capacity):
    P_cfd = cfd_results["power"].values.T
    P_fxs = pop_results[FV.P].values 

    if error_method == "rmse":
        mse = np.sqrt(np.sum((P_cfd[None] - P_fxs) ** 2, axis=2))
        return mse / capacity
    
    elif error_method == "bias":
        bias = np.abs(np.sum(P_cfd[None] - P_fxs, axis=2))
        return bias / capacity

all_results_coords = {}
all_results_data = {
    "case_name": [("case",), []],
    "time_index": [("case",), []],
    "n_turbines": [("case",), []],
    "hub_height": [("case",), []],
    "rotor_diameter": [("case",), []],
    "capacity": [("case",), []],
    "best_k": [("case",), []],
    "best_k_farm_P": [("case",), []],
    "best_k_error": [("case",), []],
}

def write_figure_calib(resfig_dir, k_values, error, i, ci, title):
    cname = f"case_{ci:03d}_calibration"
    fpath = resfig_dir / f"{cname}.png"
    print("Writing figure to", fpath)
    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.scatter(k_values, error, color="blue")
    ax.scatter(k_values[i], error[i], color="red", label=f"Best k = {k_values[i]:.3f}")
    ax.set_xlabel("k")
    ax.set_ylabel("RMSE / Capacity")
    ax.set_title(title)     
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(fpath)
    plt.close(fig)

def write_figure_flow(calfig_dir, fig, ci, title):
    cname = f"case_{ci:03d}_flow"
    fpath = calfig_dir / f"{cname}.png"
    print("Writing figure to", fpath)
    fig.suptitle(title)
    #fig.tight_layout()
    fig.savefig(fpath)
    plt.close(fig)
    
with foxes.Engine.new(
    engine, 
    chunk_size_states=chunksize_s,
    chunk_size_points=chunksize_p,
    n_procs=n_procs,
    verbosity=1,
) as engine:
    ci = 0
    for cdi, case_dir in enumerate(cases_dirs):
        print(f"\nENTERING CASE DIR {case_dir.name}")

        tdfiles = list(case_dir.glob("turbine_data*.nc"))
        assert len(tdfiles) == 1, f"Expected one turbine data file, found {tdfiles}"
        cfd_results = open_dataset(tdfiles[0])
        sysfiles = list(case_dir.glob("*system*.yaml"))
        assert len(sysfiles) == 1, f"Expected one system file, found {sysfiles}"
        wio_dict = foxes.input.yaml.windio.windio_file2dict(sysfiles[0])
        n_turbines = foxes.input.yaml.windio.read_n_turbines(wio_dict)
        hub_heights = foxes.input.yaml.windio.read_hub_heights(wio_dict)
        assert len(hub_heights) == 1, f"Only one hub height supported, got {hub_heights} for case {case_dir.name}"
        hh = hub_heights[0]
        rotor_diameters = foxes.input.yaml.windio.read_rotor_diameters(wio_dict)
        assert len(rotor_diameters) == 1, f"Only one rotor diameter supported, got {rotor_diameters} for case {case_dir.name}"
        D = rotor_diameters[0]

        physical_inputs = open_dataset(case_dir / "updated_physical_inputs.nc")
        physical_inputs = physical_inputs.interp(height=max(hub_heights))
        for v, d in physical_inputs.data_vars.items():
            if cdi == 0:
                all_results_data[v] = [("case",), []]
            if v in all_results_data:
                all_results_data[v][1] += d.values.tolist()

        n_times = physical_inputs.sizes["time"]
        all_results_data["case_name"][1] += [case_dir.name] * n_times
        all_results_data["n_turbines"][1] += [n_turbines] * n_times
        all_results_data["hub_height"][1] += [hh] * n_times
        all_results_data["rotor_diameter"][1] += [D] * n_times
        all_results_data["time_index"][1] += list(range(n_times))

        n_pop = len(k_values)
        pop_data = np.zeros((n_pop, n_turbines))
        pop_data[:] = k_values[:, None]
        pop_data = Dataset({
            FV.K: (("index", FC.TURBINE), pop_data),
        })

        idict, algo, odir = foxes.input.yaml.windio.read_windio_dict(
            wio_dict,
            rotor_model=rotor,
            population_params=dict(data_source=pop_data, verbosity=0),
            verbosity=verbosity
        )
        for wmodel in algo.wake_models.values():
            if hasattr(wmodel, "wake_k"):
                wmodel.wake_k = foxes.core.WakeK()

        print("Running foxes")
        farm_results = algo.calc_farm()
        print("Done running foxes")

        pop_results = algo.population_model.farm2pop_results(algo, farm_results)

        cap = algo.farm.get_capacity(algo)
        error = calc_error(cfd_results, pop_results, cap)
        eri = np.argmin(error, axis=0)

        futures = []
        for t in range(n_times):
            title = f"Case {ci + t}: {case_dir.name}, Time {t}"
            futures.append(
                engine.submit(
                    write_figure_calib, 
                    calfig_dir, 
                    k_values, 
                    error[:, t], 
                    eri[t], 
                    ci + t, 
                    title,
                )
            )
        for f in futures:
            engine.result(f)

        P_mean = (pop_results[FV.P] * pop_results[FV.WEIGHT]).sum(dim=FC.TURBINE).values
        P_mean = np.take_along_axis(P_mean, eri[None], axis=0)[0]
        error = np.take_along_axis(error, eri[None], axis=0)[0]
        k = k_values[eri]

        all_results_data["capacity"][1] += [cap] * n_times
        all_results_data["best_k"][1] += k.tolist()
        all_results_data["best_k_farm_P"][1] += P_mean.tolist()
        all_results_data["best_k_error"][1] += error.tolist()

        """
        farm_results = algo.select_population_member(pop_results, 2)
        o = foxes.output.FlowPlots2D(algo, farm_results)
        futures = []
        for t, fig in enumerate(o.gen_states_fig_xy(
            FV.WS,
            resolution=20,
            figsize=(6, 6),
            rotor_color="red",
        )):
            title = f"Case {ci + t}: {case_dir.name}, Time {t}, k={k[t]:.3f}"
            futures.append(
                engine.submit(
                    write_figure_flow,
                    calfig_dir, 
                    fig,
                    ci + t, 
                    title,
                )
            )
        for f in futures:
            engine.result(f)
        """
        ci += n_times

all_results_data = {v:(d[0],np.asarray(d[1])) for v,d in all_results_data.items()}    
results = Dataset(all_results_data)#, coords=all_results_coords)
print("\nResults:\n", results)
print(f"\nSaving results to {results_file}")
foxes.utils.write_nc(results, results_file)
