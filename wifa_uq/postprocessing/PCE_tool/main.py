import yaml
import os
import xarray as xr
from pce_utils import (
    build_input_output_arrays,
    construct_PCE_ot,
    compute_sobol_indices,
    plot_sobol_indices,
    plot_training_quality,
)


def main():
    # Default values for optional parameters
    defaults = {
        "pce_degree": 5,
        "copula": "normal",
        "marginals": "kernel",
        "q": 1,
        "output_variable": "model_bias_cap",
        "save_plots": True,
        "evaluate_training": False,
        "plot_options": {
            "scatter": True,
            "distribution": True,
            "metrics": ["RMSE", "KL"],
        },
    }

    # Load user config
    with open("config.yaml", "r") as f:
        user_config = yaml.safe_load(f)

    # Merge defaults with user config
    config = {**defaults, **user_config}
    config["plot_options"] = {
        **defaults["plot_options"],
        **user_config.get("plot_options", {}),
    }

    # Load dataset
    ds = xr.load_dataset(config["data_file"])
    ntimes = len(ds["wind_farm"].values)

    # Build input/output arrays
    input_array, output_array, varnames, kk_values, nvar_physical = (
        build_input_output_arrays(
            ds,
            config["stochastic_variables"],
            config["model_variables"],
            config["output_variable"],
        )
    )

    nvar = len(varnames)

    # Construct PCE
    pce_result = construct_PCE_ot(
        input_array=input_array,
        output_array=output_array,
        marginals=[config["marginals"]] * nvar,
        copula=config["copula"],
        degree=config["pce_degree"],
        q=config["q"],
    )

    PCE_metamodel = pce_result.getMetaModel()

    # Compute Sobol indices
    sobol_first, sobol_total = compute_sobol_indices(pce_result, nvar)

    # Plot Sobol indices
    plot_sobol_indices(
        sobol_first,
        sobol_total,
        varnames,
        save=config["save_plots"],
        filename="plots/sobol_indices.png",
    )

    # Optional training evaluation plots
    if config["evaluate_training"]:
        plot_training_quality(
            calibration_inputs=ds,
            stochastic_varnames_physical=config["stochastic_variables"],
            model_bias_cap=ds[config["output_variable"]].values,
            kk_values=kk_values,
            input_variable_array_physical=input_array[:ntimes, :nvar_physical],
            nvar_physical=nvar_physical,
            nvar=nvar,
            PCE_metamodel=PCE_metamodel,
            save=config["save_plots"],
            plot_options=config["plot_options"],
            seed=42,
        )


if __name__ == "__main__":
    # Ensure plot folder exists
    if not os.path.exists("plots"):
        os.makedirs("plots")
    main()
