import numpy as np
import pandas as pd
from pathlib import Path
import openturns as ot
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import wasserstein_distance, ks_2samp, entropy


def build_input_output_arrays(ds, stochastic_vars, model_vars, output_var):
    """
    Build input and output arrays for PCE from dataset
    """
    ntimes = len(ds["wind_farm"].values)
    kk_values = ds[model_vars[0]].values if model_vars else np.array([0.0])
    if not model_vars:
        nk = 1
    else:
        nk = ds.sizes["sample"]

    n_feature = len(stochastic_vars)
    nvar = n_feature + len(model_vars)

    input_array_physical = np.zeros((ntimes, n_feature))
    input_array = np.zeros((ntimes * nk, nvar))
    output_array = np.zeros((ntimes * nk))

    # Physical variables
    for j, var in enumerate(stochastic_vars):
        values = np.array(ds[var])
        input_array_physical[:, j] = values
        for i in range(nk):
            input_array[i * ntimes : (i + 1) * ntimes, j] = values

    # Model variables
    for j, var in enumerate(model_vars):
        for i in range(nk):
            input_array[i * ntimes : (i + 1) * ntimes, n_feature + j] = kk_values[i]

    # Output

    if not model_vars:
        # output_var expected shape (sample, ntimes)
        ymean = ds[output_var].mean(dim="sample").values  # shape (ntimes,)
        output_array[:] = np.tile(
            ymean, nk
        )  # ou mieux: output_array = np.tile(ymean, nk)
    else:
        for i in range(nk):
            output_array[i * ntimes : (i + 1) * ntimes] = ds[output_var][i, :]

    varnames = stochastic_vars + model_vars
    return input_array, output_array, varnames, kk_values, n_feature


def construct_PCE_ot(
    input_array, output_array, marginals, copula, degree, LARS=True, q=1
):
    """
    Construct Polynomial Chaos Expansion using OpenTURNS
    """
    Nt = input_array.shape[0]
    Nvar = input_array.shape[1]

    # Create samples
    outputSample = ot.Sample(Nt, 1)
    for i in range(Nt):
        outputSample[i, 0] = output_array[i]

    polyColl = ot.PolynomialFamilyCollection(Nvar)
    collection = ot.DistributionCollection(Nvar)
    marginal = {}
    UncorrelatedInputSample = ot.Sample(input_array)

    # Marginals
    for i in range(Nvar):
        varSample = ot.Sample(Nt, 1)
        for j in range(Nt):
            varSample[j, 0] = input_array[j, i]
        minValue = varSample.getMin()[0]
        maxValue = varSample.getMax()[0]
        if marginals[i] == "kernel":
            marginal[i] = ot.KernelSmoothing().build(varSample)
        elif marginals[i] == "uniform":
            marginal[i] = ot.Uniform(minValue - 1e-5, maxValue + 1e-5)
        else:
            marginal[i] = ot.NormalFactory().build(varSample)
        collection[i] = ot.Distribution(marginal[i])

    # Copula
    if copula == "independent":
        copula = ot.IndependentCopula(Nvar)
    elif copula in ["gaussian", "normal"]:
        copula = ot.NormalCopulaFactory().build(ot.Sample(input_array))
    else:
        copula = ot.IndependentCopula(Nvar)

    UncorrelatedInputDistribution = ot.ComposedDistribution(collection, copula)

    # Polynomial basis
    for v in range(Nvar):
        marginalv = UncorrelatedInputDistribution.getMarginal(v)
        polyColl[v] = ot.StandardDistributionPolynomialFactory(marginalv)

    enumerateFunction = ot.HyperbolicAnisotropicEnumerateFunction(Nvar, q)
    multivariateBasis = ot.OrthogonalProductPolynomialFactory(
        polyColl, enumerateFunction
    )
    P = enumerateFunction.getStrataCumulatedCardinal(degree)
    adaptativeStrategy = ot.FixedStrategy(multivariateBasis, P)

    if LARS:
        basisSequenceFactory = ot.LARS()
        fittingAlgorithm = ot.CorrectedLeaveOneOut()
        approximationAlgorithm = ot.LeastSquaresMetaModelSelectionFactory(
            basisSequenceFactory, fittingAlgorithm
        )
        projectionStrategy = ot.LeastSquaresStrategy(
            UncorrelatedInputSample, outputSample, approximationAlgorithm
        )
        algo = ot.FunctionalChaosAlgorithm(
            UncorrelatedInputSample,
            outputSample,
            UncorrelatedInputDistribution,
            adaptativeStrategy,
            projectionStrategy,
        )
    else:
        wei_exp = ot.MonteCarloExperiment(
            UncorrelatedInputDistribution, UncorrelatedInputSample.getSize()
        )
        X_UncorrelatedInputSample, weights = wei_exp.generateWithWeights()
        projectionStrategy = ot.LeastSquaresStrategy()
        algo = ot.FunctionalChaosAlgorithm(
            X_UncorrelatedInputSample,
            weights,
            outputSample,
            UncorrelatedInputDistribution,
            adaptativeStrategy,
            projectionStrategy,
        )

    algo.run()
    return algo.getResult()


def compute_sobol_indices(pce_result, nvar):
    """
    Compute first and total Sobol indices for each k
    """
    first_sobol = np.zeros((nvar))
    total_sobol = np.zeros((nvar))
    chaosSI = ot.FunctionalChaosSobolIndices(pce_result)
    for v in range(nvar):
        first_sobol[v] = chaosSI.getSobolIndex(v)
        total_sobol[v] = chaosSI.getSobolTotalIndex(v)
    return first_sobol, total_sobol


def plot_sobol_indices(
    first, total, varnames, save=False, filename="pce_sobol_indices.png"
):
    """
    Plot Sobol indices
    """
    colors = ["#002d74", "#e85113"]
    nvar = len(varnames)
    bar_width = 0.24
    bar_positions1 = np.arange(nvar) + 0.5 * bar_width
    bar_positions2 = np.arange(nvar) + 1.5 * bar_width

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(bar_positions1, first, width=bar_width, color=colors[0], label="1st order")
    ax.bar(bar_positions2, total, width=bar_width, color=colors[1], label="Total order")
    ax.set_xticks(bar_positions2)
    ax.set_xticklabels(varnames, rotation=45, ha="right")
    ax.set_xlabel("Variables")
    ax.set_ylabel("Sobol indices")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=150)
    plt.show()


def plot_training_quality(
    database,
    varnames,
    value_of_interest,
    kk_values,
    input_variable_array_physical,
    n_feature,
    nvar,
    PCE_metamodel,
    save=True,
    plot_options=None,
    seed=42,
    output_dir="./PCE_training_quality/",
):
    """
    Evaluate PCE training quality via scatter plots, distribution comparison, and metrics.

    Special case:
    - If there is no model variable (nvar == n_feature), PCE is deterministic per time index.

    Args:
        database: Dataset of inputs.
        varnames: Physical (stochastic) variable names.
        value_of_interest: Target array of shape (sample_size, ntimes).
        kk_values: Model-variable values per sample (None if no model var).
        input_variable_array_physical: Physical input array (n_samples, n_feature).
        n_feature: Number of physical variables.
        nvar: Total number of input variables (physical + model).
        PCE_metamodel: OpenTURNS PCE metamodel callable.
        save: Whether to save figures to disk.
        plot_options: Dict with flags: {"scatter": bool, "distribution": bool, "metrics": list}.
        seed: RNG seed for selecting time indices.
        output_dir: Output directory for figures.

    Returns:
        None
    """

    ntimes = value_of_interest.shape[1]

    # Always use dataset sample dimension
    sample_size = value_of_interest.shape[0]

    # Detect presence of a model variable
    has_model_var = (
        (nvar > n_feature)
        and (kk_values is not None)
        and (len(kk_values) == sample_size)
    )

    # Compute PCE realizations: shape (sample_size, ntimes)
    PCE_realizations = np.zeros((sample_size, ntimes))

    for it in range(ntimes):
        physical_var = input_variable_array_physical[it, :]

        if has_model_var:
            realization = np.zeros(nvar)
            realization[:n_feature] = physical_var
            for j, k in enumerate(kk_values):
                realization[-1] = k
                PCE_realizations[j, it] = PCE_metamodel(realization)[0]
        else:
            # deterministic (no model var): 1 prediction per time index, broadcast over samples
            realization = np.zeros(n_feature)
            realization[:] = physical_var
            yhat = PCE_metamodel(realization)[0]
            PCE_realizations[:, it] = yhat

    # ------------------------------------------------------------------
    # SPECIAL CASE: no model var -> plot value_of_interest[:, :] vs time
    # ------------------------------------------------------------------

    if not has_model_var and plot_options.get("scatter", False):
        # PCE deterministic per time index -> 1 prediction per it
        obs = np.asarray(value_of_interest)  # (sample_size, ntimes)
        pce_curve = np.asarray(PCE_realizations[0, :])  # (ntimes,)

        # Build scatter arrays over all (sample, time) pairs
        x = obs.ravel()  # all observed points
        y = np.tile(pce_curve, sample_size)  # predicted repeated over samples

        fig, ax = plt.subplots(1, 1, figsize=(7, 6))
        ax.grid(True)

        ax.plot(
            x,
            y,
            "r.",
            alpha=0.35,
            markersize=4,
            label="PCE vs Data (all samples, all times)",
        )

        # y=x reference
        xmin = float(min(np.min(x), np.min(y)))
        xmax = float(max(np.max(x), np.max(y)))
        ax.plot([xmin, xmax], [xmin, xmax], "k--", linewidth=1.5, label="y=x")

        ax.set_xlabel("Observed values")
        ax.set_ylabel("PCE prediction")
        ax.set_title(
            "Training scatter (no model var): PCE deterministic per time index"
        )
        ax.legend()

        plt.tight_layout()
        if save:
            plt.savefig(output_dir / "PCE_training_scatter_no_modelvar.png", dpi=150)
        # plt.show()

        return

    # ------------------------------------------------------------------
    # Standard case (model var exists): keep your previous behavior
    # ------------------------------------------------------------------

    # Fix random seed for reproducibility
    rng = np.random.default_rng(seed)
    n_pick = 4 if ntimes >= 4 else ntimes
    time_indices = rng.choice(ntimes, n_pick, replace=False)

    # --- Scatter plot ---
    if plot_options.get("scatter", False):
        fig, ax = plt.subplots(1, n_pick, figsize=(4.5 * n_pick, 6))
        if n_pick == 1:
            ax = [ax]

        for idx, it in enumerate(time_indices):
            line_len = 0
            str_title = ""
            for var_name in varnames:
                value_str = f"{var_name}={np.array(database[var_name])[it]:.4g}"
                if line_len + len(value_str) > 30:
                    str_title += "\n"
                    line_len = 0
                str_title += value_str + ", "
                line_len += len(value_str) + 2
            str_title = str_title.rstrip(", ")

            ax[idx].set_title(str_title)
            ax[idx].grid(True)

            ax[idx].plot(
                value_of_interest[:, it],
                PCE_realizations[:, it],
                "r.",
                label="PCE vs Data",
            )
            ax[idx].plot(
                value_of_interest[:, it], value_of_interest[:, it], "k--", label="y=x"
            )
            ax[idx].set_xlabel("Observed biases")
            ax[idx].set_ylabel("PCE biases")
            ax[idx].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        if save:
            plt.savefig(output_dir / "PCE_training_scatter.png", dpi=150)
        # plt.show()

    # --- Distribution comparison ---
    if plot_options.get("distribution", False):
        fig, ax = plt.subplots(1, n_pick, figsize=(4.5 * n_pick, 6))
        if n_pick == 1:
            ax = [ax]

        mb_min = float(value_of_interest.min().values)
        mb_max = float(value_of_interest.max().values)
        pce_min = float(np.min(PCE_realizations))
        pce_max = float(np.max(PCE_realizations))
        xs = np.linspace(min(mb_min, pce_min), max(mb_max, pce_max), 1000)

        def safe_kde(arr):
            arr = np.asarray(arr)
            if arr.size < 2:
                return None
            if np.std(arr) <= 1e-12:
                return None
            try:
                return stats.gaussian_kde(arr)
            except Exception:
                return None

        for i, it in enumerate(time_indices):
            ref = np.asarray(value_of_interest[:, it])
            pce = np.asarray(PCE_realizations[:, it])

            ref_kde = safe_kde(ref)
            pce_kde = safe_kde(pce)

            if ref_kde is not None:
                ax[i].plot(xs, ref_kde(xs), color="green", label="Observed")
            else:
                ax[i].axvline(
                    np.mean(ref),
                    color="green",
                    linestyle="--",
                    label="Observed (degenerate)",
                )

            if pce_kde is not None:
                ax[i].plot(xs, pce_kde(xs), color="blue", label="PCE")
            else:
                ax[i].axvline(
                    np.mean(pce), color="blue", linestyle="--", label="PCE (degenerate)"
                )

            ax[i].set_xlabel("Biases")
            ax[i].set_ylabel("PDF")
            ax[i].grid(True)
            if i == 0:
                ax[i].legend()

        if save:
            plt.savefig(output_dir / "PCE_training_distribution.png", dpi=150)
        # plt.show()

    # --- Metrics ---
    metrics_to_plot = plot_options.get("metrics", [])
    if metrics_to_plot:
        mb_min = float(value_of_interest.min().values)
        mb_max = float(value_of_interest.max().values)
        pce_min = float(np.min(PCE_realizations))
        pce_max = float(np.max(PCE_realizations))
        xs = np.linspace(min(mb_min, pce_min), max(mb_max, pce_max), 1000)

        metric_dict = {"RMSE": [], "R2": [], "Wasserstein": [], "KS": [], "KL": []}

        def can_kde(arr):
            arr = np.asarray(arr)
            return (arr.size >= 2) and (np.std(arr) > 1e-12)

        for it in range(ntimes):
            ref = np.asarray(value_of_interest[:, it])
            pce = np.asarray(PCE_realizations[:, it])

            metric_dict["RMSE"].append(np.sqrt(mean_squared_error(ref, pce)))
            metric_dict["R2"].append(r2_score(ref, pce))
            metric_dict["Wasserstein"].append(wasserstein_distance(ref, pce))

            ks_stat, _ = ks_2samp(ref, pce)
            metric_dict["KS"].append(ks_stat)

            if "KL" in metrics_to_plot and can_kde(ref) and can_kde(pce):
                try:
                    ref_kde = stats.gaussian_kde(ref)
                    pce_kde = stats.gaussian_kde(pce)
                    p_vals = ref_kde(xs) + 1e-12
                    q_vals = pce_kde(xs) + 1e-12
                    p_vals /= np.sum(p_vals)
                    q_vals /= np.sum(q_vals)
                    metric_dict["KL"].append(entropy(p_vals, q_vals))
                except Exception:
                    metric_dict["KL"].append(np.nan)
            else:
                metric_dict["KL"].append(np.nan)

        fig, axes = plt.subplots(
            len(metrics_to_plot), 1, figsize=(10, 3 * len(metrics_to_plot)), sharex=True
        )
        if len(metrics_to_plot) == 1:
            axes = [axes]

        for ax, metric in zip(axes, metrics_to_plot):
            ax.plot(metric_dict[metric], label=metric, color="tab:blue")
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()

        axes[-1].set_xlabel("Time index")
        plt.tight_layout()
        if save:
            plt.savefig(output_dir / "PCE_training_metrics.png", dpi=150)
        # plt.show()


def run_pce_sensitivity(database, feature_names, pce_config: dict, output_dir: Path):
    """
    PCE-based sensitivity analysis on error.

    Computes Sobol indices to determine which physical features
    contribute most to error variance.

    Args:
        database: xarray Dataset
        feature_names: List of feature names corresponding to X columns
        pce_config: Dict with PCE settings:
            - degree (int): Polynomial degree, default 5
            - marginals (str): 'kernel', 'uniform', or 'normal', default 'kernel'
            - copula (str): 'independent' or 'normal', default 'independent'
            - q (float): Hyperbolic truncation parameter, default 1.0
            plot_options: Dict with settings to evalute PCE quality
                - scatter (bool): activate scatter plot
                - distribution (bool): activate distribution plot
                - metrics (str): Metrics to plot["RMSE", "R2", "Wasserstein", "KS", "KL"]
        output_dir: Path to save plots and CSV output

    Returns:
        Dict with:
            - 'sobol_first': dict mapping feature names to first-order indices
            - 'sobol_total': dict mapping feature names to total-order indices
            - 'pce_result': the fitted PCE object (for further analysis if needed)
            - 'feature_names': list of feature names
            - 'model_coeff_name': model coefficient name
            - 'varnames': list of variable names, feature_names + model_coeff_name
            - 'value_of_interest': value_of_interest, 'ref_power_cap' if model_coeff_name = None, else 'model_bias_cap'

    """
    # Extract config with defaults
    degree = pce_config.get("degree", 5)
    marginals = pce_config.get("marginals", "kernel")
    copula = pce_config.get("copula", "independent")
    q = pce_config.get("q", 1.0)
    DEFAULT_PLOT_OPTIONS = {
        "scatter": True,
        "distribution": False,
        "metrics": [],
    }
    user_plot_options = pce_config.get("plot_options", {}) or {}
    plot_options = {**DEFAULT_PLOT_OPTIONS, **user_plot_options}

    print("pce_config keys:", pce_config.keys())
    print("pce_config:", pce_config)
    print("plot_options", plot_options)
    model_coeff_name = pce_config.get("model_coeff_name", None)

    if isinstance(model_coeff_name, str) and model_coeff_name.strip().lower() == "none":
        model_coeff_name = None

    # Decide model vars + target output (hardcoded target names)
    if model_coeff_name is None:
        model_vars = []
        value_of_interest = "ref_power_cap"
    else:
        model_vars = [model_coeff_name]
        value_of_interest = "model_bias_cap"

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n--- PCE Sensitivity (single run) ---")
    print(f"    model_coeff_name: {model_coeff_name}")
    print(f"    value_of_interest: {value_of_interest}")

    # Build arrays
    input_array, output_array, varnames, kk_values, n_feature = (
        build_input_output_arrays(
            database, feature_names, model_vars, value_of_interest
        )
    )

    # Validate shapes
    n_samples, n_features = input_array.shape

    print("--- Running PCE Sensitivity Analysis ---")
    print(f"    Samples: {n_samples}, Features: {n_features}")
    print(f"    Degree: {degree}, Marginals: {marginals}, Copula: {copula}, q: {q}")
    print(f"    Features: {feature_names}")

    # Construct PCE metamodel
    print("    Constructing PCE metamodel...")
    pce_result = construct_PCE_ot(
        input_array=input_array,
        output_array=output_array,
        marginals=[marginals] * n_features,
        copula=copula,
        degree=degree,
        q=q,
    )

    # Compute Sobol indices
    print("    Computing Sobol indices...")
    sobol_first, sobol_total = compute_sobol_indices(pce_result, n_features)

    # Create results dict
    results = {
        "sobol_first": dict(zip(varnames, sobol_first)),
        "sobol_total": dict(zip(varnames, sobol_total)),
        "pce_result": pce_result,
        "feature_names": feature_names,
        "model_coeff_name": model_coeff_name,
        "varnames": varnames,
        "value_of_interest": value_of_interest,
    }
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Plot Sobol indices

    # Save outputs (use suffix based on model_coeff_name/target)
    tag = "no_modelvar" if model_coeff_name is None else str(model_coeff_name)
    plot_file = output_dir / f"pce_sobol_indices_{tag}.png"
    csv_file = output_dir / f"pce_sobol_indices_{tag}.csv"

    plot_sobol_indices(
        sobol_first,
        sobol_total,
        varnames,
        save=True,
        filename=str(plot_file),
    )
    print(f"    Saved plot to {plot_file}")

    sobol_df = pd.DataFrame(
        {"Var Names": varnames, "First_Order": sobol_first, "Total_Order": sobol_total}
    ).sort_values("Total_Order", ascending=False)
    sobol_df.to_csv(csv_file, index=False)
    print(f"    Saved indices to {csv_file}")

    # Training quality plot
    PCE_metamodel = pce_result.getMetaModel()

    array_of_interest = database[value_of_interest]

    plot_training_quality(
        database=database,
        varnames=varnames,
        value_of_interest=array_of_interest,
        kk_values=kk_values,
        input_variable_array_physical=input_array[:n_samples, :n_feature],
        n_feature=n_feature,
        nvar=n_feature + (0 if model_coeff_name is None else 1),
        PCE_metamodel=PCE_metamodel,
        save=True,
        plot_options=plot_options,
        seed=42,
        output_dir=output_dir,
    )

    return results
