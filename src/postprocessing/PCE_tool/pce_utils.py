import numpy as np
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
    nk = len(kk_values)
    
    nvar_physical = len(stochastic_vars)
    nvar = nvar_physical + len(model_vars)
    
    input_array_physical = np.zeros((ntimes, nvar_physical))
    input_array = np.zeros((ntimes*nk, nvar))
    output_array = np.zeros((ntimes*nk))
    
    # Physical variables
    for j, var in enumerate(stochastic_vars):
        values = np.array(ds[var])
        input_array_physical[:, j] = values
        for i in range(nk):
            input_array[i*ntimes:(i+1)*ntimes, j] = values

    # Model variables
    for j, var in enumerate(model_vars):
        for i in range(nk):
            input_array[i*ntimes:(i+1)*ntimes, nvar_physical+j] = kk_values[i]

    # Output
    for i in range(nk):
        output_array[i*ntimes:(i+1)*ntimes] = ds[output_var][i, :]
    
    varnames = stochastic_vars + model_vars
    return input_array, output_array, varnames, kk_values, nvar_physical

def construct_PCE_ot(input_array, output_array, marginals, copula, degree, LARS=True, q=1):
    """
    Construct Polynomial Chaos Expansion using OpenTURNS
    """
    Nt = input_array.shape[0]
    Nvar = input_array.shape[1]

    # Create samples
    outputSample = ot.Sample(Nt, 1)
    for i in range(Nt):
        outputSample[i,0] = output_array[i]

    polyColl = ot.PolynomialFamilyCollection(Nvar)
    collection = ot.DistributionCollection(Nvar)
    marginal = {}
    UncorrelatedInputSample = ot.Sample(input_array)

    # Marginals
    for i in range(Nvar):
        varSample = ot.Sample(Nt,1)
        for j in range(Nt):
            varSample[j,0] = input_array[j,i]
        minValue = varSample.getMin()[0]
        maxValue = varSample.getMax()[0]
        if marginals[i]=="kernel":
            marginal[i] = ot.KernelSmoothing().build(varSample)
        elif marginals[i]=="uniform":
            marginal[i] = ot.Uniform(minValue-1e-5, maxValue+1e-5)
        else:
            marginal[i] = ot.NormalFactory().build(varSample)
        collection[i] = ot.Distribution(marginal[i])

    # Copula
    if copula=="independent":
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
    multivariateBasis = ot.OrthogonalProductPolynomialFactory(polyColl, enumerateFunction)
    P = enumerateFunction.getStrataCumulatedCardinal(degree)
    adaptativeStrategy = ot.FixedStrategy(multivariateBasis, P)

    if LARS:
        basisSequenceFactory = ot.LARS()
        fittingAlgorithm = ot.CorrectedLeaveOneOut()
        approximationAlgorithm = ot.LeastSquaresMetaModelSelectionFactory(basisSequenceFactory, fittingAlgorithm)
        projectionStrategy = ot.LeastSquaresStrategy(UncorrelatedInputSample, outputSample, approximationAlgorithm)
        algo = ot.FunctionalChaosAlgorithm(UncorrelatedInputSample, outputSample, UncorrelatedInputDistribution, adaptativeStrategy, projectionStrategy)
    else:
        wei_exp = ot.MonteCarloExperiment(UncorrelatedInputDistribution, UncorrelatedInputSample.getSize())
        X_UncorrelatedInputSample, weights = wei_exp.generateWithWeights()
        projectionStrategy = ot.LeastSquaresStrategy()
        algo = ot.FunctionalChaosAlgorithm(X_UncorrelatedInputSample, weights, outputSample, UncorrelatedInputDistribution, adaptativeStrategy, projectionStrategy)

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

def plot_sobol_indices(first, total, varnames, save=False, filename="plots/sobol.png"):
    """
    Plot Sobol indices
    """
    colors = ['#002d74', '#e85113']
    nvar = len(varnames)
    bar_width = 0.24
    bar_positions1 = np.arange(nvar) + 0.5*bar_width
    bar_positions2 = np.arange(nvar) + 1.5*bar_width

    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(bar_positions1, first, width=bar_width, color=colors[0], label='1st order')
    ax.bar(bar_positions2, total, width=bar_width, color=colors[1], label='Total order')
    ax.set_xticks(bar_positions2)
    ax.set_xticklabels(varnames, rotation=45, ha='right')
    ax.set_xlabel("Variables")
    ax.set_ylabel("Sobol indices")
    ax.set_ylim(0,0.7)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if save:
        plt.savefig(filename, dpi=150)
    plt.show()

def plot_training_quality(calibration_inputs, stochastic_varnames_physical, model_bias_cap,
                          kk_values, input_variable_array_physical, nvar_physical, nvar,
                          PCE_metamodel, save=False, plot_options=None, seed=42):
    """
    Evaluate PCE training quality: scatter, distributions, metrics.
    Select 4 random time indices with fixed seed for reproducibility.
    """
    if plot_options is None:
        plot_options = {"scatter": True, "distribution": True, "metrics": ["RMSE", "R2", "KL"]}

    ntimes = model_bias_cap.shape[1]
    sample_size = len(kk_values)

    # Compute PCE realizations
    PCE_realizations = np.zeros((sample_size, ntimes))
    for it in range(ntimes):
        physical_var = input_variable_array_physical[it, :]
        realization = np.zeros(nvar)
        realization[:nvar_physical] = physical_var
        for j, k in enumerate(kk_values):
            realization[-1] = k
            PCE_realizations[j, it] = PCE_metamodel(realization)[0]

    # Fix random seed for reproducibility
    rng = np.random.default_rng(seed)
    time_indices = rng.choice(ntimes, 4, replace=False)

    # --- Scatter plot ---
    if plot_options.get("scatter", False):
        fig, ax = plt.subplots(1, 4, figsize=(18,6))
        handles, labels = [], []
        for idx, it in enumerate(time_indices):
            line_len = 0
            str_title = ''
            for var_name in stochastic_varnames_physical:
                value_str = f"{var_name}={np.array(calibration_inputs[var_name])[it]:.4g}"
                if line_len + len(value_str) > 30:  # max 30 chars per line
                    str_title += "\n"
                    line_len = 0
                str_title += value_str + ", "
                line_len += len(value_str) + 2
            str_title = str_title.rstrip(", ")
            ax[idx].set_title(str_title)
            ax[idx].grid(True)
            sc1 = ax[idx].plot(model_bias_cap[:, it], PCE_realizations[:, it], 'r.', label='PCE')
            sc2 = ax[idx].plot(model_bias_cap[:, it], model_bias_cap[:, it], 'k--', label='y=x')
            if idx==0:
                handles.extend([sc1[0], sc2[0]])
                labels.extend(['PCE vs Data', 'y=x'])
                ax[idx].legend()
            ax[idx].set_xlabel("Observed biases ")
            ax[idx].set_ylabel("PCE biases")                
        plt.tight_layout(rect=[0,0,1,0.95])
        if save:
            plt.savefig("plots/training_scatter.png", dpi=150)
        plt.show()

    # --- Distribution comparison ---
    if plot_options.get("distribution", False):
        fig, ax = plt.subplots(1, 4, figsize=(18,6))
        xs = np.linspace(min(model_bias_cap.min(), PCE_realizations.min()),
                         max(model_bias_cap.max(), PCE_realizations.max()), 1000)
        for i, it in enumerate(time_indices):
            line_len = 0
            str_title = ''
            for var_name in stochastic_varnames_physical:
                value_str = f"{var_name}={np.array(calibration_inputs[var_name])[it]:.4g}"
                if line_len + len(value_str) > 30:
                    str_title += "\n"
                    line_len = 0
                str_title += value_str + ", "
                line_len += len(value_str) + 2
            str_title = str_title.rstrip(", ")
            ref_kde = stats.gaussian_kde(model_bias_cap[:, it])
            pce_kde = stats.gaussian_kde(PCE_realizations[:, it])
            ax[i].plot(xs, ref_kde(xs), color='green', label='Observed')
            ax[i].plot(xs, pce_kde(xs), color='blue', label='PCE')
            ax[i].set_title(str_title)
            ax[i].set_xlabel("Biases")
            ax[i].set_ylabel("PDF")
            ax[i].grid(True)
            if i==0:
                ax[i].legend()
        if save:
            plt.savefig("plots/training_distribution.png", dpi=150)
        plt.show()

    # --- Metrics ---
    metrics_to_plot = plot_options.get("metrics", [])
    if metrics_to_plot:
        xs = np.linspace(min(model_bias_cap.min(), PCE_realizations.min()),
                         max(model_bias_cap.max(), PCE_realizations.max()), 1000)
        metric_dict = {"RMSE": [], "R2": [], "Wasserstein": [], "KS": [], "KL": []}
        for it in range(ntimes):
            ref = model_bias_cap[:, it]
            pce = PCE_realizations[:, it]
            metric_dict["RMSE"].append(np.sqrt(mean_squared_error(ref, pce)))
            metric_dict["R2"].append(r2_score(ref, pce))
            metric_dict["Wasserstein"].append(wasserstein_distance(ref, pce))
            ks_stat, _ = ks_2samp(ref, pce)
            metric_dict["KS"].append(ks_stat)
            ref_kde = stats.gaussian_kde(ref)
            pce_kde = stats.gaussian_kde(pce)
            p_vals = ref_kde(xs) + 1e-12
            q_vals = pce_kde(xs) + 1e-12
            p_vals /= np.sum(p_vals)
            q_vals /= np.sum(q_vals)
            metric_dict["KL"].append(entropy(p_vals, q_vals))

        fig, axes = plt.subplots(len(metrics_to_plot),1, figsize=(10,3*len(metrics_to_plot)), sharex=True)
        if len(metrics_to_plot)==1:
            axes = [axes]
        for ax, metric in zip(axes, metrics_to_plot):
            ax.plot(metric_dict[metric], label=metric, color="tab:blue")
            ax.set_ylabel(metric)
            ax.grid(True)
            ax.legend()
        axes[-1].set_xlabel("Time index")
        plt.tight_layout()
        if save:
            plt.savefig("plots/training_metrics.png", dpi=150)
        plt.show()
