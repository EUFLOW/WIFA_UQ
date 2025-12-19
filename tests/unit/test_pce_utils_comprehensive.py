# tests/unit/test_pce_utils_comprehensive.py
"""
Comprehensive unit tests for wifa_uq.postprocessing.PCE_tool.pce_utils

Covers:
- build_input_output_arrays (with/without model variables, ordering, output layout)
- construct_PCE_ot (marginals, copula, LARS on/off, q, prediction sanity)
- compute_sobol_indices (bounds/shape/basic dominance sanity)
- plot_sobol_indices (file output, no GUI)
- plot_training_quality (scatter/distribution/metrics; special no-model-var branch)
- run_pce_sensitivity (end-to-end outputs + config behaviors)

These tests are designed to run fast and deterministically on CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


# Force non-interactive backend early
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wifa_uq.postprocessing.PCE_tool.pce_utils import (
    build_input_output_arrays,
    construct_PCE_ot,
    compute_sobol_indices,
    plot_sobol_indices,
    plot_training_quality,
    run_pce_sensitivity,
)


# -----------------------------------------------------------------------------
# Fixtures
# -----------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    """Prevent any plt.show() calls from blocking test runs."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture
def ds_with_model_var():
    """
    Dataset shaped like pce_utils expects:
      - wind_farm dimension is treated as "time"/flow_case axis
      - sample dimension exists for parameter samples
      - model variable appears as a coordinate with dim 'sample'
      - output variable has dims ('sample','wind_farm')
    """
    rng = np.random.default_rng(0)
    nt = 30
    ns = 10

    # Physical vars (time-like)
    ABL_height = rng.uniform(200, 1000, size=nt)
    wind_veer = rng.uniform(0, 0.01, size=nt)

    # One model variable (sample-like)
    k_b = np.linspace(0.01, 0.07, ns)

    # Output depends on both physical + model var (plus small noise)
    bias = np.zeros((ns, nt))
    for i in range(ns):
        bias[i, :] = (
            0.0008 * ABL_height
            + 30.0 * wind_veer
            + 4.0 * k_b[i]
            + rng.normal(0, 0.02, nt)
        )

    # Another output for no-model-var tests (still stored per sample, but run_pce_sensitivity
    # will average it across sample when model_vars is empty)
    ref_power = 0.5 + 0.0006 * ABL_height + rng.normal(0, 0.01, nt)
    ref_power = np.tile(ref_power, (ns, 1))

    ds = xr.Dataset(
        data_vars={
            "ABL_height": (("wind_farm",), ABL_height),
            "wind_veer": (("wind_farm",), wind_veer),
            "model_bias_cap": (("sample", "wind_farm"), bias),
            "ref_power_cap": (("sample", "wind_farm"), ref_power),
        },
        coords={
            "wind_farm": np.arange(nt),
            "sample": np.arange(ns),
            "k_b": (("sample",), k_b),
        },
    )
    return ds


@pytest.fixture
def ds_no_model_var(ds_with_model_var):
    """
    Same dataset, but we will call build_input_output_arrays/run_pce_sensitivity
    with model_vars=[] / model_coeff_name=None.
    """
    return ds_with_model_var


@pytest.fixture
def synthetic_poly_regression():
    """
    Simple polynomial regression dataset (fast) for construct_PCE_ot sanity:
      y = x1^2 + 2*x2 + 0.5*x1*x2 + noise
    """
    rng = np.random.default_rng(42)
    n = 250
    x1 = rng.uniform(-1, 1, size=n)
    x2 = rng.uniform(-1, 1, size=n)
    y = x1**2 + 2.0 * x2 + 0.5 * x1 * x2 + rng.normal(0, 0.05, size=n)
    X = np.column_stack([x1, x2])
    x2 += 1e-6 * rng.normal(size=n)  # avoid constant values / add jitter
    return X, y


# -----------------------------------------------------------------------------
# build_input_output_arrays
# -----------------------------------------------------------------------------


class TestBuildInputOutputArrays:
    def test_with_model_var_shapes_and_order(self, ds_with_model_var):
        X, y, varnames, kk_values, n_feature = build_input_output_arrays(
            ds_with_model_var,
            stochastic_vars=["ABL_height", "wind_veer"],
            model_vars=["k_b"],
            output_var="model_bias_cap",
        )

        nt = ds_with_model_var.sizes["wind_farm"]
        ns = ds_with_model_var.sizes["sample"]

        assert X.shape == (nt * ns, 3)  # 2 physical + 1 model
        assert y.shape == (nt * ns,)
        assert varnames == ["ABL_height", "wind_veer", "k_b"]
        assert n_feature == 2
        np.testing.assert_allclose(kk_values, ds_with_model_var["k_b"].values)

        # Check layout: block per sample
        # For sample i, first physical columns repeat the same nt series; last col is constant k_b[i]
        for i in range(ns):
            block = X[i * nt : (i + 1) * nt, :]
            np.testing.assert_allclose(
                block[:, 0], ds_with_model_var["ABL_height"].values
            )
            np.testing.assert_allclose(
                block[:, 1], ds_with_model_var["wind_veer"].values
            )
            np.testing.assert_allclose(block[:, 2], ds_with_model_var["k_b"].values[i])

        # Output block i matches ds[output_var][i,:]
        for i in range(ns):
            expected = ds_with_model_var["model_bias_cap"].isel(sample=i).values
            np.testing.assert_allclose(y[i * nt : (i + 1) * nt], expected)

    def test_without_model_var_output_is_sample_mean(self, ds_no_model_var):
        X, y, varnames, kk_values, n_feature = build_input_output_arrays(
            ds_no_model_var,
            stochastic_vars=["ABL_height", "wind_veer"],
            model_vars=[],
            output_var="ref_power_cap",
        )
        nt = ds_no_model_var.sizes["wind_farm"]
        # pce_utils sets nk=1 for no model vars (even if sample exists in dataset)
        assert X.shape == (nt, 2)
        assert y.shape == (nt,)
        assert varnames == ["ABL_height", "wind_veer"]
        assert n_feature == 2

        # output is mean over sample, then tiled nk=1 -> just the mean
        expected = ds_no_model_var["ref_power_cap"].mean(dim="sample").values
        np.testing.assert_allclose(y, expected)

        # kk_values is a placeholder array([0.0])
        assert isinstance(kk_values, np.ndarray)
        assert kk_values.shape == (1,)
        assert float(kk_values[0]) == 0.0

    def test_model_vars_more_than_one_is_not_supported_cleanly(self, ds_with_model_var):
        """
        The current implementation uses kk_values = ds[model_vars[0]].values
        and then applies that same kk_values to every model var column, which is a limitation.
        This test documents current behavior (not ideal), to prevent silent regressions.
        """
        ds = ds_with_model_var.assign_coords(
            ss_alpha=(
                "sample",
                np.linspace(0.75, 1.0, ds_with_model_var.sizes["sample"]),
            )
        )

        X, y, varnames, kk_values, n_feature = build_input_output_arrays(
            ds,
            stochastic_vars=["ABL_height"],
            model_vars=["k_b", "ss_alpha"],
            output_var="model_bias_cap",
        )

        nt = ds.sizes["wind_farm"]
        ns = ds.sizes["sample"]

        assert X.shape == (nt * ns, 3)  # 1 phys + 2 model vars
        assert varnames == ["ABL_height", "k_b", "ss_alpha"]

        # Current behavior: both model-var columns are filled using kk_values from model_vars[0] (k_b)
        # i.e., ss_alpha column will actually contain k_b values, not ss_alpha.
        for i in range(ns):
            block = X[i * nt : (i + 1) * nt, :]
            np.testing.assert_allclose(block[:, 1], ds["k_b"].values[i])
            np.testing.assert_allclose(
                block[:, 2], ds["k_b"].values[i]
            )  # documents current limitation


# -----------------------------------------------------------------------------
# construct_PCE_ot
# -----------------------------------------------------------------------------


class TestConstructPCEOT:
    @pytest.mark.parametrize("marginals", ["kernel", "uniform", "normal"])
    def test_constructs_and_predicts_reasonably(
        self, synthetic_poly_regression, marginals
    ):
        X, y = synthetic_poly_regression
        pce = construct_PCE_ot(
            input_array=X,
            output_array=y,
            marginals=[marginals, marginals],
            copula="independent",
            degree=3,
            q=1.0,
            LARS=True,
        )
        metamodel = pce.getMetaModel()
        yhat = np.array([metamodel(xi)[0] for xi in X])

        corr = np.corrcoef(y, yhat)[0, 1]
        assert corr > 0.85

    @pytest.mark.parametrize("copula", ["independent", "normal", "gaussian"])
    def test_copulas_run(self, synthetic_poly_regression, copula):
        X, y = synthetic_poly_regression
        pce = construct_PCE_ot(
            input_array=X,
            output_array=y,
            marginals=["kernel", "kernel"],
            copula=copula,
            degree=2,
            q=1.0,
        )
        assert pce is not None
        assert hasattr(pce, "getMetaModel")

    def test_lars_toggle(self, synthetic_poly_regression):
        X, y = synthetic_poly_regression
        pce_lars = construct_PCE_ot(
            input_array=X,
            output_array=y,
            marginals=["kernel", "kernel"],
            copula="independent",
            degree=2,
            LARS=True,
            q=1.0,
        )
        pce_no_lars = construct_PCE_ot(
            input_array=X,
            output_array=y,
            marginals=["kernel", "kernel"],
            copula="independent",
            degree=2,
            LARS=False,
            q=1.0,
        )
        assert pce_lars.getMetaModel() is not None
        assert pce_no_lars.getMetaModel() is not None


# -----------------------------------------------------------------------------
# compute_sobol_indices
# -----------------------------------------------------------------------------


class TestComputeSobolIndices:
    def test_shapes_and_bounds(self, synthetic_poly_regression):
        X, y = synthetic_poly_regression
        pce = construct_PCE_ot(
            input_array=X,
            output_array=y,
            marginals=["kernel", "kernel"],
            copula="independent",
            degree=3,
            q=1.0,
        )
        s1, st = compute_sobol_indices(pce, nvar=2)

        assert s1.shape == (2,)
        assert st.shape == (2,)
        assert np.all(s1 >= -1e-8)
        assert np.all(st >= -1e-8)
        assert np.all(st + 1e-8 >= s1)
        assert float(np.sum(s1)) <= 1.0 + 1e-3

    def test_dominant_variable_detected(self):
        rng = np.random.default_rng(2)
        n = 250
        x1 = rng.uniform(-1, 1, size=n)
        x2 = rng.uniform(-1, 1, size=n)
        y = 10.0 * x1 + 0.2 * x2 + rng.normal(0, 0.1, size=n)
        X = np.column_stack([x1, x2])

        pce = construct_PCE_ot(
            input_array=X,
            output_array=y,
            marginals=["kernel", "kernel"],
            copula="independent",
            degree=2,
            q=1.0,
        )
        s1, st = compute_sobol_indices(pce, nvar=2)
        assert s1[0] > s1[1] * 3


# -----------------------------------------------------------------------------
# plot_sobol_indices
# -----------------------------------------------------------------------------


class TestPlotSobolIndices:
    def test_saves_file(self, tmp_path):
        first = np.array([0.2, 0.5, 0.1])
        total = np.array([0.3, 0.7, 0.2])
        varnames = ["a", "b", "c"]

        out = tmp_path / "sobol.png"
        plot_sobol_indices(first, total, varnames, save=True, filename=str(out))
        assert out.exists()


# -----------------------------------------------------------------------------
# plot_training_quality
# -----------------------------------------------------------------------------


class TestPlotTrainingQuality:
    def test_scatter_distribution_metrics_with_model_var(
        self, tmp_path, ds_with_model_var
    ):
        # Build arrays
        stochastic_vars = ["ABL_height", "wind_veer"]
        model_vars = ["k_b"]
        input_array, output_array, varnames, kk_values, n_feature = (
            build_input_output_arrays(
                ds_with_model_var, stochastic_vars, model_vars, "model_bias_cap"
            )
        )
        nvar = len(varnames)

        # Fit PCE quickly
        pce = construct_PCE_ot(
            input_array=input_array,
            output_array=output_array,
            marginals=["kernel"] * nvar,
            copula="independent",
            degree=2,
            q=1.0,
        )
        metamodel = pce.getMetaModel()

        # plot_training_quality expects:
        # - value_of_interest: shape (sample, ntimes)
        # - input_variable_array_physical: (ntimes, n_feature) (we can slice the first ntimes rows)
        ntimes = ds_with_model_var.sizes["wind_farm"]

        plot_training_quality(
            database=ds_with_model_var,
            varnames=stochastic_vars,
            value_of_interest=ds_with_model_var["model_bias_cap"],
            kk_values=kk_values,
            input_variable_array_physical=input_array[:ntimes, :n_feature],
            n_feature=n_feature,
            nvar=nvar,
            PCE_metamodel=metamodel,
            save=True,
            plot_options={
                "scatter": True,
                "distribution": True,
                "metrics": ["RMSE", "R2"],
            },
            seed=123,
            output_dir=tmp_path,
        )

        assert (tmp_path / "PCE_training_scatter.png").exists()
        assert (tmp_path / "PCE_training_distribution.png").exists()
        assert (tmp_path / "PCE_training_metrics.png").exists()

    def test_no_model_var_special_scatter_branch(self, tmp_path, ds_no_model_var):
        # No model vars -> nvar == n_feature in call to plot_training_quality
        stochastic_vars = ["ABL_height", "wind_veer"]
        model_vars = []

        input_array, output_array, varnames, kk_values, n_feature = (
            build_input_output_arrays(
                ds_no_model_var, stochastic_vars, model_vars, "ref_power_cap"
            )
        )
        nvar = n_feature

        # PCE on deterministic target (mean over sample) is fine
        pce = construct_PCE_ot(
            input_array=input_array,
            output_array=output_array,
            marginals=["kernel"] * nvar,
            copula="independent",
            degree=2,
            q=1.0,
        )
        metamodel = pce.getMetaModel()

        # value_of_interest still passed as (sample, ntimes) per pce_utils docstring,
        # which triggers the special no-model-var scatter branch.
        plot_training_quality(
            database=ds_no_model_var,
            varnames=stochastic_vars,
            value_of_interest=ds_no_model_var["ref_power_cap"],
            kk_values=None,
            input_variable_array_physical=input_array,  # (ntimes, n_feature)
            n_feature=n_feature,
            nvar=nvar,
            PCE_metamodel=metamodel,
            save=True,
            plot_options={"scatter": True, "distribution": False, "metrics": []},
            seed=0,
            output_dir=tmp_path,
        )

        assert (tmp_path / "PCE_training_scatter_no_modelvar.png").exists()


# -----------------------------------------------------------------------------
# run_pce_sensitivity
# -----------------------------------------------------------------------------


class TestRunPCESensitivity:
    def test_end_to_end_with_model_coeff(self, tmp_path, ds_with_model_var):
        results = run_pce_sensitivity(
            database=ds_with_model_var,
            feature_names=["ABL_height", "wind_veer"],
            pce_config={
                "degree": 2,
                "marginals": "kernel",
                "copula": "independent",
                "q": 1.0,
                "model_coeff_name": "k_b",
                "plot_options": {"scatter": True, "distribution": False, "metrics": []},
            },
            output_dir=tmp_path,
        )

        assert results["model_coeff_name"] == "k_b"
        assert results["value_of_interest"] == "model_bias_cap"
        assert results["varnames"] == ["ABL_height", "wind_veer", "k_b"]
        assert set(results["sobol_first"].keys()) == set(results["varnames"])
        assert set(results["sobol_total"].keys()) == set(results["varnames"])

        # Outputs
        assert (tmp_path / "pce_sobol_indices_k_b.png").exists()
        assert (tmp_path / "pce_sobol_indices_k_b.csv").exists()

    def test_end_to_end_no_model_coeff(self, tmp_path, ds_no_model_var):
        results = run_pce_sensitivity(
            database=ds_no_model_var,
            feature_names=["ABL_height", "wind_veer"],
            pce_config={
                "degree": 2,
                "marginals": "kernel",
                "copula": "independent",
                "q": 1.0,
                "model_coeff_name": None,
                "plot_options": {"scatter": True, "distribution": False, "metrics": []},
            },
            output_dir=tmp_path,
        )

        assert results["model_coeff_name"] is None
        assert results["value_of_interest"] == "ref_power_cap"
        assert results["varnames"] == ["ABL_height", "wind_veer"]

        assert (tmp_path / "pce_sobol_indices_no_modelvar.png").exists()
        assert (tmp_path / "pce_sobol_indices_no_modelvar.csv").exists()

    def test_model_coeff_name_string_none_is_handled(self, tmp_path, ds_no_model_var):
        results = run_pce_sensitivity(
            database=ds_no_model_var,
            feature_names=["ABL_height", "wind_veer"],
            pce_config={
                "degree": 2,
                "marginals": "kernel",
                "copula": "independent",
                "q": 1.0,
                "model_coeff_name": "None",  # string form
                "plot_options": {"scatter": True, "distribution": False, "metrics": []},
            },
            output_dir=tmp_path,
        )
        assert results["model_coeff_name"] is None
        assert results["value_of_interest"] == "ref_power_cap"

    def test_csv_format(self, tmp_path, ds_with_model_var):
        run_pce_sensitivity(
            database=ds_with_model_var,
            feature_names=["ABL_height", "wind_veer"],
            pce_config={
                "degree": 2,
                "marginals": "kernel",
                "copula": "independent",
                "q": 1.0,
                "model_coeff_name": "k_b",
            },
            output_dir=tmp_path,
        )

        csv_path = tmp_path / "pce_sobol_indices_k_b.csv"
        df = pd.read_csv(csv_path)
        assert list(df.columns) == ["Var Names", "First_Order", "Total_Order"]
        assert set(df["Var Names"]) == {"ABL_height", "wind_veer", "k_b"}


if __name__ == "__main__":
    import pytest

    pytest.main([__file__, "-v", "--tb=short"])
