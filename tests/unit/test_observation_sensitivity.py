# tests/unit/test_observation_sensitivity.py
"""
Unit tests for run_observation_sensitivity in
wifa_uq.postprocessing.error_predictor.error_predictor

Covers:
- method="auto" dispatch for model_type: tree, sir, pce
- method="shap": saves observation_sensitivity_shap.png (with SHAP mocked)
- method="sir": saves observation_sensitivity_sir.png and observation_sensitivity_sir_shadow.png
- method="pce_sobol": calls pce_utils.run_pce_sensitivity (mocked) with expected args
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from wifa_uq.postprocessing.error_predictor.error_predictor import (
    run_observation_sensitivity,
    SIRPolynomialRegressor,
)


@pytest.fixture(autouse=True)
def _no_show(monkeypatch):
    """Prevent any plt.show() calls from blocking test runs."""
    monkeypatch.setattr(plt, "show", lambda *args, **kwargs: None)


@pytest.fixture
def tiny_bias_db_nonconstant_ref(tiny_bias_db):
    """
    Make a copy of tiny_bias_db where ref_power_cap varies across case_index,
    so SIR has >1 unique y value.
    """
    db = tiny_bias_db.copy(deep=True)
    # ref_power_cap dims are (sample, case_index) in tiny_bias_db fixture
    # introduce a simple ramp across case_index (same for all samples)
    ramp = np.linspace(0.45, 0.55, db.sizes["case_index"])
    db["ref_power_cap"] = db["ref_power_cap"] * 0.0 + ramp[None, :]
    return db


def test_auto_tree_uses_shap_and_saves_plot(monkeypatch, tmp_path: Path, tiny_bias_db):
    """
    method="auto" + model_type="tree" -> uses SHAP branch and writes plot.
    Mock SHAP TreeExplainer and summary_plot.
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=10, random_state=0)),
        ]
    )

    import wifa_uq.postprocessing.error_predictor.error_predictor as ep

    class DummyExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, X):
            return np.zeros((X.shape[0], X.shape[1]))

    called = {"summary_plot": 0}
    monkeypatch.setattr(ep.shap, "TreeExplainer", DummyExplainer)
    monkeypatch.setattr(
        ep.shap,
        "summary_plot",
        lambda *args, **kwargs: called.__setitem__(
            "summary_plot", called["summary_plot"] + 1
        ),
    )

    run_observation_sensitivity(
        database=tiny_bias_db,
        features_list=["ABL_height", "wind_veer", "lapse_rate"],
        ml_pipeline=pipe,
        model_type="tree",
        output_dir=tmp_path,
        method="auto",
    )

    assert called["summary_plot"] == 1
    assert (tmp_path / "observation_sensitivity_shap.png").exists()


def test_explicit_shap_method_saves_plot(monkeypatch, tmp_path: Path, tiny_bias_db):
    """
    method="shap" should save observation_sensitivity_shap.png
    """
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=5, random_state=0)),
        ]
    )

    import wifa_uq.postprocessing.error_predictor.error_predictor as ep

    class DummyExplainer:
        def __init__(self, model):
            self.model = model

        def shap_values(self, X):
            return np.zeros((X.shape[0], X.shape[1]))

    monkeypatch.setattr(ep.shap, "TreeExplainer", DummyExplainer)
    monkeypatch.setattr(ep.shap, "summary_plot", lambda *args, **kwargs: None)

    run_observation_sensitivity(
        database=tiny_bias_db,
        features_list=["ABL_height", "wind_veer"],
        ml_pipeline=pipe,
        model_type="tree",
        output_dir=tmp_path,
        method="shap",
    )

    assert (tmp_path / "observation_sensitivity_shap.png").exists()


def test_auto_sir_saves_importance_and_shadow_plots(
    tmp_path: Path, tiny_bias_db_nonconstant_ref
):
    """
    method="auto" + model_type="sir" -> uses SIR branch.
    Requires non-constant y (ref_power_cap).
    """
    sir_model = SIRPolynomialRegressor(n_directions=1, degree=1)

    run_observation_sensitivity(
        database=tiny_bias_db_nonconstant_ref,
        features_list=["ABL_height", "wind_veer", "lapse_rate"],
        ml_pipeline=sir_model,
        model_type="sir",
        output_dir=tmp_path,
        method="auto",
    )

    assert (tmp_path / "observation_sensitivity_sir.png").exists()
    assert (tmp_path / "observation_sensitivity_sir_shadow.png").exists()


def test_explicit_sir_method_saves_plots(tmp_path: Path, tiny_bias_db_nonconstant_ref):
    """
    method="sir" explicitly should save the same two plots.
    """
    sir_model = SIRPolynomialRegressor(n_directions=1, degree=2)

    run_observation_sensitivity(
        database=tiny_bias_db_nonconstant_ref,
        features_list=["ABL_height", "wind_veer"],
        ml_pipeline=sir_model,
        model_type="sir",
        output_dir=tmp_path,
        method="sir",
    )

    assert (tmp_path / "observation_sensitivity_sir.png").exists()
    assert (tmp_path / "observation_sensitivity_sir_shadow.png").exists()


def test_auto_pce_dispatch_calls_run_pce_sensitivity(
    monkeypatch, tmp_path: Path, tiny_bias_db
):
    """
    method="auto" + model_type="pce" -> dispatches to pce_sobol branch,
    which imports and calls pce_utils.run_pce_sensitivity. Patch that function
    in its defining module.
    """
    import wifa_uq.postprocessing.PCE_tool.pce_utils as pce_utils

    called = {}

    def fake_run_pce_sensitivity(database, feature_names, pce_config, output_dir):
        called["database"] = database
        called["feature_names"] = feature_names
        called["pce_config"] = pce_config
        called["output_dir"] = Path(output_dir)

    monkeypatch.setattr(pce_utils, "run_pce_sensitivity", fake_run_pce_sensitivity)

    dummy_pipeline = object()

    run_observation_sensitivity(
        database=tiny_bias_db,
        features_list=["ABL_height", "wind_veer"],
        ml_pipeline=dummy_pipeline,
        model_type="pce",
        output_dir=tmp_path,
        method="auto",
        pce_config={
            "degree": 2,
            "marginals": "kernel",
            "copula": "independent",
            "q": 1.0,
        },
    )

    assert called["database"] is tiny_bias_db
    assert called["feature_names"] == ["ABL_height", "wind_veer"]
    assert called["pce_config"]["degree"] == 2
    assert called["output_dir"] == tmp_path


def test_unknown_method_raises(tmp_path: Path, tiny_bias_db):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor())])

    with pytest.raises(ValueError, match="Unknown method"):
        run_observation_sensitivity(
            database=tiny_bias_db,
            features_list=["ABL_height"],
            ml_pipeline=pipe,
            model_type="tree",
            output_dir=tmp_path,
            method="not_a_method",
        )


def test_auto_unknown_model_type_raises(tmp_path: Path, tiny_bias_db):
    pipe = Pipeline([("scaler", StandardScaler()), ("model", RandomForestRegressor())])

    with pytest.raises(ValueError, match="Unknown model_type"):
        run_observation_sensitivity(
            database=tiny_bias_db,
            features_list=["ABL_height"],
            ml_pipeline=pipe,
            model_type="weird",
            output_dir=tmp_path,
            method="auto",
        )
