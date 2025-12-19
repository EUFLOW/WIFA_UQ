import numpy as np
from pathlib import Path
import wifa_uq.postprocessing.physics_insights as physics_mod
from wifa_uq import workflow


def test_run_error_prediction_returns_none_when_disabled(tmp_path: Path, tiny_bias_db):
    cfg = {
        "error_prediction": {
            "run": False,
            "features": ["ABL_height"],
            "model": "Linear",
            "model_params": {},
        },
        "sensitivity_analysis": {
            "run_observation_sensitivity": False,
            "run_bias_sensitivity": False,
        },
        "physics_insights": {"run": False},
    }
    out = workflow._run_error_prediction(cfg, tiny_bias_db, tmp_path)
    assert out == (None, None, None)


def test_run_error_prediction_runs_cv_and_saves_outputs(
    monkeypatch, tmp_path: Path, tiny_bias_db
):
    cfg = {
        "error_prediction": {
            "run": True,
            "features": ["ABL_height", "wind_veer", "lapse_rate"],
            "model": "Linear",
            "model_params": {"method": "ols"},
            "calibrator": "MinBiasCalibrator",
            "bias_predictor": "BiasPredictor",
            "cross_validation": {
                "run": True,
                "splitting_mode": "kfold_shuffled",
                "n_splits": 2,
            },
        },
        "sensitivity_analysis": {
            "run_observation_sensitivity": False,
            "run_bias_sensitivity": False,
        },
        "physics_insights": {"run": False},
    }

    # mock run_cross_validation to avoid heavy work
    cv_df = __import__("pandas").DataFrame(
        {"rmse": [0.1, 0.2], "r2": [0.9, 0.8], "mae": [0.05, 0.06]}
    )
    monkeypatch.setattr(
        workflow,
        "run_cross_validation",
        lambda **kwargs: (cv_df, [np.array([1.0])], [np.array([1.1])]),
    )

    out = workflow._run_error_prediction(cfg, tiny_bias_db, tmp_path)
    assert (tmp_path / "cv_results.csv").exists()
    assert (tmp_path / "predictions.npz").exists()
    assert out[0] is not None  # cv_df


def test_run_error_prediction_runs_observation_sensitivity(
    monkeypatch, tmp_path: Path, tiny_bias_db
):
    cfg = {
        "error_prediction": {
            "run": False,
            "features": ["ABL_height"],
            "model": "Linear",
            "model_params": {},
        },
        "sensitivity_analysis": {
            "run_observation_sensitivity": True,
            "run_bias_sensitivity": False,
            "method": "auto",
            "pce_config": {"degree": 2},
        },
        "physics_insights": {"run": False},
    }

    called = {}
    monkeypatch.setattr(
        workflow, "run_observation_sensitivity", lambda **kwargs: called.update(kwargs)
    )
    workflow._run_error_prediction(cfg, tiny_bias_db, tmp_path)
    assert called["features_list"] == ["ABL_height"]


def test_run_error_prediction_physics_insights_global(
    monkeypatch, tmp_path: Path, tiny_bias_db
):
    cfg = {
        "error_prediction": {
            "run": True,
            "features": ["ABL_height", "wind_veer", "lapse_rate"],
            "model": "Linear",
            "model_params": {"method": "ols"},
            "calibrator": "MinBiasCalibrator",
            "bias_predictor": "BiasPredictor",
            "cross_validation": {
                "run": True,
                "splitting_mode": "kfold_shuffled",
                "n_splits": 2,
            },
        },
        "sensitivity_analysis": {
            "run_observation_sensitivity": False,
            "run_bias_sensitivity": False,
        },
        "physics_insights": {"run": True, "partial_dependence": {"enabled": False}},
    }

    # CV mocked
    cv_df = __import__("pandas").DataFrame({"rmse": [0.1], "r2": [0.9], "mae": [0.05]})
    monkeypatch.setattr(
        workflow,
        "run_cross_validation",
        lambda **kwargs: (cv_df, [np.array([0.0])], [np.array([0.0])]),
    )

    # Spy on run_physics_insights
    called = {}
    monkeypatch.setattr(
        physics_mod, "run_physics_insights", lambda **kwargs: called.update(kwargs)
    )

    workflow._run_error_prediction(cfg, tiny_bias_db, tmp_path)
    assert (tmp_path / "physics_insights").exists()
    assert "database" in called
    assert "fitted_model" in called
    assert "y_bias" in called  # ensures global y_bias_all branch executed
