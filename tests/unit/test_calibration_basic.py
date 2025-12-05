# tests/unit/test_calibration_basic.py
import numpy as np

from wifa_uq.postprocessing.calibration.basic_calibration import (
    MinBiasCalibrator,
    DefaultParams,
    LocalParameterPredictor,
)


def test_min_bias_calibrator_finds_index(tiny_bias_db):
    cal = MinBiasCalibrator(tiny_bias_db)
    cal.fit()
    assert isinstance(cal.best_idx_, int)
    assert cal.best_idx_ >= 0
    assert "k_b" in cal.best_params_
    assert "ss_alpha" in cal.best_params_


def test_default_params_uses_param_defaults_dict(tiny_bias_db):
    # ensure param_defaults is a dict (conftest already does this)
    cal = DefaultParams(tiny_bias_db)
    cal.fit()
    assert isinstance(cal.best_idx_, int)
    assert cal.best_params_
    assert "k_b" in cal.best_params_


def test_local_parameter_predictor_core(tiny_bias_db):
    features = ["ABL_height", "wind_veer", "lapse_rate"]
    cal = LocalParameterPredictor(tiny_bias_db, feature_names=features)
    cal.fit()

    assert cal.optimal_indices_ is not None
    assert len(cal.optimal_indices_) == len(tiny_bias_db.case_index)

    # Build some new X for prediction (just reuse training X)
    base_df = tiny_bias_db.isel(sample=0).to_dataframe().reset_index()
    X = base_df[features]
    preds = cal.predict(X)

    assert list(preds.columns) == cal.swept_params
    assert len(preds) == len(X)

