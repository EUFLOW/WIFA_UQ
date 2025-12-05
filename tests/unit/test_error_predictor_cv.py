# tests/unit/test_error_predictor_cv.py
from pathlib import Path

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from wifa_uq.postprocessing.error_predictor.error_predictor import (
    run_cross_validation,
    BiasPredictor,
    MainPipeline,
)
from wifa_uq.postprocessing.calibration.basic_calibration import (
    MinBiasCalibrator,
    LocalParameterPredictor,
)


def test_run_cross_validation_global_tree(tmp_path, tiny_bias_db):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(max_depth=2, n_estimators=10, random_state=0)),
    ])

    cv_df, y_preds, y_tests = run_cross_validation(
        xr_data=tiny_bias_db,
        ML_pipeline=pipe,
        model_type="tree",
        Calibrator_cls=MinBiasCalibrator,
        BiasPredictor_cls=BiasPredictor,
        MainPipeline_cls=MainPipeline,
        cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
        features_list=["ABL_height", "wind_veer", "lapse_rate"],
        output_dir=tmp_path,
        sa_config={"run_bias_sensitivity": False},
        calibration_mode="global",
    )

    assert not cv_df.empty
    assert "rmse" in cv_df.columns
    assert len(y_preds) == 3
    assert len(y_tests) == 3


def test_run_cross_validation_local_tree(tmp_path, tiny_bias_db):
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(max_depth=2, n_estimators=10, random_state=1)),
    ])

    cv_df, y_preds, y_tests = run_cross_validation(
        xr_data=tiny_bias_db,
        ML_pipeline=pipe,
        model_type="tree",
        Calibrator_cls=LocalParameterPredictor,
        BiasPredictor_cls=BiasPredictor,
        MainPipeline_cls=MainPipeline,
        cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
        features_list=["ABL_height", "wind_veer", "lapse_rate"],
        output_dir=tmp_path,
        sa_config={"run_bias_sensitivity": False},
        calibration_mode="local",
    )

    assert not cv_df.empty
    assert "rmse" in cv_df.columns
    # local mode must *not* crash and must return folds
    assert len(y_preds) == 3
    assert len(y_tests) == 3

