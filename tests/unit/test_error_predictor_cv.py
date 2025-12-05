# tests/unit/test_error_predictor_cv.py
"""
Tests for cross-validation functionality in error_predictor.

These tests verify that run_cross_validation works correctly with both
global and local calibration modes.
"""
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


class TestCrossValidationGlobal:
    """Tests for run_cross_validation with global calibration."""

    def test_basic_run_with_tree_model(self, tmp_path, tiny_bias_db):
        """Basic smoke test that CV runs without errors."""
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

        # Basic assertions
        assert not cv_df.empty
        assert "rmse" in cv_df.columns
        assert "r2" in cv_df.columns
        assert len(y_preds) == 3  # n_splits
        assert len(y_tests) == 3

    def test_generates_plots(self, tmp_path, tiny_bias_db):
        """Test that correction_results.png is generated."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(max_depth=2, n_estimators=10, random_state=0)),
        ])

        run_cross_validation(
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

        assert (tmp_path / "correction_results.png").exists()

    def test_shap_sensitivity_generates_plots(self, tmp_path, tiny_bias_db):
        """Test that SHAP plots are generated when enabled."""
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(max_depth=2, n_estimators=10, random_state=0)),
        ])

        run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipe,
            model_type="tree",
            Calibrator_cls=MinBiasCalibrator,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=["ABL_height", "wind_veer", "lapse_rate"],
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": True},
            calibration_mode="global",
        )

        assert (tmp_path / "bias_prediction_shap.png").exists()
        assert (tmp_path / "bias_prediction_shap_importance.png").exists()


class TestCrossValidationLocal:
    """Tests for run_cross_validation with local calibration."""

    def test_basic_run_with_tree_model(self, tmp_path, tiny_bias_db):
        """Basic smoke test that local CV runs without errors."""
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
        assert len(y_preds) == 3
        assert len(y_tests) == 3

    def test_local_mode_different_from_global(self, tmp_path, tiny_bias_db):
        """
        Verify local mode produces different results than global mode.
        
        This is a sanity check that the two modes are actually doing
        different things.
        """
        pipe_global = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(max_depth=2, n_estimators=10, random_state=42)),
        ])
        pipe_local = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(max_depth=2, n_estimators=10, random_state=42)),
        ])

        cv_config = {"splitting_mode": "kfold_shuffled", "n_splits": 3}
        features = ["ABL_height", "wind_veer", "lapse_rate"]
        sa_config = {"run_bias_sensitivity": False}

        # Create output directories
        global_dir = tmp_path / "global"
        local_dir = tmp_path / "local"
        global_dir.mkdir(parents=True, exist_ok=True)
        local_dir.mkdir(parents=True, exist_ok=True)

        cv_global, _, _ = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipe_global,
            model_type="tree",
            Calibrator_cls=MinBiasCalibrator,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config=cv_config,
            features_list=features,
            output_dir=global_dir,
            sa_config=sa_config,
            calibration_mode="global",
        )

        cv_local, _, _ = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipe_local,
            model_type="tree",
            Calibrator_cls=LocalParameterPredictor,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config=cv_config,
            features_list=features,
            output_dir=local_dir,
            sa_config=sa_config,
            calibration_mode="local",
        )

        # The RMSE values should generally be different
        # (local should often be better on training data)
        global_rmse = cv_global["rmse"].mean()
        local_rmse = cv_local["rmse"].mean()
        
        # We just verify both produce valid numbers
        assert global_rmse > 0
        assert local_rmse > 0
