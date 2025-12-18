# tests/unit/test_error_predictor_models.py
"""
Comprehensive tests for all error prediction model types.

Tests cover:
- Model building via factory function
- Fit/predict functionality
- Cross-validation integration
- Sensitivity analysis compatibility
- Both global and local calibration modes
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score
import xgboost as xgb

from wifa_uq.postprocessing.error_predictor.error_predictor import (
    PCERegressor,
    LinearRegressor,
    SIRPolynomialRegressor,
    BiasPredictor,
    MainPipeline,
    run_cross_validation,
)
from wifa_uq.workflow import build_predictor_pipeline  # Correct import location
from wifa_uq.postprocessing.calibration.basic_calibration import (
    MinBiasCalibrator,
    DefaultParams,
    LocalParameterPredictor,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def synthetic_regression_data():
    """Generate synthetic regression data for testing."""
    np.random.seed(42)
    n_samples = 200
    n_features = 3

    X = np.random.randn(n_samples, n_features)
    # Linear relationship with some noise
    y = 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * X[:, 2] + 0.1 * np.random.randn(n_samples)

    feature_names = ["feature_1", "feature_2", "feature_3"]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X, y, X_df, feature_names


@pytest.fixture
def small_synthetic_data():
    """Smaller dataset for PCE (which has feature limits)."""
    np.random.seed(42)
    n_samples = 100
    n_features = 2

    X = np.random.randn(n_samples, n_features)
    y = X[:, 0] ** 2 + 0.5 * X[:, 1] + 0.1 * np.random.randn(n_samples)

    feature_names = ["feat_a", "feat_b"]
    X_df = pd.DataFrame(X, columns=feature_names)

    return X, y, X_df, feature_names


# =============================================================================
# Test: build_predictor_pipeline Factory Function
# =============================================================================


class TestBuildPredictorPipeline:
    """Tests for the build_predictor_pipeline factory function."""

    def test_build_xgb_pipeline(self):
        """XGB pipeline should return Pipeline with tree model_type."""
        pipeline, model_type = build_predictor_pipeline("XGB")

        assert model_type == "tree"
        assert isinstance(pipeline, Pipeline)
        assert "scaler" in pipeline.named_steps
        assert "model" in pipeline.named_steps
        assert isinstance(pipeline.named_steps["model"], xgb.XGBRegressor)

    def test_build_xgb_with_params(self):
        """XGB pipeline should respect custom parameters."""
        params = {"max_depth": 5, "n_estimators": 100, "learning_rate": 0.05}
        pipeline, model_type = build_predictor_pipeline("XGB", params)

        model = pipeline.named_steps["model"]
        assert model.max_depth == 5
        assert model.n_estimators == 100
        assert model.learning_rate == 0.05

    def test_build_sir_pipeline(self):
        """SIRPolynomial pipeline should return correct type."""
        pipeline, model_type = build_predictor_pipeline("SIRPolynomial")

        assert model_type == "sir"
        assert isinstance(pipeline, SIRPolynomialRegressor)

    def test_build_pce_pipeline(self):
        """PCE pipeline should return correct type."""
        pipeline, model_type = build_predictor_pipeline("PCE")

        assert model_type == "pce"
        assert isinstance(pipeline, PCERegressor)

    def test_build_pce_with_params(self):
        """PCE pipeline should respect custom parameters."""
        params = {"degree": 3, "marginals": "uniform", "q": 0.8}
        pipeline, model_type = build_predictor_pipeline("PCE", params)

        assert pipeline.degree == 3
        assert pipeline.marginals == "uniform"
        assert pipeline.q == 0.8

    def test_build_linear_pipeline(self):
        """Linear pipeline should return correct type."""
        pipeline, model_type = build_predictor_pipeline("Linear")

        assert model_type == "linear"
        assert isinstance(pipeline, LinearRegressor)

    def test_build_linear_with_method(self):
        """Linear pipeline should respect method parameter."""
        params = {"method": "ridge", "alpha": 0.5}
        pipeline, model_type = build_predictor_pipeline("Linear", params)

        assert pipeline.method == "ridge"
        assert pipeline.alpha == 0.5

    def test_unknown_model_raises_error(self):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            build_predictor_pipeline("UnknownModel")


# =============================================================================
# Test: Individual Model Classes
# =============================================================================


class TestXGBModel:
    """Tests for XGBoost model via Pipeline."""

    def test_fit_predict(self, synthetic_regression_data):
        """XGB should fit and predict."""
        X, y, X_df, _ = synthetic_regression_data

        pipeline, _ = build_predictor_pipeline(
            "XGB", {"n_estimators": 50, "max_depth": 3}
        )
        pipeline.fit(X, y)
        y_pred = pipeline.predict(X)

        assert len(y_pred) == len(y)
        assert r2_score(y, y_pred) > 0.5  # Should fit reasonably well

    def test_handles_dataframe_input(self, synthetic_regression_data):
        """XGB should handle DataFrame input."""
        X, y, X_df, _ = synthetic_regression_data

        pipeline, _ = build_predictor_pipeline("XGB", {"n_estimators": 50})
        pipeline.fit(X_df, y)
        y_pred = pipeline.predict(X_df)

        assert len(y_pred) == len(y)


class TestSIRPolynomialModel:
    """Tests for SIRPolynomialRegressor."""

    def test_fit_predict(self, synthetic_regression_data):
        """SIR should fit and predict."""
        X, y, _, _ = synthetic_regression_data

        model = SIRPolynomialRegressor(n_directions=1, degree=2)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert len(y_pred) == len(y)
        assert r2_score(y, y_pred) > 0.3  # Should capture some relationship

    def test_feature_importance(self, synthetic_regression_data):
        """SIR should provide feature importance."""
        X, y, _, feature_names = synthetic_regression_data

        model = SIRPolynomialRegressor(n_directions=1, degree=2)
        model.fit(X, y)
        importance = model.get_feature_importance(feature_names)

        assert len(importance) == len(feature_names)
        assert all(importance >= 0)  # Absolute values

    def test_not_fitted_error(self):
        """Should raise error if predict called before fit."""
        model = SIRPolynomialRegressor()
        X = np.random.randn(10, 3)

        with pytest.raises(Exception):  # NotFittedError
            model.predict(X)


class TestPCEModel:
    """Tests for PCERegressor."""

    def test_fit_predict(self, small_synthetic_data):
        """PCE should fit and predict."""
        X, y, _, _ = small_synthetic_data

        model = PCERegressor(degree=3, max_features=5, allow_high_dim=False)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert len(y_pred) == len(y)
        assert r2_score(y, y_pred) > 0.5

    def test_refuses_high_dim_by_default(self):
        """PCE should refuse >max_features inputs by default."""
        X = np.random.randn(50, 6)  # 6 features > default max of 5
        y = X[:, 0]

        model = PCERegressor(max_features=5, allow_high_dim=False)

        with pytest.raises(ValueError, match="refused to run"):
            model.fit(X, y)

    def test_allows_high_dim_when_enabled(self):
        """PCE should work with >5 features when allowed."""
        X = np.random.randn(100, 6)
        y = X[:, 0] + X[:, 1]

        model = PCERegressor(degree=2, max_features=5, allow_high_dim=True)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert len(y_pred) == 100

    def test_different_marginals(self, small_synthetic_data):
        """PCE should work with different marginal specifications."""
        X, y, _, _ = small_synthetic_data

        for marginals in ["kernel", "uniform", "normal"]:
            model = PCERegressor(degree=3, marginals=marginals)
            model.fit(X, y)
            y_pred = model.predict(X)
            assert len(y_pred) == len(y)


class TestLinearModel:
    """Tests for LinearRegressor."""

    def test_ols_fit_predict(self, synthetic_regression_data):
        """OLS should fit and predict."""
        X, y, _, _ = synthetic_regression_data

        model = LinearRegressor(method="ols")
        model.fit(X, y)
        y_pred = model.predict(X)

        assert len(y_pred) == len(y)
        assert r2_score(y, y_pred) > 0.9  # Linear data should fit well

    def test_ridge_fit_predict(self, synthetic_regression_data):
        """Ridge should fit and predict."""
        X, y, _, _ = synthetic_regression_data

        model = LinearRegressor(method="ridge", alpha=1.0)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert len(y_pred) == len(y)

    def test_lasso_fit_predict(self, synthetic_regression_data):
        """Lasso should fit and predict."""
        X, y, _, _ = synthetic_regression_data

        model = LinearRegressor(method="lasso", alpha=0.1)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert len(y_pred) == len(y)

    def test_elasticnet_fit_predict(self, synthetic_regression_data):
        """ElasticNet should fit and predict."""
        X, y, _, _ = synthetic_regression_data

        model = LinearRegressor(method="elasticnet", alpha=0.1, l1_ratio=0.5)
        model.fit(X, y)
        y_pred = model.predict(X)

        assert len(y_pred) == len(y)

    def test_feature_importance(self, synthetic_regression_data):
        """Linear should provide feature importance (coefficients)."""
        X, y, _, feature_names = synthetic_regression_data

        model = LinearRegressor(method="ridge")
        model.fit(X, y)
        importance = model.get_feature_importance(feature_names)

        assert len(importance) == len(feature_names)

    def test_unknown_method_raises_error(self):
        """Unknown method should raise ValueError."""
        model = LinearRegressor(method="unknown")
        X = np.random.randn(10, 3)
        y = np.random.randn(10)

        with pytest.raises(ValueError, match="Unknown method"):
            model.fit(X, y)


# =============================================================================
# Test: BiasPredictor Wrapper
# =============================================================================


class TestBiasPredictor:
    """Tests for BiasPredictor class."""

    @pytest.mark.parametrize("model_name", ["XGB", "SIRPolynomial", "Linear"])
    def test_fit_predict_with_different_models(
        self, synthetic_regression_data, model_name
    ):
        """BiasPredictor should work with all model types."""
        X, y, X_df, _ = synthetic_regression_data

        pipeline, _ = build_predictor_pipeline(
            model_name, {"n_estimators": 50} if model_name == "XGB" else {}
        )

        predictor = BiasPredictor(pipeline)
        predictor.fit(X_df, y)
        y_pred = predictor.predict(X_df)

        assert len(y_pred) == len(y)


# =============================================================================
# Test: Cross-Validation with Different Models
# =============================================================================


class TestCrossValidationModels:
    """Tests for run_cross_validation with different model types."""

    @pytest.mark.parametrize(
        "model_name,model_type",
        [
            ("XGB", "tree"),
            ("SIRPolynomial", "sir"),
            ("Linear", "linear"),
        ],
    )
    def test_cv_with_global_calibration(
        self, tmp_path, tiny_bias_db, model_name, model_type
    ):
        """Cross-validation should work with all models and global calibration."""
        if model_name == "XGB":
            params = {"max_depth": 2, "n_estimators": 10, "random_state": 42}
        elif model_name == "Linear":
            params = {"method": "ridge", "alpha": 1.0}
        else:
            params = {}

        pipeline, _ = build_predictor_pipeline(model_name, params)

        cv_df, y_preds, y_tests = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
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
        assert "r2" in cv_df.columns
        assert len(y_preds) == 3
        assert len(y_tests) == 3

    @pytest.mark.parametrize(
        "model_name,model_type",
        [
            ("XGB", "tree"),
            ("Linear", "linear"),
        ],
    )
    def test_cv_with_local_calibration(
        self, tmp_path, tiny_bias_db, model_name, model_type
    ):
        """Cross-validation should work with local calibration."""
        if model_name == "XGB":
            params = {"max_depth": 2, "n_estimators": 10, "random_state": 42}
        else:
            params = {"method": "ridge"}

        pipeline, _ = build_predictor_pipeline(model_name, params)

        cv_df, y_preds, y_tests = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
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
        assert len(y_preds) == 3

    def test_cv_with_pce_small_features(self, tmp_path, tiny_bias_db):
        """Cross-validation should work with PCE when features <= max_features."""
        # Use only 2 features to stay within PCE limits
        features = ["ABL_height", "wind_veer"]

        pipeline, model_type = build_predictor_pipeline(
            "PCE", {"degree": 3, "max_features": 5, "allow_high_dim": False}
        )

        cv_df, y_preds, y_tests = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=MinBiasCalibrator,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=features,
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": False},
            calibration_mode="global",
        )

        assert not cv_df.empty
        assert "rmse" in cv_df.columns


# =============================================================================
# Test: Sensitivity Analysis Compatibility
# =============================================================================


class TestSensitivityAnalysisCompatibility:
    """Tests for sensitivity analysis with different model types."""

    def test_shap_sensitivity_with_xgb(self, tmp_path, tiny_bias_db):
        """SHAP sensitivity should work with XGB (tree model)."""
        pipeline, model_type = build_predictor_pipeline(
            "XGB", {"max_depth": 2, "n_estimators": 10, "random_state": 42}
        )

        cv_df, _, _ = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=MinBiasCalibrator,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=["ABL_height", "wind_veer", "lapse_rate"],
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": True},
            calibration_mode="global",
        )

        # Check SHAP plots were generated
        assert (tmp_path / "bias_prediction_shap.png").exists()
        assert (tmp_path / "bias_prediction_shap_importance.png").exists()

    def test_sir_sensitivity(self, tmp_path, tiny_bias_db):
        """SIR sensitivity should generate importance plot."""
        pipeline, model_type = build_predictor_pipeline("SIRPolynomial")

        cv_df, _, _ = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=MinBiasCalibrator,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=["ABL_height", "wind_veer", "lapse_rate"],
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": True},
            calibration_mode="global",
        )

        # Check SIR importance plot was generated
        assert (tmp_path / "bias_prediction_sir_importance.png").exists()

    def test_linear_sensitivity(self, tmp_path, tiny_bias_db):
        """Linear sensitivity should generate importance plot."""
        pipeline, model_type = build_predictor_pipeline(
            "Linear", {"method": "ridge", "alpha": 1.0}
        )

        cv_df, _, _ = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=MinBiasCalibrator,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=["ABL_height", "wind_veer", "lapse_rate"],
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": True},
            calibration_mode="global",
        )

        # Check linear importance plot was generated
        assert (tmp_path / "bias_prediction_linear_importance.png").exists()


# =============================================================================
# Test: Calibrator Combinations
# =============================================================================


class TestCalibratorModelCombinations:
    """Tests for different calibrator + model combinations."""

    @pytest.mark.parametrize(
        "calibrator_cls,calibration_mode",
        [
            (MinBiasCalibrator, "global"),
            (DefaultParams, "global"),
        ],
    )
    @pytest.mark.parametrize("model_name", ["XGB", "Linear"])
    def test_global_calibrators_with_models(
        self, tmp_path, tiny_bias_db, calibrator_cls, calibration_mode, model_name
    ):
        """All global calibrators should work with all models."""
        params = {"max_depth": 2, "n_estimators": 10} if model_name == "XGB" else {}
        pipeline, model_type = build_predictor_pipeline(model_name, params)

        cv_df, y_preds, y_tests = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=calibrator_cls,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=["ABL_height", "wind_veer", "lapse_rate"],
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": False},
            calibration_mode=calibration_mode,
        )

        assert not cv_df.empty
        assert all(cv_df["rmse"] >= 0)

    @pytest.mark.parametrize("local_regressor", ["Linear", "Ridge", "RandomForest"])
    def test_local_calibrator_with_different_regressors(
        self, tmp_path, tiny_bias_db, local_regressor
    ):
        """LocalParameterPredictor should work with different internal regressors."""
        pipeline, model_type = build_predictor_pipeline(
            "XGB", {"max_depth": 2, "n_estimators": 10, "random_state": 42}
        )

        local_params = {"alpha": 1.0} if local_regressor == "Ridge" else {}

        cv_df, y_preds, y_tests = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=LocalParameterPredictor,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=["ABL_height", "wind_veer", "lapse_rate"],
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": False},
            calibration_mode="local",
            local_regressor=local_regressor,
            local_regressor_params=local_params,
        )

        assert not cv_df.empty


# =============================================================================
# Test: Output File Generation
# =============================================================================


class TestOutputFileGeneration:
    """Tests verifying all expected output files are created."""

    def test_correction_results_plot_generated(self, tmp_path, tiny_bias_db):
        """Should always generate correction_results.png."""
        pipeline, model_type = build_predictor_pipeline(
            "XGB", {"max_depth": 2, "n_estimators": 10}
        )

        run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
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

    def test_local_parameter_prediction_plot(self, tmp_path, tiny_bias_db):
        """Local calibration should generate parameter prediction plot."""
        pipeline, model_type = build_predictor_pipeline(
            "XGB", {"max_depth": 2, "n_estimators": 10}
        )

        run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=LocalParameterPredictor,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
            features_list=["ABL_height", "wind_veer", "lapse_rate"],
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": False},
            calibration_mode="local",
        )

        assert (tmp_path / "local_parameter_prediction.png").exists()


# =============================================================================
# Test: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_feature(self, tmp_path, tiny_bias_db):
        """Should work with a single feature."""
        pipeline, model_type = build_predictor_pipeline(
            "XGB", {"max_depth": 2, "n_estimators": 10}
        )

        cv_df, _, _ = run_cross_validation(
            xr_data=tiny_bias_db,
            ML_pipeline=pipeline,
            model_type=model_type,
            Calibrator_cls=MinBiasCalibrator,
            BiasPredictor_cls=BiasPredictor,
            MainPipeline_cls=MainPipeline,
            cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 2},
            features_list=["ABL_height"],  # Single feature
            output_dir=tmp_path,
            sa_config={"run_bias_sensitivity": False},
            calibration_mode="global",
        )

        assert not cv_df.empty

    def test_empty_model_params(self):
        """Should handle empty model params gracefully."""
        pipeline, model_type = build_predictor_pipeline("XGB", {})
        assert pipeline is not None

        pipeline, model_type = build_predictor_pipeline("XGB", None)
        assert pipeline is not None

    def test_reproducibility_with_seed(self, tmp_path, tiny_bias_db):
        """Results should be reproducible with same random state."""
        params = {"max_depth": 2, "n_estimators": 10, "random_state": 42}

        # Run twice
        results = []
        for _ in range(2):
            pipeline, model_type = build_predictor_pipeline("XGB", params)
            cv_df, _, _ = run_cross_validation(
                xr_data=tiny_bias_db,
                ML_pipeline=pipeline,
                model_type=model_type,
                Calibrator_cls=MinBiasCalibrator,
                BiasPredictor_cls=BiasPredictor,
                MainPipeline_cls=MainPipeline,
                cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 3},
                features_list=["ABL_height", "wind_veer"],
                output_dir=tmp_path,
                sa_config={"run_bias_sensitivity": False},
                calibration_mode="global",
            )
            results.append(cv_df["rmse"].values)

        np.testing.assert_array_almost_equal(results[0], results[1])
