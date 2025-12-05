# tests/unit/test_calibration_basic.py
"""
Tests for basic calibration classes.
"""

from wifa_uq.postprocessing.calibration.basic_calibration import (
    MinBiasCalibrator,
    DefaultParams,
    LocalParameterPredictor,
)


class TestMinBiasCalibrator:
    """Tests for MinBiasCalibrator."""

    def test_finds_best_index(self, tiny_bias_db):
        """Should find the sample index with minimum total absolute bias."""
        cal = MinBiasCalibrator(tiny_bias_db)
        cal.fit()

        assert isinstance(cal.best_idx_, int)
        assert cal.best_idx_ >= 0
        assert cal.best_idx_ < len(tiny_bias_db.sample)

    def test_extracts_best_params(self, tiny_bias_db):
        """Should extract parameter values at the best index."""
        cal = MinBiasCalibrator(tiny_bias_db)
        cal.fit()

        assert "k_b" in cal.best_params_
        assert "ss_alpha" in cal.best_params_
        assert isinstance(cal.best_params_["k_b"], float)
        assert isinstance(cal.best_params_["ss_alpha"], float)


class TestDefaultParams:
    """Tests for DefaultParams calibrator."""

    def test_uses_param_defaults_dict(self, tiny_bias_db):
        """Should find sample closest to default parameter values."""
        cal = DefaultParams(tiny_bias_db)
        cal.fit()

        assert isinstance(cal.best_idx_, int)
        assert cal.best_params_
        assert "k_b" in cal.best_params_

    def test_returns_values_near_defaults(self, tiny_bias_db):
        """The best params should be close to the specified defaults."""
        cal = DefaultParams(tiny_bias_db)
        cal.fit()

        # From fixture: param_defaults = {"k_b": 0.04, "ss_alpha": 0.875}
        assert abs(cal.best_params_["k_b"] - 0.04) < 0.02


class TestLocalParameterPredictor:
    """Tests for LocalParameterPredictor."""

    def test_core_functionality(self, tiny_bias_db):
        """Test basic fit/predict cycle."""
        features = ["ABL_height", "wind_veer", "lapse_rate"]
        cal = LocalParameterPredictor(tiny_bias_db, feature_names=features)
        cal.fit()

        assert cal.optimal_indices_ is not None
        assert len(cal.optimal_indices_) == len(tiny_bias_db.case_index)

    def test_predict_returns_dataframe(self, tiny_bias_db):
        """predict() should return DataFrame with swept param columns."""
        features = ["ABL_height", "wind_veer", "lapse_rate"]
        cal = LocalParameterPredictor(tiny_bias_db, feature_names=features)
        cal.fit()

        # Build some new X for prediction (reuse training X)
        base_df = tiny_bias_db.isel(sample=0).to_dataframe().reset_index()
        X = base_df[features]
        preds = cal.predict(X)

        assert list(preds.columns) == cal.swept_params
        assert len(preds) == len(X)

    def test_get_optimal_indices(self, tiny_bias_db):
        """get_optimal_indices() should return array of sample indices."""
        features = ["ABL_height", "wind_veer", "lapse_rate"]
        cal = LocalParameterPredictor(tiny_bias_db, feature_names=features)
        cal.fit()

        indices = cal.get_optimal_indices()

        assert len(indices) == len(tiny_bias_db.case_index)
        assert all(0 <= idx < len(tiny_bias_db.sample) for idx in indices)
