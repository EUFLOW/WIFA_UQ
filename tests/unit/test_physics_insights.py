# tests/unit/test_physics_insights.py
"""
Tests for physics insights module.
"""

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

from wifa_uq.postprocessing.physics_insights import (
    run_physics_insights,
    PhysicsInsightsReport,
)
from wifa_uq.postprocessing.physics_insights.physics_insights import (
    analyze_partial_dependence,
    analyze_regimes,
    interpret_pd_direction,
    interpret_parameter_relationship,
    describe_regime,
)


class TestInterpretationHelpers:
    """Tests for physical interpretation helper functions."""

    def test_interpret_pd_direction_increases(self):
        """Test interpretation for increasing bias."""
        result = interpret_pd_direction("ABL_height", "increases")
        assert "underestimates" in result.lower()
        assert "stable" in result.lower()

    def test_interpret_pd_direction_decreases(self):
        """Test interpretation for decreasing bias."""
        result = interpret_pd_direction("ABL_height", "decreases")
        assert "overestimates" in result.lower()

    def test_interpret_pd_direction_unknown_feature(self):
        """Should handle unknown features gracefully."""
        result = interpret_pd_direction("unknown_feature", "increases")
        assert "unknown_feature" in result

    def test_interpret_parameter_relationship_positive(self):
        """Test parameter relationship interpretation."""
        result = interpret_parameter_relationship("k_b", "ABL_height", 0.7)
        assert "increase" in result.lower()
        assert "wake" in result.lower() or "k_b" in result.lower()

    def test_interpret_parameter_relationship_weak(self):
        """Weak correlations should be noted."""
        result = interpret_parameter_relationship("k_b", "ABL_height", 0.1)
        assert "weak" in result.lower()

    def test_describe_regime_stable(self):
        """Test regime description for stable conditions."""
        centroids = {"ABL_height": 900, "wind_veer": 0.008}
        feature_stats = {
            "ABL_height": (500, 200),  # mean=500, std=200
            "wind_veer": (0.005, 0.002),
        }
        result = describe_regime(centroids, feature_stats, mean_bias=0.05)
        # High ABL_height (900 is 2 std above mean) should be detected
        assert "stable" in result.lower() or "overestim" in result.lower()


class TestPartialDependence:
    """Tests for partial dependence analysis."""

    def test_basic_analysis(self, tmp_path):
        """Test that PD analysis runs and produces results."""
        # Create synthetic data with known relationship
        np.random.seed(42)
        n_samples = 200
        X = pd.DataFrame(
            {
                "ABL_height": np.random.uniform(200, 1000, n_samples),
                "wind_veer": np.random.uniform(0, 0.01, n_samples),
            }
        )
        # Bias increases with ABL_height
        y = 0.001 * X["ABL_height"] + np.random.normal(0, 0.1, n_samples)

        # Fit model
        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=20, random_state=42)),
            ]
        )
        model.fit(X, y)

        # Run analysis
        results = analyze_partial_dependence(
            model=model,
            X=X,
            features=["ABL_height", "wind_veer"],
            output_dir=tmp_path,
        )

        assert len(results) == 2
        assert (tmp_path / "partial_dependence.png").exists()

        # Check that ABL_height shows increasing trend (since we built it that way)
        abl_result = next(r for r in results if r.feature == "ABL_height")
        assert abl_result.bias_direction in ["increases", "non-monotonic"]
        assert abl_result.effect_magnitude > 0

    def test_generates_interpretations(self, tmp_path):
        """Test that physical interpretations are generated."""
        np.random.seed(42)
        n_samples = 100
        X = pd.DataFrame(
            {
                "ABL_height": np.random.uniform(200, 1000, n_samples),
            }
        )
        y = np.random.normal(0, 1, n_samples)

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        results = analyze_partial_dependence(
            model=model,
            X=X,
            features=["ABL_height"],
            output_dir=tmp_path,
        )

        assert results[0].physical_interpretation != ""


class TestRegimeAnalysis:
    """Tests for error regime identification."""

    def test_identifies_clusters(self, tmp_path):
        """Test that regime analysis identifies distinct clusters."""
        np.random.seed(42)
        n_samples = 200

        # Create data with two distinct regimes
        X1 = pd.DataFrame(
            {
                "ABL_height": np.random.normal(300, 50, n_samples // 2),
                "wind_veer": np.random.normal(0.002, 0.001, n_samples // 2),
            }
        )
        X2 = pd.DataFrame(
            {
                "ABL_height": np.random.normal(800, 50, n_samples // 2),
                "wind_veer": np.random.normal(0.008, 0.001, n_samples // 2),
            }
        )
        X = pd.concat([X1, X2], ignore_index=True)

        # Bias depends on regime
        y_bias = np.concatenate(
            [
                np.random.normal(-0.2, 0.05, n_samples // 2),  # Regime 1: negative bias
                np.random.normal(0.2, 0.05, n_samples // 2),  # Regime 2: positive bias
            ]
        )

        results = analyze_regimes(
            X=X,
            y_bias=y_bias,
            features=["ABL_height", "wind_veer"],
            output_dir=tmp_path,
            n_clusters=2,
            bias_percentile=50,  # Use all data
        )

        assert len(results) == 2
        assert (tmp_path / "error_regimes.png").exists()

        # Check that regimes have different characteristics
        biases = [r.mean_bias for r in results]
        assert abs(biases[0] - biases[1]) > 0.1  # Should be distinct

    def test_handles_small_dataset(self, tmp_path):
        """Should handle datasets too small for requested clusters."""
        X = pd.DataFrame(
            {
                "ABL_height": [300, 400, 500],
                "wind_veer": [0.001, 0.002, 0.003],
            }
        )
        y_bias = np.array([0.1, 0.2, 0.3])

        # Request 5 clusters but only 3 data points
        results = analyze_regimes(
            X=X,
            y_bias=y_bias,
            features=["ABL_height", "wind_veer"],
            output_dir=tmp_path,
            n_clusters=5,
            bias_percentile=0,  # Use all data
        )

        # Should reduce clusters automatically
        assert len(results) <= 3


class TestPhysicsInsightsReport:
    """Tests for the report data class."""

    def test_to_dict(self):
        """Test conversion to dictionary."""
        from wifa_uq.postprocessing.physics_insights.physics_insights import (
            PartialDependenceResult,
        )

        report = PhysicsInsightsReport(
            partial_dependence=[
                PartialDependenceResult(
                    feature="ABL_height",
                    grid_values=np.array([100, 200, 300]),
                    pd_values=np.array([0.1, 0.2, 0.3]),
                    bias_direction="increases",
                    effect_magnitude=0.2,
                    physical_interpretation="Test interpretation",
                )
            ],
            summary="Test summary",
        )

        d = report.to_dict()

        assert "partial_dependence" in d
        assert len(d["partial_dependence"]) == 1
        assert d["partial_dependence"][0]["feature"] == "ABL_height"
        assert d["summary"] == "Test summary"

    def test_to_markdown(self):
        """Test markdown generation."""
        from wifa_uq.postprocessing.physics_insights.physics_insights import (
            PartialDependenceResult,
            RegimeResult,
        )

        report = PhysicsInsightsReport(
            partial_dependence=[
                PartialDependenceResult(
                    feature="ABL_height",
                    grid_values=np.array([100, 200, 300]),
                    pd_values=np.array([0.1, 0.2, 0.3]),
                    bias_direction="increases",
                    effect_magnitude=0.2,
                    physical_interpretation="Model underestimates in stable conditions",
                )
            ],
            regimes=[
                RegimeResult(
                    regime_id=0,
                    n_cases=50,
                    mean_bias=0.15,
                    feature_centroids={"ABL_height": 800},
                    description="Stable nocturnal",
                )
            ],
            summary="Key finding: bias increases with stability",
        )

        md = report.to_markdown()

        assert "# Physics Insights Report" in md
        assert "ABL_height" in md
        assert "increases" in md
        assert "Stable nocturnal" in md
        assert "Key finding" in md


class TestRunPhysicsInsights:
    """Integration tests for full physics insights pipeline."""

    def test_full_pipeline(self, tmp_path, tiny_bias_db):
        """Test complete physics insights run."""
        from wifa_uq.postprocessing.calibration import LocalParameterPredictor

        # Prepare data
        features = ["ABL_height", "wind_veer", "lapse_rate"]

        # Fit a simple model
        X_df = tiny_bias_db.isel(sample=0).to_dataframe().reset_index()
        X = X_df[features]
        y = tiny_bias_db["model_bias_cap"].isel(sample=0).values

        model = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", RandomForestRegressor(n_estimators=20, random_state=42)),
            ]
        )
        model.fit(X, y)

        # Fit calibrator
        calibrator = LocalParameterPredictor(tiny_bias_db, feature_names=features)
        calibrator.fit()

        # Run insights
        report = run_physics_insights(
            database=tiny_bias_db,
            fitted_model=model,
            calibrator=calibrator,
            features_list=features,
            y_bias=y,
            output_dir=tmp_path,
            config={
                "partial_dependence": {"enabled": True},
                "interactions": {"enabled": True},
                "regime_analysis": {"enabled": True, "n_clusters": 2},
                "parameter_relationships": {"enabled": True},
            },
        )

        # Check outputs exist
        assert (tmp_path / "partial_dependence.png").exists()
        assert (tmp_path / "error_regimes.png").exists()
        assert (tmp_path / "physics_insights_report.md").exists()
        assert (tmp_path / "physics_insights.json").exists()

        # Check report content
        assert len(report.partial_dependence) == 3
        assert report.summary != ""

    def test_respects_config_disabling(self, tmp_path, tiny_bias_db):
        """Test that config can disable analyses."""
        features = ["ABL_height", "wind_veer", "lapse_rate"]

        X_df = tiny_bias_db.isel(sample=0).to_dataframe().reset_index()
        X = X_df[features]
        y = tiny_bias_db["model_bias_cap"].isel(sample=0).values

        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)

        report = run_physics_insights(
            database=tiny_bias_db,
            fitted_model=model,
            calibrator=None,
            features_list=features,
            y_bias=y,
            output_dir=tmp_path,
            config={
                "partial_dependence": {"enabled": False},
                "interactions": {"enabled": False},
                "regime_analysis": {"enabled": False},
                "parameter_relationships": {"enabled": False},
            },
        )

        # All should be empty
        assert len(report.partial_dependence) == 0
        assert len(report.interactions) == 0
        assert len(report.regimes) == 0
        assert len(report.parameter_relationships) == 0
