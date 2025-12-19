import numpy as np
import pandas as pd
import pytest
import matplotlib

# Use non-interactive backend for testing to avoid popup windows
matplotlib.use("Agg")

from wifa_uq.postprocessing.error_predictor.error_predictor import (
    MainPipeline,
    BiasPredictor,
    plot_multi_farm_cv_metrics,
    plot_farm_wise_predictions,
    plot_generalization_matrix,
)
from wifa_uq.postprocessing.calibration.basic_calibration import MinBiasCalibrator


class TestMainPipelineUtilities:
    """Tests for MainPipeline internal utilities like feature extraction."""

    def test_extract_features_cleaning(self, tiny_bias_db):
        """Test that _extract_features cleans string brackets and handles types."""
        # Create a mock calibrator
        cal = MinBiasCalibrator(tiny_bias_db)
        predictor = BiasPredictor(None)

        # Add a "dirty" column with brackets often found in NetCDF string representations
        df = tiny_bias_db.isel(sample=0).to_dataframe().reset_index()
        df["dirty_feat"] = "[0.123]"

        features = ["ABL_height", "dirty_feat"]
        pipeline = MainPipeline(cal, predictor, features_list=features)

        cleaned_df = pipeline._extract_features(df)

        assert cleaned_df["dirty_feat"].dtype == float
        assert cleaned_df["dirty_feat"].iloc[0] == 0.123
        assert cleaned_df.shape[1] == 2


class TestPlottingUtilities:
    """Tests for CV and Multi-farm plotting functions."""

    @pytest.fixture
    def mock_cv_results(self):
        return pd.DataFrame(
            {
                "rmse": [0.1, 0.12, 0.08],
                "r2": [0.8, 0.75, 0.85],
                "mae": [0.07, 0.09, 0.06],
            }
        )

    def test_plot_multi_farm_cv_metrics(self, mock_cv_results, tmp_path):
        labels = ["GroupA", "GroupB", "GroupC"]
        plot_multi_farm_cv_metrics(mock_cv_results, labels, tmp_path)

        assert (tmp_path / "cv_fold_metrics.png").exists()
        assert (tmp_path / "cv_fold_heatmap.png").exists()
        assert (tmp_path / "cv_metrics_boxplot.png").exists()

    def test_plot_farm_wise_predictions(self, tmp_path):
        y_tests = [np.array([1, 2]), np.array([3, 4])]
        y_preds = [np.array([1.1, 1.9]), np.array([2.9, 4.2])]
        fold_labels = ["Fold1", "Fold2"]
        fold_farms = [np.array(["Farm1", "Farm1"]), np.array(["Farm2", "Farm2"])]

        plot_farm_wise_predictions(y_tests, y_preds, fold_labels, fold_farms, tmp_path)

        assert (tmp_path / "cv_predictions_by_fold.png").exists()
        assert (tmp_path / "cv_predictions_per_fold.png").exists()

    def test_plot_generalization_matrix(self, mock_cv_results, tmp_path):
        labels = ["GroupA", "GroupB", "GroupC"]
        plot_generalization_matrix(mock_cv_results, labels, tmp_path)
        assert (tmp_path / "cv_generalization_summary.png").exists()
