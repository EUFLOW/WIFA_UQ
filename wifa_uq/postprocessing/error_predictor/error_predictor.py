import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_absolute_error, r2_score
from wifa_uq.postprocessing.PCE_tool.pce_utils import construct_PCE_ot
from wifa_uq.postprocessing.calibration import (
    MinBiasCalibrator,
    #  DefaultParams,
    # LocalParameterPredictor,
)


try:
    from sliced import SlicedInverseRegression
except ImportError as e:
    print(
        "Error: 'sliced' package not found. SIRPolynomialRegressor will not be available."
    )
    raise e


"""
This script contains:
- SIRPolynomialRegressor class (NEW)
- Calibrator classes
- BiasPredictor class
- MainPipeline class
- Cross validation routine
- SHAP/SIR sensitivity analysis functions
- Multi-farm CV visualization functions (NEW)
"""


class PCERegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible regressor that wraps the OpenTURNS-based PCE
    from PCE_tool.

    Safety guard:
      - By default, refuses to run if the number of input features > max_features
        unless allow_high_dim=True is explicitly set.
    """

    def __init__(
        self,
        degree=5,
        marginals="kernel",
        copula="independent",
        q=1.0,
        max_features=5,  # safety limit on input dimension
        allow_high_dim=False,  # must be True to allow > max_features
    ):
        self.degree = degree
        self.marginals = marginals
        self.copula = copula
        self.q = q
        self.max_features = max_features
        self.allow_high_dim = allow_high_dim

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y).ravel()

        n_features = X.shape[1]

        # --- Safety check on dimensionality ---
        if n_features > self.max_features and not self.allow_high_dim:
            raise ValueError(
                f"PCERegressor refused to run: number of input variables = {n_features}, "
                f"which exceeds the default safety limit of {self.max_features}. "
                f"Set allow_high_dim=True or increase max_features to override."
            )

        marginals = [self.marginals] * n_features

        # Construct PCE using your existing helper
        self.pce_result_ = construct_PCE_ot(
            input_array=X,
            output_array=y,
            marginals=marginals,
            copula=self.copula,
            degree=self.degree,
            q=self.q,
        )
        self.metamodel_ = self.pce_result_.getMetaModel()
        return self

    def predict(self, X):
        if not hasattr(self, "metamodel_"):
            raise NotFittedError("PCERegressor instance is not fitted yet.")
        X = np.asarray(X)
        preds = np.zeros(X.shape[0])
        for i, xi in enumerate(X):
            preds[i] = self.metamodel_(xi)[0]
        return preds


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    A simple linear regression wrapper with optional regularization.

    Supports: 'ols' (ordinary least squares), 'ridge', 'lasso', 'elasticnet'
    """

    def __init__(self, method="ols", alpha=1.0, l1_ratio=0.5):
        self.method = method
        self.alpha = alpha
        self.l1_ratio = l1_ratio  # Only used for elasticnet

    def fit(self, X, y):
        from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet

        if self.method == "ols":
            self.model_ = LinearRegression()
        elif self.method == "ridge":
            self.model_ = Ridge(alpha=self.alpha)
        elif self.method == "lasso":
            self.model_ = Lasso(alpha=self.alpha)
        elif self.method == "elasticnet":
            self.model_ = ElasticNet(alpha=self.alpha, l1_ratio=self.l1_ratio)
        else:
            raise ValueError(
                f"Unknown method '{self.method}'. Use 'ols', 'ridge', 'lasso', or 'elasticnet'."
            )

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X)
        self.model_.fit(X_scaled, y)
        return self

    def predict(self, X):
        if not hasattr(self, "model_"):
            raise NotFittedError("LinearRegressor is not fitted yet.")
        X_scaled = self.scaler_.transform(X)
        return self.model_.predict(X_scaled)

    def get_feature_importance(self, feature_names):
        """Return absolute coefficients as feature importance."""
        if not hasattr(self, "model_"):
            raise NotFittedError("LinearRegressor is not fitted yet.")
        return pd.Series(np.abs(self.model_.coef_), index=feature_names)


## ------------------------------------------------------------------ ##
## NEW REGRESSOR CLASS
## ------------------------------------------------------------------ ##
class SIRPolynomialRegressor(BaseEstimator, RegressorMixin):
    """
    A scikit-learn compatible regressor that first applies SIR for
    dimension reduction and then fits a polynomial regression
    on the reduced dimension(s).
    """

    def __init__(self, n_directions=1, degree=2):
        if SlicedInverseRegression is None:
            raise ImportError(
                "The 'sliced' package is required to use SIRPolynomialRegressor."
            )
        self.n_directions = n_directions
        self.degree = degree

    def fit(self, X, y):
        # 1. Standard Scaler
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X, y)

        # 2. Sliced Inverse Regression
        self.sir_ = SlicedInverseRegression(n_directions=self.n_directions)
        # Sliced package expects y as a 1D array
        y_ravel = np.ravel(y)
        X_sir = self.sir_.fit_transform(X_scaled, y_ravel)

        # Store the directions (feature importance)
        # We take the absolute value for importance ranking
        self.sir_directions_ = np.abs(self.sir_.directions_[0, :])

        # 3. Polynomial Regression
        self.poly_reg_ = Pipeline(
            [
                ("poly", PolynomialFeatures(degree=self.degree, include_bias=False)),
                ("lin_reg", LinearRegression()),
            ]
        )
        self.poly_reg_.fit(X_sir, y)

        return self

    def predict(self, X):
        if not hasattr(self, "scaler_"):
            raise NotFittedError(
                "This SIRPolynomialRegressor instance is not fitted yet."
            )

        X_scaled = self.scaler_.transform(X)
        X_sir = self.sir_.transform(X_scaled)
        return self.poly_reg_.predict(X_sir)

    def get_feature_importance(self, feature_names):
        if not hasattr(self, "sir_directions_"):
            raise NotFittedError(
                "This SIRPolynomialRegressor instance is not fitted yet."
            )

        return pd.Series(self.sir_directions_, index=feature_names)


## ------------------------------------------------------------------ ##


class BiasPredictor:
    """
    Predict bias as a function of features and parameter samples
    """

    def __init__(self, regressor_pipeline):
        self.pipeline = regressor_pipeline

    def fit(self, X_train, y_train):
        # print('shape of X and Y training: ',X_train.shape, y_train.shape)
        self.pipeline.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        y_pred = self.pipeline.predict(X_test)
        return y_pred


class MainPipeline:
    """
    Main pipeline that combines calibration and bias prediction.

    Supports two calibration modes:
    1. Global calibration (MinBiasCalibrator, DefaultParams):
       - Single parameter set for all cases
       - calibrator.best_idx_ gives the sample index

    2. Local calibration (LocalParameterPredictor):
       - Different optimal parameters per case
       - calibrator.get_optimal_indices() gives per-case indices
    """

    def __init__(
        self,
        calibrator,
        bias_predictor,
        features_list: list,
        calibration_mode: str = "global",
    ):
        """
        Args:
            calibrator: Calibrator instance (already initialized with dataset_train)
            bias_predictor: BiasPredictor instance
            features_list: List of feature names to use for bias prediction
            calibration_mode: "global" or "local"
        """
        self.calibrator = calibrator
        self.bias_predictor = bias_predictor
        self.features_list = features_list
        self.calibration_mode = calibration_mode

        if not self.features_list:
            raise ValueError("features_list cannot be empty.")

    def fit(self, dataset_train, dataset_test):
        """
        Fit the calibrator and bias predictor.

        Returns:
            X_test, y_test, idxs (for compatibility with cross-validation)
        """
        # 1. Fit calibrator
        self.calibrator.fit()

        if self.calibration_mode == "global":
            return self._fit_global(dataset_train, dataset_test)
        elif self.calibration_mode == "local":
            return self._fit_local(dataset_train, dataset_test)
        else:
            raise ValueError(f"Unknown calibration_mode: {self.calibration_mode}")

    def _fit_global(self, dataset_train, dataset_test):
        """Fit using global calibration (single parameter set)."""
        idxs = self.calibrator.best_idx_

        # Select the calibrated sample for train and test
        dataset_train_cal = dataset_train.sel(sample=idxs)
        dataset_test_cal = dataset_test.sel(sample=idxs)

        # Prepare features
        X_train_df = dataset_train_cal.to_dataframe().reset_index()
        X_test_df = dataset_test_cal.to_dataframe().reset_index()

        X_train = self._extract_features(X_train_df)
        X_test = self._extract_features(X_test_df)

        y_train = dataset_train_cal["model_bias_cap"].values
        y_test = dataset_test_cal["model_bias_cap"].values

        # Fit bias predictor
        self.bias_predictor.fit(X_train, y_train)

        # Store for predict()
        self.X_test_ = X_test

        return X_test, y_test, idxs

    def _fit_local(self, dataset_train, dataset_test):
        """Fit using local calibration (per-case optimal parameters)."""
        # 1. Get per-case optimal sample indices from the local calibrator
        train_optimal_indices = self.calibrator.get_optimal_indices()
        n_train_cases = len(dataset_train.case_index)

        # 2. Build training feature matrix from sample=0
        #    (features do not depend on sampled parameters)
        train_base = dataset_train.isel(sample=0)
        train_df = train_base.to_dataframe().reset_index()
        X_train = self._extract_features(train_df)

        # 3. Build training targets: bias at the optimal sample for each case
        y_train = np.zeros(n_train_cases)
        for case_idx, sample_idx in enumerate(train_optimal_indices):
            y_train[case_idx] = float(
                dataset_train["model_bias_cap"]
                .isel(case_index=case_idx, sample=sample_idx)
                .values
            )

        # 4. Test features (also from sample=0)
        X_test_df = dataset_test.isel(sample=0).to_dataframe().reset_index()
        X_test_features = self._extract_features(X_test_df)

        # 5. Predict optimal parameters for test cases,
        #    then find the closest sampled parameter set in the database
        predicted_params = self.calibrator.predict(X_test_features)
        test_optimal_indices = self._find_closest_samples(
            dataset_test, predicted_params
        )

        # 6. Build test targets: bias at the chosen sample for each test case
        n_test_cases = len(dataset_test.case_index)
        y_test = np.zeros(n_test_cases)
        for case_idx, sample_idx in enumerate(test_optimal_indices):
            y_test[case_idx] = float(
                dataset_test["model_bias_cap"]
                .isel(case_index=case_idx, sample=sample_idx)
                .values
            )

        # 7. Fit the bias predictor on per-case data
        self.bias_predictor.fit(X_train, y_train)

        # Store X_test_ so .predict() can be called without args
        self.X_test_ = X_test_features

        # Return in the same shape run_cross_validation expects
        return X_test_features, y_test, test_optimal_indices

    def _extract_features(self, df):
        """Extract and clean features from dataframe."""
        try:
            X = df[self.features_list].copy()
        except KeyError as e:
            print(f"Error: Feature not found in dataset: {e}")
            print(f"Available columns: {list(df.columns)}")
            raise

        # Clean string-like columns
        for col in X.columns:
            if X[col].dtype == "object":
                if X[col].dropna().empty:
                    continue
                first_item = X[col].dropna().iloc[0]
                if isinstance(first_item, str):
                    X[col] = X[col].str.replace(r"[\[\]]", "", regex=True).astype(float)
                else:
                    X[col] = X[col].astype(float)

        return X

    def _find_closest_samples(self, dataset, predicted_params):
        """
        Find the sample index closest to predicted parameters for each case.

        Args:
            dataset: xarray Dataset with 'sample' dimension
            predicted_params: DataFrame with predicted optimal parameters

        Returns:
            Array of sample indices (one per case)

        Raises:
            ValueError: If no valid parameters found for distance calculation
        """
        n_cases = len(predicted_params)
        n_samples = len(dataset.sample)
        swept_params = self.calibrator.swept_params

        # Validation 1: Check swept_params is not empty
        if not swept_params:
            raise ValueError(
                "No swept parameters defined. Cannot find closest samples. "
                "Check that the database has 'swept_params' in attrs or that "
                "parameters were correctly inferred."
            )

        # Validation 2: Check which parameters are actually available
        available_params = []
        missing_in_dataset = []
        missing_in_predictions = []

        for param_name in swept_params:
            in_dataset = param_name in dataset.coords
            in_predictions = param_name in predicted_params.columns

            if in_dataset and in_predictions:
                available_params.append(param_name)
            elif not in_dataset:
                missing_in_dataset.append(param_name)
            elif not in_predictions:
                missing_in_predictions.append(param_name)

        # Validation 3: Ensure we have at least one parameter to use
        if not available_params:
            raise ValueError(
                f"No valid parameters for distance calculation.\n"
                f"  Swept params: {swept_params}\n"
                f"  Missing in dataset.coords: {missing_in_dataset}\n"
                f"  Missing in predicted_params: {missing_in_predictions}"
            )

        # Warn about partial matches (some params missing)
        if missing_in_dataset or missing_in_predictions:
            import warnings

            warnings.warn(
                f"Some swept parameters unavailable for distance calculation:\n"
                f"  Using: {available_params}\n"
                f"  Missing in dataset: {missing_in_dataset}\n"
                f"  Missing in predictions: {missing_in_predictions}",
                UserWarning,
            )

        # Calculate distances using only available parameters
        closest_indices = np.zeros(n_cases, dtype=int)

        for case_idx in range(n_cases):
            target_params = predicted_params.iloc[case_idx]

            distances = np.zeros(n_samples)
            for param_name in available_params:
                sample_values = dataset.coords[param_name].values
                target_value = target_params[param_name]

                # Normalize by parameter range to handle different scales
                param_range = sample_values.max() - sample_values.min()
                if param_range > 0:
                    normalized_diff = (sample_values - target_value) / param_range
                else:
                    # All samples have same value for this param
                    normalized_diff = np.zeros_like(sample_values)

                distances += normalized_diff**2

            closest_indices[case_idx] = int(np.argmin(distances))

        return closest_indices

    def predict(self, X=None):
        """Predict bias for test data."""
        if X is None:
            X = self.X_test_
        return self.bias_predictor.predict(X)


def compute_metrics(y_true, bias_samples, pw, ref, data_driv=None):
    mse = ((y_true - bias_samples) ** 2).mean()
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, bias_samples)
    r2 = r2_score(y_true, bias_samples)

    if pw is not None and ref is not None and data_driv is None:
        pw_bias = np.mean(pw - ref)
        pw_bias_corrected = np.mean((pw - bias_samples) - ref)
    else:
        pw_bias = None
        pw_bias_corrected = None

    if data_driv and ref:
        data_driv_bias = np.mean(data_driv - ref)
    else:
        data_driv_bias = None

    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2,
        "pw_bias": pw_bias,
        "pw_bias_corrected": pw_bias_corrected,
        "data_driv_bias": data_driv_bias,
    }


## ------------------------------------------------------------------ ##
## MULTI-FARM CV VISUALIZATION FUNCTIONS (NEW)
## ------------------------------------------------------------------ ##


def plot_multi_farm_cv_metrics(
    cv_results: pd.DataFrame,
    fold_labels: list,
    output_dir: Path,
    splitting_mode: str = "LeaveOneGroupOut",
):
    """
    Create visualizations for multi-farm cross-validation results.

    Shows per-fold (per-group) performance metrics to understand
    how well the model generalizes across different wind farms.

    Args:
        cv_results: DataFrame with metrics per fold (rmse, r2, mae, etc.)
        fold_labels: List of strings identifying each fold (e.g., group names left out)
        output_dir: Directory to save plots
        splitting_mode: CV splitting mode for title annotation
    """
    output_dir = Path(output_dir)
    n_folds = len(cv_results)

    # --- 1. Per-Fold Metrics Bar Chart ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        f"Cross-Validation Performance by Fold ({splitting_mode})", fontsize=14
    )

    metrics = ["rmse", "r2", "mae"]
    colors = ["#1f77b4", "#2ca02c", "#ff7f0e"]

    x = np.arange(n_folds)

    for ax, metric, color in zip(axes, metrics, colors):
        values = cv_results[metric].values
        bars = ax.bar(x, values, color=color, alpha=0.7, edgecolor="black")

        # Add value labels on bars
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f"{val:.4f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=45,
            )

        # Add mean line
        mean_val = values.mean()
        ax.axhline(
            mean_val,
            color="red",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.4f}",
        )

        ax.set_xlabel("Fold (Left-Out Group)")
        ax.set_ylabel(metric.upper())
        ax.set_title(f"{metric.upper()} per Fold")
        ax.set_xticks(x)
        ax.set_xticklabels(fold_labels, rotation=45, ha="right", fontsize=8)
        ax.legend(loc="best")
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cv_fold_metrics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved per-fold metrics plot to: {output_dir / 'cv_fold_metrics.png'}")

    # --- 2. Metrics Comparison Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))

    # Normalize metrics for heatmap (z-score within each metric)
    metrics_for_heatmap = ["rmse", "mae", "r2"]
    heatmap_data = cv_results[metrics_for_heatmap].copy()

    # For r2, higher is better; for rmse/mae, lower is better
    # Normalize so that "better" is always higher for visualization
    heatmap_normalized = heatmap_data.copy()
    heatmap_normalized["rmse"] = -heatmap_data["rmse"]  # Negate so higher = better
    heatmap_normalized["mae"] = -heatmap_data["mae"]  # Negate so higher = better

    # Create heatmap
    im = ax.imshow(heatmap_normalized.T, cmap="RdYlGn", aspect="auto")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(
        "Performance (normalized, higher = better)", rotation=270, labelpad=15
    )

    # Set ticks and labels
    ax.set_xticks(np.arange(n_folds))
    ax.set_xticklabels(fold_labels, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(metrics_for_heatmap)))
    ax.set_yticklabels([m.upper() for m in metrics_for_heatmap])

    # Add text annotations with actual values
    for i in range(len(metrics_for_heatmap)):
        for j in range(n_folds):
            ax.text(
                j,
                i,
                f"{heatmap_data.iloc[j, i]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=8,
            )

    ax.set_xlabel("Fold (Left-Out Group)")
    ax.set_title("Performance Heatmap Across CV Folds\n(Green = Better Performance)")

    plt.tight_layout()
    plt.savefig(output_dir / "cv_fold_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved heatmap plot to: {output_dir / 'cv_fold_heatmap.png'}")

    # --- 3. Box Plot Summary ---
    fig, ax = plt.subplots(figsize=(8, 6))

    # Prepare data for box plot
    metrics_data = [cv_results[m].values for m in ["rmse", "mae", "r2"]]

    bp = ax.boxplot(metrics_data, labels=["RMSE", "MAE", "R²"], patch_artist=True)

    # Color the boxes
    colors_box = ["#1f77b4", "#ff7f0e", "#2ca02c"]
    for patch, color in zip(bp["boxes"], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Add individual points
    for i, (metric_vals, color) in enumerate(zip(metrics_data, colors_box)):
        x_jitter = np.random.normal(i + 1, 0.04, size=len(metric_vals))
        ax.scatter(
            x_jitter, metric_vals, alpha=0.6, color=color, edgecolor="black", s=50
        )

    ax.set_ylabel("Metric Value")
    ax.set_title(f"Distribution of CV Metrics Across {n_folds} Folds")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "cv_metrics_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved boxplot to: {output_dir / 'cv_metrics_boxplot.png'}")


def plot_farm_wise_predictions(
    y_tests: list,
    y_preds: list,
    fold_labels: list,
    fold_farm_names: list,
    output_dir: Path,
):
    """
    Create scatter plots showing predictions vs true values, colored by farm/group.

    Args:
        y_tests: List of test targets per fold
        y_preds: List of predictions per fold
        fold_labels: List of fold identifiers (left-out groups)
        fold_farm_names: List of arrays, each containing farm names for test cases in that fold
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)

    # Combine all folds
    all_y_test = np.concatenate(y_tests)
    all_y_pred = np.concatenate(y_preds)
    all_fold_ids = np.concatenate(
        [np.full(len(y_tests[i]), fold_labels[i]) for i in range(len(y_tests))]
    )

    # Get unique fold labels for coloring
    unique_folds = np.unique(all_fold_ids)
    n_folds = len(unique_folds)

    # Create colormap
    cmap = plt.cm.get_cmap("tab10" if n_folds <= 10 else "tab20")
    colors = {fold: cmap(i / n_folds) for i, fold in enumerate(unique_folds)}

    # --- Main scatter plot colored by fold ---
    fig, ax = plt.subplots(figsize=(10, 8))

    for fold_label in unique_folds:
        mask = all_fold_ids == fold_label
        ax.scatter(
            all_y_test[mask],
            all_y_pred[mask],
            c=[colors[fold_label]],
            label=f"Left out: {fold_label}",
            alpha=0.6,
            s=30,
            edgecolor="white",
            linewidth=0.5,
        )

    # Add 1:1 line
    min_val = min(all_y_test.min(), all_y_pred.min())
    max_val = max(all_y_test.max(), all_y_pred.max())
    ax.plot(
        [min_val, max_val], [min_val, max_val], "k--", linewidth=2, label="1:1 Line"
    )

    # Calculate overall metrics
    overall_rmse = np.sqrt(np.mean((all_y_test - all_y_pred) ** 2))
    overall_r2 = r2_score(all_y_test, all_y_pred)

    ax.set_xlabel("True Bias", fontsize=12)
    ax.set_ylabel("Predicted Bias", fontsize=12)
    ax.set_title(
        f"Predictions vs True Values by Left-Out Group\n"
        f"Overall RMSE: {overall_rmse:.4f}, R²: {overall_r2:.4f}",
        fontsize=14,
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal", adjustable="box")

    plt.tight_layout()
    plt.savefig(output_dir / "cv_predictions_by_fold.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(
        f"    Saved predictions scatter plot to: {output_dir / 'cv_predictions_by_fold.png'}"
    )

    # --- Per-fold subplots ---
    n_cols = min(3, n_folds)
    n_rows = (n_folds + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    if n_folds == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)

    for idx, fold_label in enumerate(unique_folds):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]

        mask = all_fold_ids == fold_label
        y_test_fold = all_y_test[mask]
        y_pred_fold = all_y_pred[mask]

        ax.scatter(
            y_test_fold,
            y_pred_fold,
            c=[colors[fold_label]],
            alpha=0.6,
            s=30,
            edgecolor="white",
            linewidth=0.5,
        )

        # 1:1 line
        min_v = min(y_test_fold.min(), y_pred_fold.min())
        max_v = max(y_test_fold.max(), y_pred_fold.max())
        ax.plot([min_v, max_v], [min_v, max_v], "k--", linewidth=1.5)

        # Per-fold metrics
        fold_rmse = np.sqrt(np.mean((y_test_fold - y_pred_fold) ** 2))
        fold_r2 = r2_score(y_test_fold, y_pred_fold) if len(y_test_fold) > 1 else 0

        ax.set_xlabel("True Bias")
        ax.set_ylabel("Predicted Bias")
        ax.set_title(
            f"Left Out: {fold_label}\nRMSE: {fold_rmse:.4f}, R²: {fold_r2:.4f}"
        )
        ax.grid(alpha=0.3)
        ax.set_aspect("equal", adjustable="box")

    # Hide unused subplots
    for idx in range(n_folds, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    plt.suptitle("Prediction Quality per CV Fold", fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(
        output_dir / "cv_predictions_per_fold.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"    Saved per-fold predictions to: {output_dir / 'cv_predictions_per_fold.png'}"
    )


def plot_generalization_matrix(
    cv_results: pd.DataFrame, fold_labels: list, output_dir: Path
):
    """
    Create a generalization analysis visualization showing how training on
    certain farm groups affects prediction on others.

    Args:
        cv_results: DataFrame with metrics per fold
        fold_labels: List of fold identifiers (left-out groups)
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    n_folds = len(fold_labels)

    # Create summary table
    fig, ax = plt.subplots(figsize=(12, max(4, n_folds * 0.5)))

    # Prepare data for table
    table_data = []
    for i, label in enumerate(fold_labels):
        row = [
            label,
            f"{cv_results['rmse'].iloc[i]:.4f}",
            f"{cv_results['r2'].iloc[i]:.4f}",
            f"{cv_results['mae'].iloc[i]:.4f}",
        ]
        table_data.append(row)

    # Add mean row
    table_data.append(
        [
            "MEAN",
            f"{cv_results['rmse'].mean():.4f}",
            f"{cv_results['r2'].mean():.4f}",
            f"{cv_results['mae'].mean():.4f}",
        ]
    )

    # Add std row
    table_data.append(
        [
            "STD",
            f"{cv_results['rmse'].std():.4f}",
            f"{cv_results['r2'].std():.4f}",
            f"{cv_results['mae'].std():.4f}",
        ]
    )

    columns = ["Left-Out Group", "RMSE", "R²", "MAE"]

    # Hide axes
    ax.axis("off")

    # Create table
    table = ax.table(
        cellText=table_data,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Color header row
    for j, col in enumerate(columns):
        table[(0, j)].set_facecolor("#4472C4")
        table[(0, j)].set_text_props(color="white", weight="bold")

    # Color mean/std rows
    for j in range(len(columns)):
        table[(n_folds + 1, j)].set_facecolor("#E2EFDA")  # Mean row
        table[(n_folds + 2, j)].set_facecolor("#FCE4D6")  # Std row

    # Color-code RMSE cells based on value
    rmse_values = cv_results["rmse"].values
    rmse_min, rmse_max = rmse_values.min(), rmse_values.max()

    for i in range(n_folds):
        # Normalize RMSE (lower is better, so invert for color)
        if rmse_max > rmse_min:
            norm_val = (rmse_values[i] - rmse_min) / (rmse_max - rmse_min)
        else:
            norm_val = 0.5

        # Color from green (good) to red (bad)
        color = plt.cm.RdYlGn(1 - norm_val)
        table[(i + 1, 1)].set_facecolor(color)

    plt.title(
        "Cross-Validation Generalization Summary\n"
        "(Testing on each group after training on all others)",
        fontsize=14,
        pad=20,
    )

    plt.tight_layout()
    plt.savefig(
        output_dir / "cv_generalization_summary.png", dpi=150, bbox_inches="tight"
    )
    plt.close(fig)
    print(
        f"    Saved generalization summary to: {output_dir / 'cv_generalization_summary.png'}"
    )


## ------------------------------------------------------------------ ##


def run_observation_sensitivity(
    database,
    features_list,
    ml_pipeline,
    model_type,
    output_dir,
    method: str = "auto",
    pce_config: dict = None,
):
    """
    Sensitivity analysis on observations.

    Args:
        database: xarray Dataset
        features_list: List of feature names
        ml_pipeline: ML pipeline (used for shap/sir methods)
        model_type: "tree" or "sir" (used for shap/sir methods)
        output_dir: Where to save plots
        method: "auto", "shap", "sir", or "pce_sobol"
                "auto" uses shap for tree models, sir directions for sir models
        pce_config: Config dict for PCE (only used if method="pce_sobol")
    """
    print(f"--- Running Observation Sensitivity (method={method}) ---")

    # Prepare data (sample 0 = default params)
    data = database.isel(sample=0).to_dataframe().reset_index()
    X = data[features_list]
    y = data["ref_power_cap"].values  # observations

    # Determine method
    if method == "auto":
        if model_type == "tree":
            method = "shap"
        elif model_type == "sir":
            method = "sir"
        elif model_type == "pce":
            # For PCE models, default "auto" to PCE-based Sobol SA
            method = "pce_sobol"
        else:
            raise ValueError(
                f"Unknown model_type '{model_type}' for 'auto' sensitivity. "
                f"Expected one of ['tree', 'sir', 'pce']."
            )

    if method == "pce_sobol":
        # PCE-based Sobol indices
        from wifa_uq.postprocessing.PCE_tool.pce_utils import run_pce_sensitivity

        run_pce_sensitivity(
            database,
            feature_names=features_list,
            pce_config=pce_config or {},
            output_dir=output_dir,
        )

    elif method == "shap":
        # Train model, then SHAP
        ml_pipeline.fit(X, y)

        if hasattr(ml_pipeline, "named_steps"):
            model = ml_pipeline.named_steps["model"]
        else:
            model = ml_pipeline

        # Get scaled data and model for SHAP
        X_scaled = ml_pipeline.named_steps["scaler"].transform(X)
        model = ml_pipeline.named_steps["model"]

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_scaled)

        # Plot
        shap.summary_plot(shap_values, X, feature_names=features_list, show=False)
        plt.savefig(Path(output_dir) / "observation_sensitivity_shap.png", dpi=150)
        plt.close()
        print(f"    Saved SHAP plot to {output_dir}/observation_sensitivity_shap.png")

    elif method == "sir":
        # Train SIR model, use direction coefficients
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        ml_pipeline.fit(X_scaled, y)

        # Get SIR direction coefficients as importance
        directions = ml_pipeline.sir_.directions_.flatten()
        importance = np.abs(directions)

        # Identify the feature with the largest influence on the first direction
        top_idx = np.argmax(importance)
        top_feature_name = features_list[top_idx]
        print(f"    Dominant feature identified: {top_feature_name}")

        # Plot
        sorted_idx = np.argsort(importance)
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(features_list)), importance[sorted_idx])
        plt.yticks(range(len(features_list)), [features_list[i] for i in sorted_idx])
        plt.xlabel("Absolute SIR Direction Coefficient")
        plt.title("Observation Sensitivity (SIR)")
        plt.tight_layout()
        plt.savefig(Path(output_dir) / "observation_sensitivity_sir.png", dpi=150)
        plt.close()
        print(f"    Saved SIR plot to {output_dir}/observation_sensitivity_sir.png")

        # 2. Shadow Plot (Error vs. First Eigenvector, Colored by Top Feature)
        # Project data onto the first found direction
        X_projected = ml_pipeline.sir_.transform(X_scaled)
        first_component = X_projected[:, 0]

        # Get values of the top feature for coloring
        # X is likely a DataFrame here given the setup code
        color_values = X.iloc[:, top_idx].values

        plt.figure(figsize=(9, 7))
        #
        scatter = plt.scatter(
            first_component,
            y,
            c=color_values,
            cmap="viridis",
            alpha=0.7,
            edgecolor="k",
            linewidth=0.5,
        )

        cbar = plt.colorbar(scatter)
        cbar.set_label(f"Feature Value: {top_feature_name}", rotation=270, labelpad=15)

        plt.xlabel(r"Projected Input $\beta_1^T \mathbf{x}$ (1st SIR Direction)")
        plt.ylabel("Observed Error (y)")
        plt.title(f"SIR Shadow Plot\nColored by dominant feature '{top_feature_name}'")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        shadow_plot_path = Path(output_dir) / "observation_sensitivity_sir_shadow.png"
        plt.savefig(shadow_plot_path, dpi=150)
        plt.close()
        print(f"    Saved SIR shadow plot to {shadow_plot_path}")

    else:
        raise ValueError(
            f"Unknown method: {method}. Use 'auto', 'shap', 'sir', or 'pce_sobol'"
        )


def run_cross_validation(
    xr_data,
    ML_pipeline,
    model_type,
    Calibrator_cls,
    BiasPredictor_cls,
    MainPipeline_cls,
    cv_config: dict,
    features_list: list,
    output_dir: Path,
    sa_config: dict,
    calibration_mode: str = "global",
    local_regressor: str = None,
    local_regressor_params: dict = None,
):
    validation_data = xr_data[
        ["turb_rated_power", "pw_power_cap", "ref_power_cap", "case_index"]
    ]
    groups = xr_data["wind_farm"].values

    splitting_mode = cv_config.get("splitting_mode", "kfold_shuffled")

    # Track fold labels for multi-farm visualization
    fold_labels = []
    is_multi_farm = False

    if splitting_mode == "LeaveOneGroupOut":
        is_multi_farm = True
        groups = xr_data["wind_farm"].values

        groups_cfg = cv_config.get("groups")
        if groups_cfg:
            # Flatten config into a name -> label mapping
            wf_to_group = {}
            for group_label, wf_list in groups_cfg.items():
                for wf in wf_list:
                    wf_to_group[wf] = group_label  # can stay as string!

            default_group = "__OTHER__"
            manual_groups = np.array(
                [wf_to_group.get(str(w), default_group) for w in groups]
            )
        else:
            # Fully generic fallback: each wind_farm is its own group
            manual_groups = groups

        cv = LeaveOneGroupOut()
        splits = list(cv.split(xr_data.case_index, groups=manual_groups))
        n_splits = cv.get_n_splits(groups=manual_groups)

        # Extract fold labels (the left-out group for each fold)
        unique_groups = np.unique(manual_groups)
        for train_idx, test_idx in splits:
            # Find which group is left out (present in test but not in train)
            test_groups = np.unique(manual_groups[test_idx])
            fold_labels.append(
                str(test_groups[0]) if len(test_groups) == 1 else str(test_groups)
            )

        print(f"Using LeaveOneGroupOut with {n_splits} groups: {list(unique_groups)}")

    if splitting_mode == "kfold_shuffled":
        n_splits = cv_config.get("n_splits", 5)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = list(cv.split(xr_data.case_index))
        fold_labels = [f"Fold {i + 1}" for i in range(n_splits)]
        print(f"Using KFold with {n_splits} splits.")

    stats_cv, y_preds, y_tests, pw_all, ref_all = [], [], [], [], []
    fold_farm_names = []  # Track farm names per fold for visualization

    # --- Add lists to store items for SHAP ---
    all_models = []
    all_xtest_scaled = []  # Only used for tree models
    all_features_df = []

    # --- Add lists for local calibration parameter prediction tracking ---
    all_predicted_params = []  # Predicted parameter values per fold
    all_actual_optimal_params = []  # Actual optimal parameter values per fold
    swept_params = xr_data.attrs.get("swept_params", [])

    for i, (train_idx_locs, test_idx_locs) in enumerate(splits):
        # Get the actual case_index *values* at these integer locations
        train_indices = xr_data.case_index.values[train_idx_locs]
        test_indices = xr_data.case_index.values[test_idx_locs]

        dataset_train = xr_data.where(xr_data.case_index.isin(train_indices), drop=True)
        dataset_test = xr_data.where(xr_data.case_index.isin(test_indices), drop=True)

        # Track farm names for this fold (for visualization)
        if "wind_farm" in xr_data.coords:
            fold_farm_names.append(xr_data.wind_farm.values[test_idx_locs])

        if calibration_mode == "local":
            calibrator = Calibrator_cls(
                dataset_train,
                feature_names=features_list,
                regressor_name=local_regressor,
                regressor_params=local_regressor_params,
            )
        else:
            calibrator = Calibrator_cls(dataset_train)

        bias_pred = BiasPredictor_cls(ML_pipeline)

        # --- Pass features_list to MainPipeline ---
        main_pipe = MainPipeline_cls(
            calibrator,
            bias_pred,
            features_list=features_list,
            calibration_mode=calibration_mode,
        )

        x_test, y_test, idxs = main_pipe.fit(dataset_train, dataset_test)
        y_pred = main_pipe.predict(x_test)

        # Get correct validation data for this fold
        val_data_fold = validation_data.sel(sample=idxs).where(
            validation_data.case_index.isin(test_indices), drop=True
        )

        if calibration_mode == "global":
            # Single sample index for all cases
            val_data_fold = validation_data.sel(sample=idxs).where(
                validation_data.case_index.isin(test_indices), drop=True
            )

            pw = val_data_fold["pw_power_cap"].values
            ref = val_data_fold["ref_power_cap"].values
        else:
            # Local calibration: idxs is an array of per-case sample indices
            # We must build pw/ref per test case using those indices.

            # Local calibration: idxs is an array of per-case sample indices
            # We must build pw/ref per test case using those indices.
            idxs = np.asarray(idxs)
            if idxs.shape[0] != len(test_indices):
                raise ValueError(
                    f"Local calibration returned {idxs.shape[0]} indices, "
                    f"but there are {len(test_indices)} test cases."
                )

            # --- Track predicted vs actual optimal parameters for this fold ---
            X_test_features = main_pipe._extract_features(
                dataset_test.isel(sample=0).to_dataframe().reset_index()
            )
            predicted_params_fold = main_pipe.calibrator.predict(X_test_features)
            all_predicted_params.append(predicted_params_fold)

            # Get actual optimal parameters (from the sample indices we chose)
            actual_params_fold = {p: [] for p in swept_params}
            for sample_idx in idxs:
                for param_name in swept_params:
                    if param_name in dataset_test.coords:
                        actual_params_fold[param_name].append(
                            float(
                                dataset_test.coords[param_name]
                                .isel(sample=sample_idx)
                                .values
                            )
                        )
            actual_params_df = pd.DataFrame(actual_params_fold)
            all_actual_optimal_params.append(actual_params_df)

            pw_list = []
            ref_list = []

            # dataset_test.case_index is the subset used inside MainPipeline
            local_case_indices = dataset_test.case_index.values

            for local_case_idx, sample_idx in enumerate(idxs):
                case_index_val = local_case_indices[local_case_idx]

                # Pick the appropriate sample & case_index from validation_data
                this_point = validation_data.sel(
                    sample=int(sample_idx), case_index=case_index_val
                )

                pw_list.append(float(this_point["pw_power_cap"].values))
                ref_list.append(float(this_point["ref_power_cap"].values))

            pw = np.array(pw_list)
            ref = np.array(ref_list)

        stats = compute_metrics(y_test, y_pred, pw=pw, ref=ref)
        stats_cv.append(stats)

        y_preds.append(y_pred)
        y_tests.append(y_test)
        pw_all.append(pw)
        ref_all.append(ref)

        # --- Store model and data for SHAP / SIR global importance ---
        all_models.append(main_pipe.bias_predictor.pipeline)
        all_features_df.append(x_test)
        if model_type == "tree":
            X_test_scaled = main_pipe.bias_predictor.pipeline.named_steps[
                "scaler"
            ].transform(x_test)
            all_xtest_scaled.append(X_test_scaled)

    cv_results = pd.DataFrame(stats_cv)

    # --- START PLOTTING BLOCK (Visualization) ---

    # Flatten all fold results into single arrays for plotting
    y_preds_flat = np.concatenate(y_preds)
    y_tests_flat = np.concatenate(y_tests)

    have_power = all(p is not None for p in pw_all) and all(
        r is not None for r in ref_all
    )

    if have_power:
        pw_flat = np.concatenate(pw_all)
        ref_flat = np.concatenate(ref_all)
        corrected_power_flat = pw_flat - y_preds_flat

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Cross-Validation Model Performance", fontsize=16)

    # 1. Predicted vs. True Bias
    ax1.scatter(y_tests_flat, y_preds_flat, alpha=0.5, s=10)
    min_bias = min(y_tests_flat.min(), y_preds_flat.min())
    max_bias = max(y_tests_flat.max(), y_preds_flat.max())
    ax1.plot([min_bias, max_bias], [min_bias, max_bias], "r--", label="1:1 Line")
    ax1.set_xlabel("True Bias (PyWake - Ref)")
    ax1.set_ylabel("Predicted Bias (ML)")
    ax1.set_title("ML Model Performance")
    ax1.grid(True)
    ax1.legend()
    ax1.axis("equal")

    # 2. Uncorrected Power vs. Reference
    ax2.scatter(ref_flat, pw_flat, alpha=0.5, s=10, label="Data")
    min_power = min(ref_flat.min(), pw_flat.min())
    max_power = max(ref_flat.max(), pw_flat.max())
    ax2.plot([min_power, max_power], [min_power, max_power], "r--", label="1:1 Line")
    ax2.set_xlabel("Reference Power (Truth)")
    ax2.set_ylabel("Uncorrected Power (Calibrated)")
    ax2.set_title("Uncorrected Model")
    ax2.grid(True)
    ax2.legend()
    ax2.axis("equal")

    # 3. Corrected Power vs. Reference
    ax3.scatter(ref_flat, corrected_power_flat, alpha=0.5, s=10, label="Data")
    ax3.plot([min_power, max_power], [min_power, max_power], "r--", label="1:1 Line")
    ax3.set_xlabel("Reference Power (Truth)")
    ax3.set_ylabel("Corrected Power (Calibrated + ML)")
    ax3.set_title("Corrected Model")
    ax3.grid(True)
    ax3.legend()
    ax3.axis("equal")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = output_dir / "correction_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved correction plot to: {plot_path}")
    plt.close(fig)  # Close the figure

    # --- MULTI-FARM CV VISUALIZATION (NEW) ---
    if is_multi_farm or splitting_mode == "LeaveOneGroupOut":
        print("--- Generating Multi-Farm CV Visualizations ---")

        # 1. Per-fold metrics visualization
        plot_multi_farm_cv_metrics(
            cv_results=cv_results,
            fold_labels=fold_labels,
            output_dir=output_dir,
            splitting_mode=splitting_mode,
        )

        # 2. Predictions colored by fold
        plot_farm_wise_predictions(
            y_tests=y_tests,
            y_preds=y_preds,
            fold_labels=fold_labels,
            fold_farm_names=fold_farm_names,
            output_dir=output_dir,
        )

        # 3. Generalization summary table
        plot_generalization_matrix(
            cv_results=cv_results, fold_labels=fold_labels, output_dir=output_dir
        )

    # --- PARAMETER PREDICTION PLOT (Local Calibration Only) ---
    if calibration_mode == "local" and all_predicted_params and swept_params:
        print("--- Generating Parameter Prediction Quality Plot ---")

        # Concatenate all folds
        predicted_all = pd.concat(all_predicted_params, axis=0, ignore_index=True)
        actual_all = pd.concat(all_actual_optimal_params, axis=0, ignore_index=True)

        n_params = len(swept_params)
        fig, axes = plt.subplots(1, n_params, figsize=(6 * n_params, 5))
        if n_params == 1:
            axes = [axes]

        fig.suptitle("Local Calibration: Parameter Prediction Quality", fontsize=14)

        for idx, param_name in enumerate(swept_params):
            ax = axes[idx]
            pred_vals = predicted_all[param_name].values
            actual_vals = actual_all[param_name].values

            ax.scatter(actual_vals, pred_vals, alpha=0.5, s=15)

            # 1:1 line
            min_val = min(actual_vals.min(), pred_vals.min())
            max_val = max(actual_vals.max(), pred_vals.max())
            ax.plot([min_val, max_val], [min_val, max_val], "r--", label="1:1 Line")

            # Calculate R² for parameter prediction
            ss_res = np.sum((actual_vals - pred_vals) ** 2)
            ss_tot = np.sum((actual_vals - actual_vals.mean()) ** 2)
            r2_param = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

            ax.set_xlabel(f"Actual Optimal {param_name}")
            ax.set_ylabel(f"Predicted {param_name}")
            ax.set_title(f"{param_name} (R² = {r2_param:.3f})")
            ax.legend()
            ax.grid(True)
            ax.set_aspect("equal", adjustable="box")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        param_plot_path = output_dir / "local_parameter_prediction.png"
        plt.savefig(param_plot_path, dpi=150)
        print(f"Saved parameter prediction plot to: {param_plot_path}")
        plt.close(fig)

    # --- END PLOTTING BLOCK ---

    # --- START SHAP ON BIAS BLOCK ---
    if sa_config.get("run_bias_sensitivity", False):
        print(f"--- Running Bias Sensitivity (Model Type: {model_type}) ---")

        if model_type == "tree":
            try:
                print("--- Calculating Bias SHAP (TreeExplainer) ---")
                all_shap_values = []

                for i in range(n_splits):
                    model = all_models[i].named_steps["model"]
                    X_test_scaled = all_xtest_scaled[i]

                    explainer = shap.TreeExplainer(model)
                    shap_values_fold = explainer.shap_values(X_test_scaled)
                    all_shap_values.append(shap_values_fold)

                final_shap_values = np.concatenate(all_shap_values, axis=0)

                final_features_df = pd.concat(all_features_df, axis=0)
                # ... (string cleaning logic) ...
                for col in final_features_df.columns:
                    if final_features_df[col].dtype == "object":
                        if final_features_df[col].dropna().empty:
                            continue
                        first_item = final_features_df[col].dropna().iloc[0]
                        if isinstance(first_item, str):
                            print(f"    Cleaning string column in SHAP: {col}")
                            final_features_df[col] = (
                                final_features_df[col]
                                .str.replace(r"[\[\]]", "", regex=True)
                                .astype(float)
                            )
                        else:
                            final_features_df[col] = final_features_df[col].astype(
                                float
                            )
                final_features_df = final_features_df.astype(float)  # Final cast

                # --- 1. GENERATE AND SAVE BAR PLOT ---
                print("--- Calculating Bias SHAP Global Feature Importance ---")
                mean_abs_shap = np.mean(np.abs(final_shap_values), axis=0)
                shap_scores = pd.Series(
                    mean_abs_shap, index=final_features_df.columns
                ).sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(10, 8))
                shap_scores.plot(kind="barh", ax=ax)
                ax.set_title(
                    "Global SHAP Feature Importance (Mean Absolute SHAP Value)"
                )
                ax.set_xlabel("Mean |SHAP Value| (Impact on Bias Prediction)")
                plt.tight_layout()

                bar_plot_path = output_dir / "bias_prediction_shap_importance.png"
                plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved SHAP importance bar plot to: {bar_plot_path}")

                # --- 2. GENERATE AND SAVE BEESWARM PLOT ---
                shap.summary_plot(final_shap_values, final_features_df, show=False)
                plot_path = output_dir / "bias_prediction_shap.png"
                plt.savefig(plot_path, dpi=150, bbox_inches="tight")
                plt.close()
                print(f"Saved bias SHAP beeswarm plot to: {plot_path}")

            except Exception as e:
                print(f"Could not run bias SHAP (Tree) analysis: {e}")
                raise e

        elif model_type == "linear":
            try:
                print(
                    "--- Calculating Bias Linear Feature Importance (Coefficients) ---"
                )
                all_linear_scores = []
                feature_names = all_features_df[0].columns

                for i in range(n_splits):
                    model = all_models[i]
                    fold_scores = model.get_feature_importance(feature_names)
                    all_linear_scores.append(fold_scores)

                # Average importances across folds
                importance_scores = pd.concat(all_linear_scores, axis=1).mean(axis=1)
                importance_scores = importance_scores.sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(10, 8))
                importance_scores.plot(kind="barh", ax=ax)
                ax.set_title("Linear Model Feature Importance (Mean |Coefficient|)")
                ax.set_xlabel("Mean |Coefficient| (Impact on Bias Prediction)")
                plt.tight_layout()

                bar_plot_path = output_dir / "bias_prediction_linear_importance.png"
                plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved linear importance plot to: {bar_plot_path}")

            except Exception as e:
                print(f"Could not run bias linear analysis: {e}")
                raise e
        elif model_type == "sir":
            try:
                print(
                    "--- Calculating Bias SIR Feature Importance (Averaged over folds) ---"
                )
                all_sir_scores = []
                feature_names = all_features_df[
                    0
                ].columns  # Get feature names from first fold

                for i in range(n_splits):
                    model = all_models[i]  # This is the SIRPolynomialRegressor instance
                    fold_scores = model.get_feature_importance(feature_names)
                    all_sir_scores.append(fold_scores)

                # Average the importances across all folds
                shap_scores = pd.concat(all_sir_scores, axis=1).mean(axis=1)
                shap_scores = shap_scores.sort_values(ascending=True)  # Sort for barh

                # Generate and save the bar plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap_scores.plot(kind="barh", ax=ax)
                ax.set_title(
                    "Global SIR Feature Importance (Mean Absolute Direction Coefficient)"
                )
                ax.set_xlabel(
                    "Mean |SIR Direction Coefficient| (Impact on Bias Prediction)"
                )
                plt.tight_layout()

                bar_plot_path = output_dir / "bias_prediction_sir_importance.png"
                plt.savefig(bar_plot_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                print(f"Saved SIR importance bar plot to: {bar_plot_path}")
                print("--- NOTE: Beeswarm plot is not available for SIR model. ---")

            except Exception as e:
                print(f"Could not run bias SIR analysis: {e}")
                raise e

    # --- END SHAP ON BIAS BLOCK ---

    return cv_results, y_preds, y_tests


# Testing
if __name__ == "__main__":
    import xarray as xr

    xr_data = xr.load_dataset("results_stacked_hh.nc")
    pipe_xgb = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500)),
        ]
    )

    # Create a dummy output_dir for testing
    test_output_dir = Path("./test_results")
    test_output_dir.mkdir(exist_ok=True)

    cv_results, y_preds, y_tests = run_cross_validation(
        xr_data,
        ML_pipeline=pipe_xgb,
        model_type="tree",  # <-- Need to provide this for the test
        Calibrator_cls=MinBiasCalibrator,
        BiasPredictor_cls=BiasPredictor,
        MainPipeline_cls=MainPipeline,
        cv_config={"splitting_mode": "kfold_shuffled", "n_splits": 5},
        features_list=["turbulence_intensity"],  # Example features
        output_dir=test_output_dir,
        sa_config={"run_bias_shap": True},  # Example config
    )
    print(cv_results.mean())
