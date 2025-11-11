import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import shap

from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import NotFittedError

import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

try:
    from sliced import SlicedInverseRegression
except ImportError as e:
    print("Error: 'sliced' package not found. SIRPolynomialRegressor will not be available.")
    raise e


"""
This script contains:
- SIRPolynomialRegressor class (NEW)
- Calibrator classes
- BiasPredictor class
- MainPipeline class
- Cross validation routine
- SHAP/SIR sensitivity analysis functions
"""

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
            raise ImportError("The 'sliced' package is required to use SIRPolynomialRegressor.")
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
        self.poly_reg_ = Pipeline([
            ('poly', PolynomialFeatures(degree=self.degree, include_bias=False)),
            ('lin_reg', LinearRegression())
        ])
        self.poly_reg_.fit(X_sir, y)

        return self

    def predict(self, X):
        if not hasattr(self, 'scaler_'):
            raise NotFittedError("This SIRPolynomialRegressor instance is not fitted yet.")
        
        X_scaled = self.scaler_.transform(X)
        X_sir = self.sir_.transform(X_scaled)
        return self.poly_reg_.predict(X_sir)

    def get_feature_importance(self, feature_names):
        if not hasattr(self, 'sir_directions_'):
            raise NotFittedError("This SIRPolynomialRegressor instance is not fitted yet.")
        
        return pd.Series(self.sir_directions_, index=feature_names)

## ------------------------------------------------------------------ ##


class MinBiasCalibrator:
    def __init__(self, dataset_train):
        self.dataset_train = dataset_train

    def fit(self):
        abs_total_bias = np.abs(self.dataset_train['model_bias_cap'].sum(dim='case_index'))
        idx_ = int(abs_total_bias.argmin().values)
        parameters=np.round(self.dataset_train.isel(sample=idx_).k_b.values,4)
        self.best_idx_ = idx_
        self.best_params_ = parameters

        return self  # same value for all data points (this is sort of redundant)

class DefaultParams:
    def __init__(self, dataset_train):
        self.dataset_train = dataset_train

    def fit(self):
        idx_ = 0  # default parameter is hard-coded upstream as first index
        parameters = np.round(self.dataset_train.isel(sample=idx_).k_b.values, 4)
        self.best_idx_ = idx_
        self.best_params_ = parameters
        return self

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
    def __init__(self, calibrator, bias_predictor, features_list: list):
        """
        Args:
            features_list (list): List of feature names to use for training.
        """
        self.calibrator = calibrator
        self.bias_predictor = bias_predictor
        self.features_list = features_list
        if not self.features_list:
            raise ValueError("features_list cannot be empty.")

    def fit(self, dataset_train, dataset_test):
        # 1. Fit calibrator
        self.calibrator.fit()
        idxs = self.calibrator.best_idx_
        params = self.calibrator.best_params_

        # 2. Prepare training data
        dataset_train_cal = dataset_train.sel(sample=idxs)
        dataset_test_cal = dataset_test.sel(sample=idxs)

        X_train_df = dataset_train_cal.to_dataframe().reset_index()
        X_test_df = dataset_test_cal.to_dataframe().reset_index()

        # --- NEW FEATURE SELECTION ---
        # Use *only* the features specified in the list
        try:
            X_train = X_train_df[self.features_list]
            X_test = X_test_df[self.features_list]
        except KeyError as e:
            print(f"Error: Feature not found in dataset: {e}")
            print(f"Available columns: {list(X_train_df.columns)}")
            raise
        
        y_train = dataset_train_cal['model_bias_cap'].values
        y_test = dataset_test_cal['model_bias_cap'].values

        # --- FIX: Clean string-like columns before fitting ---
        for col in X_train.columns:
            if X_train[col].dtype == 'object':
                # Check for empty dataframe, common in CV folds
                if X_train[col].dropna().empty:
                    continue
                first_item = X_train[col].dropna().iloc[0]
                if isinstance(first_item, str):
                    print(f"    Cleaning string column in FIT: {col}")
                    X_train[col] = X_train[col].str.replace(r'[\[\]]', '', regex=True).astype(float)
                    X_test[col] = X_test[col].str.replace(r'[\[\]]', '', regex=True).astype(float)
                else:
                    X_train[col] = X_train[col].astype(float)
                    X_test[col] = X_test[col].astype(float)
        
        # 3. Fit bias predictor
        self.bias_predictor.fit(X_train, y_train)

        # Store X_test for predict() method
        self.X_test_ = X_test
        
        return X_test, y_test, idxs

    def predict(self, X=None):
        if X is None:
            X = self.X_test_
        return self.bias_predictor.predict(X)

def compute_metrics(y_true, bias_samples,pw, ref,data_driv=None):
    mse = ((y_true - bias_samples)**2).mean()
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, bias_samples)
    r2 = r2_score(y_true, bias_samples)

    if pw is not None and ref is not None and data_driv is None:
        pw_bias = np.mean(pw-ref)
        pw_bias_corrected=np.mean((pw-bias_samples)-ref)
    else:
        pw_bias = None
        pw_bias_corrected = None

    if data_driv and ref:
        data_driv_bias = np.mean(data_driv-ref)
    else:
        data_driv_bias = None
    
    return {"rmse": rmse, "mse": mse, "mae": mae, "r2": r2,
            "pw_bias": pw_bias, "pw_bias_corrected": pw_bias_corrected,
            "data_driv_bias": data_driv_bias}

def run_observation_sensitivity(database, features_list, ml_pipeline, model_type, output_dir):
    """
    Trains a model to predict reference power from features
    and saves feature importance plots (SHAP or SIR).
    """
    print("--- Running Observation Sensitivity ---")
    
    # Select only one sample (e.g., sample=0)
    db_slice = database.sel(sample=0).to_dataframe().reset_index()
    
    X = db_slice[features_list]
    y = db_slice['ref_power_cap']
    y = y.astype(float)

    X = X.astype(float) # Final cast
    print('y is ', y)

    # Train a new pipeline on all data
    model_pipeline = ml_pipeline.fit(X, y)
    
    # --- Model-Aware Sensitivity Logic ---
    # We check the model type to decide which importance metric to use.
    if model_type == "tree":
        print("--- Calculating SHAP (TreeExplainer) ---")
        xgb_model = model_pipeline.named_steps['model']
        X_scaled = model_pipeline.named_steps['scaler'].transform(X)
        
        explainer = shap.TreeExplainer(xgb_model)
        shap_values_obj = explainer(X_scaled) # This is the Explanation object
        shap_values = shap_values_obj.values  # This is the raw numpy array

        # --- 1. BAR PLOT (from mean abs shap) ---
        print("--- Calculating SHAP Global Feature Importance (Mean Absolute Value) ---")
        mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
        shap_scores = pd.Series(mean_abs_shap, index=X.columns).sort_values(ascending=True)

        fig, ax = plt.subplots(figsize=(10, 8))
        shap_scores.plot(kind='barh', ax=ax)
        ax.set_title('Global SHAP Feature Importance (Mean Absolute SHAP Value)')
        ax.set_xlabel('Mean |SHAP Value| (Impact on ref_power_cap)')
        plt.tight_layout()
        
        bar_plot_path = output_dir / "observation_shap_importance.png"
        plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved SHAP importance bar plot to: {bar_plot_path}")
        
        # --- 2. BEESWARM PLOT (from shap object) ---
        shap.summary_plot(shap_values_obj, X, show=False)
        plot_path = output_dir / "observation_shap.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved observation SHAP beeswarm plot to: {plot_path}")

    elif model_type == "sir":
        print("--- Calculating SIR Feature Importance (Direction Coefficients) ---")
        # model_pipeline is the SIRPolynomialRegressor instance
        shap_scores = model_pipeline.get_feature_importance(X.columns)
        shap_scores = shap_scores.sort_values(ascending=True) # Sort for barh

        # --- 1. BAR PLOT (from SIR directions) ---
        fig, ax = plt.subplots(figsize=(10, 8))
        shap_scores.plot(kind='barh', ax=ax)
        ax.set_title('Global SIR Feature Importance (Absolute Direction Coefficient)')
        ax.set_xlabel('Mean |SIR Direction Coefficient| (Impact on ref_power_cap)')
        plt.tight_layout()
        
        bar_plot_path = output_dir / "observation_sir_importance.png"
        plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved SIR importance bar plot to: {bar_plot_path}")
        print("--- NOTE: Beeswarm plot is not available for SIR model. ---")
            
def run_cross_validation(xr_data, ML_pipeline, model_type, Calibrator_cls, BiasPredictor_cls, 
                         MainPipeline_cls, cv_config: dict, features_list: list,
                         output_dir: Path, sa_config: dict):
    
    validation_data = xr_data[['turb_rated_power', 'pw_power_cap', 'ref_power_cap', 'case_index']]
    groups = xr_data['wind_farm'].values
    
    splitting_mode = cv_config.get('splitting_mode', 'kfold_shuffled')
    
    if splitting_mode == 'LeaveOneGroupOut':
        # Hardcoded groups (as in your original)
        # This part is still brittle, but we can make it work
        wf_to_group = {
            "HR1": 0, "HR2": 0, "HR3":0,
            "NYSTED1":1, "NYSTED2":1,
            "VirtWF_ABL_IEA10":2,  
            "VirtWF_ABL_IEA15_ali_DX5_DY5":3,   
            "VirtWF_ABL_IEA15_stag_DX5_DY5":4, "VirtWF_ABL_IEA15_stag_DX5_DY7p5":4, "VirtWF_ABL_IEA15_stag_DX7p5_DY5":4,  
            "VirtWF_ABL_IEA22":5,
            "KUL_LES": 6 # Add KUL_LES to a new group
        }
        # Map groups, assigning a default group (e.g., 99) if not found
        default_group = max(wf_to_group.values()) + 1 if wf_to_group else 0
        manual_groups = np.array([wf_to_group.get(w, default_group) for w in groups])
        
        if len(np.unique(manual_groups)) < 2:
            print("Warning: Only one unique group found. "
                  "Falling back to KFold with 5 splits.")
            splitting_mode = 'kfold_shuffled'
            n_splits = 5
        else:
            cv = LeaveOneGroupOut()
            splits = cv.split(xr_data.case_index, groups=manual_groups)
            n_splits = cv.get_n_splits(groups=manual_groups) # Get actual number of splits
            print(f"Using LeaveOneGroupOut with {n_splits} groups.")
    
    if splitting_mode == 'kfold_shuffled':
        n_splits = cv_config.get('n_splits', 5)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        splits = cv.split(xr_data.case_index)
        print(f"Using KFold with {n_splits} splits.")

    stats_cv, y_preds, y_tests, pw_all, ref_all = [], [], [], [], []
    
    # --- Add lists to store items for SHAP ---
    all_models = []
    all_xtest_scaled = [] # Only used for tree models
    all_features_df = []

    for i, (train_idx_locs, test_idx_locs) in enumerate(splits):
        # Get the actual case_index *values* at these integer locations
        train_indices = xr_data.case_index.values[train_idx_locs]
        test_indices = xr_data.case_index.values[test_idx_locs]

        dataset_train = xr_data.where(xr_data.case_index.isin(train_indices), drop=True)
        dataset_test = xr_data.where(xr_data.case_index.isin(test_indices), drop=True)
        
        calibrator = Calibrator_cls(dataset_train)
        bias_pred = BiasPredictor_cls(ML_pipeline)
        
        # --- Pass features_list to MainPipeline ---
        main_pipe = MainPipeline_cls(calibrator, bias_pred, features_list=features_list)
        
        x_test, y_test, idxs = main_pipe.fit(dataset_train, dataset_test)
        y_pred = main_pipe.predict(x_test)

        # Get correct validation data for this fold
        val_data_fold = validation_data.sel(sample=idxs).where(
            validation_data.case_index.isin(test_indices), drop=True
        )
        
        pw = val_data_fold['pw_power_cap'].values
        ref = val_data_fold['ref_power_cap'].values

        stats = compute_metrics(y_test, y_pred, pw=pw, ref=ref)
        stats_cv.append(stats)
        
        y_preds.append(y_pred)
        y_tests.append(y_test)
        pw_all.append(pw)    # <-- Collect pw
        ref_all.append(ref)  # <-- Collect ref
        
        # --- Store model and data for SHAP ---
        all_models.append(main_pipe.bias_predictor.pipeline) # This is either the XGB-Pipeline or the SIR-Regressor
        all_features_df.append(x_test)
        if model_type == "tree":
            all_xtest_scaled.append(main_pipe.bias_predictor.pipeline.named_steps['scaler'].transform(x_test))


    cv_results = pd.DataFrame(stats_cv)
    
    # --- START PLOTTING BLOCK (Visualization) ---
    
    # Flatten all fold results into single arrays for plotting
    y_preds_flat = np.concatenate(y_preds)
    y_tests_flat = np.concatenate(y_tests)
    pw_flat = np.concatenate(pw_all)
    ref_flat = np.concatenate(ref_all)
    corrected_power_flat = pw_flat - y_preds_flat # Note: Bias is (Model - Ref), so Corrected = Model - Bias

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Cross-Validation Model Performance', fontsize=16)

    # 1. Predicted vs. True Bias
    ax1.scatter(y_tests_flat, y_preds_flat, alpha=0.5, s=10)
    min_bias = min(y_tests_flat.min(), y_preds_flat.min())
    max_bias = max(y_tests_flat.max(), y_preds_flat.max())
    ax1.plot([min_bias, max_bias], [min_bias, max_bias], 'r--', label='1:1 Line')
    ax1.set_xlabel("True Bias (PyWake - Ref)")
    ax1.set_ylabel("Predicted Bias (ML)")
    ax1.set_title("ML Model Performance")
    ax1.grid(True); ax1.legend(); ax1.axis('equal')

    # 2. Uncorrected Power vs. Reference
    ax2.scatter(ref_flat, pw_flat, alpha=0.5, s=10, label='Data')
    min_power = min(ref_flat.min(), pw_flat.min())
    max_power = max(ref_flat.max(), pw_flat.max())
    ax2.plot([min_power, max_power], [min_power, max_power], 'r--', label='1:1 Line')
    ax2.set_xlabel("Reference Power (Truth)")
    ax2.set_ylabel("Uncorrected Power (Calibrated)")
    ax2.set_title("Uncorrected Model")
    ax2.grid(True); ax2.legend(); ax2.axis('equal')

    # 3. Corrected Power vs. Reference
    ax3.scatter(ref_flat, corrected_power_flat, alpha=0.5, s=10, label='Data')
    ax3.plot([min_power, max_power], [min_power, max_power], 'r--', label='1:1 Line')
    ax3.set_xlabel("Reference Power (Truth)")
    ax3.set_ylabel("Corrected Power (Calibrated + ML)")
    ax3.set_title("Corrected Model")
    ax3.grid(True); ax3.legend(); ax3.axis('equal')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = output_dir / "correction_results.png"
    plt.savefig(plot_path, dpi=150)
    print(f"Saved correction plot to: {plot_path}")
    plt.close(fig) # Close the figure
    
    # --- END PLOTTING BLOCK ---

    # --- START SHAP ON BIAS BLOCK ---
    if sa_config.get('run_bias_sensitivity', False):
        print(f"--- Running Bias Sensitivity (Model Type: {model_type}) ---")

        if model_type == "tree":
            try:
                print("--- Calculating Bias SHAP (TreeExplainer) ---")
                all_shap_values = []
                
                for i in range(n_splits):
                    model = all_models[i].named_steps['model']
                    X_test_scaled = all_xtest_scaled[i]
                    
                    explainer = shap.TreeExplainer(model)
                    shap_values_fold = explainer.shap_values(X_test_scaled)
                    all_shap_values.append(shap_values_fold)
                    
                final_shap_values = np.concatenate(all_shap_values, axis=0)
                
                final_features_df = pd.concat(all_features_df, axis=0)
                # ... (string cleaning logic) ...
                for col in final_features_df.columns:
                    if final_features_df[col].dtype == 'object':
                        if final_features_df[col].dropna().empty:
                            continue
                        first_item = final_features_df[col].dropna().iloc[0]
                        if isinstance(first_item, str):
                            print(f"    Cleaning string column in SHAP: {col}")
                            final_features_df[col] = final_features_df[col].str.replace(r'[\[\]]', '', regex=True).astype(float)
                        else:
                            final_features_df[col] = final_features_df[col].astype(float)
                final_features_df = final_features_df.astype(float) # Final cast
                
                # --- 1. GENERATE AND SAVE BAR PLOT ---
                print("--- Calculating Bias SHAP Global Feature Importance ---")
                mean_abs_shap = np.mean(np.abs(final_shap_values), axis=0)
                shap_scores = pd.Series(mean_abs_shap, index=final_features_df.columns).sort_values(ascending=True)

                fig, ax = plt.subplots(figsize=(10, 8))
                shap_scores.plot(kind='barh', ax=ax)
                ax.set_title('Global SHAP Feature Importance (Mean Absolute SHAP Value)')
                ax.set_xlabel('Mean |SHAP Value| (Impact on Bias Prediction)')
                plt.tight_layout()
                
                bar_plot_path = output_dir / "bias_prediction_shap_importance.png"
                plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                print(f"Saved SHAP importance bar plot to: {bar_plot_path}")
                
                # --- 2. GENERATE AND SAVE BEESWARM PLOT ---
                shap.summary_plot(final_shap_values, final_features_df, show=False)
                plot_path = output_dir / "bias_prediction_shap.png"
                plt.savefig(plot_path, dpi=150, bbox_inches='tight')
                plt.close()
                print(f"Saved bias SHAP beeswarm plot to: {plot_path}")

            except Exception as e:
                print(f"Could not run bias SHAP (Tree) analysis: {e}")
                raise e

        elif model_type == "sir":
            try:
                print("--- Calculating Bias SIR Feature Importance (Averaged over folds) ---")
                all_sir_scores = []
                feature_names = all_features_df[0].columns # Get feature names from first fold
                
                for i in range(n_splits):
                    model = all_models[i] # This is the SIRPolynomialRegressor instance
                    fold_scores = model.get_feature_importance(feature_names)
                    all_sir_scores.append(fold_scores)
                
                # Average the importances across all folds
                shap_scores = pd.concat(all_sir_scores, axis=1).mean(axis=1)
                shap_scores = shap_scores.sort_values(ascending=True) # Sort for barh

                # Generate and save the bar plot
                fig, ax = plt.subplots(figsize=(10, 8))
                shap_scores.plot(kind='barh', ax=ax)
                ax.set_title('Global SIR Feature Importance (Mean Absolute Direction Coefficient)')
                ax.set_xlabel('Mean |SIR Direction Coefficient| (Impact on Bias Prediction)')
                plt.tight_layout()
                
                bar_plot_path = output_dir / "bias_prediction_sir_importance.png"
                plt.savefig(bar_plot_path, dpi=150, bbox_inches='tight')
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
    pipe_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
    ])
    
    # Create a dummy output_dir for testing
    test_output_dir = Path("./test_results")
    test_output_dir.mkdir(exist_ok=True)
    
    cv_results,y_preds,y_tests=run_cross_validation(
        xr_data,
        ML_pipeline=pipe_xgb,
        model_type="tree",  # <-- Need to provide this for the test
        Calibrator_cls=MinBiasCalibrator,
        BiasPredictor_cls=BiasPredictor,
        MainPipeline_cls=MainPipeline,
        cv_config={'splitting_mode': 'kfold_shuffled', 'n_splits': 5},
        features_list=['turbulence_intensity'], # Example features
        output_dir=test_output_dir,
        sa_config={'run_bias_shap': True} # Example config
    )
    print(cv_results.mean())
