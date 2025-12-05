# wifa_uq/postprocessing/calibration/basic_calibration.py

import numpy as np
import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone


class DefaultParams:
    """
    Uses default parameter values from database metadata.
    Returns the sample index closest to the default parameter values.
    """
    def __init__(self, dataset_train):
        self.dataset_train = dataset_train
        self.swept_params = dataset_train.attrs.get('swept_params', [])
        self.param_defaults = dataset_train.attrs.get('param_defaults', {})
        
        self.best_idx_ = None
        self.best_params_ = None
    
    def fit(self):
        """Find sample closest to default parameter values."""
        if self.param_defaults:
            # Calculate distance to default for each sample
            distances = np.zeros(len(self.dataset_train.sample))
            for param_name, default_val in self.param_defaults.items():
                if default_val is not None and param_name in self.dataset_train.coords:
                    param_values = self.dataset_train.coords[param_name].values
                    distances += (param_values - default_val) ** 2
            
            self.best_idx_ = int(np.argmin(distances))
        else:
            self.best_idx_ = 0  # Fallback to first sample
        
        # Extract parameters at this index
        self.best_params_ = {}
        for param_name in self.swept_params:
            if param_name in self.dataset_train.coords:
                self.best_params_[param_name] = float(
                    self.dataset_train.coords[param_name].isel(sample=self.best_idx_).values
                )

        return self


class MinBiasCalibrator:
    """
    Finds the sample with minimum total absolute bias.
    Works with ANY swept parameters stored in the database.
    
    This is a GLOBAL calibrator - returns a single set of parameters
    for the entire dataset.
    """
    def __init__(self, dataset_train):
        self.dataset_train = dataset_train
        self.swept_params = dataset_train.attrs.get('swept_params', [])
        if not self.swept_params:
            self.swept_params = self._infer_swept_params()
        
        self.best_idx_ = None
        self.best_params_ = None
    
    def _infer_swept_params(self):
        """Infer which coordinates are swept parameters."""
        swept = []
        for coord_name in self.dataset_train.coords:
            coord = self.dataset_train.coords[coord_name]
            if 'sample' in coord.dims and coord_name != 'sample':
                swept.append(coord_name)
        return swept
    
    def fit(self):
        """Find sample index that minimizes total absolute bias."""
        abs_total_bias = np.abs(
            self.dataset_train['model_bias_cap'].sum(dim='case_index')
        )
        self.best_idx_ = int(abs_total_bias.argmin().values)
        
        # Extract best parameters dynamically
        self.best_params_ = {}
        for param_name in self.swept_params:
            if param_name in self.dataset_train.coords:
                self.best_params_[param_name] = float(
                    self.dataset_train.coords[param_name].isel(sample=self.best_idx_).values
                )
        
        return self


class LocalParameterPredictor:
    """
    Predicts optimal parameters as a function of flow conditions.
    Works with ANY swept parameters stored in the database.
    
    This is a LOCAL calibrator - predicts different parameters for each
    flow case based on input features.
    
    Interface differs from global calibrators:
    - fit() trains an ML model
    - predict(X) returns optimal params for new conditions
    - get_optimal_indices() returns per-case optimal sample indices for training data
    """
    def __init__(self, dataset_train, feature_names, regressor=None):
        self.dataset_train = dataset_train
        self.feature_names = feature_names
        self.swept_params = dataset_train.attrs.get('swept_params', [])
        
        if not self.swept_params:
            self.swept_params = self._infer_swept_params()
        
        # Default to RandomForest if no regressor provided
        if regressor is None:
            base_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
        else:
            base_regressor = regressor
            
        # Wrap in MultiOutputRegressor if we have multiple parameters
        if len(self.swept_params) > 1:
            self.regressor = MultiOutputRegressor(base_regressor)
        else:
            self.regressor = clone(base_regressor)
        
        self.is_fitted = False
        self.optimal_indices_ = None  # Per-case optimal sample indices
        self.optimal_params_ = None   # Per-case optimal parameter values
    
    def _infer_swept_params(self):
        """Infer which coordinates are swept parameters."""
        swept = []
        for coord_name in self.dataset_train.coords:
            coord = self.dataset_train.coords[coord_name]
            if 'sample' in coord.dims and coord_name != 'sample':
                swept.append(coord_name)
        return swept
    
    def fit(self):
        """
        For each flow case, find optimal parameters, then train
        ML model to predict optimal params from features.
        """
        n_cases = len(self.dataset_train.case_index)
        n_samples = len(self.dataset_train.sample)
        
        # Find optimal parameters for each case
        self.optimal_indices_ = np.zeros(n_cases, dtype=int)
        self.optimal_params_ = {p: np.zeros(n_cases) for p in self.swept_params}
        
        for case_idx in range(n_cases):
            # Get bias values across all samples for this case
            bias_values = self.dataset_train['model_bias_cap'].isel(case_index=case_idx).values
            best_sample_idx = int(np.argmin(np.abs(bias_values)))
            
            self.optimal_indices_[case_idx] = best_sample_idx
            
            for param_name in self.swept_params:
                if param_name in self.dataset_train.coords:
                    param_val = float(
                        self.dataset_train.coords[param_name].isel(sample=best_sample_idx).values
                    )
                    self.optimal_params_[param_name][case_idx] = param_val
        
        # Build training data for ML model
        # Use sample=0 to get feature values (they're the same across samples)
        X_df = self.dataset_train.isel(sample=0).to_dataframe().reset_index()
        
        # Check that all features exist
        missing_features = [f for f in self.feature_names if f not in X_df.columns]
        if missing_features:
            raise ValueError(f"Features not found in dataset: {missing_features}")
        
        X = X_df[self.feature_names].values
        
        # Target is optimal parameter values
        if len(self.swept_params) == 1:
            y = self.optimal_params_[self.swept_params[0]]
        else:
            y = np.column_stack([self.optimal_params_[p] for p in self.swept_params])
        
        # Train regressor
        self.regressor.fit(X, y)
        self.is_fitted = True
        
        return self
    
    def predict(self, X):
        """
        Predict optimal parameters for new flow conditions.
        
        Args:
            X: Feature matrix (n_cases, n_features) or DataFrame
        
        Returns:
            DataFrame with columns for each swept parameter
        """
        if not self.is_fitted:
            raise RuntimeError("Must call fit() before predict()")
        
        if hasattr(X, 'values'):
            X = X.values
        
        predictions = self.regressor.predict(X)
        
        if len(self.swept_params) == 1:
            predictions = predictions.reshape(-1, 1)
        
        return pd.DataFrame(predictions, columns=self.swept_params)
    
    def get_optimal_indices(self):
        """
        Get the optimal sample indices for the training data.
        Used by MainPipeline to extract training targets.
        """
        if self.optimal_indices_ is None:
            raise RuntimeError("Must call fit() first")
        return self.optimal_indices_
