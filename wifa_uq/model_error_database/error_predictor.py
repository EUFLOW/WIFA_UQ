import xarray as xr
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import VarianceThreshold
from sliced import SlicedInverseRegression

# Fix for numpy deprecation
if not hasattr(np, 'int'):
    np.int = int

# -----------------------------
# BiasCalibrator Transformer
# -----------------------------
from sklearn.base import BaseEstimator, TransformerMixin

class BiasCalibrator():
    """
    Transformer that selects a calibrated sample index for each fold.
    Modes:
        - "min_bias": chooses the sample with minimum absolute total bias across points
        - "default": chooses sample index 0
    """
    def __init__(self, mode="min_bias"):
        self.mode = mode

    def calibrate(self, train_dataset: xr.Dataset,test_dataset: xr.Dataset):
        # Determine calibrated sample
        if self.mode == "min_bias":
            # sum over case_index dimension to get total bias per sample
            abs_total_bias = np.abs(train_dataset['model_bias_cap'].sum(dim='case_index'))
            idx_ = int(abs_total_bias.argmin().values)
        elif self.mode == "default":
            idx_ = 0
        else:
            raise ValueError(f"Unknown calibration mode {self.mode}")
        
        return train_dataset.sel(sample=idx_), test_dataset.sel(sample=idx_),idx_

# -----------------------------
# BiasPredictor Class
# -----------------------------
class BiasPredictor:
    def __init__(self, pipeline,data:xr.Dataset, cv_mode='default', calibration_mode='min_bias'):
        self.data = data
        self.cv_mode = cv_mode
        self.calibration_mode = calibration_mode
        self.pipeline = pipeline

    # -------------------------
    # Preprocess
    # -------------------------
    def preprocess_data(self):
        validation_data = self.data[['turb_rated_power', 'pw_power_cap', 'ref_power_cap']]
        groups = self.data['wind_farm'].values
        return validation_data, groups

    # -------------------------
    # Train / CV
    # -------------------------
    def train(self, groups, validation_data: pd.DataFrame):
        
        # CV split
        flow_cases = self.data.case_index
        if self.cv_mode == 'logo':
            cv = LeaveOneGroupOut()
            splits = cv.split(flow_cases, groups=groups)
            print(f"Using Leave-One-Group-Out CV with {len(np.unique(groups))} folds.")
        elif self.cv_mode == 'non_shuffled_cv':
            cv = KFold(n_splits=10, shuffle=False)
            splits = cv.split(flow_cases)
            print("Using K-Fold CV with 10 folds without shuffling.")
        else:
            cv = KFold(n_splits=10, shuffle=True, random_state=0)
            splits = cv.split(flow_cases)
            print("Using K-Fold CV with 10 folds and shuffling.")

        # Metrics
        scoring_standard = {
            'r2': r2_score,
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error
        }

        scoring_pw = {
            'pw_bias': lambda ref, pw: np.mean(pw - ref),
            'pw_bias_corrected': lambda ref, pw, bias: np.mean((pw - bias) - ref)
        }

        parameters = []
        y_preds = []
        y_reals = []
        results=[]
        # -------------------------
        # CV Loop
        # -------------------------
        for train_idx, test_idx in splits:
            dataset_train = self.data.where(self.data.case_index.isin(train_idx), drop=True)
            dataset_test = self.data.where(self.data.case_index.isin(test_idx), drop=True)

            # -------------------------
            # Calibrate (fold-specific)
            # -------------------------

            calibrator=BiasCalibrator(mode='min_bias')
            data_train_cal, data_test_cal,cal_idx=calibrator.calibrate(dataset_train,dataset_test)

            # Convert to DataFrame
            X_train_df = data_train_cal.to_dataframe().reset_index()
            X_test_df = data_test_cal.to_dataframe().reset_index()

            # storing the parameters used in this fold
            parameters.append({
                'ss_alpha': np.round(data_train_cal['ss_alpha'].values,4),
                'k_b': np.round(data_train_cal['k_b'].values,4)
            })

            # -------------------------
            # Split features/target
            # -------------------------
            drop_cols = ['model_bias_cap','case_index','pw_power_cap','ref_power_cap',
                         'wind_direction','wind_farm','flow_case','sample','ss_alpha','k_b','LMO']

            X_train = X_train_df.drop(columns=drop_cols)
            X_test  = X_test_df.drop(columns=drop_cols)

            y_train = X_train_df['model_bias_cap'].values.ravel()
            y_test  = X_test_df['model_bias_cap'].values.ravel()

            # Finding the raw engineering model and reference data subset associated with the test data points
            validation_fold=validation_data.isel(sample=cal_idx).to_dataframe().reset_index()
            validation_fold=validation_fold.drop(columns=['case_index','sample','k_b','ss_alpha','wind_farm','flow_case'])
            val = validation_fold.iloc[test_idx]
            pw = val['pw_power_cap'].values
            ref = val['ref_power_cap'].values

            # -------------------------
            # Fit models
            # -------------------------
            pipeline=self.pipeline
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)


            y_preds.append(y_pred)
            y_reals.append(y_test)

            # Standard metrics
            metrics = {m: f(y_test, y_pred) for m, f in scoring_standard.items()}

            # Bias metrics
            for m, f in scoring_pw.items():
                metrics[m] = f(ref, pw) if m == 'pw_bias' else f(ref, pw, y_pred)

            results.append(metrics)

        return results, parameters, y_reals, y_preds

    # -------------------------
    # Run
    # -------------------------
    def run(self):
        print("Preprocessing the data...")
        validation_data, groups = self.preprocess_data()
        print("Training the model and evaluating cross-validation metrics...")
        results, parameters, y_reals, y_preds = self.train(groups, validation_data)
        return results, parameters, y_reals, y_preds

# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    xr_data = xr.load_dataset("results_stacked_hh.nc")

    pipe_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
    ])
    
    predictor = BiasPredictor(pipeline=pipe_xgb,data=xr_data)
    cv_df, parameters, y_reals, y_preds = predictor.run()
