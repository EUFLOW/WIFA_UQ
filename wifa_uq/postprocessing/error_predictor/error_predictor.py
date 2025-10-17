import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import xgboost as xgb  
from sklearn.model_selection import KFold, LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


"""
This script contains:
- Calibrator classes (basic approaches used so far)
- BiasPredictor class (carrying out machine learning pipeline for a given set of features and target)
- MainPipeline class (data processing, calibration, bias prediction)
- Cross validation routine

The main pipeline class is intended to be flexible and accomodate different calibration
classes which can be imported from other locations, or different bias predictor classes.
The only requirement is that they both contain .fit and .predict methods.

So far, this has only been tested with simple minimum bias calibration and XGBoost regression
and will have to be further developed to accomodate probabilistic calibration etc.
"""

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
    def __init__(self, calibrator, bias_predictor):
        self.calibrator = calibrator
        self.bias_predictor = bias_predictor

    def fit(self, dataset_train,dataset_test, drop_extra=[]):
        # 1. Fit calibrator (posterior distribution)
        self.calibrator.fit()
        idxs = self.calibrator.best_idx_
        params = self.calibrator.best_params_

        # 2. Prepare training data for bias predictor (changes here for parameter pdf or other calibration approaches)
        dataset_train_cal=dataset_train.sel(sample=idxs)
        dataset_test_cal=dataset_test.sel(sample=idxs)

        X_train_df = dataset_train_cal.to_dataframe().reset_index()
        X_test_df = dataset_test_cal.to_dataframe().reset_index()

        if isinstance(idxs,int):
            print('Single parameter: k=', params)
            drop_cols = ['wind_farm', 'flow_case','sample', 'case_index','model_bias_cap','data_driv_cap','nt','turb_rated_power',
                        'veer','veer_norm','shear','shear_norm','z0','Farm_Length', 'advection', 'mixing',
                        'capping_inversion_thickness','capping_inversion_strength','wind_direction','pw_power_cap',
                        'ref_power_cap','k_b','ss_alpha',] + ['farm_density','turbulence_intensity','Farm_Width','wind_speed'] + drop_extra
        else:
            print('Multiple parameters: k=', params)
            drop_cols = ['wind_farm', 'flow_case','sample', 'case_index','model_bias_cap','data_driv_cap','nt','turb_rated_power',
                        'veer','veer_norm','shear','shear_norm','z0','Farm_Length', 'advection', 'mixing',
                        'capping_inversion_thickness','capping_inversion_strength','wind_direction','pw_power_cap',
                        'ref_power_cap',] + ['farm_density','turbulence_intensity','Farm_Width','wind_speed'] + drop_extra
            


        X_train = X_train_df.drop(columns=drop_cols,errors='ignore')
        X_test  = X_test_df.drop(columns=drop_cols,errors='ignore')

        y_train = dataset_train_cal['model_bias_cap'].values
        y_test  = dataset_test_cal['model_bias_cap'].values

        # 3. Fit bias predictor
        self.bias_predictor.fit(X_train, y_train)

        return X_test,y_test,idxs # keeping idxs for later

    def predict(self, X=None):
        if X is None:  # use stored test data by default, but can use any datapoints ??
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

def run_cross_validation(xr_data, ML_pipeline,Calibrator_cls, BiasPredictor_cls, MainPipeline_cls):

    validation_data = xr_data[['turb_rated_power', 'pw_power_cap', 'ref_power_cap']]
    groups = xr_data['wind_farm'].values

    # cross validation routine details here are hard-coded... they could and should probably be inputs
    wf_to_group = {
    "HR1": 0, "HR2": 0, "HR3":0,
    "NYSTED1":1, "NYSTED2":1,
    "VirtWF_ABL_IEA10":2, 
    "VirtWF_ABL_IEA15_ali_DX5_DY5":3,   
    "VirtWF_ABL_IEA15_stag_DX5_DY5":4, "VirtWF_ABL_IEA15_stag_DX5_DY7p5":4, "VirtWF_ABL_IEA15_stag_DX7p5_DY5":4,  
    "VirtWF_ABL_IEA22":5
        }
            
    manual_groups = np.array([wf_to_group[w] for w in groups])
    cv = LeaveOneGroupOut()
    flow_cases = xr_data.case_index
    splits = cv.split(flow_cases, groups=manual_groups)

    stats_cv, y_preds, y_tests = [], [], []

    for train_idx, test_idx in splits:
        dataset_train = xr_data.where(xr_data.case_index.isin(train_idx), drop=True)
        dataset_test = xr_data.where(xr_data.case_index.isin(test_idx), drop=True)

        calibrator=Calibrator_cls(dataset_train)
        bias_pred=BiasPredictor_cls(ML_pipeline)
        main_pipe=MainPipeline_cls(calibrator, bias_pred)

        # may seem strange that this outputs the test data, but it is 
        # following the posterior calibration step where the database is reduced to
        # the calibrated parameter set
        x_test,y_test,idxs=main_pipe.fit(dataset_train, dataset_test)
        y_pred=main_pipe.predict(x_test)

        # rename these variables... not very intuitive
        validation_fold=validation_data.isel(sample=idxs).to_dataframe().reset_index()
        validation_fold=validation_fold.drop(columns=['case_index','sample','k_b','ss_alpha','wind_farm','flow_case'])
        val = validation_fold.iloc[test_idx]
        pw = val['pw_power_cap'].values
        ref = val['ref_power_cap'].values

        stats=compute_metrics(y_test, y_pred,pw=pw, ref=ref)
        stats_cv.append(stats)

        y_preds.append(y_pred)
        y_tests.append(y_test)

    cv_results = pd.DataFrame(stats_cv)
    return cv_results,y_preds,y_tests

# Testing
if __name__ == "__main__":
    import xarray as xr

    xr_data = xr.load_dataset("results_stacked_hh.nc")
    pipe_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
    ])
    cv_results,y_preds,y_tests=run_cross_validation(xr_data,pipe_xgb,MinBiasCalibrator, BiasPredictor, MainPipeline)