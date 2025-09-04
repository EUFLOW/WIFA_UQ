import xarray as xr
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, KFold,LeaveOneGroupOut
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint
from sklearn.feature_selection import VarianceThreshold

from applicability_domain import mahalanobis_ad, pca_percentile_ad

if not hasattr(np, 'int'):
    np.int = int

from sliced import SlicedInverseRegression

class BiasPredictor:
    def __init__(self, data:xr.Dataset,cv_mode='default',calibration_mode='min_bias'):
        self.data = data # hub height data
        self.cv_mode = cv_mode
        self.calibration_mode = calibration_mode

    def preprocess_data(self):
        """
        Preprocessing the data (removing unused columns, splitting into features and target)
        """

        # Convert xarray Dataset to pandas DataFrame & drop unused columns
        # df = self.data.to_dataframe().reset_index()
        # df['log_z0'] = np.log(df['z0'].values)

        # X = self.data.drop_vars(['wind_farm','model_bias_cap',
        #              'wind_direction', 'pw_power_cap', 'ref_power_cap','flow_case'])#,'nt','turb_rated_power'
        # y = self.data['model_bias_cap']
        validation_data=self.data[['turb_rated_power', 'pw_power_cap', 'ref_power_cap']]
        groups = self.data['wind_farm'].values

        return validation_data,groups

    def train(self,groups, validation_data:pd.DataFrame):
        """
        training the model

        cv_mode: str
            The cross-validation mode to use ('default' for KFold, 'logo' for LeaveOneGroupOut)
        """

        # --- Initializing the pipeline ---
        # Additionally, the entire pipeline could be an input to the class (specify which models, which param_dists)
        pipe_linear = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()) 
        ])

        # increasing n_estimators increases r2 score significantly beyond linear regression
        # concerns about model fitting to noise
        # changing regularization or other parameters has limited effect
        pipe_xgb = Pipeline([
            ("scaler", StandardScaler()),  
            ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
        ])

        pipe_linear_reduced = Pipeline([
            ("var_thresh", VarianceThreshold(threshold=0.0)),  # SIR won't work if any features have zero variance
            ("scaler", StandardScaler()),
            ("dim_reduction", SlicedInverseRegression()),
            ("model", LinearRegression())
        ])

        pipe_xgb_reduced = Pipeline([
            ("var_thresh", VarianceThreshold(threshold=0.0)),
            ("scaler", StandardScaler()),
            ("dim_reduction", SlicedInverseRegression()),
            ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
        ])
    
        flow_cases=self.data.case_index
        # Nested CV loop 
        if self.cv_mode == 'logo':
            # Not sure about any of the below.... probably need to fix or redo
            groups=self.data['wind_farm'].values
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

        # Generate the following metrics using the best model on the unseen data
        scoring_metrics_standard = {
            'r2': r2_score,
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error
        }

        # These metrics compare the raw power predictions to the reference data for default pywake model and bias corrected model. Used within cross-validation
        scoring_metrics_pw = {
            'pw_bias': lambda ref, pw: np.mean(pw - ref),
            'pw_bias_corrected': lambda ref, pw, bias: np.mean((pw - bias) - ref)
        }

        results = {
        "linear": [],
        "xgb": [],
        "linear_reduced": [],
        "xgb_reduced": []
        }

        parameters=[]
        for train_idx, test_idx in splits:
            # now we have xarrays with just the training indices
            dataset_train = self.data.where(self.data.case_index.isin(train_idx), drop=True)
            dataset_test  = self.data.where(self.data.case_index.isin(test_idx), drop=True)

            # Option to calibrate the model parameters for each fold
            if self.calibration_mode=="min_bias":

                bias_array = dataset_train['model_bias_cap'].values  # (n parameter samples, npoints)
                abs_total_bias = np.abs(np.sum(bias_array, axis=1))
                idx = np.argmin(abs_total_bias)  

                # mses = np.mean(bias_array**2, axis=1)
                # idx = np.argmin(mses) 

            else: # if we are using the default parameters (the default parameters have sample index 0 by definition in the database generation script)
                idx=0

            # keeping only the sample chosen from the dataset (either calibrated or default parameters)
            dataset_train_cal=dataset_train.where(self.data['sample']==idx,drop=True)
            dataset_test_cal=dataset_test.where(self.data['sample']==idx,drop=True)
            dataset_train_cal=dataset_train_cal.to_dataframe().reset_index()
            dataset_test_cal=dataset_test_cal.to_dataframe().reset_index()

            # storing the parameters used in this fold
            parameters.append({
                'ss_alpha': round(dataset_train_cal['ss_alpha'].values[0],4),
                'k_b': round(dataset_train_cal['k_b'].values[0],4)
            })

            # Dropping unused columns
            X_train=dataset_train_cal.drop(columns=['model_bias_cap','case_index','pw_power_cap','ref_power_cap',
                                                            'wind_direction','wind_farm','flow_case','sample', 'ss_alpha', 'k_b','LMO'], axis=1)
            X_test=dataset_test_cal.drop(columns=['model_bias_cap', 'case_index', 'pw_power_cap', 'ref_power_cap',
                                            'wind_direction', 'wind_farm', 'flow_case', 'sample', 'ss_alpha', 'k_b','LMO'], axis=1)

            y_train,y_test=dataset_train_cal['model_bias_cap'].values.ravel(), dataset_test_cal['model_bias_cap'].values.ravel()

            # Finding the raw engineering model and reference data subset associated with the test data points
            validation_data_inloop=validation_data.isel(sample=idx).to_dataframe().reset_index()
            validation_data_inloop=validation_data_inloop.drop(columns=['case_index','sample','k_b','ss_alpha','wind_farm','flow_case'])
            val = validation_data_inloop.iloc[test_idx]
            pw = val['pw_power_cap'].values
            ref = val['ref_power_cap'].values

            # Fit model pipelines 
            models = {
                "linear": pipe_linear,
                "xgb": pipe_xgb,
                "linear_reduced": pipe_linear_reduced,
                "xgb_reduced": pipe_xgb_reduced
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Standard regression performance metrics
                metrics = {m: f(y_test, y_pred) for m, f in scoring_metrics_standard.items()}

                # Metrics which compare the pywake model to the reference data within each fold with and without bias correction
                for m, f in scoring_metrics_pw.items():
                    if m == "pw_bias":
                        metrics[m] = f(ref, pw)
                    else:  # pw_bias_corrected (y_pred here is the predicted bias for this test dataset)
                        metrics[m] = f(ref, pw, y_pred)

                results[name].append(metrics)
        for name in results:
            results[name] = pd.DataFrame(results[name])

        return results,parameters

    def run(self):
        """
        main function to run the bias predictor
        """

        # Preprocessing the data (This step previously carried out the data splitting before calibration was done within cross-validation folds)
        print("Preprocessing the data...")
        validation_data,groups = self.preprocess_data()

        # train the model
        print("Training the model and evaluating cross validation metrics...")

        results, parameters = self.train(groups,validation_data)

        # # --- Save the best model ---
        # joblib.dump(best_model, 'best_bias_predictor.pkl')

        return results,parameters

if __name__ == "__main__":
    # Just testing
    xr_data=xr.load_dataset("results_stacked_hh.nc")

    # Creating an instance of the BiasPredictor class
    predictor = BiasPredictor(xr_data)

    # Running the predictor
    cv_df,parameters = predictor.run()

""".
Issues / To do:
"""