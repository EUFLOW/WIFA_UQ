import xarray as xr
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, KFold, cross_validate
from sklearn.metrics import make_scorer, mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import uniform, randint
import joblib

if not hasattr(np, 'int'):
    np.int = int

from sliced import SlicedInverseRegression

class BiasPredictor:
    def __init__(self, data:xr.Dataset):
        self.data = data # hub height data

    def preprocess_data(self):
        """
        Preprocessing the data (removing unused columns, splitting into features and target)
        """

        # Convert xarray Dataset to pandas DataFrame & drop unused columns
        df = self.data.to_dataframe().reset_index()
        # df['log_z0'] = np.log(df['z0'].values)

        X = df.drop(['sample', 'case_index', 'wind_farm', 'flow_case','model_bias_cap',
                     'wind_direction','ss_alpha','k_b', 'pw_power_cap', 'ref_power_cap'], axis=1)#,'nt','turb_rated_power'
        y = df[['model_bias_cap']]
        validation_data=df[['turb_rated_power', 'pw_power_cap', 'ref_power_cap']]

        return X, y, validation_data

    def train(self,X:pd.DataFrame, y:pd.Series, validation_data:pd.DataFrame):
        """
        training the model
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
            ("scaler", StandardScaler()),
            ("dim_reduction", SlicedInverseRegression()),
            ("model", LinearRegression())
        ])

        pipe_xgb_reduced = Pipeline([
            ("scaler", StandardScaler()),
            ("dim_reduction", SlicedInverseRegression()),
            ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
        ])
    
        # --- Defining distributions for random hyperparameter searches for each model ---
        # (not doing hyperparameter searches here... potentially overfitting on small dataset)
        # param_dist = [
        #     {
        #         "model": [LinearRegression()],
        #         "model__fit_intercept": [True, False]
        #     },
        #     {
        #         "model": [xgb.XGBRegressor()],
        #         "model__n_estimators": randint(50, 150),
        #         "model__max_depth": randint(2, 10),
        #         "model__learning_rate": uniform(0.01, 0.1),
        #         "model__subsample": uniform(0.5, 0.5)
        #     }
        # ]

        # Nested CV loop 
        # Within each loop find a best model based on RMSE by hyperparameter tuning
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=0)

        outer_metrics = []
        outer_metrics_xgb=[]
        outer_metrics_linear_reduced = []
        outer_metrics_xgb_reduced = []

        # Generate the following metrics using the best model on the unseen data
        scoring_metrics_standard = {
            'r2': r2_score,
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error
        }

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

        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            y_train = y_train.values.ravel()
            y_test = y_test.values.ravel()

            # Validation subset for pw/ref metrics
            val = validation_data.iloc[test_idx]
            pw = val['pw_power_cap'].values
            ref = val['ref_power_cap'].values

            # --- Fit models ---
            models = {
                "linear": pipe_linear,
                "xgb": pipe_xgb,
                "linear_reduced": pipe_linear_reduced,
                "xgb_reduced": pipe_xgb_reduced
            }

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                # Standard regression metrics
                metrics = {m: f(y_test, y_pred) for m, f in scoring_metrics_standard.items()}

                # pw/ref metrics
                for m, f in scoring_metrics_pw.items():
                    if m == "pw_bias":
                        metrics[m] = f(ref, pw)
                    else:  # pw_bias_corrected (y_pred here is the predicted bias for this test dataset)
                        metrics[m] = f(ref, pw, y_pred)

                results[name].append(metrics)
        for name in results:
            results[name] = pd.DataFrame(results[name])

        return results

    def run(self):
        """
        main function to run the bias predictor
        """

        # Preprocessing the data (removing unused columns, splitting into features and target)
        print("Preprocessing the data...")
        X, y,validation_data = self.preprocess_data()

        # train the model
        print("Nested cross validation... hyperparameter optimization and evaluation on unseen data in each fold")
        results = self.train(X, y,validation_data)

        # # --- Save the best model ---
        # joblib.dump(best_model, 'best_bias_predictor.pkl')

        return results

if __name__ == "__main__":
    # Just testing
    xr_data=xr.load_dataset("results_stacked_hh_best_sample.nc")

    # Creating an instance of the BiasPredictor class
    predictor = BiasPredictor(xr_data)

    # Running the predictor
    cv_df = predictor.run()

""".
Issues / To do:
"""