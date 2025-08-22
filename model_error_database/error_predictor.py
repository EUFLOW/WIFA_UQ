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

        X = df.drop(['sample', 'case_index', 'wind_farm', 'flow_case','power_bias_perc','wind_direction','ss_alpha'], axis=1)
        y = df[['power_bias_perc']]

        return X, y

    def dim_reduction(self,X:pd.DataFrame,y:pd.Series):
        """
        reducing the dimensions of the data

        (currently passed, using manual selection in preprocessing step)
        """

        return X,y
        

    def train(self,X:pd.DataFrame, y:pd.Series):
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
            ("model", xgb.XGBRegressor(max_depth=3, n_estimators=50, learning_rate=0.05, random_state=2
                                       ,reg_alpha=1.0,reg_lambda=10.0,subsample=0.7))
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
        outer_cv = KFold(n_splits=10, shuffle=True, random_state=42)
        outer_metrics = []
        outer_metrics_xgb=[]
        outer_models = []

        # Generate the following metrics using the best model on the unseen data
        scoring_metrics = {
            'r2': r2_score,
            'mse': mean_squared_error,
            'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error
        }
        for train_idx, test_idx in outer_cv.split(X, y):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]


            # # Inner Hyperparameter Tuning  # Not using for now
            # inner_cv=KFold(n_splits=5, shuffle=True, random_state=42)

            # random_search = RandomizedSearchCV(
            #     estimator=pipe,
            #     param_distributions=param_dist,
            #     n_iter=1,
            #     scoring='neg_root_mean_squared_error',  
            #     cv=inner_cv,
            #     refit=True,
            #     verbose=1)  # This means, after finding the best hyperparameters, the model
            # random_search.fit(X_train, y_train)

            # best_model = random_search.best_estimator_
            # y_pred = best_model.predict(X_test)


            pipe_linear.fit(X_train, y_train)
            y_pred_linear = pipe_linear.predict(X_test)
            pipe_xgb.fit(X_train, y_train)
            y_pred_xgb = pipe_xgb.predict(X_test)

            metrics_linear={metric: func(y_test, y_pred_linear) for metric, func in scoring_metrics.items()}
            metrics_xgb={metric: func(y_test, y_pred_xgb) for metric, func in scoring_metrics.items()}

            outer_metrics.append(metrics_linear)
            outer_metrics_xgb.append(metrics_xgb)

        df_metrics_linear=pd.DataFrame(outer_metrics)
        df_metrics_xgb=pd.DataFrame(outer_metrics_xgb)

        return df_metrics_linear, df_metrics_xgb


    def run(self):
        """
        main function to run the bias predictor
        """

        # Preprocessing the data (removing unused columns, splitting into features and target)
        print("Preprocessing the data...")
        X, y = self.preprocess_data()

        # reduce dimensions (placeholder for now)
        print("Running dimension reduction... currently a placeholder")
        X_red, y = self.dim_reduction(X, y)
        
        # train the model
        print("Nested cross validation... hyperparameter optimization and evaluation on unseen data in each fold")
        cv_df,cv_df_xgb = self.train(X_red, y)  #cv_df,best_model =

        # # --- Save the best model ---
        # joblib.dump(best_model, 'best_bias_predictor.pkl')
        
        cv_df.to_csv('cv_metrics.csv', index=False)
        cv_df_xgb.to_csv('cv_metrics_xgb.csv', index=False)

        print("Cross-validation metrics saved to CSV files")

        return cv_df,cv_df_xgb


# Ideally just remove the features not being used here... 
 
if __name__ == "__main__":
    # Just testing
    xr_data=xr.load_dataset("results_stacked_hh.nc")

    # Creating an instance of the BiasPredictor class
    predictor = BiasPredictor(xr_data)

    # Running the predictor
    cv_df = predictor.run()

""".
Issues / To do:
"""