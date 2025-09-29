import yaml
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from wifa_uq.model_error_database.database_gen import DatabaseGenerator
from wifa_uq.preprocessing.preprocessing import PreprocessingInputs
from wifa_uq.postprocessing.error_predictor.error_predictor import BiasPredictor
from pathlib import Path

def main(config_file):
    with open(config_file, "r") as f:
        user_config = yaml.safe_load(f)

    # location of data
    base_dir=user_config['preprocessing']['base_dir']
    base_dir = Path(base_dir)
    if user_config['preprocessing']['case_names']=='all':
        case_names= [f.name for f in base_dir.iterdir() if f.is_dir()]
    else:
        case_names = user_config['preprocessing']['case_names']

    # preprocessing
    if user_config['preprocessing']['run']:
        preprocessor = PreprocessingInputs(base_dir=base_dir, case_names=case_names)
        preprocessor.batch_update_params()


    if user_config['database_gen']['run']:
        nsamples = user_config['database_gen']['n_samples']
        param_config = user_config['database_gen']['param_config']
        param_config = {k: tuple(v) for k, v in param_config.items()}

        model = user_config['database_gen']['flow_model']
        location = user_config['database_gen']['location']
        db_generator = DatabaseGenerator(nsamples, param_config, base_dir, case_names, model=model,save_to=location)
        database=db_generator.generate_database()

    else: # if database already exists
        database = user_config['database_gen']['location']

    if user_config['error_prediction']['run']:
        pipe_xgb = Pipeline([
        ("scaler", StandardScaler()),
        ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
        ])
        predictor = BiasPredictor(pipeline=pipe_xgb,data=database)
        cv_df, parameters, y_reals, y_preds = predictor.run()

        return cv_df, parameters, y_reals, y_preds
    
    ## future planned logic for fitting and predicting and cross val (not defined yet)
    """.
    Goal here is to rewrite the error predictor pipeline to take in an arbitrary calibration class... 
    In that case steps here would simply be 
    - pipeline.fit (fit posterior distribution of parameters, and fit bias predictor as a function of parameter) 
    - pipeline.predict (predict bias on test points (functionality for single values or distributions of bias))

    (Do this within an updated cross validation routine)
    """

if __name__ == "__main__":
    cv_df, parameters, y_reals, y_preds = main("err_prediction_example.yaml")