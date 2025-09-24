from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import xgboost as xgb
from sklearn.feature_selection import VarianceThreshold
from sliced import SlicedInverseRegression


from database_gen import DatabaseGenerator
from preprocessing import PreprocessingInputs
from error_predictor import BiasPredictor

#%% Defining Inputs
base_dir = Path("EDF_datasets")

# Identifiers for the different wind farm simulations on windlab
case_names=[
    "HR1",   
    "HR2",     
    "HR3",
    "NYSTED1",   
    "NYSTED2",
    "VirtWF_ABL_IEA10", 
    "VirtWF_ABL_IEA15_ali_DX5_DY5",   
    "VirtWF_ABL_IEA15_stag_DX5_DY5",    
    "VirtWF_ABL_IEA15_stag_DX5_DY7p5",  
    "VirtWF_ABL_IEA15_stag_DX7p5_DY5",  
    "VirtWF_ABL_IEA22"
]

# defining ranges for the parameter samples
param_config = {
        "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": (0.01, 0.07),
        "attributes.analysis.blockage_model.ss_alpha": (0.75, 1.0)
    }

nsamples=1000  # First sample by definition will be default, then 100 additional random samples

wake_model='pywake' # currently only implemented with a specific deficit model and blockage model

ML_pipeline= Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
        ])

#%% carry out preprocessing
preprocessor = PreprocessingInputs(base_dir=base_dir, case_names=case_names)
preprocessor.batch_update_params()

#%% generate model bias database
db_generator = DatabaseGenerator(nsamples, param_config, base_dir, case_names, model="pywake")
database=db_generator.generate_database()

#%% execute error predictor and calculate cross validation results

# Examples of some pipelines
pipe_linear = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LinearRegression())
])
pipe_xgb = Pipeline([
    ("scaler", StandardScaler()),
    ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
])
pipe_linear_reduced = Pipeline([
    ("var_thresh", VarianceThreshold(threshold=0.0)),
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

predictor = BiasPredictor(pipeline=pipe_xgb,data=database)
cv_df, parameters, y_reals, y_preds = predictor.run()