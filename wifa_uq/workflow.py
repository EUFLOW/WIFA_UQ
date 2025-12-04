import yaml
import xarray as xr
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import pandas as pd
import numpy as np

from wifa_uq.preprocessing.preprocessing import PreprocessingInputs
from wifa_uq.postprocessing.error_predictor.error_predictor import PCERegressor
from wifa_uq.model_error_database.database_gen import DatabaseGenerator
from wifa_uq.postprocessing.error_predictor.error_predictor import (
    BiasPredictor,
    MainPipeline,
    run_cross_validation,
    compute_metrics,
    run_observation_sensitivity,
    SIRPolynomialRegressor  
)
from wifa_uq.postprocessing.bayesian_calibration import (
    BayesianCalibration, 
    BayesianCalibrationWrapper
)
from wifa_uq.postprocessing.calibration import (
    MinBiasCalibrator,
    DefaultParams,
    LocalParameterPredictor
)

# --- Dynamic Class Loading ---
CLASS_MAP = {
    # Calibrators
    "MinBiasCalibrator": MinBiasCalibrator,
    "DefaultParams": DefaultParams,
    "LocalParameterPredictor": LocalParameterPredictor,
    # Bayesian 
    "BayesianCalibration": BayesianCalibrationWrapper,
    # Predictors
    "BiasPredictor": BiasPredictor,
    # ML Models
    "XGBRegressor": xgb.XGBRegressor,
    "SIRPolynomialRegressor": SIRPolynomialRegressor
}

CALIBRATION_MODES = {
    "MinBiasCalibrator": "global",
    "DefaultParams": "global",
    "LocalParameterPredictor": "local",
    "BayesianCalibration": "global"
}

def get_class_from_map(class_name: str):
    if class_name not in CLASS_MAP:
        raise ValueError(f"Unknown class '{class_name}' in config. "
                         f"Available classes are: {list(CLASS_MAP.keys())}")
    return CLASS_MAP[class_name]


def build_predictor_pipeline(model_name: str, model_params: dict | None = None):
    """
    Factory function to build the predictor pipeline based on config.
    Returns the pipeline and a 'model_type' string for SHAP logic.
    """
    if model_params is None:
       model_params = {}
    if model_name == "XGB":
        print("Building XGBoost Regressor pipeline...")
        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500))
        ])
        model_type = "tree"
        
    elif model_name == "SIRPolynomial":
        print("Building SIR+Polynomial Regressor pipeline...")
        # Note: SIRPolynomialRegressor includes its own scaling
        pipeline = SIRPolynomialRegressor(n_directions=1, degree=2)
        model_type = "sir"
        
    elif model_name == "PCE":
        print("Building PCE Regressor pipeline...")

        # model_params come directly from YAML (error_prediction.model_params)
        # They are passed as kwargs to PCERegressor, e.g.:
        #   degree, marginals, copula, q, max_features, allow_high_dim

        pipeline = PCERegressor(**model_params)
        model_type = "pce" 
    else:
        raise ValueError(f"Unknown model '{model_name}' in config. "
                        f"Available models are: ['XGB', 'SIRPolynomial', 'PCE']")
    return pipeline, model_type

def run_workflow(config_path: str | Path):
    """
    Runs the full WIFA-UQ workflow from a configuration file.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # --- 0. Resolve Paths ---
    base_dir = config_path.parent  
    paths_config = config['paths']
    system_yaml_path = base_dir / paths_config['system_config']
    ref_power_path = base_dir / paths_config['reference_power']
    ref_resource_path = base_dir / paths_config['reference_resource']
    wf_layout_path = base_dir / paths_config['wind_farm_layout']
    
    output_dir = base_dir / paths_config['output_dir']
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_resource_path = output_dir / paths_config['processed_resource_file']
    database_path = output_dir / paths_config['database_file']
    
    print(f"Resolved output directory: {output_dir}")

    # === 1. PREPROCESSING STEP ===
    if config['preprocessing']['run']:
        print("--- Running Preprocessing ---")
        preprocessor = PreprocessingInputs(
            ref_resource_path=ref_resource_path,
            output_path=processed_resource_path,
            steps=config['preprocessing'].get('steps', [])
        )
        preprocessor.run_pipeline()
        print("Preprocessing complete.")
    else:
        print("--- Skipping Preprocessing (as per config) ---")
        processed_resource_path = ref_resource_path
        if not processed_resource_path.exists():
             raise FileNotFoundError(f"Input resource file not found: {processed_resource_path}")
        print(f"Using raw resource file: {processed_resource_path.name}")

    # === 2. DATABASE GENERATION STEP ===
    if config['database_gen']['run']:
        print("--- Running Database Generation ---")
        param_config = config['database_gen']['param_config']
        
        db_generator = DatabaseGenerator(
            nsamples=config['database_gen']['n_samples'],
            param_config=param_config,
            system_yaml_path=system_yaml_path,
            ref_power_path=ref_power_path,
            processed_resource_path=processed_resource_path,
            wf_layout_path=wf_layout_path,
            output_db_path=database_path,
            model=config['database_gen']['flow_model']
        )
        database = db_generator.generate_database()
        print("Database generation complete.")
    else:
        # ... (same as before) ...
        print("--- Loading Existing Database (as per config) ---")
        if not database_path.exists():
            raise FileNotFoundError(f"Database file not found at {database_path}. "
                                    "Set 'database_gen.run = true' to generate it.")
        database = xr.load_dataset(database_path)
        print(f"Database loaded from {database_path}")

    # --- 3. SENSITIVITY ANALYSIS (ON OBSERVATIONS) ---
    print(f"Database loaded from {database_path}")

    # --- 3. SENSITIVITY ANALYSIS (ON OBSERVATIONS) ---
    sa_config = config.get('sensitivity_analysis', {})
    err_config = config['error_prediction'] ## NEW ##
    model_name = err_config.get('model', 'XGB') ## NEW ##
    model_params = err_config.get('model_params', {})

    if sa_config.get('run_observation_sensitivity', False):
        print(f"--- Running Observation Sensitivity for model: {model_name} ---")
        # Build a fresh pipeline just for this
        obs_pipeline, obs_model_type = build_predictor_pipeline(model_name, model_params)
        
        run_observation_sensitivity(
            database=database,
            features_list=err_config['features'],
            ml_pipeline=obs_pipeline,
            model_type=obs_model_type,
            output_dir=output_dir,
            method=sa_config.get('method', 'auto'),
            pce_config=sa_config.get('pce_config', {})
        )
    else:
        print("--- Skipping Observation Sensitivity (as per config) ---")

    # === 4. ERROR PREDICTION / UQ STEP ===
    if err_config['run']: ## MODIFIED ##
        print("--- Running Error Prediction ---")
        
        # Get the predictor pipeline and its type
        ml_pipeline, model_type = build_predictor_pipeline(model_name, model_params)
        
        calibrator_name = err_config['calibrator']
        calibration_mode = CALIBRATION_MODES.get(calibrator_name, "global")
        Calibrator_cls = get_class_from_map(err_config['calibrator'])
        Predictor_cls = get_class_from_map(err_config['bias_predictor'])
        MainPipeline_cls = MainPipeline
        
        print(f"Running cross-validation with calibrator: {Calibrator_cls.__name__} "
              f"and predictor: {model_name}")
              
        cv_df, y_preds, y_tests = run_cross_validation(
            xr_data=database,
            ML_pipeline=ml_pipeline,
            model_type=model_type, ## NEW ##
            Calibrator_cls=Calibrator_cls,
            BiasPredictor_cls=Predictor_cls,
            MainPipeline_cls=MainPipeline_cls,
            cv_config=err_config['cross_validation'],
            features_list=err_config['features'],
            output_dir=output_dir,
            sa_config=sa_config,
            calibration_mode=calibration_mode,
        )
        
        print("--- Cross-Validation Results (mean) ---")
        
        print("--- Cross-Validation Results (mean) ---")
        print(cv_df.mean())
        
        # Save results
        cv_df.to_csv(output_dir / "cv_results.csv")
        np.savez(output_dir / "predictions.npz", 
                 y_preds=np.array(y_preds, dtype=object), 
                 y_tests=np.array(y_tests, dtype=object))
        print(f"Results saved to {output_dir}")
    
    print("--- Workflow complete ---")

    return cv_df, y_preds, y_tests
