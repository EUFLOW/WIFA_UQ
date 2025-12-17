# wifa_uq/workflow.py
"""
Main workflow orchestration for WIFA-UQ.

Supports single-farm and multi-farm configurations with optional physics insights.
"""

import yaml
import xarray as xr
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np

from wifa_uq.model_error_database.multi_farm_gen import generate_multi_farm_database
from wifa_uq.preprocessing.preprocessing import PreprocessingInputs
from wifa_uq.postprocessing.error_predictor.error_predictor import PCERegressor
from wifa_uq.model_error_database.database_gen import DatabaseGenerator
from wifa_uq.postprocessing.error_predictor.error_predictor import (
    BiasPredictor,
    MainPipeline,
    run_cross_validation,
    run_observation_sensitivity,
    SIRPolynomialRegressor,
)

# from wifa_uq.postprocessing.bayesian_calibration import BayesianCalibrationWrapper
from wifa_uq.postprocessing.calibration import (
    MinBiasCalibrator,
    DefaultParams,
    LocalParameterPredictor,
)

# --- Dynamic Class Loading ---
CLASS_MAP = {
    # Calibrators
    "MinBiasCalibrator": MinBiasCalibrator,
    "DefaultParams": DefaultParams,
    "LocalParameterPredictor": LocalParameterPredictor,
    # Bayesian
    # "BayesianCalibration": BayesianCalibrationWrapper,
    # Predictors
    "BiasPredictor": BiasPredictor,
    # ML Models
    "XGBRegressor": xgb.XGBRegressor,
    "SIRPolynomialRegressor": SIRPolynomialRegressor,
}

CALIBRATION_MODES = {
    "MinBiasCalibrator": "global",
    "DefaultParams": "global",
    "LocalParameterPredictor": "local",
    "BayesianCalibration": "global",
}


def get_class_from_map(class_name: str):
    if class_name not in CLASS_MAP:
        raise ValueError(
            f"Unknown class '{class_name}' in config. "
            f"Available classes are: {list(CLASS_MAP.keys())}"
        )
    return CLASS_MAP[class_name]


def build_predictor_pipeline(model_name: str, model_params: dict | None = None):
    """
    Factory function to build the predictor pipeline based on config.
    Returns the pipeline and a 'model_type' string for SHAP logic.
    """
    if model_params is None:
        model_params = {}

    if model_name == "Linear":
        from wifa_uq.postprocessing.error_predictor.error_predictor import (
            LinearRegressor,
        )

        print(
            f"Building Linear Regressor (method={model_params.get('method', 'ols')})..."
        )
        pipeline = LinearRegressor(**model_params)
        model_type = "linear"
        return pipeline, model_type

    if model_name == "XGB":
        print("Building XGBoost Regressor pipeline...")
        xgb_params = {
            "max_depth": model_params.get("max_depth", 3),
            "n_estimators": model_params.get("n_estimators", 500),
            "learning_rate": model_params.get("learning_rate", 0.1),
            "random_state": model_params.get("random_state", 42),
        }
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", xgb.XGBRegressor(**xgb_params)),
            ]
        )
        model_type = "tree"

    elif model_name == "SIRPolynomial":
        print("Building SIR+Polynomial Regressor pipeline...")
        pipeline = SIRPolynomialRegressor(n_directions=1, degree=2)
        model_type = "sir"

    elif model_name == "PCE":
        print("Building PCE Regressor pipeline...")
        pipeline = PCERegressor(**model_params)
        model_type = "pce"
    else:
        raise ValueError(
            f"Unknown model '{model_name}' in config. "
            f"Available models are: ['XGB', 'SIRPolynomial', 'PCE', 'Linear']"
        )
    return pipeline, model_type


def _is_multi_farm_config(config: dict) -> bool:
    """Check if config specifies multiple farms."""
    return "farms" in config.get("paths", {}) or "farms" in config


def _validate_farm_configs(farms: list[dict]) -> None:
    """
    Validate farm configurations.

    Each farm must have:
      - name: Unique identifier for cross-validation grouping
      - system_config: Path to wind energy system YAML
    """
    required_keys = {"name", "system_config"}
    names_seen = set()

    for i, farm in enumerate(farms):
        # Check required keys
        missing = required_keys - set(farm.keys())
        if missing:
            raise ValueError(
                f"Farm #{i + 1} is missing required keys: {missing}. "
                f"Each farm must have 'name' and 'system_config'."
            )

        # Check for duplicate names
        name = farm["name"]
        if name in names_seen:
            raise ValueError(
                f"Duplicate farm name: '{name}'. Each farm must have a unique name."
            )
        names_seen.add(name)

    print(f"Validated {len(farms)} farm configurations")


def _resolve_farm_paths(farm_config: dict, base_dir: Path) -> dict:
    """
    Resolve relative paths in a farm config to absolute paths.

    Required keys:
      - name: Farm identifier (passed through as-is)
      - system_config: Path to wind energy system YAML

    Optional keys (for explicit path overrides):
      - reference_power: Path to reference power NetCDF
      - reference_resource: Path to reference resource NetCDF
      - wind_farm_layout: Path to wind farm layout YAML
    """
    resolved = {"name": farm_config["name"]}

    path_keys = [
        "system_config",
        "reference_power",
        "reference_resource",
        "wind_farm_layout",
    ]

    for key in path_keys:
        if key in farm_config:
            resolved[key] = base_dir / farm_config[key]

    return resolved


def run_workflow(config_path: str | Path):
    """
    Runs the full WIFA-UQ workflow from a configuration file.

    Supports both single-farm and multi-farm configurations.
    """
    config_path = Path(config_path).resolve()
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    base_dir = config_path.parent

    # Detect single vs multi-farm mode
    is_multi_farm = _is_multi_farm_config(config)

    if is_multi_farm:
        return _run_multi_farm_workflow(config, base_dir)
    else:
        return _run_single_farm_workflow(config, base_dir)


def _run_single_farm_workflow(config: dict, base_dir: Path):
    """
    Original single-farm workflow (existing implementation).
    """
    from wifa_uq.model_error_database.path_inference import (
        infer_paths_from_system_config,
        validate_required_paths,
    )

    # --- 0. Resolve Paths ---
    paths_config = config["paths"]
    system_yaml_path = base_dir / paths_config["system_config"]

    # Build explicit paths dict for any paths that were provided
    explicit_paths = {}
    for key in ["reference_power", "reference_resource", "wind_farm_layout"]:
        if key in paths_config and paths_config[key] is not None:
            explicit_paths[key] = base_dir / paths_config[key]

    # Infer missing paths from windIO structure
    resolved_paths = infer_paths_from_system_config(
        system_config_path=system_yaml_path,
        explicit_paths=explicit_paths,
    )

    # Validate all required paths exist
    validate_required_paths(resolved_paths)

    # Extract resolved paths
    ref_power_path = resolved_paths["reference_power"]
    ref_resource_path = resolved_paths["reference_resource"]
    wf_layout_path = resolved_paths["wind_farm_layout"]

    output_dir = base_dir / paths_config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_resource_path = output_dir / paths_config["processed_resource_file"]
    database_path = output_dir / paths_config["database_file"]

    print(f"Resolved output directory: {output_dir}")
    print(f"Resolved reference_power: {ref_power_path}")
    print(f"Resolved reference_resource: {ref_resource_path}")
    print(f"Resolved wind_farm_layout: {wf_layout_path}")
    print("Running in SINGLE-FARM mode")

    # === 1. PREPROCESSING STEP ===
    if config["preprocessing"]["run"]:
        print("--- Running Preprocessing ---")
        preprocessor = PreprocessingInputs(
            ref_resource_path=ref_resource_path,
            output_path=processed_resource_path,
            steps=config["preprocessing"].get("steps", []),
        )
        preprocessor.run_pipeline()
        print("Preprocessing complete.")
    else:
        print("--- Skipping Preprocessing (as per config) ---")
        processed_resource_path = ref_resource_path
        if not processed_resource_path.exists():
            raise FileNotFoundError(
                f"Input resource file not found: {processed_resource_path}"
            )
        print(f"Using raw resource file: {processed_resource_path.name}")

    # === 2. DATABASE GENERATION STEP ===
    if config["database_gen"]["run"]:
        print("--- Running Database Generation ---")
        param_config = config["database_gen"]["param_config"]

        db_generator = DatabaseGenerator(
            nsamples=config["database_gen"]["n_samples"],
            param_config=param_config,
            system_yaml_path=system_yaml_path,
            ref_power_path=ref_power_path,
            processed_resource_path=processed_resource_path,
            wf_layout_path=wf_layout_path,
            output_db_path=database_path,
            model=config["database_gen"]["flow_model"],
        )
        database = db_generator.generate_database()
        print("Database generation complete.")
    else:
        print("--- Loading Existing Database (as per config) ---")
        if not database_path.exists():
            raise FileNotFoundError(
                f"Database file not found at {database_path}. "
                "Set 'database_gen.run = true' to generate it."
            )
        database = xr.load_dataset(database_path)
        print(f"Database loaded from {database_path}")

    # Continue with error prediction...
    return _run_error_prediction(config, database, output_dir)


def _run_multi_farm_workflow(config: dict, base_dir: Path):
    """
    Multi-farm workflow - processes multiple farms and combines results.
    """
    paths_config = config.get("paths", {})

    # Get farm configurations
    farms_config = config.get("farms") or paths_config.get("farms")
    if not farms_config:
        raise ValueError("Multi-farm config requires 'farms' list")

    # Validate farm configs
    _validate_farm_configs(farms_config)

    # Resolve output directory
    output_dir = base_dir / paths_config.get("output_dir", "multi_farm_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    database_filename = paths_config.get("database_file", "results_stacked_hh.nc")
    database_path = output_dir / database_filename

    print(f"Resolved output directory: {output_dir}")
    print(f"Running in MULTI-FARM mode with {len(farms_config)} farms")

    # Resolve paths for each farm
    resolved_farms = [_resolve_farm_paths(farm, base_dir) for farm in farms_config]

    # Print farm summary
    print("\nFarms to process:")
    for farm in resolved_farms:
        print(f"  - {farm['name']}: {farm['system_config']}")

    # === DATABASE GENERATION ===
    if config["database_gen"]["run"]:
        print("\n--- Running Multi-Farm Database Generation ---")

        database = generate_multi_farm_database(
            farm_configs=resolved_farms,
            param_config=config["database_gen"]["param_config"],
            n_samples=config["database_gen"]["n_samples"],
            output_dir=output_dir,
            database_file=database_filename,
            model=config["database_gen"]["flow_model"],
            preprocessing_steps=config["preprocessing"].get("steps", []),
            run_preprocessing=config["preprocessing"].get("run", True),
        )

        print("Multi-farm database generation complete.")
    else:
        print("--- Loading Existing Database ---")
        if not database_path.exists():
            raise FileNotFoundError(
                f"Database file not found at {database_path}. "
                "Set 'database_gen.run = true' to generate it."
            )
        database = xr.load_dataset(database_path)
        print(f"Database loaded from {database_path}")
        print(f"Contains {len(np.unique(database.wind_farm.values))} farms")

    # Continue with error prediction...
    return _run_error_prediction(config, database, output_dir)


def _run_error_prediction(config: dict, database: xr.Dataset, output_dir: Path):
    """
    Run error prediction, sensitivity analysis, and physics insights.

    Shared between single-farm and multi-farm workflows.
    """
    sa_config = config.get("sensitivity_analysis", {})
    err_config = config["error_prediction"]
    physics_config = config.get("physics_insights", {})
    model_name = err_config.get("model", "XGB")
    model_params = err_config.get("model_params", {})

    # --- OBSERVATION SENSITIVITY ---
    if sa_config.get("run_observation_sensitivity", False):
        print(f"--- Running Observation Sensitivity for model: {model_name} ---")
        obs_pipeline, obs_model_type = build_predictor_pipeline(
            model_name, model_params
        )

        run_observation_sensitivity(
            database=database,
            features_list=err_config["features"],
            ml_pipeline=obs_pipeline,
            model_type=obs_model_type,
            output_dir=output_dir,
            method=sa_config.get("method", "auto"),
            pce_config=sa_config.get("pce_config", {}),
        )
    else:
        print("--- Skipping Observation Sensitivity (as per config) ---")

    # === ERROR PREDICTION / UQ STEP ===
    fitted_model = None
    fitted_calibrator = None
    y_bias_all = None

    if err_config["run"]:
        print("--- Running Error Prediction ---")

        ml_pipeline, model_type = build_predictor_pipeline(model_name, model_params)

        calibrator_name = err_config["calibrator"]
        calibration_mode = CALIBRATION_MODES.get(calibrator_name, "global")
        Calibrator_cls = get_class_from_map(err_config["calibrator"])
        Predictor_cls = get_class_from_map(err_config["bias_predictor"])
        MainPipeline_cls = MainPipeline

        print(
            f"Running cross-validation with calibrator: {Calibrator_cls.__name__} "
            f"and predictor: {model_name}"
        )

        cv_df, y_preds, y_tests = run_cross_validation(
            xr_data=database,
            ML_pipeline=ml_pipeline,
            model_type=model_type,
            Calibrator_cls=Calibrator_cls,
            BiasPredictor_cls=Predictor_cls,
            MainPipeline_cls=MainPipeline_cls,
            cv_config=err_config["cross_validation"],
            features_list=err_config["features"],
            output_dir=output_dir,
            sa_config=sa_config,
            calibration_mode=calibration_mode,
            local_regressor=err_config.get("local_regressor"),
            local_regressor_params=err_config.get("local_regressor_params", {}),
        )

        print("--- Cross-Validation Results (mean) ---")
        print(cv_df.mean())

        # Save results
        cv_df.to_csv(output_dir / "cv_results.csv")
        np.savez(
            output_dir / "predictions.npz",
            y_preds=np.array(y_preds, dtype=object),
            y_tests=np.array(y_tests, dtype=object),
        )
        print(f"Results saved to {output_dir}")

        # --- FIT FINAL MODEL FOR PHYSICS INSIGHTS ---
        # Re-fit on all data for physics insights analysis
        if physics_config.get("run", False):
            print("--- Fitting Final Model for Physics Insights ---")

            # Build fresh pipeline
            final_pipeline, _ = build_predictor_pipeline(model_name, model_params)

            # Prepare full dataset
            X_df = database.isel(sample=0).to_dataframe().reset_index()
            features = err_config["features"]
            X = X_df[features]

            # Fit calibrator on full data
            if calibration_mode == "local":
                fitted_calibrator = Calibrator_cls(
                    database,
                    feature_names=features,
                    regressor_name=err_config.get("local_regressor"),
                    regressor_params=err_config.get("local_regressor_params", {}),
                )
            else:
                fitted_calibrator = Calibrator_cls(database)
            fitted_calibrator.fit()

            # Get bias values at calibrated parameters
            if calibration_mode == "local":
                optimal_indices = fitted_calibrator.get_optimal_indices()
                y_bias_all = np.array(
                    [
                        float(
                            database["model_bias_cap"]
                            .isel(case_index=i, sample=idx)
                            .values
                        )
                        for i, idx in enumerate(optimal_indices)
                    ]
                )
            else:
                best_idx = fitted_calibrator.best_idx_
                y_bias_all = database["model_bias_cap"].sel(sample=best_idx).values

            # Fit final model
            final_pipeline.fit(X, y_bias_all)
            fitted_model = final_pipeline

        # === PHYSICS INSIGHTS ===
        if physics_config.get("run", False) and fitted_model is not None:
            print("--- Running Physics Insights Analysis ---")
            from wifa_uq.postprocessing.physics_insights import run_physics_insights

            insights_dir = output_dir / "physics_insights"
            insights_dir.mkdir(exist_ok=True)

            run_physics_insights(
                database=database,
                fitted_model=fitted_model,
                calibrator=fitted_calibrator,
                features_list=err_config["features"],
                y_bias=y_bias_all,
                output_dir=insights_dir,
                config=physics_config,
            )

            print("Physics insights analysis complete.")

        print("--- Workflow complete ---")
        return cv_df, y_preds, y_tests

    print("--- Workflow complete (no error prediction) ---")
    return None, None, None
