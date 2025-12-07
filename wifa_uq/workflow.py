"""
WIFA-UQ Workflow Orchestrator.

Supports both single-farm and multi-farm configurations with automatic
path inference from windIO system configs.
"""

import yaml
import xarray as xr
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import numpy as np

from wifa_uq.model_error_database.path_inference import (
    infer_paths_from_system_config,
    validate_required_paths,
)
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
from wifa_uq.postprocessing.bayesian_calibration import BayesianCalibrationWrapper
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
    "BayesianCalibration": BayesianCalibrationWrapper,
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
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", xgb.XGBRegressor(max_depth=3, n_estimators=500)),
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


def _resolve_paths_from_config(paths_config: dict, base_dir: Path) -> dict[str, Path]:
    """
    Resolve paths from config, using smart inference when paths are not explicit.

    This is the unified path resolution logic for both single-farm and multi-farm.

    Args:
        paths_config: The 'paths' section of the config
        base_dir: Base directory for relative paths (usually config file's parent)

    Returns:
        Dict with resolved Path objects for:
            - system_config
            - reference_power
            - reference_resource
            - wind_farm_layout
            - output_dir
            - processed_resource_file (name only)
            - database_file (name only)
    """
    # system_config is always required
    if "system_config" not in paths_config:
        raise ValueError(
            "Config must specify 'paths.system_config'. "
            "Other paths can be inferred from the windIO structure."
        )

    system_config_path = base_dir / paths_config["system_config"]

    # Build explicit paths dict (only include paths that are actually specified)
    explicit_paths = {}
    for key in ["reference_power", "reference_resource", "wind_farm_layout"]:
        if key in paths_config and paths_config[key] is not None:
            explicit_paths[key] = base_dir / paths_config[key]

    # Use smart inference (explicit paths override inferred ones)
    print(f"Resolving paths from system_config: {system_config_path.name}")
    resolved = infer_paths_from_system_config(
        system_config_path=system_config_path,
        explicit_paths=explicit_paths,
    )

    # Validate that all required paths exist
    validate_required_paths(resolved)

    # Print what we found
    print("Resolved paths:")
    for key, path in resolved.items():
        source = "explicit" if key in explicit_paths else "inferred"
        print(f"  {key}: {path.name} ({source})")

    # Add output paths (these are always explicit or have defaults)
    resolved["output_dir"] = base_dir / paths_config.get(
        "output_dir", "wifa_uq_results"
    )
    resolved["processed_resource_file"] = paths_config.get(
        "processed_resource_file", "processed_physical_inputs.nc"
    )
    resolved["database_file"] = paths_config.get(
        "database_file", "results_stacked_hh.nc"
    )

    return resolved


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
        missing = required_keys - set(farm.keys())
        if missing:
            raise ValueError(
                f"Farm #{i+1} is missing required keys: {missing}. "
                f"Each farm must have 'name' and 'system_config'."
            )

        name = farm["name"]
        if name in names_seen:
            raise ValueError(
                f"Duplicate farm name: '{name}'. Each farm must have a unique name."
            )
        names_seen.add(name)

    print(f"Validated {len(farms)} farm configurations")


def _resolve_farm_paths(farm_config: dict, base_dir: Path) -> dict:
    """
    Resolve paths for a single farm in multi-farm mode.

    Uses the same smart inference as single-farm mode.
    """
    system_config_path = base_dir / farm_config["system_config"]

    # Build explicit paths dict
    explicit_paths = {}
    for key in ["reference_power", "reference_resource", "wind_farm_layout"]:
        if key in farm_config and farm_config[key] is not None:
            explicit_paths[key] = base_dir / farm_config[key]

    # Use smart inference
    resolved = infer_paths_from_system_config(
        system_config_path=system_config_path,
        explicit_paths=explicit_paths,
    )

    # Add farm name
    resolved["name"] = farm_config["name"]

    return resolved


def run_workflow(config_path: str | Path):
    """
    Runs the full WIFA-UQ workflow from a configuration file.

    Supports both single-farm and multi-farm configurations.

    Path Resolution:
        Only 'system_config' is required. Other paths (reference_power,
        reference_resource, wind_farm_layout) can be:
        - Specified explicitly in the config (for full control)
        - Inferred automatically from the windIO system structure

    Example minimal config:
        paths:
          system_config: wind_energy_system.yaml
          output_dir: results/

        # Other paths will be auto-detected from windIO !include directives
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
    Single-farm workflow with smart path inference.
    """
    # --- 0. Resolve Paths (with smart inference) ---
    paths_config = config["paths"]
    resolved_paths = _resolve_paths_from_config(paths_config, base_dir)

    system_yaml_path = resolved_paths["system_config"]
    ref_power_path = resolved_paths["reference_power"]
    ref_resource_path = resolved_paths["reference_resource"]
    wf_layout_path = resolved_paths["wind_farm_layout"]

    output_dir = resolved_paths["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_resource_path = output_dir / resolved_paths["processed_resource_file"]
    database_path = output_dir / resolved_paths["database_file"]

    print(f"\nOutput directory: {output_dir}")
    print("Running in SINGLE-FARM mode\n")

    # === 1. PREPROCESSING STEP ===
    if config["preprocessing"]["run"]:
        print("--- Running Preprocessing ---")
        preprocessor = PreprocessingInputs(
            ref_resource_path=ref_resource_path,
            output_path=processed_resource_path,
            steps=config["preprocessing"].get("steps", []),
        )
        preprocessor.run_pipeline()
        print("Preprocessing complete.\n")
    else:
        print("--- Skipping Preprocessing (as per config) ---")
        processed_resource_path = ref_resource_path
        if not processed_resource_path.exists():
            raise FileNotFoundError(
                f"Input resource file not found: {processed_resource_path}"
            )
        print(f"Using raw resource file: {processed_resource_path.name}\n")

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
        print("Database generation complete.\n")
    else:
        print("--- Loading Existing Database (as per config) ---")
        if not database_path.exists():
            raise FileNotFoundError(
                f"Database file not found at {database_path}. "
                "Set 'database_gen.run = true' to generate it."
            )
        database = xr.load_dataset(database_path)
        print(f"Database loaded from {database_path}\n")

    # Continue with error prediction...
    return _run_error_prediction(config, database, output_dir)


def _run_multi_farm_workflow(config: dict, base_dir: Path):
    """
    Multi-farm workflow with smart path inference for each farm.
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

    print(f"Output directory: {output_dir}")
    print(f"Running in MULTI-FARM mode with {len(farms_config)} farms\n")

    # Resolve paths for each farm (using smart inference)
    resolved_farms = []
    print("Resolving farm paths:")
    for farm in farms_config:
        print(f"\n  Farm: {farm['name']}")
        try:
            resolved = _resolve_farm_paths(farm, base_dir)
            resolved_farms.append(resolved)
            print(f"    system_config: {resolved['system_config'].name}")
            print(f"    reference_power: {resolved['reference_power'].name}")
            print(f"    reference_resource: {resolved['reference_resource'].name}")
            print(f"    wind_farm_layout: {resolved['wind_farm_layout'].name}")
        except FileNotFoundError as e:
            print(f"    ERROR: {e}")
            raise

    print(f"\nSuccessfully resolved paths for {len(resolved_farms)} farms\n")

    # === DATABASE GENERATION ===
    if config["database_gen"]["run"]:
        print("--- Running Multi-Farm Database Generation ---")

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

        print("Multi-farm database generation complete.\n")
    else:
        print("--- Loading Existing Database ---")
        if not database_path.exists():
            raise FileNotFoundError(
                f"Database file not found at {database_path}. "
                "Set 'database_gen.run = true' to generate it."
            )
        database = xr.load_dataset(database_path)
        print(f"Database loaded from {database_path}")
        print(f"Contains {len(np.unique(database.wind_farm.values))} farms\n")

    # Continue with error prediction...
    return _run_error_prediction(config, database, output_dir)


def _run_error_prediction(config: dict, database: xr.Dataset, output_dir: Path):
    """
    Run error prediction and sensitivity analysis.

    Shared between single-farm and multi-farm workflows.
    """
    sa_config = config.get("sensitivity_analysis", {})
    err_config = config["error_prediction"]
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

        print("\n--- Cross-Validation Results (mean) ---")
        print(cv_df.mean())

        # Save results
        cv_df.to_csv(output_dir / "cv_results.csv")
        np.savez(
            output_dir / "predictions.npz",
            y_preds=np.array(y_preds, dtype=object),
            y_tests=np.array(y_tests, dtype=object),
        )
        print(f"\nResults saved to {output_dir}")

        print("\n--- Workflow complete ---")
        return cv_df, y_preds, y_tests

    print("\n--- Workflow complete (no error prediction) ---")
    return None, None, None
