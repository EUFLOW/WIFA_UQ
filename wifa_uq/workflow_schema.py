from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional, Union, Any
from pathlib import Path


class ParamConfigDict(BaseModel):
    """Full parameter configuration with explicit fields."""

    range: tuple[float, float]
    default: Optional[float] = None
    short_name: Optional[str] = None


class PreprocessingConfig(BaseModel):
    """Configuration for the preprocessing step."""

    run: bool = False
    steps: list[Literal["recalculate_params"]] = Field(default_factory=list)


class DatabaseGenConfig(BaseModel):
    """Configuration for database generation."""

    run: bool = False
    flow_model: Literal["pywake"] = "pywake"
    n_samples: int = 100
    # Accepts either [min, max] list or full dict with range/default/short_name
    param_config: dict[str, Union[ParamConfigDict, list[float]]] = Field(
        default_factory=dict
    )


class CrossValidationConfig(BaseModel):
    """Configuration for cross-validation."""

    run: bool = False
    splitting_mode: Literal["kfold_shuffled", "LeaveOneGroupOut"] = "kfold_shuffled"
    n_splits: int = 5
    metrics: list[Literal["rmse", "r2", "mae"]] = Field(
        default_factory=lambda: ["rmse", "r2", "mae"]
    )
    # Groups for LeaveOneGroupOut CV: maps group name -> list of farm names
    groups: Optional[dict[str, list[str]]] = None


class PCEModelParams(BaseModel):
    """Parameters specific to PCE model."""

    degree: int = 5
    marginals: Literal["kernel", "uniform", "normal"] = "kernel"
    copula: Literal["independent", "normal"] = "independent"
    q: float = 1.0
    max_features: int = 5
    allow_high_dim: bool = False


class LinearModelParams(BaseModel):
    """Parameters specific to Linear model."""

    method: Literal["ols", "ridge", "lasso", "elasticnet"] = "ols"
    alpha: float = 1.0
    l1_ratio: float = 0.5  # Only used for elasticnet


class XGBModelParams(BaseModel):
    """Parameters specific to XGBoost model."""

    max_depth: int = 3
    n_estimators: int = 500
    learning_rate: float = 0.1
    random_state: Optional[int] = None


class SIRModelParams(BaseModel):
    """Parameters specific to SIR+Polynomial model."""

    n_directions: int = 1
    degree: int = 2


class ErrorPredictionConfig(BaseModel):
    """Configuration for error prediction and cross-validation."""

    run: bool = False
    features: list[str]
    model: Literal["XGB", "PCE", "SIRPolynomial", "Linear"] = "XGB"
    # Model params - validated based on model type, or pass as generic dict
    model_params: dict[str, Any] = Field(default_factory=dict)
    calibrator: Literal[
        "MinBiasCalibrator",
        "LocalParameterPredictor",
        "DefaultParams",
        "BayesianCalibration",
    ] = "MinBiasCalibrator"
    # Only used when calibrator is LocalParameterPredictor
    local_regressor: Optional[
        Literal["Ridge", "Linear", "Lasso", "ElasticNet", "RandomForest", "XGB"]
    ] = None
    local_regressor_params: dict[str, Any] = Field(default_factory=dict)
    bias_predictor: Literal["BiasPredictor"] = "BiasPredictor"
    cross_validation: CrossValidationConfig = Field(
        default_factory=CrossValidationConfig
    )

    @model_validator(mode="after")
    def validate_local_calibrator(self):
        """Warn if local_regressor is set but calibrator is not LocalParameterPredictor."""
        if (
            self.local_regressor is not None
            and self.calibrator != "LocalParameterPredictor"
        ):
            import warnings

            warnings.warn(
                f"local_regressor is set to '{self.local_regressor}' but calibrator is "
                f"'{self.calibrator}'. local_regressor is only used with LocalParameterPredictor."
            )
        return self


class PCESensitivityConfig(BaseModel):
    """PCE-specific sensitivity analysis configuration."""

    degree: int = 5
    marginals: Literal["kernel", "uniform", "normal"] = "kernel"
    copula: Literal["independent", "normal"] = "independent"
    q: float = 0.5


class SensitivityConfig(BaseModel):
    """Configuration for sensitivity analysis."""

    run_observation_sensitivity: bool = False
    run_bias_sensitivity: bool = False
    method: Literal["auto", "shap", "pce_sobol", "sir"] = "auto"
    pce_config: Optional[PCESensitivityConfig] = None


class FarmConfig(BaseModel):
    """Configuration for a single farm in multi-farm mode."""

    name: str
    system_config: Path
    # These are optional - will be inferred from windIO structure if not provided
    reference_power: Optional[Path] = None
    reference_resource: Optional[Path] = None
    wind_farm_layout: Optional[Path] = None


class PathsConfig(BaseModel):
    """Path configuration for single-farm or output paths for multi-farm."""

    # Required for single-farm mode, not needed for multi-farm
    system_config: Optional[Path] = None
    # These are optional - will be inferred from windIO structure if not provided
    reference_power: Optional[Path] = None
    reference_resource: Optional[Path] = None
    wind_farm_layout: Optional[Path] = None
    # Output configuration
    output_dir: Path
    processed_resource_file: str = "processed_physical_inputs.nc"
    database_file: str = "results_stacked_hh.nc"


class WifaUQConfig(BaseModel):
    """
    Main WIFA-UQ workflow configuration.

    Supports two modes:
    1. Single-farm: Specify paths.system_config (other paths auto-inferred)
    2. Multi-farm: Specify farms list with each farm's system_config

    Example single-farm config:
        paths:
          system_config: wind_energy_system.yaml
          output_dir: results/
        preprocessing:
          run: true
          steps: [recalculate_params]
        ...

    Example multi-farm config:
        paths:
          output_dir: results/multi_farm/
        farms:
          - name: Farm1
            system_config: farm1/system.yaml
          - name: Farm2
            system_config: farm2/system.yaml
        ...
    """

    description: Optional[str] = None
    paths: PathsConfig
    # For multi-farm mode
    farms: Optional[list[FarmConfig]] = None
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    database_gen: DatabaseGenConfig = Field(default_factory=DatabaseGenConfig)
    error_prediction: ErrorPredictionConfig
    sensitivity_analysis: SensitivityConfig = Field(default_factory=SensitivityConfig)

    @model_validator(mode="after")
    def check_paths_or_farms(self):
        """Validate that either single-farm paths or multi-farm config is provided."""
        is_multi_farm = self.farms is not None and len(self.farms) > 0
        is_single_farm = self.paths.system_config is not None

        if not is_multi_farm and not is_single_farm:
            raise ValueError(
                "Configuration must specify either:\n"
                "  - paths.system_config (for single-farm mode), or\n"
                "  - farms list (for multi-farm mode)"
            )

        if is_multi_farm and is_single_farm:
            raise ValueError(
                "Cannot specify both paths.system_config and farms list. "
                "Choose single-farm or multi-farm mode."
            )

        return self

    @model_validator(mode="after")
    def check_logo_groups(self):
        """Validate LeaveOneGroupOut has groups defined in multi-farm mode."""
        cv_config = self.error_prediction.cross_validation
        if cv_config.splitting_mode == "LeaveOneGroupOut":
            if self.farms is None:
                raise ValueError(
                    "LeaveOneGroupOut splitting requires multi-farm mode (farms list must be specified)"
                )
            if cv_config.groups is None:
                # Groups are optional - if not specified, each farm becomes its own group
                pass
        return self

    def is_multi_farm(self) -> bool:
        """Check if this is a multi-farm configuration."""
        return self.farms is not None and len(self.farms) > 0
