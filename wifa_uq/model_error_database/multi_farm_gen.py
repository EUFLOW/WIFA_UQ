"""
Multi-farm database generator for cross-validation studies.

Generates combined databases from multiple wind farms, enabling
LeaveOneGroupOut cross-validation across farm groups.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import xarray as xr

from wifa_uq.model_error_database.database_gen import DatabaseGenerator
from wifa_uq.model_error_database.path_inference import (
    infer_paths_from_system_config,
    validate_required_paths,
)
from wifa_uq.preprocessing.preprocessing import PreprocessingInputs

logger = logging.getLogger(__name__)


class MultiFarmDatabaseGenerator:
    """
    Generates a combined database from multiple wind farms.

    Each farm config must specify:
      - name: Unique identifier for the farm
      - system_config: Path to wind energy system YAML (windIO format)

    Optional (paths are auto-inferred if not provided):
      - reference_power: Path to reference power NetCDF
      - reference_resource: Path to reference resource NetCDF
      - wind_farm_layout: Path to wind farm layout YAML

    Parameters
    ----------
    farm_configs : list[dict]
        List of farm configurations with 'name' and 'system_config' keys
        (and optionally resolved paths from workflow.py)

    param_config : dict
        Parameter sampling configuration (shared across all farms)

    n_samples : int
        Number of parameter samples per farm

    output_dir : Path
        Directory for output files

    database_file : str
        Name of combined database file

    model : str
        Flow model to use (default: 'pywake')

    preprocessing_steps : list[str]
        Preprocessing steps to apply to each farm

    run_preprocessing : bool
        Whether to run preprocessing
    """

    def __init__(
        self,
        farm_configs: list[dict],
        param_config: dict,
        n_samples: int,
        output_dir: Path,
        database_file: str = "combined_database.nc",
        model: str = "pywake",
        preprocessing_steps: list[str] = None,
        run_preprocessing: bool = True,
    ):
        self.farm_configs = farm_configs
        self.param_config = param_config
        self.n_samples = n_samples
        self.output_dir = Path(output_dir)
        self.database_file = database_file
        self.model = model
        self.preprocessing_steps = preprocessing_steps or []
        self.run_preprocessing = run_preprocessing

        self._validate_farm_configs()

    def _validate_farm_configs(self) -> None:
        """Validate farm configurations."""
        required_keys = {"name", "system_config"}

        names = []
        for i, farm in enumerate(self.farm_configs):
            missing = required_keys - set(farm.keys())
            if missing:
                raise ValueError(
                    f"Farm #{i+1} missing required keys: {missing}. "
                    f"Each farm must have 'name' and 'system_config'."
                )

            name = farm["name"]
            if name in names:
                raise ValueError(f"Duplicate farm name: '{name}'")
            names.append(name)

        logger.info(f"Validated {len(self.farm_configs)} farm configurations")

    def _ensure_farm_paths(self, farm_config: dict) -> dict:
        """
        Ensure all required paths are present for a farm.

        If paths were already resolved by workflow.py, use those.
        Otherwise, infer from system_config.
        """
        farm_name = farm_config["name"]

        # Check if paths are already resolved (Path objects present)
        required_paths = [
            "system_config",
            "reference_power",
            "reference_resource",
            "wind_farm_layout",
        ]
        all_resolved = all(
            key in farm_config and isinstance(farm_config.get(key), Path)
            for key in required_paths
        )

        if all_resolved:
            # Already resolved by workflow.py
            return farm_config

        # Need to infer paths
        system_config_path = Path(farm_config["system_config"])

        # Build explicit paths dict
        explicit_paths = {}
        for key in ["reference_power", "reference_resource", "wind_farm_layout"]:
            if key in farm_config and farm_config[key] is not None:
                explicit_paths[key] = Path(farm_config[key])

        print(f"  Inferring paths for {farm_name}...")
        resolved = infer_paths_from_system_config(
            system_config_path=system_config_path,
            explicit_paths=explicit_paths,
        )

        # Validate
        validate_required_paths(resolved)

        # Add name back
        resolved["name"] = farm_name

        return resolved

    def _generate_single_farm(self, farm_config: dict) -> xr.Dataset:
        """
        Generate database for a single farm.

        Parameters
        ----------
        farm_config : dict
            Farm configuration with resolved paths

        Returns
        -------
        xr.Dataset
            Database with wind_farm coordinate set to farm name
        """
        farm_name = farm_config["name"]
        print(f"\nProcessing farm: {farm_name}")

        # Ensure all paths are resolved
        paths = self._ensure_farm_paths(farm_config)

        print(f"  system_config: {paths['system_config'].name}")
        print(f"  reference_power: {paths['reference_power'].name}")
        print(f"  reference_resource: {paths['reference_resource'].name}")
        print(f"  wind_farm_layout: {paths['wind_farm_layout'].name}")

        # Create farm-specific output directory
        farm_output_dir = self.output_dir / farm_name
        farm_output_dir.mkdir(parents=True, exist_ok=True)

        # Preprocessing
        if self.run_preprocessing and self.preprocessing_steps:
            print("  Running preprocessing...")
            processed_resource_path = farm_output_dir / "processed_physical_inputs.nc"

            preprocessor = PreprocessingInputs(
                ref_resource_path=paths["reference_resource"],
                output_path=processed_resource_path,
                steps=self.preprocessing_steps,
            )
            preprocessor.run_pipeline()
        else:
            processed_resource_path = paths["reference_resource"]

        # Database generation
        print("  Generating database...")
        database_path = farm_output_dir / "database.nc"

        generator = DatabaseGenerator(
            nsamples=self.n_samples,
            param_config=self.param_config,
            system_yaml_path=paths["system_config"],
            ref_power_path=paths["reference_power"],
            processed_resource_path=processed_resource_path,
            wf_layout_path=paths["wind_farm_layout"],
            output_db_path=database_path,
            model=self.model,
        )

        db = generator.generate_database()

        # Ensure wind_farm coordinate is set to our explicit name
        if "wind_farm" in db.dims:
            db = db.assign_coords(wind_farm=[farm_name])

        return db

    def _combine_databases(self, databases: list[xr.Dataset]) -> xr.Dataset:
        """
        Combine databases from multiple farms.

        Re-indexes case_index to ensure uniqueness across farms.
        Preserves wind_farm coordinate for LeaveOneGroupOut CV.

        Parameters
        ----------
        databases : list[xr.Dataset]
            List of per-farm databases

        Returns
        -------
        xr.Dataset
            Combined database with unique case indices
        """
        print(f"\nCombining {len(databases)} farm databases...")

        # Re-index case_index to be unique across farms
        reindexed = []
        offset = 0

        for db in databases:
            # Get the farm name from the database
            if "wind_farm" in db.coords:
                farm_name = (
                    str(db.wind_farm.values[0])
                    if db.wind_farm.size == 1
                    else str(db.wind_farm.values)
                )
            else:
                farm_name = "unknown"

            # Create new case indices
            n_cases = db.dims["case_index"]
            new_indices = np.arange(offset, offset + n_cases)

            # Update dataset
            db_reindexed = db.assign_coords(case_index=new_indices)
            reindexed.append(db_reindexed)

            print(
                f"  {farm_name}: {n_cases} cases, indices {new_indices[0]}-{new_indices[-1]}"
            )
            offset += n_cases

        # Concatenate along case_index
        combined = xr.concat(reindexed, dim="case_index")

        # Validate
        self._validate_combined_database(combined)

        print(f"\nCombined database: {combined.dims['case_index']} total cases")

        return combined

    def _validate_combined_database(self, db: xr.Dataset) -> None:
        """Validate the combined database."""
        # Check case_index uniqueness
        case_indices = db.case_index.values
        if len(case_indices) != len(np.unique(case_indices)):
            raise ValueError("case_index values are not unique")

        # Check wind_farm coordinate exists
        if "wind_farm" not in db.coords:
            raise ValueError("wind_farm coordinate missing")

        # Log summary by farm
        farms = np.unique(db.wind_farm.values)
        print("Farm summary:")
        for farm in farms:
            n = int((db.wind_farm == farm).sum())
            print(f"  {farm}: {n} cases")

    def generate_database(self) -> xr.Dataset:
        """
        Generate combined database from all farms.

        Returns
        -------
        xr.Dataset
            Combined database with:
            - Unique case_index across all farms
            - wind_farm coordinate for grouping
            - All variables from individual databases
        """
        print(
            f"Starting multi-farm database generation for {len(self.farm_configs)} farms"
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

        databases = []

        for farm_config in self.farm_configs:
            try:
                db = self._generate_single_farm(farm_config)
                databases.append(db)
            except Exception as e:
                print(f"ERROR: Failed to process farm '{farm_config['name']}': {e}")
                raise

        # Combine
        combined = self._combine_databases(databases)

        # Save
        output_path = self.output_dir / self.database_file
        combined.to_netcdf(output_path)
        print(f"\nSaved combined database to: {output_path}")

        return combined


def generate_multi_farm_database(
    farm_configs: list[dict],
    param_config: dict,
    n_samples: int,
    output_dir: Path,
    database_file: str = "combined_database.nc",
    model: str = "pywake",
    preprocessing_steps: list[str] = None,
    run_preprocessing: bool = True,
) -> xr.Dataset:
    """
    Convenience function to generate multi-farm database.

    Parameters
    ----------
    farm_configs : list[dict]
        List of farm configurations with 'name' and 'system_config' keys
        (paths can be pre-resolved or will be auto-inferred)
    param_config : dict
        Parameter sampling configuration
    n_samples : int
        Number of parameter samples
    output_dir : Path
        Output directory
    database_file : str
        Output filename
    model : str
        Flow model (default: 'pywake')
    preprocessing_steps : list[str]
        Preprocessing steps to apply
    run_preprocessing : bool
        Whether to run preprocessing

    Returns
    -------
    xr.Dataset
        Combined database
    """
    generator = MultiFarmDatabaseGenerator(
        farm_configs=farm_configs,
        param_config=param_config,
        n_samples=n_samples,
        output_dir=output_dir,
        database_file=database_file,
        model=model,
        preprocessing_steps=preprocessing_steps,
        run_preprocessing=run_preprocessing,
    )

    return generator.generate_database()
