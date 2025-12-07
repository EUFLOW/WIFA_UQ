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
from ruamel.yaml import YAML

from wifa_uq.model_error_database.database_gen import DatabaseGenerator
from wifa_uq.preprocessing.preprocessing import PreprocessingInputs

logger = logging.getLogger(__name__)


def _get_yaml_parser() -> YAML:
    """Get a basic YAML parser without !include resolution."""
    yaml_obj = YAML(typ="safe", pure=True)
    yaml_obj.default_flow_style = False
    return yaml_obj


def _extract_include_paths_windio(yaml_path: Path) -> dict[str, Path]:
    """
    Parse a windIO YAML file and extract paths from !include directives.

    Uses windIO-style parsing to recursively find all included files
    and map them to their semantic keys.

    Returns a dict with keys like:
      - 'site': path to site yaml
      - 'wind_farm': path to wind farm yaml
      - 'energy_resource': path to energy resource yaml
      - 'turbine_data': path to turbine data nc
      - 'wind_resource': path to wind resource nc (the actual resource file)
    """
    base_dir = yaml_path.parent
    includes = {}

    # Read raw YAML content to find !include directives
    with open(yaml_path, "r") as f:
        content = f.read()

    # Parse line by line to find !include patterns
    # This handles various formats: !include file.yaml, !include "file.yaml", etc.
    import re

    # Pattern for key: !include filename
    # Handles nested indentation
    include_pattern = re.compile(
        r'^\s*(\w+):\s*!include\s+["\']?([^"\'\s\n#]+)["\']?', re.MULTILINE
    )

    for match in include_pattern.finditer(content):
        key = match.group(1)
        filename = match.group(2)
        file_path = base_dir / filename

        if file_path.exists():
            includes[key] = file_path

            # If this is another YAML, recursively extract its includes
            if filename.endswith(".yaml") or filename.endswith(".yml"):
                try:
                    nested = _extract_include_paths_windio(file_path)
                    # Add nested includes, but don't overwrite top-level keys
                    for nested_key, nested_path in nested.items():
                        if nested_key not in includes:
                            includes[nested_key] = nested_path
                except Exception as e:
                    logger.debug(f"Could not parse nested YAML {file_path}: {e}")

    return includes


def _find_resource_file_from_windio(system_yaml_path: Path) -> Path | None:
    """
    Follow the windIO include chain to find the actual resource NC file.

    Path is typically:
    wind_energy_system.yaml
      -> site: !include energy_site.yaml
        -> energy_resource: !include energy_resource.yaml
          -> wind_resource: !include <resource_file>.nc

    Returns the path to the NC file or None if not found.
    """
    includes = _extract_include_paths_windio(system_yaml_path)

    # Direct wind_resource reference
    if "wind_resource" in includes:
        path = includes["wind_resource"]
        if path.suffix in [".nc", ".netcdf"]:
            return path

    # Check energy_resource (might be YAML or NC)
    if "energy_resource" in includes:
        er_path = includes["energy_resource"]
        if er_path.suffix in [".nc", ".netcdf"]:
            return er_path
        elif er_path.suffix in [".yaml", ".yml"]:
            # Parse the energy_resource YAML
            er_includes = _extract_include_paths_windio(er_path)
            if "wind_resource" in er_includes:
                return er_includes["wind_resource"]

    # Check site YAML
    if "site" in includes:
        site_path = includes["site"]
        if site_path.suffix in [".yaml", ".yml"]:
            site_includes = _extract_include_paths_windio(site_path)

            # Check for energy_resource in site
            if "energy_resource" in site_includes:
                er_path = site_includes["energy_resource"]
                if er_path.suffix in [".nc", ".netcdf"]:
                    return er_path
                elif er_path.suffix in [".yaml", ".yml"]:
                    er_includes = _extract_include_paths_windio(er_path)
                    if "wind_resource" in er_includes:
                        return er_includes["wind_resource"]

    return None


class MultiFarmDatabaseGenerator:
    """
    Generates a combined database from multiple wind farms.

    Each farm config must specify:
      - name: Unique identifier for the farm
      - system_config: Path to wind energy system YAML (windIO format)

    Optional (if data can't be auto-extracted from system_config):
      - reference_power: Path to reference power NetCDF
      - reference_resource: Path to reference resource NetCDF
      - wind_farm_layout: Path to wind farm layout YAML

    Parameters
    ----------
    farm_configs : list[dict]
        List of farm configurations with 'name' and 'system_config' keys

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

    def _infer_farm_paths(self, farm_config: dict) -> dict:
        """
        Infer paths for a farm from system_config using windIO include parsing.

        Parses the windIO YAML structure to find referenced files.
        Falls back to pattern matching on common filenames.
        """
        system_path = Path(farm_config["system_config"])
        farm_dir = system_path.parent
        farm_name = farm_config["name"]

        # Use explicit paths if provided
        paths = {
            "name": farm_name,
            "system_config": system_path,
        }

        if "reference_power" in farm_config:
            paths["reference_power"] = Path(farm_config["reference_power"])
        if "reference_resource" in farm_config:
            paths["reference_resource"] = Path(farm_config["reference_resource"])
        if "wind_farm_layout" in farm_config:
            paths["wind_farm_layout"] = Path(farm_config["wind_farm_layout"])

        # If we still need paths, parse windIO includes
        missing_keys = {
            "reference_power",
            "reference_resource",
            "wind_farm_layout",
        } - set(paths.keys())

        if missing_keys:
            print(f"  Parsing windIO structure for {farm_name}...")
            try:
                includes = _extract_include_paths_windio(system_path)
                print(f"    Found includes: {list(includes.keys())}")

                # Reference power: simulation_output.turbine_data
                if "reference_power" not in paths:
                    if "turbine_data" in includes:
                        paths["reference_power"] = includes["turbine_data"]
                        print(
                            f"    Found reference_power: {paths['reference_power'].name}"
                        )

                # Reference resource: Follow the windIO chain to find the NC file
                if "reference_resource" not in paths:
                    resource_path = _find_resource_file_from_windio(system_path)
                    if resource_path and resource_path.exists():
                        paths["reference_resource"] = resource_path
                        print(
                            f"    Found reference_resource: {paths['reference_resource'].name}"
                        )

                # Wind farm layout
                if "wind_farm_layout" not in paths:
                    if "wind_farm" in includes:
                        paths["wind_farm_layout"] = includes["wind_farm"]
                        print(
                            f"    Found wind_farm_layout: {paths['wind_farm_layout'].name}"
                        )

            except Exception as e:
                print(f"    Warning: Could not parse windIO structure: {e}")

        # Final fallback: pattern matching on common filenames
        if "reference_power" not in paths:
            for name in ["turbine_data.nc", "power.nc", "ref_power.nc"]:
                candidate = farm_dir / name
                if candidate.exists():
                    paths["reference_power"] = candidate
                    print(f"    Found reference_power by pattern: {candidate.name}")
                    break

        if "reference_resource" not in paths:
            # Try common names first
            for name in ["resource.nc", "energy_resource.nc", "originalData.nc"]:
                candidate = farm_dir / name
                if candidate.exists():
                    paths["reference_resource"] = candidate
                    print(f"    Found reference_resource by pattern: {candidate.name}")
                    break

            # If still not found, look for any NC file with resource-like variables
            if "reference_resource" not in paths:
                for nc_file in farm_dir.glob("*.nc"):
                    if nc_file.name in ["turbine_data.nc"]:
                        continue  # Skip power files
                    try:
                        with xr.open_dataset(nc_file) as ds:
                            # Check for typical resource variables
                            resource_vars = [
                                "wind_speed",
                                "WS",
                                "ws",
                                "u",
                                "U",
                                "wind_direction",
                                "WD",
                                "wd",
                            ]
                            if any(
                                v in ds.data_vars or v in ds.coords
                                for v in resource_vars
                            ):
                                paths["reference_resource"] = nc_file
                                print(
                                    f"    Found reference_resource by content: {nc_file.name}"
                                )
                                break
                    except Exception:
                        continue

        if "wind_farm_layout" not in paths:
            for name in ["wind_farm.yaml", "layout.yaml", "plant_wind_farm.yaml"]:
                candidate = farm_dir / name
                if candidate.exists():
                    paths["wind_farm_layout"] = candidate
                    print(f"    Found wind_farm_layout by pattern: {candidate.name}")
                    break

        return paths

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
        print(f"Processing farm: {farm_name}")

        # Infer any missing paths
        paths = self._infer_farm_paths(farm_config)

        # Validate required paths exist
        required_paths = [
            "system_config",
            "reference_power",
            "reference_resource",
            "wind_farm_layout",
        ]
        missing = []
        for key in required_paths:
            if key not in paths:
                missing.append(key)
            elif not paths[key].exists():
                raise FileNotFoundError(
                    f"{key} not found for farm '{farm_name}': {paths[key]}"
                )

        if missing:
            # List what files exist in the directory for debugging
            farm_dir = Path(farm_config["system_config"]).parent
            existing_files = sorted(farm_dir.glob("*"))
            raise FileNotFoundError(
                f"Could not find {missing} for farm '{farm_name}'.\n"
                f"Please specify explicitly in the config.\n"
                f"Files in {farm_dir}:\n  "
                + "\n  ".join(str(f.name) for f in existing_files)
            )

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
        # (overwrite whatever was inferred from the data files)
        if "wind_farm" in db.dims:
            # Replace the wind_farm values with our explicit name
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
        print(f"Combining {len(databases)} farm databases...")

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

        print(f"Combined database: {combined.dims['case_index']} total cases")

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
        print(f"Saved combined database to: {output_path}")

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
