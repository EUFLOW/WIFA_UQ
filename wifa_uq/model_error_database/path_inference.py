"""
Shared utilities for inferring paths from windIO system configurations.

These functions parse windIO YAML files to auto-detect paths to:
- reference_power (turbine_data.nc)
- reference_resource (wind resource NetCDF)
- wind_farm_layout (wind_farm.yaml)
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import xarray as xr

logger = logging.getLogger(__name__)


def extract_include_paths_windio(yaml_path: Path) -> dict[str, Path]:
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
                    nested = extract_include_paths_windio(file_path)
                    # Add nested includes, but don't overwrite top-level keys
                    for nested_key, nested_path in nested.items():
                        if nested_key not in includes:
                            includes[nested_key] = nested_path
                except Exception as e:
                    logger.debug(f"Could not parse nested YAML {file_path}: {e}")

    return includes


def find_resource_file_from_windio(system_yaml_path: Path) -> Path | None:
    """
    Follow the windIO include chain to find the actual resource NC file.

    Path is typically:
    wind_energy_system.yaml
      -> site: !include energy_site.yaml
        -> energy_resource: !include energy_resource.yaml
          -> wind_resource: !include <resource_file>.nc

    Returns the path to the NC file or None if not found.
    """
    includes = extract_include_paths_windio(system_yaml_path)

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
            er_includes = extract_include_paths_windio(er_path)
            if "wind_resource" in er_includes:
                return er_includes["wind_resource"]

    # Check site YAML
    if "site" in includes:
        site_path = includes["site"]
        if site_path.suffix in [".yaml", ".yml"]:
            site_includes = extract_include_paths_windio(site_path)

            # Check for energy_resource in site
            if "energy_resource" in site_includes:
                er_path = site_includes["energy_resource"]
                if er_path.suffix in [".nc", ".netcdf"]:
                    return er_path
                elif er_path.suffix in [".yaml", ".yml"]:
                    er_includes = extract_include_paths_windio(er_path)
                    if "wind_resource" in er_includes:
                        return er_includes["wind_resource"]

    return None


def infer_paths_from_system_config(
    system_config_path: Path,
    explicit_paths: dict[str, Path | str] | None = None,
) -> dict[str, Path]:
    """
    Infer all required paths from a windIO system config file.

    Explicit paths override inferred ones (for backward compatibility).

    Args:
        system_config_path: Path to the wind_energy_system.yaml file
        explicit_paths: Optional dict with explicit path overrides:
            - reference_power
            - reference_resource
            - wind_farm_layout

    Returns:
        Dict with resolved paths:
            - system_config
            - reference_power
            - reference_resource
            - wind_farm_layout

    Raises:
        FileNotFoundError: If required paths cannot be found
    """
    system_config_path = Path(system_config_path)
    farm_dir = system_config_path.parent
    explicit_paths = explicit_paths or {}

    paths = {
        "system_config": system_config_path,
    }

    # Use explicit paths if provided (convert to Path)
    for key in ["reference_power", "reference_resource", "wind_farm_layout"]:
        if key in explicit_paths and explicit_paths[key] is not None:
            paths[key] = Path(explicit_paths[key])

    # For missing paths, try to infer from windIO structure
    missing_keys = {"reference_power", "reference_resource", "wind_farm_layout"} - set(
        paths.keys()
    )

    if missing_keys:
        logger.info(f"Inferring paths for: {missing_keys}")
        try:
            includes = extract_include_paths_windio(system_config_path)
            logger.debug(f"Found windIO includes: {list(includes.keys())}")

            # Reference power: simulation_output.turbine_data
            if "reference_power" not in paths:
                if "turbine_data" in includes:
                    paths["reference_power"] = includes["turbine_data"]
                    logger.info(
                        f"Found reference_power: {paths['reference_power'].name}"
                    )

            # Reference resource: Follow the windIO chain to find the NC file
            if "reference_resource" not in paths:
                resource_path = find_resource_file_from_windio(system_config_path)
                if resource_path and resource_path.exists():
                    paths["reference_resource"] = resource_path
                    logger.info(
                        f"Found reference_resource: {paths['reference_resource'].name}"
                    )

            # Wind farm layout
            if "wind_farm_layout" not in paths:
                if "wind_farm" in includes:
                    paths["wind_farm_layout"] = includes["wind_farm"]
                    logger.info(
                        f"Found wind_farm_layout: {paths['wind_farm_layout'].name}"
                    )

        except Exception as e:
            logger.warning(f"Could not parse windIO structure: {e}")

    # Final fallback: pattern matching on common filenames
    if "reference_power" not in paths:
        for name in [
            "turbine_data.nc",
            "power.nc",
            "ref_power.nc",
            "observedPower*.nc",
        ]:
            candidates = list(farm_dir.glob(name))
            if candidates:
                paths["reference_power"] = candidates[0]
                logger.info(
                    f"Found reference_power by pattern: {paths['reference_power'].name}"
                )
                break

    if "reference_resource" not in paths:
        # Try common names first
        for name in ["resource.nc", "energy_resource.nc", "originalData.nc"]:
            candidate = farm_dir / name
            if candidate.exists():
                paths["reference_resource"] = candidate
                logger.info(f"Found reference_resource by pattern: {candidate.name}")
                break

        # If still not found, look for any NC file with resource-like variables
        if "reference_resource" not in paths:
            for nc_file in farm_dir.glob("*.nc"):
                if nc_file.name in ["turbine_data.nc"]:
                    continue  # Skip power files
                if "reference_power" in paths and nc_file == paths["reference_power"]:
                    continue
                try:
                    with xr.open_dataset(nc_file) as ds:
                        resource_vars = [
                            "wind_speed",
                            "WS",
                            "ws",
                            "u",
                            "U",
                            "wind_direction",
                            "WD",
                            "wd",
                            "potential_temperature",
                            "temperature",
                        ]
                        if any(
                            v in ds.data_vars or v in ds.coords for v in resource_vars
                        ):
                            paths["reference_resource"] = nc_file
                            logger.info(
                                f"Found reference_resource by content: {nc_file.name}"
                            )
                            break
                except Exception:
                    continue

    if "wind_farm_layout" not in paths:
        for name in [
            "wind_farm.yaml",
            "layout.yaml",
            "plant_wind_farm.yaml",
            "*wind_farm*.yaml",
        ]:
            candidates = list(farm_dir.glob(name))
            if candidates:
                paths["wind_farm_layout"] = candidates[0]
                logger.info(
                    f"Found wind_farm_layout by pattern: {paths['wind_farm_layout'].name}"
                )
                break

    return paths


def validate_required_paths(
    paths: dict[str, Path],
    required: list[str] | None = None,
) -> None:
    """
    Validate that all required paths exist.

    Args:
        paths: Dict of path names to Path objects
        required: List of required keys (default: all standard paths)

    Raises:
        FileNotFoundError: If any required path is missing or doesn't exist
    """
    if required is None:
        required = [
            "system_config",
            "reference_power",
            "reference_resource",
            "wind_farm_layout",
        ]

    missing = []
    not_found = []

    for key in required:
        if key not in paths:
            missing.append(key)
        elif not paths[key].exists():
            not_found.append(f"{key}: {paths[key]}")

    if missing or not_found:
        # Get the directory for helpful error message
        if "system_config" in paths:
            farm_dir = paths["system_config"].parent
            existing_files = sorted(farm_dir.glob("*"))
            file_list = "\n  ".join(str(f.name) for f in existing_files[:20])
            if len(existing_files) > 20:
                file_list += f"\n  ... and {len(existing_files) - 20} more files"
        else:
            file_list = "(unknown directory)"

        error_msg = "Could not find required paths:\n"
        if missing:
            error_msg += f"  Missing: {missing}\n"
        if not_found:
            error_msg += f"  Not found: {not_found}\n"
        error_msg += f"\nFiles in directory:\n  {file_list}"
        error_msg += "\n\nYou can specify these paths explicitly in your config file."

        raise FileNotFoundError(error_msg)
