# tests/unit/test_path_inference.py
"""
Unit tests for wifa_uq.model_error_database.path_inference

Covers:
- extract_include_paths_windio: parsing + recursion + "don't overwrite top-level keys"
- find_resource_file_from_windio: direct wind_resource, energy_resource YAML chain, site->energy_resource chain, None
- infer_paths_from_system_config: explicit overrides, windIO include inference, pattern fallbacks, content-based resource detection
- validate_required_paths: missing keys, non-existent paths, helpful directory listing
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from wifa_uq.model_error_database.path_inference import (
    extract_include_paths_windio,
    find_resource_file_from_windio,
    infer_paths_from_system_config,
    validate_required_paths,
)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _write_nc_power(path: Path, n_turbines: int = 2, n_times: int = 3) -> Path:
    ds = xr.Dataset(
        data_vars={"power": (("turbine", "time"), np.ones((n_turbines, n_times)))},
        coords={"turbine": np.arange(n_turbines), "time": np.arange(n_times)},
    )
    ds.to_netcdf(path)
    return path


def _write_nc_resource_like(path: Path, n_times: int = 3, n_heights: int = 2) -> Path:
    ds = xr.Dataset(
        data_vars={
            "wind_speed": (("time", "height"), np.ones((n_times, n_heights)) * 8.0),
            "wind_direction": (("time",), np.ones(n_times) * 270.0),
        },
        coords={
            "time": np.arange(n_times),
            "height": np.linspace(10.0, 100.0, n_heights),
        },
    )
    ds.to_netcdf(path)
    return path


# -----------------------------------------------------------------------------
# extract_include_paths_windio
# -----------------------------------------------------------------------------


class TestExtractIncludePathsWindIO:
    def test_parses_top_level_includes(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        site_yaml = tmp_path / "energy_site.yaml"
        wf_yaml = tmp_path / "wind_farm.yaml"

        _write_text(site_yaml, "name: Site\n")
        _write_text(wf_yaml, "name: Farm\n")
        _write_text(
            sys_yaml,
            "\n".join(
                [
                    "name: Sys",
                    "site: !include energy_site.yaml",
                    "wind_farm: !include wind_farm.yaml",
                ]
            )
            + "\n",
        )

        includes = extract_include_paths_windio(sys_yaml)
        assert includes["site"] == site_yaml
        assert includes["wind_farm"] == wf_yaml

    def test_recurses_into_nested_yaml_and_keeps_top_level_keys(self, tmp_path: Path):
        # System includes site; site includes wind_farm (nested) and energy_resource.
        # Top-level system also includes wind_farm; nested wind_farm should NOT overwrite it.
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        site_yaml = tmp_path / "energy_site.yaml"
        wf_top = tmp_path / "wind_farm_top.yaml"
        wf_nested = tmp_path / "wind_farm_nested.yaml"
        er_yaml = tmp_path / "energy_resource.yaml"
        resource_nc = tmp_path / "resource.nc"

        _write_text(wf_top, "name: FarmTop\n")
        _write_text(wf_nested, "name: FarmNested\n")
        _write_text(er_yaml, "wind_resource: !include resource.nc\n")
        _write_nc_resource_like(resource_nc)

        _write_text(
            site_yaml,
            "\n".join(
                [
                    "name: Site",
                    "wind_farm: !include wind_farm_nested.yaml",
                    "energy_resource: !include energy_resource.yaml",
                ]
            )
            + "\n",
        )
        _write_text(
            sys_yaml,
            "\n".join(
                [
                    "name: Sys",
                    "site: !include energy_site.yaml",
                    "wind_farm: !include wind_farm_top.yaml",
                ]
            )
            + "\n",
        )

        includes = extract_include_paths_windio(sys_yaml)

        # Top-level keys present
        assert includes["site"] == site_yaml
        assert includes["wind_farm"] == wf_top

        # Nested keys present (but don't overwrite wind_farm)
        assert includes["energy_resource"] == er_yaml
        assert includes["wind_resource"] == resource_nc

    def test_ignores_nonexistent_include_targets(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(
            sys_yaml,
            "\n".join(
                [
                    "name: Sys",
                    "site: !include does_not_exist.yaml",
                    "wind_farm: !include missing.yaml",
                ]
            )
            + "\n",
        )
        includes = extract_include_paths_windio(sys_yaml)
        assert includes == {}


# -----------------------------------------------------------------------------
# find_resource_file_from_windio
# -----------------------------------------------------------------------------


class TestFindResourceFileFromWindIO:
    def test_direct_wind_resource_include(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        resource_nc = tmp_path / "resource.nc"
        _write_nc_resource_like(resource_nc)

        _write_text(sys_yaml, f"wind_resource: !include {resource_nc.name}\n")
        found = find_resource_file_from_windio(sys_yaml)
        assert found == resource_nc

    def test_energy_resource_yaml_chain(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        er_yaml = tmp_path / "energy_resource.yaml"
        resource_nc = tmp_path / "resource.nc"
        _write_nc_resource_like(resource_nc)

        _write_text(er_yaml, "wind_resource: !include resource.nc\n")
        _write_text(sys_yaml, "energy_resource: !include energy_resource.yaml\n")

        found = find_resource_file_from_windio(sys_yaml)
        assert found == resource_nc

    def test_site_yaml_chain(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        site_yaml = tmp_path / "energy_site.yaml"
        er_yaml = tmp_path / "energy_resource.yaml"
        resource_nc = tmp_path / "resource.nc"
        _write_nc_resource_like(resource_nc)

        _write_text(er_yaml, "wind_resource: !include resource.nc\n")
        _write_text(site_yaml, "energy_resource: !include energy_resource.yaml\n")
        _write_text(sys_yaml, "site: !include energy_site.yaml\n")

        found = find_resource_file_from_windio(sys_yaml)
        assert found == resource_nc

    def test_returns_none_when_not_found(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(sys_yaml, "name: Sys\n")
        found = find_resource_file_from_windio(sys_yaml)
        assert found is None


# -----------------------------------------------------------------------------
# infer_paths_from_system_config
# -----------------------------------------------------------------------------


class TestInferPathsFromSystemConfig:
    def test_explicit_paths_override_inferred(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        wf_yaml = tmp_path / "wind_farm.yaml"
        power_nc = tmp_path / "turbine_data.nc"
        resource_nc = tmp_path / "resource.nc"

        _write_text(wf_yaml, "name: Farm\n")
        _write_nc_power(power_nc)
        _write_nc_resource_like(resource_nc)

        # windIO includes point to one set
        _write_text(
            sys_yaml,
            "\n".join(
                [
                    "wind_farm: !include wind_farm.yaml",
                    "turbine_data: !include turbine_data.nc",
                    "wind_resource: !include resource.nc",
                ]
            )
            + "\n",
        )

        # Explicit overrides to different files
        explicit_power = tmp_path / "explicit_power.nc"
        explicit_resource = tmp_path / "explicit_resource.nc"
        explicit_layout = tmp_path / "explicit_layout.yaml"
        _write_nc_power(explicit_power)
        _write_nc_resource_like(explicit_resource)
        _write_text(explicit_layout, "name: ExplicitFarm\n")

        out = infer_paths_from_system_config(
            system_config_path=sys_yaml,
            explicit_paths={
                "reference_power": explicit_power,
                "reference_resource": explicit_resource,
                "wind_farm_layout": explicit_layout,
            },
        )

        assert out["system_config"] == sys_yaml
        assert out["reference_power"] == explicit_power
        assert out["reference_resource"] == explicit_resource
        assert out["wind_farm_layout"] == explicit_layout

    def test_infers_from_windio_includes(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        wf_yaml = tmp_path / "wind_farm.yaml"
        power_nc = tmp_path / "turbine_data.nc"
        resource_nc = tmp_path / "resource.nc"

        _write_text(wf_yaml, "name: Farm\n")
        _write_nc_power(power_nc)
        _write_nc_resource_like(resource_nc)

        _write_text(
            sys_yaml,
            "\n".join(
                [
                    "wind_farm: !include wind_farm.yaml",
                    "turbine_data: !include turbine_data.nc",
                    "wind_resource: !include resource.nc",
                ]
            )
            + "\n",
        )

        out = infer_paths_from_system_config(system_config_path=sys_yaml)

        assert out["system_config"] == sys_yaml
        assert out["reference_power"] == power_nc
        assert out["reference_resource"] == resource_nc
        assert out["wind_farm_layout"] == wf_yaml

    def test_pattern_fallbacks_for_power_and_layout(self, tmp_path: Path):
        # No windIO includes; should fall back to filename patterns.
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(sys_yaml, "name: Sys\n")

        power_nc = tmp_path / "turbine_data.nc"
        layout_yaml = tmp_path / "wind_farm.yaml"
        _write_nc_power(power_nc)
        _write_text(layout_yaml, "name: Farm\n")

        # For resource, use a "common-name" file in this test
        resource_nc = tmp_path / "resource.nc"
        _write_nc_resource_like(resource_nc)

        out = infer_paths_from_system_config(sys_yaml)

        assert out["reference_power"] == power_nc
        assert out["wind_farm_layout"] == layout_yaml
        assert out["reference_resource"] == resource_nc

    def test_content_based_resource_detection_skips_power_file(self, tmp_path: Path):
        # No windIO includes; no common-name resource; should detect via xarray content.
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(sys_yaml, "name: Sys\n")

        power_nc = tmp_path / "turbine_data.nc"
        _write_nc_power(power_nc)

        # Another .nc with resource-like variables
        detected_resource = tmp_path / "some_weird_name.nc"
        _write_nc_resource_like(detected_resource)

        # Need layout for completeness of inference (pattern)
        layout_yaml = tmp_path / "layout.yaml"
        _write_text(layout_yaml, "name: Farm\n")

        out = infer_paths_from_system_config(sys_yaml)

        assert out["reference_power"] == power_nc
        assert out["wind_farm_layout"] == layout_yaml
        assert out["reference_resource"] == detected_resource

    def test_partial_inference_when_files_missing(self, tmp_path: Path):
        # infer_paths_from_system_config does not raise if missing; validation does.
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(sys_yaml, "name: Sys\n")

        out = infer_paths_from_system_config(sys_yaml)
        assert out["system_config"] == sys_yaml
        # Others may be absent
        assert "reference_power" not in out
        assert "reference_resource" not in out
        assert "wind_farm_layout" not in out


# -----------------------------------------------------------------------------
# validate_required_paths
# -----------------------------------------------------------------------------


class TestValidateRequiredPaths:
    def test_passes_when_all_exist(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        power_nc = tmp_path / "turbine_data.nc"
        resource_nc = tmp_path / "resource.nc"
        layout_yaml = tmp_path / "wind_farm.yaml"

        _write_text(sys_yaml, "name: Sys\n")
        _write_nc_power(power_nc)
        _write_nc_resource_like(resource_nc)
        _write_text(layout_yaml, "name: Farm\n")

        paths = {
            "system_config": sys_yaml,
            "reference_power": power_nc,
            "reference_resource": resource_nc,
            "wind_farm_layout": layout_yaml,
        }
        validate_required_paths(paths)  # should not raise

    def test_raises_with_missing_keys(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(sys_yaml, "name: Sys\n")

        paths = {"system_config": sys_yaml}  # missing others
        with pytest.raises(FileNotFoundError) as exc:
            validate_required_paths(paths)

        msg = str(exc.value)
        assert "Missing" in msg
        assert "reference_power" in msg
        assert "reference_resource" in msg
        assert "wind_farm_layout" in msg
        assert "Files in directory" in msg

    def test_raises_with_not_found_paths(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(sys_yaml, "name: Sys\n")

        paths = {
            "system_config": sys_yaml,
            "reference_power": tmp_path / "nope_power.nc",
            "reference_resource": tmp_path / "nope_resource.nc",
            "wind_farm_layout": tmp_path / "nope_layout.yaml",
        }

        with pytest.raises(FileNotFoundError) as exc:
            validate_required_paths(paths)

        msg = str(exc.value)
        assert "Not found" in msg
        assert "reference_power" in msg
        assert "reference_resource" in msg
        assert "wind_farm_layout" in msg
        assert "You can specify these paths explicitly" in msg

    def test_custom_required_list(self, tmp_path: Path):
        sys_yaml = tmp_path / "wind_energy_system.yaml"
        _write_text(sys_yaml, "name: Sys\n")

        # Only require system_config
        validate_required_paths({"system_config": sys_yaml}, required=["system_config"])
