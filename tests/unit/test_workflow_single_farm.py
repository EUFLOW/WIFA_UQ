# tests/unit/test_workflow_single_farm.py
"""
Unit tests for wifa_uq.workflow single-farm workflow path.

These tests are designed to be fast and CI-friendly by mocking:
- windIO path inference (infer_paths_from_system_config / validate_required_paths)
- preprocessing (PreprocessingInputs)
- database generation (DatabaseGenerator)
- downstream error prediction (_run_error_prediction)

Key branches covered:
- preprocessing.run True/False
- database_gen.run True/False
- loading existing database when dbgen disabled
"""

from __future__ import annotations

from pathlib import Path

import pytest
import xarray as xr

from wifa_uq import workflow
import wifa_uq.model_error_database.path_inference as path_inference


def _single_farm_config(preprocessing_run: bool = True, dbgen_run: bool = True) -> dict:
    return {
        "paths": {
            "system_config": "wind_energy_system.yaml",
            "output_dir": "out",
            "processed_resource_file": "processed.nc",
            "database_file": "db.nc",
            # optional explicit overrides omitted
        },
        "preprocessing": {"run": preprocessing_run, "steps": ["recalculate_params"]},
        "database_gen": {
            "run": dbgen_run,
            "flow_model": "pywake",
            "n_samples": 2,
            "param_config": {},
        },
        "error_prediction": {
            "run": False,
            "features": ["ABL_height"],
            "model": "Linear",
            "model_params": {},
            "calibrator": "MinBiasCalibrator",
            "bias_predictor": "BiasPredictor",
            "cross_validation": {
                "run": False,
                "splitting_mode": "kfold_shuffled",
                "n_splits": 2,
            },
        },
        "sensitivity_analysis": {
            "run_observation_sensitivity": False,
            "run_bias_sensitivity": False,
        },
        "physics_insights": {"run": False},
    }


def test_run_single_farm_preprocess_and_dbgen(monkeypatch, tmp_path: Path):
    """
    preprocessing.run=True + database_gen.run=True:
    - Uses inferred paths
    - Runs preprocessing (mocked)
    - Runs DB generation (mocked)
    - Calls _run_error_prediction and returns its output
    """
    cfg = _single_farm_config(preprocessing_run=True, dbgen_run=True)
    base_dir = tmp_path
    (base_dir / cfg["paths"]["output_dir"]).mkdir(parents=True, exist_ok=True)

    # Create dummy system config
    sys_path = base_dir / cfg["paths"]["system_config"]
    sys_path.write_text("name: Sys\n", encoding="utf-8")

    # Dummy referenced files (do NOT need to be valid NetCDF due to preprocessing mock)
    ref_power = base_dir / "turbine_data.nc"
    ref_res = base_dir / "resource.nc"
    wf_layout = base_dir / "wind_farm.yaml"
    ref_power.write_text("x", encoding="utf-8")
    ref_res.write_text("x", encoding="utf-8")
    wf_layout.write_text("x", encoding="utf-8")

    # Patch windIO inference + validation (these are imported inside _run_single_farm_workflow)
    monkeypatch.setattr(
        path_inference,
        "infer_paths_from_system_config",
        lambda system_config_path, explicit_paths=None: {
            "system_config": system_config_path,
            "reference_power": ref_power,
            "reference_resource": ref_res,
            "wind_farm_layout": wf_layout,
        },
    )
    monkeypatch.setattr(path_inference, "validate_required_paths", lambda paths: None)

    # Patch preprocessing to avoid opening NetCDF
    class FakePreproc:
        def __init__(self, ref_resource_path, output_path, steps):
            self.ref_resource_path = ref_resource_path
            self.output_path = output_path
            self.steps = steps

        def run_pipeline(self):
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            self.output_path.write_text("processed", encoding="utf-8")
            return self.output_path

    monkeypatch.setattr(workflow, "PreprocessingInputs", FakePreproc)

    # Patch DB generator
    class FakeDBGen:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def generate_database(self):
            return xr.Dataset(attrs={"generated": True})

    monkeypatch.setattr(workflow, "DatabaseGenerator", FakeDBGen)

    # Patch downstream error prediction
    monkeypatch.setattr(
        workflow,
        "_run_error_prediction",
        lambda config, database, output_dir: ("cv", "preds", "tests"),
    )

    out = workflow._run_single_farm_workflow(cfg, base_dir)
    assert out == ("cv", "preds", "tests")


def test_run_single_farm_dbgen_off_loads_existing(monkeypatch, tmp_path: Path):
    """
    preprocessing.run=False + database_gen.run=False:
    - Uses inferred paths
    - Skips preprocessing and uses raw resource file (must exist)
    - Loads existing database via xr.load_dataset
    - Calls _run_error_prediction
    """
    cfg = _single_farm_config(preprocessing_run=False, dbgen_run=False)
    base_dir = tmp_path

    sys_path = base_dir / cfg["paths"]["system_config"]
    sys_path.write_text("name: Sys\n", encoding="utf-8")

    ref_power = base_dir / "turbine_data.nc"
    ref_res = base_dir / "resource.nc"
    wf_layout = base_dir / "wind_farm.yaml"
    ref_power.write_text("x", encoding="utf-8")
    ref_res.write_text("x", encoding="utf-8")
    wf_layout.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        path_inference,
        "infer_paths_from_system_config",
        lambda system_config_path, explicit_paths=None: {
            "system_config": system_config_path,
            "reference_power": ref_power,
            "reference_resource": ref_res,
            "wind_farm_layout": wf_layout,
        },
    )
    monkeypatch.setattr(path_inference, "validate_required_paths", lambda paths: None)

    # Create "existing" database file
    out_dir = base_dir / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    db_path = out_dir / cfg["paths"]["database_file"]
    db_path.write_bytes(b"")  # just ensure it exists; xr.load_dataset is mocked below

    # Patch xarray load
    monkeypatch.setattr(
        workflow.xr, "load_dataset", lambda p: xr.Dataset(attrs={"loaded": True})
    )

    monkeypatch.setattr(
        workflow,
        "_run_error_prediction",
        lambda config, database, output_dir: ("ok", None, None),
    )

    out = workflow._run_single_farm_workflow(cfg, base_dir)
    assert out[0] == "ok"


def test_run_single_farm_preprocess_disabled_missing_resource_raises(
    monkeypatch, tmp_path: Path
):
    """
    preprocessing.run=False should raise if inferred reference_resource doesn't exist.
    """
    cfg = _single_farm_config(preprocessing_run=False, dbgen_run=False)
    base_dir = tmp_path

    sys_path = base_dir / cfg["paths"]["system_config"]
    sys_path.write_text("name: Sys\n", encoding="utf-8")

    # Create only system + others; omit reference_resource so it doesn't exist
    ref_power = base_dir / "turbine_data.nc"
    wf_layout = base_dir / "wind_farm.yaml"
    missing_ref_res = base_dir / "resource.nc"  # do NOT create
    ref_power.write_text("x", encoding="utf-8")
    wf_layout.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        path_inference,
        "infer_paths_from_system_config",
        lambda system_config_path, explicit_paths=None: {
            "system_config": system_config_path,
            "reference_power": ref_power,
            "reference_resource": missing_ref_res,
            "wind_farm_layout": wf_layout,
        },
    )
    monkeypatch.setattr(path_inference, "validate_required_paths", lambda paths: None)

    with pytest.raises(FileNotFoundError, match="Input resource file not found"):
        workflow._run_single_farm_workflow(cfg, base_dir)


def test_run_single_farm_dbgen_disabled_missing_db_raises(monkeypatch, tmp_path: Path):
    """
    database_gen.run=False should raise if the database file doesn't exist.
    """
    cfg = _single_farm_config(preprocessing_run=False, dbgen_run=False)
    base_dir = tmp_path

    sys_path = base_dir / cfg["paths"]["system_config"]
    sys_path.write_text("name: Sys\n", encoding="utf-8")

    ref_power = base_dir / "turbine_data.nc"
    ref_res = base_dir / "resource.nc"
    wf_layout = base_dir / "wind_farm.yaml"
    ref_power.write_text("x", encoding="utf-8")
    ref_res.write_text("x", encoding="utf-8")
    wf_layout.write_text("x", encoding="utf-8")

    monkeypatch.setattr(
        path_inference,
        "infer_paths_from_system_config",
        lambda system_config_path, explicit_paths=None: {
            "system_config": system_config_path,
            "reference_power": ref_power,
            "reference_resource": ref_res,
            "wind_farm_layout": wf_layout,
        },
    )
    monkeypatch.setattr(path_inference, "validate_required_paths", lambda paths: None)

    # Ensure output dir exists but DB does not
    out_dir = base_dir / cfg["paths"]["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with pytest.raises(FileNotFoundError, match="Database file not found"):
        workflow._run_single_farm_workflow(cfg, base_dir)
