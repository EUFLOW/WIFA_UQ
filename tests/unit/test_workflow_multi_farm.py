from pathlib import Path
import xarray as xr

from wifa_uq import workflow


def _multi_farm_config(tmp_path: Path, dbgen_run=True):
    return {
        "paths": {
            "output_dir": "out_multi",
            "database_file": "combined.nc",
        },
        "farms": [
            {"name": "A", "system_config": "A/system.yaml"},
            {"name": "B", "system_config": "B/system.yaml"},
        ],
        "preprocessing": {"run": True, "steps": ["recalculate_params"]},
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


def test_run_multi_farm_dbgen_calls_generator(monkeypatch, tmp_path: Path):
    cfg = _multi_farm_config(tmp_path, dbgen_run=True)
    base_dir = tmp_path

    called = {}

    def fake_gen(**kwargs):
        called.update(kwargs)
        return xr.Dataset(
            coords={"wind_farm": ("case_index", ["A", "B"])}, data_vars={}
        )

    monkeypatch.setattr(
        workflow, "generate_multi_farm_database", lambda **kwargs: fake_gen(**kwargs)
    )
    monkeypatch.setattr(
        workflow,
        "_run_error_prediction",
        lambda config, database, output_dir: ("ok", None, None),
    )

    out = workflow._run_multi_farm_workflow(cfg, base_dir)
    assert out[0] == "ok"
    assert called["n_samples"] == 2
    assert called["model"] == "pywake"
    assert called["preprocessing_steps"] == ["recalculate_params"]


def test_run_multi_farm_dbgen_off_loads_existing(monkeypatch, tmp_path: Path):
    cfg = _multi_farm_config(tmp_path, dbgen_run=False)
    base_dir = tmp_path
    out_dir = base_dir / cfg["paths"]["output_dir"]
    out_dir.mkdir()
    db_path = out_dir / cfg["paths"]["database_file"]
    xr.Dataset(coords={"wind_farm": ("case_index", ["A"])}).to_netcdf(db_path)

    monkeypatch.setattr(
        workflow.xr,
        "load_dataset",
        lambda p: xr.Dataset(coords={"wind_farm": ("case_index", ["A"])}),
    )
    monkeypatch.setattr(
        workflow,
        "_run_error_prediction",
        lambda config, database, output_dir: ("ok", None, None),
    )

    out = workflow._run_multi_farm_workflow(cfg, base_dir)
    assert out[0] == "ok"
