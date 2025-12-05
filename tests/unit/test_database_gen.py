# tests/unit/test_database_gen.py
from pathlib import Path
from unittest.mock import patch
import numpy as np
import xarray as xr

from wifa_uq.model_error_database.database_gen import DatabaseGenerator


def test_normalize_param_config_accepts_list_and_dict():
    gen = DatabaseGenerator.__new__(DatabaseGenerator)  # bypass __init__
    param_config = {
        "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": [0.01, 0.07],
        "attributes.analysis.blockage_model.ss_alpha": {
            "range": [0.75, 1.0],
            "default": 0.875,
            "short_name": "alpha",
        },
    }
    normalized = gen._normalize_param_config(param_config)

    assert set(normalized.keys()) == set(param_config.keys())
    kb_cfg = normalized["attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b"]
    assert kb_cfg["range"] == [0.01, 0.07]
    assert kb_cfg["short_name"] == "k_b"
    assert kb_cfg["default"] is None

    ss_cfg = normalized["attributes.analysis.blockage_model.ss_alpha"]
    assert ss_cfg["short_name"] == "alpha"
    assert ss_cfg["default"] == 0.875


def test_generate_database_uses_run_parameter_sweep(tmp_path, monkeypatch):
    """
    We mock run_parameter_sweep to avoid heavy simulations and just return a tiny dataset.
    """
    system_yaml = tmp_path / "system.yaml"
    ref_power = tmp_path / "ref.nc"
    processed_res = tmp_path / "processed.nc"
    wf_layout = tmp_path / "layout.yaml"
    out_db = tmp_path / "db.nc"

    # Minimal YAMLs and NCs to satisfy __init__ checks
    system_yaml.write_text("name: Tiny\nwind_farm: {turbines: {hub_height: 100, rotor_diameter: 200}}")
    wf_layout.write_text(
        "name: TinyFarm\nlayouts:\n  - coordinates:\n      x: [0, 500]\n      y: [0, 500]\n"
        "turbines:\n  hub_height: 100\n  rotor_diameter: 200\n"
    )

    ref_ds = xr.Dataset(
        data_vars=dict(
            power=(("turbine", "time"), np.ones((2, 3)) * 1.0e6),
        ),
        coords=dict(turbine=[0, 1], time=[0, 1, 2]),
    )
    ref_ds.to_netcdf(ref_power)

    # Processed resource with height & ABL variables
    phys_ds = xr.Dataset(
        data_vars=dict(
            wind_speed=(("flow_case", "height"), np.ones((3, 2)) * 8.0),
            potential_temperature=(("flow_case", "height"), np.ones((3, 2)) * 280.0),
        ),
        coords=dict(
            flow_case=[0, 1, 2],
            height=[10.0, 100.0],
        ),
    )
    phys_ds.to_netcdf(processed_res)

    # Fake result from run_parameter_sweep
    def fake_run_parameter_sweep(turb_rated_power, dat, param_config, reference_power, reference_phys, n_samples, seed, output_dir):
        sample = np.arange(n_samples)
        flow_case = np.arange(reference_power.dims["time"])
        model_bias = np.random.randn(n_samples, len(flow_case))

        return xr.Dataset(
            data_vars=dict(
                model_bias_cap=(("sample", "flow_case"), model_bias),
                pw_power_cap=(("sample", "flow_case"), model_bias + 1.0),
                ref_power_cap=(("sample", "flow_case"), np.ones_like(model_bias)),
            ),
            coords=dict(
                sample=sample,
                flow_case=flow_case,
                k_b=("sample", np.linspace(0.01, 0.07, n_samples)),
            ),
            attrs=dict(
                swept_params=["k_b"],
                param_paths=["attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b"],
                param_defaults='{"k_b": 0.04}',
            ),
        )

    monkeypatch.setattr(
        "wifa_uq.model_error_database.database_gen.run_parameter_sweep",
        fake_run_parameter_sweep,
    )

    gen = DatabaseGenerator(
        nsamples=3,
        param_config={
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": [0.01, 0.07]
        },
        system_yaml_path=system_yaml,
        ref_power_path=ref_power,
        processed_resource_path=processed_res,
        wf_layout_path=wf_layout,
        output_db_path=out_db,
        model="pywake",
    )

    stacked = gen.generate_database()
    # Should have been saved
    assert out_db.exists()

    # Check that stacking added case_index and some layout features
    assert "case_index" in stacked.dims
    assert "Blocking_Distance" in stacked
    assert "Blockage_Ratio" in stacked
    assert "Farm_Length" in stacked
    assert "Farm_Width" in stacked

