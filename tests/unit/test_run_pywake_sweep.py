# tests/unit/test_run_pywake_sweep.py
from pathlib import Path
from unittest.mock import patch
import numpy as np
import xarray as xr

from wifa_uq.model_error_database.run_pywake_sweep import run_parameter_sweep


@patch("wifa_uq.model_error_database.run_pywake_sweep.run_pywake")
def test_run_parameter_sweep_shapes_and_metadata(mock_run_pywake, tmp_path):
    """
    We mock run_pywake and xarray.open_dataset("results/turbine_data.nc")
    so no heavy simulation is run.
    """
    # Fake turbine_data.nc that pywake would write
    # shape: turbine × time
    pw_ds = xr.Dataset(
        data_vars=dict(
            power=(("turbine", "time"), np.ones((3, 5)) * 1.5e6)
        ),
        coords=dict(
            turbine=np.arange(3),
            time=np.arange(5),
        ),
    )

    # Reference power of same shape
    ref_ds = xr.Dataset(
        data_vars=dict(
            power=(("turbine", "time"), np.ones((3, 5)) * 1.4e6)
        ),
        coords=dict(
            turbine=np.arange(3),
            time=np.arange(5),
        ),
    )

    # Patch xr.open_dataset to always give us pw_ds when reading turbine_data
    def fake_open_dataset(path, *args, **kwargs):
        # We only care about "results/turbine_data.nc" in this test
        return pw_ds

    with patch("xarray.open_dataset", side_effect=fake_open_dataset):
        merged = run_parameter_sweep(
            turb_rated_power=2e6,
            dat={"attributes": {"analysis": {}}},
            param_config={
                "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": {
                    "range": [0.01, 0.07],
                    "default": 0.04,
                    "short_name": "k_b",
                }
            },
            reference_power=ref_ds,
            reference_physical_inputs=None,
            n_samples=4,
            seed=42,
            output_dir=tmp_path / "samples",
        )

    # Assertions
    assert "model_bias_cap" in merged
    # shape (sample, flow_case==time)
    assert merged.model_bias_cap.shape == (4, 5)
    assert "pw_power_cap" in merged
    assert "ref_power_cap" in merged

    # swept param coords
    assert "k_b" in merged.coords
    assert merged.attrs["swept_params"] == ["k_b"]
    assert "param_paths" in merged.attrs

