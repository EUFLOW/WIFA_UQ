# tests/unit/test_run_pywake_sweep.py
"""
Tests for run_pywake_sweep module.

These tests mock run_pywake and xarray.open_dataset to avoid actual simulations.
"""

from unittest.mock import patch
import numpy as np
import xarray as xr

from wifa import run_pywake
from wifa_uq.model_error_database.run_sweep import (
    run_parameter_sweep,
    set_nested_dict_value,
    create_parameter_samples,
)


class TestSetNestedDictValue:
    """Tests for set_nested_dict_value helper."""

    def test_sets_shallow_value(self):
        d = {"a": {"b": 1}}
        set_nested_dict_value(d, ["a", "b"], 99)
        assert d["a"]["b"] == 99

    def test_sets_deep_value(self):
        d = {"a": {"b": {"c": {"d": 1}}}}
        set_nested_dict_value(d, ["a", "b", "c", "d"], 42)
        assert d["a"]["b"]["c"]["d"] == 42

    def test_windio_style_path(self):
        """Test with actual windIO-style nested structure."""
        d = {
            "attributes": {
                "analysis": {
                    "wind_deficit_model": {"wake_expansion_coefficient": {"k_b": 0.04}}
                }
            }
        }
        path = "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b".split(
            "."
        )
        set_nested_dict_value(d, path, 0.07)
        assert (
            d["attributes"]["analysis"]["wind_deficit_model"][
                "wake_expansion_coefficient"
            ]["k_b"]
            == 0.07
        )


class TestCreateParameterSamples:
    """Tests for create_parameter_samples helper."""

    def test_generates_correct_shape(self, pywake_param_config):
        samples = create_parameter_samples(pywake_param_config, n_samples=10, seed=42)

        for param_path in pywake_param_config:
            assert param_path in samples
            assert len(samples[param_path]) == 10

    def test_respects_ranges(self, pywake_param_config):
        samples = create_parameter_samples(pywake_param_config, n_samples=100, seed=42)

        for param_path, config in pywake_param_config.items():
            min_val, max_val = config["range"]
            assert all(samples[param_path] >= min_val)
            assert all(samples[param_path] <= max_val)

    def test_first_sample_is_default(self, pywake_param_config):
        samples = create_parameter_samples(pywake_param_config, n_samples=10, seed=42)

        for param_path, config in pywake_param_config.items():
            if config.get("default") is not None:
                assert samples[param_path][0] == config["default"]

    def test_reproducible_with_seed(self, pywake_param_config):
        samples1 = create_parameter_samples(pywake_param_config, n_samples=10, seed=42)
        samples2 = create_parameter_samples(pywake_param_config, n_samples=10, seed=42)

        for param_path in pywake_param_config:
            np.testing.assert_array_equal(samples1[param_path], samples2[param_path])


class TestRunParameterSweep:
    """Tests for run_parameter_sweep function."""

    @patch("wifa_uq.model_error_database.run_sweep.run_pywake")
    def test_output_shapes_and_metadata(
        self, mock_run_pywake, tmp_path, windio_system_dict
    ):
        """Test that run_parameter_sweep returns correctly shaped data with metadata."""
        n_turbines = 3
        n_times = 5
        n_samples = 4

        # Fake turbine_data.nc that pywake would write
        pw_ds = xr.Dataset(
            data_vars=dict(
                power=(("turbine", "time"), np.ones((n_turbines, n_times)) * 1.5e6)
            ),
            coords=dict(turbine=np.arange(n_turbines), time=np.arange(n_times)),
        )

        # Reference power
        ref_ds = xr.Dataset(
            data_vars=dict(
                power=(("turbine", "time"), np.ones((n_turbines, n_times)) * 1.4e6)
            ),
            coords=dict(turbine=np.arange(n_turbines), time=np.arange(n_times)),
        )

        # Patch xr.open_dataset to return pw_ds
        def fake_open_dataset(path, *args, **kwargs):
            return pw_ds

        param_config = {
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": {
                "range": [0.01, 0.07],
                "default": 0.04,
                "short_name": "k_b",
            }
        }

        with patch("xarray.open_dataset", side_effect=fake_open_dataset):
            merged = run_parameter_sweep(
                run_func=run_pywake,
                turb_rated_power=2e6,
                dat=windio_system_dict,  # Use proper windIO structure
                param_config=param_config,
                reference_power=ref_ds,
                reference_physical_inputs=None,
                n_samples=n_samples,
                seed=42,
                output_dir=tmp_path / "samples",
            )

        # Check output structure
        assert "model_bias_cap" in merged
        assert merged.model_bias_cap.shape == (n_samples, n_times)
        assert "pw_power_cap" in merged
        assert "ref_power_cap" in merged

        # Check swept param coords
        assert "k_b" in merged.coords
        assert len(merged.coords["k_b"]) == n_samples

        # Check metadata
        assert merged.attrs["swept_params"] == ["k_b"]
        assert "param_paths" in merged.attrs

    @patch("wifa_uq.model_error_database.run_sweep.run_pywake")
    def test_multiple_parameters(self, mock_run_pywake, tmp_path, windio_system_dict):
        """Test sweep with multiple parameters."""
        n_turbines = 2
        n_times = 3
        n_samples = 5

        pw_ds = xr.Dataset(
            data_vars=dict(
                power=(("turbine", "time"), np.ones((n_turbines, n_times)) * 1.5e6)
            ),
            coords=dict(turbine=np.arange(n_turbines), time=np.arange(n_times)),
        )

        ref_ds = xr.Dataset(
            data_vars=dict(
                power=(("turbine", "time"), np.ones((n_turbines, n_times)) * 1.4e6)
            ),
            coords=dict(turbine=np.arange(n_turbines), time=np.arange(n_times)),
        )

        param_config = {
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": {
                "range": [0.01, 0.07],
                "default": 0.04,
                "short_name": "k_b",
            },
            "attributes.analysis.blockage_model.ss_alpha": {
                "range": [0.75, 1.0],
                "default": 0.875,
                "short_name": "ss_alpha",
            },
        }

        with patch("xarray.open_dataset", return_value=pw_ds):
            merged = run_parameter_sweep(
                run_func=run_pywake,
                turb_rated_power=2e6,
                dat=windio_system_dict,
                param_config=param_config,
                reference_power=ref_ds,
                reference_physical_inputs=None,
                n_samples=n_samples,
                seed=42,
                output_dir=tmp_path / "samples",
            )

        # Both parameters should be in coords
        assert "k_b" in merged.coords
        assert "ss_alpha" in merged.coords
        assert set(merged.attrs["swept_params"]) == {"k_b", "ss_alpha"}

    @patch("wifa_uq.model_error_database.run_sweep.run_pywake")
    def test_bias_calculation_correctness(
        self, mock_run_pywake, tmp_path, windio_system_dict
    ):
        """Test that bias is calculated correctly as (pywake - reference) / rated_power."""
        n_turbines = 2
        n_times = 3
        rated_power = 2e6

        # PyWake produces 1.5 MW average
        pw_ds = xr.Dataset(
            data_vars=dict(
                power=(("turbine", "time"), np.ones((n_turbines, n_times)) * 1.5e6)
            ),
            coords=dict(turbine=np.arange(n_turbines), time=np.arange(n_times)),
        )

        # Reference is 1.4 MW average
        ref_ds = xr.Dataset(
            data_vars=dict(
                power=(("turbine", "time"), np.ones((n_turbines, n_times)) * 1.4e6)
            ),
            coords=dict(turbine=np.arange(n_turbines), time=np.arange(n_times)),
        )

        param_config = {
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": {
                "range": [0.01, 0.07],
                "default": 0.04,
                "short_name": "k_b",
            },
        }

        with patch("xarray.open_dataset", return_value=pw_ds):
            merged = run_parameter_sweep(
                run_func=run_pywake,
                turb_rated_power=rated_power,
                dat=windio_system_dict,
                param_config=param_config,
                reference_power=ref_ds,
                reference_physical_inputs=None,
                n_samples=2,
                seed=42,
                output_dir=tmp_path / "samples",
            )

        # Expected bias: (1.5e6 - 1.4e6) / 2e6 = 0.05
        expected_bias = (1.5e6 - 1.4e6) / rated_power
        np.testing.assert_allclose(
            merged.model_bias_cap.values, expected_bias, rtol=1e-10
        )
