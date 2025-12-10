# tests/unit/test_database_gen.py
"""
Tests for DatabaseGenerator.

These tests use windIO-compliant fixtures and mock run_parameter_sweep
to avoid heavy simulations.
"""

import json
import numpy as np
import xarray as xr
import yaml

from wifa_uq.model_error_database.database_gen import DatabaseGenerator


class TestNormalizeParamConfig:
    """Tests for _normalize_param_config method."""

    def test_accepts_list_format(self):
        """Simple [min, max] format should be normalized to full dict."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)
        param_config = {
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": [
                0.01,
                0.07,
            ],
        }
        normalized = gen._normalize_param_config(param_config)

        key = "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b"
        assert normalized[key]["range"] == [0.01, 0.07]
        assert normalized[key]["short_name"] == "k_b"  # Inferred from path
        assert normalized[key]["default"] is None

    def test_accepts_dict_format(self):
        """Full dict format should pass through with defaults filled in."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)
        param_config = {
            "attributes.analysis.blockage_model.ss_alpha": {
                "range": [0.75, 1.0],
                "default": 0.875,
                "short_name": "alpha",
            },
        }
        normalized = gen._normalize_param_config(param_config)

        key = "attributes.analysis.blockage_model.ss_alpha"
        assert normalized[key]["short_name"] == "alpha"
        assert normalized[key]["default"] == 0.875
        assert normalized[key]["range"] == [0.75, 1.0]

    def test_mixed_formats(self):
        """Both formats in same config should work."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)
        param_config = {
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": [
                0.01,
                0.07,
            ],
            "attributes.analysis.blockage_model.ss_alpha": {
                "range": [0.75, 1.0],
                "default": 0.875,
                "short_name": "alpha",
            },
        }
        normalized = gen._normalize_param_config(param_config)

        assert len(normalized) == 2
        assert (
            normalized[
                "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b"
            ]["short_name"]
            == "k_b"
        )
        assert (
            normalized["attributes.analysis.blockage_model.ss_alpha"]["short_name"]
            == "alpha"
        )


class TestInferRatedPower:
    """Tests for _infer_rated_power method."""

    def test_from_rated_power_key(self, windio_turbine_dict):
        """Should find rated_power from performance.rated_power."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)

        wf_dat = {"turbines": windio_turbine_dict}
        system_dat = {}

        power = gen._infer_rated_power(wf_dat, system_dat)
        assert power == 15.0e6

    def test_from_power_curve(self, windio_turbine_with_power_curve):
        """Should infer rated_power from max of power_curve.power_values."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)

        wf_dat = {"turbines": windio_turbine_with_power_curve}
        system_dat = {}

        power = gen._infer_rated_power(wf_dat, system_dat)
        assert power == 10.0e6  # max of power_values

    def test_from_turbine_name(self):
        """Should parse 'XMW' from turbine name as last resort."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)

        # Turbine with only name, no performance data
        wf_dat = {
            "turbines": {
                "name": "IEA 22MW Offshore Reference",
                "hub_height": 170.0,
                "rotor_diameter": 282.0,
            }
        }
        system_dat = {}

        power = gen._infer_rated_power(wf_dat, system_dat)
        assert power == 22.0e6

    def test_from_system_dat_fallback(self, windio_turbine_dict):
        """Should check system_dat['wind_farm']['turbines'] as fallback."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)

        wf_dat = {}  # No turbines here
        system_dat = {"wind_farm": {"turbines": windio_turbine_dict}}

        power = gen._infer_rated_power(wf_dat, system_dat)
        assert power == 15.0e6

    def test_raises_when_not_found(self):
        """Should raise ValueError when rated power cannot be inferred."""
        gen = DatabaseGenerator.__new__(DatabaseGenerator)

        wf_dat = {"turbines": {"hub_height": 100}}  # No power info
        system_dat = {}

        try:
            gen._infer_rated_power(wf_dat, system_dat)
            assert False, "Should have raised ValueError"
        except ValueError as e:
            assert "Could not find or infer 'rated_power'" in str(e)


class TestGenerateDatabase:
    """Integration tests for generate_database method."""

    def test_full_pipeline_with_mocked_sweep(
        self, tmp_path, windio_system_dict, windio_wind_farm_dict, monkeypatch
    ):
        """
        Test the full generate_database pipeline with mocked run_parameter_sweep.
        """
        # Create temp files
        system_yaml = tmp_path / "system.yaml"
        ref_power = tmp_path / "ref_power.nc"
        processed_res = tmp_path / "processed.nc"
        wf_layout = tmp_path / "layout.yaml"
        out_db = tmp_path / "db.nc"

        # Write windIO-compliant YAML files
        with open(system_yaml, "w") as f:
            yaml.dump(windio_system_dict, f)

        with open(wf_layout, "w") as f:
            yaml.dump(windio_wind_farm_dict, f)

        # Create reference power dataset
        n_turbines = 3
        n_times = 5
        ref_ds = xr.Dataset(
            data_vars=dict(
                power=(("turbine", "time"), np.ones((n_turbines, n_times)) * 1.0e6),
            ),
            coords=dict(turbine=np.arange(n_turbines), time=np.arange(n_times)),
        )
        ref_ds.to_netcdf(ref_power)

        # Create processed resource dataset
        phys_ds = xr.Dataset(
            data_vars=dict(
                wind_speed=(("flow_case", "height"), np.ones((n_times, 2)) * 8.0),
                wind_direction=(("flow_case",), np.full(n_times, 270.0)),
            ),
            coords=dict(flow_case=np.arange(n_times), height=[10.0, 150.0]),
        )
        phys_ds.to_netcdf(processed_res)

        # Mock run_parameter_sweep to return synthetic data
        def fake_run_parameter_sweep(
            run_pywake,
            turb_rated_power,
            dat,
            param_config,
            reference_power,
            reference_phys,
            n_samples,
            seed,
            output_dir,
        ):
            sample = np.arange(n_samples)
            flow_case = np.arange(reference_power.dims["time"])
            model_bias = np.random.randn(n_samples, len(flow_case)) * 0.1

            return xr.Dataset(
                data_vars=dict(
                    model_bias_cap=(("sample", "flow_case"), model_bias),
                    pw_power_cap=(("sample", "flow_case"), model_bias + 0.5),
                    ref_power_cap=(
                        ("sample", "flow_case"),
                        np.ones_like(model_bias) * 0.5,
                    ),
                ),
                coords=dict(
                    sample=sample,
                    flow_case=flow_case,
                    k_b=("sample", np.linspace(0.01, 0.07, n_samples)),
                ),
                attrs=dict(
                    swept_params=["k_b"],
                    param_paths=[
                        "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b"
                    ],
                    param_defaults=json.dumps({"k_b": 0.04}),
                ),
            )

        monkeypatch.setattr(
            "wifa_uq.model_error_database.database_gen.run_parameter_sweep",
            fake_run_parameter_sweep,
        )

        # Run the generator
        gen = DatabaseGenerator(
            nsamples=4,
            param_config={
                "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": [
                    0.01,
                    0.07,
                ]
            },
            system_yaml_path=system_yaml,
            ref_power_path=ref_power,
            processed_resource_path=processed_res,
            wf_layout_path=wf_layout,
            output_db_path=out_db,
            model="pywake",
        )

        stacked = gen.generate_database()

        # Verify output
        assert out_db.exists()
        assert "case_index" in stacked.dims
        assert "model_bias_cap" in stacked
        assert "Blocking_Distance" in stacked
        assert "Blockage_Ratio" in stacked
        assert "Farm_Length" in stacked
        assert "Farm_Width" in stacked
        assert "turb_rated_power" in stacked
