# tests/unit/test_preprocessing.py
"""
Tests for preprocessing module.
"""

import numpy as np
import xarray as xr

from wifa_uq.preprocessing.preprocessing import PreprocessingInputs


class TestPreprocessingNoSteps:
    """Tests for preprocessing with no steps (pass-through mode)."""

    def test_copies_file_when_no_steps(self, tmp_path):
        """With no steps, should just copy input to output."""
        input_path = tmp_path / "resource.nc"
        ds = xr.Dataset(
            data_vars=dict(
                wind_speed=(("time",), np.array([8.0, 9.0, 10.0])),
            ),
            coords=dict(time=np.arange(3)),
        )
        ds.to_netcdf(input_path)

        output_path = tmp_path / "processed.nc"

        preproc = PreprocessingInputs(
            ref_resource_path=input_path, output_path=output_path, steps=[]
        )
        out = preproc.run_pipeline()

        assert out == output_path
        assert output_path.exists()

        ds_out = xr.load_dataset(output_path)
        assert "wind_speed" in ds_out
        np.testing.assert_allclose(ds_out.wind_speed.values, ds.wind_speed.values)


class TestPreprocessingRecalculateParams:
    """Tests for recalculate_params step."""

    def test_minimal_recalculation(self, tmp_path):
        """Test recalculate_params with minimal vertical profile data."""
        input_path = tmp_path / "resource.nc"

        height = np.linspace(0, 500, 6)
        time = np.arange(4)

        wind_speed = np.tile(np.linspace(5.0, 10.0, 6), (4, 1))  # (time, height)
        potential_temperature = 280.0 + 0.005 * height  # stable stratification
        wind_direction = 270.0 + 0.01 * height  # directional shear

        ds = xr.Dataset(
            data_vars=dict(
                wind_speed=(("time", "height"), wind_speed),
                potential_temperature=(
                    ("time", "height"),
                    np.tile(potential_temperature, (4, 1)),
                ),
                k=(("time", "height"), np.ones((4, 6)) * 0.5),
                wind_direction=(("time", "height"), np.tile(wind_direction, (4, 1))),
            ),
            coords=dict(time=time, height=height),
        )
        ds.to_netcdf(input_path)

        output_path = tmp_path / "processed.nc"

        preproc = PreprocessingInputs(
            ref_resource_path=input_path,
            output_path=output_path,
            steps=["recalculate_params"],
        )
        out = preproc.run_pipeline()
        assert out == output_path

        ds_out = xr.load_dataset(output_path)

        # TI should be calculated from k and wind_speed
        assert "turbulence_intensity" in ds_out
        assert ds_out.turbulence_intensity.dims == ("time", "height")

        # Wind veer should be calculated
        assert "wind_veer" in ds_out
        assert ds_out.wind_veer.dims == ("time", "height")

        # ABL_height should be calculated
        assert "ABL_height" in ds_out
        assert ds_out.ABL_height.dims == ("time",)

    def test_missing_k_skips_TI(self, tmp_path):
        """Without 'k' field, TI calculation should be skipped gracefully."""
        input_path = tmp_path / "resource.nc"

        height = np.linspace(0, 100, 3)
        time = np.arange(2)
        wind_speed = np.ones((2, 3)) * 8.0
        potential_temperature = 280.0 + 0.001 * height

        ds = xr.Dataset(
            data_vars=dict(
                wind_speed=(("time", "height"), wind_speed),
                potential_temperature=(
                    ("time", "height"),
                    np.tile(potential_temperature, (2, 1)),
                ),
            ),
            coords=dict(time=time, height=height),
        )
        ds.to_netcdf(input_path)

        output_path = tmp_path / "processed.nc"
        preproc = PreprocessingInputs(
            ref_resource_path=input_path,
            output_path=output_path,
            steps=["recalculate_params"],
        )
        preproc.run_pipeline()

        ds_out = xr.load_dataset(output_path)

        # TI should not exist (because 'k' is missing)
        assert "turbulence_intensity" not in ds_out
        # But other fields should still be present
        assert "wind_speed" in ds_out
