# tests/unit/test_preprocessing.py
from pathlib import Path
import numpy as np
import xarray as xr

from wifa_uq.preprocessing.preprocessing import PreprocessingInputs


def test_preprocessing_no_steps_copies_file(tmp_path):
    # Create a tiny input resource file
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
        ref_resource_path=input_path,
        output_path=output_path,
        steps=[]
    )
    out = preproc.run_pipeline()

    assert out == output_path
    assert output_path.exists()

    ds_out = xr.load_dataset(output_path)
    assert "wind_speed" in ds_out
    assert np.allclose(ds_out.wind_speed.values, ds.wind_speed.values)


def test_preprocessing_recalculate_params_minimal(tmp_path):
    # Build vertical profiles so recalculation has something to chew on
    input_path = tmp_path / "resource.nc"

    height = np.linspace(0, 500, 6)
    time = np.arange(4)
    
    wind_speed = np.tile(np.linspace(5.0, 10.0, 6), (4, 1))  # (time, height)
    
    # simple stable stratification
    potential_temperature = 280.0 + 0.005 * height  
    
    # simple directional shear profile: 270° + small change with height
    wind_direction = 270.0 + 0.01 * height
    
    ds = xr.Dataset(
        data_vars=dict(
            wind_speed=(("time", "height"), wind_speed),
            potential_temperature=(("time", "height"), np.tile(potential_temperature, (4, 1))),
            k=(("time", "height"), np.ones((4, 6)) * 0.5),
            wind_direction=(("time", "height"), np.tile(wind_direction, (4, 1))),
        ),
        coords=dict(
            time=time,
            height=height,
        ),
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

    # TI has shape (time, height)
    assert "turbulence_intensity" in ds_out
    assert ds_out.turbulence_intensity.dims == ("time", "height")

    # Wind veer, ABL_height, lapse_rate, etc. may be present
    assert "wind_veer" in ds_out
    assert ds_out.wind_veer.dims == ("time", "height")

    assert "ABL_height" in ds_out
    assert ds_out.ABL_height.dims == ("time",)

    # Lapse rate & capping inversion may or may not be present depending on ci_fitting,
    # but we at least expect the code did not crash.


def test_preprocessing_missing_k_skips_TI(tmp_path):
    # No 'k' field -> TI recalculation should be skipped, but not crash.
    input_path = tmp_path / "resource.nc"

    height = np.linspace(0, 100, 3)
    time = np.arange(2)
    wind_speed = np.ones((2, 3)) * 8.0
    potential_temperature = 280.0 + 0.001 * height

    ds = xr.Dataset(
        data_vars=dict(
            wind_speed=(("time", "height"), wind_speed),
            potential_temperature=(("time", "height"), np.tile(potential_temperature, (2, 1))),
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
    # TI should not exist (because 'k' is missing), but rest should work
    assert "turbulence_intensity" not in ds_out
    assert "wind_speed" in ds_out

