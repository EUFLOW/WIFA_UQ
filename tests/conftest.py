# tests/conftest.py
"""
Pytest fixtures for WIFA-UQ tests.

These fixtures provide windIO-compliant test data structures.
"""
import pytest
import numpy as np
import xarray as xr
from pathlib import Path


@pytest.fixture
def tiny_bias_db():
    """
    Synthetic stacked database mimicking results_stacked_hh.nc, but tiny.
    
    This fixture represents the OUTPUT of DatabaseGenerator.generate_database(),
    which includes all the variables that downstream code (like run_cross_validation)
    expects.

    Dimensions:
      - sample: 4  (parameter samples)
      - case_index: 6  (stacked wind_farm × flow_case)

    Variables required by run_cross_validation:
      - model_bias_cap(sample, case_index)
      - pw_power_cap(sample, case_index)
      - ref_power_cap(sample, case_index)
      - turb_rated_power(wind_farm)  # Added by DatabaseGenerator
      - Physical features: ABL_height, wind_veer, lapse_rate

    Coords:
      - k_b(sample), ss_alpha(sample)  # Swept parameters
      - case_index, wind_farm
    """
    n_samples = 4
    n_cases = 6

    sample = np.arange(n_samples)
    flow_case = np.arange(n_cases)

    # Swept parameters (what we're calibrating)
    k_b = np.linspace(0.01, 0.07, n_samples)
    ss_alpha = np.linspace(0.75, 1.0, n_samples)

    # Physical features per flow_case
    ABL_height = np.linspace(200.0, 800.0, n_cases)
    wind_veer = np.linspace(0.0, 0.01, n_cases)
    lapse_rate = np.linspace(-0.005, 0.005, n_cases)

    # Synthetic bias: driven by ABL_height plus param effects
    # shape (sample, flow_case)
    model_bias_cap = (
        0.5 * k_b[:, None] +
        0.2 * ss_alpha[:, None] +
        0.1 * (ABL_height[None, :] / 1000.0)
    )

    pw_power_cap = 0.5 + model_bias_cap
    ref_power_cap = 0.5 * np.ones_like(pw_power_cap)

    ds = xr.Dataset(
        data_vars=dict(
            model_bias_cap=(("sample", "flow_case"), model_bias_cap),
            pw_power_cap=(("sample", "flow_case"), pw_power_cap),
            ref_power_cap=(("sample", "flow_case"), ref_power_cap),
            ABL_height=("flow_case", ABL_height),
            wind_veer=("flow_case", wind_veer),
            lapse_rate=("flow_case", lapse_rate),
        ),
        coords=dict(
            sample=sample,
            flow_case=flow_case,
            case_index=("flow_case", np.arange(n_cases)),
            k_b=("sample", k_b),
            ss_alpha=("sample", ss_alpha),
        ),
        attrs=dict(
            swept_params=["k_b", "ss_alpha"],
            param_defaults={"k_b": 0.04, "ss_alpha": 0.875},
        ),
    )

    # Add wind_farm dimension (mimics DatabaseGenerator)
    ds = ds.expand_dims(dim={"wind_farm": ["TinyFarm"]})
    
    # Add turb_rated_power - REQUIRED by run_cross_validation
    ds["turb_rated_power"] = xr.DataArray([2.0e6], dims=["wind_farm"])

    # Stack to create case_index dimension (mimics DatabaseGenerator)
    stacked = ds.stack(case_index=("wind_farm", "flow_case"))
    stacked = stacked.reset_index("case_index")

    return stacked


@pytest.fixture
def windio_turbine_dict():
    """
    A windIO-compliant turbine definition dict.
    
    Based on windIO/plant/turbine.yaml schema.
    Includes performance.rated_power which is the cleanest way to specify power.
    """
    return {
        "name": "Test Turbine 15MW",
        "hub_height": 150.0,
        "rotor_diameter": 240.0,
        "performance": {
            "rated_power": 15.0e6,  # 15 MW in Watts
            "rated_wind_speed": 10.5,
            "cutin_wind_speed": 3.0,
            "cutout_wind_speed": 25.0,
            "Ct_curve": {
                "Ct_values": [0.0, 0.8, 0.8, 0.4, 0.0],
                "Ct_wind_speeds": [0.0, 4.0, 10.0, 15.0, 25.0],
            },
        },
    }


@pytest.fixture
def windio_turbine_with_power_curve():
    """
    A windIO-compliant turbine definition using power_curve instead of rated_power.
    
    This tests the fallback inference path in _infer_rated_power.
    """
    return {
        "name": "Test Turbine",
        "hub_height": 100.0,
        "rotor_diameter": 200.0,
        "performance": {
            "power_curve": {
                "power_values": [0.0, 1.0e6, 5.0e6, 10.0e6, 10.0e6, 0.0],
                "power_wind_speeds": [0.0, 4.0, 8.0, 12.0, 20.0, 25.0],
            },
            "Ct_curve": {
                "Ct_values": [0.0, 0.8, 0.8, 0.4, 0.0],
                "Ct_wind_speeds": [0.0, 4.0, 10.0, 15.0, 25.0],
            },
        },
    }


@pytest.fixture
def windio_wind_farm_dict(windio_turbine_dict):
    """
    A windIO-compliant wind_farm definition dict.
    
    Based on windIO/plant/wind_farm.yaml schema.
    """
    return {
        "name": "TestFarm",
        "layouts": [
            {
                "coordinates": {
                    "x": [0.0, 500.0, 250.0],  # Triangle layout
                    "y": [0.0, 0.0, 400.0],    # Not collinear!
                }
            }
        ],
        "turbines": windio_turbine_dict,
    }


@pytest.fixture
def windio_system_dict(windio_wind_farm_dict):
    """
    A windIO-compliant wind_energy_system definition dict.
    
    Based on windIO/plant/wind_energy_system.yaml schema.
    This is a minimal system config for testing.
    """
    return {
        "name": "TestSystem",
        "wind_farm": windio_wind_farm_dict,
        "attributes": {
            "analysis": {
                "wind_deficit_model": {
                    "name": "Bastankhah2014",
                    "wake_expansion_coefficient": {
                        "k_a": 0.04,
                        "k_b": 0.0,
                    },
                },
                "blockage_model": {
                    "name": "SelfSimilarityDeficit2020",
                    "ss_alpha": 0.875,
                },
            },
        },
    }


@pytest.fixture
def sample_reference_power():
    """Reference power dataset for testing."""
    return xr.Dataset(
        data_vars=dict(
            power=(("turbine", "time"), np.ones((3, 5)) * 1.0e6),
        ),
        coords=dict(
            turbine=[0, 1, 2],
            time=[0, 1, 2, 3, 4],
        ),
    )


@pytest.fixture
def sample_physical_inputs():
    """Processed physical inputs dataset for testing."""
    return xr.Dataset(
        data_vars=dict(
            wind_speed=(("flow_case", "height"), np.ones((5, 3)) * 8.0),
            wind_direction=(("flow_case",), np.array([270.0, 280.0, 290.0, 300.0, 310.0])),
            potential_temperature=(("flow_case", "height"), np.ones((5, 3)) * 280.0),
        ),
        coords=dict(
            flow_case=[0, 1, 2, 3, 4],
            height=[10.0, 100.0, 200.0],
        ),
    )


@pytest.fixture
def pywake_param_config():
    """
    Standard parameter config for PyWake sweep tests.
    
    Keys are dot-separated paths matching windIO system.yaml structure.
    """
    return {
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
