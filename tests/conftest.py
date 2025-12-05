# tests/conftest.py
import pytest
import numpy as np
import xarray as xr


@pytest.fixture
def tiny_bias_db():
    """
    Synthetic stacked database mimicking results_stacked_hh.nc, but tiny.

    Dimensions before stacking:
      - sample: 4
      - flow_case: 6
      - wind_farm: 1

    Variables:
      - model_bias_cap(sample, flow_case)
      - pw_power_cap(sample, flow_case)
      - ref_power_cap(sample, flow_case)
      - ABL_height(flow_case)
      - wind_veer(flow_case)
      - lapse_rate(flow_case)

    Coords:
      - k_b(sample)
      - ss_alpha(sample)
      - case_index(flow_case)
      - wind_farm (after expand_dims)
    """
    n_samples = 4
    n_cases = 6

    sample = np.arange(n_samples)
    flow_case = np.arange(n_cases)

    # Swept parameters
    k_b = np.linspace(0.01, 0.07, n_samples)
    ss_alpha = np.linspace(0.75, 1.0, n_samples)

    # Physical features per flow_case
    ABL_height = np.linspace(200.0, 800.0, n_cases)       # strong signal
    wind_veer = np.linspace(0.0, 0.01, n_cases)          # weak
    lapse_rate = np.linspace(-0.005, 0.005, n_cases)     # weak

    # Synthetic bias: mostly driven by ABL_height plus param effects
    # shape (sample, flow_case)
    model_bias_cap = (
        0.5 * k_b[:, None] +
        0.2 * ss_alpha[:, None] +
        0.1 * (ABL_height[None, :] / 1000.0)
    )

    pw_power_cap = 0.5 + model_bias_cap   # arbitrary
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
            # For DefaultParams we want a proper dict, not JSON string
            param_defaults={"k_b": 0.04, "ss_alpha": 0.875},
        ),
    )

    # Add wind_farm dimension (length 1), like DatabaseGenerator does
    ds = ds.expand_dims(dim={"wind_farm": ["TinyFarm"]})

    # Stack along (wind_farm, flow_case) to build case_index,
    # again mimicking DatabaseGenerator.generate_database()
    stacked = ds.stack(case_index=("wind_farm", "flow_case"))
    stacked = stacked.reset_index("case_index")

    return stacked

