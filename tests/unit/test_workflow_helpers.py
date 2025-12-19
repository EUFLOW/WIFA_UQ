import pytest
from pathlib import Path

from wifa_uq import workflow


def test_get_class_from_map_ok():
    cls = workflow.get_class_from_map("MinBiasCalibrator")
    assert cls.__name__ == "MinBiasCalibrator"


def test_get_class_from_map_unknown_raises():
    with pytest.raises(ValueError, match="Unknown class"):
        workflow.get_class_from_map("Nope")


@pytest.mark.parametrize(
    "model,expected_type",
    [
        ("Linear", "linear"),
        ("XGB", "tree"),
        ("SIRPolynomial", "sir"),
        ("PCE", "pce"),
    ],
)
def test_build_predictor_pipeline_model_types(model, expected_type):
    pipe, model_type = workflow.build_predictor_pipeline(model, {})
    assert model_type == expected_type
    assert pipe is not None


def test_build_predictor_pipeline_unknown_raises():
    with pytest.raises(ValueError, match="Unknown model"):
        workflow.build_predictor_pipeline("BADMODEL", {})


def test_is_multi_farm_config():
    assert workflow._is_multi_farm_config(
        {"farms": [{"name": "A", "system_config": "x"}], "paths": {}}
    )
    assert workflow._is_multi_farm_config(
        {"paths": {"farms": [{"name": "A", "system_config": "x"}]}}
    )
    assert not workflow._is_multi_farm_config({"paths": {"system_config": "a.yaml"}})


def test_validate_farm_configs_duplicate_raises():
    with pytest.raises(ValueError, match="Duplicate"):
        workflow._validate_farm_configs(
            [
                {"name": "A", "system_config": "a.yaml"},
                {"name": "A", "system_config": "b.yaml"},
            ]
        )


def test_resolve_farm_paths(tmp_path: Path):
    base_dir = tmp_path
    farm = {
        "name": "Farm1",
        "system_config": "sys.yaml",
        "reference_power": "pow.nc",
    }
    out = workflow._resolve_farm_paths(farm, base_dir)
    assert out["name"] == "Farm1"
    assert out["system_config"] == base_dir / "sys.yaml"
    assert out["reference_power"] == base_dir / "pow.nc"
    assert "reference_resource" not in out
