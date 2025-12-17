"""
Tests for validating example YAML configurations against the Pydantic schema.

Run with: pytest tests/unit/test_config_schema.py -v
"""

import pytest
import yaml
from pathlib import Path

from wifa_uq.workflow_schema import WifaUQConfig


EXAMPLES_DIR = Path(__file__).resolve().parent.parent.parent / "examples"


def get_example_yaml_files() -> list[Path]:
    """Get all YAML files in examples/ directory (non-recursive)."""
    if not EXAMPLES_DIR.exists():
        return []
    return sorted(EXAMPLES_DIR.glob("*.yaml"))


# Print found files at module load time
_yaml_files = get_example_yaml_files()
print(f"\nFound {len(_yaml_files)} YAML files in {EXAMPLES_DIR}:")
for f in _yaml_files:
    print(f"  - {f.name}")


@pytest.mark.parametrize("yaml_file", _yaml_files, ids=lambda p: p.name)
def test_example_config_validates(yaml_file: Path):
    """Test that each example YAML file validates against the schema."""
    with open(yaml_file, "r") as f:
        raw_config = yaml.safe_load(f)

    config = WifaUQConfig(**raw_config)

    assert config.error_prediction is not None
    assert len(config.error_prediction.features) > 0
