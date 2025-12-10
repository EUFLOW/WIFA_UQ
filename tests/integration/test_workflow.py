# tests/integration/test_workflow.py
from pathlib import Path
from wifa_uq.workflow import run_workflow

# Go up from tests/integration/ to project root, then into examples/
EXAMPLES_DIR = Path(__file__).parent.parent.parent / "examples"


def test_kul_single_farm_xgb_example():
    """Test the KUL single farm XGB example workflow."""
    config_path = EXAMPLES_DIR / "kul_single_farm_xgb_example.yaml"
    assert config_path.exists(), f"Config file not found: {config_path}"

    run_workflow(config_path)
