import pytest
from pathlib import Path
import windIO

# All system config paths from the YAML examples
SYSTEM_CONFIG_PATHS = [
    # EDF Datasets
    "examples/data/EDF_datasets/HR1/wind_energy_system.yaml",
    "examples/data/EDF_datasets/HR2/wind_energy_system.yaml",
    "examples/data/EDF_datasets/HR3/wind_energy_system.yaml",
    "examples/data/EDF_datasets/NYSTED1/wind_energy_system.yaml",
    "examples/data/EDF_datasets/NYSTED2/wind_energy_system.yaml",
    "examples/data/EDF_datasets/VirtWF_ABL_IEA10/wind_energy_system.yaml",
    "examples/data/EDF_datasets/VirtWF_ABL_IEA22/wind_energy_system.yaml",
    "examples/data/EDF_datasets/VirtWF_ABL_IEA15_ali_DX5_DY5/wind_energy_system.yaml",
    "examples/data/EDF_datasets/VirtWF_ABL_IEA15_stag_DX5_DY5/system_staggered_DX5D_DY5D_Turbine_Number100.yaml",
    "examples/data/EDF_datasets/VirtWF_ABL_IEA15_stag_DX5_DY7p5/system_staggered_DX5D_DY7.5D_Turbine_Number100.yaml",
    "examples/data/EDF_datasets/VirtWF_ABL_IEA15_stag_DX7p5_DY5/system_staggered_DX7.5D_DY5D_Turbine_Number100.yaml",
    # KUL_LES Dataset
    "examples/data/KUL_LES/wind_energy_system/system_pywake.yaml",
]


def get_base_path():
    """Determine the base path for data files relative to the windIO package."""
    # Try plant examples directory first
    plant_ex_path = Path(windIO.plant_ex.__file__).parent

    # Check if data folder exists at various levels
    for parent_level in range(4):
        candidate = (
            plant_ex_path.parents[parent_level] if parent_level > 0 else plant_ex_path
        )
        data_path = candidate / "data"
        if data_path.exists():
            return candidate

    # Fall back to current working directory
    return Path.cwd()


def get_existing_system_files():
    """Return list of system config files that exist on the filesystem."""
    base_path = get_base_path()
    existing_files = []

    for rel_path in SYSTEM_CONFIG_PATHS:
        full_path = base_path / rel_path
        if full_path.exists():
            existing_files.append((rel_path, full_path))

    return existing_files


@pytest.fixture(scope="module")
def base_path():
    """Fixture providing the base path for data files."""
    return get_base_path()


class TestWindEnergySystemSchema:
    """Test suite for validating wind energy system configuration files."""

    @pytest.mark.parametrize(
        "rel_path,full_path",
        get_existing_system_files(),
        ids=[p[0] for p in get_existing_system_files()],
    )
    def test_system_file_validates_against_schema(self, rel_path, full_path):
        """
        Test that each system configuration file validates against the
        plant/wind_energy_system schema.
        """
        windIO.validate(
            input=full_path,
            schema_type="plant/wind_energy_system",
            restrictive=True,
            defaults=False,
        )

    def test_at_least_one_system_file_exists(self, base_path):
        """Ensure we're actually testing something - at least one file should exist."""
        existing = get_existing_system_files()
        assert len(existing) > 0, (
            f"No system config files found. Base path: {base_path}. "
            f"Checked paths: {SYSTEM_CONFIG_PATHS[:3]}..."
        )

    def test_report_missing_files(self, base_path):
        """Report which expected system files are missing (warning, not failure)."""
        existing_paths = {p[0] for p in get_existing_system_files()}
        missing = [p for p in SYSTEM_CONFIG_PATHS if p not in existing_paths]

        if missing:
            pytest.skip(f"Missing {len(missing)} system files: {missing[:3]}...")


# Alternative: Test all files in a single test with detailed reporting
def test_validate_all_system_configs():
    """
    Validate all system configuration files against the wind_energy_system schema.
    Collects all failures and reports them together.
    """
    base_path = get_base_path()
    failures = []
    successes = []
    skipped = []

    for rel_path in SYSTEM_CONFIG_PATHS:
        full_path = base_path / rel_path

        if not full_path.exists():
            skipped.append(rel_path)
            continue

        try:
            windIO.validate(
                input=full_path,
                schema_type="plant/wind_energy_system",
                restrictive=True,
                defaults=False,
            )
            successes.append(rel_path)
        except Exception as e:
            failures.append((rel_path, str(e)))

    # Print summary
    print(f"\n{'=' * 60}")
    print("System Schema Validation Summary")
    print(f"{'=' * 60}")
    print(f"  Passed:  {len(successes)}")
    print(f"  Failed:  {len(failures)}")
    print(f"  Skipped: {len(skipped)} (file not found)")
    print(f"{'=' * 60}")

    if failures:
        failure_msg = "\n\nValidation failures:\n"
        for path, error in failures:
            failure_msg += f"\n{'-' * 40}\n{path}:\n{error[:500]}...\n"
        pytest.fail(failure_msg)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
