# tests/unit/test_multi_farm_gen.py
"""
Tests for multi-farm database generation.
"""

import pytest
import numpy as np
import xarray as xr
from unittest.mock import patch

from wifa_uq.model_error_database.multi_farm_gen import (
    MultiFarmDatabaseGenerator,
    generate_multi_farm_database,
)


class TestMultiFarmDatabaseGenerator:
    """Tests for MultiFarmDatabaseGenerator class."""

    def test_validates_farm_configs(self, tmp_path):
        """Should raise ValueError for missing required keys."""
        invalid_configs = [
            {"name": "TestFarm"}  # Missing other required keys
        ]

        with pytest.raises(ValueError, match="missing required keys"):
            MultiFarmDatabaseGenerator(
                farm_configs=invalid_configs,
                param_config={},
                n_samples=10,
                output_dir=tmp_path,
            )

    def test_detects_duplicate_farm_names(self, tmp_path, multi_farm_configs):
        """Should raise ValueError for duplicate farm names."""
        # Duplicate the first farm config with same name
        configs = multi_farm_configs.copy()
        configs.append(configs[0].copy())  # Duplicate

        with pytest.raises(ValueError, match="Duplicate farm name"):
            MultiFarmDatabaseGenerator(
                farm_configs=configs,
                param_config={},
                n_samples=10,
                output_dir=tmp_path,
            )

    @patch("wifa_uq.model_error_database.multi_farm_gen.DatabaseGenerator")
    def test_generates_combined_database(
        self, mock_db_gen_cls, tmp_path, multi_farm_configs
    ):
        """Test full pipeline with mocked DatabaseGenerator."""
        n_samples = 4

        # Create mock databases for each farm
        def create_mock_db(farm_name, n_cases):
            return xr.Dataset(
                data_vars=dict(
                    model_bias_cap=(
                        ("sample", "case_index"),
                        np.random.randn(n_samples, n_cases) * 0.1,
                    ),
                    pw_power_cap=(
                        ("sample", "case_index"),
                        np.ones((n_samples, n_cases)) * 0.5,
                    ),
                    ref_power_cap=(
                        ("sample", "case_index"),
                        np.ones((n_samples, n_cases)) * 0.45,
                    ),
                    ABL_height=("case_index", np.linspace(200, 800, n_cases)),
                ),
                coords=dict(
                    sample=np.arange(n_samples),
                    case_index=np.arange(n_cases),
                    wind_farm=("case_index", [farm_name] * n_cases),
                    k_b=("sample", np.linspace(0.01, 0.07, n_samples)),
                ),
                attrs=dict(
                    swept_params=["k_b"],
                ),
            )

        # Setup mock to return different databases per call
        farm_dbs = [
            create_mock_db("FarmA", 5),
            create_mock_db("FarmB", 6),
            create_mock_db("FarmC", 7),
        ]

        call_count = [0]

        def mock_generate():
            db = farm_dbs[call_count[0]]
            call_count[0] += 1
            return db

        mock_instance = mock_db_gen_cls.return_value
        mock_instance.generate_database.side_effect = mock_generate

        # Run generator
        param_config = {
            "attributes.analysis.wind_deficit_model.wake_expansion_coefficient.k_b": {
                "range": [0.01, 0.07],
                "default": 0.04,
                "short_name": "k_b",
            }
        }

        generator = MultiFarmDatabaseGenerator(
            farm_configs=multi_farm_configs,
            param_config=param_config,
            n_samples=n_samples,
            output_dir=tmp_path,
            run_preprocessing=False,  # Skip preprocessing for test
        )

        combined = generator.generate_database()

        # Verify combined database
        assert len(combined.case_index) == 5 + 6 + 7  # Total cases
        assert len(combined.sample) == n_samples

        # Verify wind_farm values are preserved
        unique_farms = np.unique(combined.wind_farm.values)
        assert len(unique_farms) == 3

        # Verify case_index is unique
        assert len(np.unique(combined.case_index.values)) == len(combined.case_index)

    def test_combine_databases_reindexes_correctly(self, tmp_path):
        """Test that _combine_databases creates unique case indices."""
        generator = MultiFarmDatabaseGenerator.__new__(MultiFarmDatabaseGenerator)

        # Create two small databases with overlapping case indices
        db1 = xr.Dataset(
            data_vars=dict(value=(("sample", "case_index"), np.ones((2, 3)))),
            coords=dict(
                sample=[0, 1],
                case_index=[0, 1, 2],
                wind_farm=("case_index", ["A", "A", "A"]),
            ),
            attrs=dict(swept_params=["k"]),
        )

        db2 = xr.Dataset(
            data_vars=dict(value=(("sample", "case_index"), np.ones((2, 4)) * 2)),
            coords=dict(
                sample=[0, 1],
                case_index=[0, 1, 2, 3],  # Overlapping indices!
                wind_farm=("case_index", ["B", "B", "B", "B"]),
            ),
            attrs=dict(swept_params=["k"]),
        )

        combined = generator._combine_databases([db1, db2])

        # Should have 7 unique case indices
        assert len(combined.case_index) == 7
        assert list(combined.case_index.values) == [0, 1, 2, 3, 4, 5, 6]

        # Values should be preserved
        assert combined.value.isel(sample=0, case_index=0).values == 1  # From db1
        assert combined.value.isel(sample=0, case_index=3).values == 2  # From db2


class TestGenerateMultiFarmDatabase:
    """Tests for the convenience function."""

    @patch("wifa_uq.model_error_database.multi_farm_gen.MultiFarmDatabaseGenerator")
    def test_calls_generator_correctly(
        self, mock_gen_cls, tmp_path, multi_farm_configs
    ):
        """Should instantiate and call generator."""
        mock_instance = mock_gen_cls.return_value
        mock_instance.generate_database.return_value = xr.Dataset()

        generate_multi_farm_database(
            farm_configs=multi_farm_configs,
            param_config={"test": [0, 1]},
            n_samples=10,
            output_dir=tmp_path,
        )

        mock_gen_cls.assert_called_once()
        mock_instance.generate_database.assert_called_once()
