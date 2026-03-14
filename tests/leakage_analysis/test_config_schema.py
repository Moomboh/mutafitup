"""Tests for the leakage_analysis config schema validation."""

import json
from pathlib import Path

import pytest
import yaml

try:
    import jsonschema
except ImportError:
    jsonschema = None


SCHEMA_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "schemas" / "config.schema.yml"
)


def _load_schema() -> dict:
    with SCHEMA_PATH.open() as fh:
        return yaml.safe_load(fh)


def _validate_leakage_analysis(instance: dict, schema: dict | None = None):
    """Validate *instance* against the top-level leakage_analysis sub-schema.

    The instance should have the form: {"profiles": {"name": { ... }}}.
    """
    if schema is None:
        schema = _load_schema()
    leakage_schema = schema.get("properties", {}).get("leakage_analysis")
    if leakage_schema is None:
        pytest.fail("leakage_analysis not found in config schema")
    resolver = jsonschema.RefResolver.from_schema(schema)
    jsonschema.validate(instance, leakage_schema, resolver=resolver)


def _validate_profile(instance: dict, schema: dict | None = None):
    """Validate *instance* against the leakage_profile definition.

    The instance should have the form: {"datasets": [...], "tools": {...}}.
    """
    if schema is None:
        schema = _load_schema()
    profile_schema = schema.get("definitions", {}).get("leakage_profile")
    if profile_schema is None:
        pytest.fail("leakage_profile definition not found in config schema")
    resolver = jsonschema.RefResolver.from_schema(schema)
    jsonschema.validate(instance, profile_schema, resolver=resolver)


def _validate_parameter_counts(instance: dict, schema: dict | None = None):
    if schema is None:
        schema = _load_schema()
    sub_schema = schema.get("properties", {}).get("parameter_counts")
    if sub_schema is None:
        pytest.fail("parameter_counts not found in config schema")
    resolver = jsonschema.RefResolver.from_schema(schema)
    jsonschema.validate(instance, sub_schema, resolver=resolver)


def _minimal_profile(**overrides):
    """Return a minimal valid leakage profile config, with optional overrides."""
    cfg = {
        "datasets": ["dataset_a", "dataset_b"],
        "tools": {
            "mmseqs": {"enabled": True, "min_seq_id": 0.3},
            "foldseek": {"enabled": True, "min_seq_id": 0.3},
        },
    }
    cfg.update(overrides)
    return cfg


def _minimal_config(**profile_overrides):
    """Return a minimal valid leakage_analysis config with one profile."""
    return {"profiles": {"test": _minimal_profile(**profile_overrides)}}


@pytest.mark.skipif(jsonschema is None, reason="jsonschema not installed")
class TestLeakageAnalysisSchema:
    def test_valid_config(self):
        """A well-formed leakage_analysis config passes validation."""
        _validate_leakage_analysis(_minimal_config())

    def test_valid_profile(self):
        """A well-formed profile passes validation."""
        _validate_profile(_minimal_profile())

    def test_valid_config_with_gpu(self):
        """Config with gpu options passes validation."""
        _validate_profile(
            {
                "datasets": ["secstr", "disorder"],
                "tools": {
                    "mmseqs": {"enabled": True, "min_seq_id": 0.3, "gpu": 0},
                    "foldseek": {"enabled": False, "min_seq_id": 0.3, "gpu": 1},
                },
            }
        )

    def test_multiple_profiles(self):
        """Config with multiple named profiles passes validation."""
        _validate_leakage_analysis(
            {
                "profiles": {
                    "relaxed": _minimal_profile(),
                    "strict": _minimal_profile(
                        tools={
                            "mmseqs": {"enabled": True, "min_seq_id": 0.5},
                            "foldseek": {"enabled": False, "min_seq_id": 0.5},
                        }
                    ),
                }
            }
        )

    def test_missing_profiles(self):
        """Missing 'profiles' field should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_leakage_analysis({})

    def test_empty_profiles(self):
        """Empty 'profiles' dict should fail (minProperties: 1)."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_leakage_analysis({"profiles": {}})

    def test_missing_datasets(self):
        """Missing 'datasets' field in a profile should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_missing_tools(self):
        """Missing 'tools' field in a profile should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                }
            )

    def test_missing_mmseqs_tool(self):
        """Missing 'mmseqs' key in tools should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_missing_foldseek_tool(self):
        """Missing 'foldseek' key in tools should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_missing_enabled_field(self):
        """Missing 'enabled' in a tool config should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_missing_min_seq_id(self):
        """Missing 'min_seq_id' in a tool config should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True},
                    },
                }
            )

    def test_threshold_out_of_range_high(self):
        """Threshold > 1 should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 1.5},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_threshold_out_of_range_low(self):
        """Threshold < 0 should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": -0.1},
                    },
                }
            )

    def test_too_few_datasets(self):
        """Less than 2 datasets should fail (minItems: 2)."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["only_one"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_invalid_dataset_name_uppercase(self):
        """Dataset names with uppercase should fail (pattern: ^[a-z0-9_]+$)."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["InvalidName", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_extra_properties_rejected(self):
        """Additional properties in a profile should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                    "extra_field": True,
                }
            )

    def test_extra_tool_properties_rejected(self):
        """Additional properties in a tool config should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {
                            "enabled": True,
                            "min_seq_id": 0.3,
                            "unknown_key": 42,
                        },
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_all_production_datasets(self):
        """The production config with all datasets passes."""
        _validate_profile(
            {
                "datasets": ["secstr", "disorder", "meltome", "subloc"],
                "tools": {
                    "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                    "foldseek": {"enabled": True, "min_seq_id": 0.3},
                },
            }
        )

    def test_boundary_thresholds(self):
        """Thresholds at exactly 0 and 1 should be valid."""
        _validate_profile(
            {
                "datasets": ["dataset_a", "dataset_b"],
                "tools": {
                    "mmseqs": {"enabled": True, "min_seq_id": 0},
                    "foldseek": {"enabled": True, "min_seq_id": 1},
                },
            }
        )

    def test_gpu_optional(self):
        """Config without gpu in tool config should pass (gpu is optional)."""
        _validate_profile(_minimal_profile())

    def test_gpu_valid_values(self):
        """GPU set to 0 and 1 should pass."""
        _validate_profile(
            {
                "datasets": ["dataset_a", "dataset_b"],
                "tools": {
                    "mmseqs": {"enabled": True, "min_seq_id": 0.3, "gpu": 0},
                    "foldseek": {"enabled": True, "min_seq_id": 0.3, "gpu": 1},
                },
            }
        )

    def test_gpu_invalid_value_high(self):
        """GPU value > 1 should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3, "gpu": 2},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_gpu_invalid_value_negative(self):
        """GPU value < 0 should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3, "gpu": -1},
                    },
                }
            )

    def test_gpu_invalid_type_float(self):
        """GPU value must be integer, not float."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3, "gpu": 0.5},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_foldseek_disabled(self):
        """Config with foldseek disabled passes validation."""
        _validate_profile(
            {
                "datasets": ["dataset_a", "dataset_b"],
                "tools": {
                    "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                    "foldseek": {"enabled": False, "min_seq_id": 0.3},
                },
            }
        )

    def test_both_disabled(self):
        """Config with both tools disabled passes validation (schema allows it)."""
        _validate_profile(
            {
                "datasets": ["dataset_a", "dataset_b"],
                "tools": {
                    "mmseqs": {"enabled": False, "min_seq_id": 0.3},
                    "foldseek": {"enabled": False, "min_seq_id": 0.3},
                },
            }
        )

    def test_unknown_tool_rejected(self):
        """Unknown tool names in tools should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_profile(
                {
                    "datasets": ["dataset_a", "dataset_b"],
                    "tools": {
                        "mmseqs": {"enabled": True, "min_seq_id": 0.3},
                        "foldseek": {"enabled": True, "min_seq_id": 0.3},
                        "blast": {"enabled": True, "min_seq_id": 0.3},
                    },
                }
            )

    def test_extra_leakage_analysis_properties_rejected(self):
        """Additional properties at the leakage_analysis level should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_leakage_analysis(
                {
                    "profiles": {"test": _minimal_profile()},
                    "extra_field": True,
                }
            )

    def test_invalid_profile_name_rejected(self):
        """Profile names with invalid characters should fail."""
        with pytest.raises(jsonschema.ValidationError):
            _validate_leakage_analysis(
                {
                    "profiles": {"invalid-name": _minimal_profile()},
                }
            )

    def test_parameter_counts_accepts_run_keys(self):
        _validate_parameter_counts(
            {
                "include_runs": [
                    "heads_only/esmc_300m_all_heads_only",
                    "align_lora/esmc_300m_all_r4",
                ]
            }
        )

    def test_parameter_counts_rejects_bad_run_key_format(self):
        with pytest.raises(jsonschema.ValidationError):
            _validate_parameter_counts({"include_runs": ["esmc_300m_all_r4"]})
