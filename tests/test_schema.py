"""Validate docs/results_schema.json and the package's loader."""

from __future__ import annotations

import pytest

from sparsefire.schema import load_schema, validate


def _valid_record() -> dict:
    return {
        "phase": "phase0_baseline",
        "model_id": "meta-llama/Llama-3.2-1B-Instruct",
        "timestamp_utc": "2026-04-21T17:30:00Z",
        "config": {
            "dtype": "float16",
            "attn_impl": "eager",
            "n_runs": 50,
            "n_tokens": 256,
            "seed": 0,
        },
        "energy": {
            "joules_per_token": {"mean": 12.3, "ci_low": 12.0, "ci_high": 12.6, "n": 50},
            "total_energy_j": {"mean": 3148.8, "ci_low": 3072.0, "ci_high": 3225.6, "n": 50},
            "wallclock_s": {"mean": 8.5, "ci_low": 8.4, "ci_high": 8.6, "n": 50},
            "mean_power_w": {"mean": 370.5, "ci_low": 366.0, "ci_high": 375.0, "n": 50},
        },
        "accuracy": {
            "perplexity_wikitext2": 9.87,
            "hellaswag_acc": 0.451,
        },
        "metadata": {
            "git_sha": "abc1234",
            "python_version": "3.11.9",
            "torch_version": "2.5.1",
            "gpu_name": "NVIDIA GeForce RTX 3090",
        },
    }


def test_schema_loads():
    schema = load_schema()
    assert schema["$schema"].startswith("http://json-schema.org/")
    assert "Energy" in schema["definitions"]


def test_valid_record_passes():
    validate(_valid_record())


def test_missing_phase_field_fails():
    import jsonschema

    r = _valid_record()
    del r["phase"]
    with pytest.raises(jsonschema.ValidationError):
        validate(r)


def test_bad_attn_impl_fails():
    import jsonschema

    r = _valid_record()
    r["config"]["attn_impl"] = "flash"
    with pytest.raises(jsonschema.ValidationError):
        validate(r)


def test_hellaswag_acc_out_of_range_fails():
    import jsonschema

    r = _valid_record()
    r["accuracy"]["hellaswag_acc"] = 1.5
    with pytest.raises(jsonschema.ValidationError):
        validate(r)
