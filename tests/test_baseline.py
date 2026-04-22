"""Unit tests for sparsefire.baseline. All heavy deps (model, nvml, datasets, lm-eval) are mocked."""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from sparsefire.config import Config


class FakeNvml:
    """Minimal pynvml mock for EnergyMeter."""

    def __init__(self):
        self._calls = 0

    def nvmlInit(self): ...

    def nvmlDeviceGetHandleByIndex(self, i):
        return "handle"

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self._calls += 1
        # Return increasing energy: 1000 mJ per call pair → 1 J delta
        return 1_000_000 + (self._calls - 1) * 1000

    def nvmlDeviceGetPowerUsage(self, h):
        return 250_000  # 250 W


@pytest.fixture
def fake_nvml(monkeypatch):
    mod = types.ModuleType("pynvml")
    fake = FakeNvml()
    for attr in dir(fake):
        if attr.startswith("nvml"):
            setattr(mod, attr, getattr(fake, attr))
    monkeypatch.setitem(sys.modules, "pynvml", mod)
    return fake


@pytest.fixture
def mock_model():
    """A fake model that pretends to generate tokens."""
    import torch

    model = MagicMock()
    model.device = "cpu"
    model.eval = MagicMock()
    # generate returns tensor of shape (1, prompt_len + n_generated)
    # prompt_len=8 (from mock tokenizer), n_generated=16 (from test_cfg.n_tokens)
    fake_output = torch.zeros(1, 8 + 16, dtype=torch.long)
    model.generate = MagicMock(return_value=fake_output)
    # For perplexity: model(ids, labels=target) returns loss
    fake_fwd = MagicMock()
    fake_fwd.loss = torch.tensor(2.0)
    model.__call__ = MagicMock(return_value=fake_fwd)
    model.return_value = fake_fwd
    return model


@pytest.fixture
def test_cfg(tmp_path):
    """Minimal config for testing — tiny runs."""
    return Config(
        n_runs=3,
        n_tokens=16,
        n_prompts=2,
        warmup_s=0,
        lock_clocks=False,
        device="cpu",
        results_dir=tmp_path / "results",
    )


def _make_mock_tokenizer():
    """Create a mock tokenizer that returns proper tensor-like objects."""
    import torch

    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"

    # tokenizer(text, ...) returns an object with .to() that returns dict-like with input_ids
    fake_encoded = MagicMock()
    fake_encoded.to.return_value = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
    tok.return_value = fake_encoded
    return tok


def _run_baseline_with_mocks(fake_nvml, mock_model, test_cfg, ppl=15.5, acc=0.45, acc_norm=0.60):
    """Helper: run baseline.run() with all heavy deps mocked."""
    mock_tokenizer = _make_mock_tokenizer()

    with (
        patch("transformers.AutoModelForCausalLM") as mock_auto_model,
        patch("transformers.AutoTokenizer") as mock_auto_tok,
        patch("sparsefire.prompts.load_prompts", return_value=["Hello world", "Test prompt"]),
        patch("sparsefire.evaluate.perplexity_wikitext2", return_value=ppl),
        patch(
            "sparsefire.evaluate.hellaswag_0shot", return_value={"acc": acc, "acc_norm": acc_norm}
        ),
    ):
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        from sparsefire.baseline import run

        return run(test_cfg)


def test_baseline_run_produces_valid_json(fake_nvml, mock_model, test_cfg, monkeypatch):
    """baseline.run() should produce a dict that validates against the results schema."""
    result = _run_baseline_with_mocks(fake_nvml, mock_model, test_cfg)

    # Validate structure
    assert result["phase"] == "phase0_baseline"
    assert result["model_id"] == test_cfg.model_id
    assert "energy" in result
    assert "accuracy" in result
    assert "metadata" in result

    # Energy stats should have CI fields
    jpt = result["energy"]["joules_per_token"]
    assert "mean" in jpt and "ci_low" in jpt and "ci_high" in jpt
    assert jpt["n"] == test_cfg.n_runs

    # Accuracy values should match mocks
    assert result["accuracy"]["perplexity_wikitext2"] == 15.5
    assert result["accuracy"]["hellaswag_acc"] == 0.45
    assert result["accuracy"]["hellaswag_acc_norm"] == 0.60

    # Sparsity fields should be null for baseline
    assert result["sparsity"]["target_mlp"] is None
    assert result["sparsity"]["quantization"] is None

    # Results file should be written
    out_path = Path(test_cfg.results_dir) / "phase0_baseline.json"
    assert out_path.exists()
    written = json.loads(out_path.read_text())
    assert written["phase"] == "phase0_baseline"


def test_baseline_run_validates_against_schema(fake_nvml, mock_model, test_cfg, monkeypatch):
    """The result dict should pass jsonschema validation."""
    result = _run_baseline_with_mocks(
        fake_nvml, mock_model, test_cfg, ppl=12.0, acc=0.42, acc_norm=0.58
    )

    # This should not raise
    from sparsefire.schema import validate

    validate(result)


def test_cli_phase0_dispatches(monkeypatch):
    """--phase 0 should dispatch to baseline.run without raising NotImplementedError."""
    from sparsefire import cli

    called = {}

    def fake_run(cfg):
        called["cfg"] = cfg
        return {"phase": "phase0_baseline"}

    monkeypatch.setattr("sparsefire.baseline.run", fake_run)
    # Should not raise NotImplementedError
    cli.main(["--phase", "0", "--n-runs", "1"])
    assert "cfg" in called
