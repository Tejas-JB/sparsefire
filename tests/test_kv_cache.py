"""Unit tests for sparsefire.kv_cache. All heavy deps mocked."""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock, patch

import pytest
import torch

from sparsefire.config import Config


class FakeNvml:
    def __init__(self):
        self._calls = 0

    def nvmlInit(self): ...
    def nvmlDeviceGetHandleByIndex(self, i):
        return "handle"

    def nvmlDeviceGetTotalEnergyConsumption(self, h):
        self._calls += 1
        return 1_000_000 + (self._calls - 1) * 1000

    def nvmlDeviceGetPowerUsage(self, h):
        return 250_000


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
def test_cfg(tmp_path):
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
    tok = MagicMock()
    tok.pad_token = None
    tok.eos_token = "<eos>"
    fake_encoded = MagicMock()
    fake_encoded.to.return_value = {"input_ids": torch.zeros(1, 8, dtype=torch.long)}
    tok.return_value = fake_encoded
    return tok


def _make_mock_model():
    model = MagicMock()
    model.device = "cpu"
    model.eval = MagicMock()
    fake_output = torch.zeros(1, 8 + 16, dtype=torch.long)
    model.generate = MagicMock(return_value=fake_output)
    fake_fwd = MagicMock()
    fake_fwd.loss = torch.tensor(2.0)
    model.__call__ = MagicMock(return_value=fake_fwd)
    model.return_value = fake_fwd
    return model


def test_kv_cache_run_use_cache_true(fake_nvml, test_cfg):
    """kv_cache.run(use_cache=True) produces valid result."""
    mock_tokenizer = _make_mock_tokenizer()
    mock_model = _make_mock_model()

    with (
        patch("transformers.AutoModelForCausalLM") as mock_auto_model,
        patch("transformers.AutoTokenizer") as mock_auto_tok,
        patch("sparsefire.prompts.load_prompts", return_value=["Hello world", "Test prompt"]),
        patch("sparsefire.evaluate.perplexity_wikitext2", return_value=11.5),
        patch("sparsefire.evaluate.hellaswag_0shot", return_value={"acc": 0.45, "acc_norm": 0.60}),
    ):
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        from sparsefire.kv_cache import run

        result = run(test_cfg, use_cache=True)

    assert result["phase"] == "phase1_kvcache_cache_on"
    assert result["energy"]["joules_per_token"]["n"] == test_cfg.n_runs

    from sparsefire.schema import validate

    validate(result)


def test_kv_cache_run_use_cache_false(fake_nvml, test_cfg):
    """kv_cache.run(use_cache=False) produces valid result with correct phase name."""
    mock_tokenizer = _make_mock_tokenizer()
    mock_model = _make_mock_model()

    with (
        patch("transformers.AutoModelForCausalLM") as mock_auto_model,
        patch("transformers.AutoTokenizer") as mock_auto_tok,
        patch("sparsefire.prompts.load_prompts", return_value=["Hello world", "Test prompt"]),
        patch("sparsefire.evaluate.perplexity_wikitext2", return_value=11.5),
        patch("sparsefire.evaluate.hellaswag_0shot", return_value={"acc": 0.45, "acc_norm": 0.60}),
    ):
        mock_auto_tok.from_pretrained.return_value = mock_tokenizer
        mock_auto_model.from_pretrained.return_value = mock_model

        from sparsefire.kv_cache import run

        result = run(test_cfg, use_cache=False)

    assert result["phase"] == "phase1_kvcache_cache_off"
    assert result["accuracy"]["perplexity_wikitext2"] == 11.5

    from sparsefire.schema import validate

    validate(result)
