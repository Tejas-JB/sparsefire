"""Unit tests for sparsefire.quantization — fused AWQ kernel toggle."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def _mock_transformers():
    """Patch transformers imports used by load_quantized_model."""
    with (
        patch("transformers.AutoModelForCausalLM") as mock_model_cls,
        patch("transformers.AutoTokenizer") as mock_tok_cls,
        patch("transformers.AwqConfig") as mock_awq_cfg,
    ):
        import torch

        tok = MagicMock()
        tok.pad_token = None
        tok.eos_token = "<eos>"
        mock_tok_cls.from_pretrained.return_value = tok

        model = MagicMock()
        mock_model_cls.from_pretrained.return_value = model

        yield {
            "AutoModelForCausalLM": mock_model_cls,
            "AutoTokenizer": mock_tok_cls,
            "AwqConfig": mock_awq_cfg,
            "model": model,
            "tokenizer": tok,
        }


def test_load_quantized_fused_uses_do_fuse_true(_mock_transformers, tmp_path):
    """When fused=True, AwqConfig should be called with do_fuse=True."""
    from sparsefire.quantization import load_quantized_model

    load_quantized_model(tmp_path, attn_impl="eager", fused=True)

    _mock_transformers["AwqConfig"].assert_called_once_with(
        bits=4, do_fuse=True, fuse_max_seq_len=512
    )


def test_load_quantized_unfused_uses_do_fuse_false(_mock_transformers, tmp_path):
    """When fused=False, AwqConfig should be called with do_fuse=False."""
    from sparsefire.quantization import load_quantized_model

    load_quantized_model(tmp_path, attn_impl="eager", fused=False)

    _mock_transformers["AwqConfig"].assert_called_once_with(bits=4, do_fuse=False)


def test_load_quantized_default_is_fused(_mock_transformers, tmp_path):
    """Default value of fused should be True."""
    from sparsefire.quantization import load_quantized_model

    load_quantized_model(tmp_path)

    _mock_transformers["AwqConfig"].assert_called_once_with(
        bits=4, do_fuse=True, fuse_max_seq_len=512
    )


def test_run_standalone_uses_fused(tmp_path):
    """run() without stack_sparsity should load model with fused=True."""
    from sparsefire.config import Config

    cfg = Config(
        n_runs=1,
        n_tokens=16,
        n_prompts=1,
        warmup_s=0,
        lock_clocks=False,
        device="cpu",
        results_dir=tmp_path / "results",
    )

    with (
        patch("sparsefire.quantization.quantize_model", return_value=tmp_path),
        patch("sparsefire.quantization.load_quantized_model") as mock_load,
        patch("sparsefire.quantization.load_prompts", return_value=["Hello"]),
        patch("sparsefire.quantization.tokenize_prompts", return_value=[{}]),
        patch("sparsefire.quantization.measure_energy", return_value={
            "joules_per_token": {"mean": 0.01, "ci_low": 0.009, "ci_high": 0.011, "n": 1},
            "total_energy_j": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.1, "n": 1},
            "wallclock_s": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6, "n": 1},
            "mean_power_w": {"mean": 100, "ci_low": 90, "ci_high": 110, "n": 1},
            "peak_power_w": 120,
            "tokens_per_second": {"mean": 32, "ci_low": 28, "ci_high": 36, "n": 1},
        }),
        patch("sparsefire.quantization.run_accuracy", return_value={
            "perplexity_wikitext2": 15.0,
            "hellaswag_acc": 0.45,
            "hellaswag_acc_norm": 0.60,
        }),
        patch("sparsefire.quantization.validate_and_write"),
    ):
        mock_load.return_value = (MagicMock(), MagicMock())

        from sparsefire.quantization import run

        run(cfg)

        # Standalone (no stack_sparsity) should use fused=True
        mock_load.assert_called_once_with(tmp_path, attn_impl=cfg.attn_impl, fused=True)


def test_run_stacked_uses_unfused(tmp_path):
    """run() with stack_sparsity should load model with fused=False."""
    from sparsefire.config import Config

    cfg = Config(
        n_runs=1,
        n_tokens=16,
        n_prompts=1,
        warmup_s=0,
        lock_clocks=False,
        device="cpu",
        results_dir=tmp_path / "results",
    )

    with (
        patch("sparsefire.quantization.quantize_model", return_value=tmp_path),
        patch("sparsefire.quantization.load_quantized_model") as mock_load,
        patch("sparsefire.quantization.load_prompts", return_value=["Hello"]),
        patch("sparsefire.quantization.tokenize_prompts", return_value=[{}]),
        patch("sparsefire.quantization.measure_energy", return_value={
            "joules_per_token": {"mean": 0.01, "ci_low": 0.009, "ci_high": 0.011, "n": 1},
            "total_energy_j": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.1, "n": 1},
            "wallclock_s": {"mean": 0.5, "ci_low": 0.4, "ci_high": 0.6, "n": 1},
            "mean_power_w": {"mean": 100, "ci_low": 90, "ci_high": 110, "n": 1},
            "peak_power_w": 120,
            "tokens_per_second": {"mean": 32, "ci_low": 28, "ci_high": 36, "n": 1},
        }),
        patch("sparsefire.quantization.run_accuracy", return_value={
            "perplexity_wikitext2": 15.0,
            "hellaswag_acc": 0.45,
            "hellaswag_acc_norm": 0.60,
        }),
        patch("sparsefire.quantization.validate_and_write"),
        patch("sparsefire.activation_sparsity.calibrate_thresholds", return_value={}),
        patch("sparsefire.hooks.sparse_mlp_hooks"),
    ):
        mock_load.return_value = (MagicMock(), MagicMock())

        from sparsefire.quantization import run

        run(cfg, stack_sparsity=0.5)

        # Stacked (with hooks) should use fused=False
        mock_load.assert_called_once_with(tmp_path, attn_impl=cfg.attn_impl, fused=False)
