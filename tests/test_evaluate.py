"""Unit tests for sparsefire.evaluate — verifies hellaswag_0shot uses actual model objects."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestHellaswag0shot:
    """Verify hellaswag_0shot accepts model+tokenizer and wraps them with HFLM."""

    def _make_fake_results(self, acc=0.45, acc_norm=0.60):
        return {
            "results": {
                "hellaswag": {
                    "acc,none": acc,
                    "acc_norm,none": acc_norm,
                }
            }
        }

    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.models.huggingface.HFLM")
    def test_receives_model_object_not_string(self, mock_hflm_cls, mock_simple_eval):
        """hellaswag_0shot should accept a model object, not a string path."""
        from sparsefire.evaluate import hellaswag_0shot

        model = MagicMock()
        tokenizer = MagicMock()
        mock_simple_eval.return_value = self._make_fake_results()

        hellaswag_0shot(model, tokenizer)

        # The model passed to HFLM should be the object, not a string
        call_kwargs = mock_hflm_cls.call_args
        assert call_kwargs is not None, "HFLM was never instantiated"
        assert call_kwargs.kwargs.get("pretrained") is model or call_kwargs[1].get("pretrained") is model

    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.models.huggingface.HFLM")
    def test_hflm_wrapper_created_with_model_and_tokenizer(self, mock_hflm_cls, mock_simple_eval):
        """HFLM should be created with both the model and tokenizer."""
        from sparsefire.evaluate import hellaswag_0shot

        model = MagicMock()
        tokenizer = MagicMock()
        mock_simple_eval.return_value = self._make_fake_results()

        hellaswag_0shot(model, tokenizer)

        mock_hflm_cls.assert_called_once()
        call_kwargs = mock_hflm_cls.call_args.kwargs
        assert call_kwargs["pretrained"] is model
        assert call_kwargs["tokenizer"] is tokenizer

    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.models.huggingface.HFLM")
    def test_simple_evaluate_receives_hflm_instance(self, mock_hflm_cls, mock_simple_eval):
        """simple_evaluate should receive the HFLM wrapper, not a string."""
        from sparsefire.evaluate import hellaswag_0shot

        model = MagicMock()
        tokenizer = MagicMock()
        fake_lm = MagicMock()
        mock_hflm_cls.return_value = fake_lm
        mock_simple_eval.return_value = self._make_fake_results()

        hellaswag_0shot(model, tokenizer)

        mock_simple_eval.assert_called_once()
        call_kwargs = mock_simple_eval.call_args.kwargs
        assert call_kwargs["model"] is fake_lm

    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.models.huggingface.HFLM")
    def test_result_contains_acc_and_acc_norm_as_floats(self, mock_hflm_cls, mock_simple_eval):
        """Return dict should have acc and acc_norm as floats."""
        from sparsefire.evaluate import hellaswag_0shot

        model = MagicMock()
        tokenizer = MagicMock()
        mock_simple_eval.return_value = self._make_fake_results(acc=0.42, acc_norm=0.58)

        result = hellaswag_0shot(model, tokenizer)

        assert "acc" in result
        assert "acc_norm" in result
        assert isinstance(result["acc"], float)
        assert isinstance(result["acc_norm"], float)
        assert result["acc"] == pytest.approx(0.42)
        assert result["acc_norm"] == pytest.approx(0.58)

    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.models.huggingface.HFLM")
    def test_batch_size_and_device_forwarded(self, mock_hflm_cls, mock_simple_eval):
        """batch_size and device should be forwarded to HFLM."""
        from sparsefire.evaluate import hellaswag_0shot

        model = MagicMock()
        tokenizer = MagicMock()
        mock_simple_eval.return_value = self._make_fake_results()

        hellaswag_0shot(model, tokenizer, batch_size=16, device="cuda:1")

        call_kwargs = mock_hflm_cls.call_args.kwargs
        assert call_kwargs["batch_size"] == 16
        assert call_kwargs["device"] == "cuda:1"
