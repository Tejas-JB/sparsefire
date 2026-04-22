"""Phase 2 — TEAL-style activation sparsity at down_proj input.

Calibrates per-layer thresholds on n_samples WikiText-2 texts, then measures.
"""

from __future__ import annotations

from .config import Config


def calibrate_thresholds(
    model, tokenizer, target_sparsity: float, n_samples: int = 512, seq_len: int = 2048
) -> dict[int, float]:
    """Return {layer_idx: threshold} where threshold is the target_sparsity-th
    percentile of |down_proj input| across n_samples forward passes."""
    raise NotImplementedError


def run(cfg: Config, sparsity: float = 0.40) -> dict:
    raise NotImplementedError("Phase 3 implementer: fill this in.")
