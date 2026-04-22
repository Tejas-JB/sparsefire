"""Phase 3 — AutoAWQ INT4 quantization, loaded with do_fuse=False.

Optionally stacks activation sparsity on top (the compound experiment).
"""

from __future__ import annotations

from pathlib import Path

from .config import Config


def quantize_model(
    source_model_id: str, output_dir: Path, group_size: int = 128, w_bit: int = 4
) -> Path:
    """Run AutoAWQ quantization; save to output_dir; return output_dir."""
    raise NotImplementedError


def run(cfg: Config, stack_sparsity: float | None = None) -> dict:
    raise NotImplementedError("Phase 4 implementer: fill this in.")
