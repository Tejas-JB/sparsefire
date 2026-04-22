"""Phase 6 — waterfall attribution chart and neuron-firing animation."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def make_waterfall(results_dir: Path, out_path: Path | None = None) -> Path:
    """Load all results/phase*.json, build waterfall chart with brain-equivalent bar."""
    raise NotImplementedError


def make_neuron_firing_video(
    activations_by_sparsity: dict[float, np.ndarray],
    out_path: Path,
    grid_shape: tuple[int, int] = (32, 32),
    fps: int = 30,
    duration_s: int = 30,
) -> Path:
    raise NotImplementedError
