"""Phase 0 — dense fp16 baseline. Eager attention, use_cache=True."""

from __future__ import annotations

from .config import Config


def run(cfg: Config) -> dict:
    raise NotImplementedError(
        "Phase 1 implementer: fill this in. See docs/architecture.md §phases."
    )
