"""Phase 1 — KV cache A/B (use_cache True vs False)."""

from __future__ import annotations

from .config import Config


def run(cfg: Config, use_cache: bool = True) -> dict:
    raise NotImplementedError("Phase 2 implementer: fill this in.")
