"""Phase 4 — post-softmax top-k attention sparsity with attention-sink preservation.

Implemented via F.softmax monkeypatch (see hooks.sparse_attention).
Requires attn_implementation='eager' at model load.
"""

from __future__ import annotations

from .config import Config


def run(cfg: Config, top_k_frac: float = 0.5, stack_phases: list[str] | None = None) -> dict:
    raise NotImplementedError("Phase 5 implementer: fill this in.")
