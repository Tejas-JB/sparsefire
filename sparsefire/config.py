"""Run-level configuration. See docs/architecture.md §Config."""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path


@dataclass(frozen=True)
class Config:
    # Model
    model_id: str = "meta-llama/Llama-3.2-1B-Instruct"
    dtype: str = "float16"
    attn_impl: str = "eager"
    device: str = "cuda"
    seed: int = 0

    # Measurement protocol
    n_runs: int = 50
    n_tokens: int = 256
    n_prompts: int = 50
    warmup_s: int = 60
    sample_interval_ms: int = 50
    lock_clocks: bool = True
    gpu_lock_freq_mhz: int = 1395  # RTX 3090 base; overridden per-device if needed

    # Eval
    wikitext_split: str = "wikitext-2-raw-v1"
    hellaswag_batch_size: int = 8

    # Paths
    results_dir: Path = Path("results")
    hf_cache_dir: Path | None = None

    def override(self, **kwargs: object) -> Config:
        return replace(self, **kwargs)  # type: ignore[arg-type]
