"""Phase 1 — KV cache A/B (use_cache True vs False).

Measures the energy delta introduced by KV caching. Runs the same prompts
with use_cache=True (default, already baseline) and use_cache=False to
isolate the caching contribution.
"""

from __future__ import annotations

import logging

from ._runner import (
    assemble_result,
    load_model_and_tokenizer,
    measure_energy,
    run_accuracy,
    tokenize_prompts,
    validate_and_write,
)
from .config import Config
from .prompts import load_prompts

logger = logging.getLogger(__name__)


def run(cfg: Config, use_cache: bool = True) -> dict:
    """Run KV cache phase. Call twice: once with use_cache=True, once False."""
    model, tokenizer = load_model_and_tokenizer(cfg)
    prompts = load_prompts(n_prompts=cfg.n_prompts, seed=cfg.seed, split=cfg.wikitext_split)
    prompt_inputs = tokenize_prompts(cfg, tokenizer, prompts)

    cache_label = "cache_on" if use_cache else "cache_off"
    phase_name = f"phase1_kvcache_{cache_label}"

    energy = measure_energy(
        cfg,
        model,
        prompt_inputs,
        use_cache=use_cache,
        phase_label=f"KV-cache ({cache_label})",
    )

    accuracy = run_accuracy(cfg, model, tokenizer)

    result = assemble_result(cfg, phase_name, energy, accuracy)
    validate_and_write(result, cfg, f"{phase_name}.json")
    return result
