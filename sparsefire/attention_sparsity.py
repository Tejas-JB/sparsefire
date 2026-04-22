"""Phase 4 — post-softmax top-k attention sparsity with attention-sink preservation.

Zeroes all but top-k% of attention scores after softmax, renormalizes,
and preserves the first token (attention sink). Measures energy impact.
"""

from __future__ import annotations

import logging

from ._runner import (
    assemble_result,
    load_model_and_tokenizer,
    measure_energy,
    tokenize_prompts,
    validate_and_write,
)
from .config import Config
from .hooks import sparse_attention
from .prompts import load_prompts

logger = logging.getLogger(__name__)


def run(cfg: Config, top_k_frac: float = 0.5, stack_phases: list[str] | None = None) -> dict:
    """Run attention sparsity at the given top-k fraction.

    top_k_frac: fraction of attention scores to keep (e.g. 0.5 = keep top 50%).
    Lower = more aggressive sparsity.
    """
    model, tokenizer = load_model_and_tokenizer(cfg)
    prompts = load_prompts(n_prompts=cfg.n_prompts, seed=cfg.seed, split=cfg.wikitext_split)
    prompt_inputs = tokenize_prompts(cfg, tokenizer, prompts)

    keep_pct = int(top_k_frac * 100)
    phase_name = f"phase4_attn_topk{keep_pct}"
    phase_label = f"AttnSparse-top{keep_pct}%"

    logger.info(
        "Attention sparsity: keeping top %d%% of scores (first token always preserved)",
        keep_pct,
    )

    energy = measure_energy(
        cfg,
        model,
        prompt_inputs,
        hook_ctx=sparse_attention(top_k_frac=top_k_frac, preserve_first_token=True),
        phase_label=phase_label,
    )

    sparsity_info = {
        "target_mlp": None,
        "achieved_mlp_mean": None,
        "achieved_mlp_per_layer": None,
        "target_attn_top_k_frac": top_k_frac,
        "attention_sink_preserved": True,
        "quantization": None,
    }

    result = assemble_result(cfg, phase_name, energy, {}, sparsity=sparsity_info)
    validate_and_write(result, cfg, f"{phase_name}.json")
    return result
