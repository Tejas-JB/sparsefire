"""Phase 3 — AutoAWQ INT4 quantization, loaded with do_fuse=False.

Quantizes Llama-3.2-1B-Instruct to INT4 AWQ (group_size=128), saves locally,
then measures energy. Optionally stacks activation sparsity on top.
"""

from __future__ import annotations

import logging
from pathlib import Path

from ._runner import (
    assemble_result,
    measure_energy,
    tokenize_prompts,
    validate_and_write,
)
from .config import Config
from .prompts import load_prompts

logger = logging.getLogger(__name__)

_QUANT_DIR = Path("quantized/llama-3.2-1b-awq")


def quantize_model(
    source_model_id: str, output_dir: Path = _QUANT_DIR, group_size: int = 128, w_bit: int = 4
) -> Path:
    """Run AutoAWQ quantization; save to output_dir; return output_dir."""
    from awq import AutoAWQForCausalLM
    from transformers import AutoTokenizer

    output_dir = Path(output_dir)
    if (output_dir / "config.json").exists():
        logger.info("Quantized model already exists at %s, skipping", output_dir)
        return output_dir

    logger.info("Quantizing %s to INT%d (group_size=%d)...", source_model_id, w_bit, group_size)
    tokenizer = AutoTokenizer.from_pretrained(source_model_id)
    model = AutoAWQForCausalLM.from_pretrained(source_model_id)

    quant_config = {
        "zero_point": True,
        "q_group_size": group_size,
        "w_bit": w_bit,
        "version": "GEMM",
    }
    model.quantize(tokenizer, quant_config=quant_config)

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    logger.info("Quantized model saved to %s", output_dir)
    return output_dir


def load_quantized_model(quant_dir: Path = _QUANT_DIR, attn_impl: str = "eager"):
    """Load quantized model with do_fuse=False for hook compatibility."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

    logger.info("Loading quantized model from %s", quant_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = AwqConfig(bits=4, do_fuse=False)
    model = AutoModelForCausalLM.from_pretrained(
        str(quant_dir),
        quantization_config=quant_config,
        attn_implementation=attn_impl,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer


def run(cfg: Config, stack_sparsity: float | None = None) -> dict:
    """Run quantization phase. Optionally stack activation sparsity on top."""
    # Step 1: Quantize (or load cached)
    quant_dir = quantize_model(cfg.model_id)

    # Step 2: Load quantized model
    model, tokenizer = load_quantized_model(quant_dir, attn_impl=cfg.attn_impl)

    # Step 3: Prompts
    prompts = load_prompts(n_prompts=cfg.n_prompts, seed=cfg.seed, split=cfg.wikitext_split)
    prompt_inputs = tokenize_prompts(cfg, tokenizer, prompts)

    # Step 4: Optionally calibrate sparsity hooks
    hook_ctx = None
    if stack_sparsity is not None:
        from .activation_sparsity import calibrate_thresholds
        from .hooks import sparse_mlp_hooks

        thresholds = calibrate_thresholds(
            model, tokenizer, stack_sparsity, n_samples=64, seq_len=256
        )
        hook_ctx = sparse_mlp_hooks(model, thresholds)

    # Step 5: Measure energy
    suffix = f"_sparse{int(stack_sparsity * 100)}" if stack_sparsity else ""
    phase_label = (
        f"Quant-INT4{'+sparse' + str(int(stack_sparsity * 100)) + '%' if stack_sparsity else ''}"
    )
    phase_name = f"phase3_quant{suffix}"

    energy = measure_energy(
        cfg,
        model,
        prompt_inputs,
        hook_ctx=hook_ctx,
        phase_label=phase_label,
    )

    sparsity_info = {
        "target_mlp": stack_sparsity,
        "achieved_mlp_mean": None,
        "achieved_mlp_per_layer": None,
        "target_attn_top_k_frac": None,
        "attention_sink_preserved": None,
        "quantization": {"method": "awq", "bits": 4, "group_size": 128},
    }

    result = assemble_result(cfg, phase_name, energy, {}, sparsity=sparsity_info)
    validate_and_write(result, cfg, f"{phase_name}.json")
    return result
