"""Phase 2 — TEAL-style activation sparsity at down_proj input.

Calibrates per-layer thresholds on WikiText-2 calibration texts, then measures
energy with magnitude-thresholded MLP activations at the specified sparsity level.
"""

from __future__ import annotations

import logging

import torch

from ._runner import (
    assemble_result,
    load_model_and_tokenizer,
    measure_energy,
    run_accuracy,
    tokenize_prompts,
    validate_and_write,
)
from .config import Config
from .hooks import sparse_mlp_hooks
from .prompts import load_prompts

logger = logging.getLogger(__name__)


def calibrate_thresholds(
    model,
    tokenizer,
    target_sparsity: float,
    n_samples: int = 512,
    seq_len: int = 2048,
) -> dict[int, float]:
    """Compute per-layer magnitude thresholds for down_proj input.

    Uses reservoir sampling to collect a fixed-size sample of activation
    magnitudes per layer, then computes the threshold via torch.quantile.
    Memory-efficient: keeps at most reservoir_size values per layer.
    """
    import gc

    from datasets import load_dataset

    logger.info(
        "Calibrating thresholds for %.0f%% sparsity on %d samples...",
        target_sparsity * 100,
        n_samples,
    )

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    texts = [x["text"] for x in ds if len(x["text"].split()) >= 16][:n_samples]

    n_layers = len(model.model.layers)
    # Keep a capped reservoir of magnitudes per layer (memory-bounded)
    reservoir_size = 500_000  # 500K values per layer ≈ 2MB per layer
    reservoirs: dict[int, list[torch.Tensor]] = {i: [] for i in range(n_layers)}
    counts: dict[int, int] = {i: 0 for i in range(n_layers)}
    handles = []

    def make_capture_hook(layer_idx, _reservoirs=reservoirs, _counts=counts):
        def hook(module, args):
            x = args[0]
            mags = x.abs().detach().cpu().flatten()
            n = mags.shape[0]
            _counts[layer_idx] += n
            if _counts[layer_idx] <= reservoir_size:
                _reservoirs[layer_idx].append(mags)
            else:
                keep = max(1, int(n * reservoir_size / _counts[layer_idx]))
                idx = torch.randperm(n)[:keep]
                _reservoirs[layer_idx].append(mags[idx])

        return hook

    for i, layer in enumerate(model.model.layers):
        h = layer.mlp.down_proj.register_forward_pre_hook(make_capture_hook(i))
        handles.append(h)

    try:
        for i, text in enumerate(texts):
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len)
            ids = ids.to(model.device)
            with torch.no_grad():
                model(**ids)
            if (i + 1) % 50 == 0:
                logger.info("  calibration %d/%d", i + 1, len(texts))
    finally:
        for h in handles:
            h.remove()

    # Compute per-layer thresholds using quantile (much faster than full sort)
    thresholds = {}
    for layer_idx in range(n_layers):
        all_mags = torch.cat(reservoirs[layer_idx])
        thresholds[layer_idx] = torch.quantile(all_mags.float(), target_sparsity).item()
        del all_mags
    del reservoirs
    gc.collect()

    logger.info("Calibrated thresholds: %s", {k: f"{v:.4f}" for k, v in thresholds.items()})
    return thresholds


def _measure_achieved_sparsity(
    model, tokenizer, thresholds: dict[int, float], n_check: int = 10
) -> tuple[float, list[float]]:
    """Verify achieved sparsity on a few samples. Returns (mean, per_layer)."""
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [x["text"] for x in ds if len(x["text"].split()) >= 16][:n_check]

    n_layers = len(model.model.layers)
    zero_counts = {i: 0 for i in range(n_layers)}
    total_counts = {i: 0 for i in range(n_layers)}
    handles = []

    def make_count_hook(layer_idx, threshold):
        def hook(module, args):
            x = args[0]
            mask = x.abs() <= threshold
            zero_counts[layer_idx] += mask.sum().item()
            total_counts[layer_idx] += x.numel()

        return hook

    for i, layer in enumerate(model.model.layers):
        h = layer.mlp.down_proj.register_forward_pre_hook(make_count_hook(i, thresholds[i]))
        handles.append(h)

    try:
        for text in texts:
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            ids = ids.to(model.device)
            with torch.no_grad():
                model(**ids)
    finally:
        for h in handles:
            h.remove()

    per_layer = []
    for i in range(n_layers):
        sp = zero_counts[i] / total_counts[i] if total_counts[i] > 0 else 0.0
        per_layer.append(sp)

    mean_sp = sum(per_layer) / len(per_layer)
    return mean_sp, per_layer


def run(cfg: Config, sparsity: float = 0.40) -> dict:
    """Run activation sparsity phase at the given sparsity level."""
    model, tokenizer = load_model_and_tokenizer(cfg)
    prompts = load_prompts(n_prompts=cfg.n_prompts, seed=cfg.seed, split=cfg.wikitext_split)
    prompt_inputs = tokenize_prompts(cfg, tokenizer, prompts)

    # Calibrate thresholds
    thresholds = calibrate_thresholds(model, tokenizer, sparsity)

    # Verify achieved sparsity
    achieved_mean, achieved_per_layer = _measure_achieved_sparsity(model, tokenizer, thresholds)
    logger.info(
        "Target sparsity: %.1f%%, achieved: %.1f%%",
        sparsity * 100,
        achieved_mean * 100,
    )

    # Measure energy with sparsity hooks active
    phase_label = f"ActSparsity-{int(sparsity * 100)}%"
    phase_name = f"phase2_actsparse_{int(sparsity * 100)}"
    energy = measure_energy(
        cfg,
        model,
        prompt_inputs,
        hook_ctx=sparse_mlp_hooks(model, thresholds),
        phase_label=phase_label,
    )

    accuracy = run_accuracy(cfg, model, tokenizer)

    sparsity_info = {
        "target_mlp": sparsity,
        "achieved_mlp_mean": achieved_mean,
        "achieved_mlp_per_layer": achieved_per_layer,
        "target_attn_top_k_frac": None,
        "attention_sink_preserved": None,
        "quantization": None,
    }

    result = assemble_result(cfg, phase_name, energy, accuracy, sparsity=sparsity_info)
    validate_and_write(result, cfg, f"{phase_name}.json")
    return result
