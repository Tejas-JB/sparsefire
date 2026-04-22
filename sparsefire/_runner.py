"""Shared measurement runner used by all phase modules.

Extracts the common pattern: load model, tokenize prompts, warmup, run N
energy-metered generations, compute bootstrap CIs, run accuracy evals,
assemble + validate + write the results JSON.
"""

from __future__ import annotations

import json
import logging
import platform
import subprocess
from contextlib import nullcontext
from datetime import UTC, datetime
from pathlib import Path

from .config import Config

logger = logging.getLogger(__name__)


def _get_git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _is_git_dirty() -> bool:
    try:
        result = subprocess.run(["git", "diff", "--quiet"], capture_output=True)
        return result.returncode != 0
    except Exception:  # noqa: BLE001
        return False


def collect_metadata(cfg: Config) -> dict:
    import torch

    meta = {
        "git_sha": _get_git_sha(),
        "git_dirty": _is_git_dirty(),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "gpu_name": "unknown",
        "host": platform.node(),
    }
    try:
        import transformers

        meta["transformers_version"] = transformers.__version__
    except ImportError:
        pass
    if torch.cuda.is_available():
        meta["gpu_name"] = torch.cuda.get_device_name(0)
        meta["cuda_version"] = torch.version.cuda or "unknown"
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            meta["driver_version"] = out.split("\n")[0]
        except Exception:  # noqa: BLE001
            pass
    return meta


def load_model_and_tokenizer(cfg: Config):
    """Load model + tokenizer with standard config. Returns (model, tokenizer)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info("Loading model %s (dtype=%s, attn=%s)", cfg.model_id, cfg.dtype, cfg.attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_impl,
        device_map=cfg.device,
    )
    model.eval()
    return model, tokenizer


def tokenize_prompts(cfg: Config, tokenizer, prompts: list[str]) -> list[dict]:
    """Tokenize prompts, truncate to ~32 tokens so generation has room."""
    prompt_inputs = []
    for p in prompts:
        ids = tokenizer(p, return_tensors="pt", truncation=True, max_length=32)
        prompt_inputs.append(ids.to(cfg.device))
    return prompt_inputs


def measure_energy(
    cfg: Config,
    model,
    prompt_inputs: list[dict],
    use_cache: bool = True,
    hook_ctx=None,
    phase_label: str = "phase",
) -> dict:
    """Run the energy measurement loop. Returns dict with energy stats.

    hook_ctx: optional context manager to apply hooks during generation.
    """
    import torch

    from .energy import EnergyMeter, bootstrap_ci, locked_clocks, warmup

    def gen_fn():
        with torch.no_grad():
            model.generate(
                **prompt_inputs[0],
                max_new_tokens=cfg.n_tokens,
                use_cache=use_cache,
                do_sample=False,
            )

    with locked_clocks(cfg.gpu_lock_freq_mhz, enable=cfg.lock_clocks):
        logger.info("Warming up for %ds", cfg.warmup_s)
        warmup(gen_fn, seconds=cfg.warmup_s)

        logger.info("Running %d energy measurement runs × %d tokens", cfg.n_runs, cfg.n_tokens)
        jpt_values = []
        energy_values = []
        wallclock_values = []
        power_values = []
        peak_power = 0.0

        ctx = hook_ctx if hook_ctx is not None else nullcontext()
        with ctx:
            for i in range(cfg.n_runs):
                inp = prompt_inputs[i % len(prompt_inputs)]
                prompt_len = inp["input_ids"].shape[-1]
                with (
                    EnergyMeter(sample_interval_ms=cfg.sample_interval_ms) as meter,
                    torch.no_grad(),
                ):
                    output_ids = model.generate(
                        **inp, max_new_tokens=cfg.n_tokens, use_cache=use_cache, do_sample=False
                    )
                r = meter.result
                n_generated = output_ids.shape[-1] - prompt_len
                jpt = r.per_token(n_generated)
                jpt_values.append(jpt)
                energy_values.append(r.total_energy_j)
                wallclock_values.append(r.wallclock_s)
                power_values.append(r.mean_power_w)
                peak_power = max(peak_power, r.peak_power_w)
                if (i + 1) % 10 == 0:
                    logger.info(
                        "  run %d/%d: %.4f J/tok (%d tok), %.1f W",
                        i + 1,
                        cfg.n_runs,
                        jpt,
                        n_generated,
                        r.mean_power_w,
                    )

    jpt_mean, jpt_lo, jpt_hi = bootstrap_ci(jpt_values, seed=cfg.seed)
    e_mean, e_lo, e_hi = bootstrap_ci(energy_values, seed=cfg.seed)
    wc_mean, wc_lo, wc_hi = bootstrap_ci(wallclock_values, seed=cfg.seed)
    pw_mean, pw_lo, pw_hi = bootstrap_ci(power_values, seed=cfg.seed)
    tps_values = [cfg.n_tokens / w for w in wallclock_values]
    tps_mean, tps_lo, tps_hi = bootstrap_ci(tps_values, seed=cfg.seed)

    logger.info(
        "%s: %.4f J/tok [%.4f, %.4f], %.1f tok/s, %.1f W mean",
        phase_label,
        jpt_mean,
        jpt_lo,
        jpt_hi,
        tps_mean,
        pw_mean,
    )

    return {
        "joules_per_token": {
            "mean": jpt_mean,
            "ci_low": jpt_lo,
            "ci_high": jpt_hi,
            "n": cfg.n_runs,
        },
        "total_energy_j": {"mean": e_mean, "ci_low": e_lo, "ci_high": e_hi, "n": cfg.n_runs},
        "wallclock_s": {"mean": wc_mean, "ci_low": wc_lo, "ci_high": wc_hi, "n": cfg.n_runs},
        "mean_power_w": {"mean": pw_mean, "ci_low": pw_lo, "ci_high": pw_hi, "n": cfg.n_runs},
        "peak_power_w": peak_power,
        "tokens_per_second": {
            "mean": tps_mean,
            "ci_low": tps_lo,
            "ci_high": tps_hi,
            "n": cfg.n_runs,
        },
    }


def run_accuracy(cfg: Config, model, tokenizer) -> dict:
    """Run perplexity + HellaSwag and return accuracy dict."""
    from .evaluate import hellaswag_0shot, perplexity_wikitext2

    logger.info("Evaluating perplexity on WikiText-2...")
    ppl = perplexity_wikitext2(model, tokenizer, split=cfg.wikitext_split)
    logger.info("WikiText-2 perplexity: %.2f", ppl)

    logger.info("Evaluating HellaSwag 0-shot...")
    hs = hellaswag_0shot(
        cfg.model_id,
        batch_size=cfg.hellaswag_batch_size,
        device=f"{cfg.device}:0" if ":" not in cfg.device else cfg.device,
        attn_impl=cfg.attn_impl,
    )
    logger.info("HellaSwag acc=%.4f, acc_norm=%.4f", hs["acc"], hs["acc_norm"])

    return {
        "perplexity_wikitext2": ppl,
        "hellaswag_acc": hs["acc"],
        "hellaswag_acc_norm": hs["acc_norm"],
    }


def assemble_result(
    cfg: Config,
    phase_name: str,
    energy: dict,
    accuracy: dict,
    sparsity: dict | None = None,
) -> dict:
    """Assemble the full result dict matching results_schema.json."""
    if sparsity is None:
        sparsity = {
            "target_mlp": None,
            "achieved_mlp_mean": None,
            "achieved_mlp_per_layer": None,
            "target_attn_top_k_frac": None,
            "attention_sink_preserved": None,
            "quantization": None,
        }
    return {
        "phase": phase_name,
        "model_id": cfg.model_id,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "config": {
            "dtype": cfg.dtype,
            "attn_impl": cfg.attn_impl,
            "n_runs": cfg.n_runs,
            "n_tokens": cfg.n_tokens,
            "n_prompts": cfg.n_prompts,
            "seed": cfg.seed,
            "lock_clocks": cfg.lock_clocks,
            "sample_interval_ms": cfg.sample_interval_ms,
        },
        "energy": energy,
        "accuracy": accuracy,
        "sparsity": sparsity,
        "metadata": collect_metadata(cfg),
    }


def validate_and_write(result: dict, cfg: Config, filename: str) -> Path:
    """Validate against schema and write to results dir."""
    from .schema import validate

    validate(result)
    out_path = Path(cfg.results_dir) / filename
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    logger.info("Results written to %s", out_path)
    return out_path
