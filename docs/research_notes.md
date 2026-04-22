# Research notes — implementation gotchas

Findings that diverge from or clarify the PRD. Reconciled in the deviation table of [action_plan.md](action_plan.md).

## 1. TEAL hook site (activation sparsity)

The PRD code hooks `layer.mlp.act_fn` output. This is actually the **CATS** approach, which TEAL explicitly benchmarks and *beats*.

TEAL sparsifies **inputs to weight matrices**:
- Input to `gate_proj` (hidden state `x`, Gaussian-ish)
- Input to `up_proj` (same `x`)
- Input to `down_proj` (the `gate*up` product — Laplacian, concentrated near zero, most sparsifiable)

**Our choice:** hook on `down_proj`'s input via `register_forward_pre_hook` on `layer.mlp.down_proj`. Intercepts `args[0]` = `act_fn(gate_proj(x)) * up_proj(x)` (the intermediate). This is the highest-impact site per TEAL's own analysis.

TEAL's reported "free zone" for Llama-3 is 25–40%, not 50% or 90%. 50% causes measurable ppl regression; 90% is a collapse test, not a deployment target.

Sources:
- TEAL paper: https://arxiv.org/abs/2408.14690
- TEAL repo: https://github.com/FasterDecoding/TEAL

## 2. AutoAWQ pipeline

No pre-quantized `meta-llama/Llama-3.2-1B-Instruct` INT4 AWQ checkpoint exists on HuggingFace. We quantize ourselves — ~2–5 minutes on a 3090, one-time, cached locally.

Critical loading flag: **`do_fuse=False`** in `AwqConfig`. Fused modules collapse gate/up/down into single CUDA kernels, breaking per-layer forward hooks we need for activation sparsity stacking.

AutoAWQ pins `transformers==4.47.1`. We match this across the project so installs don't fight.

Loading:
```python
from transformers import AutoModelForCausalLM, AwqConfig
quant = AwqConfig(bits=4, do_fuse=False)
model = AutoModelForCausalLM.from_pretrained(
    "./quantized/llama-3.2-1b-awq",
    quantization_config=quant,
    attn_implementation="eager",
    torch_dtype="float16",
    device_map="cuda",
)
```

Sources: https://github.com/casper-hansen/AutoAWQ · https://huggingface.co/docs/transformers/main/en/quantization/awq

## 3. lm-evaluation-harness HellaSwag

CLI:
```bash
lm_eval --model hf \
  --model_args pretrained=meta-llama/Llama-3.2-1B-Instruct,attn_implementation=eager \
  --tasks hellaswag --num_fewshot 0 --batch_size 8 --device cuda:0
```

Works with AWQ-quantized checkpoints through the `hf` backend; `vllm` backend also supports AWQ first-class if we need faster eval. Expected runtime on 3090: ~10–20 min per phase.

Source: https://github.com/EleutherAI/lm-evaluation-harness

## 4. NVML energy API

Prefer `nvmlDeviceGetTotalEnergyConsumption(handle)` — returns cumulative millijoules since device init. Delta before/after generation = exact energy, no numerical integration of sampled power. Polling `nvmlDeviceGetPowerUsage` at 50 ms is kept only as a diagnostic trace for reporting mean/peak power.

Package: `nvidia-ml-py` (official NVIDIA). The `pynvml` package on PyPI is the community fork and is effectively deprecated.

RTX 3090: full NVML telemetry supported (Ampere consumer cards are fine, unlike some older Pascal/Turing consumer parts).

Clock locking requires root/sudo or the persistence daemon: `sudo nvidia-smi -lgc 1395,1395` (3090 base clock). The container will need `--privileged` or the right cgroup permissions.

## 5. Eager vs SDPA attention

By default `AutoModelForCausalLM` loads Llama with `LlamaSdpaAttention` → uses `torch.nn.functional.scaled_dot_product_attention` → the fused CUDA kernel **does not expose post-softmax attention weights**. Phase 5 (attention sparsity) fundamentally cannot work under SDPA.

Options:
- (A) SDPA for Phases 0–4, eager for Phase 5 → baseline number shifts between phases (invalid comparison).
- (B) **Eager across all phases** → slower (~30%) but every bar in the waterfall chart is comparable.

**Our choice: (B).** We explicitly caveat in the README that a production serving stack would use SDPA/FlashAttention, so our absolute J/token numbers are pessimistic vs a real deployment; the *deltas* between phases are what matter, and deltas are preserved either way.

Load with:
```python
AutoModelForCausalLM.from_pretrained(..., attn_implementation="eager")
```

Patching softmax: we monkeypatch `torch.nn.functional.softmax` inside a context manager (scoped per forward call). Only activates on 4-D tensors (attention-weight shape); other softmax calls pass through unchanged.

Sources: https://github.com/huggingface/transformers/issues/31990 · https://huggingface.co/docs/transformers/en/attention_interface

## Cross-cutting watchlist

- **Transformers 4.47.1 everywhere** — avoid installing AutoAWQ after other pins; put it first in pyproject or install order.
- **First-token preservation in attention sparsity** — the "attention sink" phenomenon (Xiao et al., 2023) means the first token always receives disproportionate attention. We pin `mask[..., 0] = True` before renormalization.
- **Quantized models + hooks** — must test Phase 3 (sparsity) on the AWQ-quantized model before claiming the stack works; compressed-weight forward paths differ slightly.
- **Clock lock in container** — needs `docker run --privileged` or appropriate capabilities; verify on first PC session.
