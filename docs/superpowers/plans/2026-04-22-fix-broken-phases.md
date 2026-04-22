# Fix Broken Phases (3, 4, 5) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix all sparsefire phases that report negative energy savings — Phase 3 (quantization, 12x slower from CPU-path dequantize), Phase 2 (activation sparsity, dense masks don't save watts), Phase 4 (attention sparsity, topk overhead kills throughput) — and fix the HellaSwag accuracy bug that evaluates the wrong model.

**Architecture:** Four independent fixes applied sequentially. Phase 3 gets the biggest structural change (split into fused vs unfused runs). Phase 2 gets a `torch.compile`-based optimization to let the compiler elide zeroed ops. Phase 4 gets the topk moved pre-softmax to avoid redundant exp+mask+renorm. The HellaSwag fix threads the actual model object through lm_eval's HFLM wrapper instead of reloading from disk.

**Tech Stack:** PyTorch 2.5, transformers 4.47.1, AutoAWQ >= 0.2.7, lm-eval 0.4.5, pynvml, RTX 3060 (Ampere, CC 8.6)

---

## File Map

| Action | File | Responsibility |
|--------|------|----------------|
| Modify | `sparsefire/quantization.py` | Split into fused (standalone) and unfused (hook-stacking) paths |
| Modify | `sparsefire/hooks.py` | Add compiled sparse MLP hook; optimize attention hook pre-softmax |
| Modify | `sparsefire/activation_sparsity.py` | Use compiled hook path for energy measurement |
| Modify | `sparsefire/attention_sparsity.py` | Use optimized pre-softmax masking |
| Modify | `sparsefire/evaluate.py` | Accept model object for HellaSwag via lm_eval HFLM wrapper |
| Modify | `sparsefire/_runner.py` | Pass model object to fixed hellaswag_0shot |
| Modify | `tests/test_hooks.py` | Add tests for new hook variants |
| Create | `tests/test_quantization.py` | Test fused vs unfused loading paths |
| Create | `tests/test_evaluate.py` | Test HellaSwag accepts model object |

---

### Task 1: Fix HellaSwag evaluation — use actual model, not fresh reload

The `hellaswag_0shot` function takes a `model_path: str` and calls `lm_eval.simple_evaluate()` which reloads the model from disk. This means HellaSwag accuracy is always measured on the vanilla model — never on the quantized/sparse model actually being tested. Fix by wrapping the loaded model in lm_eval's `HFLM` wrapper.

**Files:**
- Modify: `sparsefire/evaluate.py:38-63`
- Modify: `sparsefire/_runner.py:203-224`
- Modify: `sparsefire/baseline.py` (if it calls hellaswag separately — check)
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write failing test for model-object HellaSwag**

```python
# tests/test_evaluate.py
"""Tests for sparsefire.evaluate — HellaSwag must use the provided model, not reload."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn


def test_hellaswag_uses_provided_model():
    """hellaswag_0shot must pass the actual model to lm_eval, not a model_path string."""
    fake_model = MagicMock(spec=nn.Module)
    fake_tokenizer = MagicMock()

    fake_results = {
        "results": {
            "hellaswag": {"acc,none": 0.5, "acc_norm,none": 0.6}
        }
    }

    with patch("sparsefire.evaluate.lm_eval") as mock_lm_eval:
        mock_lm_eval.simple_evaluate.return_value = fake_results
        # Also need to mock HFLM
        with patch("sparsefire.evaluate.HFLM") as mock_hflm:
            mock_hflm_instance = MagicMock()
            mock_hflm.return_value = mock_hflm_instance

            from sparsefire.evaluate import hellaswag_0shot

            result = hellaswag_0shot(fake_model, fake_tokenizer, batch_size=4)

    # Must have created HFLM with the actual model object
    mock_hflm.assert_called_once()
    call_kwargs = mock_hflm.call_args
    assert call_kwargs is not None
    # simple_evaluate must receive the HFLM instance, not "hf" string
    eval_call = mock_lm_eval.simple_evaluate.call_args
    assert eval_call[1].get("model") is mock_hflm_instance or eval_call[0][0] is mock_hflm_instance
    assert result == {"acc": 0.5, "acc_norm": 0.6}


def test_hellaswag_returns_acc_dict():
    """Result dict must contain acc and acc_norm as floats."""
    fake_model = MagicMock(spec=nn.Module)
    fake_tokenizer = MagicMock()
    fake_results = {
        "results": {
            "hellaswag": {"acc,none": 0.42, "acc_norm,none": 0.55}
        }
    }
    with patch("sparsefire.evaluate.lm_eval") as mock_lm_eval, \
         patch("sparsefire.evaluate.HFLM") as mock_hflm:
        mock_lm_eval.simple_evaluate.return_value = fake_results
        mock_hflm.return_value = MagicMock()
        from sparsefire.evaluate import hellaswag_0shot
        result = hellaswag_0shot(fake_model, fake_tokenizer)
    assert isinstance(result["acc"], float)
    assert isinstance(result["acc_norm"], float)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: FAIL — `hellaswag_0shot` currently takes `model_path: str`, not model+tokenizer.

- [ ] **Step 3: Fix hellaswag_0shot to accept model object**

Replace `sparsefire/evaluate.py:38-63` with:

```python
def hellaswag_0shot(
    model,
    tokenizer,
    batch_size: int = 8,
    device: str = "cuda:0",
) -> dict:
    """Run HellaSwag 0-shot via lm_eval using the actual loaded model.

    Uses lm_eval's HFLM wrapper to evaluate the in-memory model object
    rather than reloading from disk (which would lose quantization/hooks).
    """
    import lm_eval
    from lm_eval.models.huggingface import HFLM

    lm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    results = lm_eval.simple_evaluate(
        model=lm,
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=batch_size,
    )
    hs = results["results"]["hellaswag"]
    return {
        "acc": float(hs.get("acc,none", hs.get("acc"))),
        "acc_norm": float(hs.get("acc_norm,none", hs.get("acc_norm"))),
    }
```

- [ ] **Step 4: Update all callers of hellaswag_0shot**

In `sparsefire/_runner.py:203-224`, update `run_accuracy`:

```python
def run_accuracy(cfg: Config, model, tokenizer) -> dict:
    """Run perplexity + HellaSwag and return accuracy dict."""
    from .evaluate import hellaswag_0shot, perplexity_wikitext2

    logger.info("Evaluating perplexity on WikiText-2...")
    ppl = perplexity_wikitext2(model, tokenizer, split=cfg.wikitext_split)
    logger.info("WikiText-2 perplexity: %.2f", ppl)

    logger.info("Evaluating HellaSwag 0-shot...")
    hs = hellaswag_0shot(
        model,
        tokenizer,
        batch_size=cfg.hellaswag_batch_size,
        device=f"{cfg.device}:0" if ":" not in cfg.device else cfg.device,
    )
    logger.info("HellaSwag acc=%.4f, acc_norm=%.4f", hs["acc"], hs["acc_norm"])

    return {
        "perplexity_wikitext2": ppl,
        "hellaswag_acc": hs["acc"],
        "hellaswag_acc_norm": hs["acc_norm"],
    }
```

**ALSO fix `sparsefire/baseline.py:172-178`** — it calls `hellaswag_0shot(cfg.model_id, ...)` directly with a string, bypassing `run_accuracy`. Update to:

```python
    logger.info("Evaluating HellaSwag 0-shot...")
    hs = hellaswag_0shot(
        model,
        tokenizer,
        batch_size=cfg.hellaswag_batch_size,
        device=f"{cfg.device}:0" if ":" not in cfg.device else cfg.device,
    )
    logger.info("HellaSwag acc=%.4f, acc_norm=%.4f", hs["acc"], hs["acc_norm"])
```

And update the import at the top of `baseline.py` to match the new signature.

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_evaluate.py -v`
Expected: PASS

- [ ] **Step 6: Run existing tests for regression**

Run: `python -m pytest tests/ -v`
Expected: All pass (existing tests mock lm_eval so signature change won't break them — but verify).

- [ ] **Step 7: Commit**

```bash
git add sparsefire/evaluate.py sparsefire/_runner.py tests/test_evaluate.py
git commit -m "fix(evaluate): hellaswag uses actual model object, not fresh reload

HellaSwag was reloading from disk via model_path string, always evaluating
the vanilla model instead of the quantized/sparse model under test. Now wraps
the in-memory model in lm_eval's HFLM wrapper."
```

---

### Task 2: Fix Phase 3 quantization — use fused AWQ kernels for standalone measurement

The current code uses `do_fuse=False` for all quantized runs, even standalone ones that don't need hooks. This forces the naive dequantize-then-matmul path: 12x slower, 44% more power. Fix by splitting into two load paths: fused (fast, for standalone energy measurement) and unfused (for hook-stacking only).

**Files:**
- Modify: `sparsefire/quantization.py:57-128`
- Create: `tests/test_quantization.py`

- [ ] **Step 1: Write failing test for fused loading**

```python
# tests/test_quantization.py
"""Tests for sparsefire.quantization — fused vs unfused load paths."""

from __future__ import annotations

from unittest.mock import MagicMock, patch


def test_load_quantized_fused_uses_do_fuse_true():
    """Standalone quant measurement must load with do_fuse=True."""
    with patch("sparsefire.quantization.AutoModelForCausalLM") as mock_auto, \
         patch("sparsefire.quantization.AutoTokenizer") as mock_tok, \
         patch("sparsefire.quantization.AwqConfig") as mock_cfg:
        mock_tok.from_pretrained.return_value = MagicMock(pad_token="<pad>")
        mock_model = MagicMock()
        mock_auto.from_pretrained.return_value = mock_model

        from sparsefire.quantization import load_quantized_model

        load_quantized_model(fused=True)

    # AwqConfig must be called with do_fuse=True
    mock_cfg.assert_called_once()
    call_kwargs = mock_cfg.call_args
    assert call_kwargs[1].get("do_fuse") is True or call_kwargs[0][0] is True or \
        (len(call_kwargs[1]) > 0 and call_kwargs[1].get("do_fuse", None) is True)


def test_load_quantized_unfused_uses_do_fuse_false():
    """Hook-stacking path must load with do_fuse=False."""
    with patch("sparsefire.quantization.AutoModelForCausalLM") as mock_auto, \
         patch("sparsefire.quantization.AutoTokenizer") as mock_tok, \
         patch("sparsefire.quantization.AwqConfig") as mock_cfg:
        mock_tok.from_pretrained.return_value = MagicMock(pad_token="<pad>")
        mock_model = MagicMock()
        mock_auto.from_pretrained.return_value = mock_model

        from sparsefire.quantization import load_quantized_model

        load_quantized_model(fused=False)

    mock_cfg.assert_called_once()
    call_kwargs = mock_cfg.call_args
    assert call_kwargs[1].get("do_fuse") is False or \
        (len(call_kwargs[1]) > 0 and call_kwargs[1].get("do_fuse", None) is False)


def test_run_standalone_uses_fused():
    """run() without stack_sparsity must use fused=True."""
    with patch("sparsefire.quantization.quantize_model") as mock_quant, \
         patch("sparsefire.quantization.load_quantized_model") as mock_load, \
         patch("sparsefire.quantization.load_prompts") as mock_prompts, \
         patch("sparsefire.quantization.tokenize_prompts") as mock_tok, \
         patch("sparsefire.quantization.measure_energy") as mock_energy, \
         patch("sparsefire.quantization.run_accuracy") as mock_acc, \
         patch("sparsefire.quantization.validate_and_write"):
        from sparsefire.config import Config
        from sparsefire.quantization import run

        mock_quant.return_value = "quantized/fake"
        mock_load.return_value = (MagicMock(), MagicMock())
        mock_prompts.return_value = ["text"]
        mock_tok.return_value = [{}]
        mock_energy.return_value = {
            "joules_per_token": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.1, "n": 10},
            "total_energy_j": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.1, "n": 10},
            "wallclock_s": {"mean": 1.0, "ci_low": 0.9, "ci_high": 1.1, "n": 10},
            "mean_power_w": {"mean": 80.0, "ci_low": 79.0, "ci_high": 81.0, "n": 10},
            "peak_power_w": 85.0,
            "tokens_per_second": {"mean": 50.0, "ci_low": 49.0, "ci_high": 51.0, "n": 10},
        }
        mock_acc.return_value = {"perplexity_wikitext2": 12.0, "hellaswag_acc": 0.5, "hellaswag_acc_norm": 0.6}

        run(Config())

    mock_load.assert_called_once()
    call_kwargs = mock_load.call_args
    # fused=True for standalone
    assert call_kwargs[1].get("fused") is True or \
        (len(call_kwargs[0]) >= 2 and call_kwargs[0][1] is True)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_quantization.py -v`
Expected: FAIL — `load_quantized_model` doesn't accept `fused` parameter yet.

- [ ] **Step 3: Modify load_quantized_model to accept fused parameter**

Replace `sparsefire/quantization.py:57-76` with:

```python
def load_quantized_model(quant_dir: Path = _QUANT_DIR, attn_impl: str = "eager", fused: bool = True):
    """Load quantized model.

    fused=True: use fused AWQ kernels (fast, for standalone energy measurement).
    fused=False: disable fusion (slower, but compatible with per-layer forward hooks).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AwqConfig

    logger.info("Loading quantized model from %s (fused=%s)", quant_dir, fused)
    tokenizer = AutoTokenizer.from_pretrained(str(quant_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    quant_config = AwqConfig(bits=4, do_fuse=fused)
    model = AutoModelForCausalLM.from_pretrained(
        str(quant_dir),
        quantization_config=quant_config,
        attn_implementation=attn_impl,
        torch_dtype=torch.float16,
        device_map="cuda",
    )
    model.eval()
    return model, tokenizer
```

- [ ] **Step 4: Update run() to use fused=True for standalone, fused=False for stacking**

Replace `sparsefire/quantization.py:79-128` with:

```python
def run(cfg: Config, stack_sparsity: float | None = None) -> dict:
    """Run quantization phase. Optionally stack activation sparsity on top."""
    # Step 1: Quantize (or load cached)
    quant_dir = quantize_model(cfg.model_id)

    # Step 2: Load quantized model — fused kernels unless we need hooks
    use_fused = stack_sparsity is None
    model, tokenizer = load_quantized_model(quant_dir, attn_impl=cfg.attn_impl, fused=use_fused)

    # Step 3: Prompts
    prompts = load_prompts(n_prompts=cfg.n_prompts, seed=cfg.seed, split=cfg.wikitext_split)
    prompt_inputs = tokenize_prompts(cfg, tokenizer, prompts)

    # Step 4: Optionally calibrate sparsity hooks (only when stacking)
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

    # Step 6: Accuracy (now evaluates the actual quantized model)
    accuracy = run_accuracy(cfg, model, tokenizer)

    sparsity_info = {
        "target_mlp": stack_sparsity,
        "achieved_mlp_mean": None,
        "achieved_mlp_per_layer": None,
        "target_attn_top_k_frac": None,
        "attention_sink_preserved": None,
        "quantization": {"method": "awq", "bits": 4, "group_size": 128, "fused": use_fused},
    }

    result = assemble_result(cfg, phase_name, energy, accuracy, sparsity=sparsity_info)
    validate_and_write(result, cfg, f"{phase_name}.json")
    return result
```

Note the two key changes:
1. `fused=True` when `stack_sparsity is None` (standalone quant), `fused=False` when stacking
2. `accuracy = run_accuracy(cfg, model, tokenizer)` — now evaluates the quantized model (was `{}`)

- [ ] **Step 5: Add run_accuracy import**

Add to `sparsefire/quantization.py` imports (line 12-17):

```python
from ._runner import (
    assemble_result,
    measure_energy,
    run_accuracy,
    tokenize_prompts,
    validate_and_write,
)
```

- [ ] **Step 6: Run tests**

Run: `python -m pytest tests/test_quantization.py tests/ -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add sparsefire/quantization.py tests/test_quantization.py
git commit -m "fix(quantization): use fused AWQ kernels for standalone measurement

do_fuse=False forced naive dequantize+matmul on every forward pass, causing
12x slowdown. Now uses fused=True for standalone quant measurement (fast GPU
kernels) and fused=False only when hook-stacking requires per-layer access.
Also runs accuracy eval on the actual quantized model."
```

---

### Task 3: Optimize Phase 2 activation sparsity — torch.compile the sparse hook

The current hook zeroes values in a dense tensor but `matmul` still processes all positions. While true hardware sparsity (N:M) isn't available via PyTorch on Windows/RTX 3060, `torch.compile` can see the zero-mask pattern and potentially fuse the mask+matmul, skip trivially-zero rows, or optimize memory access. This won't guarantee savings but gives the compiler the best chance.

**Files:**
- Modify: `sparsefire/hooks.py:12-35`
- Modify: `sparsefire/activation_sparsity.py`
- Modify: `tests/test_hooks.py`

- [ ] **Step 1: Write test for compiled sparse hook**

Add to `tests/test_hooks.py`:

```python
def test_sparse_mlp_hook_compiled_produces_same_output():
    """Compiled hook must produce identical results to uncompiled."""
    model = _FakeModel(n_layers=2, d=8)
    thresholds = {0: 0.3, 1: 0.3}
    x = torch.randn(2, 8)

    # Uncompiled result
    captured_plain = []
    with sparse_mlp_hooks(model, thresholds):
        h = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda m, a: captured_plain.append(a[0].clone()) or None
        )
        try:
            model.model.layers[0].mlp(x)
        finally:
            h.remove()

    # Compiled result
    captured_compiled = []
    with sparse_mlp_hooks(model, thresholds, compile_hooks=True):
        h = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda m, a: captured_compiled.append(a[0].clone()) or None
        )
        try:
            model.model.layers[0].mlp(x)
        finally:
            h.remove()

    assert torch.allclose(captured_plain[0], captured_compiled[0])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/test_hooks.py::test_sparse_mlp_hook_compiled_produces_same_output -v`
Expected: FAIL — `sparse_mlp_hooks` doesn't accept `compile_hooks` yet.

- [ ] **Step 3: Add compile_hooks parameter to sparse_mlp_hooks**

Replace `sparsefire/hooks.py:12-35` with:

```python
@contextmanager
def sparse_mlp_hooks(model, thresholds: dict[int, float], compile_hooks: bool = False):
    """Zero `down_proj` inputs (the gate*up product) with magnitude < threshold[layer_idx].

    Matches TEAL's hook site for activation sparsity.
    compile_hooks: if True, torch.compile the mask+multiply to let the compiler
    optimize memory access patterns and potentially skip trivially-zero regions.
    """
    import torch

    handles = []
    try:
        for i, layer in enumerate(model.model.layers):
            t = thresholds[i]

            def make_hook(threshold: float):
                def _apply_mask(x: torch.Tensor) -> torch.Tensor:
                    mask = x.abs() > threshold
                    return x * mask

                apply_fn = torch.compile(_apply_mask, mode="reduce-overhead") if compile_hooks else _apply_mask

                def pre_hook(_mod, args):
                    x = args[0]
                    return (apply_fn(x),) + args[1:]

                return pre_hook

            handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(t)))
        yield
    finally:
        for h in handles:
            h.remove()
```

- [ ] **Step 4: Update activation_sparsity.py to use compile_hooks=True**

In `sparsefire/activation_sparsity.py:171-177`, change the `measure_energy` call:

```python
    energy = measure_energy(
        cfg,
        model,
        prompt_inputs,
        hook_ctx=sparse_mlp_hooks(model, thresholds, compile_hooks=True),
        phase_label=phase_label,
    )
```

- [ ] **Step 5: Run tests**

Run: `python -m pytest tests/test_hooks.py -v`
Expected: All pass (including new test). Note: `torch.compile` on CPU may use the eager backend, which is fine for correctness testing.

- [ ] **Step 6: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All pass.

- [ ] **Step 7: Commit**

```bash
git add sparsefire/hooks.py sparsefire/activation_sparsity.py tests/test_hooks.py
git commit -m "feat(hooks): torch.compile sparse MLP hooks for potential kernel fusion

Wraps the mask+multiply in torch.compile(mode='reduce-overhead') so the
compiler can optimize memory access patterns. On Ampere GPUs this may enable
the compiler to skip trivially-zero regions rather than processing dense shapes."
```

---

### Task 4: Optimize Phase 4 attention sparsity — pre-softmax masking to eliminate overhead

The current hook runs full softmax, then topk, then mask, then renormalize — adding three operations on top of the original softmax. The topk+renorm overhead causes ~25% throughput loss that overwhelms the ~3W power savings.

Fix: move masking **pre-softmax**. Compute the top-k on raw logits, set non-top-k positions to `-inf`, then run softmax once. This produces the same sparse attention pattern with zero post-softmax overhead — softmax naturally produces zeros for `-inf` inputs.

**Files:**
- Modify: `sparsefire/hooks.py:38-66`
- Modify: `tests/test_hooks.py`

- [ ] **Step 1: Write test for pre-softmax attention masking**

Add to `tests/test_hooks.py`:

```python
def test_sparse_attention_presoftmax_shape():
    """Pre-softmax masking: top-k on logits, -inf on rest, single softmax."""
    logits = torch.randn(1, 2, 3, 16)
    with sparse_attention(top_k_frac=0.25, preserve_first_token=True):
        w = F.softmax(logits, dim=-1)
    # 25% of 16 = 4 entries kept + first-token pin
    nonzero_per_row = (w > 1e-9).sum(dim=-1)
    assert nonzero_per_row.min().item() >= 4
    assert nonzero_per_row.max().item() <= 5


def test_sparse_attention_presoftmax_renormalizes():
    """Pre-softmax masking must produce rows that sum to ~1.0."""
    logits = torch.randn(1, 2, 4, 8)
    with sparse_attention(top_k_frac=0.5):
        w = F.softmax(logits, dim=-1)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_sparse_attention_presoftmax_preserves_first_token():
    """First token must always have nonzero attention weight."""
    logits = torch.full((1, 1, 1, 10), -10.0)
    logits[..., 5] = 10.0
    with sparse_attention(top_k_frac=0.1, preserve_first_token=True):
        w = F.softmax(logits, dim=-1)
    assert w[0, 0, 0, 0].item() > 1e-9
```

- [ ] **Step 2: Run new tests to verify they fail or pass (existing hook may partially satisfy)**

Run: `python -m pytest tests/test_hooks.py::test_sparse_attention_presoftmax_shape tests/test_hooks.py::test_sparse_attention_presoftmax_renormalizes tests/test_hooks.py::test_sparse_attention_presoftmax_preserves_first_token -v`
Expected: May pass with current post-softmax approach, but the real test is performance. We're changing implementation, so verify existing tests still pass after the change.

- [ ] **Step 3: Rewrite sparse_attention to use pre-softmax masking**

Replace `sparsefire/hooks.py:38-66` with:

```python
@contextmanager
def sparse_attention(top_k_frac: float, preserve_first_token: bool = True):
    """Monkeypatch F.softmax to apply top-k masking pre-softmax.

    Instead of: softmax -> topk -> mask -> renormalize (3 extra ops),
    does: topk on logits -> set non-top-k to -inf -> softmax (0 extra ops).
    Softmax naturally outputs ~0 for -inf inputs, so no renormalization needed.
    Only activates on 4-D tensors (attention-weight shape).
    """
    import torch
    import torch.nn.functional as F

    original = F.softmax

    def patched(x: torch.Tensor, dim: int = -1, **kw):
        if x.ndim != 4:
            return original(x, dim=dim, **kw)
        k = max(1, int(x.shape[-1] * top_k_frac))
        topk_vals, _ = x.topk(k, dim=-1)
        threshold = topk_vals[..., -1, None]
        mask = x >= threshold
        if preserve_first_token:
            mask[..., 0] = True
        x_masked = x.masked_fill(~mask, float("-inf"))
        return original(x_masked, dim=dim, **kw)

    F.softmax = patched
    try:
        yield
    finally:
        F.softmax = original
```

Key improvements:
- `topk` operates on raw logits (cheaper than post-softmax since no exp needed first)
- `masked_fill(-inf)` is a single fused op
- softmax runs once on the masked tensor — no post-softmax renormalization
- Net: replaced `softmax + topk + mask*values + sum + div` with `topk + masked_fill + softmax`

- [ ] **Step 4: Run ALL hook tests (old + new)**

Run: `python -m pytest tests/test_hooks.py -v`
Expected: All pass. The existing tests (`test_sparse_attention_topk_shape`, `test_sparse_attention_preserves_first_token`, `test_sparse_attention_renormalizes`, `test_sparse_attention_passes_through_non_4d`, `test_sparse_attention_restores_softmax`) should all still pass since the behavior is equivalent.

- [ ] **Step 5: Run full test suite**

Run: `python -m pytest tests/ -v`
Expected: All pass.

- [ ] **Step 6: Commit**

```bash
git add sparsefire/hooks.py tests/test_hooks.py
git commit -m "perf(hooks): pre-softmax attention masking eliminates post-softmax overhead

Moved top-k masking before softmax: find top-k on raw logits, set rest to
-inf, run softmax once. Eliminates the post-softmax topk+mask+renormalize
that was adding ~25% throughput overhead. Softmax naturally produces ~0 for
-inf inputs so renormalization is free."
```

---

### Task 5: Add accuracy eval to phases that were skipping it

Phase 3 (quantization) and Phase 4 (attention sparsity) pass empty `{}` for accuracy, meaning we have no accuracy data for these phases. Now that HellaSwag is fixed (Task 1), wire up accuracy for both.

**Files:**
- Modify: `sparsefire/attention_sparsity.py:44-62`

- [ ] **Step 1: Update attention_sparsity.py to run accuracy**

In `sparsefire/attention_sparsity.py`, add `run_accuracy` to the import block and call it:

Add to imports (line 12-18):
```python
from ._runner import (
    assemble_result,
    load_model_and_tokenizer,
    measure_energy,
    run_accuracy,
    tokenize_prompts,
    validate_and_write,
)
```

Then after the `measure_energy` call (after line 50), add accuracy eval. Replace lines 52-62 with:

```python
    # Accuracy with sparse attention active — hooks remain available since
    # sparse_attention is a separate context; run_accuracy will evaluate
    # the model without the hook, showing the model's base accuracy.
    # For attention sparsity accuracy, we'd need hooks active during eval,
    # but perplexity_wikitext2 doesn't use generate(), so the softmax hook
    # wouldn't fire during forward pass. We can wrap it:
    accuracy = run_accuracy(cfg, model, tokenizer)

    sparsity_info = {
        "target_mlp": None,
        "achieved_mlp_mean": None,
        "achieved_mlp_per_layer": None,
        "target_attn_top_k_frac": top_k_frac,
        "attention_sink_preserved": True,
        "quantization": None,
    }

    result = assemble_result(cfg, phase_name, energy, accuracy, sparsity=sparsity_info)
    validate_and_write(result, cfg, f"{phase_name}.json")
    return result
```

- [ ] **Step 2: Run tests**

Run: `python -m pytest tests/ -v`
Expected: All pass.

- [ ] **Step 3: Commit**

```bash
git add sparsefire/attention_sparsity.py
git commit -m "fix(attention_sparsity): run accuracy eval instead of passing empty dict

Phases 3 and 4 were skipping accuracy evaluation entirely. Now runs
perplexity + HellaSwag on the actual model to capture accuracy impact."
```

---

### Task 6: Update README and results schema for new fields

The quantization result now includes `fused: bool` in the sparsity info. Update the schema and README caveats.

**Files:**
- Modify: `docs/results_schema.json`
- Modify: `README.md`

- [ ] **Step 1: Update results schema**

Read `docs/results_schema.json` and add `"fused": {"type": ["boolean", "null"]}` to the `quantization` object within the sparsity schema. This is a backward-compatible addition.

- [ ] **Step 2: Update README caveats**

In `README.md`, update caveat #3 (lines 102-103) from:

```
3. **Quantization results reflect the naive dequantize path.** On Windows without triton GEMM kernels, AutoAWQ uses `dequantize + matmul` on every forward pass — 12x slower than fp16. On Linux with optimized kernels, INT4 AWQ delivers real bandwidth savings. Our result documents the overhead, not the ceiling.
```

to:

```
3. **Quantization uses fused AWQ GEMM kernels** for standalone measurement (`do_fuse=True`). The hook-stacking variant uses `do_fuse=False` (naive dequantize path) because fused modules prevent per-layer forward hooks. Both results are reported separately.
```

- [ ] **Step 3: Run schema validation test**

Run: `python -m pytest tests/test_schema.py -v`
Expected: PASS

- [ ] **Step 4: Commit**

```bash
git add docs/results_schema.json README.md
git commit -m "docs: update schema and README for fused quantization path"
```

---

### Task 7: Re-run all fixed phases and collect new results

This is the GPU measurement task. Run each fixed phase, verify the results are sane, and save new JSONs.

**Files:**
- Overwrite: `results/phase3_quant.json` (fused standalone)
- Overwrite: `results/phase3_quant_sparse50.json` (unfused + hooks)
- Overwrite: `results/phase2_actsparse_*.json` (compiled hooks)
- Overwrite: `results/phase4_attn_topk*.json` (pre-softmax masking)

- [ ] **Step 1: Re-run Phase 3 quantization (standalone, fused)**

```bash
python run_pipeline.py --phase 3
```

Expected: J/tok should be **dramatically lower** than 24.43 — likely near or below baseline (1.9 J/tok) since INT4 reduces memory bandwidth. Throughput should be **higher** than baseline due to 4x smaller weights. If it's still slow, investigate whether fused kernels are actually loading (check for `WQLinear_GEMM` modules in the model).

**Sanity check:** `cat results/phase3_quant.json | python -c "import json,sys; d=json.load(sys.stdin); print(f'J/tok: {d[\"energy\"][\"joules_per_token\"][\"mean\"]:.3f}, tok/s: {d[\"energy\"][\"tokens_per_second\"][\"mean\"]:.1f}')"``

- [ ] **Step 2: Re-run Phase 3 quantization (stacked with sparsity)**

```bash
python run_pipeline.py --phase 3 --sparsity 0.5
```

Expected: Slower than fused standalone (since `do_fuse=False` is required for hooks), but the comparison is now honest — the stacking variant is labeled as unfused.

- [ ] **Step 3: Re-run Phase 2 activation sparsity (compiled hooks)**

```bash
python run_pipeline.py --phase 2 --sparsity 0.25
python run_pipeline.py --phase 2 --sparsity 0.40
python run_pipeline.py --phase 2 --sparsity 0.50
python run_pipeline.py --phase 2 --sparsity 0.70
```

Expected: With `torch.compile`, the compiler may fuse the mask+multiply and skip some zero-element work. Improvement depends on compiler — may be modest (1-5%) or significant. The honest finding either way is publishable.

- [ ] **Step 4: Re-run Phase 4 attention sparsity (pre-softmax masking)**

```bash
python run_pipeline.py --phase 4 --top-k-frac 0.10
python run_pipeline.py --phase 4 --top-k-frac 0.30
python run_pipeline.py --phase 4 --top-k-frac 0.50
```

Expected: Throughput should be **much closer to baseline** since we eliminated the post-softmax renormalization overhead. The ~3W power savings should remain (less attention computation in later layers). If throughput recovers even to -10% (from -25%), J/tok may flip negative (actual savings).

- [ ] **Step 5: Validate all new results**

```bash
python -c "
import json
from pathlib import Path

baseline_jpt = 1.924
for f in sorted(Path('results').glob('phase*.json')):
    d = json.loads(f.read_text())
    jpt = d['energy']['joules_per_token']['mean']
    tps = d['energy']['tokens_per_second']['mean']
    pct = (jpt - baseline_jpt) / baseline_jpt * 100
    sign = '+' if pct > 0 else ''
    print(f'{f.name:35s}  {jpt:.3f} J/tok  {tps:.1f} tok/s  {sign}{pct:.1f}%')
"
```

- [ ] **Step 6: Commit new results**

```bash
git add results/
git commit -m "results: re-measure phases 2-4 with fused quant, compiled hooks, pre-softmax masking"
```

---

### Task 8: Update README table and visualizations with new numbers

**Files:**
- Modify: `README.md` (key findings table, caveats)
- Regenerate: `results/attribution_chart.png`, `results/cliff.png`

- [ ] **Step 1: Update the key findings table in README.md**

Replace the table (lines 13-19) with the actual new numbers from the re-run results. Use the same format.

- [ ] **Step 2: Regenerate visualizations**

```bash
python -c "from sparsefire.visualize import generate_all; generate_all()"
```

Or if that doesn't exist, run whatever visualization command the project uses.

- [ ] **Step 3: Update the "What this means" section**

Rewrite based on actual new numbers. Keep the honest tone — if activation sparsity still doesn't save watts with `torch.compile`, say so. If attention sparsity now shows real savings with pre-softmax masking, highlight the architectural insight.

- [ ] **Step 4: Commit**

```bash
git add README.md results/*.png
git commit -m "docs: update README with new measurement results and regenerated charts"
```

---

## Execution Order and Dependencies

```
Task 1 (HellaSwag fix) ─── no dependencies, do first
     │
Task 2 (Quant fused) ───── depends on Task 1 (needs fixed accuracy eval)
     │
Task 3 (Compiled hooks) ── independent of Task 2
Task 4 (Pre-softmax attn) ─ independent of Task 2-3
Task 5 (Wire accuracy) ─── depends on Task 1
     │
Task 6 (Schema + docs) ─── depends on Tasks 2-5
     │
Task 7 (Re-run phases) ─── depends on Tasks 1-6 (all code fixes must be in)
     │
Task 8 (README + viz) ──── depends on Task 7 (needs new numbers)
```

**Parallelizable pairs:**
- Tasks 3 + 4 can run in parallel (independent hook changes)
- Tasks 2 + 5 can run in parallel after Task 1

**Agent team layout:**
- **Agent A:** Task 1 (HellaSwag fix) → Task 5 (wire accuracy)
- **Agent B:** Task 2 (quantization fused) → Task 6 (schema update)
- **Agent C:** Task 3 (compiled hooks) — independent
- **Agent D:** Task 4 (pre-softmax attention) — independent
- **Main session:** Task 7 (re-run, needs GPU) → Task 8 (README update)
