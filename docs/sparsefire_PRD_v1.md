# sparsefire — PRD v1.0
**Model:** Llama-3.2-1B-Instruct
**Hardware:** Single consumer GPU (4090-class)
**Timeline:** 1–2 weeks
**Goal:** End-to-end pipeline that attributes energy savings phase-by-phase, making the brain-to-AI gap visible and measurable.

---

## 1. The one-line pitch

A reproducible pipeline that runs Llama-3.2-1B through four increasingly brain-like optimization phases, measures real GPU energy at each phase in joules per token, and attributes exactly how much each biomimetic trick contributes to closing the brain-GPU efficiency gap.

---

## 2. The core idea

The brain closes the energy gap through a stack of compounding tricks — not one silver bullet. sparsefire mirrors that stack exactly, applying each trick as a discrete phase, measuring the energy at each step, and showing the cumulative curve.

The pipeline has four phases applied sequentially to the same model:

| Phase | What it does | Brain analogy |
|---|---|---|
| **Baseline** | Dense Llama-3.2-1B, no modifications | A brain where every neuron fires on every thought |
| **Phase 1: KV cache** | Enable KV caching (may already be default — we verify and measure the delta) | Working memory — don't reprocess what you've already seen |
| **Phase 2: Activation sparsity** | Apply TEAL-style magnitude thresholding to MLP hidden states | Neural silence — 95% of neurons quiet at any moment |
| **Phase 3: Quantization** | Stack 4-bit weight quantization (AWQ or bitsandbytes) on top of Phase 2 | Low-precision signaling — spikes are 1-bit, not 32-bit floats |
| **Phase 4: Attention sparsity** | Apply top-k sparsity post-softmax to attention scores | Selective attention — the brain doesn't attend to everything equally |

Each phase is additive. The final number is the compound energy saving across all four. The chart that results — energy per token at each phase, with the brain's equivalent marked as the target — is the deliverable.

---

## 3. What we are NOT building

- A new sparsity algorithm (we use TEAL's magnitude thresholding as-is)
- A production serving system
- A training-time modification (inference only)
- A multi-model benchmark (one model, clean and focused)
- A custom CUDA kernel (we use existing sparse kernels from TEAL/bitsandbytes)

---

## 4. Technical success criteria

These are the exact conditions under which the project is done and shippable:

1. **Baseline energy measured.** We have a stable, reproducible joules-per-token number for dense Llama-3.2-1B at fp16, with variance bars across N=50 runs.
2. **Each phase produces a measurable delta.** Every phase must show a statistically meaningful energy reduction vs the previous phase. If a phase shows no delta, we document why and move on — that finding is still publishable.
3. **Accuracy is tracked at every phase.** Perplexity on WikiText-2 and accuracy on HellaSwag at each phase. We need to know the accuracy cost of each energy saving.
4. **Attribution is clean.** The final output is a bar or waterfall chart: "Phase 1 saved X joules/token, Phase 2 saved Y, Phase 3 saved Z, Phase 4 saved W. Total saving: N%. Remaining gap to brain: M%."
5. **The pipeline is one-command reproducible.** `python run_pipeline.py --model llama-3.2-1b` runs all four phases end to end, outputs a JSON results file and auto-generates the attribution chart. Someone with a 4090 should be able to reproduce our results in under 2 hours.
6. **The neuron firing visualization exists.** A video artifact showing MLP activations at baseline vs Phase 2 (sparsity) — a grid of neurons, most dark, a few firing. This is the content money shot.

---

## 5. Energy measurement methodology

This is the most important technical decision in the whole project. Everything else is software — this is where rigor lives.

**Tool:** `pynvml` — reads real GPU power draw directly from the NVIDIA Management Library. Available on any NVIDIA GPU, no root required.

**Protocol:**
- Warm up GPU for 60 seconds of inference before any measurement begins (eliminates thermal ramp variance)
- Lock GPU clocks with `nvidia-smi --lock-gpu-clocks` to eliminate boost clock noise
- Generate exactly 256 tokens per run, from a fixed prompt set (50 prompts drawn from WikiText-2 test set)
- Sample power draw at 50ms intervals during generation
- Energy per run = mean power draw × wall-clock generation time
- Energy per token = total energy / 256
- Report: mean ± 95% CI across 50 runs
- Report both: raw joules/token AND joules/token normalized to brain equivalent (brain ≈ 20W / ~10 tokens/sec conscious thought ≈ ~2J/token equivalent — we will caveat this comparison carefully)

**What we report honestly:**
GPU sparsity savings are partially theoretical — the GPU still processes the tensor shapes unless we use a sparse kernel. We will explicitly state: "theoretical FLOP savings" vs "measured wattage savings" and explain the gap. This honesty is actually the most interesting finding and becomes reel content on its own.

---

## 6. Accuracy measurement methodology

**Perplexity:** Run model on WikiText-2 test set (standard, reproducible, everyone uses it). Lower = better. Track at each phase.

**Downstream accuracy:** HellaSwag 0-shot via `lm-evaluation-harness`. 10-minute run per phase. Gives us a real-world task accuracy number, not just a statistical proxy.

**The "cliff" experiment:** After the four-phase pipeline is complete, run a secondary sweep on Phase 2 (activation sparsity) specifically — push sparsity from 0% to 99% in 10% increments and find where perplexity collapses. This generates the tradeoff curve that is the most viral chart in the project. Nobody has published this cleanly for Llama-3.2-1B.

---

## 7. Phases — detailed technical spec

### Phase 0: Baseline

```python
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-1B-Instruct",
    torch_dtype=torch.float16,
    device_map="cuda"
)
# No modifications. Measure energy here.
```

**What we're measuring:** The true cost of dense inference. Every weight fetched, every neuron active, full fp16 precision.

**Expected result:** ~X joules/token (we don't know yet — this is the first number we establish on day 1).

---

### Phase 1: KV cache

**What it does:** Caches the key and value tensors from attention layers so they don't get recomputed for each new token during generation. Standard in HuggingFace — likely already on by default. We verify explicitly and measure the delta.

**Implementation:**
```python
# HuggingFace enables KV cache by default in .generate()
# We run baseline WITH and WITHOUT use_cache=False to isolate the delta
outputs = model.generate(..., use_cache=True)   # Phase 1
outputs = model.generate(..., use_cache=False)  # Baseline comparison
```

**Brain analogy for attribution chart:** "Working memory — stopped reprocessing the past"

**Expected energy delta:** Significant for longer sequences, modest for 256-token generation. We measure it honestly and report whatever we find.

**Accuracy impact:** Zero — KV cache is mathematically equivalent to recomputation.

---

### Phase 2: Activation sparsity (MLP)

**What it does:** After the SwiGLU gate in each MLP block, zero out all hidden state values below a magnitude threshold. This skips fetching the corresponding weight columns from VRAM on the next matmul.

**Method:** TEAL-style magnitude thresholding — the simplest, training-free, no-finetuning approach. We calibrate the threshold on 512 samples from WikiText-2 training set to achieve target sparsity levels (we test 50% and 90%).

**Implementation:**
```python
class SparseMLPHook:
    def __init__(self, sparsity_threshold):
        self.threshold = sparsity_threshold

    def __call__(self, module, input, output):
        # Zero out activations below threshold magnitude
        mask = output.abs() > self.threshold
        return output * mask

# Register hook on every MLP activation layer
for layer in model.model.layers:
    layer.mlp.act_fn.register_forward_hook(
        SparseMLPHook(threshold=calibrated_threshold)
    )
```

**Calibration:** Compute threshold as the k-th percentile of activation magnitudes across calibration set. k=50 → 50% sparsity. k=90 → 90% sparsity. We run both and report both.

**Brain analogy for attribution chart:** "Neural silence — most neurons went quiet"

**Expected energy delta:** Theoretical FLOP savings: 50-90%. Measured wattage savings on GPU: likely 10-30% (the gap between these two numbers IS the story — we explain it clearly).

**Accuracy impact:** At 50% sparsity: minimal (~1-2% perplexity increase). At 90%: needs measurement — this is the experiment.

---

### Phase 3: Quantization

**What it does:** Reduces weight precision from fp16 (2 bytes per weight) to int4 (0.5 bytes per weight). Each weight fetch from VRAM now moves 4x less data.

**Method:** AWQ (Activation-aware Weight Quantization) — the cleanest 4-bit method with minimal accuracy loss, pre-quantized checkpoints available on HuggingFace.

**Implementation:**
```python
# Use pre-quantized checkpoint — no quantization runtime needed
model = AutoModelForCausalLM.from_pretrained(
    "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
    # nearest available AWQ checkpoint for our model
)
# Then apply Phase 2 sparsity hooks on top
```

**Note:** We stack Phase 2 sparsity ON TOP of Phase 3 quantization. This is the key compound experiment. TEAL explicitly shows these are independent and stack cleanly.

**Brain analogy for attribution chart:** "Low-precision signaling — neurons fire in 1-bit spikes, not 32-bit floats"

**Expected energy delta:** 1.5-2.5x memory bandwidth reduction = proportional energy reduction in memory traffic.

**Accuracy impact:** AWQ at 4-bit: ~1-3% accuracy drop on HellaSwag vs fp16 baseline. Well-established in literature.

---

### Phase 4: Attention sparsity

**What it does:** After the softmax in each attention head, zero out all but the top-k attention scores per query. The model only "attends" to the most relevant tokens.

**This is the novel piece.** CATS and TEAL don't touch attention. This has not been cleanly benchmarked on Llama-3.2-1B.

**Implementation:**
```python
class SparseAttentionHook:
    def __init__(self, top_k_fraction=0.1):
        self.top_k = top_k_fraction

    def __call__(self, module, input, output):
        # output is (attn_output, attn_weights, past_key_value)
        attn_weights = output[1]
        if attn_weights is None:
            return output
        # Keep only top-k% of attention scores, zero the rest
        k = max(1, int(attn_weights.shape[-1] * self.top_k))
        top_k_vals, _ = attn_weights.topk(k, dim=-1)
        threshold = top_k_vals[..., -1, None]
        sparse_weights = attn_weights * (attn_weights >= threshold)
        # Renormalize
        sparse_weights = sparse_weights / sparse_weights.sum(dim=-1, keepdim=True).clamp(min=1e-9)
        return (output[0], sparse_weights) + output[2:]
```

**Brain analogy for attribution chart:** "Selective attention — stopped attending to everything at once"

**Expected result:** Unknown — this is the experiment. Hypothesis: attention can tolerate 50-70% sparsity before accuracy degrades significantly. The "attention sink" phenomenon (first token always gets high attention) may force a floor of ~10-20% non-sparse.

**What if it fails:** If attention sparsity causes immediate accuracy collapse, that IS the finding. "Attention is fundamentally different from MLPs — here's why the brain analogy breaks down here." Still interesting, still ships.

---

## 8. The attribution chart (the core deliverable)

A waterfall chart. X-axis: phases. Y-axis: joules per token. Each bar shows the energy at that phase. The delta between bars is labeled with the brain analogy and the percentage saving.

```
Baseline:          ████████████████████  100% (X J/token)
+ KV cache:        ███████████████████   ~95% (saves ~5%)   "Working memory"
+ Sparsity 50%:    ████████████          ~60% (saves ~35%)  "Neural silence"  
+ Quantization:    ██████                ~35% (saves ~25%)  "1-bit signaling"
+ Attn sparsity:   ████                  ~25% (saves ~10%)  "Selective attention"
                                                             
Brain equivalent:  █                     ~0.X% of baseline  ← The remaining gap
```

The remaining gap to the brain is explicitly shown. We don't pretend we closed it. We show exactly how much each trick helped and how much is left. That honesty is the differentiator.

---

## 9. The neuron firing visualization

A 30-second video artifact. Python-generated, not hand-animated.

**What it shows:** A 32×32 grid of neurons (representing one MLP layer's hidden states). At baseline: nearly all lit up (white/bright). As sparsity increases from 0% → 90%: neurons go dark one by one, until only a sparse constellation remains — like a night sky clearing.

**Implementation:**
```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

# Capture actual activation magnitudes during a forward pass
# at each sparsity level — use real model activations, not synthetic data
activations_by_sparsity = {}  # {sparsity_level: activation_grid}

fig, ax = plt.subplots(figsize=(8, 8))
# Animate sparsity sweeping from 0% to 90%
# Each frame = one sparsity level
# Color = activation magnitude (bright = active, dark = silent)
```

**Why this matters for content:** This is the shot that stops the scroll. A transformer "learning to think like a brain" — visually, in 30 seconds. No technical knowledge required to feel the impact.

---

## 10. Repository structure

```
sparsefire/
├── README.md                    # The mini-paper
├── run_pipeline.py              # One-command end-to-end runner
├── sparsefire/
│   ├── baseline.py              # Phase 0 measurement
│   ├── kv_cache.py              # Phase 1
│   ├── activation_sparsity.py   # Phase 2 — TEAL-style hooks
│   ├── quantization.py          # Phase 3 — AWQ integration
│   ├── attention_sparsity.py    # Phase 4 — novel piece
│   ├── energy.py                # pynvml measurement utilities
│   ├── evaluate.py              # perplexity + HellaSwag
│   └── visualize.py             # Attribution chart + neuron animation
├── results/
│   ├── results.json             # Raw numbers, auto-generated
│   ├── attribution_chart.png    # The waterfall chart
│   └── neuron_firing.mp4        # The visualization
├── notebooks/
│   └── sparsity_sweep.ipynb     # The 0→99% cliff experiment
└── requirements.txt
```

---

## 11. Action plan — day by day

### Day 1 — Environment + baseline
- [ ] Set up repo, install dependencies (`transformers`, `pynvml`, `lm-evaluation-harness`, `bitsandbytes`, `autoawq`)
- [ ] Load Llama-3.2-1B-Instruct in fp16
- [ ] Implement `energy.py` — GPU power sampling loop, verify readings are stable
- [ ] Run baseline measurement: 50 runs × 256 tokens, record mean joules/token ± CI
- [ ] Run baseline perplexity on WikiText-2
- [ ] **Checkpoint:** We have our ground truth number. Everything else is a delta from here.

### Day 2 — Phase 1 (KV cache) + Phase 2 setup
- [ ] Measure baseline with `use_cache=False` explicitly to isolate KV cache delta
- [ ] Record Phase 1 energy and confirm delta
- [ ] Implement `activation_sparsity.py` — forward hooks on MLP activation layers
- [ ] Calibrate thresholds on 512 WikiText-2 training samples for 50% and 90% targets
- [ ] Verify actual sparsity achieved matches target (count zeros in activations)
- [ ] **Checkpoint:** Hooks work, sparsity levels confirmed, no crashes.

### Day 3 — Phase 2 measurement + accuracy
- [ ] Run Phase 2 energy measurement at 50% sparsity (50 runs)
- [ ] Run Phase 2 energy measurement at 90% sparsity (50 runs)
- [ ] Run perplexity for both sparsity levels
- [ ] Run HellaSwag for both sparsity levels via lm-eval
- [ ] **Checkpoint:** We know the sparsity energy delta and its accuracy cost.

### Day 4 — Phase 3 (quantization) + stacking
- [ ] Load AWQ int4 checkpoint (or apply bitsandbytes 4-bit on the fly)
- [ ] Measure Phase 3 energy (quantization alone, no sparsity)
- [ ] Stack Phase 2 sparsity on top of Phase 3 quantization — measure compound energy
- [ ] Run accuracy for stacked Phase 2+3
- [ ] **Checkpoint:** We have the compound savings number.

### Day 5 — Phase 4 (attention sparsity)
- [ ] Implement `attention_sparsity.py` — post-softmax top-k hooks
- [ ] Test at 50% and 90% attention sparsity
- [ ] Handle attention sink tokens carefully (never zero the first token's attention)
- [ ] Measure energy for full Phase 1+2+3+4 stack
- [ ] Run accuracy for full stack
- [ ] **Checkpoint:** Full pipeline done. We have all four phase numbers.

### Day 6 — The cliff experiment
- [ ] Sweep activation sparsity from 0% to 99% in 5% increments on Phase 2 alone
- [ ] Record perplexity at each step
- [ ] Find the cliff — where does accuracy collapse?
- [ ] Generate the sparsity-vs-perplexity curve
- [ ] **Checkpoint:** The most interesting single chart in the project is done.

### Day 7 — Visualization + pipeline polish
- [ ] Build `visualize.py` — neuron firing animation using real activation captures
- [ ] Build attribution waterfall chart
- [ ] Polish `run_pipeline.py` — one command runs everything, outputs JSON + charts
- [ ] Test full pipeline on a clean environment (or fresh conda env)
- [ ] **Checkpoint:** Repo is reproducible end-to-end.

### Day 8 — README + content
- [ ] Write README as mini-paper: motivation → methodology → results → honest caveats → reproduce in 10 min
- [ ] Export all charts at publication quality
- [ ] Export neuron firing video
- [ ] Final numbers review — sanity check everything
- [ ] **Checkpoint:** Ship-ready.

---

## 12. Honest caveats to include in README (non-negotiable)

These are not weaknesses. They're what separates rigorous work from hype.

1. **FLOP savings ≠ wattage savings on GPU.** GPUs process dense tensor operations even when activations are zero unless a sparse kernel is used. We use PyTorch hooks for sparsity — the wattage savings will be smaller than the FLOP savings. We report both and explain the gap. The gap itself shows why neuromorphic hardware matters.

2. **The brain comparison is approximate.** "20W / ~10 tokens per second" is a rough analogy. Brains don't generate language the way transformers do. We use the comparison to establish the scale of the gap, not to make precise claims.

3. **Single model, single GPU.** These results are for Llama-3.2-1B on a 4090-class GPU. Different model sizes and hardware will produce different numbers. We invite replication.

4. **Attention sparsity is experimental.** Phase 4 is the novel piece. We don't claim it's better than existing methods — we claim it's the first clean public measurement of its energy impact on this architecture.

---

## 13. Definition of done

The project ships when:
- [ ] `run_pipeline.py` runs end-to-end without errors on a 4090
- [ ] `results/results.json` contains all four phases with energy + accuracy numbers
- [ ] `results/attribution_chart.png` is clean and publication-quality
- [ ] `results/neuron_firing.mp4` is render-complete and visually compelling
- [ ] README explains methodology, results, and caveats clearly enough that a researcher could critique it
- [ ] Reel 2 ("existing solutions") script is drafted using the real numbers from Phase 2 results
- [ ] Reel 3 ("the invention") hook is written using the real total savings number

---

*sparsefire PRD v1.0 — ready to build*
