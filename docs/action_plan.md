# sparsefire — action plan v1.0

**Maps to:** `docs/sparsefire_PRD_v1.md`
**Owner:** Tejas + Claude (agent-assisted)
**Repo:** github.com/<tejas>/sparsefire (public)
**Dev machine:** macOS (Darwin), no GPU — lint/unit tests only
**Run machine:** Linux PC with RTX 3090 (24GB) — all measurement + model loads
**Containerization:** single `Dockerfile` with CUDA 12.4 base; same image on both machines

---

## Guiding principles

1. **Evidence before assertions.** No phase is "done" until the CI of its energy delta is computed and its accuracy regression is measured.
2. **Honest nulls are publishable.** If a phase shows no delta, we ship that finding with the explanation.
3. **Every green checkpoint → commit + push + tag.** Progress must be visible on GitHub.
4. **Dev/run split.** Mac writes code & runs unit tests; PC runs anything touching CUDA, pynvml, or real model weights.
5. **TDD for the measurement stack.** `energy.py` and `evaluate.py` are load-bearing — bad measurements invalidate the whole project. They get tests first.

---

## Agent teams & when to use each

| Team | When | Tools |
|---|---|---|
| **feature-dev:code-architect** | Once at Phase 0 — design module APIs, data contracts, results.json schema, before any implementation | read/search only, produces blueprint doc |
| **feature-dev:code-explorer** | Parallel with architect — dissect TEAL repo, AutoAWQ repo, lm-eval-harness to extract the 2–3 patterns we'll reuse | read only |
| **general-purpose (worktree)** | Per-phase implementation — one agent per phase branch, self-contained scope | full code tools in isolated worktree |
| **superpowers:test-driven-development** | `energy.py`, `evaluate.py`, hook correctness tests | TDD skill |
| **feature-dev:code-reviewer** | Gate between phases — reviews diffs before merge to main | read only, reports findings |
| **superpowers:verification-before-completion** | Before I claim any phase done | checklist enforcer |

**Parallelization rule:** GPU is a single shared resource — measurement runs are sequential. Code dev can go in parallel (different worktrees), but only one agent at a time hits the 3090.

---

## Worktree & branching strategy

```
/Users/Tejas/Documents/Sparsefire/          ← main worktree, trunk
/Users/Tejas/Documents/Sparsefire-wt/
  ├── phase-0-scaffold/                      ← initial infra
  ├── phase-1-baseline/
  ├── phase-2-kvcache/
  ├── phase-3-actsparsity/
  ├── phase-4-quant/
  ├── phase-5-attn/
  └── phase-6-viz/
```

Lifecycle per phase: create worktree → implement → unit tests green on Mac → push branch → (if GPU needed) user pulls on PC, runs, reports numbers → code-reviewer gates → merge to main → tag → delete worktree → push tag.

**Tags:** `v0.0-scaffold`, `v0.1-baseline`, `v0.2-kvcache`, `v0.3-actsparsity`, `v0.4-quant`, `v0.5-attn`, `v0.6-viz`, `v1.0-ship`.

---

## Interaction protocol (Tejas ↔ Claude)

- **Checkpoints** = I stop and show you numbers + a decision. You say "proceed" or redirect.
- **GPU handoffs** = I announce "GPU time: pull branch X on the PC, run `make phaseN`, paste me the JSON + stdout tail". I then analyze and continue.
- **Memory** = facts about you/the project go into persistent memory; ephemeral task state stays in TodoWrite.
- **Push cadence** = every green checkpoint auto-pushes; I never force-push, never touch main without review.
- **Course-correct anywhere** = drop in, I adapt.

---

# PHASE 0 — Infrastructure & scaffolding *(Mac, no GPU)*

**Goal:** A skeleton repo where Phase 1 can drop in and just work.

## 0.1 Git + GitHub
- [ ] `git init` in `/Users/Tejas/Documents/Sparsefire`
- [ ] `.gitignore` (python, macos, .env, results/, *.pt, hf cache, __pycache__)
- [ ] `gh repo create sparsefire --public --source . --remote origin`
- [ ] Initial commit: existing PRD + this plan
- [ ] Push `main`

## 0.2 Python project
- [ ] `pyproject.toml` — uv-managed, Python 3.11
- [ ] Dependencies pinned: `torch==2.5.*`, `transformers==4.46.*`, `accelerate`, `pynvml`, `autoawq`, `lm-eval==0.4.*`, `datasets`, `numpy`, `matplotlib`, `pytest`, `ruff`
- [ ] `ruff.toml` — line length 100, strict
- [ ] `Makefile` — `install`, `lint`, `test`, `phase0`, `phase1`, ..., `all`

## 0.3 Containerization
- [ ] `Dockerfile` — `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`, install uv + deps
- [ ] `docker-compose.yml` — GPU passthrough, mounts `./results` and HF cache
- [ ] `scripts/run_on_gpu.sh` — wrapper: `docker compose run --rm gpu python -m sparsefire.<cmd>`
- [ ] Verify Dockerfile builds (on Mac — CPU arch is fine for syntax check; full image build on PC)

## 0.4 Module scaffolding (architect-designed)
Dispatch `feature-dev:code-architect` to produce API contracts. Stubs to land:
```
sparsefire/
  __init__.py
  energy.py              # EnergyMeter class, measure_generation(model, prompts, n_tokens, n_runs)
  evaluate.py            # perplexity_wikitext2, hellaswag_0shot
  baseline.py            # run() → dict
  kv_cache.py
  activation_sparsity.py
  quantization.py
  attention_sparsity.py
  visualize.py
  cli.py                 # argparse front
run_pipeline.py          # calls sparsefire.cli
tests/
  test_energy.py         # mocks pynvml
  test_evaluate.py       # mocks dataset load
  test_hooks.py          # tiny synthetic tensors
```

## 0.5 Results schema (locked early)
- [ ] `docs/results_schema.json` — JSON schema every phase must emit
- [ ] `tests/test_schema.py` — validates any `results_*.json` conforms

## 0.6 Measurement harness (TDD)
Dispatch `superpowers:test-driven-development` for `energy.py`:
- Test: sampler produces N power readings at configured interval
- Test: energy integral = mean(power) × duration
- Test: bootstrap CI computation correct on known input
- Test: lock/unlock clocks wrapper handles missing nvidia-smi gracefully (fallback: warn, no-op on Mac)
- Implementation after tests pass

## 0.7 CI
- [ ] `.github/workflows/ci.yml` — Python 3.11, `ruff check`, `pytest` on Mac-compatible tests only (mark GPU tests with `@pytest.mark.gpu`, skip in CI)

## 0.8 Docs
- [ ] `README.md` skeleton: motivation, methodology, results table (placeholders), caveats, reproduce-in-10-min section
- [ ] `docs/brain_anchor.md` — derivation of the ~2 J/token equivalent with citations

### Phase 0 checkpoint
- Repo pushed, CI green on main
- Dockerfile builds on PC (you'll run `docker compose build` and paste output)
- `pytest` passes on Mac (all GPU tests skipped)
- **Tag `v0.0-scaffold`**, proceed.

---

# PHASE 1 — Baseline *(GPU required)*

**Goal:** The one number everything else deltas from.

## 1.1 PC setup (first GPU handoff)
- [ ] You: `git clone` on PC, `docker compose build`, export `HF_TOKEN`, `huggingface-cli whoami`
- [ ] Smoke: `make smoke` — loads Llama-3.2-1B-Instruct fp16, generates "Hello, world" × 10 tokens, prints latency

## 1.2 Energy harness validation on real hardware
- [ ] `nvidia-smi --lock-gpu-clocks=<base>,<base>` wrapper
- [ ] 60s warmup loop
- [ ] Sanity: idle power reading stable (±5%); loaded-model power >> idle
- [ ] Verify 50ms sampling doesn't skip

## 1.3 Baseline measurement
- [ ] 50 prompts drawn deterministically from WikiText-2 test split (seed=0)
- [ ] 50 runs × 256 tokens generated
- [ ] `results/phase1_baseline.json`: mean ± 95% CI joules/token, p50/p95 latency, tok/s, mean power W

## 1.4 Baseline accuracy
- [ ] WikiText-2 perplexity (standard sliding-window protocol)
- [ ] HellaSwag 0-shot via `lm-eval --model hf --tasks hellaswag --num_fewshot 0`
- [ ] Append to `phase1_baseline.json`

### Phase 1 checkpoint
- **You review:** joules/token, ppl, HellaSwag acc, CV of energy measurement (<5% = healthy)
- If CV > 10% we debug before proceeding — unstable baseline poisons all deltas
- **Tag `v0.1-baseline`**, proceed.

---

# PHASE 2 — KV cache *(GPU)*

## 2.1 Implementation
- [ ] `kv_cache.py`: two generation modes wrapping the same model (`use_cache=True/False`)
- [ ] Unit test: outputs bitwise-identical between modes (mathematical equivalence proof)

## 2.2 Measurement
- [ ] 50 runs × 256 tokens, `use_cache=False` (this is the TRUE dense baseline without caching)
- [ ] 50 runs × 256 tokens, `use_cache=True`
- [ ] `results/phase2_kvcache.json` with delta + CI

## 2.3 Note
Phase 1 already uses `use_cache=True` by default. Phase 2's experiment isolates the delta by *disabling* caching. We report this honestly as "measured KV cache contribution vs hypothetical no-cache dense".

### Phase 2 checkpoint
- **You review:** delta magnitude — likely small at 256 tokens, that's fine and honest
- **Tag `v0.2-kvcache`**

---

# PHASE 3 — Activation sparsity *(dev Mac, measure GPU)*

## 3.1 Hook implementation (Mac)
- [ ] `activation_sparsity.py`: `SparseMLPHook` class, magnitude thresholding post-SwiGLU (on `down_proj` input, following TEAL's recipe — post-act, pre-down)
- [ ] Calibration utility: given target sparsity k%, compute per-layer thresholds from 512 WikiText-2 samples
- [ ] Unit tests (tiny tensors, on Mac): hook zeros correct fraction; mask shape correct; batch dim handled

## 3.2 Calibration run (GPU)
- [ ] Calibrate thresholds for 50% and 90% targets
- [ ] Verify achieved sparsity matches target ±2pp across layers (log per-layer table)

## 3.3 Measurement (GPU)
- [ ] Energy at 50% sparsity (50 runs)
- [ ] Energy at 90% sparsity (50 runs)
- [ ] Perplexity + HellaSwag at both

## 3.4 Results
- [ ] `results/phase3_actsparse_50.json`, `phase3_actsparse_90.json`
- [ ] Honest reporting: theoretical FLOP savings vs measured wattage — both in the JSON

### Phase 3 checkpoint
- **You review:** accuracy regression within budget? (propose: <5% HellaSwag drop at 50%, unbounded at 90% to find the cliff)
- **Tag `v0.3-actsparsity`**

---

# PHASE 4 — Quantization (stacked on sparsity) *(GPU)*

## 4.1 Quantize
- [ ] Run AutoAWQ on Llama-3.2-1B-Instruct → INT4 group-size 128
- [ ] Cache quantized weights locally (+ optionally push to your HF account, your call)
- [ ] Smoke: quantized model generates coherent text

## 4.2 Measurement
- [ ] Energy: quant only (50 runs)
- [ ] Energy: quant + 50% act sparsity stacked (50 runs) — **this is the headline compound number**
- [ ] Energy: quant + 90% act sparsity stacked (50 runs)
- [ ] Accuracy for each configuration

## 4.3 Results
- [ ] `results/phase4_quant.json`, `phase4_quant_plus_sparse50.json`, `phase4_quant_plus_sparse90.json`

### Phase 4 checkpoint
- **You review:** does stacking behave additively as TEAL claims? If the savings don't compound we need to understand why before Phase 5.
- **Tag `v0.4-quant`**

---

# PHASE 5 — Attention sparsity *(GPU)* — the novel piece

## 5.1 Hook implementation (Mac)
- [ ] `attention_sparsity.py`: post-softmax top-k mask, with first-token pin (attention sink)
- [ ] Renormalize masked scores; avoid div-by-zero
- [ ] Unit test: first token always survives; top-k count correct; no NaN on random attn matrices

## 5.2 Measurement (GPU)
- [ ] 50%, 70%, 90% attention sparsity — energy + accuracy at each
- [ ] Full stack: KV + act-sparse-50 + quant + attn-sparse-50 — **the final compound number**
- [ ] Full stack at aggressive: attn-sparse-70

## 5.3 Results
- [ ] `results/phase5_attn_*.json`, `results/phase5_fullstack.json`

### Phase 5 checkpoint
- **You review:** does attention sparsity add savings, or does it break accuracy? Either is a publishable finding.
- **Tag `v0.5-attn`**

---

# PHASE 6 — Cliff experiment + visualizations

## 6.1 Cliff sweep (GPU)
- [ ] Act-sparsity 0% → 99% in 5% increments, perplexity at each
- [ ] Plot: sparsity-vs-perplexity curve → `results/cliff.png`

## 6.2 Neuron firing video (GPU for activation capture, Mac for rendering)
- [ ] Capture MLP activations on a fixed prompt at sparsity levels 0, 10, 20, …, 90%
- [ ] Reshape a chosen layer's hidden states → 32×32 grid
- [ ] Matplotlib animation, 30s, `results/neuron_firing.mp4`

## 6.3 Attribution waterfall (Mac)
- [ ] Pull all `results/phase*.json` → build waterfall chart → `results/attribution_chart.png`
- [ ] Include brain-equivalent bar with explicit "approximate" label

### Phase 6 checkpoint
- **You review:** are the charts ready for social? Is the cliff clear? Is the video visually arresting?
- **Tag `v0.6-viz`**

---

# PHASE 7 — Ship

- [ ] README mini-paper: 1-paragraph pitch, methodology, results table, waterfall embed, neuron-firing gif, caveats section (non-negotiable 4 caveats from PRD §12), reproduce-in-10-min
- [ ] `run_pipeline.py` end-to-end dry run on a fresh docker container
- [ ] All results, charts, video committed (use git-lfs for the mp4)
- [ ] Reel 2 + Reel 3 scripts drafted using real numbers
- [ ] **Tag `v1.0-ship`**, announce

---

## Test points summary (what "green" means at every phase)

| Test | Check |
|---|---|
| Energy measurement stability | CV across 50 runs < 5% |
| Energy delta significance | 95% CIs of phase N and N-1 do not overlap |
| Sparsity achieved | Measured zeros within ±2pp of target |
| Hook correctness | Unit test: zeros the right elements, preserves the right ones |
| Accuracy regression | Within phase-specific budget or explicitly documented as cliff |
| Results schema | JSON validates against `docs/results_schema.json` |
| Reproducibility | `make all` on fresh docker → same numbers ±CI |

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| pynvml readings noisy | Lock clocks, warmup, 50-run bootstrap CI |
| HF hook API drift on transformers bump | Pin transformers version in pyproject |
| AWQ + activation hook incompatibility | Test hook on quantized model early in Phase 4 |
| Attention sparsity crashes on KV cache | Test together before claiming Phase 5 done |
| 3090 thermal throttle during long runs | 60s warmup + clock lock + monitor GPU temp |
| FLOP savings ≈ 0 wattage on dense kernels | Expected; we ship the gap as the headline finding |
