# sparsefire

**Brain-inspired energy-efficiency pipeline for Llama-3.2-1B.**
Four additive phases of biomimetic tricks — KV cache, activation sparsity, INT4 quantization, attention sparsity — each measured in joules-per-token on real hardware, with honest per-phase attribution of the savings.

> Status: **Phase 0 — scaffolding.** Measurement numbers below are placeholders until the 3090 runs land.

---

## The idea in one paragraph

The brain closes the energy gap to digital computation not through one silver bullet but through a stack of compounding tricks: working memory so past inputs aren't reprocessed; >95% of neurons silent at any instant; communication in 1-bit spikes rather than 32-bit floats; attention that focuses rather than attending to everything equally. `sparsefire` applies each of these tricks as a discrete, measurable phase to the same Llama-3.2-1B model on an RTX 3090 and reports how much of the brain-to-GPU gap each trick actually closes.

---

## The phases

| # | Trick | Brain analogy | Status |
|---|---|---|---|
| 0 | Dense fp16 baseline | "Every neuron fires on every thought" | pending |
| 1 | KV cache A/B | Working memory | pending |
| 2 | MLP activation sparsity (TEAL, hooks on `down_proj` input) | Neural silence | pending |
| 3 | INT4 AWQ quantization (stacked on sparsity) | 1-bit spike signaling | pending |
| 4 | Post-softmax top-k attention | Selective attention | pending |
| 5 | Cliff sweep + visualizations | — | pending |

---

## Deliverables

- `results/results.json` — per-phase energy + accuracy with 95% CI
- `results/attribution_chart.png` — waterfall of per-phase savings vs brain anchor
- `results/neuron_firing.mp4` — 30-second MLP activation animation across sparsity levels
- `results/cliff.png` — activation-sparsity-vs-perplexity curve

---

## Reproducing

On a Linux box with an NVIDIA GPU (≥24GB VRAM) and Docker with NVIDIA Container Toolkit:

```bash
git clone https://github.com/Tejas-JB/sparsefire
cd sparsefire
export HF_TOKEN=<your gated-Llama token>
docker compose build
docker compose run --rm gpu python run_pipeline.py --all
```

Full pipeline runs in ~2h on an RTX 3090.

---

## Methodology highlights (ships honest)

1. **Energy via `nvmlDeviceGetTotalEnergyConsumption`** — cumulative-energy delta before/after generation, not polled-power numerical integration.
2. **Clock-locked GPU** — `nvidia-smi --lock-gpu-clocks` pins the clock to the 3090 base frequency to eliminate boost-clock noise.
3. **60-second warmup** before every measurement block eliminates thermal ramp variance.
4. **50 runs × 256 tokens per configuration** with bootstrapped 95% confidence intervals.
5. **Eager attention across all phases** — SDPA doesn't expose post-softmax weights. Keeping eager everywhere trades ~30% headline latency for apples-to-apples comparisons across phases. See [docs/research_notes.md](docs/research_notes.md#5).

---

## Caveats (non-negotiable)

1. **FLOP savings ≠ wattage savings.** PyTorch hooks zero activations; the GPU still processes the dense tensor shapes. Theoretical FLOP reductions are larger than measured wattage reductions. We report both and explain the gap — that gap is the argument for neuromorphic hardware.
2. **Brain comparison is approximate.** "~2 J/token" is derived in [docs/brain_anchor.md](docs/brain_anchor.md) with caveats. Use it for scale, not precision.
3. **Single model, single GPU.** RTX 3090, Llama-3.2-1B-Instruct only. Other configurations will produce different numbers.
4. **Attention sparsity is experimental.** First clean public measurement of post-softmax top-k sparsity's energy impact on Llama-3.2-1B.

---

## Docs

- [docs/sparsefire_PRD_v1.md](docs/sparsefire_PRD_v1.md) — product requirements
- [docs/action_plan.md](docs/action_plan.md) — phase-by-phase build plan
- [docs/architecture.md](docs/architecture.md) — module APIs + hook patterns
- [docs/research_notes.md](docs/research_notes.md) — implementation gotchas
- [docs/brain_anchor.md](docs/brain_anchor.md) — the 2 J/token derivation
- [docs/results_schema.json](docs/results_schema.json) — JSON schema for per-phase results

---

## License

MIT.
