# KT: sparsefire — Phase 0 → Phase 1 handoff

**Read this first.** You are a fresh Claude Code instance on a Linux PC with an RTX 3090. Everything you need to continue the project is here. The other half of the work is happening on Tejas's Mac (no GPU). This repo is the shared truth.

---

## What the project is

**sparsefire** — a 4-phase inference pipeline on Llama-3.2-1B-Instruct that stacks biomimetic energy tricks and attributes per-phase joules-per-token savings via NVML on real hardware. Ship artifacts: per-phase results JSON, waterfall attribution chart, neuron-firing video, mini-paper README.

The four phases, in order, applied additively to the same model:
1. **Baseline** — dense fp16, eager attention
2. **KV cache** A/B — measures the cost of *not* caching
3. **MLP activation sparsity** — TEAL-style, zero `down_proj` inputs below a per-layer magnitude threshold
4. **INT4 AWQ quantization** — stacked on top of sparsity
5. **Post-softmax top-k attention sparsity** — with attention-sink (first token) preservation

(Numbering starts from Phase 0 in code; human-facing phase numbers in the PRD start from 1. Both conventions appear.)

Canonical docs in the repo:
- `docs/sparsefire_PRD_v1.md` — product requirements (authoritative for *what*)
- `docs/action_plan.md` — phase-by-phase plan + **deviation table from PRD** (authoritative for *how*)
- `docs/architecture.md` — module APIs, hook patterns, sequence diagram
- `docs/research_notes.md` — five PRD-diverging findings you MUST honor
- `docs/brain_anchor.md` — derivation of the ~2 J/token brain equivalent
- `docs/results_schema.json` — JSON Schema every `results/phase*.json` must pass

Read all six before making any code changes.

---

## Hardware & dev/run split

- **Mac (Tejas's dev box):** scaffolding, unit tests, lint. No CUDA. Cannot run real model or NVML.
- **This PC (you):** RTX 3090, 24 GB VRAM, Linux, Docker + NVIDIA Container Toolkit. All measurement happens here.
- **One container image on both machines.** `Dockerfile` uses `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04`. The Mac uses it only for syntax validation.

When a task doesn't need GPU (code, docs, tests with mocks), do it either place. When it does (model load, NVML, lm-eval, AWQ quantization), do it here and paste results back.

---

## Current state (as of this KT)

- GitHub: **[github.com/Tejas-JB/sparsefire](https://github.com/Tejas-JB/sparsefire)** (public)
- Open PR: **[#1 — Phase 0: scaffolding](https://github.com/Tejas-JB/sparsefire/pull/1)** (branch `phase-0-scaffold` → `main`)
- 21/21 unit tests green on Mac (`pytest tests/`). Zero require GPU.
- Phase 1 (baseline) = first GPU work, unimplemented. `sparsefire/baseline.py` is a `NotImplementedError` stub.

**Blocker on Tejas's side before Phase 1 can run:**
1. Merge PR #1.
2. Run `gh auth refresh -s workflow` so CI can be installed (`.ci-pending/ci.yml` → `.github/workflows/ci.yml`).
3. Rotate the HF token that was exposed in chat and re-export `HF_TOKEN` fresh.

---

## Locked decisions (do NOT relitigate without strong reason)

1. **Activation-sparsity hook site = `down_proj` input** (TEAL's site, not the PRD's `act_fn` output which is CATS). See `sparsefire/hooks.py::sparse_mlp_hooks`.
2. **Sparsity test levels = 25%, 40%, 50%, 70%** — PRD's 50/90 is outside Llama-3's free zone.
3. **Quantization path:** AutoAWQ → INT4 group-size 128, save locally, load via Transformers-native `AwqConfig(bits=4, do_fuse=False)`. No public 1B checkpoint exists.
4. **Energy primary metric = `nvmlDeviceGetTotalEnergyConsumption` delta.** Polled power (50 ms) is diagnostic only.
5. **`attn_implementation="eager"` across ALL phases.** Costs ~30% latency but is the only way Phase 5 works and keeps every phase's baseline comparable. Documented caveat in README.
6. **Package pins:** `torch==2.5.*`, `transformers==4.47.1` (AutoAWQ requires), `nvidia-ml-py`. See `pyproject.toml`.
7. **Use `nvidia-ml-py`, not `pynvml`** (same import name, different package).
8. **Container runs `--privileged`** because `nvidia-smi --lock-gpu-clocks` needs root. See `docker-compose.yml`.

---

## What you (this Claude instance) are likely picking up next

Assuming PR #1 is merged and token is rotated:

### Phase 1 — baseline implementation + first GPU run

1. `git checkout -b phase-1-baseline` off `main`.
2. Implement `sparsefire/baseline.py::run(cfg)` per `docs/architecture.md`. It should:
   - Load the model with `attn_implementation="eager"`, fp16, `device_map="cuda"`.
   - Acquire 50 prompts via `sparsefire.prompts.load_prompts`.
   - Call `energy.warmup(...)` for 60 s, `energy.lock_gpu_clocks(1395)` (3090 base).
   - Loop 50 runs × 256 tokens per prompt, wrapping each with `EnergyMeter`.
   - Compute `joules_per_token`, `wallclock_s`, `mean_power_w` with `bootstrap_ci`.
   - Call `evaluate.perplexity_wikitext2` and `evaluate.hellaswag_0shot`.
   - Assemble a dict matching `docs/results_schema.json` → write `results/phase0_baseline.json`.
   - Validate with `sparsefire.schema.validate()` before returning.
3. Wire `cli.py` so `python run_pipeline.py --phase 0` calls it.
4. Smoke first: `docker compose run --rm gpu python run_pipeline.py --smoke` must load the model and generate 10 tokens.
5. Real run: `docker compose run --rm gpu python run_pipeline.py --phase 0`.
6. Green checkpoint criteria: CV across 50 runs < 5%; joules/token bootstrap CI width < 10% of the mean.
7. Open PR `phase-1-baseline` → `main`, paste stdout tail + `results/phase0_baseline.json` in the PR body.

### Environment setup on this PC (first-time only)

```bash
git clone https://github.com/Tejas-JB/sparsefire && cd sparsefire
export HF_TOKEN=<freshly rotated token, NOT the one leaked in chat>
huggingface-cli whoami            # confirm Llama-3.2 gated access works
docker compose build              # builds the CUDA image
docker compose run --rm gpu python -c "import torch; print(torch.cuda.get_device_name(0))"
# Expect: NVIDIA GeForce RTX 3090
```

---

## Working conventions

- **Trunk-only pushes are blocked.** Always work on a branch and open a PR. `phase-N-*` naming.
- **Tag after merge:** `v0.1-baseline`, `v0.2-kvcache`, …, `v1.0-ship`.
- **Commit messages** explain the *why*, co-authored by Claude per repo convention.
- **Never skip pre-commit hooks** or force-push main. Never commit secrets.
- **TDD for the measurement stack.** Tests mock NVML (`tests/test_energy.py` shows the pattern). GPU-dependent tests get `@pytest.mark.gpu` and are skipped in CI.
- **Every phase result file must validate** against `docs/results_schema.json` (see `sparsefire/schema.py::validate`).
- **Honest nulls are fine.** If a phase shows no delta, ship that finding with the explanation — don't tune to make it look better.

---

## Agent/subagent pattern

The Mac-side Claude uses:
- `feature-dev:code-architect` once, early, for API design.
- `feature-dev:code-explorer` for reading upstream repos (TEAL, AutoAWQ, lm-eval).
- `general-purpose` subagents in git worktrees for independent implementation chunks.
- `feature-dev:code-reviewer` as gate between phases.
- `superpowers:test-driven-development` for critical utilities.

Feel free to match this. Worktrees live at `../sparsefire-wt/phase-N-*/`. One worktree per phase. Merge, delete worktree, tag, repeat.

---

## Contact with the Mac side

The Mac Claude's memory at `~/.claude/projects/-Users-Tejas-Documents-Sparsefire/memory/` contains:
- `project_sparsefire.md` — project summary
- `user_hardware.md` — the dev/run split

You don't have access to that memory from this PC, but everything substantive is in the repo docs. If you learn something non-obvious worth persisting, add a memory note *on your side* and also update the appropriate `docs/*.md` file so it survives across machines.

---

## Quick checklist before you do anything

- [ ] Read `docs/sparsefire_PRD_v1.md`, `docs/action_plan.md`, `docs/architecture.md`, `docs/research_notes.md`.
- [ ] Confirm `git status` clean, on `main`, PR #1 merged.
- [ ] Confirm `HF_TOKEN` in env and not the leaked one.
- [ ] `docker compose build` succeeds.
- [ ] `python run_pipeline.py --smoke` inside container generates text.
- [ ] Only then start writing Phase 1 code.

Ship honestly. The gap to the brain is the story — closing it partially with clean measurements is the deliverable, not pretending it's closed.
