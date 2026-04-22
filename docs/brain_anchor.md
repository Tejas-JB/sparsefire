# Brain energy anchor — derivation

Sparsefire reports GPU joules-per-token alongside an **approximate** brain equivalent. This file shows the math and the caveats. The comparison is intentionally rough — brains don't generate next-token predictions — but it anchors the scale of the gap.

## The number we cite: ~2 J / "token-equivalent"

### Inputs

| Quantity | Value | Source |
|---|---|---|
| Human brain resting power | ~20 W | Attwell & Laughlin (2001), *J. Cereb. Blood Flow Metab.* 21(10): 1133–1145 — the canonical accounting of the brain's energy budget from synaptic + action-potential activity |
| Conscious reading/speaking rate | ~5–10 tokens/s | Standard psycholinguistic estimates: silent reading ≈ 250 wpm ≈ 5 tok/s assuming ~1.3 tokens/word; inner speech can reach ~10 tok/s |
| Fraction of brain engaged in language | ~0.1 (order-of-magnitude) | Language network (Broca + Wernicke + associated) comprises a small fraction of cortical activity; full 20 W is not dedicated to language alone — but for a scale-of-the-gap comparison we use the full 20 W conservatively, which *overestimates* the brain's cost and *understates* the true gap |

### Calculation

```
Brain power (upper bound for language use):   20 W
Token rate:                                   10 tokens/s
Energy per token:                             20 / 10 = 2 J/token
```

Using the lower reading rate (5 tok/s): 4 J/token.

We report the 2 J/token figure in charts and document both bounds.

## Why this is a ceiling, not a floor

- The entire brain is not doing language. A tighter estimate using only the language network would yield ~0.2 J/token.
- We intentionally use the whole-brain figure because any improvement to a part of the brain's budget is still "brain-like efficiency," and the loose bound makes our claim harder to attack.
- The literature's neuromorphic comparison (e.g., Levy & Calvert 2021, "Communication consumes 35 times more energy than computation in the human cortex," *PNAS* 118(18)) suggests that *neural communication* is the real bottleneck, not neuron compute — this reinforces why Phase 2 (activation sparsity, i.e., silencing most neurons) maps onto the largest fraction of the brain's efficiency.

## What this comparison is NOT

1. **Not a claim the brain generates English tokens.** Next-token prediction is a transformer concept; brains produce language via a fundamentally different process.
2. **Not a thermodynamic lower bound.** The Landauer limit at body temperature is ~3·10⁻²¹ J/bit — much smaller still. The brain operates many orders of magnitude above Landauer; GPUs many orders above the brain.
3. **Not a claim that closing this gap is possible on current silicon.** A 3090 in dense fp16 is ~350 W delivering ~30 tok/s on Llama-3.2-1B ≈ **~12 J/token** — about 6× the brain figure. The point of sparsefire is to measure how much of that 6× each biomimetic trick actually closes.

## Citations

- Attwell, D., & Laughlin, S. B. (2001). An energy budget for signaling in the grey matter of the brain. *Journal of Cerebral Blood Flow & Metabolism*, 21(10), 1133–1145. https://doi.org/10.1097/00004647-200110000-00001
- Levy, W. B., & Calvert, V. G. (2021). Communication consumes 35 times more energy than computation in the human cortex. *PNAS*, 118(18), e2008173118. https://doi.org/10.1073/pnas.2008173118
- Hasson, U., Chen, J., & Honey, C. J. (2015). Hierarchical process memory: memory as an integral component of information processing. *Trends in Cognitive Sciences*, 19(6), 304–313 — context for language-specific cortical budgets.
