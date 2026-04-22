"""Deterministic prompt loader for measurement runs.

Draws n_prompts texts from WikiText-2 test split (seed=0), truncates each to
the first ~32 tokens so generation has room for n_tokens more.
"""

from __future__ import annotations


def load_prompts(n_prompts: int = 50, seed: int = 0, split: str = "wikitext-2-raw-v1") -> list[str]:
    import random

    from datasets import load_dataset

    ds = load_dataset("wikitext", split, split="test")
    texts = [x["text"] for x in ds if len(x["text"].split()) >= 16]
    rng = random.Random(seed)
    rng.shuffle(texts)
    return texts[:n_prompts]
