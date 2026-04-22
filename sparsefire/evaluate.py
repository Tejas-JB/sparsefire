"""Accuracy evaluation: perplexity on WikiText-2 and HellaSwag 0-shot via lm-eval."""

from __future__ import annotations


def perplexity_wikitext2(
    model, tokenizer, split: str = "wikitext-2-raw-v1", stride: int = 512, max_length: int = 2048
) -> float:
    """Sliding-window perplexity on WikiText-2 test set."""
    import math

    import torch
    from datasets import load_dataset

    ds = load_dataset("wikitext", split, split="test")
    text = "\n\n".join(x["text"] for x in ds)
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids.to(model.device)

    nlls = []
    seq_len = input_ids.size(1)
    prev_end = 0
    for begin in range(0, seq_len, stride):
        end = min(begin + max_length, seq_len)
        trg_len = end - prev_end
        ids = input_ids[:, begin:end]
        target = ids.clone()
        target[:, :-trg_len] = -100
        with torch.no_grad():
            out = model(ids, labels=target)
        nlls.append(out.loss.float() * trg_len)
        prev_end = end
        if end == seq_len:
            break
    return math.exp((torch.stack(nlls).sum() / end).item())


def hellaswag_0shot(
    model,
    tokenizer,
    batch_size: int = 8,
    device: str = "cuda:0",
) -> dict:
    """Run HellaSwag 0-shot via lm_eval using the actual loaded model."""
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
