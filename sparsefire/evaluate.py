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
    model_path: str,
    batch_size: int = 8,
    device: str = "cuda:0",
    attn_impl: str = "eager",
    extra_model_args: str = "",
) -> dict:
    """Run HellaSwag 0-shot via lm_eval Python API. Returns {acc, acc_norm}."""
    import lm_eval

    model_args = f"pretrained={model_path},attn_implementation={attn_impl}"
    if extra_model_args:
        model_args = f"{model_args},{extra_model_args}"
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=["hellaswag"],
        num_fewshot=0,
        batch_size=batch_size,
        device=device,
    )
    hs = results["results"]["hellaswag"]
    return {
        "acc": float(hs.get("acc,none", hs.get("acc"))),
        "acc_norm": float(hs.get("acc_norm,none", hs.get("acc_norm"))),
    }
