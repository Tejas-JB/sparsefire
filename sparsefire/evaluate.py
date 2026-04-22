"""Accuracy evaluation: perplexity on WikiText-2 and HellaSwag 0-shot via lm-eval."""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path


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
    """Run `lm_eval` as subprocess, parse results JSON, return {acc, acc_norm}."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "results.json"
        model_args = f"pretrained={model_path},attn_implementation={attn_impl}"
        if extra_model_args:
            model_args = f"{model_args},{extra_model_args}"
        cmd = [
            "lm_eval",
            "--model",
            "hf",
            "--model_args",
            model_args,
            "--tasks",
            "hellaswag",
            "--num_fewshot",
            "0",
            "--batch_size",
            str(batch_size),
            "--device",
            device,
            "--output_path",
            str(out_path),
        ]
        subprocess.run(cmd, check=True)
        # lm-eval writes a nested structure; glob for the actual file
        files = list(Path(tmp).rglob("*.json"))
        if not files:
            raise RuntimeError("lm_eval produced no output")
        data = json.loads(files[0].read_text())
        hs = data["results"]["hellaswag"]
        return {
            "acc": float(hs.get("acc,none", hs.get("acc"))),
            "acc_norm": float(hs.get("acc_norm,none", hs.get("acc_norm"))),
        }
