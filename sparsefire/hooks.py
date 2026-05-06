"""Hook context managers for activation and attention sparsity.

See docs/architecture.md §Hook registration pattern and docs/research_notes.md §1,5
for justification of hook sites.
"""

from __future__ import annotations

from contextlib import contextmanager


@contextmanager
def sparse_mlp_hooks(model, thresholds: dict[int, float]):
    """Zero `down_proj` inputs (the gate*up product) with magnitude < threshold[layer_idx].

    Matches TEAL's hook site for activation sparsity.
    Uses in-place masked_fill_ to minimize temporary tensor allocations.
    """
    handles = []
    try:
        for i, layer in enumerate(model.model.layers):
            t = thresholds[i]

            def make_hook(threshold: float):
                def pre_hook(_mod, args):
                    x = args[0]
                    out = x.clone()
                    out.masked_fill_(x.abs() <= threshold, 0)
                    return (out,) + args[1:]

                return pre_hook

            handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(t)))
        yield
    finally:
        for h in handles:
            h.remove()


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
