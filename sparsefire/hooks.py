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
    """
    handles = []
    try:
        for i, layer in enumerate(model.model.layers):
            t = thresholds[i]

            def make_hook(threshold: float):
                def pre_hook(_mod, args):
                    x = args[0]
                    mask = x.abs() > threshold
                    return (x * mask,) + args[1:]

                return pre_hook

            handles.append(layer.mlp.down_proj.register_forward_pre_hook(make_hook(t)))
        yield
    finally:
        for h in handles:
            h.remove()


@contextmanager
def sparse_attention(top_k_frac: float, preserve_first_token: bool = True):
    """Monkeypatch F.softmax with a top-k-then-renormalize variant.

    Only activates on 4-D tensors (attention-weight shape). Other softmax calls pass through.
    """
    import torch
    import torch.nn.functional as F

    original = F.softmax

    def patched(x: torch.Tensor, dim: int = -1, **kw):
        w = original(x, dim=dim, **kw)
        if w.ndim != 4:
            return w
        k = max(1, int(w.shape[-1] * top_k_frac))
        topk, _ = w.topk(k, dim=-1)
        threshold = topk[..., -1, None]
        mask = w >= threshold
        if preserve_first_token:
            mask[..., 0] = True
        w_sparse = w * mask
        return w_sparse / w_sparse.sum(dim=-1, keepdim=True).clamp_min(1e-9)

    F.softmax = patched
    try:
        yield
    finally:
        F.softmax = original
