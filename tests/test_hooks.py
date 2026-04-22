"""Unit tests for sparsefire.hooks. Runs on CPU with tiny synthetic tensors."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from sparsefire.hooks import sparse_attention, sparse_mlp_hooks


class _FakeMLP(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.down_proj = nn.Linear(d, d, bias=False)

    def forward(self, x):
        return self.down_proj(x)


class _FakeLayer(nn.Module):
    def __init__(self, d: int = 8):
        super().__init__()
        self.mlp = _FakeMLP(d)


class _FakeModel(nn.Module):
    def __init__(self, n_layers: int = 2, d: int = 8):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_FakeLayer(d) for _ in range(n_layers)])


def test_sparse_mlp_hook_zeros_below_threshold():
    model = _FakeModel(n_layers=2, d=4)
    thresholds = {0: 0.5, 1: 0.5}
    x = torch.tensor([[0.1, 0.9, -0.3, 0.7]])
    captured: list[torch.Tensor] = []

    # Sparse hooks register first (inside context), then capture registers after —
    # hooks run in registration order, so capture sees the zeroed tensor.
    with sparse_mlp_hooks(model, thresholds):
        h = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
            lambda m, a: captured.append(a[0].clone()) or None
        )
        try:
            _ = model.model.layers[0].mlp(x)
        finally:
            h.remove()

    expected = torch.tensor([[0.0, 0.9, 0.0, 0.7]])
    assert torch.allclose(captured[0], expected)


def test_sparse_mlp_hook_cleans_up():
    model = _FakeModel(n_layers=1, d=4)
    thresholds = {0: 0.5}
    x = torch.tensor([[0.1, 0.9, -0.3, 0.7]])
    with sparse_mlp_hooks(model, thresholds):
        pass
    # After context exit, hooks should be removed — feeding small values should survive.
    captured: list[torch.Tensor] = []
    h = model.model.layers[0].mlp.down_proj.register_forward_pre_hook(
        lambda m, a: captured.append(a[0].clone()) or None
    )
    try:
        _ = model.model.layers[0].mlp(x)
    finally:
        h.remove()
    assert torch.allclose(captured[0], x)


def test_sparse_attention_topk_shape():
    # 4-D attention weight tensor: [batch, heads, q, k]
    logits = torch.randn(1, 2, 3, 16)
    with sparse_attention(top_k_frac=0.25, preserve_first_token=True):
        w = F.softmax(logits, dim=-1)
    # 25% of 16 = 4 entries survive + first-token pin (may already be in top-k)
    nonzero_per_row = (w > 0).sum(dim=-1)
    # Each row: at least 4, at most 5 (top-4 plus first-token if not already in top-4)
    assert nonzero_per_row.min().item() >= 4
    assert nonzero_per_row.max().item() <= 5


def test_sparse_attention_preserves_first_token():
    logits = torch.full((1, 1, 1, 10), -10.0)
    logits[..., 5] = 10.0  # token 5 would dominate
    with sparse_attention(top_k_frac=0.1, preserve_first_token=True):
        w = F.softmax(logits, dim=-1)
    assert w[0, 0, 0, 0].item() > 0.0  # first token pinned non-zero


def test_sparse_attention_renormalizes():
    logits = torch.randn(1, 2, 4, 8)
    with sparse_attention(top_k_frac=0.5):
        w = F.softmax(logits, dim=-1)
    sums = w.sum(dim=-1)
    assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


def test_sparse_attention_passes_through_non_4d():
    """F.softmax on 2-D tensors (e.g., classifier heads) must be unaffected."""
    logits = torch.randn(4, 10)
    with sparse_attention(top_k_frac=0.1):
        w = F.softmax(logits, dim=-1)
    # All entries should be >0 (standard softmax), no top-k mask applied
    assert (w > 0).all()


def test_sparse_attention_restores_softmax():
    original = F.softmax
    with sparse_attention(top_k_frac=0.5):
        pass
    assert F.softmax is original
