"""Phase 6 — waterfall attribution chart, cliff plot, and neuron-firing animation."""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def make_waterfall(results_dir: Path, out_path: Path | None = None) -> Path:
    """Build a waterfall attribution chart from phase results JSONs.

    Shows J/tok at each phase with delta labels and brain-equivalent target.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    results_dir = Path(results_dir)
    out_path = out_path or results_dir / "attribution_chart.png"

    # Collect phase results
    phases = []

    # Baseline
    p0 = results_dir / "phase0_baseline.json"
    if p0.exists():
        d = json.loads(p0.read_text())
        phases.append(("Baseline\n(dense fp16)", d["energy"]["joules_per_token"]["mean"]))

    # KV cache off (if exists, show the "without cache" reference)
    kv_off = results_dir / "phase1_kvcache_cache_off.json"
    if kv_off.exists():
        d = json.loads(kv_off.read_text())
        phases.append(("No KV cache", d["energy"]["joules_per_token"]["mean"]))

    # KV cache on
    kv_on = results_dir / "phase1_kvcache_cache_on.json"
    if kv_on.exists():
        d = json.loads(kv_on.read_text())
        phases.append(("+KV cache\n(working memory)", d["energy"]["joules_per_token"]["mean"]))

    # Best activation sparsity
    sp40 = results_dir / "phase2_actsparse_50.json"
    if sp40.exists():
        d = json.loads(sp40.read_text())
        phases.append(
            ("+Act. sparsity\n50% (neural silence)", d["energy"]["joules_per_token"]["mean"])
        )

    # Attention sparsity (best = top-30%)
    attn = results_dir / "phase4_attn_topk30.json"
    if attn.exists():
        d = json.loads(attn.read_text())
        phases.append(
            ("+Attn. sparsity\ntop-30% (sel. attn)", d["energy"]["joules_per_token"]["mean"])
        )

    if not phases:
        raise RuntimeError("No phase results found in " + str(results_dir))

    labels, values = zip(*phases, strict=True)
    x = np.arange(len(labels))
    colors = ["#2196F3", "#f44336", "#4CAF50", "#FF9800", "#9C27B0"]

    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(
        x, values, color=colors[: len(values)], width=0.6, edgecolor="white", linewidth=1.5
    )

    # Brain equivalent line
    brain_jpt = 2.0
    ax.axhline(y=brain_jpt, color="#E91E63", linestyle="--", linewidth=2, alpha=0.7)
    ax.text(
        len(labels) - 0.5,
        brain_jpt + 0.05,
        "Brain ~2 J/tok\n(approximate)",
        color="#E91E63",
        fontsize=9,
        ha="right",
        va="bottom",
        fontstyle="italic",
    )

    # Value labels on bars
    for bar, val in zip(bars, values, strict=False):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.03,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=10,
        )

    # Delta labels
    for i in range(1, len(values)):
        delta_pct = (values[i] - values[i - 1]) / values[i - 1] * 100
        sign = "+" if delta_pct > 0 else ""
        color = "#f44336" if delta_pct > 0 else "#4CAF50"
        ax.annotate(
            f"{sign}{delta_pct:.1f}%",
            xy=(i, values[i]),
            xytext=(i - 0.3, max(values[i], values[i - 1]) + 0.15),
            fontsize=8,
            color=color,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Energy (J/token)", fontsize=12)
    ax.set_title("sparsefire: Energy Attribution by Phase\n(Llama-3.2-1B on RTX 3060)", fontsize=14)
    ax.set_ylim(0, max(values) * 1.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Waterfall chart saved to %s", out_path)
    return out_path


def make_cliff_plot(cliff_json: Path, out_path: Path | None = None) -> Path:
    """Plot sparsity vs perplexity cliff curve."""
    import matplotlib.pyplot as plt

    data = json.loads(cliff_json.read_text())
    sparsity = data["sparsity_levels"]
    ppl = data["perplexity"]

    out_path = out_path or cliff_json.parent / "cliff.png"

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(sparsity, ppl, "o-", color="#2196F3", linewidth=2, markersize=8)
    ax.fill_between(sparsity, ppl, alpha=0.1, color="#2196F3")

    # Mark the "free zone" (< 5% PPL increase)
    baseline_ppl = ppl[0]
    threshold_ppl = baseline_ppl * 1.05
    ax.axhline(y=threshold_ppl, color="#4CAF50", linestyle="--", alpha=0.5, linewidth=1.5)
    ax.text(5, threshold_ppl + 0.2, "5% degradation threshold", color="#4CAF50", fontsize=9)

    # Annotate each point
    for s, p in zip(sparsity, ppl, strict=True):
        delta = (p - baseline_ppl) / baseline_ppl * 100
        if delta > 5:
            ax.annotate(
                f"PPL={p:.1f}\n(+{delta:.0f}%)",
                (s, p),
                textcoords="offset points",
                xytext=(10, 10),
                fontsize=7,
                color="#f44336",
            )

    ax.set_xlabel("Activation Sparsity (%)", fontsize=12)
    ax.set_ylabel("WikiText-2 Perplexity", fontsize=12)
    ax.set_title(
        "The Cliff: Activation Sparsity vs Accuracy\n(Llama-3.2-1B, TEAL-style on down_proj)",
        fontsize=14,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Cliff plot saved to %s", out_path)
    return out_path


def make_neuron_firing_video(
    model,
    tokenizer,
    thresholds_by_sparsity: dict[float, dict[int, float]],
    out_path: Path,
    layer_idx: int = 8,
    grid_shape: tuple[int, int] = (32, 32),
    fps: int = 5,
    prompt: str = "The brain is remarkably efficient at processing information",
) -> Path:
    """Create neuron firing animation showing activations at increasing sparsity.

    Each frame = one sparsity level. Color = activation magnitude.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    import torch

    logger.info("Capturing activations for neuron firing video (layer %d)", layer_idx)
    n_neurons = grid_shape[0] * grid_shape[1]

    # Capture activations at each sparsity level
    frames = []
    sparsity_labels = []

    ids = tokenizer(prompt, return_tensors="pt").to(model.device)

    for sp_level in sorted(thresholds_by_sparsity.keys()):
        thresholds = thresholds_by_sparsity[sp_level]
        captured = []

        def capture_hook(module, args, _captured=captured):
            x = args[0]
            _captured.append(x.detach().cpu())

        handle = model.model.layers[layer_idx].mlp.down_proj.register_forward_pre_hook(capture_hook)

        if sp_level == 0.0:
            with torch.no_grad():
                model(**ids)
        else:
            from .hooks import sparse_mlp_hooks

            with sparse_mlp_hooks(model, thresholds), torch.no_grad():
                model(**ids)

        handle.remove()

        if captured:
            act = captured[0].squeeze()  # (seq_len, hidden_dim)
            # Take last token's activations, truncate to grid size
            neuron_vals = act[-1, :n_neurons].abs().numpy()
            frames.append(neuron_vals.reshape(grid_shape))
            sparsity_labels.append(f"{int(sp_level * 100)}%")

    if not frames:
        raise RuntimeError("No activations captured")

    # Normalize across all frames
    vmax = max(f.max() for f in frames)

    # Build animation
    fig, ax = plt.subplots(figsize=(8, 8))
    img = ax.imshow(frames[0], cmap="hot", vmin=0, vmax=vmax, interpolation="nearest")
    title = ax.set_title("Neuron Firing — 0% sparsity", fontsize=16)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.colorbar(img, ax=ax, label="Activation magnitude", shrink=0.8)

    # Each sparsity level gets multiple frames for smooth viewing
    frames_per_level = max(1, fps * 3)  # 3 seconds per level
    total_frames = len(frames) * frames_per_level

    def update(frame_idx):
        level_idx = min(frame_idx // frames_per_level, len(frames) - 1)
        img.set_data(frames[level_idx])
        title.set_text(f"Neuron Firing — {sparsity_labels[level_idx]} sparsity")
        return [img, title]

    ani = animation.FuncAnimation(fig, update, frames=total_frames, interval=1000 // fps, blit=True)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as gif (mp4 needs ffmpeg which may not be installed)
    gif_path = out_path.with_suffix(".gif")
    ani.save(str(gif_path), writer="pillow", fps=fps)
    plt.close(fig)
    logger.info("Neuron firing animation saved to %s", gif_path)
    return gif_path
