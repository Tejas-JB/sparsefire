"""sparsefire CLI entrypoint. See `python run_pipeline.py --help`."""

from __future__ import annotations

import argparse
import sys

from .config import Config


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sparsefire", description=__doc__)
    g = p.add_mutually_exclusive_group()
    g.add_argument("--phase", type=int, choices=list(range(7)))
    g.add_argument("--all", action="store_true", help="run phases 0..6 sequentially")
    g.add_argument("--cliff", action="store_true", help="sparsity 0..0.99 sweep")
    g.add_argument("--smoke", action="store_true", help="minimal smoke test: load model + generate")
    p.add_argument("--sparsity", type=float, default=None, help="phase 2 target sparsity")
    p.add_argument("--top-k-frac", type=float, default=None, help="phase 4 top-k fraction")
    p.add_argument("--use-cache", action="store_true", help="phase 1 A/B knob")
    p.add_argument("--no-use-cache", action="store_true")
    p.add_argument("--n-runs", type=int, default=None)
    p.add_argument("--results-dir", type=str, default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    overrides: dict[str, object] = {}
    if args.n_runs is not None:
        overrides["n_runs"] = args.n_runs
    if args.results_dir is not None:
        overrides["results_dir"] = args.results_dir
    cfg = Config().override(**overrides)

    if args.smoke:
        _smoke(cfg)
        return 0

    if args.all:
        raise NotImplementedError("--all: implemented in Phase 6")
    if args.cliff:
        raise NotImplementedError("--cliff: implemented in Phase 2 follow-up")
    if args.phase is None:
        build_parser().print_help()
        return 2

    _run_phase(args.phase, cfg, args)
    return 0


def _run_phase(phase: int, cfg: Config, args) -> dict:
    """Dispatch to the appropriate phase module."""
    import logging

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if phase == 0:
        from . import baseline

        return baseline.run(cfg)
    if phase == 1:
        from . import kv_cache

        use_cache = not args.no_use_cache  # default True unless --no-use-cache
        return kv_cache.run(cfg, use_cache=use_cache)
    if phase == 2:
        from . import activation_sparsity

        sparsity = args.sparsity if args.sparsity is not None else 0.40
        return activation_sparsity.run(cfg, sparsity=sparsity)
    if phase == 3:
        from . import quantization

        return quantization.run(cfg, stack_sparsity=args.sparsity)
    if phase == 4:
        from . import attention_sparsity

        top_k = args.top_k_frac if args.top_k_frac is not None else 0.5
        return attention_sparsity.run(cfg, top_k_frac=top_k)
    raise NotImplementedError(f"Phase {phase} not yet implemented")


def _smoke(cfg: Config) -> None:
    """Minimal check: load the model and generate 10 tokens. Phase 1 Day 1 task."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.model_id)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model_id,
        torch_dtype=getattr(torch, cfg.dtype),
        attn_implementation=cfg.attn_impl,
        device_map=cfg.device,
    )
    inputs = tok("Hello, world.", return_tensors="pt").to(cfg.device)
    out = model.generate(**inputs, max_new_tokens=10, use_cache=True)
    print(tok.decode(out[0]))


if __name__ == "__main__":
    sys.exit(main())
