# sparsefire Energy Benchmarks

Measure joules-per-token for LLM inference using NVIDIA GPU energy counters.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run benchmark (requires NVIDIA GPU)
python measure_joules_per_token.py --prompt "Hello world" --num-tokens 1000
```

Results saved to `results/<timestamp>.json` and `results/aggregate.csv`.

## Methodology

### Energy Measurement Approach

This benchmark uses **pynvml** (NVIDIA Management Library Python bindings) to read hardware energy counters directly from the GPU. This is the most accurate method available for measuring GPU inference energy consumption.

**Why pynvml:**
- Direct hardware counter access (no estimation or profiling overhead)
- Millijoule precision
- Supported on all modern NVIDIA GPUs
- Industry standard for energy measurement

### What We Measure

**GPU inference energy only:**
- Prompt processing phase (prefill): Energy to process input tokens in parallel
- Token generation phase (decode): Energy per generated token (sequential)

**Controlled variables:**
- Batch size: 1 (single sequence)
- Sequence length: user-specified
- Temperature: 0 (deterministic sampling)
- Model: Llama-3.2-1B (default)

**Reproducibility metadata:**
- GPU model and driver version
- CUDA version
- Model checkpoint identifier
- Timestamp
- Statistical variance across runs

### What We Don't Measure

- Training energy (inference only)
- Multi-GPU setups (single GPU only)
- CPU-only inference (requires NVIDIA GPU)
- System-level energy (GPU only, not CPU/RAM/disk)

**Why narrow scope:** Each measurement context requires different approaches. Clear, focused methodology is more valuable than broad coverage.

### Variance Handling

- **Warmup run:** Initializes GPU state before measurement
- **Multiple runs:** Default 5 runs per benchmark
- **Statistical analysis:** Reports mean, standard deviation, coefficient of variation
- **Variance warning:** Flags if CV% > 5%

**Variance sources:**
- GPU thermal state
- Background processes
- Driver scheduling
- Memory allocation patterns

## Usage

### Basic Usage

```bash
# Default: 1000 tokens, 5 runs
python measure_joules_per_token.py

# Custom prompt and token count
python measure_joules_per_token.py --prompt "Explain quantum computing" --num-tokens 500

# More runs for lower variance
python measure_joules_per_token.py --num-runs 10

# Specify output file
python measure_joules_per_token.py --output my_benchmark.json
```

### Command-Line Options

```
--prompt TEXT          Input prompt for inference (default: "The quick brown fox...")
--num-tokens INT       Number of tokens to generate (default: 1000)
--num-runs INT         Number of measurement runs (default: 5)
--model TEXT           HuggingFace model name (default: meta-llama/Llama-3.2-1B)
--output PATH          Output JSON file path (default: results/<timestamp>.json)
--test                 Run basic test mode
```

### Output Format

**JSON (per benchmark):**

```json
{
  "model": "meta-llama/Llama-3.2-1B",
  "gpu": "NVIDIA RTX 4090",
  "driver_version": "535.129.03",
  "cuda_version": "12.2",
  "timestamp": "2026-05-28T10:30:00Z",
  "prompt": "The quick brown fox",
  "prompt_tokens": 5,
  "generated_tokens": 995,
  "total_tokens": 1000,
  "prompt_phase_joules": 12.3,
  "generation_phase_joules": 32.9,
  "total_energy_joules_mean": 45.2,
  "total_energy_joules_stdev": 0.8,
  "joules_per_token_overall_mean": 0.0452,
  "joules_per_token_overall_stdev": 0.0008,
  "joules_per_token_generation_mean": 0.0330,
  "joules_per_token_generation_stdev": 0.0007,
  "coefficient_of_variation": 1.77,
  "num_runs": 5
}
```

**CSV (aggregate across benchmarks):**

```csv
timestamp,model,gpu,total_tokens,total_energy_joules_mean,joules_per_token_overall_mean,coefficient_of_variation,num_runs
2026-05-28T10:30:00Z,Llama-3.2-1B,RTX-4090,1000,45.2,0.0452,1.77,5
```

## Requirements

### Hardware

- NVIDIA GPU with energy counter support (most modern GPUs)
- 8GB+ GPU memory for Llama-3.2-1B

### Software

- Linux (pynvml driver requirement)
- Python 3.10+
- CUDA 11.8+ (12.0+ recommended)

**Note:** pynvml is not consistently supported on Windows or macOS for energy counters. Use Linux for best results.

### Dependencies

```
pynvml>=11.5.0
torch>=2.0.0
transformers>=4.30.0
```

Install with: `pip install -r requirements.txt`

## Known Limitations

1. **Single GPU only:** Multi-GPU measurement requires different approach
2. **Linux only:** pynvml energy counters not consistently available on other platforms
3. **NVIDIA only:** Requires NVIDIA GPU (no AMD/Intel support)
4. **GPU-only energy:** Does not measure CPU, RAM, or disk energy
5. **No system overhead:** Does not account for OS or background process energy
6. **Inference only:** Does not measure training energy

These are deliberate scope limitations. Each would require different measurement methodology.

## Reproducibility Checklist

To reproduce results:

- [ ] GPU model matches (e.g., RTX 4090)
- [ ] Driver version matches or is close (e.g., 535.x)
- [ ] CUDA version matches major version (e.g., 12.x)
- [ ] Same model checkpoint (e.g., Llama-3.2-1B from HuggingFace)
- [ ] Same prompt and token count
- [ ] Same number of warmup and measurement runs
- [ ] GPU not under heavy load from background processes

**Expected variance:** ±5% coefficient of variation is normal. Higher variance suggests GPU load or thermal effects.

## Troubleshooting

**Error: "pynvml not installed"**

```bash
pip install pynvml
```

**Error: "CUDA not available"**

Install PyTorch with CUDA support:
```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

**Error: "GPU does not support energy counters"**

Your GPU may not support energy monitoring. Check NVIDIA documentation for your GPU model.

**Warning: "High variance (>5%)"**

- Close background GPU applications
- Run more iterations (--num-runs 10)
- Wait for GPU to reach stable thermal state
- Check for background processes with `nvidia-smi`

**Out of memory error**

Reduce token count or use smaller model:
```bash
python measure_joules_per_token.py --num-tokens 100
```

## Future Work

- Multi-GPU support
- System-level energy measurement (CPU + RAM + disk)
- Support for AMD/Intel GPUs
- Comparison against other models (Llama-2, GPT variants)
- Training energy measurement
- Blog post with detailed methodology writeup

## License

MIT License - see repository root LICENSE file

## Citation

If you use this benchmark in research, please cite:

```
@software{sparsefire2026,
  author = {Bhat, Tejas},
  title = {sparsefire: LLM Inference Energy Benchmarks},
  year = {2026},
  url = {https://github.com/<username>/sparsefire}
}
```

## Contact

Tejas Bhat - tejasjb@gmail.com

Part of the NeuroAI research initiative.
