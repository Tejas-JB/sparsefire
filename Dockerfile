FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        git curl ca-certificates build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python3

RUN pip install --upgrade pip uv

WORKDIR /workspace

# Copy dependency manifest first to maximize layer cache
COPY pyproject.toml README.md /workspace/

# Install core deps (no sparsefire package yet; installed later with source)
RUN uv pip install --system \
        "torch==2.5.*" \
        "transformers==4.47.1" \
        "accelerate>=1.0" \
        "datasets>=3.0" \
        "autoawq>=0.2.7" \
        "nvidia-ml-py>=12.555" \
        "lm-eval>=0.4.5" \
        "numpy>=1.26" \
        "matplotlib>=3.9" \
        "jsonschema>=4.23" \
        "pyyaml>=6.0" \
        "tqdm>=4.66"

COPY . /workspace
RUN uv pip install --system -e .

# Default entrypoint; override via `docker compose run`
CMD ["python", "-c", "import torch; print('CUDA:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"]
