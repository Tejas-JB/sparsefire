.PHONY: install lint test smoke phase0 phase1 phase2 phase3 phase4 phase5 cliff all clean build-gpu shell-gpu

PYTHON ?= python
UV ?= uv

install:
	$(UV) pip install -e ".[dev]"

lint:
	ruff check sparsefire tests
	ruff format --check sparsefire tests

fmt:
	ruff format sparsefire tests
	ruff check --fix sparsefire tests

test:
	pytest

# Phase-specific targets (each runs inside docker on GPU host)
smoke:
	$(PYTHON) -m sparsefire.cli --smoke

phase0:
	$(PYTHON) run_pipeline.py --phase 0

phase1:
	$(PYTHON) run_pipeline.py --phase 1

phase2:
	$(PYTHON) run_pipeline.py --phase 2 --sparsity 0.25
	$(PYTHON) run_pipeline.py --phase 2 --sparsity 0.40
	$(PYTHON) run_pipeline.py --phase 2 --sparsity 0.50
	$(PYTHON) run_pipeline.py --phase 2 --sparsity 0.70

phase3:
	$(PYTHON) run_pipeline.py --phase 3

phase4:
	$(PYTHON) run_pipeline.py --phase 4 --top-k-frac 0.5
	$(PYTHON) run_pipeline.py --phase 4 --top-k-frac 0.3

cliff:
	$(PYTHON) run_pipeline.py --cliff

all:
	$(PYTHON) run_pipeline.py --all

# Docker helpers (run on the 3090 host)
build-gpu:
	docker compose build

shell-gpu:
	docker compose run --rm gpu bash

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__
