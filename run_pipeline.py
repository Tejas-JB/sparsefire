#!/usr/bin/env python
"""Entrypoint: `python run_pipeline.py --phase N | --all | --cliff | --smoke`."""
from __future__ import annotations

import sys

from sparsefire.cli import main

if __name__ == "__main__":
    sys.exit(main())
