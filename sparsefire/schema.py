"""Results JSON schema loader + validator."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

_SCHEMA_PATH = Path(__file__).parent.parent / "docs" / "results_schema.json"


@lru_cache(maxsize=1)
def load_schema() -> dict:
    return json.loads(_SCHEMA_PATH.read_text())


def validate(result: dict) -> None:
    """Raise jsonschema.ValidationError if the result dict is non-conforming."""
    import jsonschema  # lazy to keep import-time cheap

    jsonschema.validate(instance=result, schema=load_schema())
