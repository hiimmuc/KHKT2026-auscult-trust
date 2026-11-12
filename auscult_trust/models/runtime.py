"""Runtime configuration of a released model: the fully resolved training arguments, versioned and frozen.

A released model loads this file instead of rebuilding the arguments from the current training parser, so later changes of
training defaults cannot alter an old model.
"""

import json
from types import SimpleNamespace

SCHEMA = 1


def runtime_config(args, **extra):
    """JSON-serialisable runtime configuration from resolved training `args`."""
    return {"schema": SCHEMA, "args": {**vars(args), **extra}}


def load_runtime(path):
    """Runtime arguments as a namespace; rejects an unknown schema."""
    config = json.loads(open(path).read())
    if config.get("schema") != SCHEMA:
        raise ValueError(
            f"Runtime configuration {path} has schema {config.get('schema')}, expected {SCHEMA}"
        )
    return SimpleNamespace(**config["args"])
