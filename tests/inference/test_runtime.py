import json
from types import SimpleNamespace

import pytest

from auscult_trust.models.runtime import load_runtime, runtime_config


def test_runtime_config_round_trips_and_rejects_unknown_schema(tmp_path):
    path = tmp_path / "runtime.json"
    path.write_text(json.dumps(runtime_config(SimpleNamespace(encoder="beats", n_cls=4), beats_config={"a": 1})))
    args = load_runtime(path)
    assert (args.encoder, args.n_cls, args.beats_config) == ("beats", 4, {"a": 1})
    path.write_text(json.dumps({"schema": 99, "args": {}}))
    with pytest.raises(ValueError, match="schema"):
        load_runtime(path)
