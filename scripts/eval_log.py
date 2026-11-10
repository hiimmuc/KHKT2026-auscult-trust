"""JSONL event log and logged subprocess runner shared by the frozen-evaluation scripts."""
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def logger(log_dir):
    """Return `(event, run)` that append to `<log_dir>/events.jsonl`."""
    log = Path(log_dir)

    def event(**data):
        log.mkdir(parents=True, exist_ok=True)
        data["utc"] = datetime.now(timezone.utc).isoformat()
        with (log / "events.jsonl").open("a") as f:
            f.write(json.dumps(data) + "\n")
        print(json.dumps(data), flush=True)

    def run(args):
        cmd = [sys.executable, *map(str, args)]
        event(status="started", command=shlex.join(cmd))
        result = subprocess.run(cmd)
        event(status="passed" if result.returncode == 0 else "failed", command=shlex.join(cmd), exit_code=result.returncode)
        result.check_returncode()

    return event, run
