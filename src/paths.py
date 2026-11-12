"""Repo-relative locations, independent of the working directory.

`DATA` is `$AT_DATA` or the `data/` folder next to the repo (the wrapper directory); `THIRD_PARTY` holds the vendored
upstream code (see `third_party/README.md`).
"""

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("AT_DATA", REPO.parent / "data"))
THIRD_PARTY = REPO / "third_party"
OUTPUTS = REPO / "outputs"
CHECKPOINTS = REPO / "checkpoints"
