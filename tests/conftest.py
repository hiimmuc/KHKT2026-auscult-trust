"""Test collection rules shared by every test folder."""

import os
import sys

# Headless plotting: an inherited Jupyter inline backend would break matplotlib in subprocesses.
os.environ["MPLBACKEND"] = "Agg"

# The campaign drivers in scripts/ lock their work with fcntl, which exists only on POSIX systems.
# Their tests run on Linux (and WSL2); on native Windows they are not collected.
collect_ignore_glob = ["infrastructure/test_*campaign*.py"] if sys.platform == "win32" else []
