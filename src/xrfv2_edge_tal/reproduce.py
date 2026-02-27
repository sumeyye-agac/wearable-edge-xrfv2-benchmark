"""Utilities for deterministic end-to-end reproduction runs."""

from __future__ import annotations

import re

RUN_DIR_RE = re.compile(r"run dir:\s*(runs/\S+)", re.IGNORECASE)


def extract_run_dir(output: str) -> str:
    """Extract `runs/<id>` from CLI output."""
    match = RUN_DIR_RE.search(output)
    if not match:
        raise ValueError(f"Could not extract run dir from output:\n{output}")
    return match.group(1)
