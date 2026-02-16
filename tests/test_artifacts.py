from __future__ import annotations

import json
from pathlib import Path

from xrfv2_edge_tal.artifacts import create_run_dir, write_metrics

REQUIRED_FILES = {
    "resolved_config.yaml",
    "env.json",
    "git.json",
    "command.txt",
}


def test_create_run_dir_and_metrics(tmp_path: Path) -> None:
    cfg = {"seed": 7, "model": {"name": "tiny_tcn"}}
    run_dir = create_run_dir(tmp_path / "runs", cfg, "xrfv2-edge-tal train --seed 7")

    assert run_dir.exists()
    existing = {p.name for p in run_dir.iterdir()}
    assert REQUIRED_FILES.issubset(existing)

    metrics_path = write_metrics(run_dir, {"loss": 0.5, "map": 0.12})
    assert metrics_path.exists()

    payload = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert payload["loss"] == 0.5
    assert payload["map"] == 0.12
