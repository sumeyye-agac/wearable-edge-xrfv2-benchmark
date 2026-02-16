"""Run artifact helpers for reproducibility."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def _short_config_hash(config_dict: dict[str, Any]) -> str:
    payload = json.dumps(config_dict, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha1(payload).hexdigest()[:8]


def _safe_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_resolved_config(run_dir: Path, config_dict: dict[str, Any]) -> Path:
    path = run_dir / "resolved_config.yaml"
    path.write_text(yaml.safe_dump(config_dict, sort_keys=False), encoding="utf-8")
    return path


def write_env_info(run_dir: Path) -> Path:
    path = run_dir / "env.json"
    payload: dict[str, Any] = {
        "python": sys.version,
        "platform": platform.platform(),
        "python_executable": sys.executable,
    }
    _safe_write_json(path, payload)
    return path


def _run_git(args: list[str]) -> tuple[bool, str]:
    try:
        out = subprocess.check_output(["git", *args], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return False, ""
    return True, out


def write_git_info(run_dir: Path) -> Path:
    path = run_dir / "git.json"

    ok_sha, sha = _run_git(["rev-parse", "HEAD"])
    ok_branch, branch = _run_git(["rev-parse", "--abbrev-ref", "HEAD"])
    ok_dirty, dirty_raw = _run_git(["status", "--porcelain"])

    payload: dict[str, Any] = {
        "available": bool(ok_sha or ok_branch),
        "commit": sha if ok_sha else None,
        "branch": branch if ok_branch else None,
        "dirty": bool(dirty_raw) if ok_dirty else None,
    }
    _safe_write_json(path, payload)
    return path


def write_metrics(run_dir: Path, metrics: dict[str, Any]) -> Path:
    path = run_dir / "metrics.json"
    _safe_write_json(path, metrics)
    return path


def create_run_dir(base_dir: str | Path, config_dict: dict[str, Any], command_str: str) -> Path:
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    run_name = f"{_utc_timestamp()}_{_short_config_hash(config_dict)}"
    run_dir = base / run_name
    suffix = 1
    while run_dir.exists():
        run_dir = base / f"{run_name}_{suffix}"
        suffix += 1
    run_dir.mkdir(parents=True, exist_ok=False)

    write_resolved_config(run_dir, config_dict)
    write_env_info(run_dir)
    write_git_info(run_dir)
    (run_dir / "command.txt").write_text(command_str + "\n", encoding="utf-8")
    return run_dir
