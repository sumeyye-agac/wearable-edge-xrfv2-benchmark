#!/usr/bin/env python3
"""Run a deterministic full train/eval/calibration pipeline and save a manifest."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime
from pathlib import Path

from xrfv2_edge_tal.reproduce import extract_run_dir


def _exec(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{stdout}\nstderr:\n{stderr}"
        )
    return stdout


def _git_head() -> str:
    proc = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return (proc.stdout or "").strip()
    return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/event_presence_mobility.yaml")
    parser.add_argument("--adapter", default="xrfv2", choices=["dummy", "xrfv2"])
    parser.add_argument("--data-root", default="data/raw/xrfv2_kaggle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-profile", default="earbuds_glasses")
    parser.add_argument("--profiles", default="earbuds_glasses,glasses_only")
    parser.add_argument("--train-device", default="auto")
    parser.add_argument("--eval-device", default="auto")
    parser.add_argument("--fp-hour-budget", type=float, default=10.0)
    parser.add_argument("--thresholds", default="0.70,0.75,0.80,0.835,0.85,0.90")
    parser.add_argument("--cooldowns", default="0.00,0.50,2.00")
    parser.add_argument(
        "--output",
        default="runs/repro_full_latest.json",
        help="Path to summary manifest JSON",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    train_cmd = [
        "xrfv2-edge-tal",
        "event-train",
        "--config",
        str(args.config),
        "--adapter",
        str(args.adapter),
        "--data-root",
        str(args.data_root),
        "--profile",
        str(args.train_profile),
        "--seed",
        str(int(args.seed)),
        "--override",
        "train.max_train_samples=0",
        "--override",
        f"runtime.device={args.train_device}",
    ]
    train_out = _exec(train_cmd)
    train_run = extract_run_dir(train_out)
    checkpoint = f"{train_run}/checkpoints/last.npz"

    eval_cmd = [
        "xrfv2-edge-tal",
        "event-eval",
        "--config",
        str(args.config),
        "--adapter",
        str(args.adapter),
        "--data-root",
        str(args.data_root),
        "--checkpoint",
        checkpoint,
        "--profiles",
        str(args.profiles),
        "--seed",
        str(int(args.seed)),
        "--override",
        "eval.max_eval_samples=0",
        "--override",
        f"runtime.device={args.eval_device}",
    ]
    eval_out = _exec(eval_cmd)
    eval_run = extract_run_dir(eval_out)

    cal_cmd = [
        "xrfv2-edge-tal",
        "event-calibrate",
        "--config",
        str(args.config),
        "--adapter",
        str(args.adapter),
        "--data-root",
        str(args.data_root),
        "--checkpoint",
        checkpoint,
        "--profiles",
        str(args.profiles),
        "--metric-mode",
        "sample_presence",
        "--fp-hour-budget",
        str(float(args.fp_hour_budget)),
        "--thresholds",
        str(args.thresholds),
        "--cooldowns",
        str(args.cooldowns),
        "--seed",
        str(int(args.seed)),
        "--override",
        "eval.max_eval_samples=0",
        "--override",
        f"runtime.device={args.eval_device}",
    ]
    cal_out = _exec(cal_cmd)
    cal_run = extract_run_dir(cal_out)

    manifest = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "git_commit": _git_head(),
        "config": str(args.config),
        "adapter": str(args.adapter),
        "data_root": str(args.data_root),
        "seed": int(args.seed),
        "train_profile": str(args.train_profile),
        "profiles": [item.strip() for item in str(args.profiles).split(",") if item.strip()],
        "commands": {
            "train": train_cmd,
            "eval": eval_cmd,
            "calibrate": cal_cmd,
        },
        "runs": {
            "train": train_run,
            "eval": eval_run,
            "calibrate": cal_run,
        },
        "checkpoint": checkpoint,
        "artifacts": {
            "train_metrics": f"{train_run}/metrics.json",
            "eval_metrics": f"{eval_run}/metrics.json",
            "profile_metrics": f"{eval_run}/profile_metrics.json",
            "profile_report": f"{eval_run}/profile_report.md",
            "calibration_metrics": f"{cal_run}/metrics.json",
            "calibration_grid": f"{cal_run}/calibration_grid.json",
            "calibration_report": f"{cal_run}/calibration_report.md",
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Mirror into the calibration run for full run-local traceability.
    repro_path = Path(cal_run) / "repro_manifest.json"
    repro_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Repro manifest: {output_path}")
    print(f"Train run: {train_run}")
    print(f"Eval run: {eval_run}")
    print(f"Calibration run: {cal_run}")


if __name__ == "__main__":
    main()
