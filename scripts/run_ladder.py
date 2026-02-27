#!/usr/bin/env python3
"""Run deploy-track experiment ladders and save a consolidated summary."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

RUN_DIR_RE = re.compile(r"run dir:\s*(runs/\S+)", re.IGNORECASE)


@dataclass
class LadderRun:
    name: str
    train_profile: str
    train_run_dir: str
    eval_run_dir: str
    calibrate_run_dir: str


def _exec(cmd: list[str]) -> tuple[str, str]:
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if proc.returncode != 0:
        message = (
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n"
            f"stdout:\n{stdout}\n\nstderr:\n{stderr}\n"
        )
        raise RuntimeError(message)
    return stdout, stderr


def _extract_run_dir(output: str) -> str:
    match = RUN_DIR_RE.search(output)
    if not match:
        raise RuntimeError(f"Could not extract run dir from output:\n{output}")
    return match.group(1)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_single_ladder(
    *,
    name: str,
    train_profile: str,
    config: str,
    adapter: str,
    data_root: str,
    seed: int,
    train_device: str,
    eval_device: str,
    event_mode: str,
    epochs: int,
    profiles_csv: str,
    calibrate_max_eval_samples: int,
    fp_hour_budget: float,
    calibrate_thresholds_csv: str,
    calibrate_cooldowns_csv: str,
) -> LadderRun:
    train_cmd = [
        "xrfv2-edge-tal",
        "event-train",
        "--config",
        config,
        "--adapter",
        adapter,
        "--data-root",
        data_root,
        "--profile",
        train_profile,
        "--seed",
        str(seed),
        "--override",
        f"train.epochs={epochs}",
        "--override",
        f"runtime.device={train_device}",
        "--override",
        f"train.event_mode={event_mode}",
        "--override",
        f"eval.event_mode={event_mode}",
    ]
    train_stdout, _ = _exec(train_cmd)
    train_run_dir = _extract_run_dir(train_stdout)
    checkpoint = f"{train_run_dir}/checkpoints/last.npz"

    eval_cmd = [
        "xrfv2-edge-tal",
        "event-eval",
        "--config",
        config,
        "--adapter",
        adapter,
        "--data-root",
        data_root,
        "--checkpoint",
        checkpoint,
        "--profiles",
        profiles_csv,
        "--seed",
        str(seed),
        "--override",
        f"runtime.device={eval_device}",
        "--override",
        f"train.event_mode={event_mode}",
        "--override",
        f"eval.event_mode={event_mode}",
    ]
    eval_stdout, _ = _exec(eval_cmd)
    eval_run_dir = _extract_run_dir(eval_stdout)

    calibrate_cmd = [
        "xrfv2-edge-tal",
        "event-calibrate",
        "--config",
        config,
        "--adapter",
        adapter,
        "--data-root",
        data_root,
        "--checkpoint",
        checkpoint,
        "--profiles",
        profiles_csv,
        "--metric-mode",
        "sample_presence",
        "--fp-hour-budget",
        str(fp_hour_budget),
        "--seed",
        str(seed),
        "--thresholds",
        calibrate_thresholds_csv,
        "--cooldowns",
        calibrate_cooldowns_csv,
        "--override",
        f"runtime.device={eval_device}",
        "--override",
        f"eval.max_eval_samples={calibrate_max_eval_samples}",
        "--override",
        f"train.event_mode={event_mode}",
        "--override",
        f"eval.event_mode={event_mode}",
    ]
    calibrate_stdout, _ = _exec(calibrate_cmd)
    calibrate_run_dir = _extract_run_dir(calibrate_stdout)

    return LadderRun(
        name=name,
        train_profile=train_profile,
        train_run_dir=train_run_dir,
        eval_run_dir=eval_run_dir,
        calibrate_run_dir=calibrate_run_dir,
    )


def _build_result_record(run: LadderRun) -> dict[str, Any]:
    train_metrics = _read_json(Path(run.train_run_dir) / "metrics.json")
    eval_metrics = _read_json(Path(run.eval_run_dir) / "profile_metrics.json")
    calibrate_metrics = _read_json(Path(run.calibrate_run_dir) / "metrics.json")
    return {
        "name": run.name,
        "train_profile": run.train_profile,
        "run_dirs": {
            "train": run.train_run_dir,
            "eval": run.eval_run_dir,
            "calibrate": run.calibrate_run_dir,
        },
        "train": {
            "epochs": int(train_metrics.get("train", {}).get("epochs", 0)),
            "final_loss": float(train_metrics.get("train", {}).get("final_loss", 0.0)),
        },
        "eval_profiles": eval_metrics,
        "calibration": calibrate_metrics.get("calibration", {}),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/event_presence_mobility.yaml")
    parser.add_argument("--adapter", default="xrfv2", choices=["dummy", "xrfv2"])
    parser.add_argument("--data-root", default="data/raw/xrfv2_kaggle")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-device", default="mps")
    parser.add_argument("--eval-device", default="cpu")
    parser.add_argument("--event-mode", default="hierarchical", choices=["flat", "hierarchical"])
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--profiles",
        default="wifi_all,all_imu,earbuds_glasses,glasses_only",
        help="Comma-separated evaluation/calibration profiles",
    )
    parser.add_argument("--calibrate-max-eval-samples", type=int, default=2048)
    parser.add_argument("--fp-hour-budget", type=float, default=10.0)
    parser.add_argument(
        "--calibrate-thresholds",
        default="0.70,0.75,0.80,0.835,0.85,0.90",
        help="Comma-separated threshold grid for event-calibrate",
    )
    parser.add_argument(
        "--calibrate-cooldowns",
        default="0.00,0.50,2.00",
        help="Comma-separated cooldown grid (seconds) for event-calibrate",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output summary JSON path (default: runs/ladder_summary_<timestamp>.json)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = (
        Path(args.output)
        if args.output
        else Path("runs") / f"ladder_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, Any]] = []
    for name, train_profile in [
        ("teacher_baseline", "wifi_all"),
        ("product_baseline", "earbuds_glasses"),
    ]:
        run = _run_single_ladder(
            name=name,
            train_profile=train_profile,
            config=args.config,
            adapter=args.adapter,
            data_root=args.data_root,
            seed=int(args.seed),
            train_device=args.train_device,
            eval_device=args.eval_device,
            event_mode=args.event_mode,
            epochs=int(args.epochs),
            profiles_csv=str(args.profiles),
            calibrate_max_eval_samples=int(args.calibrate_max_eval_samples),
            fp_hour_budget=float(args.fp_hour_budget),
            calibrate_thresholds_csv=str(args.calibrate_thresholds),
            calibrate_cooldowns_csv=str(args.calibrate_cooldowns),
        )
        records.append(_build_result_record(run))

    summary = {
        "created_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "config": {
            "config_path": args.config,
            "adapter": args.adapter,
            "data_root": args.data_root,
            "seed": int(args.seed),
            "train_device": args.train_device,
            "eval_device": args.eval_device,
            "event_mode": args.event_mode,
            "epochs": int(args.epochs),
            "profiles": str(args.profiles),
            "calibrate_max_eval_samples": int(args.calibrate_max_eval_samples),
            "fp_hour_budget": float(args.fp_hour_budget),
            "calibrate_thresholds": str(args.calibrate_thresholds),
            "calibrate_cooldowns": str(args.calibrate_cooldowns),
        },
        "runs": records,
    }
    output_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Summary written: {output_path}")


if __name__ == "__main__":
    main()
