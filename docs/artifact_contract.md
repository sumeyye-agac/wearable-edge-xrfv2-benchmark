# Artifact Contract

Each run must emit a self-contained artifact directory under `runs/`.

## Required files

- `resolved_config.yaml`: fully resolved config used for the run.
- `env.json`: environment details (python, platform, selected package versions).
- `git.json`: git metadata (commit hash, branch, dirty flag) when available.
- `command.txt`: exact CLI command used to launch the run.
- `metrics.json`: train/eval scalar metrics and summary values.
- `dataset_fingerprint.json`: lightweight dataset fingerprint (file sizes + hashes + counts).
- `benchmark.json`: edge benchmarking metrics (params, model size, latency).

## Event-track files

- `profile_metrics.json`: profile-wise metrics map for `earbuds_glasses`, `glasses_only`, optional `all_imu`.
- `profile_report.md`: table report for profile comparison.
- `event_predictions.json`: emitted triggers with score/time/profile.
- `event_ground_truth.json`: GT event starts used for matching.
- `calibration_report.md`: best threshold/cooldown per profile (`event-calibrate`).
- `calibration_grid.json`: full threshold/cooldown sweep rows (`event-calibrate`).

## Directory shape

```text
runs/
  20260218_123045_1a2b3c4d/
    resolved_config.yaml
    env.json
    git.json
    command.txt
    metrics.json
    dataset_fingerprint.json
    benchmark.json
    profile_metrics.json
    profile_report.md
    calibration_report.md
    calibration_grid.json
    checkpoints/
      last.npz
```

## Reproducibility expectations

- The run can be traced from config + command + commit.
- Metrics are machine-readable JSON.
- Dataset fingerprint enables data drift detection.
- Benchmark file captures edge constraints directly.
