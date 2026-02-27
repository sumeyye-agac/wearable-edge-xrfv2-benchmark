# Artifact Contract

Each run writes a self-contained artifact directory under `runs/`.

## Core files

- `resolved_config.yaml`: fully resolved config used for the run
- `env.json`: python/platform/runtime details
- `git.json`: commit, branch, dirty state (when available)
- `command.txt`: exact CLI invocation
- `metrics.json`: scalar metrics and summary blocks
- `dataset_fingerprint.json`: lightweight data fingerprint for reproducibility
- `benchmark.json`: model params, size, latency (for benchmark runs)

## Event-track files

- `profile_metrics.json`: per-profile metrics for `earbuds_glasses`, `glasses_only`, optional upper bounds
- `profile_report.md`: readable profile comparison table
- `event_predictions.json`: predicted trigger times and scores
- `event_ground_truth.json`: ground-truth event starts used for matching
- `calibration_report.md`: selected threshold/cooldown operating point (`event-calibrate`)
- `calibration_grid.json`: full threshold/cooldown sweep (`event-calibrate`)

## Repository-level references

- `docs/event/mobility_transition_spec.md`: deploy-track definition
- `docs/event/results_latest.md`: latest reproducible run ledger and operating point

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
- Latest cross-run comparison is documented in `docs/event/results_latest.md`.
