# Event Track Spec (Phone Interaction / Proxy)

## Goal

Build an edge-oriented wearable event detector with deploy-style metrics and explicit sensor profiles.

Primary product profile:

- `earbuds_glasses` = `airpods` + `imu_gl`

Fallback:

- `glasses_only` = `imu_gl`

## Dataset Contract

Expected raw files:

- `train_data.h5`
- `train_label.json`
- `test_data.h5`
- `test_label.json`
- `info.json`

No redistribution in this repository.

## Labeling Modes

Configured under `labels.task_variant`:

- `phone_interaction`: positive = answering/using phone actions
- `hand_to_head_proxy`: physically observable proxy set

Resolution order:

1. `labels.positive_label_ids` if explicitly provided
2. resolver from `info.json` action maps
3. error with actionable fallback instructions

Label source modality is controlled by `labels.source_modality` (`imu` default, `merged_dedup` optional diagnostic).

## Event Modes

Configured under `train.event_mode` and `eval.event_mode`:

- `flat`:
  - framewise classification over full sequence
  - trigger extraction from frame probabilities
- `hierarchical`:
  - Stage A: candidate generation from `imu_gl` motion energy
  - Stage B: classify candidate windows
  - trigger-level filtering with threshold/cooldown/hysteresis

Candidate configuration:

- `energy_threshold`
- `min_active_s`
- `cooldown_s`
- `pre_s`, `post_s`
- `window_len_s`
- `max_windows`

## Trigger and Matching

Trigger generation:

1. optional smoothing
2. threshold/hysteresis
3. cooldown suppression
4. emit timestamped triggers

Metrics:

- `onset_strict`: match if `|pred_time - gt_start| <= onset_tolerance_s`
- `within_segment`: match if `gt_start <= pred_time <= gt_end`
- `precision`, `recall`, `f1`
- `fp_per_hour`
- onset delay stats (`mean`, `p50`, `p90`)

## Calibration Procedure

`event-calibrate` sweeps threshold/cooldown grids and selects the best row under optional FP/hour budget.

Recommended budgeted selection:

- `--metric-mode within_segment`
- `--fp-hour-budget 10`

Outputs:

- `calibration_grid.json`
- `calibration_report.md`
- selection summary in `metrics.json`

## Edge and Reproducibility Outputs

Each run stores:

- `resolved_config.yaml`
- `command.txt`
- `env.json`
- `git.json`
- `metrics.json`
- `dataset_fingerprint.json`
- `benchmark.json`

Evaluation/calibration additionally store:

- `profile_metrics.json`
- `profile_report.md`
- `event_predictions.json`
- `event_ground_truth.json`
- `calibration_grid.json`
- `calibration_report.md`

See `docs/event/results_latest.md` for the latest run ledger and decisions.
