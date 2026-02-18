# Phone Interaction Event Spec (XRF V2)

## Goal

Ship-able phone interaction event detection from wearable IMU, with:
- default profile: earbuds + glasses
- fallback profile: glasses-only

## Dataset Scope

- Source: XRF V2 multi-device IMU (phone/watch/earbuds/glasses)
- Redistribution: not allowed in this repo
- Expected raw files:
  - `train_data.h5`
  - `train_label.json`
  - `test_data.h5`
  - `test_label.json`
  - `info.json`

## Positive Event Definition

A frame/segment is positive if action is either:
- `Answer the phone`
- `Use phone`

Label resolution order:
1. Parse `info.json` mapping keys (`labels`, `label_map`, `actions`, `action_map`, `id2label`, `label2id`)
2. Match names to positive actions
3. Fallback to explicit config `labels.positive_label_ids`

## Model Output -> Trigger

1. Framewise probability for positive class
2. Smoothing
3. Threshold/hysteresis
4. Cooldown suppression
5. Emit trigger timestamps

## Matching Rule

Predicted trigger matches GT event if:
- same sequence
- `|pred_time - gt_start| <= onset_tolerance_s`
- greedy one-to-one matching

## Metrics

- Precision / Recall / F1
- FP/hour
- onset delay (mean, p50, p90)
- edge metrics: params, checkpoint size MB, CPU latency median/p90

Duration handling:
- Prefer adapter-provided timestamps/duration
- Else use `eval.frame_time_s` (default 0.02 seconds)

## Deployment Profiles

- `earbuds_glasses`: default production path
- `glasses_only`: fallback profile
- `all_imu`: diagnostic upper bound only

## Artifacts

Event runs include standard contract plus:
- `profile_metrics.json`
- `profile_report.md`
- `event_predictions.json`
- `event_ground_truth.json`
