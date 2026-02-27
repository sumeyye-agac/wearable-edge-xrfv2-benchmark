# XRF V2 Edge Benchmark: Wearable Event Detection

This repo contains an end-to-end benchmark for wearable event detection on XRF V2, with a practical sensor setup: **earbuds + smart glasses**.

## What This Repo Does

- Defines a deploy-oriented event task: **Mobility Transition Presence**
- Trains and evaluates lightweight models on profile-restricted sensors
- Calibrates operating points under a false-positive budget (`FP/hour`)
- Produces machine-readable artifacts for every run

## Task Definition

- Positive labels: `Walking (24)`, `Standing Up (26)`, `Lying Down (27)`
- Input: time-series windows from XRF V2 modalities (`airpods`, split IMU receivers, optional Wi-Fi)
- Output: event triggers + profile-level metrics + calibrated threshold/cooldown
- Main decision metric: `sample_presence` F1 with `FP/hour <= 10`

The scope is intentionally narrow: this is a reliability-first event track, not a broad activity suite.

## Deployment Profiles

| Profile | Sensors | Purpose |
|---|---|---|
| `earbuds_glasses` | `airpods + imu_gl` | default profile |
| `glasses_only` | `imu_gl` | fallback profile |
| `all_imu` | all IMU streams | diagnostic upper bound |
| `wifi_all` | Wi-Fi + all IMU | non-product upper bound |

## Quickstart (No Dataset)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_presence_mobility.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_presence_mobility.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only
```

## Data

XRF V2 is not redistributed in this repository.

Expected local layout:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

Canonical handling in this repo:

- `imu` is exposed as `imu_gl`, `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp`
- `airpods` is reduced to 6 channels (`acc + rot`)

## Current Reproducible Result

Reference runs:

- train: `runs/20260227_021605_0bc9e9f1`
- eval: `runs/20260227_030049_5a32e2cf`
- calibrate: `runs/20260227_030614_5a32e2cf`

Budgeted operating point (`sample_presence`, `FP/hour<=10`):

| Profile | F1 | Precision | Recall | FP/hour | Threshold | Cooldown(s) |
|---|---:|---:|---:|---:|---:|---:|
| `earbuds_glasses` | **0.6117** | 0.8014 | 0.4946 | 6.77 | 0.835 | 0.0 |
| `glasses_only` | **0.5816** | 0.8414 | 0.4433 | 5.67 | 0.900 | 0.0 |

Additional signal (`earbuds_glasses`): `within_segment F1 = 0.4052`.

Detailed ledger: `docs/event/results_latest.md`.

## Full Reproduction (One Command)

```bash
python scripts/reproduce_full_run.py \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 \
  --data-root data/raw/xrfv2_kaggle \
  --seed 42 \
  --train-profile earbuds_glasses \
  --profiles earbuds_glasses,glasses_only \
  --train-device auto \
  --eval-device auto
```

Generated manifests:

- `runs/repro_full_latest.json`
- `runs/<calibrate_run_id>/repro_manifest.json`

For closest numeric repeatability across machines, use `--train-device cpu --eval-device cpu`.

## Run On Real XRF V2 (Manual Steps)

```bash
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes

xrfv2-edge-tal event-train \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 \
  --data-root data/raw/xrfv2_kaggle \
  --profile earbuds_glasses \
  --override train.max_train_samples=0

xrfv2-edge-tal event-eval \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 \
  --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run_id>/checkpoints/last.npz \
  --profiles earbuds_glasses,glasses_only \
  --override eval.max_eval_samples=0

xrfv2-edge-tal event-calibrate \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 \
  --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run_id>/checkpoints/last.npz \
  --profiles earbuds_glasses,glasses_only \
  --metric-mode sample_presence \
  --fp-hour-budget 10 \
  --override eval.max_eval_samples=0
```

## What Is Solid, What Is Not Yet

Solid:

- Reproducible run trail with structured artifacts
- Clear profile separation (default + fallback + upper bounds)
- Budget-based calibration instead of raw threshold reporting

Still improving:

- `onset_strict` is low for this sensor/task setup
- recall can be higher at the chosen FP budget
- richer semantic events remain harder with wearable-only inputs

## Docs

- deploy spec: `docs/event/mobility_transition_spec.md`
- dataset notes: `docs/dataset_xrfv2.md`
- artifact contract: `docs/artifact_contract.md`
