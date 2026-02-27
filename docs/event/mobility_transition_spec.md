# Mobility Transition Deploy Spec

## Deploy objective

This deploy track uses a simplified, product-friendly objective:

- Event class: `Walking (24)` OR `Standing Up (26)` OR `Lying Down (27)`
- Sensors: earbuds + glasses (`earbuds_glasses`) with glasses-only fallback
- Primary metric: `sample_presence` F1 under `FP/hour <= 10`

Model input/output:

- Input: profile-filtered IMU windows (`airpods + imu_gl` by default)
- Output: discrete event triggers and profile-level metrics

`sample_presence` means:

- Each sample window is binary (`event present` / `not present`).
- Prediction is positive if at least one trigger is emitted in that sample.

This is intentionally simpler than strict trigger localization and better aligned with deploy gates.

## Pipeline

- TinyTCN, hierarchical candidate generation from glasses IMU energy
- Candidate scoring: max positive probability
- Trigger timing: peak frame
- Calibration: threshold + cooldown sweep by profile

## Latest reproducible operating point

Checkpoint:

- `runs/20260227_021605_0bc9e9f1/checkpoints/last.npz`

Calibration run:

- `runs/20260227_030614_5a32e2cf`

Best budgeted point (`FP/hour <= 10`):

- threshold: `0.835`
- cooldown: `0.0s`
- `sample_presence` F1: `0.6117`
- `sample_presence` FP/hour: `6.77`

## Commands

```bash
xrfv2-edge-tal event-train \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle \
  --profile glasses_only

xrfv2-edge-tal event-calibrate \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run>/checkpoints/last.npz \
  --profiles earbuds_glasses,glasses_only \
  --metric-mode sample_presence --fp-hour-budget 10
```
