# XRF V2 Edge Benchmark: Deploy-Ready Wearable Event Detection

This repository implements a reproducible, edge-first event detector on XRF V2 for the realistic wearable sensor pair: **earbuds + smart glasses**.

## Problem, input, output

- Task: detect **Mobility Transition Presence**.
- Positive labels: `Walking (24)`, `Standing Up (26)`, `Lying Down (27)`.
- Input: multi-modal time-series windows from XRF V2 (`airpods`, split IMU receivers, optional Wi-Fi).
- Output: calibrated event triggers and profile-wise reports.
- Primary deploy metric: `sample_presence` F1 under `FP/hour <= 10`.

This task is intentionally scoped to a physically observable behavior so that the model can meet practical false-positive constraints on edge devices.

## Data

XRF V2 is not redistributed here. Obtain it from the official XRFV2 project/release channels.

Expected local directory:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

Data normalization in this repo:

- `imu` is canonically exposed as `imu_gl`, `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp`
- `airpods` is reduced to 6 channels (`acc + rot`) for stable profile behavior

## Deployment profiles

| Profile | Sensors | Intended use |
|---|---|---|
| `earbuds_glasses` | `airpods + imu_gl` | default product profile |
| `glasses_only` | `imu_gl` | fallback profile |
| `all_imu` | all IMU streams | diagnostic upper bound |
| `wifi_all` | Wi-Fi + all IMU | non-product upper bound |

## Quickstart (no dataset required)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_presence_mobility.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_presence_mobility.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only
```

## Real XRF V2 run

```bash
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes

xrfv2-edge-tal event-train \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 \
  --data-root data/raw/xrfv2_kaggle \
  --profile glasses_only

xrfv2-edge-tal event-calibrate \
  --config configs/event_presence_mobility.yaml \
  --adapter xrfv2 \
  --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run_id>/checkpoints/last.npz \
  --profiles earbuds_glasses,glasses_only \
  --metric-mode sample_presence \
  --fp-hour-budget 10
```

## Latest reproducible operating point

Reference full calibration run: `runs/20260227_030614_5a32e2cf`

- profile: `earbuds_glasses`
- `sample_presence` F1: `0.6117`
- precision: `0.8014`
- recall: `0.4946`
- FP/hour: `6.77`
- threshold: `0.835`
- cooldown: `0.0s`

Full result ledger and run links: `docs/event/results_latest.md`

## Reproducibility and artifacts

- Deploy spec: `docs/event/mobility_transition_spec.md`
- Dataset notes: `docs/dataset_xrfv2.md`
- Artifact contract: `docs/artifact_contract.md`

Each run writes machine-readable artifacts under `runs/<run_id>/`.

## License

MIT
