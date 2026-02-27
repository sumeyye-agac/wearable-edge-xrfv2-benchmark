# XRF V2 Edge Benchmark: Deploy-Ready Wearable Event Detection

This repo is an edge ML benchmark for wearable IMU event detection on XRF V2, focused on **earbuds + smart-glasses** deployment profiles.

## What is deployed here

The default deploy track is a simplified, physically observable objective:

- **Mobility Transition Presence**
- positive labels: `Walking (24)`, `Standing Up (26)`, `Lying Down (27)`
- deployment metric: `sample_presence` F1 under `FP/hour <= 10`

Why simplification:

- strict phone-semantic triggers were not stable enough under hard FP budgets
- this track provides a reliable deploy operating point with reproducible artifacts

## Data

XRF V2 is not redistributed in this repo.

Expected folder:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

Modality handling:

- `imu` is split to: `imu_gl`, `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp`
- `airpods` is reduced to 6 channels (`acc + rot`)

## Profiles

| Profile | Sensors | Purpose |
|---|---|---|
| `earbuds_glasses` | `airpods + imu_gl` | default product profile |
| `glasses_only` | `imu_gl` | fallback profile |
| `all_imu` | all IMU streams | diagnostic |
| `wifi_all` | Wi-Fi + all IMU | non-product diagnostic |

## Quickstart (dummy)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_presence_mobility.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_presence_mobility.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only
```

## Real XRF V2: Deploy Track

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

## Latest reproducible result

Run: `runs/20260227_010805_f46605ff`

Budgeted best (`sample_presence`, `FP/hour<=10`):

- **F1 = 0.6109**
- precision = `0.7289`
- recall = `0.5258`
- FP/hour = `9.91`
- threshold = `0.835`, cooldown = `0.0s`

Full ledger: `docs/event/results_latest.md`

## Specs and artifacts

- Deploy spec: `docs/event/mobility_transition_spec.md`
- Phone-semantic research spec: `docs/event/phone_interaction_spec.md`
- Artifact contract: `docs/artifact_contract.md`

Every run writes machine-readable artifacts under `runs/<run_id>/`.

## License

MIT
