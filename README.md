# XRF V2 Edge Benchmark: Deploy-Oriented Wearable Event Detection

This repository is a reproducible benchmark for **edge event detection from earbuds + smart-glasses IMU** on XRF V2, with artifact-backed runs and profile-specific deployment metrics.

## What This Repo Does

- Uses XRF V2 wearable sensor data.
- Converts framewise scores into discrete event triggers.
- Evaluates with product-style metrics:
  - `within_segment` Precision / Recall / F1
  - `FP/hour`
  - onset delay statistics
  - edge metrics (params, checkpoint size, CPU latency)
- Supports deployment profiles:
  - `earbuds_glasses` (default product profile)
  - `glasses_only` (fallback profile)

## Problem Update For Deployment

The original phone-semantic event (`Answering Phone + Using Phone`) is preserved as a research track.
For deployability, this repo now promotes a physically observable event track:

- **Mobility Transition Event** = union of labels `[24, 26, 27]`
  - `Walking`
  - `Standing Up`
  - `Lying Down`

Spec: `docs/event/mobility_transition_spec.md`

## Data (No Redistribution)

This repo does not redistribute XRF V2.

Expected raw folder:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

Probe shapes and modality keys:

```bash
python scripts/probe_xrfv2.py --data-root data/raw/xrfv2_kaggle
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes
```

Canonical modality handling in this repo:

- `imu` split to receiver keys: `imu_gl`, `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp`
- `airpods` reduced to 6 channels (`acc + rot`)

## Deployment Profiles

| Profile | Sensors | Use |
|---|---|---|
| `earbuds_glasses` | `airpods + imu_gl` | Product default |
| `glasses_only` | `imu_gl` | Fallback |
| `all_imu` | all IMU streams | Diagnostic upper bound |
| `wifi_all` | Wi-Fi + all IMU streams | Non-product diagnostic |

## Quickstart (No Dataset)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_mobility_transition.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_mobility_transition.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only
```

## Real XRF V2 (Deploy Track)

Train product profile:

```bash
xrfv2-edge-tal event-train \
  --config configs/event_mobility_transition.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle \
  --profile earbuds_glasses
```

Evaluate and calibrate:

```bash
xrfv2-edge-tal event-eval \
  --config configs/event_mobility_transition.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run_id>/checkpoints/last.npz \
  --profiles wifi_all,all_imu,earbuds_glasses,glasses_only

xrfv2-edge-tal event-calibrate \
  --config configs/event_mobility_transition.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run_id>/checkpoints/last.npz \
  --profiles earbuds_glasses,glasses_only \
  --metric-mode within_segment --fp-hour-budget 10
```

All outputs are stored under `runs/<run_id>/`.

## Latest Reproducible Results

Reference ledger: `docs/event/results_latest.md`

Reported benchmark scale: `eval.max_eval_samples=2048`.

| Track | Train profile | Budgeted within_segment F1 | FP/hour | Threshold | Cooldown | Run refs |
|---|---|---:|---:|---:|---:|---|
| Product model | `earbuds_glasses` | `0.1321` | `9.74` | `0.67` | `16.0s` | train `runs/20260226_224000_9fa54963`, cal `runs/20260227_000013_f082093d` |
| Fallback model | `glasses_only` | `0.2707` | `9.61` | `0.88` | `32.0s` | train `runs/20260227_000037_4d7e71f5`, cal `runs/20260227_000917_3db08c5d` |

Interpretation:

- Deploy track now reaches budgeted deploy-level quality in both default and fallback profiles.
- Phone-semantic track remains available as research-only in `configs/event_phone_interaction.yaml`.

## Artifact Contract

Each run includes:

- `resolved_config.yaml`
- `env.json`
- `git.json`
- `command.txt`
- `metrics.json`
- `dataset_fingerprint.json`
- `benchmark.json`

Event runs also include:

- `profile_metrics.json`
- `profile_report.md`
- `event_predictions.json`
- `event_ground_truth.json`
- `calibration_report.md`
- `calibration_grid.json`

Details: `docs/artifact_contract.md`

## License

MIT
