# XRF V2 Edge Event Detection (Earbuds + Glasses)

This repository is a reproducible, edge-first benchmark for **wearable event detection on XRF V2**, focused on a realistic product sensor set: **AirPods IMU + glasses receiver IMU**.

## What This Is

- **Data**: XRF V2 (2025 multi-device wearable dataset; IMU + Wi-Fi available).
- **Task**: convert model scores into discrete event triggers (not just frame logits).
- **Default product profile**: `earbuds_glasses` (`airpods` + `imu_gl`).
- **Fallback profile**: `glasses_only` (`imu_gl`).
- **Why this matters**: we report product metrics (within-segment F1, FP/hour, onset delay) plus edge metrics (params, checkpoint size, CPU latency) with full run artifacts.

## Data (No Redistribution)

This repo does **not** redistribute XRF V2 data. Obtain data from the official XRFV2 project.

Expected raw folder:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

XRF V2 includes multiple IMU devices (phone/watch/earbuds/glasses) and Wi-Fi.  
This repo uses a canonical IMU representation:

- `imu` is split into receiver keys: `imu_gl`, `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp`
- `airpods` is reduced to 6 channels (`acc + rot`)

Probe real shapes:

```bash
python scripts/probe_xrfv2.py --data-root data/raw/xrfv2_kaggle
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes
```

## Event Definition

Phone Interaction event:

- positive action = `Answering Phone` OR `Using Phone`
- output = timestamp triggers after smoothing + threshold + cooldown (+ optional hysteresis)

Primary evaluation:

- `within_segment`: prediction time falls inside GT segment
- `onset_strict`: prediction time is within tolerance of GT onset
- plus `FP/hour` and onset delay stats (`mean`, `p50`, `p90`)

Why event detection (vs full TAL) in the flagship path:

- closer to product trigger behavior
- directly optimizable under false-alarm budgets
- easier to reason about edge deployment constraints

## Deployment Profiles

| Profile | Sensors | Intended use |
|---|---|---|
| `earbuds_glasses` | `airpods` + `imu_gl` | Default product target |
| `glasses_only` | `imu_gl` | Fallback when earbuds disconnect |
| `all_imu` | `airpods` + all IMU receivers | Diagnostic upper bound |
| `wifi_all` | `wifi` + all IMU receivers | Non-product upper bound |

Restricting sensors is intentional: it avoids unrealistic reliance on phone/watch channels and keeps the benchmark aligned to deployable wearable products.

## Quickstart (No Dataset Required)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_phone_interaction.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_phone_interaction.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only
```

## Real XRF V2 Run

```bash
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes
xrfv2-edge-tal event-train --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --profile earbuds_glasses --override train.event_mode=hierarchical --override eval.event_mode=hierarchical
xrfv2-edge-tal event-eval --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles wifi_all,all_imu,earbuds_glasses,glasses_only --override train.event_mode=hierarchical --override eval.event_mode=hierarchical
xrfv2-edge-tal event-calibrate --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles wifi_all,all_imu,earbuds_glasses,glasses_only --metric-mode within_segment --fp-hour-budget 10 --override eval.max_eval_samples=2048 --override train.event_mode=hierarchical --override eval.event_mode=hierarchical
```

All outputs are written under `runs/<run_id>/`.

## Results (Truthful, Reproducible)

Latest experiment ledger is in `docs/event/results_latest.md`.

Current status from full runs in this branch:

- Flat baseline (Stage 0): budgeted `within_segment F1` is `0.0` across profiles.
  - Summary: `runs/ladder_stage0_flat.json`
- Hierarchical phone-interaction run (Stage 2): budgeted `within_segment F1` remains `0.0`.
  - Train: `runs/20260219_093913_d2eba0a6`
  - Eval: `runs/20260219_100616_b1665395`
  - Calibrate: `runs/20260219_100729_294f368e`
- Proxy variant run (Stage 4, explicit IDs `{5,6,16,21}`): still below deploy threshold.
  - Train: `runs/20260219_104805_47987fce`
  - Eval: `runs/20260219_110732_e1a63005`
  - Calibrate: `runs/20260219_110845_e270b93a`
  - Budgeted (`FP/hour<=10`) best:
    - `earbuds_glasses`: `within_segment F1=0.0000`
    - `glasses_only`: `within_segment F1=0.0064`
  - Unbudgeted best (for context only): `within_segment F1=0.0976` at `FP/hour=151.10`

Interpretation: current lightweight pipelines are not yet deploy-ready under the target FP/hour budget for this sensor restriction.

## Artifact Contract

Every run writes:

- `resolved_config.yaml`
- `env.json`
- `git.json`
- `command.txt`
- `metrics.json`
- `dataset_fingerprint.json`
- `benchmark.json`

Event runs also write:

- `profile_metrics.json`
- `profile_report.md`
- `event_predictions.json`
- `event_ground_truth.json`
- `calibration_report.md`
- `calibration_grid.json`

See `docs/artifact_contract.md` for details.

## Notes

- TAL-focused materials are kept as secondary/archive content under `docs/tal/archive/`.
- No dataset or model binaries are committed.

## License

MIT
