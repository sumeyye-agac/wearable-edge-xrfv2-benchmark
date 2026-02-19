# XRF V2 Wearable Event Detection (Edge-First)

`xrfv2-edge-tal` is a reproducible benchmark for **phone interaction event detection** on the XRF V2 wearable dataset, with a product-first sensor target: **AirPods IMU + glasses receiver IMU**.

## What This Repository Does

- Uses XRF V2 (multi-device IMU + Wi-Fi) without redistributing any dataset files.
- Trains lightweight frame models and converts frame probabilities into event triggers.
- Reports product-style metrics:
  - `onset_strict` (timestamp tolerance matching)
  - `within_segment` (trigger lands anywhere inside GT segment)
  - FP/hour, onset delay, model size, CPU latency
- Enforces reproducibility with run artifacts under `runs/<run_id>/`.

## Data (No Redistribution)

This repo does **not** ship XRF V2 data. Obtain it from the official XRFV2 sources.

Expected raw directory:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

## Data Format Note (Important)

XRF V2 `imu` is a packed 5-receiver tensor. This repository explicitly splits it into receiver keys:

- `imu_gl` (glasses receiver)
- `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp`

`airpods` is reduced to 6 channels (`acc + rot`) from the original 9-channel tensor.

You can probe real shapes with:

```bash
python scripts/probe_xrfv2.py --data-root data/raw/xrfv2_kaggle
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes
```

## Deployment Profiles (True Keys)

Profiles are configured with raw modality keys to avoid ambiguity:

- `earbuds_glasses`: `airpods`, `imu_gl` (default product target)
- `glasses_only`: `imu_gl` (fallback)
- `all_imu`: `airpods`, `imu_gl`, `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp` (upper bound)
- `wifi_all`: `wifi` + all IMU keys (upper bound, non-product)

## Event Definition

Positive event = union of action names:

- `Answer the phone`
- `Use phone`

Default label source is `labels.source_modality=imu` to prevent cross-modality label duplication. Optional `merged_dedup` is supported for diagnostics.

## Quickstart (Dummy)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_phone_interaction.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_phone_interaction.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only
```

## Real XRF V2 Run (Reproducible)

```bash
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes
xrfv2-edge-tal event-train --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --profile wifi_all --override train.epochs=1 --override train.max_train_samples=512
xrfv2-edge-tal event-eval --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles wifi_all,all_imu,earbuds_glasses,glasses_only
xrfv2-edge-tal event-calibrate --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles wifi_all,all_imu,earbuds_glasses,glasses_only --metric-mode within_segment
```

## Results: Deployment Profile Trade-Offs (Honest Story)

This project uses a strict comparison ladder:

1. Upper bounds (`wifi_all`, `all_imu`)
2. Product target (`earbuds_glasses`)
3. Fallback (`glasses_only`)

If upper bounds are non-trivial while product profiles drop, that is a valid and expected deployment trade-off, not hidden by unrealistic sensor assumptions.

### Full-run artifacts (real XRF V2)

Primary full run (teacher, upper-bound profile):

- Train (`wifi_all`, full train split, 8 epochs): `runs/20260219_013737_b4a5a303/`
- Full eval (all profiles): `runs/20260219_033317_25b6180b/`
- Product-aware calibration (subset for speed, `max_eval_samples=2048`): `runs/20260219_040033_e4468f56/`

Distillation attempt (student on `earbuds_glasses`):

- Train (`earbuds_glasses`, distilled, full train split, 6 epochs): `runs/20260219_040058_afb39261/`
- Full eval: `runs/20260219_043058_25b6180b/`
- Product-profile calibration: `runs/20260219_043826_4686cd1d/`

Measured timing (Apple Silicon + MPS):

- Full teacher run average epoch: `~239s` (`runs/20260219_013737_b4a5a303/metrics.json`)
- Full distilled student average epoch: `~208s` (`runs/20260219_040058_afb39261/metrics.json`)

Read generated reports directly:

- `runs/<eval_run_id>/profile_report.md`
- `runs/<eval_run_id>/profile_metrics.json`
- `runs/<calibration_run_id>/calibration_report.md`
- `runs/<calibration_run_id>/calibration_grid.json`

Current status from reproducible artifacts:

- `wifi_all` can reach non-zero budgeted quality in calibrated subset mode:
  - `within_segment F1=0.3576`, `FP/hour=6.824` in `runs/20260219_040033_e4468f56/`
- `earbuds_glasses` and `glasses_only` are not yet deploy-ready under the same FP budget:
  - budget-constrained runs collapse to near-zero recall in current baselines.
- This is an explicit product gap, not hidden. The repo keeps profile reports and calibration grids so the trade-off is inspectable.

Table schema (generated, not hand-written):

| Profile | Sensors | Onset F1 | Within F1 | Onset FP/hour | Within FP/hour | p90 onset delay (s) | Notes |
|---|---|---:|---:|---:|---:|---:|---|

## Artifact Contract

Each run writes reproducibility files under `runs/<run_id>/`:

- `resolved_config.yaml`
- `env.json`
- `git.json`
- `command.txt`
- `metrics.json`
- `dataset_fingerprint.json`
- `benchmark.json`
- `profile_metrics.json` (event eval)
- `profile_report.md` (event eval)
- `event_predictions.json` / `event_ground_truth.json`

See `docs/artifact_contract.md` for details.

## License

MIT
