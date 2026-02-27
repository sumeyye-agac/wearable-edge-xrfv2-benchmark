# XRF V2 Edge Benchmark: Deploy-Oriented Wearable Event Detection

This repository is a flagship, reproducible edge-ML benchmark for **wearable event detection** on XRF V2, designed for realistic product sensors: **earbuds + smart glasses**.

## Why This Project Exists

Most academic wearable benchmarks optimize broad activity recognition or full temporal localization with sensor setups that are hard to ship. This project intentionally targets a narrower, product-relevant question:

- Can we get useful event detection quality on constrained, wearable-first sensors?
- Can we hold false positives to a deployment budget?
- Can the full pipeline be reproduced end-to-end with strict artifacts?

Primary audience fit:

- **AI Research Engineer / Scientist Lead**: clear task framing, robust metrics, reproducible artifacts.
- **Startup Founder / CTO**: profile-based deployment story, latency/size visibility, known risks and next actions.

## Final Task Definition

- Task: **Mobility Transition Presence** detection
- Positive labels: `Walking (24)`, `Standing Up (26)`, `Lying Down (27)`
- Input: time-series windows from XRF V2 modalities (`airpods`, split IMU receivers, optional Wi-Fi)
- Output: trigger stream + profile-level metrics + calibrated operating point
- Primary decision metric: `sample_presence` F1 under `FP/hour <= 10`

This is a deliberate simplification from harder semantic tasks, chosen to reach a practical edge operating point.

## Deployment Profiles

| Profile | Sensors | Role |
|---|---|---|
| `earbuds_glasses` | `airpods + imu_gl` | product default |
| `glasses_only` | `imu_gl` | fail-safe fallback |
| `all_imu` | all IMU streams | diagnostic upper bound |
| `wifi_all` | Wi-Fi + all IMU | non-product upper bound |

Why profiles matter:

- They prevent unrealistic claims.
- They expose sensor-dependency risk explicitly.
- They give a CTO-friendly path: default profile + degraded fallback profile.

## Data Contract

XRF V2 is not redistributed here. Obtain it from the official XRFV2 source.

Expected directory:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

In-repo canonicalization:

- `imu` -> `imu_gl`, `imu_lh`, `imu_rh`, `imu_lp`, `imu_rp`
- `airpods` -> 6 channels (`acc + rot`)

## System Overview

1. Ingest and validate XRF V2 H5/JSON schema
2. Select profile modalities (`earbuds_glasses` by default)
3. Hierarchical event pipeline:
   - candidate generation from glasses motion energy
   - window-level tiny model scoring
   - trigger filtering (threshold/cooldown/hysteresis)
4. Evaluate with `onset_strict`, `within_segment`, and `sample_presence`
5. Calibrate threshold/cooldown under `FP/hour` budget

## Latest Reproducible Result (Full Run)

Reference runs:

- train: `runs/20260227_021605_0bc9e9f1`
- eval: `runs/20260227_030049_5a32e2cf`
- calibrate: `runs/20260227_030614_5a32e2cf`

Budgeted operating point (`metric_mode=sample_presence`, `FP/hour<=10`):

| Profile | F1 | Precision | Recall | FP/hour | Threshold | Cooldown(s) |
|---|---:|---:|---:|---:|---:|---:|
| `earbuds_glasses` | **0.6117** | 0.8014 | 0.4946 | 6.77 | 0.835 | 0.0 |
| `glasses_only` | **0.5816** | 0.8414 | 0.4433 | 5.67 | 0.900 | 0.0 |

Secondary quality signal (`earbuds_glasses`): `within_segment F1 = 0.4052`.

Full details: `docs/event/results_latest.md`.

## Strengths, Limits, and Improvement Path

Current strengths:

- Reproducible full pipeline with deterministic artifacts
- Explicit deployment profile logic (default + fallback)
- Budget-aware calibration (`FP/hour`) instead of raw-threshold reporting
- Edge footprint is transparent in run artifacts (model size, latency blocks)

Known limitations:

- Onset-strict trigger metric remains low for this task/sensor pair
- Recall is moderate at the chosen low-FP operating point
- Semantically richer event definitions are harder under wearable-only constraints

Planned improvements (without scope drift):

- Add confidence calibration per profile with validation-split guardrails
- Improve recall under fixed FP budget via hard-negative mining and temporal context tuning
- Add stricter deployment gate report (`F1`, `FP/hour`, latency, checkpoint size) as one summary artifact

## Quickstart (No Dataset Required)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_presence_mobility.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_presence_mobility.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only
```

## Run on Real XRF V2

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

## Reproducibility Contract

- Deploy spec: `docs/event/mobility_transition_spec.md`
- Dataset notes: `docs/dataset_xrfv2.md`
- Artifact contract: `docs/artifact_contract.md`

Every run writes machine-readable files under `runs/<run_id>/` so claims can be audited and reproduced.
