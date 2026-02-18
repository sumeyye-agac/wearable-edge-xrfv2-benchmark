# Deploy-Ready Phone Interaction Event Detection (XRF V2 IMU)

This repository is a reproducible, edge-first benchmark for **phone interaction event detection** on **XRF V2 multi-device IMU**, with a default deploy profile that uses only **earbuds + smart glasses**.

## What This Repo Does

- Data: XRF V2 (2025), a modern multi-device wearable dataset (phone/watch/earbuds/glasses IMU streams).
- Task: detect discrete **Phone Interaction Events** (trigger timestamps), not full TAL by default.
- Product sensor target:
  - default profile: `earbuds_glasses`
  - fallback profile: `glasses_only`
- Product metrics: Precision, Recall, F1, FP/hour, onset delay, CPU latency, model size.
- Reproducibility: every run writes a strict artifact bundle under `runs/<run_id>/`.

## Why Earbuds + Glasses

This is an intentional deployment constraint:
- avoids unrealistic dependence on phone/watch signals that may be unavailable in product runtime
- matches always-on wearable sensing for hands-free interaction scenarios
- gives a clear engineering story: robust default path + graceful fallback on earbud disconnect

## Data

We do **not** redistribute XRF V2. Obtain it from the **Official XRFV2 repo** / official distribution channels.

Expected raw directory (example: `data/raw/xrfv2/`):
- `train_data.h5`
- `train_label.json`
- `test_data.h5`
- `test_label.json`
- `info.json`

Notes:
- XRF V2 can include IMU from phone/watch/earbuds/glasses. This repo defaults to earbuds+glasses only.
- Earbuds are often sampled lower (commonly 25Hz) while others can be 50Hz; pipeline code converts to a canonical frame representation before training/eval.

## Event Definition

Positive event = union of two XRF V2 actions by name:
- `Answer the phone`
- `Use phone`

Model output is converted to event triggers with:
- smoothing
- thresholding
- cooldown
- optional hysteresis

Metrics:
- Precision / Recall / F1: trigger correctness under onset matching tolerance.
- FP/hour: false alarms normalized by duration.
- onset delay: delay between matched prediction and GT start (mean/p50/p90).

Why event detection first (instead of full TAL):
- closer to a ship-able feature (triggering UX logic)
- easier false-alarm control
- cleaner latency and energy budgeting on edge hardware

## Deployment Profiles

| Profile | Sensors | Intended use |
|---|---|---|
| `earbuds_glasses` | Earbuds + Glasses IMU | Default product target |
| `glasses_only` | Glasses IMU | Fallback when earbuds disconnect |
| `all_imu` | Phone + Watch + Earbuds + Glasses IMU | Diagnostic upper bound (non-product) |

Restricting sensors is deliberate. It improves realism and portfolio differentiation, and avoids inflated scores from signals not guaranteed at runtime. Performance may drop as sensors are removed; this repository reports that explicitly via profile reports.

## Quickstart (No Dataset Required)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal event-train --config configs/event_phone_interaction.yaml --adapter dummy
xrfv2-edge-tal event-eval --config configs/event_phone_interaction.yaml --adapter dummy --checkpoint runs/<train_run_id>/checkpoints/last.npz
```

Expected JSON output keys (example):

```json
{
  "event_metrics": {},
  "profile": "earbuds_glasses",
  "profile_metrics": {},
  "edge": {},
  "calibration": {},
  "diagnostics": {}
}
```

## Run on Real XRF V2

1. Obtain the raw files listed above and place them under `data/raw/xrfv2` (or pass your own path).
2. Prepare:

```bash
xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2 --list-modalities
xrfv2-edge-tal prepare --adapter xrfv2 --data-root data/raw/xrfv2 --output-dir data/processed
```

3. Train and evaluate event detector:

```bash
xrfv2-edge-tal event-train --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2 --profile earbuds_glasses
xrfv2-edge-tal event-eval --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2 --profile earbuds_glasses --checkpoint runs/<train_run_id>/checkpoints/last.npz
```

4. Multi-profile comparison report:

```bash
xrfv2-edge-tal event-eval --config configs/event_phone_interaction.yaml --adapter xrfv2 --data-root data/raw/xrfv2 --profiles earbuds_glasses,glasses_only --checkpoint runs/<train_run_id>/checkpoints/last.npz
```

Artifacts are saved in `runs/<run_id>/...` including `metrics.json`, `profile_metrics.json`, and `profile_report.md`.

## Reproducibility Contract

Every run writes:
- `resolved_config.yaml`
- `env.json`
- `git.json`
- `command.txt`
- `metrics.json`
- `dataset_fingerprint.json`
- `benchmark.json`
- `profile_metrics.json` (event multi-profile eval)
- `profile_report.md` (event multi-profile eval)

See `docs/artifact_contract.md`.

## Additional Track (TAL)

Full TAL baselines and historical TAL reports are kept as a secondary track. Archive docs are under `docs/tal/archive/`.

## License

MIT
