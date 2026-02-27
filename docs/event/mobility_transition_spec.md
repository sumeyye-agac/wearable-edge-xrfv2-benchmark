# Mobility Transition Event Spec (Deploy Track)

## Why this track exists

Phone semantics from head-mounted IMU are hard under strict false-alarm budgets.
For deployment, we need a physically observable target with stable edge operating points.

This track defines a deploy-oriented event:

- positive labels: `24 (Walking)`, `26 (Standing Up)`, `27 (Lying Down)`
- sensors: earbuds + glasses by default, glasses-only fallback
- objective: maximize `within_segment` F1 under `FP/hour <= 10`

## Input profiles

- `earbuds_glasses` (default): `airpods + imu_gl`
- `glasses_only` (fallback): `imu_gl`
- `all_imu`, `wifi_all`: diagnostic only

## Model + pipeline

- model: tiny TCN
- event mode: hierarchical
- candidate generation: glasses motion energy
- candidate scoring: max positive probability in candidate window
- trigger timing: peak frame in candidate window
- calibration grid: threshold + cooldown sweep per profile

## Metrics for go/no-go

Primary:

- `within_segment` precision / recall / F1
- `FP/hour`

Secondary:

- `onset_strict` metrics (more sensitive to onset ambiguity)
- edge metrics (params, checkpoint size, CPU latency)

## Latest reproducible operating points (2048-eval benchmark)

From `docs/event/results_latest.md`:

- product (`earbuds_glasses`): threshold `0.67`, cooldown `16.0s`
- fallback (`glasses_only`): threshold `0.88`, cooldown `32.0s`

## Reproduction

```bash
xrfv2-edge-tal event-train \
  --config configs/event_mobility_transition.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --profile earbuds_glasses

xrfv2-edge-tal event-calibrate \
  --config configs/event_mobility_transition.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run>/checkpoints/last.npz \
  --profiles earbuds_glasses,glasses_only \
  --metric-mode within_segment --fp-hour-budget 10
```
