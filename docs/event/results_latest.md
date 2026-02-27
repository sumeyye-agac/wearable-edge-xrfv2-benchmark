# Latest Event Results (Branch: `exp/hierarchical-event-v3`)

## Executive Summary

A deploy-oriented problem update was applied:

- from: phone-interaction semantic event
- to: mobility-transition event (`Walking`, `Standing Up`, `Lying Down`)

Reason: stronger physical observability from head-mounted IMU under strict false-alarm budget.

Primary decision metric: `within_segment` F1 at `FP/hour <= 10`.

## Deploy Track (Mobility Transition)

Config: `configs/event_mobility_transition.yaml`

### Product model (`earbuds_glasses` training profile)

- train run: `runs/20260226_224000_9fa54963`
  - profile: `earbuds_glasses`
  - labels: `[24, 26, 27]`
  - epochs: `2`, train samples: `512`
- eval run (2048 samples): `runs/20260226_235728_f082093d`
- calibrate run (fine sweep): `runs/20260227_000013_f082093d`

Best budgeted operating point (`earbuds_glasses`):

- threshold: `0.67`
- cooldown: `16.0s`
- within_segment: `F1=0.1321`, `precision=0.3694`, `recall=0.0804`, `FP/hour=9.74`
- counts: `TP=133`, `FP=227`, `FN=1521`

### Fallback model (`glasses_only` training profile)

- train run: `runs/20260227_000037_4d7e71f5`
  - profile: `glasses_only`
  - labels: `[24, 26, 27]`
  - epochs: `3`, train samples: `1024`
- eval run (2048 samples): `runs/20260227_000553_f082093d`
- calibrate run (high-threshold fine sweep): `runs/20260227_000917_3db08c5d`

Best budgeted operating point (`glasses_only`):

- threshold: `0.88`
- cooldown: `32.0s`
- within_segment: `F1=0.2707`, `precision=0.5676`, `recall=0.1778`, `FP/hour=9.61`
- counts: `TP=294`, `FP=224`, `FN=1360`

## Notes

- The fallback model is calibrated separately from the product model.
- Profile-specific thresholds are required for stable budget behavior.
- `onset_strict` remains low and is treated as secondary for this deploy track.

## Archived Research Track (Phone Interaction)

Phone-interaction history is preserved in previous run artifacts and commit history.
This track remains useful for research, but current lightweight models are not as stable as the deploy track above under strict FP/hour budgets.
