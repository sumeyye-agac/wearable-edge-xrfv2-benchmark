# Phone Interaction Spec (Research Track)

This document defines the original semantic target:

- positive actions: `Answering Phone` + `Using Phone`
- profiles: `earbuds_glasses` (default), `glasses_only` (fallback)
- evaluation: `within_segment` and `onset_strict` with `FP/hour`

## Status

In current lightweight edge models, this semantic target is still hard under strict false-alarm budgets.
It remains in the repo as a research track.

For deployment-oriented results, use:

- `docs/event/mobility_transition_spec.md`
- `configs/event_mobility_transition.yaml`

## Reproduction

```bash
xrfv2-edge-tal event-train \
  --config configs/event_phone_interaction.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --profile earbuds_glasses \
  --override train.event_mode=hierarchical --override eval.event_mode=hierarchical

xrfv2-edge-tal event-calibrate \
  --config configs/event_phone_interaction.yaml \
  --adapter xrfv2 --data-root data/raw/xrfv2_kaggle \
  --checkpoint runs/<train_run>/checkpoints/last.npz \
  --profiles earbuds_glasses,glasses_only \
  --metric-mode within_segment --fp-hour-budget 10
```
