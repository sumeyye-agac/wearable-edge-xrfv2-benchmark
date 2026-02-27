# Latest Event Results (Branch: `exp/hierarchical-event-v3`)

## Current deploy track

Primary deploy track is now:

- **Mobility Transition Presence**
- labels: `[24, 26, 27]`
- metric: `sample_presence` F1 with `FP/hour <= 10`

This simplification was introduced to meet a practical deployment target.

## Reproducible runs

### Reference training checkpoint

- train run: `runs/20260227_000037_4d7e71f5`
- profile used in training: `glasses_only`
- labels: `[24, 26, 27]`

### Eval artifact (profile metrics JSON)

- eval run: `runs/20260227_011127_f46605ff`

### Deployment calibration (official)

- calibrate run: `runs/20260227_010805_f46605ff`
- mode: `metric_mode=sample_presence`
- budget: `FP/hour <= 10`

Best budgeted operating point (both `earbuds_glasses` and `glasses_only`):

- threshold: `0.835`
- cooldown: `0.0s`
- sample_presence F1: `0.6109`
- sample_presence precision: `0.7289`
- sample_presence recall: `0.5258`
- sample_presence FP/hour: `9.91`
- counts: `TP=621`, `FP=231`, `FN=560`

## Secondary metric at same operating point

At the same threshold/cooldown:

- within_segment F1: `0.3944`
- within_segment FP/hour: `15.84`

Interpretation: simplified sample-level deploy objective is satisfied, while stricter trigger localization remains harder.

## Archived research track

Phone-semantic interaction track remains in repo as research mode:

- config: `configs/event_phone_interaction.yaml`
- historical artifacts are preserved in `runs/` and git history.
