# Latest Event Results

## Deploy track

- task: **Mobility Transition Presence**
- labels: `[24, 26, 27]`
- optimization target: `sample_presence` F1 under `FP/hour <= 10`

## Reproducible runs

- train run: `runs/20260227_000037_4d7e71f5`
- eval run: `runs/20260227_011127_f46605ff`
- calibration run: `runs/20260227_010805_f46605ff`

## Official budgeted operating point

- metric mode: `sample_presence`
- FP/hour budget: `<= 10`
- threshold: `0.835`
- cooldown: `0.0s`
- F1: `0.6109`
- precision: `0.7289`
- recall: `0.5258`
- FP/hour: `9.91`
- confusion counts: `TP=621`, `FP=231`, `FN=560`

## Secondary metric at same operating point

- within_segment F1: `0.3944`
- within_segment FP/hour: `15.84`

The selected deploy point satisfies the sample-level false-positive budget and keeps trigger-level metrics visible for ongoing improvement work.
