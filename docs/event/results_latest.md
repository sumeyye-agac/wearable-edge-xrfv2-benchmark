# Latest Event Results (Branch: `exp/hierarchical-event-v3`)

## Scope

This report summarizes the latest reproducible runs for:

- flat phone-interaction baseline
- hierarchical phone-interaction
- hard-negative-mining attempts
- proxy task variant (`hand_to_head_proxy`)
- hierarchical trigger scoring update (`score_mode=max`, `trigger_time=peak`)

Decision metric: `within_segment` F1 under `FP/hour <= 10`.

## Run Registry

- Stage 0 flat baseline summary: `runs/ladder_stage0_flat.json`
- Stage 2 hierarchical phone-interaction:
  - train: `runs/20260219_093913_d2eba0a6`
  - eval: `runs/20260219_100616_b1665395`
  - calibrate: `runs/20260219_100729_294f368e`
- Stage 3 hard-negative-mining exploratory run (subset):
  - train: `runs/20260219_101503_9142165c`
  - eval: `runs/20260219_101919_e1b788ed_1`
  - calibrate: `runs/20260219_101919_e1b788ed`
- Stage 4 proxy full run (`labels.task_variant=hand_to_head_proxy`, `labels.positive_label_ids=5,6,16,21`):
  - train: `runs/20260219_104805_47987fce`
  - eval: `runs/20260219_110732_e1a63005`
  - calibrate: `runs/20260219_110845_e270b93a`
- Stage 5 trigger scoring update on Stage 4 checkpoint:
  - eval: `runs/20260225_232318_c595a4a2`
  - calibrate: `runs/20260225_232426_c644e669`
- Stage 5 trigger scoring check on phone-interaction teacher checkpoint:
  - eval: `runs/20260225_232913_4019b83e`
  - calibrate: `runs/20260225_233020_4c7ddef0`

## Budgeted Results (`FP/hour <= 10`)

### Stage 0 Flat

| Profile | within_segment F1 | FP/hour |
|---|---:|---:|
| `wifi_all` | 0.0000 | 0.000 |
| `all_imu` | 0.0000 | 0.000 |
| `earbuds_glasses` | 0.0000 | 0.000 |
| `glasses_only` | 0.0000 | 0.000 |

### Stage 2 Hierarchical (Phone Interaction)

| Profile | within_segment F1 | FP/hour |
|---|---:|---:|
| `wifi_all` | 0.0000 | 0.000 |
| `all_imu` | 0.0000 | 0.000 |
| `earbuds_glasses` | 0.0000 | 0.000 |
| `glasses_only` | 0.0000 | 0.000 |

### Stage 4 Proxy (Full Run)

| Profile | within_segment F1 | TP | FP | FP/hour | Threshold | Cooldown (s) |
|---|---:|---:|---:|---:|---:|---:|
| `wifi_all` | 0.0000 | 0 | 0 | 0.000 | 0.25 | 0.5 |
| `all_imu` | 0.0000 | 0 | 0 | 0.000 | 0.25 | 0.5 |
| `earbuds_glasses` | 0.0000 | 0 | 0 | 0.000 | 0.25 | 0.5 |
| `glasses_only` | 0.0064 | 3 | 122 | 5.236 | 0.20 | 8.0 |

### Stage 5 Proxy + Trigger Scoring Update (`max` + `peak`)

Reason for change:

- previous hierarchical mode used candidate score = mean window probability and trigger time = window start
- this under-used peak evidence and penalized timing
- updated mode uses candidate score = max positive probability and trigger time = peak frame in candidate window

Budgeted results on same Stage 4 checkpoint:

| Profile | within_segment F1 | TP | FP | FP/hour | Threshold | Cooldown (s) |
|---|---:|---:|---:|---:|---:|---:|
| `wifi_all` | 0.0285 | 14 | 163 | 6.995 | 0.21 | 8.0 |
| `all_imu` | 0.0285 | 14 | 163 | 6.995 | 0.21 | 8.0 |
| `earbuds_glasses` | 0.0285 | 14 | 163 | 6.995 | 0.21 | 8.0 |
| `glasses_only` | 0.0108 | 5 | 112 | 4.807 | 0.21 | 2.0 |

Result:

- product profile improved from `0.0000` to `0.0285` under budget
- still below deploy-ready threshold

## Non-Budget Context (Stage 4 Proxy)

At very low thresholds, the model can produce non-zero recall but FP/hour becomes too high for product use:

- best unbudgeted `within_segment F1`: `0.0976`
- operating point: threshold `0.05`, cooldown `2.0`
- TP `222`, FP `3521`, FP/hour `151.10`

This is not deployable under the target budget.

## Runtime Notes (MPS Training)

- `runs/20260219_093913_d2eba0a6`: avg epoch `143.12s` (8 epochs)
- `runs/20260219_104805_47987fce`: avg epoch `143.63s` (6 epochs)

## Decisions

- Decision A triggered after Stage 2: budgeted quality remained zero.
- Stage 3 hard-negative mining did not produce a meaningful budgeted gain.
- Stage 4 proxy variant remained below deploy-ready threshold.
- Stage 5 trigger-scoring update improved budgeted `earbuds_glasses` to `F1=0.0285` but still below target.

Final status: no plan reached deploy-ready quality for `earbuds_glasses` at `FP/hour <= 10` in current lightweight setup.
