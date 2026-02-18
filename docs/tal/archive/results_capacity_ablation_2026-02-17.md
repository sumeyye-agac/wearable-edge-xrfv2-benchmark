# Capacity Ablation (Base vs Plus) - 2026-02-17

This experiment checks whether slightly larger edge-small models improve TAL quality.

## Protocol

- train: `2` epochs, `1024` samples, `max_windows_per_sample=1`
- eval: `1024` samples
- data: `data/raw/xrfv2_kaggle`
- backend: torch + MPS

## TinyTCN

| Variant | Config | Params | Train Time | Eval Thr | mAP avg | F1@0.50 | Latency ms (median / p90) |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | `configs/paper_light_tiny_tcn.yaml` | `8,238` | `134.23s` | `0.05` | `0.00001856` | `0.00296696` | `2.0848 / 2.3165` |
| Plus | `configs/paper_light_plus_tiny_tcn.yaml` | `16,446` | `135.89s` | `0.05` | `0.00005100` | `0.00223113` | `2.1442 / 2.3953` |
| Plus (tuned) | `configs/paper_light_plus_tiny_tcn.yaml` | `16,446` | - | `0.12` | `0.00001845` | `0.00299097` | - |

Interpretation:
- Plus-TCN improves mAP under low threshold (`0.05`) but loses F1 due to many more false positives.
- With threshold tuning (`0.12`), F1 recovers and slightly exceeds base.

## TinyTransformer

| Variant | Config | Params | Train Time | Eval Thr | mAP avg | F1@0.50 | Latency ms (median / p90) |
|---|---|---:|---:|---:|---:|---:|---:|
| Base | `configs/paper_light_tiny_transformer.yaml` | `8,070` | `161.01s` | `0.01` | `0.00001964` | `0.00102402` | `9.3058 / 10.0759` |
| Plus | `configs/paper_light_plus_tiny_transformer.yaml` | `14,046` | `161.33s` | `0.01` | `0.00001613` | `0.00102402` | `5.7048 / 7.3897` |

Interpretation:
- In this ablation, Plus-Transformer does not improve TAL quality.
- It remains a valid variant, but not the best candidate for immediate promotion.

## Recommendation

- Promote `paper_light_plus_tiny_tcn` as the first “larger edge-small” option.
- Keep `paper_light_plus_tiny_transformer` as optional for further tuning rather than default.

## Raw Metrics

- `docs/results_capacity_ablation_2026-02-17.json`
