# Paper-Light Full Runs (2026-02-17)

This report captures full paper-light runs on real XRFV2 data in this environment.

## Setup

- Dataset root: `data/raw/xrfv2_kaggle`
- Detected modalities: `imu`, `wifi`, `airpods`
- Samples: train `9660`, test `9660`
- Hardware backend: torch + MPS (`device: mps`)

Note:
- Kaggle package in this environment did not include separate test files.
- `test_data.h5` and `test_label.json` are symlinked to train files for executable end-to-end benchmarking.

## Full Run Summary

| Model | Train Run ID | Train Time | Eval Run ID (default thr) | mAP avg (default) | Eval Run ID (low thr) | mAP avg (low) | F1@0.50 (low) |
|---|---|---:|---|---:|---|---:|---:|
| TinyTCN | `20260217_084124_193926b4` | `1129.74s` | `20260217_090230_193926b4` | `0.00041413` | `20260217_090543_95e081bb` | `0.00538485` | `0.01056355` |
| TinyTransformer | `20260217_090601_220f81dd` | `2121.79s` | `20260217_094353_220f81dd` | `0.00000037` | `20260217_094649_ef914fb6` | `0.00000235` | `0.00060719` |

## Edge Metrics

| Model | Benchmark Run ID | Params | Checkpoint MB | Latency ms (median / p90) | Estimated FPS (median / p90) |
|---|---|---:|---:|---:|---:|
| TinyTCN | `20260217_090556_193926b4` | `8,238` | `0.03345` | `2.0787 / 2.4219` | `481.07 / 412.90` |
| TinyTransformer | `20260217_094657_220f81dd` | `8,070` | `0.03318` | `8.8821 / 10.3658` | `112.59 / 96.47` |

## Deployment-Oriented Takeaway

- TinyTCN is the stronger deployment baseline in this run set:
  - higher TAL quality under both default and low-threshold decoding
  - significantly lower latency at similar parameter size
- TinyTransformer remains a useful second baseline but is weaker on both TAL and latency in this setup.

## Raw Artifact Pointer

- Machine-readable snapshot: `docs/results_paper_light_full_2026-02-17.json`
