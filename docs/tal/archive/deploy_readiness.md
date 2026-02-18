# Deploy Readiness Status

This document tracks whether current edge models are ready for production deployment.

## Scope

- Task: Temporal Action Localization (TAL) on XRFV2-style wearable streams
- Scope preserved: IMU-centric, edge-first, lightweight models only

## Latest Iteration Summary

| Candidate | Config | Best Threshold | mAP avg | F1@0.50 | Precision@0.50 | Recall@0.50 | Params | Latency ms (median / p90) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Deploy-TCN | `configs/deploy_tiny_tcn.yaml` | `0.10` | `0.00674104` | `0.01220300` | `0.00701292` | `0.04694786` | `2,958` | `0.8968 / 1.3080` |
| Deploy-TCN-BG | `configs/deploy_tiny_tcn_bg.yaml` | `0.25` | `0.00732479` | `0.01982424` | `0.01258492` | `0.04667131` | `3,007` | `0.9078 / 1.8876` |
| Deploy-TCN-BG-Plus | `configs/deploy_tiny_tcn_bg_plus.yaml` | `0.25` | `0.00530928` | `0.02080425` | `0.01365863` | `0.04362919` | `4,639` | `1.3576 / 1.6313` |
| Deploy-TCN-BG-Focal | `configs/deploy_tiny_tcn_bg_focal.yaml` | `0.10` | `0.00698337` | `0.01549867` | `0.00890660` | `0.05964086` | `3,007` | `0.8882 / 1.9123` |

Best observed:
- Best mAP avg: `deploy_tiny_tcn_bg` (`0.00732479`)
- Best F1@0.50: `deploy_tiny_tcn_bg_plus` (`0.02080425`)

Full machine-readable summary:
- `docs/results_deploy_iterations_2026-02-17.json`

## Readiness Decision

- Runtime/edge constraints: **PASS**
- TAL quality for production localization: **NOT YET**

## Why Not Yet

- Current localization quality remains too low for reliable product behavior.
- Multiple in-scope strategies improved results, but not enough to cross production-ready TAL quality.

## Next Steps (Still In Scope)

1. Add edge-small but stronger temporal head (for example compact dilated residual TCN blocks).
2. Add calibration pipeline for per-class decode thresholds on a held-out validation split.
3. Add robust class-imbalance handling across full training (class-balanced focal + sampling).
4. Verify on a true unseen test split (current environment aliases test to train-formatted files).
