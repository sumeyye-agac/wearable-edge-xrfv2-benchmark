# Deploy Readiness Status

This document tracks whether current edge models are ready for production deployment.

## Scope

- Task: Temporal Action Localization (TAL) on XRFV2-style wearable streams
- Scope preserved: IMU-centric, edge-first, lightweight models only

## Latest Deploy Candidate

- Config: `configs/deploy_tiny_tcn.yaml`
- Train run: `20260217_104534_bcb84607`
- Eval run (best tested threshold): `20260217_110645_231e9644` with `decode.score_threshold=0.10`
- Benchmark run: `20260217_110652_bcb84607`

## Measured Metrics

- `mAP_avg`: `0.00674104`
- `F1@0.50`: `0.01220300`
- `Precision@0.50`: `0.00701292`
- `Recall@0.50`: `0.04694786`
- Params: `2,958`
- CPU latency (`seq_len=2048`): median `0.8968 ms`, p90 `1.3080 ms`

## Readiness Decision

- Runtime/edge constraints: **PASS**
- TAL quality for production localization: **NOT YET**

## Why Not Yet

- Current localization quality remains too low for reliable product behavior.
- Threshold tuning changes precision/recall balance, but absolute quality remains below practical expectations.

## Next Steps (Still In Scope)

1. Add lightweight sequence heads with stronger temporal context (still edge-small).
2. Introduce class-balanced/focal-style training in torch backend.
3. Add validation-based threshold calibration and per-class calibration.
4. Add stronger IMU-centric augmentation and label denoising checks.
