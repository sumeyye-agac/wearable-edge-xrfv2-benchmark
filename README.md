# XRF V2 – Edge-First Multi-IMU Temporal Action Localization (TAL) Benchmark

`xrfv2-edge-tal` is a reproducible, edge-focused benchmark for **temporal action localization** on wearable IMU streams (phone/watch/earbuds/glasses).

This codebase emphasizes:
- simple, correct TAL baselines (`TinyTCN`, `TinyTransformer`)
- edge-first reporting (parameter count, checkpoint size, CPU latency)
- robustness to missing modalities (modality dropout + gating fusion)
- optional distillation hook (teacher checkpoint -> student subset)
- reproducible run artifacts for every command

## Why This Repo

Most TAL examples optimize only mAP. This benchmark treats **deployment constraints** as first-class metrics alongside TAL quality.

## Quickstart (Dummy Data, 3 commands)

```bash
pip install -e ".[dev]"
xrfv2-edge-tal train --config configs/dummy_tiny_tcn.yaml --adapter dummy --data-root data/raw/dummy --seed 42
xrfv2-edge-tal eval --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/dummy_tiny_tcn.yaml --adapter dummy --data-root data/raw/dummy --seed 42
```

Notes:
- Replace `<TRAIN_RUN>` with the generated run folder under `runs/`.
- Dummy data is synthetic and generated in-memory for testability.

## Run On Real XRF V2 Data

### 1) Obtain dataset

Expected directory format:
- `train_data.h5`
- `train_label.json`
- `test_data.h5`
- `test_label.json`
- `info.json`

See `docs/dataset_xrfv2.md` for acquisition details.

### 2) Inspect + prepare

```bash
xrfv2-edge-tal inspect --adapter xrfv2 --data-root /path/to/xrfv2
xrfv2-edge-tal prepare --adapter xrfv2 --data-root /path/to/xrfv2 --output-dir data/processed --seed 42
```

### 3) Train + evaluate

```bash
xrfv2-edge-tal train --config configs/dummy_tiny_tcn.yaml --adapter xrfv2 --data-root /path/to/xrfv2 --seed 42
xrfv2-edge-tal eval --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/dummy_tiny_tcn.yaml --adapter xrfv2 --data-root /path/to/xrfv2 --seed 42
```

## Paper-Aligned Lightweight Track (Edge-Oriented)

This repo now supports a **paper-aligned data/training protocol** with lighter models:
- sliding windows (`clip_len=2048`, overlap stride)
- optional interpolation to fixed temporal length
- partial-segment coverage rule (`min_segment_coverage=0.25`)
- overlap-averaged inference for evaluation
- cosine LR schedule and light numeric augmentation

Use:

```bash
xrfv2-edge-tal train --config configs/paper_light_tiny_tcn.yaml --adapter xrfv2 --data-root /path/to/xrfv2 --seed 42
xrfv2-edge-tal eval --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/paper_light_tiny_tcn.yaml --adapter xrfv2 --data-root /path/to/xrfv2 --seed 42
xrfv2-edge-tal benchmark --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/paper_light_tiny_tcn.yaml --seed 42
```

Why this track:
- aligned with paper-level protocol choices where possible
- intentionally different architecture choice (TinyTCN/TinyTransformer) for edge deployment constraints

### Paper-Light Pilot Snapshot (2026-02-17, 256-sample real subset, MPS)

| Model | Train Run ID | Eval Run ID | Decode Thr | mAP avg | F1@0.50 | Params | CPU Latency ms (median / p90) |
|---|---|---|---:|---:|---:|---:|---:|
| TinyTCN (paper_light) | `20260217_082536_b7df6361` | `20260217_082752_8cadc1c9` | `0.05` | `0.00096571` | `0.00286970` | `8,238` | `3.276 / 3.779` |
| TinyTransformer (paper_light) | `20260217_082801_06d434b9` | `20260217_083024_c731d84d` | `0.01` | `0.00000113` | `0.00019417` | `8,070` | `5.560 / 6.504` |

Notes:
- These are short pilot runs (`epochs=1`, `max_train_samples=256`) to validate the protocol and edge behavior quickly.
- Full training is expected to improve TAL quality but require significantly more wall-clock time.

## Latest Real-Run Snapshot (2026-02-17)

Measured on the full available Kaggle XRFV2 release in this environment.

| Model | Train Run ID | Eval Run ID (thr=0.35) | mAP avg (thr=0.35 decode) | mAP avg (thr=0.05 decode) | Params | CPU Latency ms (median / p90) |
|---|---|---|---:|---:|---:|---:|
| TinyTCN | `20260217_005508_d0613500` | `20260217_011540_d0613500` | `0.00000000` | `0.00013746` | `10,974` | `35.7483 / 45.0604` |
| TinyTransformer | `20260217_011550_2a6307aa` | `20260217_020811_2a6307aa` | `0.00000000` | `0.00000090` | `14,046` | `94.7648 / 194.0851` |

Notes:
- Kaggle package currently lacks `test_data.h5` and `test_label.json`; local run used aliases to the available train files for executable end-to-end validation.
- Full analysis artifacts were generated under `/tmp/real_xrfv2_runs/` including `real_analysis_full.md` and `real_analysis_full.json`.

## Edge Metrics

Benchmark a checkpoint:

```bash
xrfv2-edge-tal benchmark --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/dummy_tiny_tcn.yaml --seed 42
```

Outputs include:
- total parameters
- checkpoint/model size (MB)
- CPU latency (`median`, `p90`) after warmup
- estimated FPS (`median`, `p90`)
- simple edge readiness flag (`latency_budget_pass_50ms`)

ONNX export:
```bash
xrfv2-edge-tal export-onnx --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/dummy_tiny_tcn.yaml --output-path artifacts/model.onnx --seed 42
```
(`torch` + `onnxruntime` are required for ONNX export verification.)

## Reproducibility Contract

Each run writes artifacts under `runs/<run_id>/`:
- `resolved_config.yaml`
- `env.json`
- `git.json`
- `command.txt`
- `metrics.json`
- `dataset_fingerprint.json`
- `benchmark.json`

See `docs/artifact_contract.md` for full contract.

## Baselines

- `TinyTCN`: lightweight temporal smoothing + framewise classifier
- `TinyTransformer`: tiny self-attention framewise classifier
- Shared postprocessing: framewise-to-segment decoding + temporal NMS

## TAL Metrics

Implemented TAL evaluation:
- segment tIoU
- AP@tIoU
- mAP averaged over thresholds `0.50:0.05:0.95`

## Citation

If you use this benchmark, cite this repository and the original XRF V2 dataset source.

```bibtex
@software{agac2026xrfv2edgetal,
  title = {XRF V2 Edge TAL Benchmark},
  author = {Agac, Sumeyye and contributors},
  year = {2026},
  url = {https://github.com/sumeyye-agac/wearable-edge-xrfv2-benchmark}
}
```

## Scope

Current scope is IMU-centric TAL. Wi-Fi and video streams are intentionally out of scope for v1.

## Roadmap

- knowledge distillation hooks (teacher all modalities -> student subset)
- INT8 quantization benchmarking on edge CPUs
- richer ablations for modality failure modes

## License

MIT
