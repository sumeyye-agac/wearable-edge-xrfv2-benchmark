# Paper-Aligned Lightweight Track

This track keeps the repository edge-first and architecture-light, while aligning major protocol choices with the XRFV2 paper:

- temporal window protocol
- overlap-based evaluation
- fixed-length temporal processing
- segment coverage filtering

Reference paper:
- XRFV2 benchmark paper: `https://arxiv.org/html/2501.19034v2`

## Design Choice

The original paper reports strongest numbers with larger sequence models.  
This track intentionally uses lightweight baselines (`TinyTCN`, `TinyTransformer`) for deployability:

- fewer parameters
- lower latency
- straightforward export path

## Configs

- `configs/paper_light_tiny_tcn.yaml`
- `configs/paper_light_tiny_transformer.yaml`

## Run

```bash
xrfv2-edge-tal train --config configs/paper_light_tiny_tcn.yaml --adapter xrfv2 --data-root /path/to/xrfv2 --seed 42
xrfv2-edge-tal eval --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/paper_light_tiny_tcn.yaml --adapter xrfv2 --data-root /path/to/xrfv2 --seed 42
xrfv2-edge-tal benchmark --checkpoint runs/<TRAIN_RUN>/checkpoints/last.npz --config configs/paper_light_tiny_tcn.yaml --seed 42
```

## Full-Run Timing (Observed 2026-02-17)

- TinyTCN full train (`8` epochs, full `9660` samples): about `18.8` minutes
- TinyTransformer full train (`8` epochs, full `9660` samples): about `35.4` minutes
- Full eval (all samples): about `2.3` to `3.3` minutes

## Results Snapshot

- Detailed report: `docs/results_paper_light_full_2026-02-17.md`
- Machine-readable metrics: `docs/results_paper_light_full_2026-02-17.json`

## Key Knobs

- `train.paper_track.clip_len`
- `train.paper_track.stride`
- `train.paper_track.resample_to`
- `train.paper_track.min_segment_coverage`
- `train.lr_schedule` (`constant` or `cosine`)
