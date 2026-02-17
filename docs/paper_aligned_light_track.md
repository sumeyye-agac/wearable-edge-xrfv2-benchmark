# Paper-Aligned Lightweight Track

This track keeps the repository edge-first and architecture-light, while aligning major protocol choices with the XRFV2 paper:

- temporal window protocol
- overlap-based evaluation
- fixed-length temporal processing
- segment coverage filtering

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

## Key Knobs

- `train.paper_track.clip_len`
- `train.paper_track.stride`
- `train.paper_track.resample_to`
- `train.paper_track.min_segment_coverage`
- `train.lr_schedule` (`constant` or `cosine`)
