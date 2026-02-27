# AI Agent Instructions

This repository is centered on one flagship track:

- **Deploy-ready wearable event detection on XRF V2**
- default product profile: `earbuds_glasses` (`airpods + imu_gl`)
- fallback profile: `glasses_only`

## Core commands

- install: `pip install -e ".[dev]"`
- lint: `ruff check .`
- format: `ruff format .`
- tests: `pytest -q`

## Data contract

Expected raw folder:

```text
data/raw/xrfv2_kaggle/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

Use:

- `xrfv2-edge-tal inspect --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --list-modalities --show-shapes`

## Event pipeline commands

- train: `xrfv2-edge-tal event-train --config configs/event_presence_mobility.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --profile glasses_only`
- eval: `xrfv2-edge-tal event-eval --config configs/event_presence_mobility.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --checkpoint runs/<run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only`
- calibrate: `xrfv2-edge-tal event-calibrate --config configs/event_presence_mobility.yaml --adapter xrfv2 --data-root data/raw/xrfv2_kaggle --checkpoint runs/<run_id>/checkpoints/last.npz --profiles earbuds_glasses,glasses_only --metric-mode sample_presence --fp-hour-budget 10`

## Output artifacts

All run artifacts are under `runs/<run_id>/`. Required files are documented in:

- `docs/artifact_contract.md`
- `docs/event/results_latest.md`
