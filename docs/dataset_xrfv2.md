# XRF V2 Dataset Access

This repository does **not** redistribute XRF V2 data.

## Expected local directory

Point CLI commands to a directory containing:
- `train_data.h5`
- `train_label.json`
- `test_data.h5`
- `test_label.json`
- `info.json`

## Download options

1. Kaggle/official release pages used by XRFV2 maintainers
2. SDP/academic distribution channels referenced by official docs

## Validation behavior in this repo

`XRFV2H5Adapter` validates required files and raises clear errors if:
- files are missing
- modality keys cannot be resolved from H5
- label schema is malformed

Supported label shapes include:
- `[start, end, label]`
- `[start, end, score, label]` (score ignored)
- dict schema with `start`, `end`, `label`
- modality-specific nested label dictionaries

## Redistribution policy

Do not commit dataset files into git. `data/` and `runs/` artifacts are ignored by default.
