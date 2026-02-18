# XRF V2 Dataset Access

This repository does **not** redistribute XRF V2.

## Source

Obtain the dataset from the **Official XRFV2 repo** and linked release channels.

## Expected Local Directory

Point CLI commands to a directory containing:
- `train_data.h5`
- `train_label.json`
- `test_data.h5`
- `test_label.json`
- `info.json`

Example:

```text
data/raw/xrfv2/
  train_data.h5
  train_label.json
  test_data.h5
  test_label.json
  info.json
```

## IMU Scope In This Repo

XRF V2 may include IMU from phone/watch/earbuds/glasses. This flagship track defaults to:
- `earbuds_glasses` (primary)
- `glasses_only` (fallback)

`all_imu` can be used only as a diagnostic upper-bound profile.

## Validation Behavior

`XRFV2H5Adapter` validates required files and raises actionable errors for:
- missing required files
- unresolved modality datasets in H5
- unsupported label schema

Supported label segment formats include:
- `[start, end, label]`
- `[start, end, score, label]` (score ignored)
- dict with `start`, `end`, `label`
- nested modality-specific lists

## Sample Rate Note

Earbuds streams can differ from others (often 25Hz vs 50Hz). The pipeline normalizes data to a canonical frame representation for training/eval.

## Redistribution Policy

Never commit raw data or generated runs. `.gitignore` excludes `data/`, `runs/`, checkpoints, ONNX, and large artifacts.
