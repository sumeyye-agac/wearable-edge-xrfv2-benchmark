#!/usr/bin/env python3
"""Probe XRFV2 H5 file structure and basic stats."""

from __future__ import annotations

import argparse
import sys

from xrfv2_edge_tal.data.probe import probe_xrfv2_h5_json


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe XRFV2 H5 modalities and shapes")
    parser.add_argument("--data-root", required=True, help="Path containing train_data.h5/etc")
    parser.add_argument("--sample-index", type=int, default=0, help="Train sample index to inspect")
    args = parser.parse_args()

    try:
        print(probe_xrfv2_h5_json(data_root=args.data_root, sample_index=args.sample_index))
    except Exception as exc:  # pragma: no cover - CLI surface
        print(f"Probe failed: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
