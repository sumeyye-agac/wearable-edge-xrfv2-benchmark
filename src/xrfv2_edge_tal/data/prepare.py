"""Dataset preparation helpers for manifest, splits, and fingerprinting."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from xrfv2_edge_tal.data.adapters import RawAdapter
from xrfv2_edge_tal.data.splits import create_default_split, create_lopo_splits


def build_manifest(adapter: RawAdapter) -> list[dict[str, Any]]:
    manifest: list[dict[str, Any]] = []
    for split in ["train", "test"]:
        for sample_id in adapter.split_ids(split):
            x, _, meta = adapter.get_sample(sample_id, split)
            first_modality = next(iter(x.keys()))
            seq_len = int(x[first_modality].shape[0])
            manifest.append(
                {
                    "sample_id": sample_id,
                    "source_split": split,
                    "subject_id": meta.get("subject_id"),
                    "seq_len": seq_len,
                    "modalities": sorted(list(x.keys())),
                }
            )
    return manifest


def write_manifest_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compute_dataset_fingerprint(
    data_root: str | Path, manifest: list[dict[str, Any]]
) -> dict[str, Any]:
    root = Path(data_root)
    files = []
    for child in sorted(root.glob("*")):
        if child.is_file():
            files.append(
                {
                    "name": child.name,
                    "size_bytes": child.stat().st_size,
                    "sha256": _sha256_file(child),
                }
            )

    modalities = sorted({m for row in manifest for m in row.get("modalities", [])})
    return {
        "data_root": str(root),
        "num_samples": len(manifest),
        "modalities": modalities,
        "files": files,
    }


def prepare_dataset(
    adapter: RawAdapter, data_root: str | Path, output_dir: str | Path, seed: int = 42
) -> dict[str, Path]:
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest = build_manifest(adapter)
    manifest_path = out_dir / "manifest.jsonl"
    write_manifest_jsonl(manifest_path, manifest)

    split = create_default_split(manifest, seed=seed, subject_stratified=True)
    splits_dir = out_dir / "splits"
    splits_dir.mkdir(parents=True, exist_ok=True)
    default_split_path = splits_dir / "default.json"
    default_split_path.write_text(
        json.dumps(split, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lopo = create_lopo_splits(manifest)
    (splits_dir / "lopo.json").write_text(
        json.dumps(lopo, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    fingerprint = compute_dataset_fingerprint(data_root, manifest)
    fingerprint_path = out_dir / "dataset_fingerprint.json"
    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "manifest": manifest_path,
        "default_split": default_split_path,
        "fingerprint": fingerprint_path,
        "lopo": splits_dir / "lopo.json",
    }
