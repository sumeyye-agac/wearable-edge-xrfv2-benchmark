from __future__ import annotations

from pathlib import Path

from xrfv2_edge_tal.data.prepare import compute_dataset_fingerprint


def test_compute_dataset_fingerprint_uses_full_and_sampled_hash_modes(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir(parents=True, exist_ok=True)

    small = data_root / "small.bin"
    large = data_root / "large.bin"
    small.write_bytes(b"a" * 128)
    large.write_bytes(b"b" * 8192)

    manifest = [{"sample_id": "0", "modalities": ["airpods", "imu_gl"]}]
    fp = compute_dataset_fingerprint(
        data_root=data_root,
        manifest=manifest,
        max_full_hash_bytes=1024,
    )

    files = {entry["name"]: entry for entry in fp["files"]}
    assert files["small.bin"]["hash_mode"] == "full_sha256"
    assert files["large.bin"]["hash_mode"] == "sampled_sha256_head_tail_1mb"
    assert isinstance(files["small.bin"]["sha256"], str) and files["small.bin"]["sha256"]
    assert isinstance(files["large.bin"]["sha256"], str) and files["large.bin"]["sha256"]
    assert fp["max_full_hash_bytes"] == 1024
