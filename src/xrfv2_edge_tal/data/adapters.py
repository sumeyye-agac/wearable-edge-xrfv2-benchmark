"""Dataset adapters for synthetic and XRFV2 H5 sources."""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import h5py
import numpy as np

Segment = dict[str, float | int]


class RawAdapter(ABC):
    """Abstract interface for raw sequence adapters."""

    @property
    @abstractmethod
    def modalities(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def split_ids(self, split: str) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_sample(
        self, sample_id: str, split: str
    ) -> tuple[dict[str, np.ndarray], list[Segment], dict[str, Any]]:
        raise NotImplementedError


class DummyAdapter(RawAdapter):
    """Synthetic adapter that requires no external dataset."""

    def __init__(
        self,
        seed: int = 42,
        num_train: int = 24,
        num_test: int = 8,
        seq_len: int = 160,
        feat_dim: int = 6,
        num_classes: int = 5,
        modalities: list[str] | None = None,
    ) -> None:
        self._rng = np.random.default_rng(seed)
        self._num_classes = num_classes
        self._seq_len = seq_len
        self._feat_dim = feat_dim
        self._modalities = modalities or ["imu_phone", "imu_watch", "imu_earbuds", "imu_glasses"]
        self._data: dict[str, dict[str, dict[str, Any]]] = {"train": {}, "test": {}}

        self._build_split("train", num_train)
        self._build_split("test", num_test)

    @property
    def modalities(self) -> list[str]:
        return list(self._modalities)

    def _build_split(self, split: str, count: int) -> None:
        for idx in range(count):
            sample_id = str(idx)
            subject_id = f"subject_{idx % 4}"
            segments = self._sample_segments()
            sample: dict[str, Any] = {
                "subject_id": subject_id,
                "segments": segments,
                "x": {},
            }
            for modality in self._modalities:
                x = self._rng.normal(0.0, 0.2, size=(self._seq_len, self._feat_dim)).astype(
                    np.float32
                )
                for seg in segments:
                    start = int(seg["start"])
                    end = int(seg["end"])
                    label = int(seg["label"])
                    x[start:end, :] += label * 0.3
                sample["x"][modality] = x
            self._data[split][sample_id] = sample

    def _sample_segments(self) -> list[Segment]:
        out: list[Segment] = []
        n_segments = int(self._rng.integers(1, 4))
        for _ in range(n_segments):
            start = int(self._rng.integers(0, self._seq_len - 20))
            duration = int(self._rng.integers(8, 30))
            end = min(self._seq_len, start + duration)
            label = int(self._rng.integers(1, self._num_classes))
            out.append({"start": float(start), "end": float(end), "label": label})
        out.sort(key=lambda x: float(x["start"]))
        return out

    def split_ids(self, split: str) -> list[str]:
        self._require_split(split)
        return list(self._data[split].keys())

    def get_sample(
        self, sample_id: str, split: str
    ) -> tuple[dict[str, np.ndarray], list[Segment], dict[str, Any]]:
        self._require_split(split)
        if sample_id not in self._data[split]:
            raise KeyError(f"Unknown sample_id={sample_id} for split={split}")

        sample = self._data[split][sample_id]
        x = {k: np.array(v, copy=True) for k, v in sample["x"].items()}
        segments = [dict(s) for s in sample["segments"]]
        meta = {"subject_id": sample["subject_id"]}
        return x, segments, meta

    @staticmethod
    def _require_split(split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be one of train/test, got: {split}")


class XRFV2H5Adapter(RawAdapter):
    """Loader for XRFV2 Kaggle/SDP-style H5 + JSON format."""

    REQUIRED_FILES = {
        "train": ["train_data.h5", "train_label.json"],
        "test": ["test_data.h5", "test_label.json"],
        "meta": ["info.json"],
    }

    def __init__(self, data_root: str | Path) -> None:
        self.data_root = Path(data_root)
        self._validate_required_files()

        info_path = self.data_root / "info.json"
        self.info = self._load_json(info_path)
        self._modalities = self._infer_modalities(self.info)

        self._labels: dict[str, dict[str, list[Segment]]] = {
            "train": self._load_label_file(self.data_root / "train_label.json"),
            "test": self._load_label_file(self.data_root / "test_label.json"),
        }

        self._counts = {
            "train": self._sample_count(self.data_root / "train_data.h5"),
            "test": self._sample_count(self.data_root / "test_data.h5"),
        }

    @property
    def modalities(self) -> list[str]:
        return list(self._modalities)

    def split_ids(self, split: str) -> list[str]:
        self._require_split(split)
        return [str(i) for i in range(self._counts[split])]

    def get_sample(
        self, sample_id: str, split: str
    ) -> tuple[dict[str, np.ndarray], list[Segment], dict[str, Any]]:
        self._require_split(split)
        try:
            idx = int(sample_id)
        except ValueError as exc:
            raise ValueError(f"sample_id must be an integer-like string, got: {sample_id}") from exc

        if idx < 0 or idx >= self._counts[split]:
            raise IndexError(f"sample_id out of range for split {split}: {sample_id}")

        h5_path = self.data_root / f"{split}_data.h5"
        x: dict[str, np.ndarray] = {}
        with h5py.File(h5_path, "r") as h5f:
            for modality in self._modalities:
                dataset = self._resolve_modality_dataset(h5f, modality)
                arr = np.asarray(dataset[idx], dtype=np.float32)
                # Normalize modality tensors to [T, D] for baseline models.
                if arr.ndim < 2:
                    arr = arr.reshape(-1, 1)
                elif arr.ndim > 2:
                    arr = arr.reshape(arr.shape[0], -1)
                x[modality] = arr

        segments = self._labels[split].get(str(idx), [])
        meta = {"sample_id": str(idx), "source": split}
        return x, [dict(s) for s in segments], meta

    def _validate_required_files(self) -> None:
        missing: list[str] = []
        for group in self.REQUIRED_FILES.values():
            for name in group:
                if not (self.data_root / name).exists():
                    missing.append(name)
        if missing:
            joined = ", ".join(sorted(missing))
            raise FileNotFoundError(
                f"XRFV2 directory is missing required files: {joined}. "
                "Expected train_data.h5/train_label.json/test_data.h5/test_label.json/info.json"
            )

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"Expected JSON object in {path}, got {type(raw)!r}")
        return raw

    def _infer_modalities(self, info: dict[str, Any]) -> list[str]:
        candidates = [
            info.get("modalities"),
            info.get("modality_list"),
            info.get("sensors"),
        ]
        for candidate in candidates:
            if isinstance(candidate, list) and all(isinstance(x, str) for x in candidate):
                return list(candidate)

        train_h5 = self.data_root / "train_data.h5"
        with h5py.File(train_h5, "r") as h5f:
            keys = self._collect_h5_paths(h5f)
        if not keys:
            raise ValueError("No datasets found in train_data.h5 to infer modalities")
        return keys

    def _sample_count(self, h5_path: Path) -> int:
        with h5py.File(h5_path, "r") as h5f:
            first = self._resolve_modality_dataset(h5f, self._modalities[0])
            return int(first.shape[0])

    def _load_label_file(self, path: Path) -> dict[str, list[Segment]]:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return self._parse_label_payload(raw, path)

    def _parse_label_payload(self, raw: Any, source_path: Path) -> dict[str, list[Segment]]:
        if isinstance(raw, dict):
            # Modality-specific schema:
            # {"imu": {"0": [[...], ...]}, "wifi": {...}, "imu_file_names": {"0": "..."}}
            if raw and all(isinstance(v, dict) for v in raw.values()):
                merged_by_sample: dict[str, list[Segment]] = {}
                for top_key, per_sample in raw.items():
                    if str(top_key).endswith("_file_names"):
                        continue
                    for sample_id, payload in per_sample.items():
                        try:
                            segs = self._parse_segments(payload, source_path)
                        except ValueError:
                            # Ignore non-segment payloads in mixed schemas.
                            continue
                        merged_by_sample.setdefault(str(sample_id), []).extend(segs)
                if merged_by_sample:
                    return merged_by_sample

            out: dict[str, list[Segment]] = {}
            for key, value in raw.items():
                if isinstance(value, dict):
                    # modality-specific labels: merge all segment lists.
                    merged: list[Segment] = []
                    for maybe_segments in value.values():
                        merged.extend(self._parse_segments(maybe_segments, source_path))
                    out[str(key)] = merged
                else:
                    out[str(key)] = self._parse_segments(value, source_path)
            return out

        if isinstance(raw, list):
            out = {}
            for idx, entry in enumerate(raw):
                out[str(idx)] = self._parse_segments(entry, source_path)
            return out

        raise ValueError(
            f"Unsupported label JSON structure in {source_path}. "
            "Expected dict or list of segment entries."
        )

    def _parse_segments(self, value: Any, source_path: Path) -> list[Segment]:
        if value is None:
            return []
        if isinstance(value, dict):
            if {"start", "end", "label"}.issubset(value.keys()):
                return [
                    {
                        "start": float(value["start"]),
                        "end": float(value["end"]),
                        "label": int(value["label"]),
                    }
                ]
            # nested dict that still contains segments by key
            merged: list[Segment] = []
            for inner in value.values():
                merged.extend(self._parse_segments(inner, source_path))
            return merged

        if isinstance(value, list):
            if len(value) in {3, 4} and all(
                isinstance(v, (int, float, np.integer, np.floating)) for v in value
            ):
                start, end = float(value[0]), float(value[1])
                label = int(value[2] if len(value) == 3 else value[3])
                return [{"start": start, "end": end, "label": label}]
            out: list[Segment] = []
            for item in value:
                out.extend(self._parse_segments(item, source_path))
            return out

        if isinstance(value, (tuple, np.ndarray)):
            arr = list(value)
            if len(arr) not in {3, 4}:
                raise ValueError(
                    f"Invalid label tuple length={len(arr)} in {source_path}; expected 3 or 4 values"
                )
            start, end = float(arr[0]), float(arr[1])
            label = int(arr[2] if len(arr) == 3 else arr[3])
            return [{"start": start, "end": end, "label": label}]

        if isinstance(value, (int, float, str)):
            raise ValueError(
                f"Unrecognized scalar label value {value!r} in {source_path}; expected segment list/tuple/dict"
            )

        raise ValueError(
            f"Unsupported label item type {type(value)!r} in {source_path}; expected dict/list/tuple"
        )

    @staticmethod
    def _collect_h5_paths(group: h5py.Group, prefix: str = "") -> list[str]:
        out: list[str] = []
        for key, obj in group.items():
            path = f"{prefix}/{key}" if prefix else key
            if isinstance(obj, h5py.Dataset):
                out.append(path)
            elif isinstance(obj, h5py.Group):
                out.extend(XRFV2H5Adapter._collect_h5_paths(obj, path))
        return out

    def _resolve_modality_dataset(self, h5f: h5py.File, modality: str) -> h5py.Dataset:
        if modality in h5f and isinstance(h5f[modality], h5py.Dataset):
            return h5f[modality]

        for path in self._collect_h5_paths(h5f):
            if path.endswith(f"/{modality}") or path == modality:
                obj = h5f[path]
                if isinstance(obj, h5py.Dataset):
                    return obj

        available = ", ".join(self._collect_h5_paths(h5f))
        raise KeyError(
            f"Modality '{modality}' not found in {h5f.filename}. Available datasets: {available}"
        )

    @staticmethod
    def _require_split(split: str) -> None:
        if split not in {"train", "test"}:
            raise ValueError(f"split must be one of train/test, got: {split}")
