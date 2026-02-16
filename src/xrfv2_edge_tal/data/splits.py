"""Split creation helpers."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np


def create_default_split(
    manifest: list[dict[str, Any]], seed: int = 42, subject_stratified: bool = True
) -> dict[str, list[str]]:
    """Create train/val/test split from manifest entries.

    If subject IDs are present and `subject_stratified=True`, split by subject to reduce leakage.
    """
    if not manifest:
        return {"train": [], "val": [], "test": []}

    has_subject = subject_stratified and all("subject_id" in m and m["subject_id"] is not None for m in manifest)
    rng = np.random.default_rng(seed)

    if has_subject:
        subject_to_ids: dict[str, list[str]] = defaultdict(list)
        for row in manifest:
            subject_to_ids[str(row["subject_id"])].append(str(row["sample_id"]))

        subjects = np.array(list(subject_to_ids.keys()))
        rng.shuffle(subjects)

        n = len(subjects)
        n_train = max(1, int(round(0.7 * n)))
        n_val = max(1, int(round(0.15 * n))) if n >= 3 else 0
        if n_train + n_val >= n:
            n_val = max(0, n - n_train - 1)

        train_subjects = set(subjects[:n_train])
        val_subjects = set(subjects[n_train : n_train + n_val])
        test_subjects = set(subjects[n_train + n_val :])
        if not test_subjects and subjects.size > 0:
            test_subjects = {subjects[-1]}
            train_subjects.discard(subjects[-1])

        split = {"train": [], "val": [], "test": []}
        for subject, ids in subject_to_ids.items():
            if subject in train_subjects:
                split["train"].extend(ids)
            elif subject in val_subjects:
                split["val"].extend(ids)
            elif subject in test_subjects:
                split["test"].extend(ids)
        return split

    ids = np.array([str(row["sample_id"]) for row in manifest])
    rng.shuffle(ids)
    n = len(ids)
    n_train = max(1, int(round(0.7 * n)))
    n_val = max(1, int(round(0.15 * n))) if n >= 3 else 0
    if n_train + n_val >= n:
        n_val = max(0, n - n_train - 1)

    return {
        "train": ids[:n_train].tolist(),
        "val": ids[n_train : n_train + n_val].tolist(),
        "test": ids[n_train + n_val :].tolist(),
    }


def create_lopo_splits(manifest: list[dict[str, Any]]) -> list[dict[str, list[str]]]:
    """Create leave-one-participant-out splits if subject IDs are available."""
    if not manifest:
        return []
    if not all("subject_id" in m and m["subject_id"] is not None for m in manifest):
        return []

    subject_to_ids: dict[str, list[str]] = defaultdict(list)
    for row in manifest:
        subject_to_ids[str(row["subject_id"])].append(str(row["sample_id"]))

    subjects = sorted(subject_to_ids.keys())
    out: list[dict[str, list[str]]] = []
    for held_out in subjects:
        train_ids: list[str] = []
        test_ids = list(subject_to_ids[held_out])
        for subject in subjects:
            if subject == held_out:
                continue
            train_ids.extend(subject_to_ids[subject])
        out.append({"held_out_subject": held_out, "train": train_ids, "val": [], "test": test_ids})
    return out
