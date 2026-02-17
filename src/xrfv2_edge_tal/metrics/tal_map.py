"""Temporal Action Localization mAP metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

SegmentLike = dict[str, Any]
DEFAULT_THRESHOLDS = np.arange(0.5, 1.0, 0.05)


def segment_tiou(a: SegmentLike, b: SegmentLike) -> float:
    """Temporal IoU between two segments with `start` and `end`."""
    start_a, end_a = float(a["start"]), float(a["end"])
    start_b, end_b = float(b["start"]), float(b["end"])

    inter = max(0.0, min(end_a, end_b) - max(start_a, start_b))
    union = max(end_a, end_b) - min(start_a, start_b)
    if union <= 0.0:
        return 0.0
    return float(inter / union)


def _interpolated_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    mrec = np.concatenate([[0.0], recalls, [1.0]])
    mpre = np.concatenate([[0.0], precisions, [0.0]])

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = max(mpre[i - 1], mpre[i])

    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def ap_at_tiou(preds: list[SegmentLike], gts: list[SegmentLike], tiou: float) -> float:
    """Compute class-averaged AP at a single tIoU threshold.

    Input schema:
    - `preds`: list of dicts with keys: `sample_id`, `label`, `start`, `end`, `score`
    - `gts`:   list of dicts with keys: `sample_id`, `label`, `start`, `end`
    """
    if not gts:
        return 0.0

    labels = sorted({int(gt["label"]) for gt in gts})
    ap_values: list[float] = []

    for label in labels:
        gt_label = [gt for gt in gts if int(gt["label"]) == label]
        pred_label = [p for p in preds if int(p["label"]) == label]

        if not gt_label:
            continue

        gt_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for gt in gt_label:
            gt_by_sample[str(gt["sample_id"])].append({**gt, "matched": False})

        pred_sorted = sorted(pred_label, key=lambda p: float(p.get("score", 0.0)), reverse=True)

        tp = np.zeros(len(pred_sorted), dtype=np.float64)
        fp = np.zeros(len(pred_sorted), dtype=np.float64)

        for idx, pred in enumerate(pred_sorted):
            sample_id = str(pred["sample_id"])
            candidates = gt_by_sample.get(sample_id, [])

            best_iou = 0.0
            best_gt_idx = -1
            for c_idx, candidate in enumerate(candidates):
                if candidate["matched"]:
                    continue
                iou = segment_tiou(pred, candidate)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = c_idx

            if best_gt_idx >= 0 and best_iou >= tiou:
                tp[idx] = 1.0
                candidates[best_gt_idx]["matched"] = True
            else:
                fp[idx] = 1.0

        if len(pred_sorted) == 0:
            ap_values.append(0.0)
            continue

        tp_cum = np.cumsum(tp)
        fp_cum = np.cumsum(fp)
        recalls = tp_cum / max(len(gt_label), 1)
        precisions = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
        ap_values.append(_interpolated_ap(recalls, precisions))

    if not ap_values:
        return 0.0
    return float(np.mean(ap_values))


def ap_by_class_at_tiou(
    preds: list[SegmentLike],
    gts: list[SegmentLike],
    tiou: float,
) -> dict[int, float]:
    """Return AP per class at one tIoU threshold."""
    if not gts:
        return {}

    labels = sorted({int(gt["label"]) for gt in gts})
    out: dict[int, float] = {}
    for label in labels:
        class_ap = ap_at_tiou(
            preds=[p for p in preds if int(p["label"]) == label],
            gts=[g for g in gts if int(g["label"]) == label],
            tiou=tiou,
        )
        out[label] = float(class_ap)
    return out


def match_predictions_at_tiou(
    preds: list[SegmentLike],
    gts: list[SegmentLike],
    tiou: float,
) -> dict[str, Any]:
    """Greedy confidence-sorted matching summary at one tIoU threshold."""
    preds_sorted = sorted(preds, key=lambda p: float(p.get("score", 0.0)), reverse=True)
    gt_by_key: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for gt in gts:
        key = (str(gt["sample_id"]), int(gt["label"]))
        gt_by_key[key].append({**gt, "matched": False})

    tp = 0
    fp = 0
    matched_tious: list[float] = []
    for pred in preds_sorted:
        key = (str(pred["sample_id"]), int(pred["label"]))
        candidates = gt_by_key.get(key, [])

        best_iou = 0.0
        best_idx = -1
        for idx, candidate in enumerate(candidates):
            if candidate["matched"]:
                continue
            iou = segment_tiou(pred, candidate)
            if iou > best_iou:
                best_iou = iou
                best_idx = idx

        if best_idx >= 0 and best_iou >= tiou:
            tp += 1
            candidates[best_idx]["matched"] = True
            matched_tious.append(float(best_iou))
        else:
            fp += 1

    fn = len(gts) - tp
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 0.0
    if precision + recall > 0:
        f1 = 2.0 * precision * recall / (precision + recall)

    tiou_stats = {
        "count": len(matched_tious),
        "mean": float(np.mean(matched_tious)) if matched_tious else 0.0,
        "p50": float(np.percentile(matched_tious, 50)) if matched_tious else 0.0,
        "p90": float(np.percentile(matched_tious, 90)) if matched_tious else 0.0,
        "max": float(np.max(matched_tious)) if matched_tious else 0.0,
    }
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_tiou": tiou_stats,
    }


def map_over_thresholds(
    preds: list[SegmentLike],
    gts: list[SegmentLike],
    thresholds: np.ndarray | list[float] | None = None,
) -> dict[str, Any]:
    if thresholds is None:
        thresholds = DEFAULT_THRESHOLDS
    thrs = [float(x) for x in thresholds]
    per_threshold = {f"{thr:.2f}": ap_at_tiou(preds, gts, thr) for thr in thrs}
    mean_ap = float(np.mean(list(per_threshold.values()))) if per_threshold else 0.0
    return {
        "thresholds": thrs,
        "ap": per_threshold,
        "map_avg": mean_ap,
    }
