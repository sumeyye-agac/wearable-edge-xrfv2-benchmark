"""Temporal NMS for segment predictions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from xrfv2_edge_tal.metrics.tal_map import segment_tiou

SegmentLike = dict[str, Any]


def _nms_one_class(segments: list[SegmentLike], tiou_threshold: float) -> list[SegmentLike]:
    ordered = sorted(segments, key=lambda x: float(x.get("score", 0.0)), reverse=True)
    keep: list[SegmentLike] = []

    while ordered:
        current = ordered.pop(0)
        keep.append(current)
        ordered = [seg for seg in ordered if segment_tiou(current, seg) < tiou_threshold]

    return keep


def temporal_nms(
    segments: list[SegmentLike],
    tiou_threshold: float = 0.5,
    classwise: bool = True,
) -> list[SegmentLike]:
    if not segments:
        return []

    if not classwise:
        return _nms_one_class(segments, tiou_threshold)

    by_label: dict[int, list[SegmentLike]] = defaultdict(list)
    for seg in segments:
        by_label[int(seg["label"])].append(seg)

    out: list[SegmentLike] = []
    for _, group in sorted(by_label.items(), key=lambda kv: kv[0]):
        out.extend(_nms_one_class(group, tiou_threshold))

    return sorted(out, key=lambda x: float(x.get("score", 0.0)), reverse=True)
