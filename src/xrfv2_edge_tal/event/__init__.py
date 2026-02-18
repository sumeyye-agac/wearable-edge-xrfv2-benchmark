"""Event-detection utilities."""

from xrfv2_edge_tal.event.metrics import compute_event_metrics
from xrfv2_edge_tal.event.trigger import frame_probs_to_event_triggers

__all__ = ["compute_event_metrics", "frame_probs_to_event_triggers"]
