"""Event-detection utilities."""

from xrfv2_edge_tal.event.calibrate_event import calibrate_event_main
from xrfv2_edge_tal.event.metrics import compute_event_metrics
from xrfv2_edge_tal.event.trigger import frame_probs_to_event_triggers

__all__ = ["calibrate_event_main", "compute_event_metrics", "frame_probs_to_event_triggers"]
