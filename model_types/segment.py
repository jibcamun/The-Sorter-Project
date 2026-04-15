from typing import Any, Dict, List, Tuple

from pydantic import Field


PositionTuple = Tuple[int, int, int, bool]
SegmentActionPayload = Dict[str, PositionTuple]


class SegmentActionMixin:
    segment: SegmentActionPayload = Field(default_factory=dict)


class SegmentObservationMixin:
    positions_segment: Dict[str, PositionTuple] = Field(default_factory=dict)
    positions: List[PositionTuple] = Field(default_factory=list)
    observed_objects: List[Dict[str, Any]] = Field(default_factory=list)
    last_segment_attempt: Dict[str, PositionTuple] = Field(default_factory=dict)
