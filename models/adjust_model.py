from typing import Dict, Tuple

from pydantic import Field

from .segment_model import PositionTuple


AdjustActionPayload = Dict[str, Tuple[str, int]]


class AdjustActionMixin:
    adjust: AdjustActionPayload = Field(default_factory=dict)


class AdjustObservationMixin:
    positions_adjust: Dict[str, PositionTuple] = Field(default_factory=dict)
