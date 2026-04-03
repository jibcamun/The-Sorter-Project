from typing import Dict, Tuple

from pydantic import Field

from .segment_model import PositionTuple


AdjustActionPayload = Tuple[str, str, int] | Tuple[()]


class AdjustActionMixin:
    adjust: AdjustActionPayload = Field(default_factory=tuple)


class AdjustObservationMixin:
    positions_adjust: Dict[str, PositionTuple] = Field(default_factory=dict)
