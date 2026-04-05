from typing import Any, Dict, List, Tuple

from pydantic import Field

from .segment_model import PositionTuple


AdjustActionPayload = Tuple[str, int] | Tuple[()]


class AdjustActionMixin:
    adjust: AdjustActionPayload = Field(default_factory=tuple)


class AdjustObservationMixin:
    positions_adjust: Dict[str, PositionTuple] = Field(default_factory=dict)
    adjustable_objects: List[Dict[str, Any]] = Field(default_factory=list)
    adjust_focus_object: str = Field(default="")
    adjust_start_position: PositionTuple | Tuple[()] = Field(default_factory=tuple)
    adjust_visited_positions: List[Tuple[int, int, int]] = Field(default_factory=list)
    adjust_action_options: List[List[Any]] = Field(default_factory=list)
