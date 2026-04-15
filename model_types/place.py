from typing import Dict

from pydantic import Field

from .segment import PositionTuple


PlaceActionPayload = Dict[str, PositionTuple]


class PlaceActionMixin:
    place: PlaceActionPayload = Field(default_factory=dict)


class PlaceObservationMixin:
    positions_place: Dict[str, PositionTuple] = Field(default_factory=dict)
