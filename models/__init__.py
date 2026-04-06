from typing import Dict, List, Tuple

from numpy.typing import NDArray
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field, field_serializer

from .adjust_model import AdjustActionMixin, AdjustObservationMixin
from .place_model import PlaceActionMixin, PlaceObservationMixin
from .segment_model import PositionTuple, SegmentActionMixin, SegmentObservationMixin


class SorterAction(
    Action,
    SegmentActionMixin,
    AdjustActionMixin,
    PlaceActionMixin,
):
    pass


class SorterObservation(
    Observation,
    PlaceObservationMixin,
    SegmentObservationMixin,
    AdjustObservationMixin,
):
    grid_dims: Tuple[int, int, int]
    weighted_grid: NDArray
    current_grid: NDArray
    objects_present: Dict[str, PositionTuple] = Field(default_factory=dict)
    reward: float = Field(default=0.0)
    reward_details: Tuple[List[float], List[str]] = Field(
        default_factory=lambda: ([], [])
    )
    advisory: List[str] = Field(default_factory=list)
    done: bool = Field(default=False)

    @field_serializer("weighted_grid", "current_grid")
    def _serialize_ndarray(self, value: NDArray):
        return value.tolist()


class SorterState(
    State,
    PlaceObservationMixin,
    SegmentObservationMixin,
    AdjustObservationMixin,
):
    grid_dims: Tuple[int, int, int]
    weighted_grid: NDArray
    current_grid: NDArray
    objects_present: Dict[str, PositionTuple] = Field(default_factory=dict)
    reward: Tuple[List[float], List[str]] = Field(default_factory=lambda: ([], []))
    advisory: List[str] = Field(default_factory=list)
    done: bool = Field(default=False)

    @field_serializer("weighted_grid", "current_grid")
    def _serialize_ndarray(self, value: NDArray):
        return value.tolist()
