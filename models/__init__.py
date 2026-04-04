from typing import Dict, List, Tuple

from numpy.typing import NDArray
from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field

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
    reward: Tuple[List[float], List[str]] = Field(default_factory=lambda: ([], []))
    advisory: List[str] = Field(default_factory=list)
    done: bool = Field(default=False)


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
