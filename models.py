from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field
from typing import List, Dict, Tuple
from numpy.typing import NDArray


class SorterAction(Action):

    segment: Dict[str, Tuple[int, int, int, bool]] = Field(default_factory=dict)
    adjust: Dict[str, Tuple[str, int]] = Field(default_factory=dict)
    place: Dict[str, Tuple[int, int, int, bool]] = Field(default_factory=dict)


class SorterObservation(Observation):

    grid_dims: Tuple[int, int, int]
    weighted_grid: NDArray
    current_grid: NDArray
    objects_present: Dict[str, Tuple[int, int, int, bool]] = Field(default_factory=dict)
    positions_place: Dict[str, Tuple[int, int, int, bool]] = Field(default_factory=dict)
    positions_segment: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict
    )
    positions_adjust: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict
    )
    reward: Tuple[List[float], List[str]] = Field(default_factory=lambda: ([], []))
    positions: List[Tuple[int, int, int, bool]] = Field(default_factory=list)
    done: bool = Field(default=False)


class SorterState(State, SorterObservation):

    grid_dims: Tuple[int, int, int]
    weighted_grid: NDArray
    objects_present: Dict[str, Tuple[int, int, int, bool]] = Field(default_factory=dict)
    current_grid: NDArray
    positions_place: Dict[str, Tuple[int, int, int, bool]] = Field(default_factory=dict)
    positions_segment: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict
    )
    positions_adjust: Dict[str, Tuple[int, int, int, bool]] = Field(
        default_factory=dict
    )
    reward: Tuple[List[float], List[str]] = Field(default_factory=lambda: ([], []))
    done: bool = Field(default=False)
