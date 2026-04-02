from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import SorterAction, SorterObservation, SorterState
    from ..utils.grids import init_grid, weighted_grid
    from ..tasks.segment import segment
    from ..tasks.adjust import adjust
    from ..tasks.place import place
except ImportError:
    from models import SorterAction, SorterObservation, SorterState
    from utils.grids import init_grid, weighted_grid
    from tasks.segment import segment
    from tasks.adjust import adjust
    from tasks.place import place


class SorterEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def _return_observation(self, state: SorterState):

        return SorterObservation(
            grid_dims=state.grid_dims,
            weighted_grid=state.weighted_grid,
            current_grid=state.current_grid,
            reward=state.reward,
            done=state.done,
            positions_place=state.positions_place,
            positions_adjust=state.positions_adjust,
            positions_segment=state.positions_segment,
            positions=list(state.objects_present.values()),
        )

    def __init__(self, task):

        fresh_grid, objs_present = init_grid()
        grid_dims = tuple(int(dim) for dim in fresh_grid.shape)
        self.task = task
        self.step_count = 0
        self._state = SorterState(
            step_count=0,
            grid_dims=grid_dims,
            current_grid=fresh_grid,
            done=False,
            reward=([], []),
            weighted_grid=weighted_grid(grid_dims),
            objects_present=objs_present,
            positions_place={},
            positions_adjust={},
            positions_segment={},
        )
        self._reset_count = 0

    def reset(self) -> SorterObservation:

        fresh_grid, objs_present = init_grid()
        grid_dims = tuple(int(dim) for dim in fresh_grid.shape)

        self._state = SorterState(
            step_count=0,
            grid_dims=grid_dims,
            current_grid=fresh_grid,
            done=False,
            reward=([], []),
            weighted_grid=weighted_grid(grid_dims),
            objects_present=objs_present,
            positions_place={},
            positions_adjust={},
            positions_segment={},
        )

        self._reset_count += 1

        return self._return_observation(self._state)

    def step(self, action: SorterAction) -> SorterObservation:
        self.step_count += 1
        self._state.step_count += 1

        if self.task == "segment":
            segment(self._state, action.segment)
        if self.task == "adjust":
            adjust(self._state, action.adjust)
        if self.task == "place":
            place(self._state, action.place)

        return self._return_observation(self._state)

    @property
    def state(self) -> State:
        return self._state
