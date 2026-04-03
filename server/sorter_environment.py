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

    def _initial_state_kwargs(self, grid_dims, fresh_grid, objs_present):
        state_kwargs = {
            "step_count": 0,
            "grid_dims": grid_dims,
            "current_grid": fresh_grid,
            "done": False,
            "reward": ([], []),
            "weighted_grid": weighted_grid(grid_dims),
            # Internal evaluator source of truth. Exposed selectively via _state_kwargs.
            "objects_present": objs_present,
        }

        if self.task == "segment":
            state_kwargs.update(
                positions_segment={},
                positions=[],
            )
        elif self.task == "place":
            state_kwargs.update(
                positions_place={},
            )
        elif self.task == "adjust":
            state_kwargs.update(
                positions_adjust={},
            )

        return state_kwargs

    def _state_kwargs(self, state: SorterState):
        state_kwargs = {
            "episode_id": state.episode_id,
            "step_count": state.step_count,
            "grid_dims": state.grid_dims,
            "weighted_grid": state.weighted_grid,
            "current_grid": state.current_grid,
            "reward": state.reward,
            "done": state.done,
        }

        if self.task == "segment":
            state_kwargs.update(
                positions_segment=state.positions_segment,
                positions=list(state.objects_present.values()),
            )
        elif self.task == "place":
            state_kwargs.update(
                objects_present=state.objects_present,
                positions_place=state.positions_place,
            )
        elif self.task == "adjust":
            state_kwargs.update(
                objects_present=state.objects_present,
                positions_adjust=state.positions_adjust,
            )

        return state_kwargs

    def _return_observation(self, state: SorterState):
        observation_kwargs = self._state_kwargs(state)
        observation_kwargs.pop("episode_id", None)
        observation_kwargs.pop("step_count", None)
        return SorterObservation(**observation_kwargs)

    def __init__(self, task):

        fresh_grid, objs_present = init_grid()
        grid_dims = tuple(int(dim) for dim in fresh_grid.shape)
        self.task = task
        self.step_count = 0
        self._state = SorterState(
            **self._initial_state_kwargs(grid_dims, fresh_grid, objs_present)
        )
        self._reset_count = 0

    def reset(self) -> SorterObservation:

        fresh_grid, objs_present = init_grid()
        grid_dims = tuple(int(dim) for dim in fresh_grid.shape)

        self._state = SorterState(
            **self._initial_state_kwargs(grid_dims, fresh_grid, objs_present)
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
        return SorterState(**self._state_kwargs(self._state))
