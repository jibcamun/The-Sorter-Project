import os
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from models import SorterAction, SorterObservation, SorterState
    from config.objects import OBJECTS
    from utils.grids import init_grid, weighted_grid
    from tasks.segment import segment
    from tasks.adjust import (
        _position_score,
        adjust,
        build_adjust_candidates,
        top_k_legal_adjustment_positions,
    )
    from tasks.place import place
except ImportError:
    from ..models import SorterAction, SorterObservation, SorterState
    from ..config.objects import OBJECTS
    from ..utils.grids import init_grid, weighted_grid
    from ..tasks.segment import segment
    from ..tasks.adjust import (
        _position_score,
        adjust,
        build_adjust_candidates,
        top_k_legal_adjustment_positions,
    )
    from ..tasks.place import place


class SorterEnvironment(Environment):
    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    DEFAULT_TASK = "place"
    VALID_TASKS = {"segment", "place", "adjust"}

    def _segment_observed_objects(self, state: SorterState):
        observed_objects = []
        for obj_name, pos in sorted(
            state.objects_present.items(), key=lambda item: item[1][:3]
        ):
            dims = list(OBJECTS[obj_name]["dims"])
            stackable = bool(OBJECTS[obj_name]["stack"])
            observed_objects.append(
                {
                    "position": list(pos),
                    "dims": dims,
                    "stackable": stackable,
                    "volume": int(dims[0] * dims[1] * dims[2]),
                }
            )
        return observed_objects

    def _adjust_observed_objects(self, state: SorterState):
        adjustable_positions = build_adjust_candidates(state)
        observed_objects = []
        for obj_name, current_pos in adjustable_positions.items():
            if state.adjust_focus_object and state.adjust_focus_object != obj_name:
                continue
            current_score = _position_score(state, obj_name, tuple(current_pos))
            ranked_targets = top_k_legal_adjustment_positions(state, obj_name)
            observed_objects.append(
                {
                    "object_name": obj_name,
                    "current_position": list(current_pos),
                    "dims": list(OBJECTS[obj_name]["dims"]),
                    "legal_targets": [
                        {
                            "option_index": index,
                            "position": list(pos),
                            "projected_score": float(_position_score(state, obj_name, pos)),
                            "score_delta": float(
                                _position_score(state, obj_name, pos) - current_score
                            ),
                        }
                        for index, pos in enumerate(ranked_targets)
                    ],
                }
            )
        return observed_objects

    def _adjust_action_options(self, state: SorterState):
        options = []
        for item in self._adjust_observed_objects(state):
            for index, _target in enumerate(item.get("legal_targets", [])):
                options.append([item["object_name"], index])
        return options

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
                observed_objects=[],
                last_segment_attempt={},
            )
        elif self.task == "place":
            state_kwargs.update(
                positions_place={},
            )
        elif self.task == "adjust":
            state_kwargs.update(
                positions_adjust={},
                adjustable_objects=[],
                adjust_focus_object="",
                adjust_start_position=(),
                adjust_visited_positions=[],
                adjust_action_options=[],
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
                observed_objects=self._segment_observed_objects(state),
                last_segment_attempt=state.last_segment_attempt,
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
                adjustable_objects=self._adjust_observed_objects(state),
                adjust_focus_object=state.adjust_focus_object,
                adjust_start_position=state.adjust_start_position,
                adjust_visited_positions=state.adjust_visited_positions,
                adjust_action_options=self._adjust_action_options(state),
            )

        return state_kwargs

    def _return_observation(self, state: SorterState):
        observation_kwargs = self._state_kwargs(state)
        observation_kwargs.pop("episode_id", None)
        observation_kwargs.pop("step_count", None)
        return SorterObservation(**observation_kwargs)

    def __init__(self, task: str | None = None):
        fresh_grid, objs_present = init_grid()
        grid_dims = tuple(int(dim) for dim in fresh_grid.shape)
        resolved_task = task or os.getenv(
            "THE_SORTER_PROJECT_TASK", self.DEFAULT_TASK
        )
        if resolved_task not in self.VALID_TASKS:
            raise ValueError(
                f"Unsupported task '{resolved_task}'. Expected one of {sorted(self.VALID_TASKS)}."
            )
        self.task = resolved_task
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
