from typing import Dict, Tuple
from numpy import mean, any

try:
    from ..models import SorterObservation, SorterState
    from ..config.objects import OBJECTS
    from ..utils.rewards import compute_reward
except:
    from models import SorterObservation, SorterState
    from config.objects import OBJECTS
    from utils.rewards import compute_reward


def _remove_object(state: SorterState, obj_name: str, pos: tuple):
    dims = OBJECTS[obj_name]["dims"]
    state.current_grid[
        pos[0] : pos[0] + dims[0], pos[1] : pos[1] + dims[1], pos[2] : pos[2] + dims[2]
    ] = 0


def _is_adjustable(grid: dict, obj_name: str, pos: tuple):
    dims = OBJECTS[obj_name]["dims"]

    if (
        pos[0] < 0
        or pos[1] < 0
        or pos[2] < 0
        or pos[0] + dims[0] > grid.shape[0]
        or pos[1] + dims[1] > grid.shape[1]
        or pos[2] + dims[2] > grid.shape[2]
    ):
        return False, f"Object '{obj_name}' is out of bounds with respect to grid"

    above_slice = grid[
        pos[0] : pos[0] + dims[0],
        pos[1] : pos[1] + dims[1],
        pos[2] + dims[2] : pos[2] + dims[2] + 1,
    ]
    below_slice = (
        grid[pos[0] : pos[0] + dims[0], pos[1] : pos[1] + dims[1], pos[2] - 1]
        if pos[2] > 0
        else None
    )

    if pos[2] == 0 and any(above_slice != 0):
        return (
            False,
            "there is an object above, thus this object can not be adjusted as the stack might collapse",
        )
    elif any(above_slice != 0) or (below_slice is not None and any(below_slice != 0)):
        return (
            False,
            "there is an object below, thus this object can not be adjusted as the stack might collapse",
        )

    return True, ""


def build_adjust_candidates(observation: SorterObservation) -> Dict[str, list[int]]:
    adjustable_objects: Dict[str, list[int]] = {}

    for obj_name, pos in observation.objects_present.items():
        is_adjustable, _ = _is_adjustable(observation.current_grid, obj_name, pos[:3])
        if is_adjustable:
            adjustable_objects[obj_name] = list(pos[:3])

    return adjustable_objects


def _in_bounds(grid, dims: tuple, pos: tuple) -> bool:
    return not (
        pos[0] < 0
        or pos[1] < 0
        or pos[2] < 0
        or pos[0] + dims[0] > grid.shape[0]
        or pos[1] + dims[1] > grid.shape[1]
        or pos[2] + dims[2] > grid.shape[2]
    )


def _has_support(grid, dims: tuple, pos: tuple) -> bool:
    if pos[2] == 0:
        return True
    below_slice = grid[
        pos[0] : pos[0] + dims[0],
        pos[1] : pos[1] + dims[1],
        pos[2] - 1,
    ]
    return not any(below_slice == 0)


def _is_legal_adjustment(state: SorterState, obj_name: str, new_pos: tuple) -> bool:
    init_pos = state.objects_present[obj_name]
    dims = tuple(OBJECTS[obj_name]["dims"])

    is_adjustable, _ = _is_adjustable(state.current_grid, obj_name, init_pos[:3])
    if not is_adjustable or not _in_bounds(state.current_grid, dims, new_pos):
        return False

    grid_without_object = state.current_grid.copy()
    grid_without_object[
        init_pos[0] : init_pos[0] + dims[0],
        init_pos[1] : init_pos[1] + dims[1],
        init_pos[2] : init_pos[2] + dims[2],
    ] = 0

    target_slice = grid_without_object[
        new_pos[0] : new_pos[0] + dims[0],
        new_pos[1] : new_pos[1] + dims[1],
        new_pos[2] : new_pos[2] + dims[2],
    ]

    if any(target_slice != 0):
        return False

    return _has_support(grid_without_object, dims, new_pos)


def _position_score(state: SorterState, obj_name: str, pos: tuple) -> float:
    reward_per_obj = 30.0 / len(state.objects_present)
    dims = tuple(OBJECTS[obj_name]["dims"])
    return (
        mean(
            state.weighted_grid[
                pos[0] : pos[0] + dims[0],
                pos[1] : pos[1] + dims[1],
                pos[2] : pos[2] + dims[2],
            ]
        )
        * reward_per_obj
    )


def _best_adjustment_position(state: SorterState, obj_name: str) -> tuple | None:
    init_pos = state.objects_present[obj_name]
    dims = tuple(OBJECTS[obj_name]["dims"])
    best_pos = None
    best_score = _position_score(state, obj_name, init_pos)

    for x in range(state.grid_dims[0] - dims[0] + 1):
        for y in range(state.grid_dims[1] - dims[1] + 1):
            for z in range(state.grid_dims[2] - dims[2] + 1):
                new_pos = (x, y, z)
                if new_pos == init_pos[:3]:
                    continue
                if not _is_legal_adjustment(state, obj_name, new_pos):
                    continue
                candidate_score = _position_score(state, obj_name, new_pos)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_pos = new_pos

    return best_pos


def _adjustment_reward(
    current_score: float, new_score: float, best_score: float, reward_per_obj: float
) -> float:
    if new_score < current_score:
        reward = -(current_score - new_score) * reward_per_obj
    elif new_score == current_score:
        reward = 0.0
    elif best_score <= current_score:
        reward = 0.0
    else:
        reward = (
            (new_score - current_score) / (best_score - current_score)
        ) * reward_per_obj

    return max(-reward_per_obj, min(reward, reward_per_obj))


def _is_adjust_done(
    state: SorterState, object_to_check: Tuple[str, str, int] | Tuple[()]
) -> bool:
    if not object_to_check:
        return False
    obj_name = object_to_check[0]
    return _best_adjustment_position(state, obj_name) is None


def adjust(state: SorterState, adjustment: Tuple[str, str, int] | Tuple[()]):
    state.done = False
    reward_per_obj = 30.0 / len(state.objects_present)
    if len(adjustment) != 3:
        compute_reward(
            state,
            -reward_per_obj,
            "adjust: Exactly one object must be selected for adjustment.",
        )
        return

    obj, _direction, _amount = adjustment
    if obj not in state.objects_present:
        compute_reward(
            state,
            -reward_per_obj,
            f"adjust: Object '{obj}' is not present in the grid.",
        )
        return

    init_pos = state.objects_present[obj]
    is_adjustable, msg = _is_adjustable(state.current_grid, obj, init_pos[:3])
    if not is_adjustable:
        compute_reward(
            state,
            -reward_per_obj,
            f"adjust: Object '{obj}' can not be adjusted due to {msg}",
        )
        return

    best_pos = _best_adjustment_position(state, obj)
    if best_pos is None:
        compute_reward(
            state,
            0.0,
            f"adjust: Object '{obj}' is already at its best legal position.",
        )
        state.done = True
        return

    current_score = _position_score(state, obj, init_pos[:3])
    best_score = _position_score(state, obj, best_pos)
    dimns = OBJECTS[obj]["dims"]
    updated_pos = (*best_pos, init_pos[3])
    _remove_object(state, obj, init_pos)
    state.current_grid[
        best_pos[0] : best_pos[0] + dimns[0],
        best_pos[1] : best_pos[1] + dimns[1],
        best_pos[2] : best_pos[2] + dimns[2],
    ] = 1
    state.positions_adjust[obj] = updated_pos
    state.objects_present[obj] = updated_pos
    new_score = _position_score(state, obj, best_pos)
    compute_reward(
        state,
        _adjustment_reward(current_score, new_score, best_score, reward_per_obj),
        (
            f"adjust: Object '{obj}' adjusted successfully to a legal position."
            if new_score < best_score
            else f"adjust: Object '{obj}' adjusted successfully to its best legal position."
        ),
    )
    state.done = _is_adjust_done(state, adjustment)
