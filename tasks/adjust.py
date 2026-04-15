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

    if any(above_slice != 0):
        return (
            False,
            "there is an object above, thus this object can not be adjusted as the stack might collapse",
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


def _legal_adjustment_error(
    state: SorterState, obj_name: str, new_pos: tuple
) -> str | None:
    init_pos = state.objects_present[obj_name]
    dims = tuple(OBJECTS[obj_name]["dims"])

    is_adjustable, adjust_msg = _is_adjustable(
        state.current_grid, obj_name, init_pos[:3]
    )
    if not is_adjustable:
        return adjust_msg
    if not _in_bounds(state.current_grid, dims, new_pos):
        return "the target position is out of bounds for this object's dimensions"

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
        return "the target space is already occupied"

    if not _has_support(grid_without_object, dims, new_pos):
        return "the target position does not have full support below it"

    return None


def _is_legal_adjustment(state: SorterState, obj_name: str, new_pos: tuple) -> bool:
    return _legal_adjustment_error(state, obj_name, new_pos) is None


def _position_score(state: SorterState, obj_name: str, pos: tuple) -> float:
    dims = tuple(OBJECTS[obj_name]["dims"])
    return mean(
        state.weighted_grid[
            pos[0] : pos[0] + dims[0],
            pos[1] : pos[1] + dims[1],
            pos[2] : pos[2] + dims[2],
        ]
    )


def _is_improvement_significant(
    current_score: float, new_score: float, eps: float = 0.01
) -> bool:
    return (new_score - current_score) > eps


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


def _legal_adjustment_positions(state: SorterState, obj_name: str) -> list[tuple]:
    dims = tuple(OBJECTS[obj_name]["dims"])
    legal_positions = []
    visited_positions = {
        tuple(pos)
        for pos in getattr(state, "adjust_visited_positions", [])
        if isinstance(pos, (list, tuple)) and len(pos) == 3
    }

    for x in range(state.grid_dims[0] - dims[0] + 1):
        for y in range(state.grid_dims[1] - dims[1] + 1):
            for z in range(state.grid_dims[2] - dims[2] + 1):
                new_pos = (x, y, z)
                if new_pos == state.objects_present[obj_name][:3]:
                    continue
                if visited_positions and new_pos in visited_positions:
                    continue
                if _is_legal_adjustment(state, obj_name, new_pos):
                    legal_positions.append(new_pos)

    return legal_positions


def _improving_adjustment_positions(state: SorterState, obj_name: str) -> list[tuple]:
    current_pos = state.objects_present[obj_name][:3]
    current_score = _position_score(state, obj_name, current_pos)
    improving_positions = []

    for pos in _legal_adjustment_positions(state, obj_name):
        if _is_improvement_significant(
            current_score, _position_score(state, obj_name, pos)
        ):
            improving_positions.append(pos)

    return improving_positions


def top_k_legal_adjustment_positions(
    state: SorterState, obj_name: str, k: int = 5
) -> list[tuple]:
    legal_positions = _legal_adjustment_positions(state, obj_name)
    return legal_positions[:k]


def _adjustment_reward(
    current_score: float, new_score: float, reward_per_obj: float
) -> float:
    reward = (new_score - current_score) * reward_per_obj
    return max(-reward_per_obj, min(reward, reward_per_obj))


def _adjustment_feedback(obj_name: str, current_score: float, new_score: float) -> str:
    if new_score < current_score:
        return f"adjust: Object '{obj_name}' moved legally to a worse target position."
    if new_score == current_score:
        return (
            f"adjust: Object '{obj_name}' moved legally but did not improve its score. "
            "Choose a different legal target position."
        )
    return f"adjust: Object '{obj_name}' adjusted successfully to a better target position."


def adjust(state: SorterState, adjustment: Tuple[str, int] | Tuple[()]):
    state.done = False
    reward_per_obj = 30.0 / len(state.objects_present)
    if len(adjustment) != 2:
        compute_reward(
            state,
            -reward_per_obj,
            "adjust: Exactly one object and one target option index must be selected for adjustment.",
        )
        return

    obj, option_index = adjustment
    if obj not in state.objects_present:
        compute_reward(
            state,
            -reward_per_obj,
            f"adjust: Object '{obj}' is not present in the grid.",
        )
        return
    if not isinstance(option_index, int):
        compute_reward(
            state,
            -reward_per_obj,
            "adjust: Target option index must be an integer.",
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

    exposed_targets = top_k_legal_adjustment_positions(state, obj)
    if not exposed_targets:
        compute_reward(
            state,
            0.0,
            f"adjust: Object '{obj}' has no unvisited legal target positions available.",
        )
        state.done = True
        return

    if not getattr(state, "adjust_focus_object", ""):
        state.adjust_focus_object = obj
        state.adjust_start_position = init_pos
        state.adjust_visited_positions = [tuple(init_pos[:3])]
    elif state.adjust_focus_object != obj:
        compute_reward(
            state,
            -reward_per_obj,
            (
                f"adjust: This episode is locked to object '{state.adjust_focus_object}'. "
                f"Do not switch to '{obj}'. Keep adjusting the same object within the episode."
            ),
        )
        return

    if option_index < 0 or option_index >= len(exposed_targets):
        compute_reward(
            state,
            -reward_per_obj,
            (
                f"adjust: Target option index {option_index} is invalid for object '{obj}'. "
                f"Choose an index between 0 and {len(exposed_targets) - 1}."
            ),
        )
        return

    requested_pos = exposed_targets[option_index]
    current_score = _position_score(state, obj, init_pos[:3])

    legality_error = _legal_adjustment_error(state, obj, requested_pos)
    if legality_error is not None:
        compute_reward(
            state,
            -reward_per_obj,
            (
                f"adjust: Object '{obj}' could not be adjusted to target position "
                f"{requested_pos} because {legality_error}. "
                "Choose a different target option index for this object."
            ),
        )
        return

    dimns = OBJECTS[obj]["dims"]
    updated_pos = (*requested_pos, init_pos[3])
    _remove_object(state, obj, init_pos)
    state.current_grid[
        requested_pos[0] : requested_pos[0] + dimns[0],
        requested_pos[1] : requested_pos[1] + dimns[1],
        requested_pos[2] : requested_pos[2] + dimns[2],
    ] = 1
    state.positions_adjust[obj] = updated_pos
    state.objects_present[obj] = updated_pos
    visited_positions = list(getattr(state, "adjust_visited_positions", []))
    visited_positions.append(tuple(requested_pos))
    state.adjust_visited_positions = visited_positions
    new_score = _position_score(state, obj, requested_pos)
    compute_reward(
        state,
        _adjustment_reward(current_score, new_score, reward_per_obj),
        _adjustment_feedback(obj, current_score, new_score),
    )
    state.done = not bool(_improving_adjustment_positions(state, obj))
