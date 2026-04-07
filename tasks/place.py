from typing import Dict, Tuple

from numpy import zeros, mean, all
from ortools.sat.python import cp_model

try:
    from ..models import SorterState
    from ..config.objects import OBJECTS
    from ..utils.rewards import compute_reward
except:
    from models import SorterState
    from config.objects import OBJECTS
    from utils.rewards import compute_reward


OBJECTIVE_SCALE = 1000
PLACE_IMPROVEMENT_EPS = 1e-3


def _placement_score(weighted_grid, obj_name: str, pos: tuple) -> float:
    dims = tuple(OBJECTS[obj_name]["dims"])
    return mean(
        weighted_grid[
            pos[0] : pos[0] + dims[0],
            pos[1] : pos[1] + dims[1],
            pos[2] : pos[2] + dims[2],
        ]
    )


def _placement_objective_score(weighted_grid, obj_name: str, pos: tuple) -> int:
    return int(round(_placement_score(weighted_grid, obj_name, pos) * OBJECTIVE_SCALE))


def _total_placement_score(weighted_grid, positions: Dict[str, tuple]) -> float:
    return sum(
        _placement_score(weighted_grid, obj_name, pos)
        for obj_name, pos in positions.items()
    )


def _placement_reward_delta(
    weighted_grid, previous_positions: Dict[str, tuple], new_positions: Dict[str, tuple]
) -> float:
    previous_score = _total_placement_score(weighted_grid, previous_positions)
    new_score = _total_placement_score(weighted_grid, new_positions)
    return new_score - previous_score


def _enumerate_candidates(grid_dims: tuple, obj_name: str):
    dims = tuple(OBJECTS[obj_name]["dims"])
    stack = OBJECTS[obj_name]["stack"]

    for x in range(grid_dims[0] - dims[0] + 1):
        for y in range(grid_dims[1] - dims[1] + 1):
            z_values = range(grid_dims[2] - dims[2] + 1) if stack else [0]
            for z in z_values:
                yield (x, y, z, stack)


def _cells_for_position(obj_name: str, pos: tuple):
    dims = tuple(OBJECTS[obj_name]["dims"])
    for x in range(pos[0], pos[0] + dims[0]):
        for y in range(pos[1], pos[1] + dims[1]):
            for z in range(pos[2], pos[2] + dims[2]):
                yield (x, y, z)


def _footprint_below(obj_name: str, pos: tuple):
    dims = tuple(OBJECTS[obj_name]["dims"])
    if pos[2] == 0:
        return []
    return [
        (x, y, pos[2] - 1)
        for x in range(pos[0], pos[0] + dims[0])
        for y in range(pos[1], pos[1] + dims[1])
    ]


def _optimal_placements(state: SorterState):
    object_names = sorted(state.objects_present.keys())
    model = cp_model.CpModel()
    candidate_vars = {}
    candidate_scores = {}
    cell_cover = {}

    for obj_name in object_names:
        choices = []
        for idx, pos in enumerate(_enumerate_candidates(state.grid_dims, obj_name)):
            var = model.NewBoolVar(f"{obj_name}_{idx}")
            candidate_vars[(obj_name, pos)] = var
            candidate_scores[(obj_name, pos)] = _placement_objective_score(
                state.weighted_grid, obj_name, pos
            )
            choices.append(var)

            for cell in _cells_for_position(obj_name, pos):
                cell_cover.setdefault(cell, []).append(var)

        model.AddExactlyOne(choices)

    for vars_for_cell in cell_cover.values():
        model.Add(sum(vars_for_cell) <= 1)

    for obj_name in object_names:
        for pos in _enumerate_candidates(state.grid_dims, obj_name):
            if pos[2] == 0:
                continue
            var = candidate_vars[(obj_name, pos)]
            for below_cell in _footprint_below(obj_name, pos):
                supporting_vars = cell_cover.get(below_cell, [])
                if supporting_vars:
                    model.Add(sum(supporting_vars) >= 1).OnlyEnforceIf(var)
                else:
                    model.Add(var == 0)

    model.Maximize(
        sum(
            var * candidate_scores[(obj_name, pos)]
            for (obj_name, pos), var in candidate_vars.items()
        )
    )

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 5.0
    solver.parameters.num_search_workers = 8
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return None, status

    optimal = {}
    for (obj_name, pos), var in candidate_vars.items():
        if solver.Value(var):
            optimal[obj_name] = pos

    return optimal, status


def _wrong_objects_feedback(
    agent_positions: Dict[str, tuple], optimal_positions: Dict[str, tuple]
) -> str:
    wrong_objects = sorted(
        obj_name
        for obj_name, optimal_pos in optimal_positions.items()
        if agent_positions.get(obj_name) != optimal_pos
    )

    if not wrong_objects:
        return "place: All objects match the optimal placement."

    return f"place: Objects needing better placement: {', '.join(wrong_objects)}"


def _has_full_support(grid, obj_name: str, pos: tuple) -> bool:
    dims = tuple(OBJECTS[obj_name]["dims"])
    if pos[2] == 0:
        return True
    return all(
        grid[
            pos[0] : pos[0] + dims[0],
            pos[1] : pos[1] + dims[1],
            pos[2] - 1,
        ]
        == 1
    )


def _is_complete_valid_layout(
    state: SorterState, placements: Dict[str, Tuple[int, int, int, bool]]
) -> tuple[bool, str | None, object]:
    grid = zeros(state.grid_dims)

    for obj, pos in placements.items():
        if obj not in OBJECTS:
            return False, f"place: Object '{obj}' is not a valid known object", grid

        dimns = OBJECTS[obj]["dims"]
        stack = OBJECTS[obj]["stack"]

        if (
            pos[0] < 0
            or pos[1] < 0
            or pos[2] < 0
            or pos[0] + dimns[0] > grid.shape[0]
            or pos[1] + dimns[1] > grid.shape[1]
            or pos[2] + dimns[2] > grid.shape[2]
        ):
            return (
                False,
                f"place: Object '{obj}' is out of bounds with respect to grid",
                grid,
            )

        if not all(
            grid[
                pos[0] : pos[0] + dimns[0],
                pos[1] : pos[1] + dimns[1],
                pos[2] : pos[2] + dimns[2],
            ]
            == 0
        ):
            return (
                False,
                f"place: Object '{obj}' could not be placed as the given space is already occupied",
                grid,
            )

        if pos[2] > 0 and (not stack or not _has_full_support(grid, obj, pos)):
            return (
                False,
                f"place: Object '{obj}' could not be placed as there was no support below it",
                grid,
            )

        grid[
            pos[0] : pos[0] + dimns[0],
            pos[1] : pos[1] + dimns[1],
            pos[2] : pos[2] + dimns[2],
        ] = 1

    return True, None, grid


def _placement_feedback(
    wt_grid,
    previous_positions: Dict[str, tuple],
    new_positions: Dict[str, tuple],
) -> str:
    changed_objects = sorted(
        obj_name
        for obj_name, pos in new_positions.items()
        if previous_positions.get(obj_name) != pos
    )
    if not changed_objects:
        return "place: Layout kept unchanged."

    delta = _placement_reward_delta(wt_grid, previous_positions, new_positions)
    if delta > PLACE_IMPROVEMENT_EPS:
        return f"place: Valid layout improved for objects: {', '.join(changed_objects)}"
    if delta < -PLACE_IMPROVEMENT_EPS:
        return f"place: Valid layout worsened for objects: {', '.join(changed_objects)}"
    return f"place: Valid layout updated with negligible score change for objects: {', '.join(changed_objects)}"


def place(state: SorterState, placements: Dict[str, Tuple[int, int, int, bool]]):
    state.done = False
    state.advisory = []
    wt_grid = state.weighted_grid
    target_objects = set(state.objects_present.keys())
    reward_per_obj = 50.0 / len(target_objects) if target_objects else 0.0
    previous_positions = dict(state.positions_place)
    proposed_positions = dict(previous_positions)
    proposed_positions.update(placements)

    if set(proposed_positions.keys()) != target_objects:
        missing_objects = sorted(target_objects - set(proposed_positions.keys()))
        compute_reward(
            state,
            -reward_per_obj,
            (
                "place: Layout is incomplete. Missing placements for objects: "
                f"{', '.join(missing_objects)}"
            ),
        )
        return

    is_valid, error_message, grid = _is_complete_valid_layout(state, proposed_positions)
    if not is_valid:
        compute_reward(state, -reward_per_obj, error_message)
        return

    delta_score = _placement_reward_delta(
        wt_grid, previous_positions, proposed_positions
    )
    reward = max(-50.0, min(delta_score * reward_per_obj, 50.0))
    compute_reward(
        state,
        reward,
        _placement_feedback(wt_grid, previous_positions, proposed_positions),
    )

    state.current_grid = grid
    state.positions_place = proposed_positions
    state.done = True

    optimal_positions, optimal_status = _optimal_placements(state)
    if optimal_positions is None:
        return

    if optimal_status == cp_model.OPTIMAL:
        achieved_score = _total_placement_score(
            state.weighted_grid, state.positions_place
        )
        optimal_score = _total_placement_score(state.weighted_grid, optimal_positions)
        tolerance = len(state.objects_present) / OBJECTIVE_SCALE
        state.advisory.append(
            "place: Current placement achieves the optimal total score."
            if abs(achieved_score - optimal_score) <= tolerance
            else _wrong_objects_feedback(state.positions_place, optimal_positions)
        )
    else:
        state.advisory.append(
            "place: Valid layout accepted. Exact optimal comparison unavailable because the solver returned only a feasible solution."
        )
