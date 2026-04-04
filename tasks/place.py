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
        return None

    optimal = {}
    for (obj_name, pos), var in candidate_vars.items():
        if solver.Value(var):
            optimal[obj_name] = pos

    return optimal


def _wrong_objects_feedback(agent_positions: Dict[str, tuple], optimal_positions: Dict[str, tuple]) -> str:
    wrong_objects = sorted(
        obj_name
        for obj_name, optimal_pos in optimal_positions.items()
        if agent_positions.get(obj_name) != optimal_pos
    )

    if not wrong_objects:
        return "place: All objects match the optimal placement."

    return f"place: Objects needing better placement: {', '.join(wrong_objects)}"


def _matches_optimal_score(state: SorterState, optimal_positions: Dict[str, tuple] | None) -> bool:
    if optimal_positions is None:
        return False
    if set(state.positions_place.keys()) != set(state.objects_present.keys()):
        return False

    achieved_score = _total_placement_score(state.weighted_grid, state.positions_place)
    optimal_score = _total_placement_score(state.weighted_grid, optimal_positions)
    tolerance = len(state.objects_present) / OBJECTIVE_SCALE
    return abs(achieved_score - optimal_score) <= tolerance


def _is_place_done(state: SorterState, optimal_positions: Dict[str, tuple] | None) -> bool:
    return _matches_optimal_score(state, optimal_positions)


def place(state: SorterState, placements: Dict[str, Tuple[int, int, int, bool]]):
    grid_dims = state.grid_dims
    state.done = False
    state.advisory = []

    grid = zeros(grid_dims)
    wt_grid = state.weighted_grid
    staged_positions = {}

    reward_per_obj = (
        50.0 / len(placements.keys()) if len(placements.keys()) > 0 else 0.0
    )

    if set(placements.keys()) != set(state.objects_present.keys()):
        compute_reward(
            state,
            -reward_per_obj,
            "place: All the objects present in the grid was not passed as input. It must be passed irrespective of whether the position is changed or not.",
        )
        state.done = False
        return

    objs_alrdy_present = dict(state.positions_place)

    for obj, pos in placements.items():
        if obj not in OBJECTS:
            compute_reward(
                state,
                -reward_per_obj,
                f"place: Object '{obj}' is not a valid known object",
            )
            state.done = False
            return

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
            compute_reward(
                state,
                -reward_per_obj,
                f"place: Object '{obj}' is out of bounds with respect to grid",
            )
            state.done = False
            return

        if all(
            grid[
                pos[0] : pos[0] + dimns[0],
                pos[1] : pos[1] + dimns[1],
                pos[2] : pos[2] + dimns[2],
            ]
            == 0
        ):
            if pos[2] == 0:
                grid[
                    pos[0] : pos[0] + dimns[0],
                    pos[1] : pos[1] + dimns[1],
                    pos[2] : pos[2] + dimns[2],
                ] += 1
                if (
                    objs_alrdy_present.get(obj) is None
                    or objs_alrdy_present[obj] != pos
                ):
                    compute_reward(
                        state,
                        _placement_score(wt_grid, obj, pos) * reward_per_obj,
                        f"place: Object '{obj}' placed successfully",
                    )
                else:
                    compute_reward(
                        state,
                        0.0,
                        f"place: Object '{obj}' kept at its current valid placement",
                    )
                staged_positions[obj] = pos
            elif stack and all(
                grid[
                    pos[0] : pos[0] + dimns[0],
                    pos[1] : pos[1] + dimns[1],
                    pos[2] - 1,
                ]
                == 1
            ):
                grid[
                    pos[0] : pos[0] + dimns[0],
                    pos[1] : pos[1] + dimns[1],
                    pos[2] : pos[2] + dimns[2],
                ] += 1
                if (
                    objs_alrdy_present.get(obj) is None
                    or objs_alrdy_present[obj] != pos
                ):
                    compute_reward(
                        state,
                        _placement_score(wt_grid, obj, pos) * reward_per_obj,
                        f"place: Object '{obj}' placed successfully",
                    )
                else:
                    compute_reward(
                        state,
                        0.0,
                        f"place: Object '{obj}' kept at its current valid placement",
                    )
                staged_positions[obj] = pos
            else:
                compute_reward(
                    state,
                    -reward_per_obj,
                    f"place: Object '{obj}' could not be placed as there was no support below it",
                )
        else:
            compute_reward(
                state,
                -reward_per_obj,
                f"place: Object '{obj}' could not be placed as the give space is already occupied",
            )

    state.current_grid = grid
    state.positions_place = staged_positions
    optimal_positions = _optimal_placements(state)
    state.done = _is_place_done(state, optimal_positions)
    if optimal_positions is not None:
        state.advisory.append(
            "place: Current placement achieves the optimal total score."
            if state.done
            else _wrong_objects_feedback(state.positions_place, optimal_positions)
        )
