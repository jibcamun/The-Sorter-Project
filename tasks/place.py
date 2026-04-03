from typing import Dict, Tuple

from numpy import zeros, mean, all

try:
    from ..models import SorterState
    from ..config.objects import OBJECTS
    from ..utils.rewards import compute_reward
except:
    from models import SorterState
    from config.objects import OBJECTS
    from utils.rewards import compute_reward


def _placement_score(weighted_grid, obj_name: str, pos: tuple) -> float:
    dims = tuple(OBJECTS[obj_name]["dims"])
    return mean(
        weighted_grid[
            pos[0] : pos[0] + dims[0],
            pos[1] : pos[1] + dims[1],
            pos[2] : pos[2] + dims[2],
        ]
    )


def _is_place_done(state: SorterState) -> bool:
    return set(state.positions_place.keys()) == set(state.objects_present.keys())


def place(state: SorterState, placements: Dict[str, Tuple[int, int, int, bool]]):
    grid_dims = state.grid_dims
    state.done = False

    grid = zeros(grid_dims)
    wt_grid = state.weighted_grid
    staged_positions = dict(state.positions_place)

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
    state.done = _is_place_done(state)
