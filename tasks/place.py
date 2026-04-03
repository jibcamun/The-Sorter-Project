from typing import Dict, Tuple

from numpy import zeros, mean, all

try:
    from ..models import SorterState
    from ..config.objects import OBJECTS
    from ..utils.rewards import compute_reward
    from ..utils.grids import weighted_grid
except:
    from models import SorterState
    from config.objects import OBJECTS
    from utils.rewards import compute_reward
    from utils.grids import weighted_grid


def place(state: SorterState, placements: Dict[str, Tuple[int, int, int, bool]]):
    grid_dims = state.grid_dims

    grid = zeros(grid_dims)
    state.weighted_grid = weighted_grid(grid_dims)
    wt_grid = state.weighted_grid

    reward_per_obj = (
        50.0 / len(placements.keys()) if len(placements.keys()) > 0 else 0.0
    )

    if set(placements.keys()) != set(state.objects_present.keys()):
        compute_reward(
            state,
            -reward_per_obj,
            "place: All the objects present in the grid was not passed as input. It must be passed irrespective of whether the position is changed or not.",
        )
        return

    objs_alrdy_present = state.positions_place

    for obj, pos in placements.items():
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
                        mean(
                            wt_grid[
                                pos[0] : pos[0] + dimns[0],
                                pos[1] : pos[1] + dimns[1],
                                pos[2] : pos[2] + dimns[2],
                            ]
                        )
                        * reward_per_obj,
                        f"place: Object '{obj}' placed successfully",
                    )
                    state.positions_place[obj] = pos
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
                        mean(
                            wt_grid[
                                pos[0] : pos[0] + dimns[0],
                                pos[1] : pos[1] + dimns[1],
                                pos[2] : pos[2] + dimns[2],
                            ]
                        )
                        * reward_per_obj,
                        f"place: Object '{obj}' placed successfully",
                    )
                    state.positions_place[obj] = pos
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
