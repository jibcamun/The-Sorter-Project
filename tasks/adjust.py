from numpy import mean, any

try:
    from ..models import SorterState, SorterAction
    from ..config.objects import OBJECTS
    from ..utils.rewards import compute_reward
except:
    from models import SorterState, SorterAction
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


def _adjust_position(
    state: SorterState,
    obj: str,
    init_pos: tuple,
    new_pos: tuple,
    dimns: tuple,
    direction: str,
):
    wt_grid = state.weighted_grid
    reward_per_obj = 30.0 / len(state.objects_present)
    is_adjust, msg = _is_adjustable(state.current_grid, obj, new_pos)
    if is_adjust:
        _remove_object(state, obj, init_pos)
        state.current_grid[
            new_pos[0] : new_pos[0] + dimns[0],
            new_pos[1] : new_pos[1] + dimns[1],
            new_pos[2] : new_pos[2] + dimns[2],
        ] = 1
        state.positions_adjust[obj] = new_pos
        state.objects_present[obj] = new_pos
        compute_reward(
            state,
            mean(
                wt_grid[
                    new_pos[0] : new_pos[0] + dimns[0],
                    new_pos[1] : new_pos[1] + dimns[1],
                    new_pos[2] : new_pos[2] + dimns[2],
                ]
            )
            * reward_per_obj,
            f"adjust: Object '{obj}' adjusted {direction} succesfully , can be adjusted better to maximise reward",
        )
    else:
        compute_reward(
            state,
            -reward_per_obj,
            f"adjust : Object '{obj}' did not adjust in '{direction}' due to {msg}",
        )


def adjust(state: SorterState, action: SorterAction):
    reward_per_obj = 30.0 / len(state.objects_present)
    for obj, obj_adjust_info in action.adjust.items():
        init_pos = state.objects_present[obj]
        direction = obj_adjust_info[0]
        mag = obj_adjust_info[1]
        dimns = OBJECTS[obj]["dims"]

        if direction == "RIGHT":
            new_pos = (init_pos[0] + mag, init_pos[1], init_pos[2])
            _adjust_position(state, obj, init_pos, new_pos, dimns, direction)

        elif direction == "LEFT":
            new_pos = (init_pos[0] - mag, init_pos[1], init_pos[2])
            _adjust_position(state, obj, init_pos, new_pos, dimns, direction)

        elif direction == "FORWARD":
            new_pos = (init_pos[0], init_pos[1] + mag, init_pos[2])
            _adjust_position(state, obj, init_pos, new_pos, dimns, direction)

        elif direction == "BACKWARD":
            new_pos = (init_pos[0], init_pos[1] - mag, init_pos[2])
            _adjust_position(state, obj, init_pos, new_pos, dimns, direction)

        elif direction == "UP":
            new_pos = (init_pos[0], init_pos[1], init_pos[2] + mag)
            _adjust_position(state, obj, init_pos, new_pos, dimns, direction)

        elif direction == "DOWN":
            new_pos = (init_pos[0], init_pos[1], init_pos[2] - mag)
            _adjust_position(state, obj, init_pos, new_pos, dimns, direction)

        else:
            compute_reward(
                state,
                -reward_per_obj,
                'adjust: Invalid choice of adjust, valid choices are: "RIGHT", "LEFT", "FORWARD", "BACKWARD", "UP" and "DOWN"',
            )
            return
