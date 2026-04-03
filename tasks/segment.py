from typing import Dict, Tuple

try:
    from ..models import SorterState
    from ..utils.rewards import compute_reward
except:
    from models import SorterState
    from utils.rewards import compute_reward


def segment(state: SorterState, objects_found: Dict[str, Tuple[int, int, int, bool]]):

    obj_loc_mapper = state.objects_present
    reward_per_object_placed = 20.0 / len(obj_loc_mapper)

    remaining_objects = dict(obj_loc_mapper)

    for object_name, object_found in objects_found.items():
        expected_position = remaining_objects.get(object_name)

        if expected_position is None:
            compute_reward(
                state,
                -reward_per_object_placed,
                f"The object name, '{object_name}' is not present in the grid",
            )
            continue

        if object_found == expected_position:
            compute_reward(
                state,
                reward_per_object_placed,
                f"the right object, '{object_name}' was found at correct position",
            )
            state.positions_segment[object_name] = remaining_objects.pop(object_name)
        else:
            compute_reward(
                state,
                -reward_per_object_placed,
                f"The object, '{object_name}' was assigned the wrong position: '{object_found}'",
            )
