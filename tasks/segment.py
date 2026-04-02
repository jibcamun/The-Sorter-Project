try:
    from ..models import SorterState, SorterAction
    from ..utils.rewards import compute_reward
except:
    from models import SorterState, SorterAction
    from utils.rewards import compute_reward


def segment(state: SorterState, action: SorterAction):

    obj_loc_mapper = state.objects_present
    reward_per_object_placed = 20.0 / len(obj_loc_mapper)
    objects_found = action.segment

    remaining_objects = dict(obj_loc_mapper)

    for object_found in objects_found:
        matched_name = None
        for object_present, object_position in remaining_objects.items():
            if object_found == object_position:
                matched_name = object_present
                break

        if matched_name is not None:
            compute_reward(
                state,
                reward_per_object_placed,
                f"the right object, '{matched_name}' was found at correct position",
            )
            state.positions_segment[matched_name] = remaining_objects.pop(matched_name)
        else:
            compute_reward(
                state,
                -reward_per_object_placed,
                f"The position, '{object_found}' does not correspond to any object in the grid",
            )
