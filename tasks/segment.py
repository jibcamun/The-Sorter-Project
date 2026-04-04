from typing import Dict, Tuple

try:
    from ..models import SorterState
    from ..utils.rewards import compute_reward
except:
    from models import SorterState
    from utils.rewards import compute_reward


def _normalize_position(position) -> tuple:
    return tuple(position)


def _is_segment_done(state: SorterState) -> bool:
    target = state.objects_present
    found = state.positions_segment

    return (
        len(found) == len(target)
        and set(found.keys()) == set(target.keys())
        and all(_normalize_position(found[obj]) == _normalize_position(target[obj]) for obj in target)
    )


def segment(state: SorterState, objects_found: Dict[str, Tuple[int, int, int, bool]]):
    obj_loc_mapper = state.objects_present
    reward_per_object_placed = 20.0 / len(obj_loc_mapper)
    previous_attempt = dict(getattr(state, "last_segment_attempt", {}))
    state.advisory = []
    state.last_segment_attempt = dict(objects_found)
    invalid_task_keys = {"segment", "place", "adjust"}.intersection(objects_found.keys())

    if objects_found and objects_found == previous_attempt:
        state.advisory.append(
            "segment: Submitted mapping is unchanged from the previous failed attempt."
        )

    if invalid_task_keys:
        compute_reward(
            state,
            -reward_per_object_placed,
            (
                "segment: Invalid nested task keys inside segment payload: "
                f"{sorted(invalid_task_keys)}. Expected a flat mapping from object name to position."
            ),
        )
        state.done = False
        return

    if len(objects_found) != len(obj_loc_mapper):
        compute_reward(
            state,
            -reward_per_object_placed,
            (
                "segment: Incorrect number of objects provided."
                # f"Expected {len(obj_loc_mapper)}, received {len(objects_found)}."
            ),
        )
        state.done = False
        return

    correct_labels = []
    incorrect_labels = []

    for object_name, object_found in objects_found.items():
        expected_position = obj_loc_mapper.get(object_name)
        if object_name not in obj_loc_mapper:
            compute_reward(
                state,
                -reward_per_object_placed,
                f"The object name, '{object_name}' is not present in the grid",
            )
            incorrect_labels.append(object_name)
            continue

        normalized_found_position = _normalize_position(object_found)
        normalized_expected_position = _normalize_position(expected_position)

        if normalized_found_position == normalized_expected_position:
            compute_reward(
                state,
                reward_per_object_placed,
                f"the right object, '{object_name}' was found at correct position",
            )
            state.positions_segment[object_name] = normalized_found_position
            correct_labels.append(object_name)
        else:
            compute_reward(
                state,
                -reward_per_object_placed,
                f"The object, '{object_name}' was assigned the wrong position: '{object_found}'",
            )
            incorrect_labels.append(object_name)

    remaining_labels = sorted(
        obj_name for obj_name in obj_loc_mapper.keys() if obj_name not in correct_labels
    )
    state.advisory.append(
        (
            f"segment: {len(correct_labels)}/{len(obj_loc_mapper)} labels correct. "
            f"Correct labels: {sorted(correct_labels)}. "
            f"Incorrect labels: {sorted(incorrect_labels)}. "
            f"Remaining labels to fix: {remaining_labels}."
        )
    )

    state.done = _is_segment_done(state)
