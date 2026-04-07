from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Literal, Mapping

try:
    from .models import SorterAction, SorterObservation, SorterState
    from .tasks.adjust import adjust as run_adjust
    from .tasks.adjust import _legal_adjustment_positions, _position_score
    from .tasks.place import place as run_place
    from .tasks.segment import segment as run_segment
except ImportError:
    from models import SorterAction, SorterObservation, SorterState
    from tasks.adjust import adjust as run_adjust
    from tasks.adjust import _legal_adjustment_positions, _position_score
    from tasks.place import place as run_place
    from tasks.segment import segment as run_segment


TaskName = Literal["segment", "place", "adjust"]
TASK_FIELDS = ("segment", "place", "adjust")

TASK_MAX_SCORES: Dict[TaskName, float] = {
    "segment": 20.0,
    "place": 50.0,
    "adjust": 30.0,
}
SUCCESS_SCORE_THRESHOLD = 0.1


@dataclass(slots=True)
class GradeResult:
    task: TaskName
    done: bool
    raw_score: float
    max_score: float
    step_max_score: float
    normalized_score: float
    latest_reward: float
    rewards: list[float]
    feedback: list[str]
    final_state: SorterState

    @property
    def passed(self) -> bool:
        return self.normalized_score >= SUCCESS_SCORE_THRESHOLD

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["final_state"] = self.final_state.model_dump()
        payload["passed"] = self.passed
        return payload


@dataclass(slots=True)
class ParsedAction:
    segment: Dict[str, Any]
    place: Dict[str, Any]
    adjust: tuple[Any, ...]


def _normalize_position_payload_map(payload: Mapping[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for object_name, position in payload.items():
        if isinstance(position, (list, tuple)):
            normalized[object_name] = tuple(position)
        else:
            normalized[object_name] = position
    return normalized


def _normalize_adjust_payload(payload: Any) -> tuple[Any, ...]:
    return tuple(payload) if isinstance(payload, (list, tuple)) else ()


def _coerce_state(state: SorterState | SorterObservation | Mapping[str, Any]):
    if isinstance(state, SorterState):
        return state.model_copy(deep=True)
    if isinstance(state, SorterObservation):
        payload = state.model_dump()
        payload["reward"] = payload.pop("reward_details", ([], []))
        return SorterState(**payload)
    if isinstance(state, Mapping):
        return SorterState(**dict(state))
    raise TypeError("state must be a SorterState, SorterObservation, or mapping.")


def _coerce_action(action: SorterAction | Mapping[str, Any]):
    if isinstance(action, SorterAction):
        return ParsedAction(
            segment=_normalize_position_payload_map(action.segment),
            place=_normalize_position_payload_map(action.place),
            adjust=_normalize_adjust_payload(action.adjust),
        )
    if isinstance(action, Mapping):
        action_dict = dict(action)
        segment = action_dict.get("segment", {})
        place = action_dict.get("place", {})
        adjust = action_dict.get("adjust", ())

        if not isinstance(segment, Mapping):
            raise TypeError("segment action must be a mapping.")
        if not isinstance(adjust, (list, tuple)):
            raise TypeError("adjust action must be a list or tuple.")

        return ParsedAction(
            segment=_normalize_position_payload_map(segment),
            place=_normalize_position_payload_map(place),
            adjust=_normalize_adjust_payload(adjust),
        )
    raise TypeError("action must be a SorterAction or mapping.")


def _fallback_action(task: TaskName, action: Any) -> ParsedAction:
    segment: Dict[str, Any] = {}
    place: Dict[str, Any] = {}
    adjust: tuple[Any, ...] = ()

    if isinstance(action, Mapping):
        action_payload = action.get(task, action)
        if isinstance(action_payload, Mapping):
            if task == "segment":
                segment = _normalize_position_payload_map(action_payload)
            elif task == "place":
                place = _normalize_position_payload_map(action_payload)
        elif task == "adjust" and isinstance(action_payload, (list, tuple)):
            adjust = _normalize_adjust_payload(action_payload)

    return ParsedAction(segment=segment, place=place, adjust=adjust)


def _step_max_reward(task: TaskName, state: SorterState, action: ParsedAction) -> float:
    if task == "segment":
        return TASK_MAX_SCORES["segment"]

    if task == "place":
        return TASK_MAX_SCORES["place"]

    if task == "adjust":
        total_objects = len(state.objects_present)
        adjusted_objects = 1 if len(action.adjust) == 2 else 0
        if total_objects <= 0 or adjusted_objects <= 0:
            return 0.0
        return (TASK_MAX_SCORES["adjust"] / total_objects) * (adjusted_objects)

    raise ValueError(f"Unsupported task: {task}")


def _episode_max_reward(task: TaskName, state: SorterState) -> float:
    if task in ("segment", "place"):
        return TASK_MAX_SCORES[task]

    if task == "adjust":
        total_objects = len(state.objects_present)
        if total_objects <= 0:
            return 0.0
        if getattr(state, "adjust_focus_object", ""):
            return TASK_MAX_SCORES["adjust"] / total_objects
        return TASK_MAX_SCORES["adjust"]

    raise ValueError(f"Unsupported task: {task}")


def _result_from_state(task: TaskName, state: SorterState, action: ParsedAction):
    rewards = list(state.reward[0])
    feedback = list(state.reward[1])
    raw_score = float(sum(rewards))
    max_score = _episode_max_reward(task, state)
    step_max_score = _step_max_reward(task, state, action)
    normalized_score = raw_score / max_score if max_score > 0 else 0.0
    normalized_score = max(0.0, min(1.0, normalized_score))
    latest_reward = rewards[-1] if rewards else 0.0

    return GradeResult(
        task=task,
        done=bool(state.done),
        raw_score=raw_score,
        max_score=max_score,
        step_max_score=step_max_score,
        normalized_score=normalized_score,
        latest_reward=latest_reward,
        rewards=rewards,
        feedback=feedback,
        final_state=state,
    )


def _adjust_progress_fraction(state: SorterState) -> float:
    focus_object = getattr(state, "adjust_focus_object", "")
    if not focus_object or focus_object not in state.objects_present:
        return 0.0

    current_pos = state.objects_present[focus_object][:3]
    start_pos = (
        getattr(state, "adjust_start_position", ())
        or state.objects_present[focus_object]
    )
    initial_score = _position_score(state, focus_object, start_pos[:3])
    current_score = _position_score(state, focus_object, current_pos)
    legal_positions = _legal_adjustment_positions(state, focus_object)
    best_reachable_score = max(
        [
            current_score,
            *(_position_score(state, focus_object, pos) for pos in legal_positions),
        ]
    )

    achievable_improvement = best_reachable_score - initial_score
    achieved_improvement = current_score - initial_score
    if achievable_improvement <= 0:
        return 1.0
    return max(0.0, min(achieved_improvement / achievable_improvement, 1.0))


def _segment_completion_fraction(state: SorterState) -> float:
    total_objects = len(state.objects_present)
    if total_objects <= 0:
        return 0.0

    confirmed_positions = getattr(state, "positions_segment", {}) or {}
    correct_count = 0
    for object_name, expected_position in state.objects_present.items():
        if confirmed_positions.get(object_name) == expected_position:
            correct_count += 1

    return max(0.0, min(correct_count / total_objects, 1.0))


def grade_adjust_progress(state: SorterState | SorterObservation | Mapping[str, Any]):
    graded_state = _coerce_state(state)
    _validate_state_for_task("adjust", graded_state)
    parsed_action = ParsedAction(segment={}, place={}, adjust=())
    result = _result_from_state("adjust", graded_state, parsed_action)
    progress_fraction = _adjust_progress_fraction(graded_state)
    result.raw_score = result.max_score * progress_fraction
    result.normalized_score = progress_fraction
    result.feedback = [
        *result.feedback,
        (
            "adjust progress grading: rollout ended before task completion; "
            f"awarded {progress_fraction:.3f} of achievable improvement for "
            f"'{getattr(graded_state, 'adjust_focus_object', '')}'."
        ),
    ]
    return result


def _record_failure(
    task: TaskName, state: SorterState, action: ParsedAction, message: str
) -> GradeResult:
    penalty = -_step_max_reward(task, state, action)
    state.done = False
    state.reward[0].append(penalty)
    state.reward[1].append(message)
    return _result_from_state(task, state, action)


def _validate_state_for_task(task: TaskName, state: SorterState):
    if task == "segment" and not state.objects_present:
        raise ValueError(
            "segment grading requires a state with objects_present populated."
        )
    if task == "place" and not state.objects_present:
        raise ValueError(
            "place grading requires a state with objects_present populated."
        )
    if task == "adjust" and not state.objects_present:
        raise ValueError(
            "adjust grading requires a state with objects_present populated."
        )


def grade_segment(
    state: SorterState | SorterObservation | Mapping[str, Any],
    action: SorterAction | Mapping[str, Any],
):
    graded_state = _coerce_state(state)
    _validate_state_for_task("segment", graded_state)
    try:
        parsed_action = _coerce_action(action)
        run_segment(graded_state, parsed_action.segment)
    except (KeyError, TypeError, ValueError) as exc:
        fallback_action = _fallback_action("segment", action)
        return _record_failure(
            "segment", graded_state, fallback_action, f"segment grading failed: {exc}"
        )
    result = _result_from_state("segment", graded_state, parsed_action)
    completion_fraction = _segment_completion_fraction(graded_state)
    result.raw_score = result.max_score * completion_fraction
    result.normalized_score = completion_fraction
    return result


def grade_place(
    state: SorterState | SorterObservation | Mapping[str, Any],
    action: SorterAction | Mapping[str, Any],
):
    graded_state = _coerce_state(state)
    _validate_state_for_task("place", graded_state)
    try:
        parsed_action = _coerce_action(action)
        run_place(graded_state, parsed_action.place)
    except (KeyError, TypeError, ValueError) as exc:
        fallback_action = _fallback_action("place", action)
        return _record_failure(
            "place", graded_state, fallback_action, f"place grading failed: {exc}"
        )
    return _result_from_state("place", graded_state, parsed_action)


def grade_adjust(
    state: SorterState | SorterObservation | Mapping[str, Any],
    action: SorterAction | Mapping[str, Any] | list[Any] | tuple[Any, ...],
):
    graded_state = _coerce_state(state)
    _validate_state_for_task("adjust", graded_state)
    try:
        action_input = (
            {"adjust": action} if isinstance(action, (list, tuple)) else action
        )
        parsed_action = _coerce_action(action_input)
        run_adjust(graded_state, parsed_action.adjust)
    except (KeyError, TypeError, ValueError) as exc:
        fallback_action = _fallback_action("adjust", action)
        return _record_failure(
            "adjust", graded_state, fallback_action, f"adjust grading failed: {exc}"
        )
    return _result_from_state("adjust", graded_state, parsed_action)


def grade_task(
    task: TaskName,
    state: SorterState | SorterObservation | Mapping[str, Any],
    action: SorterAction | Mapping[str, Any] | list[Any] | tuple[Any, ...],
):
    if task == "segment":
        return grade_segment(state, action)
    if task == "place":
        return grade_place(state, action)
    if task == "adjust":
        return grade_adjust(state, action)
    raise ValueError(f"Unsupported task: {task}")


def grade_payload(
    task: TaskName,
    state: SorterState | SorterObservation | Mapping[str, Any],
    payload: Any,
):
    if not isinstance(payload, Mapping):
        if task != "adjust":
            raise ValueError(
                f"Payload for task '{task}' must be a mapping with top-level key '{task}'."
            )
        payload = {task: payload}

    payload_keys = set(payload.keys())
    task_field_keys = payload_keys.intersection(TASK_FIELDS)

    if task in payload:
        if task_field_keys - {task}:
            raise ValueError(
                f"Payload for task '{task}' contains unrelated task fields: {sorted(task_field_keys - {task})}"
            )
    elif task_field_keys:
        raise ValueError(
            f"Payload for task '{task}' is missing its top-level key and contains other task fields: {sorted(task_field_keys)}"
        )
    else:
        payload = {task: payload}

    return grade_task(task, state, payload)


__all__ = [
    "GradeResult",
    "SUCCESS_SCORE_THRESHOLD",
    "TASK_MAX_SCORES",
    "grade_adjust",
    "grade_adjust_progress",
    "grade_payload",
    "grade_place",
    "grade_segment",
    "grade_task",
]
