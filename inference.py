import os
import textwrap
from dotenv import load_dotenv
import json
from json import JSONDecodeError
from typing import Any, Dict, List, Optional
import numpy as np
from openai import OpenAI
from pydantic import ValidationError
from config.objects import OBJECTS
from server.sorter_environment import SorterEnvironment
from models import SorterObservation, SorterState

try:
    from graders import GradeResult, grade_task
except ImportError:
    from .graders import GradeResult, grade_task

try:
    from tasks.adjust import build_adjust_candidates
except ImportError:
    from .tasks.adjust import build_adjust_candidates


load_dotenv()

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://integrate.api.nvidia.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-120b"
TASK_NAME = os.getenv("THE_SORTER_PROJECT_TASK", "adjust") or "adjust"
BENCHMARK = os.getenv("THE_SORTER_PROJECT_BENCHMARK", "the_sorter_project")
MAX_STEPS = 8
TEMPERATURE = 0.7

TASK_ACTION_FIELDS = {
    "segment": "segment",
    "place": "place",
    "adjust": "adjust",
}

TASK_OBSERVATION_FIELDS = {
    "segment": {
        "positions_segment",
        "positions",
        "observed_objects",
        "last_segment_attempt",
        "reward",
        "advisory",
        "done",
    },
    "place": {"objects_present", "positions_place", "reward", "advisory", "done"},
    "adjust": {
        "grid_dims",
        "objects_present",
        "positions_adjust",
        "reward",
        "advisory",
        "done",
    },
}

SYSTEM_PROMPT = textwrap.dedent(
    f"""
    You are a professional logistician, facility expert and layout engineer.
    Your current task is: {TASK_NAME}.

    Task definitions:
    a) segment: identify object names for the observed objects using only observable information. Each observed object includes its position, dimensions, stackability, and volume. Objects with identical observable signatures may be interchangeable for scoring, so focus on assigning labels consistently within each compatible group. Return exactly one flat mapping from object name to position under the segment key. Do not nest place or adjust inside segment.
    b) place: place all objects into an empty grid efficiently while respecting stackability, always return the name and position of all the objects, both the objects modified and not modified, do not return an empty dictionary
    c) adjust: return a single tuple of the form ["object_name", "DIRECTION", amount]. The requested direction and amount are the actual move that will be applied if legal. Rewards compare that requested move against the best legal position for the same object, so keep improving one object until it reaches its best legal position.

    Active action schema for this task:
    a) segment: {{"segment": {{"object_name": [x, y, z, stackable], ...}}}}
    b) place: {{"place": {{"object_name": [x, y, z, stackable], ...}}}}
    c) adjust: {{"adjust": ["object_name", "DIRECTION", amount]}}

    Return only the JSON object for the active task.
    Do not include inactive task fields.
    Do not include explanations, commentary, or extra keys.
    If there is no valid move, return an empty object for the active task.
    For adjust, return exactly one tuple when valid_adjustments is not empty.

    Additional context:
    Objects and dimensions: {OBJECTS}
    """
).strip()


def empty_action_json() -> str:
    empty_value = [] if TASK_NAME == "adjust" else {}
    return json.dumps({TASK_ACTION_FIELDS[TASK_NAME]: empty_value})


def _empty_action_payload_dict() -> Dict[str, Any]:
    return {
        "segment": {},
        "place": {},
        "adjust": (),
    }


def current_reward_events(observation: SorterObservation) -> List[float]:
    return list(observation.reward[0]) if observation.reward else []


def current_feedback_events(observation: SorterObservation) -> List[str]:
    return list(observation.reward[1]) if observation.reward else []


def latest_advisory(observation: SorterObservation) -> str:
    advisory_values = observation.advisory if observation.advisory else []
    return advisory_values[-1] if advisory_values else "No advisory available."


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(
    step: int, action: str, reward: float, done: bool, error: Optional[str]
) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_user_prompt(
    step: int,
    last_observation: SorterObservation,
    last_reward_total: float,
    last_reward_events: List[float],
    last_reward_feedback_events: List[str],
    last_advisory: str,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    observation_payload = last_observation.model_dump()
    allowed_fields = TASK_OBSERVATION_FIELDS[TASK_NAME]
    observation_payload = {
        key: value
        for key, value in observation_payload.items()
        if key in allowed_fields
    }

    valid_adjustments: Dict[str, List[int]] = {}
    if TASK_NAME == "adjust":
        valid_adjustments = build_adjust_candidates(last_observation)

    advisory_line = (
        f"Advisory: {last_advisory}"
        if last_advisory != "No advisory available."
        else "Advisory: None"
    )
    segment_guidance = ""
    if TASK_NAME == "segment":
        segment_guidance = (
            "Segment guidance: assign one object label to each observed position using only observed_objects[].dims, stackable, and volume. "
            "Do not rely on hidden candidate label lists; infer from the object catalog and the observation only. "
            "Return a flat mapping like "
            '{"segment":{"book":[x,y,z,true],"bottle":[x,y,z,false]}}. '
            "Do not put a place key inside segment."
        )

    return textwrap.dedent(
        f"""
        Step: {step}
        Last observation: {json.dumps(observation_payload, default=_json_default)}
        Adjustable objects: {json.dumps(valid_adjustments, default=_json_default)}
        Last step reward total: {last_reward_total:.2f}
        Last step reward events: {json.dumps(last_reward_events, default=_json_default)}
        Last step feedback: {json.dumps(last_reward_feedback_events, default=_json_default)}
        {advisory_line}
        {segment_guidance}
        Previous steps:
        {history_block}
        Run the next step.
        """
    ).strip()


def _json_default(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def get_model_message(
    client: OpenAI,
    step: int,
    last_observation: SorterObservation,
    last_reward_total: float,
    last_reward_events: List[float],
    last_reward_feedback_events: List[str],
    last_advisory: str,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(
        step,
        last_observation,
        last_reward_total,
        last_reward_events,
        last_reward_feedback_events,
        last_advisory,
        history,
    )
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
        )
        text = (completion.choices[0].message.content or "").strip()
        return text if text else empty_action_json()
    except Exception as exc:
        print(f"[DEBUG] Model request failed: {exc}", flush=True)
        return empty_action_json()


def _extract_json_payload(output_str: str) -> str:
    output_str = output_str.strip()

    if output_str.startswith("```"):
        lines = output_str.splitlines()
        if len(lines) >= 3:
            output_str = "\n".join(lines[1:-1]).strip()

    start = output_str.find("{")
    end = output_str.rfind("}")

    if start == -1 or end == -1 or end < start:
        raise JSONDecodeError("No JSON object found in model output", output_str, 0)

    return output_str[start : end + 1]


def _normalize_action_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    action_field = TASK_ACTION_FIELDS[TASK_NAME]
    action_value = payload.get(action_field, payload)
    normalized_payload = _empty_action_payload_dict()

    if TASK_NAME == "adjust":
        if action_value is None:
            action_value = []
        if isinstance(action_value, dict):
            if not action_value:
                action_value = []
            elif len(action_value) == 1:
                obj_name, move = next(iter(action_value.items()))
                if isinstance(move, (list, tuple)) and len(move) == 2:
                    action_value = [obj_name, move[0], move[1]]
                else:
                    raise ValueError(
                        "'adjust' dict form must be {'obj_name': ['DIRECTION', amount]}."
                    )
            else:
                raise ValueError(
                    "'adjust' dict form must contain exactly one object entry."
                )
        if not isinstance(action_value, (list, tuple)):
            raise ValueError(f"'{action_field}' must be a JSON array.")
        normalized_payload[action_field] = tuple(action_value)
        return normalized_payload

    if action_value == [] or action_value is None:
        action_value = {}
    if not isinstance(action_value, dict):
        raise ValueError(f"'{action_field}' must be a JSON object.")

    normalized_payload[action_field] = action_value
    return normalized_payload


def parse_action(message: str) -> Dict[str, Any]:
    payload = json.loads(_extract_json_payload(message))
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object at the top level.")
    return _normalize_action_payload(payload)


def _get_internal_state(env: SorterEnvironment) -> SorterState:
    state = getattr(env, "_state", None)
    if state is None:
        raise RuntimeError("SorterEnvironment does not expose internal state for grading.")
    return state


def _build_observation(env: SorterEnvironment, state: SorterState) -> SorterObservation:
    build_observation = getattr(env, "_return_observation", None)
    if build_observation is None:
        raise RuntimeError(
            "SorterEnvironment does not expose observation construction for grading."
        )
    return build_observation(state)


def _step_reward_chunk(
    previous_rewards: List[float],
    previous_feedback: List[str],
    final_grade: GradeResult,
) -> tuple[List[float], List[str]]:
    reward_start = len(previous_rewards)
    feedback_start = len(previous_feedback)
    return (
        final_grade.rewards[reward_start:],
        final_grade.feedback[feedback_start:],
    )


def _summarize_step_feedback(step_feedback: List[str]) -> str:
    if not step_feedback:
        return "No feedback available."
    return " | ".join(step_feedback)


def apply_graded_step(
    env: SorterEnvironment, task_name: str, action: Dict[str, Any]
) -> tuple[SorterObservation, GradeResult]:
    env.step_count += 1
    internal_state = _get_internal_state(env)
    internal_state.step_count += 1
    graded_result = grade_task(task_name, internal_state, action)
    env._state = graded_result.final_state
    observation = _build_observation(env, graded_result.final_state)
    return observation, graded_result


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = SorterEnvironment(TASK_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    final_grade: Optional[GradeResult] = None

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        last_observation = result
        last_reward_events = current_reward_events(last_observation)
        last_reward_feedback_events = current_feedback_events(last_observation)
        last_reward_total = sum(last_reward_events) if last_reward_events else 0.0
        last_advisory = latest_advisory(last_observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(
                client,
                step,
                last_observation,
                last_reward_total,
                last_reward_events,
                last_reward_feedback_events,
                last_advisory,
                history,
            )

            parse_error: Optional[str] = None
            try:
                action = parse_action(message)
            except (JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                print(f"[DEBUG] Action parse failed: {exc}", flush=True)
                parse_error = str(exc)
                action = json.loads(empty_action_json())

            previous_rewards = list(_get_internal_state(env).reward[0])
            previous_feedback = list(_get_internal_state(env).reward[1])
            result, final_grade = apply_graded_step(env, TASK_NAME, action)
            step_rewards, step_feedback = _step_reward_chunk(
                previous_rewards, previous_feedback, final_grade
            )

            obs = result
            reward = sum(step_rewards) if step_rewards else 0.0
            feedback = _summarize_step_feedback(step_feedback)

            done = final_grade.done

            error = parse_error
            if error is None and feedback.lower().startswith(f"{TASK_NAME} grading failed:"):
                error = feedback

            rewards.append(reward)
            steps_taken = step
            last_observation = obs
            last_reward_total = reward
            last_reward_events = step_rewards
            last_reward_feedback_events = step_feedback
            last_advisory = latest_advisory(obs)

            log_step(
                step=step,
                action=json.dumps(action),
                reward=reward,
                done=done,
                error=error,
            )

            history.append(
                f"Step {step}: {message!r} -> reward_total {reward:+.2f}, reward_events : {step_rewards}, feedback : {step_feedback}, advisory : {last_advisory}"
            )

            if done:
                break

        if final_grade is not None:
            score = final_grade.normalized_score
            success = final_grade.passed
        else:
            score = 0.0
            success = False

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
