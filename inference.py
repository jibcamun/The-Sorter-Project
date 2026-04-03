"""
Inference Script Example
===================================
MANDATORY
- Before submitting, ensure the following variables are defined in your environment configuration:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key.
    LOCAL_IMAGE_NAME The name of the local image to use for the environment if you are using from_docker_image()
                     method

- Defaults are set only for API_BASE_URL and MODEL_NAME
    (and should reflect your active inference setup):
    API_BASE_URL = os.getenv("API_BASE_URL", "<your-active-endpoint>")
    MODEL_NAME = os.getenv("MODEL_NAME", "<your-active-model>")

- The inference script must be named `inference.py` and placed in the root directory of the project
- Participants must use OpenAI Client for all LLM calls using above variables

STDOUT FORMAT
- The script must emit exactly three line types to stdout, in this order:

    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>

  Rules:
    - One [START] line at episode begin.
    - One [STEP] line per step, immediately after env.step() returns.
    - One [END] line after env.close(), always emitted (even on exception).
    - reward and rewards are formatted to 2 decimal places.
    - done and success are lowercase booleans: true or false.
    - error is the raw last_action_error string, or null if none.
    - All fields on a single line with no newlines within a line.

  Example:
    [START] task=click-test env=miniwob model=Qwen3-VL-30B
    [STEP] step=1 action=click('123') reward=0.00 done=false error=null
    [STEP] step=2 action=fill('456','text') reward=0.00 done=false error=null
    [STEP] step=3 action=click('789') reward=1.00 done=true error=null
    [END] success=true steps=3 rewards=0.00,0.00,1.00
"""

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
from models import SorterObservation, SorterAction, SorterState
from config.grid import random_grid_dims


load_dotenv()

IMAGE_NAME = os.getenv("IMAGE_NAME")  # If you are using docker image
API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")

API_BASE_URL = os.getenv("API_BASE_URL") or "https://integrate.api.nvidia.com/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "openai/gpt-oss-120b"
TASK_NAME = os.getenv("THE_SORTER_PROJECT_TASK", "segment") or "segment"
BENCHMARK = os.getenv("THE_SORTER_PROJECT_BENCHMARK", "the_sorter_project")
MAX_STEPS = 8
TEMPERATURE = 0.7
MAX_TOKENS = 150
SUCCESS_SCORE_THRESHOLD = 0.1  # normalized score in [0, 1]

# Max possible reward: each token contributes 0.1, across all steps
_MAX_REWARD_PER_STEP = MAX_TOKENS * 0.1
MAX_TOTAL_REWARD = MAX_STEPS * _MAX_REWARD_PER_STEP

SYSTEM_PROMPT = (
    textwrap.dedent(
        """
    You are a professional logistician, facility expert and layout engineer. You will be provided a particular task that can be:
    a) segment: Identify objects by returning the exact object position tuples that are present in the current grid.
    b) place: Place all the objects into an empty grid efficiently. Do not place objects in the air, and respect each object's stackability.
    c) adjust: Adjust the position of one or many objects such that they do not overlap with other objects.

    The SorterAction schema is:
    a) segment: Dict[str, Tuple[int, int, int, bool]]
    b) place: Dict[str, Tuple[int, int, int, bool]]
    c) adjust: Dict[str, Tuple[str, int]]

    Use only the field that matches the current task. The other two fields must use their correct empty type.
    Never use [] for object fields or {} for list fields.

    Field details:
    a) segment: Return a mapping from object name to exact position tuple in the form (x, y, z, stackable).
    b) place: Return a mapping from object name to placement tuple (x, y, z, stackable).
    c) adjust: Return a mapping from object name to a tuple (direction, amount). Direction must be one of RIGHT, LEFT, UP, DOWN, FORWARD, BACKWARD.

    Output:
    1. Always return a valid JSON object conforming to the schema.
    2. Do not include any extra text, explanations, or commentary.
    
    Always return JSON that conforms to the SorterAction Pydantic model:
    a) segment task: {"segment": {"object_name": [x, y, z, stackable], ...}, "place": {}, "adjust": {}}
    b) place task: {"segment": {}, "place": {"object_name": [x, y, z, stackable], ...}, "adjust": {}}
    c) adjust task: {"segment": {}, "place": {}, "adjust": {"object_name": ["DIRECTION", amount], ...}}

    Empty value rules:
    - segment must always be a JSON object, so when empty use {}
    - place must always be a JSON object, so when empty use {}
    - adjust must always be a JSON object, so when empty use {}
    
    Goal:
    1. Maximize cumulative reward.
    2. Learn from reward to improve placement in future steps.
    
    Do not include explanations, text, or extra fields.
    If there is no valid move, return an empty value of the correct type for that field:
    a) segment: {}
    b) place: {}
    c) adjust: {}
    
    The output must be parseable and valid for SorterAction(**json_output).
    """
    ).strip()
    + textwrap.dedent(
        f"Your current task is: {TASK_NAME}. These are the objects with their dimensions: {OBJECTS}. The dimensions of the grid is: {random_grid_dims()}"
    )
)


def empty_action_json() -> str:
    return json.dumps({"segment": {}, "place": {}, "adjust": {}})


def latest_reward(observation: SorterObservation) -> float:
    reward_values = observation.reward[0] if observation.reward else []
    return reward_values[-1] if reward_values else 0.0


def latest_feedback(observation: SorterObservation) -> str:
    feedback_values = observation.reward[1] if observation.reward else []
    return feedback_values[-1] if feedback_values else "No feedback available."


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
    last_reward: float,
    last_reward_feedback: str,
    history: List[str],
) -> str:
    history_block = "\n".join(history[-4:]) if history else "None"
    observation_payload = last_observation.model_dump()
    if TASK_NAME == "segment":
        observation_payload.pop("objects_present", None)

    return textwrap.dedent(
        f"""
        Step: {step}
        Last observation: {json.dumps(observation_payload, default=_json_default)}
        Last reward: {last_reward:.2f}
        Last reward reason: {last_reward_feedback}
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
    last_reward: float,
    last_reward_feedback: str,
    history: List[str],
) -> str:
    user_prompt = build_user_prompt(
        step, last_observation, last_reward, last_reward_feedback, history
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
    normalized = {
        "segment": payload.get("segment", {}),
        "place": payload.get("place", {}),
        "adjust": payload.get("adjust", {}),
    }

    if normalized["segment"] == []:
        normalized["segment"] = {}
    if normalized["place"] == []:
        normalized["place"] = {}
    if normalized["adjust"] == []:
        normalized["adjust"] = {}

    return normalized


def parse_action(message: str) -> SorterAction:
    payload = json.loads(_extract_json_payload(message))
    if not isinstance(payload, dict):
        raise ValueError("Model output JSON must be an object at the top level.")
    return SorterAction(**_normalize_action_payload(payload))


def main():
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

    env = SorterEnvironment(TASK_NAME)

    history: List[str] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=TASK_NAME, env=BENCHMARK, model=MODEL_NAME)

    try:
        result = env.reset()
        last_observation = result
        last_reward = latest_reward(last_observation)
        last_reward_feedback = latest_feedback(last_observation)

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            message = get_model_message(
                client,
                step,
                last_observation,
                last_reward,
                last_reward_feedback,
                history,
            )

            try:
                action = parse_action(message)
            except (JSONDecodeError, ValidationError, ValueError, TypeError) as exc:
                print(f"[DEBUG] Action parse failed: {exc}", flush=True)
                action = SorterAction(**json.loads(empty_action_json()))

            result = env.step(action)

            obs = result
            reward = latest_reward(obs)
            feedback = latest_feedback(obs)

            done = result.done

            error = None

            rewards.append(reward)
            steps_taken = step
            last_observation = obs
            last_reward = reward

            log_step(step=step, action=message, reward=reward, done=done, error=error)

            history.append(
                f"Step {step}: {message!r} -> reward {reward:+.2f}, feedback : {feedback}"
            )

            if done:
                break

        score = sum(rewards) / len(rewards) if rewards else 0.0
        if rewards:
            min_reward = min(rewards)
            max_reward = max(rewards)
            reward_range = max_reward - min_reward
            score = (score - min_reward) / reward_range if reward_range else 0.0
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        try:
            env.close()
        except Exception as e:
            print(f"[DEBUG] env.close() error (container cleanup): {e}", flush=True)
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
