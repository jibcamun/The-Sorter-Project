---
title: The Sorter Project
emoji: "📊"
colorFrom: purple
colorTo: yellow
sdk: docker
sdk_version: "4.66"
python_version: "3.13"
app_file: server/app.py
pinned: false
app_port: 8000
---

# The Sorter Project

`sorter` is an OpenEnv environment for warehouse slotting and re-slotting. It models three tasks that human warehouse operators and inventory planners actually perform:

1. identify incoming items from known inventory metadata
2. reposition one item to a better legal slot
3. reorganize an entire layout to improve storage quality

The environment is designed for agent evaluation rather than low-level robotics. Collision checks, support constraints, bounds, and stackability are enforced by the environment, while the agent is evaluated on recognition, layout decisions, and improvement over time.

## Why This Is A Real-World Task

This environment maps to common operations in warehouses, storage rooms, micro-fulfillment centers, and factory floor inventory areas:

- `segment`: match observed object geometry to known SKU metadata during inbound triage
- `adjust`: move a single misplaced or newly arrived item to a better slot
- `place`: perform full re-slotting when utilization changes or the area must be reorganized

The environment is intentionally framed as warehouse slotting instead of a toy rearrangement puzzle. A human planner would recognize the task immediately: identify what is in the aisle, decide where one object should move next, or propose a better global arrangement under physical constraints.

## Environment Summary

The world state is a 3D grid:

- `current_grid`: current occupancy of the warehouse volume
- `weighted_grid`: a dense preference field that defines which placements are more valuable
- `objects_present`: ground-truth object placements, exposed selectively by task

The reward is shaped across the whole trajectory:

- partial credit for correct segmentation
- incremental reward for better local moves in `adjust`
- incremental reward for improving the total layout in `place`
- penalties for malformed, illegal, destructive, or clearly unhelpful actions

## Tasks

### Task 1: `segment` (Easy)

Objective: identify every observed object by name from its visible properties and position.

Why it is easy:

- the inventory catalog is known
- object dimensions and stackability are exposed
- the task is discrete and fully auditable

What the agent sees:

- observed object descriptors
- candidate positions
- grid dimensions
- reward and advisory history

What the agent must return:

```json
{
  "segment": {
    "book": [0, 0, 0, false],
    "bottle": [2, 1, 0, true]
  }
}
```

Done condition:

- the task ends when every object is labeled at the exact correct position

Grader rule:

- exact-match mapping from object name to position
- score normalized into `0.0-1.0`

Reward behavior:

- positive reward for each exact correct mapping
- negative reward for wrong labels, wrong positions, or malformed payloads

### Task 2: `adjust` (Medium)

Objective: improve the location of one movable object using the legal candidate moves exposed by the environment.

Why it is medium:

- the agent must choose among legal but not necessarily optimal moves
- the focus locks onto one object after the first valid move
- reward is based on improvement, not just validity

What the agent sees:

- current object placements
- adjustable objects
- ranked legal targets with score deltas
- current adjustment focus and visited positions

What the agent must return:

```json
{
  "adjust": ["book", 0]
}
```

Done condition:

- no legal improving moves remain
- no legal targets remain for the chosen object
- or the task logic marks the episode complete

Grader rule:

- validates that the selected `(object_name, option_index)` is legal for the current state
- grades by realized improvement versus achievable improvement
- returns normalized score in `0.0-1.0`

Reward behavior:

- positive reward for better legal moves
- zero for legal but non-improving moves
- negative reward for worsening or invalid moves

### Task 3: `place` (Hard)

Objective: propose a full legal layout for all objects that improves total layout quality.

Why it is hard:

- every object must be placed
- overlap, support, bounds, and stackability must all remain valid
- reward depends on total layout quality, not one local move
- the optimal solution is hidden

What the agent sees:

- current object placements
- full object set
- weighted preference field
- reward and optimizer advisory

What the agent must return:

```json
{
  "place": {
    "book": [0, 0, 0, false],
    "bottle": [2, 1, 0, true],
    "box": [4, 0, 0, false]
  }
}
```

Done condition:

- a valid accepted layout may end the episode when the task determines the reorganization is complete

Grader rule:

- validates completeness and physical legality of the full layout
- compares achieved layout quality against the current state
- returns normalized score in `0.0-1.0`

Reward behavior:

- positive reward for improving total layout score
- negative reward for incomplete, overlapping, unsupported, or out-of-bounds layouts

## Difficulty Progression

The three tasks intentionally progress from local recognition to local optimization to global optimization:

| Task | Difficulty | Core skill |
| --- | --- | --- |
| `segment` | Easy | Recognition and exact mapping |
| `adjust` | Medium | Constrained local search |
| `place` | Hard | Global layout optimization |

## Action Space

The environment exposes a typed `SorterAction` Pydantic model composed of three task-specific action fields.

| Field | Type | Used in | Meaning |
| --- | --- | --- | --- |
| `segment` | `Dict[str, PositionTuple]` | `segment` | Predicted object-name to position mapping |
| `adjust` | `Tuple[str, int]` or empty tuple | `adjust` | Choose one legal move by object name and exposed option index |
| `place` | `Dict[str, PositionTuple]` | `place` | Proposed full layout for every object |

`PositionTuple = (x, y, z, rotated)`

- `x`, `y`, `z` are integer coordinates in the grid
- `rotated` is a boolean indicating whether the object is rotated relative to its default orientation

One action payload should target the active task and leave the other task fields empty.

## Observation Space

The environment exposes a typed `SorterObservation` Pydantic model. Some fields are global, while others are revealed only for the active task.

### Global Observation Fields

| Field | Type | Visible in | Meaning |
| --- | --- | --- | --- |
| `grid_dims` | `Tuple[int, int, int]` | All tasks | Dimensions of the warehouse grid |
| `weighted_grid` | `NDArray` | All tasks | Preference field defining what counts as a better placement |
| `current_grid` | `NDArray` | All tasks | Current occupancy grid |
| `reward` | `float` | All tasks | Latest scalar reward |
| `reward_details` | `Tuple[List[float], List[str]]` | All tasks | Reward event log and matching textual feedback |
| `advisory` | `List[str]` | All tasks | Guidance, including optimizer messages |
| `done` | `bool` | All tasks | Whether the current episode is finished |

### Task-Specific Observation Fields

| Field | Type | Task | Meaning |
| --- | --- | --- | --- |
| `positions_segment` | `Dict[str, PositionTuple]` | `segment` | Segment-specific internal positions tracked by the task |
| `positions` | `List[PositionTuple]` | `segment` | Candidate positions shown during segmentation |
| `observed_objects` | `List[Dict[str, Any]]` | `segment` | Visible descriptors such as dimensions, stackability, and volume |
| `last_segment_attempt` | `Dict[str, PositionTuple]` | `segment` | Most recent segmentation payload |
| `objects_present` | `Dict[str, PositionTuple]` | `adjust`, `place` | Currently exposed object placements |
| `positions_adjust` | `Dict[str, PositionTuple]` | `adjust` | Positions relevant to adjustment |
| `adjustable_objects` | `List[Dict[str, Any]]` | `adjust` | Objects eligible for movement and their legal targets |
| `adjust_focus_object` | `str` | `adjust` | Object currently locked for multi-step adjustment |
| `adjust_start_position` | `PositionTuple` or empty tuple | `adjust` | Original position of the focused object |
| `adjust_visited_positions` | `List[Tuple[int, int, int]]` | `adjust` | Legal coordinates already explored |
| `adjust_action_options` | `List[List[Any]]` | `adjust` | Currently valid `(object_name, option_index)` choices |
| `positions_place` | `Dict[str, PositionTuple]` | `place` | Placement map used during full-layout optimization |

## Environment Design Choices

### State Management

- `reset()` creates a fresh random layout, clears reward history, clears task-specific caches, and starts a new episode
- `step(action)` applies the task-specific transition and returns a typed `SorterObservation`
- `state` returns the current typed internal state for inspection and grading

### Why `weighted_grid` Is Exposed

`weighted_grid` reveals what the environment prefers without exposing the optimizer answer. This gives agents useful reward-shaping context while preserving the optimization problem.

### Why `adjust` Uses Option Indices

The environment surfaces legal candidate moves instead of the full coordinate search space. This keeps the task tractable and better matches real planning systems that propose feasible actions before a policy chooses among them.

### Why Constraints Are Enforced By The Environment

Bounds checking, overlap prevention, support requirements, and stackability are handled internally so the task evaluates planning quality rather than low-level physics bookkeeping.

## Reward Function

### Mathematical Form

$$
Reward =
\begin{cases}
\sum_{o \in O} \frac{20}{N}(2I[o]-1), & \text{valid segment} \\
\mathrm{clamp}\left(\frac{30}{N}(\mathrm{score}(o,p_{\text{new}})-\mathrm{score}(o,p_{\text{old}})), -\frac{30}{N}, \frac{30}{N}\right), & \text{valid adjust} \\
\mathrm{clamp}\left(\frac{50}{N}(L(P_{\text{new}})-L(P_{\text{old}})), -50, 50\right), & \text{valid place} \\
-\frac{20}{N}, & \text{invalid segment} \\
-\frac{30}{N}, & \text{invalid adjust} \\
-\frac{50}{N}, & \text{invalid place}
\end{cases}
$$

Where:

- `N` is the number of objects in the episode
- `score(obj, pos)` is the mean `weighted_grid` value covered by the object at that position
- `L(layout)` is the total layout score across all objects

### Reward Interpretation

- `segment` rewards exact object identification and penalizes wrong or malformed submissions
- `adjust` rewards local score improvement and penalizes invalid or harmful moves
- `place` rewards global layout improvement and penalizes illegal full-layout proposals

The reward is meaningful across the trajectory, not only at the terminal step. Agents can detect partial progress, plateauing, and regressions from the reward stream and advisory feedback.

## Determinism And Grading

Task graders are implemented in `graders.py` and return normalized results in `0.0-1.0`.

- `grade_segment(...)` checks exact segmentation correctness
- `grade_adjust(...)` checks legal adjustment execution and progress
- `grade_adjust_progress(...)` assigns partial credit when the rollout ends before full completion
- `grade_place(...)` checks legality and global layout quality
- `grade_task(...)` dispatches to the appropriate task grader

For a fixed state and action payload, grading is deterministic. The environment itself resets to randomized layouts, but once a specific episode state exists, the same action produces the same grade and reward transitions.

## Episode Boundaries

Episode boundaries are task-dependent and are designed to be sensible for each workflow:

- `segment`: ends when all objects are labeled correctly
- `adjust`: ends when no improving legal moves remain, no legal targets remain, or the task logic completes
- `place`: ends when a valid full layout is accepted and the task marks the episode done

Invalid actions do not silently pass:

- malformed payloads are penalized
- illegal moves are penalized
- incomplete or unsupported layouts are penalized

The baseline inference rollout is capped by `MAX_STEPS = 8` in `inference.py`.

## API

The FastAPI app is defined in `server/app.py`.

Available endpoints:

- `POST /reset`
- `POST /step`
- `GET /state`
- `GET /schema`
- `WS /ws`

Typical interaction:

1. call `POST /reset`
2. inspect the task-relevant observation fields
3. call `POST /step` with one `SorterAction`
4. continue until `done=true`

## Setup

### Environment Variables

The baseline and submission flow use the following variables:

| Variable | Required | Purpose |
| --- | --- | --- |
| `OPENAI_API_KEY` | Yes for inference | API key consumed by the OpenAI client |
| `API_BASE_URL` | Yes for submission | Base URL for the LLM provider endpoint |
| `MODEL_NAME` | Yes for submission | Model identifier used by the baseline |
| `HF_TOKEN` | Yes for submission config | Hugging Face token for Space or submission workflows |

Notes:

- `inference.py` uses the OpenAI client and now prefers `OPENAI_API_KEY`
- `HF_TOKEN` is included because the submission instructions require it in environment configuration
- `API_KEY` is still accepted as a backward-compatible fallback, but `OPENAI_API_KEY` is the intended variable

Example `.env`:

```bash
OPENAI_API_KEY=your-provider-key
API_BASE_URL=https://integrate.api.nvidia.com/v1
MODEL_NAME=openai/gpt-oss-120b
HF_TOKEN=your-hugging-face-token
```

### Local Installation

```bash
uv sync
```

### Run The Server

```bash
uv run --project . server --host 0.0.0.0 --port 8000
```

### Run The Baseline

```bash
python inference.py
```

To run one task only:

```bash
THE_SORTER_PROJECT_TASK=segment python inference.py
THE_SORTER_PROJECT_TASK=adjust python inference.py
THE_SORTER_PROJECT_TASK=place python inference.py
```

### Validate OpenEnv Compliance

```bash
openenv validate
```

### Validate The Submission End To End

```bash
bash validate-submission.sh https://your-space.hf.space .
```

### Docker

Build:

```bash
docker build -t sorter .
```

Run:

```bash
docker run --rm -p 8000:8000 sorter
```

## Baseline Scores

The baseline script is `inference.py` in the project root, as required by the submission rules. It emits structured stdout in the required `[START]`, `[STEP]`, and `[END]` format and uses the OpenAI client for model calls.

Because baseline scores depend on the configured `API_BASE_URL`, `MODEL_NAME`, and credentials, record the measured values from your current run before submission.

Suggested reporting table:

| Task | Model | Score | Steps | Notes |
| --- | --- | --- | --- | --- |
| `segment` | `MODEL_NAME` | `TBD` | `TBD` | Populate from `[END]` |
| `adjust` | `MODEL_NAME` | `TBD` | `TBD` | Populate from `[END]` |
| `place` | `MODEL_NAME` | `TBD` | `TBD` | Populate from `[END]` |
| `overall` | `MODEL_NAME` | `TBD` | `TBD` | Average or aggregated score |

## Example Payloads

### Example `segment` Action

```json
{
  "segment": {
    "book": [0, 0, 0, false],
    "bottle": [2, 1, 0, true]
  }
}
```

### Example `adjust` Action

```json
{
  "adjust": ["book", 0]
}
```

### Example `place` Action

```json
{
  "place": {
    "book": [0, 0, 0, false],
    "bottle": [2, 1, 0, true],
    "box": [4, 0, 0, false]
  }
}
```

### Example `segment` Observation Slice

```json
{
  "grid_dims": [8, 8, 4],
  "positions": [[0, 0, 0, false], [2, 1, 0, true]],
  "observed_objects": [
    {
      "position": [0, 0, 0, false],
      "dims": [1, 2, 1],
      "stackable": false,
      "volume": 2
    }
  ],
  "reward": 0.0,
  "done": false
}
```

### Example `adjust` Observation Slice

```json
{
  "objects_present": {
    "book": [0, 0, 0, false]
  },
  "adjust_focus_object": "",
  "adjust_action_options": [["book", 0], ["book", 1]],
  "reward": 0.0,
  "done": false
}
```

### Example `place` Observation Slice

```json
{
  "objects_present": {
    "book": [0, 0, 0, false],
    "bottle": [2, 1, 0, true]
  },
  "positions_place": {},
  "reward": 0.0,
  "done": false
}
```

## Project Structure

```text
sorter/
|-- config/                  # Grid and object configuration
|-- model_types/             # Task-specific type fragments
|-- models/                  # Typed action, observation, and state models
|-- server/                  # FastAPI app and OpenEnv environment wrapper
|-- tasks/                   # segment, adjust, place task logic
|-- utils/                   # Grid and reward helpers
|-- client.py                # Client utilities
|-- graders.py               # Deterministic task graders
|-- inference.py             # Baseline agent runner
|-- openenv.yaml             # OpenEnv metadata
|-- Dockerfile               # Container build
|-- validate-submission.sh   # Pre-submission validation helper
`-- README.md
```

## Links

Hugging Face repository: https://huggingface.co/spaces/Jibrann/sorter

Hugging Face Space: https://jibrann-sorter.hf.space

GitHub repository: https://github.com/jibcamun/The-Sorter-Project

## Related Work

- [Jumanji](https://github.com/instadeepai/jumanji): RL environments for structured decision-making and optimization
- [miniRL](https://proxyapps.exascaleproject.org/app/minirl/): lightweight RL experimentation framework
- [BabyAI](https://arxiv.org/abs/1810.08272): benchmark for learning complex behavior from simpler sub-tasks
