---
title: The Sorter Project
emoji: 🔊
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

## The Purpose 
Building an environment to make AI models learn on how to **identify**, **place** and **adjust** the position of things in the environment which are scattered in a *random* fashion.

## Real Life Application
We came up with this idea, keeping in mind its application in factories, warehouses and storage facilities. _(and even your coffee table!)_

## The Problem
### **The Industrial Perspective / Micro Perspective**

Companies spend milllions if not billions on establishing, maintaining and organisising warehouses and storage facilities, and in a densely populated country like India with increasing demand for land and with the surging property prices efficient storage and orgsation becomes the ***need of the hour***, leading to the demand for an environment or an agent that can help companies and organisations and provide them with ways for the maximum efficient and logical storage of their "objects".
The environments and agents that specialise in full fledged identifying, sorting, stacking and organising of objects or warehouse material are few in number, and ***we are here to fill that gap***.

### **The Populational Perspective / Macro Persepective**

With increase in population causing decrease of 'Open Spaces' it becomes extremely important to **build societies and localities that can cater to a huge chunk of population** and in such a case, The Sorter Project though being mainly built for industrial application, becomes an extremely useful tool that allows proper space utilisation to accomatodate more people whilst taking minimum space. _(so in the near future we might not have to shift to mars)_

## Our Solution
_We have developed this environment with 'ease' thanks to OpenEnv!_ <br>
Our Sorter Project consists of ***_3_ different parts or tasks***:

### **TASK 1**: The Segmentation Problem (Easy)<br>
Our Project has an ****Segmentation Action**** that makes agents identify the names of objects which are present in the environment, based on the positions and predefined knowledge of the dimensions of objects. (In a typical factory or warehouse setting the number of objects in the inventory and their dimensions are highly documented.)

### **TASK 2**: The Adjustment Problem (Medium)<br>
Our Project also provides an ****Adjustment Action**** for the agent to adjust the position of a particular object and place it in the most optimal position by trying multiple times throught the episode. _(because no one's good in their first try!)_
    
### **TASK 3**: The Placement Problem (Hard)<br>
Our Project has a ****Placement Action**** that allows agents to place all the objects present in the environment in the best possible positions to maximize space and optimise the arrangement. This acts as a complete reorganisation of the objects in the environment.<br>

## Quick Question: Wait... Then what is the difference between `adjust` and `place`?<br>
That is a very valid thought that might strike anyone reading the task description, and there is a marginal difference between the two actions.<br><br>
The `adjust` task is to modify the position *one* object, for example, to adjust the position of a stack of boxes that have just arrived at the warehouse to the best possible position.<br>
The `place` task is to modify or place *all* the objects in the most "optimal" positions, for example, to reorganise a warehouse to accomodate for a surge in the inventory or just to organise the warehouse better or god forbid, an earthquake hits the warehouse and all the objects fall everywhere, stacks of objects or boxes are scattered and misplaced, then this can help reorganise the warehouse.

**NOTE**: In a real life scenario, idealy all tasks would be done sequentially, in a chronological order for the agent to function independently without any external context on the objects present, allowing it to function on its own volition and giving it full freedom.

## Environment Design

- We have designed our environment to simulate a warehouse or any space for that matter as a three dimensional grid, although not an entirely accurate representation, most companies like NVIDIA, Siemens, Dassault Systèmes, Microsoft and IBM use similiar or near same systems as recorded in [Warehouse Digital Twins](https://docs.nvidia.com/learning/physical-ai/assembling-digital-twins/latest/index.html) by NVIDIA and [Supply Chain 2.0](https://www.microsoft.com/en-us/industry/blog/manufacturing-and-mobility/2026/03/24/supply-chain-2-0-how-microsoft-is-powering-simulations-ai-agents-and-physical-ai/?msockid=2baed25235d468720632c43e34b669fd) by Microsoft. However these designs are mostly used by the massive coorporations for inventory management, automations and other related aspects of warehouse management, and we take inspiration from their logic to build something novel.
- To achieve our goal we created two grid namely, the main grid called `current_grid` which contains a few objects randomly initialised to simulate objects present in the warehouse and `weighted_grid`, which is a duplicate of the main grid made in order to introduce noise and indicate the preference of placement.
- Weighted grid is exposed because it defines what "better" means, but not the "optimal" positions of objects in the warehouse. This preserves the optimization problem while still making the reward landscape quite interpretable.
- We intentionally expose only small, task-relevant slices of state, in order to provide better context for the model to perform each action, without exposing vital "ground truths" and destroying the reinforcement learning aspect of it.
- The environment keeps `objects_present` as internal truth, but reveals it selectively depending on the task. It is revealed in `adjust` and `place` functions, in order to ensure that the ability of the tasks of functions to run independently is not compromised. 
- We return the latest scalar reward together with textual feedback and advisory messages. This was a choice made to support both the reinforcement learning logic and help the LLM in iterative correction.
- Validity constraints such as bounds, non-overlap, stackability, and support are enforced inside the environment instead of delegated to the agent. This keeps the task focused on decision quality and finding the best "optimal" position, rather than simulating low level frivolous tasks such as obstacle prevention, etc, which does not fit our pursuit.

### Segmentation Action (`segment`)

- `segment` exposes object geometry and positions to be identified, but not solved labels. This forces the policy to perform recognition rather than exploit a direct answer key. 
- The task is framed as exact mapping from object name to position. This keeps the objective discrete and auditable, which is useful for both grading and debugging agent failures. The model is aware of the objects that are a part of the inventory and their dimensions but unaware of where various objects are located, so its job here is to simply match the object's name that it is aware of to the positions in the grid.

### Adjustment Action (`adjust`)

- `adjust` exposes only legal candidate targets for one movable object at a time. This reduces the action space on purpose and turns the task into local improvement under constraints instead of brute force search over the full grid.
- Adjustment episodes are locked to a single focus object after the first valid move. This was designed to prevent agents from gaming reward by hopping across objects and to encourage coherent multiple steps of refinement.
- We expose legal options (the options that are not out of bounds or intersect with other objects) in `adjust` instead of every possible coordinate. This keeps inference tractable, prevents combinatorial explosion, and matches real systems where a planner often proposes feasible candidate actions rather than raw continuous freedom. 
- This might not look like optimal environment design but the goal here is to find the "best" and "optimal" position to place the object not figure out if the position is valid or not, which in real life can be done with pre existing computer vision models and with IR Sensors (in the robotic aspect).

### Placement Action (`place`)

- `place` is the most open ended task, so it exposes the object set and current placements while still hiding the optimizer's solution. The agent must learn to improve layouts from reward and constraints rather than copy an oracle plan.
- The full layout requirement is clear, the policy is evaluated on global arrangement quality, not isolated object moves. This makes `place` a true reorganization task rather than a repeated version of `adjust`.

## Technical Details

### Reward Logic
#### Mathematical Formulation
$$
\begin{aligned}
R =\;& \text{valid segment: } \sum_{o \in O} \frac{20}{N}(2I[o]-1) \\
& \text{valid adjust: } \mathrm{clamp}\left(\frac{30}{N}(\mathrm{score}(o,p_{\text{new}})-\mathrm{score}(o,p_{\text{old}})), -\frac{30}{N}, \frac{30}{N}\right) \\
& \text{valid place: } \mathrm{clamp}\left(\frac{50}{N}(L(P_{\text{new}})-L(P_{\text{old}})), -50, 50\right) \\
& \text{invalid segment: } -\frac{20}{N} \\
& \text{invalid adjust: } -\frac{30}{N} \\
& \text{invalid place: } -\frac{50}{N}
\end{aligned}
$$


#### Explanation
Let:

- `N` = total number of objects in the current episode
- `score(obj, pos)` = mean value of the `weighted_grid` over the cells occupied by an object at `pos`
- `layout_score(layout)` = sum of `score(obj, pos)` over all placed objects

Rewards are stored as an event log: the environment appends both the numeric reward and a matching text feedback message after each action. The observation returns the latest numeric reward as `reward`, and the full log as `reward_details`.

**Task 1 (`segment`)**

- Base unit: `20 / N`
- For each submitted object:
- `+20 / N` if the object name is valid and its predicted position exactly matches the ground-truth position
- `-20 / N` if the object name is unknown or the position is wrong
- `-20 / N` for malformed submissions, such as the wrong number of objects or nested task keys in the payload
- The task finishes only when every object has been labeled correctly

**Task 2 (`adjust`)**

- Base unit: `30 / N`
- Only one object can be adjusted per step, and the episode stays locked onto that same object after the first valid choice
- Legal moves must:
- stay inside the grid
- move to empty space
- keep full support underneath the object
- avoid moving objects that have something stacked on top of them
- For a legal move, reward is based on score improvement:
- `reward = clamp((new_score - current_score) * (30 / N), -(30 / N), +(30 / N))`
- This means:
- positive reward for improving the object's weighted-grid score
- zero reward if the move is legal but does not improve the score
- negative reward for legal moves that make the position worse
- Invalid actions receive `-30 / N`
- If there are no remaining legal target positions for the chosen object, the environment returns `0` and ends the episode
- The episode also ends once there are no unvisited legal moves that improve the score further

**Task 3 (`place`)**

- Base unit: `50 / N`
- The action is evaluated as a complete layout: every object must have a placement
- A proposed layout is rejected with `-50 / N` if it is incomplete, out of bounds, overlapping, or unsupported
- For a valid layout, reward is based on total layout improvement relative to the previous accepted layout:
- `reward = clamp((layout_score(new) - layout_score(previous)) * (50 / N), -50, +50)`
- This means the agent is rewarded for improving the total weighted-grid score of the whole arrangement, not just for matching a single hard-coded target placement
- After accepting a valid layout, the environment compares it against an OR-Tools optimizer result and adds advisory feedback indicating whether:
- the layout is optimal
- some objects still have better placements available
- or only a feasible reference solution was found within the solver time limit
            
### Demonstration
To view the demonstation of how this environment is supposed to work clone this repository and run the `inference.py`. Set the following environment variables in a `.env` file:
- API_BASE_URL
- MODEL_NAME
- API_KEY

### Project Structure

```
sorter/
├── config/                  # Basic configurations
│   ├── grid.py
│   └── objects.py
│
├── models/                  # Task based models (action, observation, state)
│   ├── __init__.py
│   ├── adjust_model.py
│   ├── place_model.py
│   └── segment_model.py
│
├── server/                  # FastAPI server
│   ├── app.py
│   ├── sorter_environment.py
│   └── requirements.txt
│
├── tasks/                   # Core logic for each task
│   ├── adjust.py
│   ├── place.py
│   └── segment.py
│
├── utils/                   # Utility functions
│   ├── grids.py
│   └── rewards.py
│
├── __init__.py              # Environment initialization
├── client.py                # API client processing
├── graders.py               # Task evaluation logic
├── inference.py             # Baseline agent 
├── openenv.yaml             # OpenEnv configuration
├── pyproject.toml           # Project configuration
├── Dockerfile               # Container setup
└── README.md                # This file
```

## Links
****Huggingface Repository Link****: https://huggingface.co/spaces/Jibrann/sorter <br>
****Huggingface Spaces Link****: https://jibrann-sorter.hf.space <br>
****Github Link**** (this page): https://github.com/jibcamun/The-Sorter-Project

## Related Works
[Jumanji](https://github.com/instadeepai/jumanji)<br>
[miniRL](https://proxyapps.exascaleproject.org/app/minirl/)<br>
[BabyAI](https://arxiv.org/abs/1810.08272)<br>
