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
Our Project also provides an ****Adjust Action**** for the agent to adjust the position of a particular object and place it in the most optimal position. _(because no one's good in their first try!)_
    
### **TASK 3**: The Placement Problem (Hard)<br>
Our Project has a ****Placement Action**** that allows agents to place all the objects present in the environment in the best possible positions to maximize space and optimise the arrangement. This acts as a complete reorganisation of the objects in the environment.<br>

## Quick Question: Wait... Then what is the difference between `adjust` and `place`?<br>
That is a very valid thought that might strike anyone reading the task description, and there is a marginal difference between the two actions.<br><br>
The `adjust` task is to modify the position *one* object, for example, to adjust the position of a stack of boxes that have just arrived at the warehouse to the best possible position.<br>
The `place` task is to modify or place *all* the objects in the most "optimal" positions, for example, to reorganise a warehouse to accomodate for a surge in the inventory or just to organise the warehouse better or god forbid, an earthquake hits the warehouse and all the objects fall everywhere, stacks of objects or boxes are scattered and misplaced, then this can help reorganise the warehouse.

**NOTE**: In a real life scenario, idealy all tasks would be done sequentially, in a chronological order for the agent to function independently without any external context on the objects present, allowing it to function on its own volition and giving it full freedom.

## Environment Design

## Technical Details

### Reward Logic
***N := total number of objects***
***w:= the mean value of the region of the weighted grid taken by the object***

**Task 1 (Segmentation):** +20/N per correct position and object found
                           -20/N in all other cases

**Task 2 (Adjustment):**    (+30/N)*w for each correct adjustment
                            -30/N in all othe
r cases
**Task 3 (Placement) :** (+50   /N)*w for each correct placement
               

             -50/N in all other cases
            
### Demonstration


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

### LLM Used
GPT oss 120B

## How to Run
Just run `python inference.py`to utilse the environment by running the baseline agent

## Links
****Huggingface Link**** (to run `inference.py`): https://huggingface.co/spaces/Jibrann/app <br>
****Github Link**** (this page): https://github.com/jibcamun/Reinforcement-Learning-Object-Placement

## Related Works
[Jumanji](https://github.com/instadeepai/jumanji)<br>
[miniRL](https://proxyapps.exascaleproject.org/app/minirl/)<br>
[BabyAI](https://arxiv.org/abs/1810.08272)<br>
