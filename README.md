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
Our Sorter Project consists of ***_3_ different parts*** and ***_4_ different processes/tasks***:

### **Part 1**: The Segmentation Problem<br>
**Task 1:** Our Project has an ****Segmentation Action**** that makes agents identify objects which is rare to find in multiple similar environments.
### **Part 2**: The Identification Problem<br>
**Task 2:** Our Project has a ****Identification Action**** which is though a part of the Segmentation task, is slightly different, it allows agents to segregate objects into **stackable** and **not stackable** which will be of high importance while addressing the next problem.
    
### **Part 3**: The Placement Problem<br>
**Task 3:** Our Project has a ****Placement Action**** that allows agents to place things it has found.<br>
**Task 4:** It also provides an ****Adjust Action**** for the agent to adjust things _(because no one's good in their first try ! )_

## Technical Details
### Reward Logic


### Demonstration

## Links
****Huggingface Link**** (to run `inference.py`): https://huggingface.co/spaces/Jibrann/app <br>
****Github Link**** (this page): https://github.com/jibcamun/Reinforcement-Learning-Object-Placement

## Related Works
[Jumanji](https://github.com/instadeepai/jumanji)<br>
[miniRL](https://proxyapps.exascaleproject.org/app/minirl/)<br>
[BabyAI](https://arxiv.org/abs/1810.08272)<br>
