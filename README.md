Doc2Graph
==============================
We put the code for constructing **Concept Interaction Graph** and run experiments in our ACL 2019 submission here. 

We also put the datasets here: **Chinese News Same Event dataset (CNSE)** and **Chinese News Same Story dataset (CNSS)**. 

Requirement
-------------------
To run the code successfully, you will need (just install the most recent version)

- Pytorch
- Graph-tool

How to use
------------------
**Run experiments: ** Please go to src/models/CCIG, and run script.sh.

**Process data: ** Please go to src/models/CCIG/data/ and run feature_extractor.py.

**Datasets:** The CNSE dataset in the paper is in data/raw/event-story-cluster/same_event_doc_pair.txt,   and the CNSS dataset is located in data/raw/event-story-cluster/same_story_doc_pair.txt.

I have uploaded the dataset to baidupan as well. You can download via:
link:https://pan.baidu.com/s/1EDna1rspOceQUENprgXZFg  password:nhzs

**CIG visualization: ** we put some figures of CIG with community detection in the folder ``CIG visualization by graph-tool''.


Project Organization
------------

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │
    ├── src                <- Source code for use in this .
       │
       └──models/CCIG         
           │
           ├── data     <- code for extract graph features 
           ├── util     <- code for some functions 
           ├── loader.py  <- code for loading graph features
           ├── main.py  <- main code for running models
           └── models   <- pytorch code for our model
    
