# Chinese Medicine Large Language Model(CMLLM)
CMLLM includes several agents to process your question with high accuracy and runs on a knowledge graph constructed by Chinese Medicine expert.

## Download
In order to implement the code, clone the project and create a conda virtual environment first.

`git clone https://github.com/MILK-BIOS/CMLLM.git`

`conda create -n cmllm python=3.10`

`conda activate cmllm`

Then install the dependencies as follows.
 
`cd CMLLM/`

`pip install -r requirements.txt`

## Prepare the knowledge graph

First, please check whether neo4j get installed on your device. You can install neo4j from https://neo4j.com/.

Then copy the csv file in `./database/` to `${NEO4J_HOME}/import/`.

Next open the neo4j cmd and input content in `./database/create.txt`. You will get a knowledge graph now.

## Implement

We prepare a demo of assistant in `test.py`, you can change the prompt sent to LLM by change `task`

`python test.py`