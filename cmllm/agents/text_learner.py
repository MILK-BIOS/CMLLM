import langchain
from langchain.graphs import Neo4jGraph
import torch


class TextLearner:
    def __init__(self, template: str=None):
        if template:
            self.template = template
        else:
            self.template = '''Answer the following questions as best you can. You have access to the following tools:

                                {tools}

                                Use the following format:

                                Question: the input question you must answer
                                Thought: you should always think about what to do
                                Action: the action to take, should be one of [{tool_names}]
                                Action Input: the input to the action
                                Observation: the result of the action
                                ... (this Thought/Action/Action Input/Observation can repeat N times)
                                Thought: I now know the final answer
                                Final Answer: the final answer to the original input question

                                Begin!

                                Question: {input}
                                Thought:{agent_scratchpad}'''
        self.prompt = PromptTemplate.from_template(template)


    def create_KG(
        self,
        graph: Neo4jGraph,
        database
    ):
        """Create knowledge graph via TL output.

        Use externel database to create own Chinese Medicine knowledge graph

        Args: 
        graph: Neo4j graph API by langchain
        database: externel database storing the spcific knowledge

        Result:
        Update the neo4j knowledge graph

        """
        self.graph = graph
        self.database = database

    def load_database(self, database):
        pass
