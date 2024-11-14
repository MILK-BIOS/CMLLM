import langchain
import json
import typing
from pydantic import Field
from langchain_core.prompts import PromptTemplate
from langchain_community.graphs.graph_store import GraphStore
from langchain_community.graphs import Neo4jGraph

from langchain_community.chains.graph_qa.cypher_utils import (
    CypherQueryCorrector,
    Schema,
)


class KnowledgeRetriever:
    def __init__(
        self,
        llm,
        graph: GraphStore=Field(exclude=True),
        template: str=None,
        validate_cypher: bool=False
        ):
        self.llm = llm
        if graph:
            self.graph = graph
            self.graph_schema = graph.get_structured_schema
        if template:
            self.template = template
        else:
            self.template = """
                You are an expert of Neo4jGraph. Write a query based on the given graph Schema and problem.
                schema as follows：
                ---
                {schema}
                ---
                question as follows：
                ---
                {question}
                ---
                Now please write the query：
            """
        if validate_cypher:
            corrector_schema = [
                Schema(el["start"], el["type"], el["end"])
                for el in kwargs["graph"].structured_schema.get("relationships")
            ]
            cypher_query_corrector = CypherQueryCorrector(corrector_schema)

    def run(self):
        pass
