from langchain.tools import BaseTool
from typing import Union


class kg_write_tools(BaseTool):
    name = "Knowledge graph writer"
    description = "Use this tool when you need to write or update new knowledge to knowledge graph"

    def run(self, cypher_msg):
        pass
