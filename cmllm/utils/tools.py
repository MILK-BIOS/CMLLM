from pydantic import BaseModel, Field
from langchain.tools import tool
from langchain.chains import GraphCypherQAChain
from typing import Union


@tool
class kg_write(BaseModel):
    name = "KGWriter"
    description = "Use this tool when you need to write or update new knowledge to knowledge graph"

    def run(self, cypher_msg):
        pass



def extract_cypher(text: str) -> str:
    """Extract Cypher code from a text.

    Args:
        text: Text to extract Cypher code from.

    Returns:
        Cypher code extracted from the text.
    """
    # The pattern to find Cypher code enclosed in triple backticks
    pattern = r"```(.*?)```"

    # Find all matches in the input text
    matches = re.findall(pattern, text, re.DOTALL)

    return matches[0] if matches else text