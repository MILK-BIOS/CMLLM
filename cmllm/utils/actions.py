from pydantic import BaseModel, Field, ValidationError
from typing import Dict, Optional, Any


class Prompt(BaseModel):
    query: str = Field(description="Message to pass to the agent")
    tags: Optional[Dict[str, Any]] = Field(description="Additional parameters for the prompt")

class Action(BaseModel):
    """Structural tools class"""
    name: str = Field(description="Tool's name")
    prompt: Optional[Prompt] = Field(description="The necessary message pass to agent")

class Choice(BaseModel):
    """Structural tools class"""
    name: str = Field(description="Tool's name")
    score: int = Field(description="The score passed to tool")

