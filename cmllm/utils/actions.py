from pydantic import BaseModel


class Action(BaseModel):
    def __init__(self, tools):
        self.tools = tools