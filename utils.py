from pydantic import BaseModel
from typing import Annotated
from langgraph.graph import add_messages

class State(BaseModel):
    messages:       Annotated[list, add_messages] = []
    response:       str = ""
    safety:         str = "unknown"
    unsafe_retries: int = 0
