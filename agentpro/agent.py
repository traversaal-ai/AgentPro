from typing import Optional, List, Any
from pydantic import BaseModel, Field
import json

# Define the structure of an Action
class Action(BaseModel):
    action_type: str  # Dynamic action type based on tool (e.g., "search", "calculate", "generate_slides")
    input: Any = Field(default_factory=lambda: None)  # Flexible input: string, list, or dict

    def get_input(self) -> Any:
        """
        Safely returns the action input (str, list, dict).
        """
        return self.input

# Define the structure of an Observation
class Observation(BaseModel):
    result: Any  # Stores the result after running an action

# Define the structure of a single thought step
class ThoughtStep(BaseModel):
    thought: Optional[str] = None  # Agent's reasoning at this step
    action: Optional[Action] = None  # Action taken (optional)
    observation: Optional[Observation] = None  # Result observed after action
    pause_reflection: Optional[str] = None  # Optional reflection if agent paused

# Define the full agent response
class AgentResponse(BaseModel):
    tool_calls: Optional[List[str]]
    full_thoughts: Optional[List[str]]
    thought_process: List[ThoughtStep]  # Steps including thoughts, actions, and observations
    final_answer: Optional[str] = None  # Final answer after reasoning
