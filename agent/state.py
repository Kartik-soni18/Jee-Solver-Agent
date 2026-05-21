"""State definitions for the LangGraph math agent."""

from typing import TypedDict, Annotated, Optional
import operator


class AgentState(TypedDict):
    """Typed dictionary representing the agent's state throughout the graph execution."""

    problem: str
    thought: str
    code: Optional[str]
    code_result: Optional[str]
    answer: Optional[str]
    reasoning_trace: Annotated[list[str], operator.add]
    verification: Optional[str]
    reflection: Optional[str]
    retry_count: int
    final_answer: Optional[str]
    done: bool
