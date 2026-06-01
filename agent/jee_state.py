"""Extended state definitions for JEE calculus problem-solving agent."""

from typing import TypedDict, Annotated, Optional
import operator


class JEEAgentState(TypedDict):

    # -- Core problem fields ------------------------------------------------
    problem: str
    problem_type: Optional[str]
    key_concepts: list[str]
    solution_plan: Optional[str]

    # -- Current working state ----------------------------------------------
    thought: str
    code: Optional[str]
    sympy_code: Optional[str]
    sympy_result: Optional[str]
    code_result: Optional[str]
    answer: Optional[str]

    # -- Accumulated steps --------------------------------------------------
    # NOTE: steps and reasoning_trace use plain list (not Annotated with
    # operator.add) because nodes mutate them in-place and return the same
    # list object. Using operator.add would cause unbounded duplication on
    # every graph self-loop, leading to exponential memory growth.
    steps: list[dict]
    symbolic_steps: list[str]
    reasoning_trace: list[str]

    # -- Step tracking ------------------------------------------------------
    current_step_index: int
    _all_steps_done: bool
    _solve_loop_count: int

    # -- Verification / reflection ------------------------------------------
    verification: Optional[str]
    reflection: Optional[str]
    retry_count: int

    # -- Image input --------------------------------------------------------
    image_data: Optional[str]
    extracted_latex: Optional[str]

    # -- Final output -------------------------------------------------------
    final_answer: Optional[str]
    latex_answer: Optional[str]
    confidence: float
    done: bool
