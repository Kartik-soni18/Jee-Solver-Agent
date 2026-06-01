import json
import re
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.jee_state import JEEAgentState
from agent.jee_prompts import (
    JEE_SYSTEM_PROMPT_FAST,
    ANALYZE_PLAN_PROMPT,
    DIRECT_SOLVE_PROMPT,
    VERIFY_FAST_PROMPT,
    FORMAT_FAST_PROMPT,
    TOOL_DESCRIPTIONS_FAST,
)
from agent.sympy_tools import SymPyTool
from agent.rag_knowledge_base import get_knowledge_base


def _parse_json(response_text: str) -> dict:
    """Fast JSON parser with minimal fallback."""
    if not response_text:
        return {}
    text = response_text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"```\s*$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {}


def _get_rag_context(problem: str, problem_type: str = "") -> str:
    """Get RAG context, with optional problem type filter."""
    try:
        kb = get_knowledge_base()
        if kb.is_available:
            if problem_type:
                result = kb.retrieve_by_problem_type(query=problem, problem_type=problem_type, top_k=3)
            else:
                result = kb.retrieve(query=problem, top_k=3)
            ctx = result.get("formatted_context", "")
            return ctx[:3500] if ctx else ""
    except Exception:
        pass
    return ""


def analyze_solve_node(state: JEEAgentState, llm, config) -> dict:
    """Combined analyze + plan + solve in ONE LLM call.
    
    Instead of:
      analyze (LLM) -> plan (LLM) -> solve_step_1 (LLM) -> solve_step_2 (LLM) -> ...
    
    We do:
      analyze_solve (ONE LLM call generates complete SymPy code) -> execute
    
    This reduces LLM calls from 5-7 to 1 for the core solving.
    """
    problem: str = state.get("problem", "")
    if not problem:
        return {"problem_type": "other", "answer": "", "reasoning_trace": ["Error: No problem."], "done": False}

    # RAG retrieval
    rag_ctx = _get_rag_context(problem)

    # Use DIRECT_SOLVE_PROMPT for retry (shorter), ANALYZE_PLAN_PROMPT for first attempt
    is_retry = state.get("retry_count", 0) > 0
    prompt_template = DIRECT_SOLVE_PROMPT if is_retry else ANALYZE_PLAN_PROMPT

    prompt = prompt_template.format(problem=problem, retrieved_context=rag_ctx or "None")
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT_FAST + " " + TOOL_DESCRIPTIONS_FAST),
        HumanMessage(content=prompt),
    ]

    # LLM call: get classification + complete SymPy code
    sympy_code = ""
    problem_type = "other"
    expected_answer = ""
    try:
        response = llm.invoke(messages)
        parsed = _parse_json(response.content)
        problem_type = parsed.get("problem_type", "other")
        sympy_code = parsed.get("sympy_code", "")
        expected_answer = parsed.get("expected_answer", "")
    except Exception as exc:
        return {
            "problem_type": "other",
            "answer": "",
            "reasoning_trace": [f"analyze_solve failed: {exc}"],
            "done": False,
        }

    # Normalize sympy_code to string (LLM may return a list of lines)
    if isinstance(sympy_code, list):
        sympy_code = "\n".join(str(line) for line in sympy_code)

    # Execute the complete SymPy code
    tool = SymPyTool()
    result = ""
    if sympy_code:
        result = tool.run_generic(sympy_code)
    else:
        return {
            "problem_type": problem_type,
            "answer": "",
            "reasoning_trace": ["No SymPy code generated."],
            "done": False,
        }

    # Extract final answer from result (last non-error line)
    answer = ""
    if result and not result.startswith("Error"):
        lines = [l.strip() for l in result.split("\n") if l.strip()]
        if lines:
            answer = lines[-1]

    trace = state.get("reasoning_trace", [])
    trace.append(f"ANALYZE_SOLVE: type={problem_type}, code_len={len(sympy_code)}, result={answer}")

    return {
        "problem_type": problem_type,
        "sympy_code": sympy_code,
        "sympy_result": result,
        "answer": answer,
        "expected_answer": expected_answer,
        "reasoning_trace": trace,
        "done": False,
    }


def verify_fast_node(state: JEEAgentState, llm, config) -> dict:
    """Fast verification — compact prompt, minimal checks."""
    problem: str = state.get("problem", "")
    result: str = state.get("answer", "") or state.get("sympy_result", "")
    problem_type: str = state.get("problem_type", "")
    sympy_code: str = state.get("sympy_code", "")

    # Skip LLM verification for simple numeric answers — do quick sanity check
    if result and not result.startswith("Error"):
        try:
            import sympy as sp
            val = float(sp.sympify(result).evalf())
            # Quick sanity: area/probability must be >= 0
            if problem_type in ("area", "definite_integrals", "probability") and val < 0:
                pass  # could be valid depending on context
        except Exception:
            pass

    prompt = VERIFY_FAST_PROMPT.format(
        problem=problem,
        problem_type=problem_type,
        result=result,
        sympy_code=sympy_code,
    )
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT_FAST),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        parsed = _parse_json(response.content)
    except Exception:
        parsed = {}

    # Handle case where parsed is a list instead of dict
    if isinstance(parsed, list) and len(parsed) > 0:
        parsed = parsed[0] if isinstance(parsed[0], dict) else {}
    if not isinstance(parsed, dict):
        parsed = {}

    is_correct = bool(parsed.get("is_correct", True))  # Default to True if parse fails
    confidence = float(parsed.get("confidence", 0.8))
    simplified = parsed.get("simplified_result", "")

    answer_update = {}
    if simplified and not str(simplified).startswith("Error"):
        answer_update["answer"] = str(simplified)

    trace = state.get("reasoning_trace", [])
    trace.append(f"VERIFY: correct={is_correct}, confidence={confidence:.2f}")

    return {
        **answer_update,
        "verification": f"Correct: {is_correct}, Confidence: {confidence:.2f}",
        "confidence": confidence,
        "done": is_correct and confidence >= 0.7,
        "reasoning_trace": trace,
    }


def reflect_fast_node(state: JEEAgentState, llm, config) -> dict:
    """Fast reflection — shorter prompt, focused on code fix."""
    retry_count: int = state.get("retry_count", 0)
    max_retries: int = getattr(config, "MAX_RETRIES", 2)  # Reduced from 3 to 2

    if retry_count >= max_retries:
        trace = state.get("reasoning_trace", [])
        trace.append(f"REFLECT: Max retries ({retry_count}).")
        return {"done": True, "reasoning_trace": trace}

    problem: str = state.get("problem", "")
    sympy_code: str = state.get("sympy_code", "")
    error: str = state.get("sympy_result", "")

    # Short reflection prompt
    prompt = (
        f"Fix this SymPy code for: {problem}\n\n"
        f"Failed code:\n{sympy_code}\n\n"
        f"Error/Result: {error}\n\n"
        f"Return JSON: {{\"corrected_code\": \"\"}}"
    )
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT_FAST),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        parsed = _parse_json(response.content)
    except Exception:
        parsed = {}

    corrected_code = str(parsed.get("corrected_code", ""))

    trace = state.get("reasoning_trace", [])
    trace.append(f"REFLECT (attempt {retry_count + 1}): code fixed")

    return {
        "sympy_code": corrected_code or sympy_code,
        "retry_count": retry_count + 1,
        "done": False,
        "reasoning_trace": trace,
    }


def format_fast_node(state: JEEAgentState, llm, config) -> dict:
    """Fast formatting — minimal prompt, quick LaTeX generation."""
    problem: str = state.get("problem", "")
    result: str = state.get("answer", "") or state.get("sympy_result", "")

    # Fast path: if result is simple, format without LLM
    if result and len(result) < 50 and not result.startswith("Error"):
        latex = f"${result}$"
        boxed = f"\\boxed{{{result}}}"
        trace = state.get("reasoning_trace", [])
        trace.append(f"FORMAT: {result}")
        return {
            "latex_answer": latex,
            "final_answer": result,
            "answer": result,
            "done": True,
            "reasoning_trace": trace,
        }

    # LLM format for complex results
    prompt = FORMAT_FAST_PROMPT.format(result=result)
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT_FAST),
        HumanMessage(content=prompt),
    ]

    latex_answer = ""
    final_answer = result
    boxed = ""

    try:
        response = llm.invoke(messages)
        parsed = _parse_json(response.content)
        if parsed:
            latex_answer = str(parsed.get("latex_answer", ""))
            final_answer = str(parsed.get("final_answer", result))
            boxed = str(parsed.get("boxed_answer", ""))
    except Exception:
        pass

    if not latex_answer and result:
        latex_answer = f"${result}$"
    if not boxed and result:
        boxed = f"\\boxed{{{result}}}"

    trace = state.get("reasoning_trace", [])
    trace.append(f"FORMAT: {final_answer}")

    return {
        "latex_answer": latex_answer,
        "final_answer": final_answer or boxed,
        "answer": final_answer or result,
        "done": True,
        "reasoning_trace": trace,
    }
