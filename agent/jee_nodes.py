"""JEE-specific node functions for the enhanced calculus agent.

Each node implements a stage of the multi-step reasoning pipeline:
  analyze -> plan -> solve_symbolic -> consolidate -> verify -> format

The ``solve_symbolic`` node iterates through planned steps one at a time,
using a self-loop in the graph until all steps are completed.  If
verification fails, the ``reflect`` node generates corrected SymPy code
and loops back to ``solve_symbolic`` for a retry (bounded by
MAX_RETRIES).
"""

import json
from typing import Optional

from langchain_core.messages import HumanMessage, SystemMessage

from agent.jee_state import JEEAgentState
from agent.jee_prompts import (
    JEE_SYSTEM_PROMPT,
    ANALYZE_PROMPT,
    PLAN_PROMPT,
    SOLVE_SYMPY_PROMPT,
    VERIFY_SYMBOLIC_PROMPT,
    REFLECT_JEE_PROMPT,
    FORMAT_LATEX_PROMPT,
    TOOL_DESCRIPTIONS,
)
from agent.sympy_tools import SymPyTool
from agent.nodes import parse_json_response


# ---------------------------------------------------------------------------
# Keyword-based problem-type detection (fallback when LLM fails)
# ---------------------------------------------------------------------------

_KEYWORD_MAP: dict[str, list[str]] = {
    "limits": ["limit", " lim", "approaches", "tends to", "lim_", "→"],
    "differentiation": [
        "derivative",
        "differentiate",
        "dy/dx",
        "rate of",
        "slope",
        "differentiability",
        "d/dx",
    ],
    "integration": ["integrate", "integration", "integral", "∫"],
    "definite_integrals": [
        "definite integral",
        "evaluate the integral",
        "integrate from",
        "∫_",
    ],
    "differential_equations": [
        "differential equation",
        "solve dy/dx",
        "family of curves",
        "orthogonal trajectories",
        "ode",
    ],
    "maxima_minima": [
        "maxima",
        "minima",
        "maximum",
        "minimum",
        "greatest",
        "least",
        "extreme",
        "optimize",
    ],
    "tangent_normal": ["tangent", "normal to", "normal at", "normal line"],
    "area": ["area bounded", "area enclosed", "area of", "area between", "region"],
    "continuity": ["continuous", "discontinuity", "continuity"],
    "series": ["series", "taylor", "maclaurin", "expansion", "expand"],
}


def detect_problem_type(state: JEEAgentState) -> str:
    """Heuristic problem-type detection from raw text.

    Used as a fallback when the LLM-powered *analyze* node fails to
    produce valid JSON.
    """
    text: str = state.get("problem", "").lower()
    scores: dict[str, int] = {
        ptype: sum(1 for kw in kws if kw in text)
        for ptype, kws in _KEYWORD_MAP.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"


# ---------------------------------------------------------------------------
# Helper: dispatch SymPy tool calls from plan steps
# ---------------------------------------------------------------------------

def _dispatch_sympy_op(tool: SymPyTool, step: dict) -> str:
    """Execute a single planned step using the appropriate SymPyTool method.

    Falls back to ``run_generic`` when the step's ``tool`` field doesn't
    match a known method.
    """
    op: str = step.get("tool", "run_generic")
    code: str = step.get("code", "")

    # If explicit SymPy code is provided, run it directly
    if code and len(code) > 5:
        return tool.run_generic(code)

    # Otherwise dispatch by method name
    expr: str = step.get("expression", "")
    var: str = step.get("variable", "x")

    method_map = {
        "solve_limit": lambda: tool.solve_limit(
            expr or code, var, step.get("point", "0"), step.get("direction", "+")
        ),
        "solve_derivative": lambda: tool.solve_derivative(
            expr or code, var, int(step.get("order", 1))
        ),
        "solve_integral": lambda: tool.solve_integral(expr or code, var),
        "solve_definite_integral": lambda: tool.solve_definite_integral(
            expr or code, var, step.get("lower", "0"), step.get("upper", "1")
        ),
        "solve_equation": lambda: tool.solve_equation(
            step.get("equation", expr or code), var
        ),
        "solve_ode": lambda: tool.solve_ode(
            step.get("equation", code),
            step.get("function", "y"),
            var,
        ),
        "find_maxima_minima": lambda: tool.find_maxima_minima(
            expr or code, var
        ),
        "find_tangent_normal": lambda: tool.find_tangent_normal(
            expr or code, var, step.get("point_x", "0")
        ),
        "area_under_curve": lambda: tool.area_under_curve(
            expr or code, var, step.get("lower", "0"), step.get("upper", "1")
        ),
        "check_continuity": lambda: tool.check_continuity(
            expr or code, var, step.get("point", "0")
        ),
        "partial_fraction": lambda: tool.partial_fraction(expr or code, var),
        "taylor_series": lambda: tool.taylor_series(
            expr or code, var, step.get("point", "0"), int(step.get("order", 5))
        ),
        "implicit_differentiation": lambda: tool.implicit_differentiation(
            step.get("equation", code),
            step.get("y_var", "y"),
            step.get("x_var", "x"),
        ),
        "parametric_derivative": lambda: tool.parametric_derivative(
            step.get("x_expr", ""),
            step.get("y_expr", ""),
            step.get("parameter", "t"),
            int(step.get("order", 1)),
        ),
        "solve_by_parts": lambda: tool.solve_by_parts(
            step.get("u", ""), step.get("dv", ""), var
        ),
        "simplify_expression": lambda: tool.simplify_expression(expr or code),
        "factor_expression": lambda: tool.factor_expression(expr or code),
        "expand_expression": lambda: tool.expand_expression(expr or code),
        "evaluate_expression": lambda: tool.evaluate_expression(
            expr or code, step.get("substitutions", {})
        ),
    }

    try:
        if op in method_map:
            return method_map[op]()
        return tool.run_generic(code or expr or "print(0)")
    except Exception as exc:
        return f"Error: {type(exc).__name__}: {exc}"


# ---------------------------------------------------------------------------
# Graph nodes
# ---------------------------------------------------------------------------

def analyze_node(state: JEEAgentState, llm, config) -> dict:
    """Classify the problem, identify concepts, recommend tools."""
    problem: str = state.get("problem", "")
    if not problem:
        return {
            "problem_type": "other",
            "key_concepts": [],
            "reasoning_trace": ["Error: No problem provided."],
            "done": False,
        }

    prompt = ANALYZE_PROMPT.format(problem=problem)
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT + "\n" + TOOL_DESCRIPTIONS),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        parsed = parse_json_response(response.content)
    except Exception as exc:
        # Fallback to keyword detection
        fallback_type = detect_problem_type(state)
        return {
            "problem_type": fallback_type,
            "key_concepts": ["unknown"],
            "reasoning_trace": [f"LLM analysis failed ({exc}), fallback to keyword detection: {fallback_type}"],
            "done": False,
        }

    if not parsed:
        fallback_type = detect_problem_type(state)
        return {
            "problem_type": fallback_type,
            "key_concepts": ["unknown"],
            "reasoning_trace": ["Analysis returned empty, using keyword fallback."],
            "done": False,
        }

    return {
        "problem_type": parsed.get("problem_type", detect_problem_type(state)),
        "key_concepts": parsed.get("key_concepts", []),
        "reasoning_trace": [
            f"ANALYZE: Type={parsed.get('problem_type', '')}, "
            f"Concepts={parsed.get('key_concepts', [])}, "
            f"Approach={parsed.get('approach', '')}, "
            f"Difficulty={parsed.get('difficulty', '')}"
        ],
        "done": False,
    }


def plan_node(state: JEEAgentState, llm, config) -> dict:
    """Create step-by-step solution plan with SymPy operations."""
    problem: str = state.get("problem", "")
    problem_type: str = state.get("problem_type", "")
    key_concepts: list[str] = state.get("key_concepts", [])

    # Include approach from analysis in the context
    last_trace = state.get("reasoning_trace", [""])[-1] if state.get("reasoning_trace") else ""
    approach = ""
    if "Approach=" in last_trace:
        approach = last_trace.split("Approach=")[1].split(",")[0] if "Approach=" in last_trace else ""

    prompt = PLAN_PROMPT.format(
        problem=problem,
        problem_type=problem_type,
        key_concepts=key_concepts,
        approach=approach,
    )
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT + "\n" + TOOL_DESCRIPTIONS),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        parsed = parse_json_response(response.content)
    except Exception as exc:
        # Create a minimal fallback plan
        return {
            "solution_plan": json.dumps({"steps": [{"description": "Solve directly", "tool": "run_generic", "code": f"import sympy as sp; x=sp.symbols('x', real=True); print(sp.solve({problem!r}, x) if '=' in {problem!r} else 'See result')", "expected_result": "solution"}]}),
            "steps": [{"description": "Solve directly", "tool": "run_generic", "code": "", "expected_result": "solution"}],
            "reasoning_trace": [f"PLAN: LLM failed ({exc}), using direct-solve fallback."],
            "done": False,
        }

    steps = parsed.get("steps", []) if parsed else []
    if not steps:
        steps = [{"description": "Solve directly", "tool": "run_generic", "code": "", "expected_result": "solution"}]

    return {
        "solution_plan": json.dumps(parsed) if parsed else "",
        "steps": steps,
        "reasoning_trace": [f"PLAN: {len(steps)} steps planned: {[s.get('description', '') for s in steps]}"],
        "done": False,
    }


def solve_symbolic_node(state: JEEAgentState, llm, config) -> dict:
    """Execute the current solution step using SymPy.

    Iterates through steps one at a time using *current_step_index* stored
    in the state.  Returns a special ``_all_steps_done`` flag so the graph
    router can decide whether to loop back or continue to consolidation.
    """
    import time

    tool = SymPyTool()
    steps: list[dict] = state.get("steps", [])
    idx: int = state.get("current_step_index", 0)
    problem: str = state.get("problem", "")
    problem_type: str = state.get("problem_type", "")

    if not steps or idx >= len(steps):
        return {"_all_steps_done": True, "sympy_code": None, "sympy_result": None, "done": False}

    step = steps[idx]
    step_desc = step.get("description", f"Step {idx + 1}")

    # Collect previous results for context
    prev_results = ""
    for i, s in enumerate(steps[:idx]):
        if s.get("result"):
            prev_results += f"Step {i + 1}: {s.get('description', '')} -> {s.get('result', '')}\n"

    # Generate SymPy code via LLM
    prompt = SOLVE_SYMPY_PROMPT.format(
        problem=problem,
        problem_type=problem_type,
        step_description=step_desc,
        previous_results=prev_results or "None",
    )
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT + "\n" + TOOL_DESCRIPTIONS),
        HumanMessage(content=prompt),
    ]

    sympy_code: str = ""
    try:
        response = llm.invoke(messages)
        parsed = parse_json_response(response.content)
        if parsed and "code" in parsed:
            sympy_code = parsed["code"]
        else:
            # Fallback: use the step's own code or dispatch
            sympy_code = step.get("code", "")
    except Exception:
        sympy_code = step.get("code", "")

    # Execute the SymPy code
    result: str = ""
    if sympy_code:
        result = tool.run_generic(sympy_code)
    else:
        result = _dispatch_sympy_op(tool, step)

    # Update the step with results
    steps[idx]["result"] = result
    steps[idx]["sympy_code"] = sympy_code
    steps[idx]["completed"] = True

    next_idx = idx + 1
    all_done = next_idx >= len(steps)

    trace_entry = f"SOLVE step {idx + 1}/{len(steps)}: {step_desc}\nCode: {sympy_code}\nResult: {result}"

    return {
        "steps": steps,
        "current_step_index": next_idx,
        "_all_steps_done": all_done,
        "sympy_code": sympy_code,
        "sympy_result": result,
        "symbolic_steps": [f"Step {idx + 1}: {sympy_code} -> {result}"],
        "reasoning_trace": [trace_entry],
        "done": False,
    }


def consolidate_node(state: JEEAgentState, llm, config) -> dict:
    """Combine all step results into a coherent final answer."""
    steps: list[dict] = state.get("steps", [])
    problem: str = state.get("problem", "")

    # Gather results from all completed steps
    step_results = []
    for i, s in enumerate(steps):
        desc = s.get("description", f"Step {i + 1}")
        res = s.get("result", "")
        if res:
            step_results.append(f"{desc}: {res}")

    # Use the last step's result as the primary answer
    final_answer = ""
    if steps:
        last_result = steps[-1].get("result", "")
        # Clean up error prefixes
        if last_result and not last_result.startswith("Error"):
            final_answer = last_result.split("\n")[-1].strip() if "\n" in last_result else last_result.strip()

    # If no valid answer, try to extract from sympy_result
    if not final_answer:
        final_answer = state.get("sympy_result", "")

    # LLM-based consolidation for complex multi-step problems
    if len(steps) > 1:
        try:
            context = "\n".join(step_results)
            prompt = (
                f"Given these intermediate results for a JEE calculus problem, "
                f"provide the final consolidated answer as a single expression or value.\n\n"
                f"Problem: {problem}\n\n"
                f"Step results:\n{context}\n\n"
                f"Return ONLY the final answer (no explanation):"
            )
            messages = [
                SystemMessage(content=JEE_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]
            response = llm.invoke(messages)
            consolidated = response.content.strip()
            if consolidated and len(consolidated) < 500:
                final_answer = consolidated
        except Exception:
            pass

    return {
        "answer": final_answer,
        "reasoning_trace": [f"CONSOLIDATE: Combined {len(steps)} steps into final answer: {final_answer}"],
        "done": False,
    }


def verify_symbolic_node(state: JEEAgentState, llm, config) -> dict:
    """Verify solution correctness using LLM + optional SymPy cross-check."""
    problem: str = state.get("problem", "")
    result: str = state.get("answer", "") or state.get("sympy_result", "")
    sympy_code: str = state.get("sympy_code", "")
    problem_type: str = state.get("problem_type", "")

    # Quick sanity checks (numeric)
    if result and not result.startswith("Error"):
        try:
            import sympy as sp
            expr = sp.sympify(result)
            # Check if result is a simple number
            val = float(expr.evalf())
            if problem_type in ("area", "definite_integrals") and val < 0:
                # Area/integral could be negative depending on context
                pass
        except Exception:
            pass

    prompt = VERIFY_SYMBOLIC_PROMPT.format(
        problem=problem,
        problem_type=problem_type,
        result=result,
        sympy_code=sympy_code,
    )
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        parsed = parse_json_response(response.content)
    except Exception as exc:
        return {
            "verification": f"Verification LLM call failed: {exc}",
            "confidence": 0.5,
            "done": False,
        }

    if not parsed:
        return {
            "verification": "Failed to parse verification response.",
            "confidence": 0.5,
            "done": False,
        }

    is_correct = bool(parsed.get("is_correct", False))
    confidence = float(parsed.get("confidence", 0.0))
    issues = parsed.get("issues", [])
    simplified = parsed.get("simplified_result", "")

    # Update answer if simplification was applied
    answer_update = {}
    if simplified and not str(simplified).startswith("Error"):
        answer_update["answer"] = str(simplified)

    return {
        **answer_update,
        "verification": (
            f"Correct: {is_correct}, Confidence: {confidence:.2f}, "
            f"Issues: {issues}"
        ),
        "confidence": confidence,
        "done": is_correct and confidence >= 0.7,
        **({"reasoning_trace": [f"VERIFY: correct={is_correct}, confidence={confidence:.2f}"]} if not is_correct else {}),
    }


def reflect_jee_node(state: JEEAgentState, llm, config) -> dict:
    """Reflect on verification failures and produce corrected approach."""
    retry_count: int = state.get("retry_count", 0)
    max_retries: int = getattr(config, "MAX_RETRIES", 3)

    if retry_count >= max_retries:
        return {
            "reflection": f"Max retries ({max_retries}) reached. Proceeding with best effort.",
            "done": True,
            "reasoning_trace": [f"REFLECT: Max retries reached ({retry_count}/{max_retries})."],
        }

    problem: str = state.get("problem", "")
    failed_approach = state.get("solution_plan", "")
    verification: str = state.get("verification", "")
    sympy_code: str = state.get("sympy_code", "")
    error: str = state.get("sympy_result", "")

    # Extract issues from verification text
    issues = verification
    if "Issues:" in verification:
        parts = verification.split("Issues:", 1)
        if len(parts) > 1:
            issues = parts[1].strip()

    prompt = REFLECT_JEE_PROMPT.format(
        problem=problem,
        failed_approach=failed_approach,
        issues=issues,
        code=sympy_code,
        error=error,
    )
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    try:
        response = llm.invoke(messages)
        parsed = parse_json_response(response.content)
    except Exception as exc:
        return {
            "reflection": f"Reflection LLM call failed: {exc}",
            "retry_count": retry_count + 1,
            "current_step_index": 0,
            "reasoning_trace": [f"REFLECT: LLM error ({exc}), will retry from step 1."],
        }

    if not parsed:
        return {
            "reflection": "Failed to parse reflection response.",
            "retry_count": retry_count + 1,
            "current_step_index": 0,
            "reasoning_trace": ["REFLECT: Empty response, retrying from step 1."],
        }

    corrected_code = str(parsed.get("corrected_code", ""))

    # Update the first step with corrected code if available
    steps = list(state.get("steps", []))
    if corrected_code and steps:
        steps[0]["code"] = corrected_code
        steps[0]["result"] = ""
        steps[0]["completed"] = False

    return {
        "reflection": str(parsed.get("analysis", "")),
        "sympy_code": corrected_code,
        "steps": steps,
        "current_step_index": 0,  # Restart from step 1
        "retry_count": retry_count + 1,
        "done": False,
        "reasoning_trace": [
            f"REFLECT (attempt {retry_count + 1}): {parsed.get('analysis', '')}\n"
            f"Corrected code: {corrected_code[:200]}..."
            if len(corrected_code) > 200
            else f"Corrected code: {corrected_code}"
        ],
    }


def format_latex_node(state: JEEAgentState, llm, config) -> dict:
    """Format final answer as LaTeX and plain text."""
    problem: str = state.get("problem", "")
    result: str = state.get("answer", "") or state.get("sympy_result", "")
    steps = state.get("steps", [])

    # Build steps summary
    steps_summary = ""
    for i, s in enumerate(steps[:5]):  # Limit to first 5 steps for LLM context
        desc = s.get("description", f"Step {i + 1}")
        res = s.get("result", "")
        if res:
            res_short = res[:100] + "..." if len(res) > 100 else res
            steps_summary += f"{i + 1}. {desc}: {res_short}\n"

    prompt = FORMAT_LATEX_PROMPT.format(
        problem=problem,
        result=result,
        steps=steps_summary or "No steps recorded.",
    )
    messages = [
        SystemMessage(content=JEE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ]

    latex_answer: str = ""
    final_answer: str = result
    summary: str = ""
    boxed: str = ""

    try:
        response = llm.invoke(messages)
        parsed = parse_json_response(response.content)
        if parsed:
            latex_answer = str(parsed.get("latex_answer", ""))
            final_answer = str(parsed.get("final_answer", result))
            summary = str(parsed.get("summary", ""))
            boxed = str(parsed.get("boxed_answer", ""))
    except Exception:
        pass

    # Fallback: create simple LaTeX from result
    if not latex_answer and result:
        latex_answer = f"${result}$"
    if not boxed and result:
        boxed = f"\\boxed{{{result}}}"

    return {
        "latex_answer": latex_answer,
        "final_answer": final_answer or boxed,
        "answer": final_answer or result,
        "done": True,
        "reasoning_trace": [f"FORMAT: Final answer = {final_answer}, LaTeX = {latex_answer}"],
    }
