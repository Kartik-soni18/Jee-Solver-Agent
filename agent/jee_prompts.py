"""Fast/short prompt templates for JEE problem solving.

These prompts are designed to minimize token usage and LLM latency by:
- Removing verbose instructions
- Combining steps where possible
- Using compact JSON schemas
"""

# ---------------------------------------------------------------------------
# Compact system prompt
# ---------------------------------------------------------------------------

JEE_SYSTEM_PROMPT_FAST: str = (
    "You are a JEE Math expert. Solve using reasoning + SymPy. "
    "Use sp.symbols('x', real=True). Use sp.Eq() for equations. "
    "Print final result. "
    "Topics: limits, derivatives, integrals, DEs, complex, matrices, probability, vectors, geometry. "
    "CRITICAL: Before brute-force computation, check for contradictions or impossibilities. "
    "If no valid solutions exist, the sum over an empty set is 0. "
    "For involutions (f(f(x))=x), remember f'(f(x))*f'(x)=1. "
    "For fixed points (f(x)=x), check derivative constraints."
)

# ---------------------------------------------------------------------------
# Combined analyze + plan prompt
# ---------------------------------------------------------------------------

ANALYZE_PLAN_PROMPT: str = (
    """Solve this JEE problem. Analyze and plan, then provide SymPy code.

**Problem:** {problem}

**Relevant Context:**
{retrieved_context}

CRITICAL RULES:
1. DO NOT write "import sympy" — it is already imported as "sp".
2. First check for mathematical contradictions or impossibilities (e.g., involution f(f(x))=x implies f'(f(x))*f'(x)=1; fixed points have derivative constraints). If impossible, answer is 0 (empty sum).
3. Use SymPy to VERIFY your reasoning, not replace it.
4. Print the final result with print(result).

Return JSON with:
- "problem_type": one of [limits,differentiation,integration,definite_integrals,differential_equations,maxima_minima,complex_numbers,quadratic,permutations_combinations,binomial_theorem,matrices_determinants,probability,sequences_series,trigonometric_identities,trigonometric_equations,inverse_trig,straight_lines,circles,parabola,ellipse,hyperbola,vectors,three_d_geometry,statistics,mathematical_reasoning]
- "key_concepts": list of key concepts
- "approach": brief strategy
- "difficulty": easy/medium/hard
- "sympy_code": complete SymPy code (NO import lines) to solve in ONE go
- "expected_answer": expected final answer"""
)

# ---------------------------------------------------------------------------
# Direct solve prompt (skip analyze/plan, go straight to code)
# ---------------------------------------------------------------------------

DIRECT_SOLVE_PROMPT: str = (
    """Generate SymPy Python code to solve this JEE problem.

**Problem:** {problem}

**Relevant Context:**
{retrieved_context}

CRITICAL RULES:
1. DO NOT write "import sympy" or "import sympy as sp" — sympy is already imported as "sp"
2. Define ALL symbols with sp.symbols('...', real=True)
3. Use sp.Eq() for equations
4. Print the final result with print(result)
5. Before coding, check for contradictions/impossibilities (e.g., involution f(f(x))=x implies f'(f(x))*f'(x)=1). If no solutions exist, answer is 0.
6. Use SymPy to VERIFY reasoning, not replace it.

Return JSON with:
- "problem_type": classification
- "sympy_code": complete runnable code (NO import lines)
- "explanation": 1-line explanation"""
)

# ---------------------------------------------------------------------------
# Compact verification prompt
# ---------------------------------------------------------------------------

VERIFY_FAST_PROMPT: str = (
    """Verify: Problem: {problem} | Result: {result} | Type: {problem_type}

Checks: sanity, domain, cross-check, form.

Return JSON with keys:
- "is_correct": boolean
- "confidence": float 0-1
- "issues": list of strings
- "simplified_result": string or empty"""
)

# ---------------------------------------------------------------------------
# Compact format prompt
# ---------------------------------------------------------------------------

FORMAT_FAST_PROMPT: str = (
    """Format this math result as LaTeX: {result}

Return JSON with keys:
- "latex_answer": LaTeX string
- "final_answer": plain text
- "boxed_answer": boxed LaTeX"""
)

# ---------------------------------------------------------------------------
# Tool descriptions (compact)
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS_FAST: str = """
SymPy tools: solve_limit, solve_derivative, solve_integral, solve_definite_integral, solve_equation, solve_system, solve_ode, find_maxima_minima, find_tangent_normal, area_under_curve, check_continuity, taylor_series, implicit_differentiation, parametric_derivative, solve_by_parts, simplify_expression, factor_expression, expand_expression, evaluate_expression, run_generic
"""
