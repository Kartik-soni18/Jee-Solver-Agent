"""JEE-specific prompt templates for calculus problem solving.

Each prompt is a string template with named ``{placeholders}`` that the
calling node fills at runtime via ``str.format(...)`` or an f-string
wrapper.

Usage::

    from agent.jee_prompts import ANALYZE_PROMPT, TOOL_DESCRIPTIONS

    filled = ANALYZE_PROMPT.format(problem="Evaluate lim x->0 sin(x)/x")
"""

# ---------------------------------------------------------------------------
# System prompt — injected into the LLM's system message
# ---------------------------------------------------------------------------

JEE_SYSTEM_PROMPT: str = (
    """You are an expert JEE Advanced mathematics instructor with deep expertise in calculus. You solve problems using rigorous mathematical reasoning and SymPy symbolic computation. You follow a structured approach: analyze → plan → solve step-by-step using SymPy → verify.

Core principles:
1. Always reason from first principles — definitions, theorems, and standard results.
2. Identify the problem type (limits, differentiation, integration, differential equations, applications) before solving.
3. Use SymPy for all symbolic computation — differentiation, integration, limits, solving equations, series expansions.
4. Check edge cases and domain restrictions (e.g., division by zero, negative logs, absolute values).
5. Verify results using alternative methods when possible (e.g., numerical check, graphical check, known standard results).
6. Format final answers using proper mathematical notation.
7. Classify difficulty as easy, medium, or hard based on concepts involved and expected solution time.

JEE Advanced topics you master:
- Limits: L'Hopital's rule, standard limits, Sandwich theorem, expansion methods
- Differentiation: Chain rule, implicit, parametric, higher-order derivatives, logarithmic differentiation
- Integration: Substitution, by parts, partial fractions, definite integral properties, reduction formulas
- Differential Equations: Variable separable, homogeneous, linear first-order, exact equations
- Applications: Maxima/minima, rate of change, tangent/normal, Rolle's & LMVT
- Definite Integrals: Properties, Leibniz rule, areas, Beta/Gamma functions"""
)

# ---------------------------------------------------------------------------
# Analysis prompt — classifies the problem
# ---------------------------------------------------------------------------

ANALYZE_PROMPT: str = (
    """Analyze this JEE mathematics problem and classify it.

**Problem:** {problem}

1. Identify the problem type: limits, differentiation, integration, definite_integrals, differential_equations, maxima_minima, tangent_normal, area, continuity, series, or other.
2. List the key mathematical concepts and theorems involved.
3. Identify the best approach (substitution, by parts, L'Hopital, etc.).
4. Note any tricky aspects or common mistakes.
5. Classify the difficulty level.

Return ONLY valid JSON with keys:
- "problem_type": string classification
- "key_concepts": list of concept strings
- "approach": string describing the approach
- "tricky_aspects": list of strings
- "difficulty": "easy", "medium", or "hard"
- "recommended_tools": list of tool names from the available tools list"""
)

# ---------------------------------------------------------------------------
# Planning prompt — generates step-by-step solution plan
# ---------------------------------------------------------------------------

PLAN_PROMPT: str = (
    """Create a detailed step-by-step solution plan for this JEE problem.

**Problem:** {problem}
**Type:** {problem_type}
**Key Concepts:** {key_concepts}
**Approach:** {approach}

Create a plan where each step specifies:
1. What to compute in this step
2. Which SymPy tool/method to use
3. What the expected intermediate result should be

Return ONLY valid JSON with key:
- "steps": list of dicts, each with:
  - "description": what to do in this step (human-readable)
  - "tool": which SymPy function to call (e.g., "solve_limit", "solve_integral")
  - "code": SymPy Python code to execute for this step
  - "expected_result": what to expect from this step
  - "verification": how to verify this step's result"""
)

# ---------------------------------------------------------------------------
# SymPy code generation prompt — produces code for the current step
# ---------------------------------------------------------------------------

SOLVE_SYMPY_PROMPT: str = (
    """Generate SymPy Python code to solve the current step of this JEE problem.

**Problem:** {problem}
**Type:** {problem_type}
**Current Step:** {step_description}
**Previous Results:** {previous_results}

Write clean SymPy code that:
1. Uses proper SymPy symbols (x, y, etc.) with `sp.symbols(..., real=True)`
2. Performs the exact computation needed for this step
3. Prints the result using `print(result)`
4. Uses `sp.simplify()` for clean output when appropriate
5. Includes a comment explaining the key operation

Return ONLY valid JSON with:
- "code": the SymPy Python code string (no markdown fences)
- "explanation": brief explanation of what the code does"""
)

# ---------------------------------------------------------------------------
# Verification prompt — checks symbolic results
# ---------------------------------------------------------------------------

VERIFY_SYMBOLIC_PROMPT: str = (
    """Verify the solution to this JEE problem using the computed result.

**Problem:** {problem}
**Expected Answer Type:** {problem_type}
**Computed Result:** {result}
**SymPy Code Used:** {sympy_code}

Verify:
1. Does the result make mathematical sense? (e.g., correct sign, expected magnitude)
2. Are the units/dimensions correct? (e.g., area should be positive, definite integral bounds respected)
3. Can we cross-check with an alternative method? (e.g., numerical evaluation, different technique)
4. Is the answer in the expected form for JEE? (simplified, no unnecessary complexity)
5. Check edge cases: does the solution hold at boundary points?

Return ONLY valid JSON with:
- "is_correct": boolean
- "confidence": float in [0, 1]
- "issues": list of any issues found (empty if none)
- "alternative_check": string describing cross-verification method performed
- "simplified_result": the result in simplest form if simplification was applied"""
)

# ---------------------------------------------------------------------------
# Reflection prompt — triggered when verification fails
# ---------------------------------------------------------------------------

REFLECT_JEE_PROMPT: str = (
    """The solution attempt had issues. Analyze and correct.

**Problem:** {problem}
**Failed Approach:** {failed_approach}
**Issues:** {issues}
**SymPy Code Used:** {code}
**Error/Result:** {error}

1. Why did the approach fail? Identify the root cause.
2. What mathematical concept or technique was missed?
3. Provide a corrected SymPy code approach.
4. Suggest an alternative method if applicable.

Common failure modes in JEE problems:
- Forgetting to check domain restrictions (e.g., log argument must be positive)
- Missing the absolute value in area calculations
- Not considering all critical points in maxima/minima
- Forgetting to add +C for indefinite integrals
- Wrong substitution choice in integration
- Not checking both left and right limits for piecewise functions
- Algebraic simplification errors before applying calculus

Return ONLY valid JSON with:
- "analysis": string explaining what went wrong
- "corrected_code": string with corrected SymPy code (no markdown fences)
- "alternative_approach": string with different method if applicable
- "confidence": float in [0, 1] for the corrected approach"""
)

# ---------------------------------------------------------------------------
# Formatting prompt — produces LaTeX + plain-text final answer
# ---------------------------------------------------------------------------

FORMAT_LATEX_PROMPT: str = (
    """Format the final answer for a JEE problem.

**Problem:** {problem}
**Result:** {result}
**Steps:** {steps}

1. Format the final answer as clean LaTeX suitable for rendering.
2. Provide a concise final answer string.
3. Summarize the key steps briefly (2-3 lines max).
4. Ensure the LaTeX uses proper math delimiters ($...$ or $$...$$).

Return ONLY valid JSON with:
- "latex_answer": LaTeX string for the final answer
- "final_answer": plain text final answer
- "summary": brief solution summary (2-3 sentences)
- "boxed_answer": answer in \\boxed{{}} notation for JEE style"""
)

# ---------------------------------------------------------------------------
# Tool descriptions — injected into LLM context so it knows what it can call
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS: str = """
Available SymPy tools:
- solve_limit(expression, variable, point, direction='+') — Compute limits (including one-sided)
- solve_derivative(expression, variable, order=1) — Differentiation of any order
- solve_integral(expression, variable) — Indefinite integration (+ C included)
- solve_definite_integral(expression, variable, lower, upper) — Definite integration
- solve_equation(equation, variable) — Solve equations (linear, polynomial, transcendental)
- solve_system(equations, variables) — Solve system of equations
- solve_ode(equation, function, variable) — Differential equations
- find_maxima_minima(expression, variable) — Critical points with classification
- find_tangent_normal(curve, variable, point_x) — Tangent and normal lines
- area_under_curve(curve, variable, lower, upper) — Area calculation (handles sign changes)
- check_continuity(expression, variable, point) — Continuity check with limit analysis
- check_differentiability(expression, variable, point) — Differentiability check
- partial_fraction(expression, variable) — Partial fraction decomposition
- taylor_series(expression, variable, point, order) — Taylor/Maclaurin series expansion
- series_expansion_limit(expression, variable, point, order) — Evaluate limit via series
- lhospital_limit(expression, variable, point) — L'Hopital's rule for indeterminate forms
- implicit_differentiation(equation, y_var, x_var) — Implicit differentiation dy/dx
- parametric_derivative(x_expr, y_expr, parameter, order) — Parametric derivatives
- solve_by_parts(u, dv, variable) — Integration by parts with full working
- substitution_method(expression, variable, substitution, new_var) — Integration by substitution
- simplify_expression(expression) — Simplify mathematical expressions
- factor_expression(expression) — Factor polynomials and rational expressions
- expand_expression(expression) — Expand products and powers
- evaluate_expression(expression, substitutions) — Numerical evaluation
- run_generic(code) — Execute arbitrary SymPy code (escape hatch)
"""
