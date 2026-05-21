"""SymPy-based symbolic mathematics toolkit for JEE calculus problems.

This module provides a sandboxed, timeout-protected environment for executing
SymPy symbolic mathematics operations commonly found in JEE Advanced calculus
problems: limits, differentiation, integration, differential equations,
maxima/minima analysis, tangent/normal lines, area calculations, continuity
checks, partial fractions, Taylor series, implicit differentiation, parametric
derivatives, and integration by parts.

All computation runs in a restricted namespace with thread-based timeout
protection to prevent hanging on complex symbolic operations.
"""

import re
import io
import sys
import threading
from typing import Optional, Tuple, List, Dict, Any

# Import sympy at module level so it can be injected into the sandbox
# without relying on __import__ (which is blocked by restricted builtins).
try:
    import sympy as _sympy_module
except ImportError:  # pragma: no cover
    _sympy_module = None  # type: ignore[assignment]


class SymPyToolError(Exception):
    """Raised when SymPy tool execution fails."""
    pass


class SymPyTool:
    """Execute SymPy symbolic mathematics with sandboxing and timeout.

    Provides a comprehensive set of methods for solving JEE-level calculus
    problems using SymPy. Each method generates the appropriate SymPy Python
    code and executes it in a restricted, timeout-protected environment.

    Attributes:
        TIMEOUT: Maximum execution time in seconds (default: 10).
        ALLOWED_MODULES: Set of module names permitted in the sandbox.
    """

    ALLOWED_MODULES: frozenset[str] = frozenset(
        {"sympy", "math", "fractions", "decimal"}
    )
    TIMEOUT: int = 10  # seconds

    # ------------------------------------------------------------------
    # Construction / setup
    # ------------------------------------------------------------------

    def __init__(self) -> None:
        self._setup_sympy()

    def _setup_sympy(self) -> None:
        """Pre-configure SymPy environment with common symbols and settings."""
        import sympy as sp

        # Common symbols used in JEE problems
        self.x, self.y, self.z, self.t = sp.symbols("x y z t", real=True)
        self.a, self.b, self.c, self.n, self.m = sp.symbols(
            "a b c n m", real=True
        )
        self.k = sp.symbols("k", real=True, positive=True)
        self.theta = sp.symbols("theta", real=True)
        self.pi = sp.pi
        self.e = sp.E
        self.inf = sp.oo

        # Operator dictionary for quick access to common SymPy functions
        self.operators: Dict[str, Any] = {
            # Trigonometric
            "sin": sp.sin,
            "cos": sp.cos,
            "tan": sp.tan,
            "cot": sp.cot,
            "sec": sp.sec,
            "csc": sp.csc,
            # Inverse trigonometric
            "asin": sp.asin,
            "acos": sp.acos,
            "atan": sp.atan,
            "acot": sp.acot,
            "asec": sp.asec,
            "acsc": sp.acsc,
            # Hyperbolic
            "sinh": sp.sinh,
            "cosh": sp.cosh,
            "tanh": sp.tanh,
            "coth": sp.coth,
            "sech": sp.sech,
            "csch": sp.csch,
            # Logarithmic / exponential
            "log": sp.log,
            "ln": lambda x: sp.log(x),
            "exp": sp.exp,
            "sqrt": sp.sqrt,
            # Absolute value
            "abs": sp.Abs,
            # Calculus operators
            "diff": sp.diff,
            "integrate": sp.integrate,
            "limit": sp.limit,
            "summation": sp.summation,
            "Derivative": sp.Derivative,
            "Integral": sp.Integral,
            "Limit": sp.Limit,
            "Sum": sp.Sum,
            # Special constants
            "oo": sp.oo,
            "pi": sp.pi,
            "E": sp.E,
            "I": sp.I,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_namespace(self) -> Dict[str, Any]:
        """Build a safe execution namespace with SymPy objects.

        Returns a dictionary that serves as the *globals* namespace for the
        sandboxed ``exec()``.  Only a minimal set of Python builtins and
        SymPy symbols/functions are exposed.
        """
        if _sympy_module is None:
            raise SymPyToolError(
                "SymPy is not installed. Install it with: pip install sympy>=1.12"
            )

        namespace: Dict[str, Any] = {
            "__builtins__": {
                "print": print,
                "range": range,
                "len": len,
                "int": int,
                "float": float,
                "str": str,
                "list": list,
                "tuple": tuple,
                "dict": dict,
                "set": set,
                "sum": sum,
                "max": max,
                "min": min,
                "abs": abs,
                "round": round,
                "pow": pow,
                "True": True,
                "False": False,
                "None": None,
                "type": type,
                "isinstance": isinstance,
                "enumerate": enumerate,
                "zip": zip,
                "map": map,
                "filter": filter,
                "sorted": sorted,
                "reversed": reversed,
                "all": all,
                "any": any,
                "Exception": Exception,
                "ArithmeticError": ArithmeticError,
                "ValueError": ValueError,
                "ZeroDivisionError": ZeroDivisionError,
            },
            "sympy": _sympy_module,
            "sp": _sympy_module,
            # Pre-defined symbols
            "x": self.x,
            "y": self.y,
            "z": self.z,
            "t": self.t,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "n": self.n,
            "m": self.m,
            "k": self.k,
            "theta": self.theta,
            "pi": self.pi,
            "e": self.e,
            "oo": self.inf,
            # Common SymPy functions exposed directly
            **self.operators,
        }
        return namespace

    def _preprocess_code(self, code: str) -> str:
        """Strip redundant ``import sympy`` lines from *code*.

        The sandbox already injects ``sp`` and ``sympy`` into the namespace,
        so importing again is unnecessary and may fail because ``__import__
        is not exposed in the restricted builtins.
        """
        # Remove lines that import sympy (with optional 'as sp')
        lines = code.splitlines()
        filtered = [
            line
            for line in lines
            if not re.match(r"^\s*import\s+sympy(?:\s+as\s+\w+)?\s*$", line)
        ]
        return "\n".join(filtered)

    def _execute(self, code: str) -> str:
        """Execute SymPy code safely with a thread-based timeout.

        Parameters
        ----------
        code:
            Raw Python code string (may contain markdown fences).

        Returns
        -------
        str
            Printed output from the code, or an error message prefixed with
            ``"Error: "``.
        """
        code = self._strip_markdown(code)
        code = self._preprocess_code(code)
        if not code.strip():
            return "Error: No code provided."

        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        result: Dict[str, Any] = {
            "output": "",
            "error": None,
            "result": None,
        }

        def execute() -> None:
            try:
                ns = self._build_namespace()
                exec(code, ns)
                result["output"] = sys.stdout.getvalue()
                # If nothing was printed but a *result* variable was assigned,
                # capture its string representation automatically.
                if not result["output"].strip() and "result" in ns:
                    result["output"] = str(ns["result"])
            except Exception as exc:
                result["error"] = f"{type(exc).__name__}: {str(exc)}"

        thread = threading.Thread(target=execute)
        thread.daemon = True
        thread.start()
        thread.join(timeout=self.TIMEOUT)
        sys.stdout = old_stdout

        if thread.is_alive():
            return (
                f"Error: Execution timed out after {self.TIMEOUT}s.\n"
                "The expression may be too complex or have no closed-form solution.\n"
                "Try simplifying the expression or using numerical methods."
            )
        if result["error"]:
            return f"Error: {result['error']}"
        return result["output"].strip() or "(no output)"

    def _strip_markdown(self, code: str) -> str:
        """Remove markdown fences (```python ... ```) from *code*."""
        code = re.sub(r"^```python\s*", "", code, flags=re.MULTILINE)
        code = re.sub(r"^```\s*", "", code, flags=re.MULTILINE)
        code = re.sub(r"```\s*$", "", code, flags=re.MULTILINE)
        return code.strip()

    # ------------------------------------------------------------------
    # Public API — JEE calculus methods
    # ------------------------------------------------------------------

    def solve_limit(
        self,
        expression: str,
        variable: str,
        point: str,
        direction: str = "+",
    ) -> str:
        """Compute limit of *expression* as *variable* approaches *point*.

        Examples
        --------
        >>> tool = SymPyTool()
        >>> tool.solve_limit("sin(x)/x", "x", "0")
        '1'
        >>> tool.solve_limit("(1+1/x)**x", "x", "oo")
        'E'
        """
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
result = sp.limit(expr, {variable}, {point}, dir='{direction}')
print(result)
"""
        return self._execute(code)

    def solve_derivative(
        self, expression: str, variable: str, order: int = 1
    ) -> str:
        """Compute the *n*-th derivative of *expression* w.r.t. *variable*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
result = sp.diff(expr, {variable}, {order})
print(result)
"""
        return self._execute(code)

    def solve_integral(self, expression: str, variable: str) -> str:
        """Compute the indefinite integral of *expression*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
result = sp.integrate(expr, {variable})
print(result)
"""
        return self._execute(code)

    def solve_definite_integral(
        self, expression: str, variable: str, lower: str, upper: str
    ) -> str:
        """Compute the definite integral of *expression* from *lower* to *upper*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
result = sp.integrate(expr, ({variable}, {lower}, {upper}))
print(result)
"""
        return self._execute(code)

    def solve_equation(self, equation: str, variable: str) -> str:
        """Solve *equation* = 0 for *variable*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
result = sp.solve({equation}, {variable})
print(result)
"""
        return self._execute(code)

    def solve_ode(
        self, equation: str, function: str, variable: str
    ) -> str:
        """Solve an ordinary differential equation.

        *equation* should be a SymPy Eq object or expression = 0 form.
        *function* is the name of the dependent function (e.g. ``'f'``).
        """
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
f = sp.Function('{function}')
result = sp.dsolve({equation}, f({variable}))
print(result)
"""
        return self._execute(code)

    def simplify_expression(self, expression: str) -> str:
        """Simplify a mathematical expression using SymPy."""
        code = f"""\
import sympy as sp
x, y, z = sp.symbols('x y z', real=True)
expr = {expression}
result = sp.simplify(expr)
print(result)
"""
        return self._execute(code)

    def factor_expression(self, expression: str) -> str:
        """Factor a mathematical expression."""
        code = f"""\
import sympy as sp
x, y, z = sp.symbols('x y z', real=True)
expr = {expression}
result = sp.factor(expr)
print(result)
"""
        return self._execute(code)

    def expand_expression(self, expression: str) -> str:
        """Expand a mathematical expression."""
        code = f"""\
import sympy as sp
x, y, z = sp.symbols('x y z', real=True)
expr = {expression}
result = sp.expand(expr)
print(result)
"""
        return self._execute(code)

    def evaluate_expression(
        self,
        expression: str,
        substitutions: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Evaluate *expression* numerically with optional substitutions.

        Parameters
        ----------
        substitutions:
            Mapping of symbol names to numeric values, e.g.
            ``{"x": 1, "y": 2}``.
        """
        sub_lines = ""
        if substitutions:
            for var_name, value in substitutions.items():
                # Reference the pre-defined symbol directly by name
                # (x, y, z, a, b, c, n, m, k, theta are pre-injected)
                sub_lines += f"\nexpr = expr.subs({var_name}, {value})"

        code = f"""\
import sympy as sp
x, y, z = sp.symbols('x y z', real=True)
expr = {expression}
{sub_lines}
result = expr.evalf()
print(result)
"""
        return self._execute(code)

    def find_maxima_minima(
        self,
        expression: str,
        variable: str,
        domain: Optional[Tuple[str, str]] = None,
    ) -> str:
        """Find critical points and classify as maxima / minima.

        Parameters
        ----------
        domain:
            Optional ``(lower, upper)`` tuple restricting the analysis to a
            closed interval.
        """
        domain_str = ""
        if domain:
            domain_str = (
                f", domain=sp.Interval({domain[0]}, {domain[1]})"
            )

        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
f_prime = sp.diff(expr, {variable})
critical = sp.solve(f_prime, {variable})
second = sp.diff(f_prime, {variable})
print(f"Expression: ${{expr}}")
print(f"First derivative: ${{f_prime}}")
print(f"Second derivative: ${{second}}")
print(f"Critical points: ${{critical}}")
if not critical:
    print("No critical points found.")
for cp in critical:
    if cp.is_real{domain_str.replace('domain=', '').replace('sp.Interval', ' and ').replace(',', ' <= cp <= ')}:
        val = second.subs({variable}, cp)
        func_val = sp.simplify(expr.subs({variable}, cp))
        if val > 0:
            print(f"x={{cp}} is a local MINIMUM, f={{func_val}}")
        elif val < 0:
            print(f"x={{cp}} is a local MAXIMUM, f={{func_val}}")
        elif val == 0:
            print(f"x={{cp}} is a possible inflection point (second derivative = 0)")
        else:
            print(f"x={{cp}}: inconclusive (second derivative = {{val}})")
"""
        return self._execute(code)

    def find_tangent_normal(
        self, curve: str, variable: str, point_x: str
    ) -> str:
        """Find equations of tangent and normal to *curve* at *point_x*.

        *curve* must be an expression in terms of *variable* representing
        y = curve(variable).
        """
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
y = sp.Function('y')
curve = {curve}
dy_dx = sp.diff(curve, {variable})
slope = dy_dx.subs({variable}, {point_x})
point_y = sp.simplify(curve.subs({variable}, {point_x}))
print(f"Point: ({point_x}, {{point_y}})")
print(f"Slope of tangent: {{slope}}")
if slope == sp.oo or slope == -sp.oo:
    print(f"Tangent: x = {point_x} (vertical line)")
    print(f"Normal: y = {{point_y}} (horizontal line)")
elif slope == 0:
    print(f"Tangent: y = {{point_y}} (horizontal line)")
    print(f"Normal: x = {point_x} (vertical line)")
else:
    tangent = slope * (sp.Symbol('x') - {point_x}) + point_y
    normal = (-1/slope) * (sp.Symbol('x') - {point_x}) + point_y
    print(f"Tangent: y = {{sp.simplify(tangent)}}")
    print(f"Normal: y = {{sp.simplify(normal)}}")
"""
        return self._execute(code)

    def area_under_curve(
        self, curve: str, variable: str, lower: str, upper: str
    ) -> str:
        """Calculate the (signed) area under *curve* between *lower* and *upper*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {curve}
# Split integration at roots to handle sign changes properly
roots = sp.solve(expr, {variable})
relevant_roots = [r for r in roots if r.is_real and {lower} < r < {upper}]
relevant_roots = sorted([float(r.evalf()) if hasattr(r, 'evalf') else r for r in relevant_roots])
bounds = [{lower}] + relevant_roots + [{upper}]
total_area = 0
for i in range(len(bounds)-1):
    a, b = bounds[i], bounds[i+1]
    mid = (a + b) / 2
    sign = sp.sign(expr.subs({variable}, mid))
    piece = sp.integrate(expr, ({variable}, a, b))
    total_area += sp.Abs(piece)
    print(f"Interval [{{a}}, {{b}}]: integral = {{piece}}, contribution = {{sp.Abs(piece)}}")
print(f"Total area: {{sp.simplify(total_area)}}")
"""
        return self._execute(code)

    def check_continuity(
        self, expression: str, variable: str, point: str
    ) -> str:
        """Check whether *expression* is continuous at *point*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
left = sp.limit(expr, {variable}, {point}, dir='-')
right = sp.limit(expr, {variable}, {point}, dir='+')
try:
    val = expr.subs({variable}, {point})
    print(f"Function value at {point}: {{val}}")
except Exception:
    val = None
    print(f"Function value at {point}: undefined")
print(f"Left limit: {{left}}")
print(f"Right limit: {{right}}")
if left == right:
    if val is not None and left == val:
        print(f"Result: Continuous at {point}")
    elif val is not None:
        print(f"Result: Removable discontinuity at {point} (limit exists but f({point}) != limit)")
    else:
        print(f"Result: Removable discontinuity at {point} (limit exists but function undefined)")
else:
    print(f"Result: Discontinuous at {point} (jump/infinite discontinuity)")
"""
        return self._execute(code)

    def check_differentiability(
        self, expression: str, variable: str, point: str
    ) -> str:
        """Check whether *expression* is differentiable at *point*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
# Check left and right derivatives at the point
h = sp.symbols('h', positive=True)
left_diff = sp.limit((expr.subs({variable}, {point} - h) - expr.subs({variable}, {point})) / (-h), h, 0, dir='+')
right_diff = sp.limit((expr.subs({variable}, {point} + h) - expr.subs({variable}, {point})) / h, h, 0, dir='+')
print(f"Left derivative at {point}: {{left_diff}}")
print(f"Right derivative at {point}: {{right_diff}}")
if left_diff == right_diff and left_diff != sp.oo and left_diff != -sp.oo:
    print(f"Result: Differentiable at {point}, derivative = {{left_diff}}")
else:
    print(f"Result: Not differentiable at {point}")
"""
        return self._execute(code)

    def partial_fraction(self, expression: str, variable: str) -> str:
        """Decompose rational *expression* into partial fractions."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
result = sp.apart(expr, {variable})
print(result)
"""
        return self._execute(code)

    def taylor_series(
        self, expression: str, variable: str, point: str, order: int
    ) -> str:
        """Compute the Taylor / Maclaurin series expansion of *expression*."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
result = sp.series(expr, {variable}, {point}, {order + 1}).removeO()
print(result)
"""
        return self._execute(code)

    def implicit_differentiation(
        self, equation: str, y_var: str, x_var: str
    ) -> str:
        """Perform implicit differentiation to find dy/dx.

        *equation* should be a SymPy expression representing F(x, y) = 0.
        """
        code = f"""\
import sympy as sp
{x_var} = sp.symbols('{x_var}', real=True)
{y_var} = sp.Function('{y_var}')
expr = {equation}
dy_dx = sp.idiff(expr, {y_var}({x_var}), {x_var})
print(dy_dx)
"""
        return self._execute(code)

    def parametric_derivative(
        self,
        x_expr: str,
        y_expr: str,
        parameter: str,
        order: int = 1,
    ) -> str:
        """Compute dy/dx for parametric equations x(t), y(t)."""
        code = f"""\
import sympy as sp
{parameter} = sp.symbols('{parameter}', real=True)
x = {x_expr}
y = {y_expr}
dx_dt = sp.diff(x, {parameter})
dy_dt = sp.diff(y, {parameter})
dy_dx = sp.simplify(dy_dt / dx_dt)
print(f"dx/dt = {{dx_dt}}")
print(f"dy/dt = {{dy_dt}}")
print(f"dy/dx = {{dy_dx}}")
if {order} >= 2:
    d2y_dx2 = sp.simplify(sp.diff(dy_dx, {parameter}) / dx_dt)
    print(f"d2y/dx2 = {{d2y_dx2}}")
"""
        return self._execute(code)

    def solve_by_parts(
        self, u: str, dv: str, variable: str
    ) -> str:
        """Integration by parts: integral(u * dv) = u*v - integral(v*du)."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
u = {u}
dv = {dv}
du = sp.diff(u, {variable})
v = sp.integrate(dv, {variable})
result = u*v - sp.integrate(v*du, {variable})
print(f"u = {{u}}")
print(f"dv = {{dv}}")
print(f"du = {{du}}")
print(f"v = {{v}}")
print(f"Result: {{sp.simplify(result)}}")
"""
        return self._execute(code)

    def substitution_method(
        self, expression: str, variable: str, substitution: str, new_var: str
    ) -> str:
        """Perform integration using substitution.

        *substitution* is the expression for the new variable in terms of
        the old one, e.g. ``'x**2'`` for u = x**2.
        """
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
{new_var} = sp.symbols('{new_var}', real=True)
expr = {expression}
# Substitute: express {variable} in terms of {new_var} and compute dx/du
sub_expr = sp.solve(sp.Eq({new_var}, {substitution}), {variable})
if not sub_expr:
    print("Error: Could not solve substitution equation")
else:
    x_in_u = sub_expr[0]
    dx_du = sp.diff(x_in_u, {new_var})
    transformed = expr.subs({variable}, x_in_u) * dx_du
    transformed = sp.simplify(transformed)
    print(f"Substitution: {variable} = {{x_in_u}}")
    print(f"dx/d{new_var} = {{dx_du}}")
    print(f"Transformed integral: integral({{transformed}}, d{new_var})")
    result = sp.integrate(transformed, {new_var})
    # Substitute back
    final = result.subs({new_var}, {substitution})
    print(f"After integration: {{result}}")
    print(f"After back-substitution: {{sp.simplify(final)}}")
"""
        return self._execute(code)

    def solve_system(
        self, equations: List[str], variables: List[str]
    ) -> str:
        """Solve a system of equations.

        *equations* is a list of SymPy expression strings (each = 0).
        *variables* is a list of variable names to solve for.
        """
        eq_str = ", ".join(equations)
        var_str = ", ".join(variables)
        code = f"""\
import sympy as sp
{chr(10).join(f"{v} = sp.symbols('{v}', real=True)" for v in variables)}
equations = [{eq_str}]
result = sp.solve(equations, [{var_str}])
print(result)
"""
        return self._execute(code)

    def series_expansion_limit(
        self, expression: str, variable: str, point: str, order: int = 5
    ) -> str:
        """Evaluate a limit using Taylor / Maclaurin series expansion.

        Useful for indeterminate forms where direct limit is hard.
        """
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
series_num = sp.series(sp.numer(expr), {variable}, {point}, {order + 1}).removeO()
series_den = sp.series(sp.denom(expr), {variable}, {point}, {order + 1}).removeO()
print(f"Numerator series: {{series_num}}")
print(f"Denominator series: {{series_den}}")
simplified = sp.simplify(series_num / series_den)
result = sp.limit(simplified, {variable}, {point})
print(f"Limit from series: {{result}}")
# Verify with direct limit
direct = sp.limit(expr, {variable}, {point})
print(f"Direct limit (verification): {{direct}}")
"""
        return self._execute(code)

    def lhospital_limit(
        self, expression: str, variable: str, point: str
    ) -> str:
        """Evaluate a 0/0 or oo/oo limit using L'Hopital's Rule.

        Applies successive differentiation of numerator and denominator
        until the limit is determinate or max iterations reached.
        """
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
expr = {expression}
num = sp.numer(expr)
den = sp.denom(expr)
point = {point}
print(f"Original: limit({{num}}/{{den}}, {variable} -> {{point}})")
for i in range(1, 6):
    lim_num = sp.limit(num, {variable}, point)
    lim_den = sp.limit(den, {variable}, point)
    print(f"Step {{i}}: num_limit={{lim_num}}, den_limit={{lim_den}}")
    if lim_den != 0:
        result = sp.limit(num/den, {variable}, point)
        print(f"Result after {{i}} application(s) of L'Hopital: {{result}}")
        break
    elif lim_num == 0 and lim_den == 0:
        num = sp.diff(num, {variable})
        den = sp.diff(den, {variable})
        print(f"  -> Applying L'Hopital: num'={{num}}, den'={{den}}")
    else:
        print(f"Indeterminate form not resolved. Final limit: {{sp.limit(num/den, {variable}, point)}}")
        break
else:
    print(f"L'Hopital did not converge after 5 iterations. Limit: {{sp.limit(expr, {variable}, point)}}")
"""
        return self._execute(code)

    def run_generic(self, code: str) -> str:
        """Execute arbitrary SymPy code safely.

        This is an escape-hatch for operations not covered by the typed
        methods above.  The code still runs in the restricted sandbox with
        timeout protection.
        """
        return self._execute(code)

    def help(self) -> str:
        """Return a description of all available tools."""
        return TOOL_DESCRIPTIONS


# ---------------------------------------------------------------------------
# Module-level constant — identical to the one in jee_prompts.py so that
# both the LLM and the tool class can reference the same descriptions.
# ---------------------------------------------------------------------------

TOOL_DESCRIPTIONS: str = """
Available SymPy tools:
- solve_limit(expression, variable, point, direction='+') — Compute limits
- solve_derivative(expression, variable, order=1) — Differentiation
- solve_integral(expression, variable) — Indefinite integration
- solve_definite_integral(expression, variable, lower, upper) — Definite integration
- solve_equation(equation, variable) — Solve equations
- solve_ode(equation, function, variable) — Differential equations
- solve_system(equations, variables) — System of equations
- find_maxima_minima(expression, variable) — Critical points & classification
- find_tangent_normal(curve, variable, point_x) — Tangent / normal lines
- area_under_curve(curve, variable, lower, upper) — Area calculation
- check_continuity(expression, variable, point) — Continuity check
- check_differentiability(expression, variable, point) — Differentiability check
- partial_fraction(expression, variable) — Partial fractions
- taylor_series(expression, variable, point, order) — Series expansion
- series_expansion_limit(expression, variable, point, order) — Limit via series
- lhospital_limit(expression, variable, point) — L'Hopital's rule
- implicit_differentiation(equation, y_var, x_var) — Implicit differentiation
- parametric_derivative(x_expr, y_expr, parameter, order) — Parametric derivatives
- solve_by_parts(u, dv, variable) — Integration by parts
- substitution_method(expression, variable, substitution, new_var) — Substitution
- simplify_expression(expression) — Simplify expressions
- factor_expression(expression) — Factor expressions
- expand_expression(expression) — Expand expressions
- evaluate_expression(expression, substitutions) — Numerical evaluation
- run_generic(code) — Execute arbitrary SymPy code
"""
