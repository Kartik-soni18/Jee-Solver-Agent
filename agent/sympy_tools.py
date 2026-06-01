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
        {"sympy", "math", "fractions", "decimal", "numpy", "itertools", "statistics"}
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
                "getattr": getattr,
                "eval": eval,
                "Exception": Exception,
                "ArithmeticError": ArithmeticError,
                "ValueError": ValueError,
                "ZeroDivisionError": ZeroDivisionError,
                "itertools": __import__("itertools"),
                "math": __import__("math"),
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
            # Matrix and complex helpers
            "Matrix": _sympy_module.Matrix,
            "I": _sympy_module.I,
            "re": _sympy_module.re,
            "im": _sympy_module.im,
            "arg": _sympy_module.arg,
            "conjugate": _sympy_module.conjugate,
            "factorial": _sympy_module.factorial,
            "binomial": _sympy_module.binomial,
            "gamma": _sympy_module.gamma,
            " summation": _sympy_module.summation,
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
        if isinstance(code, list):
            code = "\n".join(str(line) for line in code)
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

    # ------------------------------------------------------------------
    # Algebra tools
    # ------------------------------------------------------------------

    def solve_quadratic(self, a: str, b: str, c: str) -> str:
        """Solve ax^2 + bx + c = 0 and return roots, discriminant, sum/product."""
        code = f"""\
import sympy as sp
x = sp.symbols('x', real=True)
a = sp.sympify('{a}')
b = sp.sympify('{b}')
c = sp.sympify('{c}')
disc = b**2 - 4*a*c
roots = sp.solve(a*x**2 + b*x + c, x)
print(f"Discriminant: {{disc}}")
print(f"Roots: {{roots}}")
print(f"Sum of roots: {{-b/a}}")
print(f"Product of roots: {{c/a}}")
if disc > 0:
    print("Nature: Real and distinct")
elif disc == 0:
    print("Nature: Real and equal")
else:
    print("Nature: Complex conjugate")
"""
        return self._execute(code)

    def complex_operations(self, z1: str, z2: str, op: str) -> str:
        """Perform operation on complex numbers. op: add, sub, mul, div, conjugate, modulus, argument, real, imag."""
        code = f"""\
import sympy as sp
z1 = sp.sympify('{z1}')
z2 = sp.sympify('{z2}')
op = '{op}'
if op == 'add':
    result = z1 + z2
elif op == 'sub':
    result = z1 - z2
elif op == 'mul':
    result = z1 * z2
elif op == 'div':
    result = sp.simplify(z1 / z2)
elif op == 'conjugate':
    result = sp.conjugate(z1)
elif op == 'modulus':
    result = sp.Abs(z1)
elif op == 'argument':
    result = sp.arg(z1)
elif op == 'real':
    result = sp.re(z1)
elif op == 'imag':
    result = sp.im(z1)
else:
    result = "Error: unknown operation"
print(result)
"""
        return self._execute(code)

    def binomial_expansion(self, a: str, b: str, n: str, term_index: Optional[int] = None) -> str:
        """Expand (a + b)^n or return a specific term."""
        ti = f", term_index={term_index}" if term_index is not None else ""
        code = f"""\
import sympy as sp
a = sp.sympify('{a}')
b = sp.sympify('{b}')
n = sp.sympify('{n}')
expr = (a + b)**n
if {term_index is not None}:
    k = {term_index if term_index is not None else 0} - 1
    result = sp.binomial(n, k) * a**(n-k) * b**k
    print(f"Term {{k+1}}: {{result}}")
else:
    expanded = sp.expand(expr)
    print(f"Expansion: {{expanded}}")
"""
        return self._execute(code)

    def evaluate_permutation(self, n: str, r: str) -> str:
        """Compute P(n, r) = n! / (n-r)!."""
        code = f"""\
import sympy as sp
n = sp.sympify('{n}')
r = sp.sympify('{r}')
result = sp.factorial(n) / sp.factorial(n - r)
print(sp.simplify(result))
"""
        return self._execute(code)

    def evaluate_combination(self, n: str, r: str) -> str:
        """Compute C(n, r) = n! / (r!(n-r)!)."""
        code = f"""\
import sympy as sp
n = sp.sympify('{n}')
r = sp.sympify('{r}')
result = sp.binomial(n, r)
print(result)
"""
        return self._execute(code)

    def matrix_determinant(self, matrix_str: str) -> str:
        """Compute determinant of a matrix given as string e.g. '[[1,2],[3,4]]'."""
        code = f"""\
import sympy as sp
M = sp.Matrix({matrix_str})
print(M.det())
"""
        return self._execute(code)

    def matrix_inverse(self, matrix_str: str) -> str:
        """Compute inverse of a matrix."""
        code = f"""\
import sympy as sp
M = sp.Matrix({matrix_str})
print(M.inv())
"""
        return self._execute(code)

    def matrix_multiply(self, a_str: str, b_str: str) -> str:
        """Multiply two matrices."""
        code = f"""\
import sympy as sp
A = sp.Matrix({a_str})
B = sp.Matrix({b_str})
print(A * B)
"""
        return self._execute(code)

    def solve_linear_system(self, equations: List[str], variables: List[str]) -> str:
        """Solve a linear system Ax = b."""
        eq_str = ", ".join(equations)
        var_str = ", ".join(variables)
        code = f"""\
import sympy as sp
{chr(10).join(f"{v} = sp.symbols('{v}', real=True)" for v in variables)}
eqs = [{eq_str}]
result = sp.solve(eqs, [{var_str}])
print(result)
"""
        return self._execute(code)

    def arithmetic_series(self, a: str, d: str, n: str) -> str:
        """Sum of AP: n/2 * (2a + (n-1)d)."""
        code = f"""\
import sympy as sp
a = sp.sympify('{a}')
d = sp.sympify('{d}')
n = sp.sympify('{n}')
result = n/2 * (2*a + (n-1)*d)
print(sp.simplify(result))
"""
        return self._execute(code)

    def geometric_series(self, a: str, r: str, n: str) -> str:
        """Sum of GP: a(1-r^n)/(1-r) for r != 1."""
        code = f"""\
import sympy as sp
a = sp.sympify('{a}')
r = sp.sympify('{r}')
n = sp.sympify('{n}')
result = a * (1 - r**n) / (1 - r)
print(sp.simplify(result))
"""
        return self._execute(code)

    def sum_series(self, expr: str, var: str, start: str, end: str) -> str:
        """Sum a symbolic series."""
        code = f"""\
import sympy as sp
{var} = sp.symbols('{var}', real=True, integer=True)
expr = sp.sympify('{expr}')
result = sp.summation(expr, ({var}, {start}, {end}))
print(result)
"""
        return self._execute(code)

    # ------------------------------------------------------------------
    # Trigonometry tools
    # ------------------------------------------------------------------

    def solve_trig_equation(self, equation: str, variable: str) -> str:
        """Solve a trigonometric equation."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
eq = sp.sympify('{equation}')
result = sp.solve(eq, {variable})
print(f"General solutions: {{result}}")
# Also print principal solutions in [0, 2*pi)
print(f"Principal check: substitute values...")
"""
        return self._execute(code)

    def prove_trig_identity(self, lhs: str, rhs: str) -> str:
        """Simplify LHS and RHS and check equality."""
        code = f"""\
import sympy as sp
x = sp.symbols('x', real=True)
lhs = sp.sympify('{lhs}')
rhs = sp.sympify('{rhs}')
lhs_s = sp.trigsimp(lhs)
rhs_s = sp.trigsimp(rhs)
print(f"LHS simplified: {{lhs_s}}")
print(f"RHS simplified: {{rhs_s}}")
diff = sp.simplify(lhs_s - rhs_s)
print(f"Difference: {{diff}}")
print(f"Equal: {{diff == 0}}")
"""
        return self._execute(code)

    # ------------------------------------------------------------------
    # Coordinate Geometry tools
    # ------------------------------------------------------------------

    def line_equation(self, p1: str, p2: str) -> str:
        """Find equation of line through two points p1=(x1,y1), p2=(x2,y2)."""
        code = f"""\
import sympy as sp
x, y = sp.symbols('x y', real=True)
p1 = {p1}
p2 = {p2}
x1, y1 = p1
x2, y2 = p2
slope = sp.Rational(y2 - y1, x2 - x1) if (x2 - x1) != 0 else sp.oo
eq = sp.Eq(y - y1, slope * (x - x1))
print(f"Slope: {{slope}}")
print(f"Equation: {{sp.simplify(eq.lhs - eq.rhs)}} = 0")
"""
        return self._execute(code)

    def distance_point_line(self, point: str, line_coeffs: str) -> str:
        """Distance from point (x0,y0) to line ax+by+c=0. line_coeffs=(a,b,c)."""
        code = f"""\
import sympy as sp
x0, y0 = {point}
a, b, c = {line_coeffs}
dist = sp.Abs(a*x0 + b*y0 + c) / sp.sqrt(a**2 + b**2)
print(sp.simplify(dist))
"""
        return self._execute(code)

    def point_of_intersection(self, l1: str, l2: str) -> str:
        """Find intersection of two lines given as (a,b,c) tuples for ax+by+c=0."""
        code = f"""\
import sympy as sp
x, y = sp.symbols('x y', real=True)
a1, b1, c1 = {l1}
a2, b2, c2 = {l2}
result = sp.solve([a1*x + b1*y + c1, a2*x + b2*y + c2], [x, y])
print(result)
"""
        return self._execute(code)

    def circle_equation(self, center: str, radius: str) -> str:
        """Equation of circle with given center and radius."""
        code = f"""\
import sympy as sp
x, y = sp.symbols('x y', real=True)
h, k = {center}
r = sp.sympify('{radius}')
eq = sp.Eq((x - h)**2 + (y - k)**2, r**2)
print(f"Standard: {{eq}}")
print(f"Expanded: {{sp.expand(eq.lhs - eq.rhs)}} = 0")
"""
        return self._execute(code)

    def tangent_to_circle(self, circle_eq: str, point: str) -> str:
        """Find tangent to circle at a point. circle_eq should be x^2+y^2+Dx+Ey+F=0 form."""
        code = f"""\
import sympy as sp
x, y = sp.symbols('x y', real=True)
x0, y0 = {point}
# For circle S=0, tangent at (x0,y0) is S1=0
expr = sp.sympify('{circle_eq}')
# Replace x^2 -> x*x0, y^2 -> y*y0, x -> (x+x0)/2, y -> (y+y0)/2
# Simplified: evaluate T = 0
S = sp.sympify('{circle_eq}')
T = S.subs({{x**2: x*x0, y**2: y*y0, x: (x+x0)/2, y: (y+y0)/2}})
print(f"Tangent: {{sp.simplify(T)}} = 0")
"""
        return self._execute(code)

    def conic_properties(self, equation: str, conic_type: str) -> str:
        """Analyze conic: parabola, ellipse, or hyperbola."""
        code = f"""\
import sympy as sp
x, y = sp.symbols('x y', real=True)
eq = sp.sympify('{equation}')
conic = '{conic_type}'
if conic == 'parabola':
    # Try to find vertex and focus by completing square
    print(f"Equation: {{eq}} = 0")
    print("For parabola: find vertex by completing square in x or y")
elif conic == 'ellipse':
    print(f"Equation: {{eq}} = 0")
    print("Standard form: (x-h)^2/a^2 + (y-k)^2/b^2 = 1")
elif conic == 'hyperbola':
    print(f"Equation: {{eq}} = 0")
    print("Standard form: (x-h)^2/a^2 - (y-k)^2/b^2 = 1")
else:
    print("Unknown conic type")
"""
        return self._execute(code)

    # ------------------------------------------------------------------
    # Vectors & 3D Geometry tools
    # ------------------------------------------------------------------

    def vector_dot(self, a: str, b: str) -> str:
        """Dot product of two vectors (lists)."""
        code = f"""\
import sympy as sp
a = sp.Matrix({a})
b = sp.Matrix({b})
print(a.dot(b))
"""
        return self._execute(code)

    def vector_cross(self, a: str, b: str) -> str:
        """Cross product of two 3D vectors (lists)."""
        code = f"""\
import sympy as sp
a = sp.Matrix({a})
b = sp.Matrix({b})
result = a.cross(b)
print(result.T)
"""
        return self._execute(code)

    def vector_triple_product(self, a: str, b: str, c: str) -> str:
        """Scalar triple product [a b c] = a . (b x c)."""
        code = f"""\
import sympy as sp
a = sp.Matrix({a})
b = sp.Matrix({b})
c = sp.Matrix({c})
result = a.dot(b.cross(c))
print(result)
"""
        return self._execute(code)

    def vector_magnitude(self, v: str) -> str:
        """Magnitude of a vector."""
        code = f"""\
import sympy as sp
v = sp.Matrix({v})
print(sp.sqrt(sum(vi**2 for vi in v)))
"""
        return self._execute(code)

    def angle_between_vectors(self, a: str, b: str) -> str:
        """Angle between two vectors in radians."""
        code = f"""\
import sympy as sp
import math
a = sp.Matrix({a})
b = sp.Matrix({b})
dot = a.dot(b)
mag_a = sp.sqrt(sum(vi**2 for vi in a))
mag_b = sp.sqrt(sum(vi**2 for vi in b))
cos_theta = sp.simplify(dot / (mag_a * mag_b))
print(f"cos(theta) = {{cos_theta}}")
print(f"theta = {{sp.acos(cos_theta)}} radians")
"""
        return self._execute(code)

    def line_3d(self, point: str, direction: str) -> str:
        """Equation of line in 3D through point with direction ratios."""
        code = f"""\
import sympy as sp
x, y, z = sp.symbols('x y z', real=True)
x0, y0, z0 = {point}
a, b, c = {direction}
print("Symmetric: (x -", x0, ")/", a, " = (y -", y0, ")/", b, " = (z -", z0, ")/", c, sep="")
"""
        return self._execute(code)

    def plane_equation(self, point: str, normal: str) -> str:
        """Equation of plane through point with normal vector."""
        code = f"""\
import sympy as sp
x, y, z = sp.symbols('x y z', real=True)
x0, y0, z0 = {point}
a, b, c = {normal}
eq = a*(x - x0) + b*(y - y0) + c*(z - z0)
print(f"Plane: {{sp.expand(eq)}} = 0")
"""
        return self._execute(code)

    def distance_point_plane(self, point: str, plane: str) -> str:
        """Distance from point to plane ax+by+cz+d=0. plane=(a,b,c,d)."""
        code = f"""\
import sympy as sp
x0, y0, z0 = {point}
a, b, c, d = {plane}
dist = sp.Abs(a*x0 + b*y0 + c*z0 + d) / sp.sqrt(a**2 + b**2 + c**2)
print(sp.simplify(dist))
"""
        return self._execute(code)

    # ------------------------------------------------------------------
    # Probability & Statistics tools
    # ------------------------------------------------------------------

    def probability_event(self, favorable: str, total: str) -> str:
        """P = favorable / total."""
        code = f"""\
import sympy as sp
f = sp.sympify('{favorable}')
t = sp.sympify('{total}')
print(sp.Rational(f, t) if f == int(f) and t == int(t) else sp.simplify(f/t))
"""
        return self._execute(code)

    def conditional_probability(self, a_and_b: str, b: str) -> str:
        """P(A|B) = P(A∩B) / P(B)."""
        code = f"""\
import sympy as sp
ab = sp.sympify('{a_and_b}')
b = sp.sympify('{b}')
print(sp.simplify(ab / b))
"""
        return self._execute(code)

    def mean_median_mode(self, data_list: str) -> str:
        """Compute mean, median, mode of a data list."""
        code = f"""\
data = {data_list}
n = len(data)
mean = sum(data) / n
sorted_data = sorted(data)
if n % 2 == 1:
    median = sorted_data[n // 2]
else:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
freq = {{}}
for d in data:
    freq[d] = freq.get(d, 0) + 1
max_freq = max(freq.values())
mode_vals = [k for k, v in freq.items() if v == max_freq]
print(f"Mean: {{mean}}")
print(f"Median: {{median}}")
print(f"Mode: {{mode_vals}}")
"""
        return self._execute(code)

    def standard_deviation(self, data_list: str) -> str:
        """Standard deviation of a data list."""
        code = f"""\
data = {data_list}
n = len(data)
mean = sum(data) / n
variance = sum((x - mean) ** 2 for x in data) / (n - 1)
print(variance ** 0.5)
"""
        return self._execute(code)

    def correlation_coefficient(self, x_list: str, y_list: str) -> str:
        """Pearson correlation coefficient r."""
        code = f"""\
x = {x_list}
y = {y_list}
n = len(x)
mean_x = sum(x) / n
mean_y = sum(y) / n
num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
den = (sum((xi - mean_x) ** 2 for xi in x) * sum((yi - mean_y) ** 2 for yi in y)) ** 0.5
print(num / den)
"""
        return self._execute(code)

    # ------------------------------------------------------------------
    # General tools
    # ------------------------------------------------------------------

    def solve_inequality(self, ineq: str, variable: str) -> str:
        """Solve an inequality for a variable."""
        code = f"""\
import sympy as sp
{variable} = sp.symbols('{variable}', real=True)
ineq_str = '{ineq}'
expr = None
for op, rel in [('>=', 'Ge'), ('<=', 'Le'), ('>', 'Gt'), ('<', 'Lt'), ('=', 'Eq')]:
    if op in ineq_str:
        lhs_str, rhs_str = ineq_str.split(op, 1)
        lhs = eval(lhs_str)
        rhs = eval(rhs_str)
        expr = getattr(sp, rel)(lhs, rhs)
        break
if expr is None:
    expr = eval(ineq_str)
try:
    result = sp.solve_univariate_inequality(expr, {variable}, relational=False)
    print(result)
except Exception as e:
    print(f"Error: {{e}}")
"""
        return self._execute(code)

    # ------------------------------------------------------------------
    # Generic escape hatch
    # ------------------------------------------------------------------

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

--- Calculus ---
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

--- Algebra ---
- solve_quadratic(a, b, c) — Solve ax^2+bx+c=0, return roots & discriminant
- complex_operations(z1, z2, op) — op: add, sub, mul, div, conjugate, modulus, argument, real, imag
- binomial_expansion(a, b, n, term_index=None) — Expand (a+b)^n or get specific term
- evaluate_permutation(n, r) — P(n,r)
- evaluate_combination(n, r) — C(n,r)
- matrix_determinant(matrix_str) — e.g. '[[1,2],[3,4]]'
- matrix_inverse(matrix_str)
- matrix_multiply(a_str, b_str)
- solve_linear_system(equations, variables)
- arithmetic_series(a, d, n) — Sum of AP
- geometric_series(a, r, n) — Sum of GP
- sum_series(expr, var, start, end) — Symbolic summation

--- Trigonometry ---
- solve_trig_equation(equation, variable)
- prove_trig_identity(lhs, rhs)

--- Coordinate Geometry ---
- line_equation(p1, p2) — p1,p2 as (x,y) tuples
- distance_point_line(point, line_coeffs) — line_coeffs=(a,b,c) for ax+by+c=0
- point_of_intersection(l1, l2) — l1,l2 as (a,b,c) tuples
- circle_equation(center, radius) — center=(h,k)
- tangent_to_circle(circle_eq, point)
- conic_properties(equation, conic_type) — parabola, ellipse, hyperbola

--- Vectors & 3D Geometry ---
- vector_dot(a, b) — a,b as component lists
- vector_cross(a, b)
- vector_triple_product(a, b, c)
- vector_magnitude(v)
- angle_between_vectors(a, b)
- line_3d(point, direction) — point=(x0,y0,z0), direction=(a,b,c)
- plane_equation(point, normal)
- distance_point_plane(point, plane) — plane=(a,b,c,d) for ax+by+cz+d=0

--- Probability & Statistics ---
- probability_event(favorable, total)
- conditional_probability(a_and_b, b)
- mean_median_mode(data_list)
- standard_deviation(data_list)
- correlation_coefficient(x_list, y_list)

--- General ---
- solve_inequality(ineq, variable)
- run_generic(code) — Execute arbitrary SymPy code
"""
