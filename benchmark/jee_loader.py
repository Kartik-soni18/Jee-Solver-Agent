"""JEE Advanced mathematics problem loader with embedded dataset."""

import re
from typing import Optional


JEE_PROBLEMS = [
    # === LIMITS (8 problems) ===
    {
        "question": r"Evaluate: $\lim_{x \to 0} \frac{\sin(3x) - 3\sin(x)}{x^3}$",
        "answer": "-4",
        "problem_type": "limits",
        "topic": "Trigonometric Limits",
        "difficulty": "medium",
        "solution_method": "L'Hopital's Rule or Taylor Series",
    },
    {
        "question": r"Evaluate: $\lim_{x \to \infty} \left( \frac{x+3}{x+1} \right)^{x+2}$",
        "answer": "e^2",
        "problem_type": "limits",
        "topic": r"Exponential Limits ($1^\infty$ form)",
        "difficulty": "medium",
        "solution_method": "Standard limit form",
    },
    {
        "question": r"Evaluate: $\lim_{x \to 0} \frac{\tan(x) - \sin(x)}{x^3}$",
        "answer": "1/2",
        "problem_type": "limits",
        "topic": "Trigonometric Limits",
        "difficulty": "easy",
        "solution_method": "Taylor expansion or L'Hopital",
    },
    {
        "question": r"Evaluate: $\lim_{x \to 0} \frac{e^x - 1 - x}{x^2}$",
        "answer": "1/2",
        "problem_type": "limits",
        "topic": "Exponential Limits",
        "difficulty": "easy",
        "solution_method": "L'Hopital's Rule",
    },
    {
        "question": r"Evaluate: $\lim_{x \to 0} \frac{\ln(1+x) - x}{x^2}$",
        "answer": "-1/2",
        "problem_type": "limits",
        "topic": "Logarithmic Limits",
        "difficulty": "easy",
        "solution_method": "L'Hopital's Rule",
    },
    {
        "question": r"If $\lim_{x \to 0} \frac{a\sin(x) - b\sin(2x) + c\sin(3x)}{x^5} = 1$, find $a + b + c$.",
        "answer": "6",
        "problem_type": "limits",
        "topic": "Coefficient determination from limits",
        "difficulty": "hard",
        "solution_method": "Taylor series expansion",
    },
    {
        "question": r"Evaluate: $\lim_{x \to 0} \frac{\sqrt{1+x} - \sqrt{1-x}}{x}$",
        "answer": "1",
        "problem_type": "limits",
        "topic": "Algebraic Limits",
        "difficulty": "easy",
        "solution_method": "Rationalization",
    },
    {
        "question": r"Evaluate: $\lim_{x \to \infty} x \left( \ln(1+x) - \ln(x) \right)$",
        "answer": "1",
        "problem_type": "limits",
        "topic": "Logarithmic Limits",
        "difficulty": "medium",
        "solution_method": "Substitution and standard limits",
    },
    
    # === DIFFERENTIATION (7 problems) ===
    {
        "question": r"If $y = \ln\left(\frac{1+\sin(x)}{1-\sin(x)}\right)$, find $\frac{dy}{dx}$ at $x = \frac{\pi}{6}$.",
        "answer": "4",
        "problem_type": "differentiation",
        "topic": "Logarithmic Differentiation",
        "difficulty": "medium",
        "solution_method": "Chain rule and log simplification",
    },
    {
        "question": r"If $x = a(t - \sin(t))$ and $y = a(1 - \cos(t))$, find $\frac{d^2y}{dx^2}$.",
        "answer": "-1/(a*(1-cos(t))^2)",
        "problem_type": "differentiation",
        "topic": "Parametric Differentiation",
        "difficulty": "medium",
        "solution_method": "Parametric derivative formula",
    },
    {
        "question": r"If $x^y = y^x$, find $\frac{dy}{dx}$.",
        "answer": "(y*(y-x*ln(y)))/(x*(x-y*ln(x)))",
        "problem_type": "differentiation",
        "topic": "Implicit Differentiation",
        "difficulty": "hard",
        "solution_method": "Logarithmic differentiation",
    },
    {
        "question": r"Find the derivative of $\tan^{-1}\left(\frac{\sqrt{1+x^2} - 1}{x}\right)$ with respect to $x$.",
        "answer": "1/(2*(1+x^2))",
        "problem_type": "differentiation",
        "topic": "Inverse Trigonometric Differentiation",
        "difficulty": "hard",
        "solution_method": "Substitution and chain rule",
    },
    {
        "question": r"If $f(x) = x^x$ for $x > 0$, find $f'(2)$.",
        "answer": "4*(ln(2)+1)",
        "problem_type": "differentiation",
        "topic": "Exponential Differentiation",
        "difficulty": "medium",
        "solution_method": "Logarithmic differentiation",
    },
    {
        "question": r"Find $\frac{d}{dx} \left( e^{\sin(x)} \cdot \ln(\cos(x)) \right)$.",
        "answer": "e^sin(x)*(cos(x)*ln(cos(x)) - tan(x))",
        "problem_type": "differentiation",
        "topic": "Product Rule",
        "difficulty": "easy",
        "solution_method": "Product rule + chain rule",
    },
    {
        "question": r"If $y = \sin^{-1}(2x\sqrt{1-x^2})$, find $\frac{dy}{dx}$ for $0 < x < \frac{1}{\sqrt{2}}$.",
        "answer": "2/sqrt(1-x^2)",
        "problem_type": "differentiation",
        "topic": "Inverse Trigonometric Substitution",
        "difficulty": "hard",
        "solution_method": "Substitution x = sin(\\theta)",
    },
    
    # === INTEGRATION (8 problems) ===
    {
        "question": r"Evaluate: $\int \frac{dx}{x^2 + 4x + 8}$",
        "answer": "atan((x+2)/2)/2",
        "problem_type": "integration",
        "topic": "Integration by Completing the Square",
        "difficulty": "easy",
        "solution_method": "Complete square + arctan formula",
    },
    {
        "question": r"Evaluate: $\int \frac{x^2 + 1}{x^4 + 1} dx$",
        "answer": "atan((x^2-1)/x)/sqrt(2)",
        "problem_type": "integration",
        "topic": "Algebraic Integration",
        "difficulty": "hard",
        "solution_method": "Divide by x^2, substitution",
    },
    {
        "question": r"Evaluate: $\int_0^{\pi/2} \frac{\sqrt{\sin(x)}}{\sqrt{\sin(x)} + \sqrt{\cos(x)}} dx$",
        "answer": "pi/4",
        "problem_type": "definite_integrals",
        "topic": "Definite Integral Properties (King's Property)",
        "difficulty": "medium",
        "solution_method": "Property: integral_0^a f(x) = integral_0^a f(a-x)",
    },
    {
        "question": r"Evaluate: $\int \frac{dx}{1 + \sin(x) + \cos(x)}$",
        "answer": "ln(1+tan(x/2))",
        "problem_type": "integration",
        "topic": "Trigonometric Integration",
        "difficulty": "medium",
        "solution_method": "Universal substitution t = tan(x/2)",
    },
    {
        "question": r"Evaluate: $\int_0^1 x \cdot e^x dx$",
        "answer": "1",
        "problem_type": "definite_integrals",
        "topic": "Integration by Parts",
        "difficulty": "easy",
        "solution_method": "Integration by parts",
    },
    {
        "question": r"Evaluate: $\int \frac{x \cdot e^x}{(x+1)^2} dx$",
        "answer": "e^x/(x+1)",
        "problem_type": "integration",
        "topic": "Integration by Parts (special)",
        "difficulty": "hard",
        "solution_method": "Recognize derivative of e^x/(x+1)",
    },
    {
        "question": r"Evaluate: $\int_0^{\pi/4} \ln(1 + \tan(x)) dx$",
        "answer": "pi*ln(2)/8",
        "problem_type": "definite_integrals",
        "topic": "Definite Integral using King's Property",
        "difficulty": "hard",
        "solution_method": "King's property: integral_0^{pi/4} f(x) = integral_0^{pi/4} f(pi/4-x)",
    },
    {
        "question": r"Evaluate: $\int \sin^{-1}(x) dx$",
        "answer": "x*asin(x) + sqrt(1-x^2)",
        "problem_type": "integration",
        "topic": "Integration by Parts (inverse trig)",
        "difficulty": "medium",
        "solution_method": "Integration by parts",
    },
    
    # === DIFFERENTIAL EQUATIONS (4 problems) ===
    {
        "question": r"Solve: $\frac{dy}{dx} = \frac{x + y}{x - y}$",
        "answer": "atan(y/x) = ln(sqrt(x^2+y^2)) + C",
        "problem_type": "differential_equations",
        "topic": "Homogeneous Differential Equation",
        "difficulty": "medium",
        "solution_method": "Substitution y = vx",
    },
    {
        "question": r"Solve: $\frac{dy}{dx} + y \cdot \tan(x) = \cos(x)$",
        "answer": "y = (x+C)*cos(x)",
        "problem_type": "differential_equations",
        "topic": "Linear First-Order DE",
        "difficulty": "medium",
        "solution_method": "Integrating factor",
    },
    {
        "question": r"Solve: $\frac{d^2y}{dx^2} + y = \sin(x)$ with $y(0) = 0$ and $y'(0) = 0$.",
        "answer": "-x*cos(x)/2 + sin(x)/2",
        "problem_type": "differential_equations",
        "topic": "Second-Order Linear DE",
        "difficulty": "hard",
        "solution_method": "Complementary function + particular integral",
    },
    {
        "question": r"Solve: $x \frac{dy}{dx} + 2y = x^2 \ln(x)$",
        "answer": "y = x^2*ln(x)/3 - x^2/9 + C/x^2",
        "problem_type": "differential_equations",
        "topic": "Linear First-Order DE (Bernoulli)",
        "difficulty": "hard",
        "solution_method": "Integrating factor after standard form",
    },
    
    # === APPLICATION OF DERIVATIVES (3 problems) ===
    {
        "question": r"Find the maximum value of $f(x) = x \cdot e^{-x}$ for $x \geq 0$.",
        "answer": "1/e",
        "problem_type": "maxima_minima",
        "topic": "Maxima/Minima",
        "difficulty": "easy",
        "solution_method": "Find critical points, second derivative test",
    },
    {
        "question": r"Find the point on the curve $y = x^2$ closest to the point $(3, 0)$.",
        "answer": "(1, 1)",
        "problem_type": "maxima_minima",
        "topic": "Distance Minimization",
        "difficulty": "medium",
        "solution_method": "Minimize distance function",
    },
    {
        "question": r"Find the equation of the tangent to $y = x^3 - 2x + 1$ at $x = 2$.",
        "answer": "y = 10*x - 15",
        "problem_type": "tangent_normal",
        "topic": "Tangent Line",
        "difficulty": "easy",
        "solution_method": "Point-slope form",
    },
    
    # === DEFINITE INTEGRALS / AREA (3 problems) ===
    {
        "question": r"Find the area enclosed by $y = x^2$ and $y = \sqrt{x}$.",
        "answer": "1/3",
        "problem_type": "area",
        "topic": "Area Between Curves",
        "difficulty": "medium",
        "solution_method": "Find intersection, integrate difference",
    },
    {
        "question": r"Evaluate: $\int_{-1}^{1} |x| dx$",
        "answer": "1",
        "problem_type": "definite_integrals",
        "topic": "Modulus Integration",
        "difficulty": "easy",
        "solution_method": "Split at x=0",
    },
    {
        "question": r"Evaluate: $\int_0^{\pi/2} \frac{\sin^3(x)}{\sin^3(x) + \cos^3(x)} dx$",
        "answer": "pi/4",
        "problem_type": "definite_integrals",
        "topic": "Definite Integral Symmetry",
        "difficulty": "medium",
        "solution_method": "Property: integral_0^{pi/2} f(sin,cos) = integral_0^{pi/2} f(cos,sin)",
    },
]


class JEELoader:
    """Load JEE problems from embedded dataset."""

    @staticmethod
    def load_problems(
        max_problems: int = 0,
        problem_type: Optional[str] = None,
        difficulty: Optional[str] = None,
    ) -> list[dict]:
        """Load JEE problems with optional filtering.

        Args:
            max_problems: Maximum number of problems to return (0 = all).
            problem_type: Filter by problem type (e.g., 'limits', 'integration').
            difficulty: Filter by difficulty ('easy', 'medium', 'hard').

        Returns:
            List of problem dictionaries.
        """
        problems = JEE_PROBLEMS.copy()
        if problem_type:
            problems = [p for p in problems if p["problem_type"] == problem_type]
        if difficulty:
            problems = [p for p in problems if p["difficulty"] == difficulty]
        if max_problems > 0:
            problems = problems[:max_problems]
        return problems

    @staticmethod
    def get_problem_types() -> list[str]:
        """Return all unique problem types in the dataset."""
        return sorted(set(p["problem_type"] for p in JEE_PROBLEMS))

    @staticmethod
    def get_topics() -> list[str]:
        """Return all unique topics in the dataset."""
        return sorted(set(p["topic"] for p in JEE_PROBLEMS))

    @staticmethod
    def get_difficulty_distribution() -> dict:
        """Return the distribution of problems by difficulty."""
        dist = {}
        for p in JEE_PROBLEMS:
            d = p["difficulty"]
            dist[d] = dist.get(d, 0) + 1
        return dist

    @staticmethod
    def get_type_distribution() -> dict:
        """Return the distribution of problems by type."""
        dist = {}
        for p in JEE_PROBLEMS:
            t = p["problem_type"]
            dist[t] = dist.get(t, 0) + 1
        return dist

    @staticmethod
    def get_dataset_summary() -> dict:
        """Return a comprehensive summary of the dataset."""
        return {
            "total_problems": len(JEE_PROBLEMS),
            "problem_types": JEELoader.get_problem_types(),
            "topics": JEELoader.get_topics(),
            "difficulty_distribution": JEELoader.get_difficulty_distribution(),
            "type_distribution": JEELoader.get_type_distribution(),
        }
