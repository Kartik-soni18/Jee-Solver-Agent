"""RAG Knowledge Base for JEE Mathematics Problem Solving.

This module provides a Retrieval-Augmented Generation (RAG) system that:
1. Stores JEE problems, formulas, theorems, and solution templates
2. Retrieves top-k relevant documents based on semantic similarity
3. Injects retrieved context into LLM prompts for better problem solving

The knowledge base is built from:
- Embedded JEE problem dataset (~140 problems)
- NCERT formulas and theorems by topic
- Common solution methods and pitfalls
"""

import os
import json
from typing import Optional
from dataclasses import dataclass, field

import numpy as np

# Try to import sentence-transformers and chromadb
try:
    from sentence_transformers import SentenceTransformer
    _HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    _HAS_SENTENCE_TRANSFORMERS = False

try:
    import chromadb
    from chromadb.config import Settings
    _HAS_CHROMADB = True
except ImportError:
    _HAS_CHROMADB = False


# ---------------------------------------------------------------------------
# JEE Formula & Theorem Knowledge Base (NCERT-based)
# ---------------------------------------------------------------------------

JEE_FORMULAS: list[dict] = [
    # === LIMITS ===
    {
        "topic": "limits",
        "title": "Standard Limits",
        "content": "lim(x→0) sin(x)/x = 1, lim(x→0) tan(x)/x = 1, lim(x→0) (e^x - 1)/x = 1, lim(x→0) ln(1+x)/x = 1, lim(x→0) (a^x - 1)/x = ln(a)",
        "type": "formula",
    },
    {
        "topic": "limits",
        "title": "L'Hopital's Rule",
        "content": "If lim(x→a) f(x)/g(x) is 0/0 or ∞/∞, then lim(x→a) f(x)/g(x) = lim(x→a) f'(x)/g'(x), provided the latter limit exists. Apply repeatedly if needed.",
        "type": "theorem",
    },
    {
        "topic": "limits",
        "title": "Exponential Limit Form",
        "content": "lim(x→∞) (1 + a/x)^x = e^a. For 1^∞ form: if lim f(x)^g(x) where f→1 and g→∞, rewrite as e^{lim g(x)·(f(x)-1)}.",
        "type": "formula",
    },
    {
        "topic": "limits",
        "title": "Taylor Series Expansions",
        "content": "sin(x) = x - x³/3! + x⁵/5! - ..., cos(x) = 1 - x²/2! + x⁴/4! - ..., e^x = 1 + x + x²/2! + ..., ln(1+x) = x - x²/2 + x³/3 - ..., (1+x)^n = 1 + nx + n(n-1)x²/2! + ...",
        "type": "formula",
    },
    # === DIFFERENTIATION ===
    {
        "topic": "differentiation",
        "title": "Basic Derivatives",
        "content": "d/dx(x^n) = nx^(n-1), d/dx(sin x) = cos x, d/dx(cos x) = -sin x, d/dx(tan x) = sec²x, d/dx(e^x) = e^x, d/dx(ln x) = 1/x, d/dx(a^x) = a^x ln(a)",
        "type": "formula",
    },
    {
        "topic": "differentiation",
        "title": "Product and Quotient Rules",
        "content": "Product: d/dx(uv) = u'v + uv'. Quotient: d/dx(u/v) = (u'v - uv')/v². Chain rule: d/dx(f(g(x))) = f'(g(x))·g'(x).",
        "type": "formula",
    },
    {
        "topic": "differentiation",
        "title": "Parametric Differentiation",
        "content": "If x = x(t), y = y(t), then dy/dx = (dy/dt)/(dx/dt). Second derivative: d²y/dx² = d/dt(dy/dx) / (dx/dt).",
        "type": "formula",
    },
    {
        "topic": "differentiation",
        "title": "Implicit Differentiation",
        "content": "For F(x,y) = 0, differentiate both sides w.r.t. x, treating y as function of x. Collect dy/dx terms and solve.",
        "type": "method",
    },
    {
        "topic": "differentiation",
        "title": "Logarithmic Differentiation",
        "content": "For y = f(x)^g(x), take ln: ln(y) = g(x)·ln(f(x)), then differentiate: y'/y = g'(x)·ln(f(x)) + g(x)·f'(x)/f(x).",
        "type": "method",
    },
    {
        "topic": "differentiation",
        "title": "Inverse Trigonometric Derivatives",
        "content": "d/dx(sin⁻¹x) = 1/√(1-x²), d/dx(cos⁻¹x) = -1/√(1-x²), d/dx(tan⁻¹x) = 1/(1+x²), d/dx(cot⁻¹x) = -1/(1+x²), d/dx(sec⁻¹x) = 1/(|x|√(x²-1)), d/dx(csc⁻¹x) = -1/(|x|√(x²-1))",
        "type": "formula",
    },
    # === INTEGRATION ===
    {
        "topic": "integration",
        "title": "Basic Integrals",
        "content": "∫x^n dx = x^(n+1)/(n+1) + C (n≠-1), ∫1/x dx = ln|x| + C, ∫e^x dx = e^x + C, ∫a^x dx = a^x/ln(a) + C, ∫sin(x) dx = -cos(x) + C, ∫cos(x) dx = sin(x) + C, ∫sec²(x) dx = tan(x) + C",
        "type": "formula",
    },
    {
        "topic": "integration",
        "title": "Integration by Parts (LIATE)",
        "content": "∫u dv = uv - ∫v du. Choose u by LIATE priority: Logarithmic, Inverse trig, Algebraic, Trigonometric, Exponential.",
        "type": "method",
    },
    {
        "topic": "integration",
        "title": "Standard Integral Forms",
        "content": "∫dx/(x²+a²) = (1/a)tan⁻¹(x/a) + C, ∫dx/√(a²-x²) = sin⁻¹(x/a) + C, ∫dx/√(x²+a²) = ln|x+√(x²+a²)| + C, ∫dx/(x²-a²) = (1/2a)ln|(x-a)/(x+a)| + C",
        "type": "formula",
    },
    {
        "topic": "integration",
        "title": "Partial Fraction Decomposition",
        "content": "For rational functions P(x)/Q(x): (1) If deg P ≥ deg Q, do polynomial division first. (2) Factor Q(x). (3) For linear factor (ax+b): A/(ax+b). (4) For repeated linear: A/(ax+b) + B/(ax+b)² + ... (5) For irreducible quadratic: (Ax+B)/(ax²+bx+c).",
        "type": "method",
    },
    {
        "topic": "definite_integrals",
        "title": "King's Property",
        "content": "∫₀^a f(x) dx = ∫₀^a f(a-x) dx. Useful when f(x) + f(a-x) is simple. Example: ∫₀^{π/2} sinⁿx/(sinⁿx+cosⁿx) dx = π/4.",
        "type": "property",
    },
    {
        "topic": "definite_integrals",
        "title": "Definite Integral Properties",
        "content": "∫ₐᵇ f(x) dx = ∫ₐᵇ f(a+b-x) dx. ∫₀^{2a} f(x) dx = 2∫₀^a f(x) dx if f(2a-x)=f(x), else 0 if f(2a-x)=-f(x). ∫_{-a}^a f(x) dx = 2∫₀^a f(x) dx if even, 0 if odd.",
        "type": "property",
    },
    # === DIFFERENTIAL EQUATIONS ===
    {
        "topic": "differential_equations",
        "title": "Separable DE",
        "content": "If dy/dx = f(x)g(y), separate: dy/g(y) = f(x)dx, then integrate both sides.",
        "type": "method",
    },
    {
        "topic": "differential_equations",
        "title": "Linear First-Order DE",
        "content": "dy/dx + P(x)y = Q(x). Integrating factor: IF = e^{∫P(x)dx}. Solution: y·IF = ∫Q(x)·IF dx + C.",
        "type": "method",
    },
    {
        "topic": "differential_equations",
        "title": "Homogeneous DE",
        "content": "If dy/dx = f(y/x), substitute y = vx, so dy/dx = v + x dv/dx. Separate and integrate.",
        "type": "method",
    },
    # === MAXIMA/MINIMA ===
    {
        "topic": "maxima_minima",
        "title": "Critical Points and Classification",
        "content": "Find f'(x) = 0 for critical points. Second derivative test: f''(c) > 0 → local min, f''(c) < 0 → local max, f''(c) = 0 → inconclusive (use first derivative test). Check endpoints for absolute extrema on closed intervals.",
        "type": "method",
    },
    {
        "topic": "maxima_minima",
        "title": "Optimization Strategy",
        "content": "1. Define objective function. 2. Use constraints to reduce variables. 3. Find critical points. 4. Verify using second derivative or boundary check. 5. Check if answer makes physical/mathematical sense.",
        "type": "method",
    },
    # === COMPLEX NUMBERS ===
    {
        "topic": "complex_numbers",
        "title": "Complex Number Properties",
        "content": "|z| = √(x²+y²), arg(z) = tan⁻¹(y/x). |z₁·z₂| = |z₁|·|z₂|, arg(z₁·z₂) = arg(z₁)+arg(z₂). |z₁/z₂| = |z₁|/|z₂|. z·z̄ = |z|². De Moivre: (cos θ + i sin θ)ⁿ = cos(nθ) + i sin(nθ).",
        "type": "formula",
    },
    {
        "topic": "complex_numbers",
        "title": "Cube Roots of Unity",
        "content": "1, ω, ω² where ω = e^(2πi/3) = (-1+i√3)/2. Properties: 1+ω+ω² = 0, ω³ = 1, ω² = ω̄, |ω| = 1. Useful for factoring x³±1 and cyclic sums.",
        "type": "formula",
    },
    # === QUADRATIC ===
    {
        "topic": "quadratic",
        "title": "Quadratic Equation Relations",
        "content": "For ax²+bx+c=0 with roots α,β: α+β = -b/a, αβ = c/a. α²+β² = (α+β)² - 2αβ. α³+β³ = (α+β)³ - 3αβ(α+β). Discriminant D = b²-4ac. D>0: real distinct, D=0: equal, D<0: complex conjugate.",
        "type": "formula",
    },
    # === PERMUTATIONS & COMBINATIONS ===
    {
        "topic": "permutations_combinations",
        "title": "P&C Formulas",
        "content": "nPr = n!/(n-r)!. nCr = n!/(r!(n-r)!). nCr = nC(n-r). nCr + nC(r-1) = (n+1)Cr. Total subsets of n elements = 2ⁿ. Circular permutations = (n-1)!. Permutations with repetition: n!/(n₁!·n₂!·...).",
        "type": "formula",
    },
    # === BINOMIAL THEOREM ===
    {
        "topic": "binomial_theorem",
        "title": "Binomial Expansion",
        "content": "(a+b)ⁿ = Σ C(n,r) a^(n-r) b^r for r=0 to n. General term: T_(r+1) = C(n,r) a^(n-r) b^r. Middle term: if n even, T_(n/2+1); if n odd, T_((n+1)/2) and T_((n+3)/2). Sum of coefficients: substitute a=b=1.",
        "type": "formula",
    },
    # === MATRICES & DETERMINANTS ===
    {
        "topic": "matrices_determinants",
        "title": "Determinant Properties",
        "content": "|Aᵀ| = |A|. |kA| = kⁿ|A| for n×n matrix. |AB| = |A||B|. |A⁻¹| = 1/|A|. If two rows/columns are identical: |A|=0. Row operation: adding multiple of one row to another doesn't change determinant. Swapping rows: sign changes.",
        "type": "formula",
    },
    {
        "topic": "matrices_determinants",
        "title": "Matrix Inverse",
        "content": "A⁻¹ = adj(A)/|A| where adj(A) is transpose of cofactor matrix. For 2×2 [[a,b],[c,d]]: A⁻¹ = (1/(ad-bc)) [[d,-b],[-c,a]]. (AB)⁻¹ = B⁻¹A⁻¹. (Aᵀ)⁻¹ = (A⁻¹)ᵀ.",
        "type": "formula",
    },
    # === PROBABILITY ===
    {
        "topic": "probability",
        "title": "Probability Rules",
        "content": "P(A∪B) = P(A) + P(B) - P(A∩B). P(A|B) = P(A∩B)/P(B). For independent events: P(A∩B) = P(A)·P(B). Bayes: P(A|B) = P(B|A)·P(A)/P(B). Total probability: P(B) = Σ P(B|Aᵢ)·P(Aᵢ).",
        "type": "formula",
    },
    # === SEQUENCES & SERIES ===
    {
        "topic": "sequences_series",
        "title": "AP and GP Formulas",
        "content": "AP: nth term = a + (n-1)d. Sum = n/2 [2a+(n-1)d] = n/2(a+l). GP: nth term = ar^(n-1). Sum = a(1-rⁿ)/(1-r) for r≠1. Infinite GP sum = a/(1-r) for |r|<1. AM ≥ GM for positive reals.",
        "type": "formula",
    },
    # === TRIGONOMETRY ===
    {
        "topic": "trigonometric_identities",
        "title": "Fundamental Identities",
        "content": "sin²x + cos²x = 1. 1 + tan²x = sec²x. 1 + cot²x = csc²x. sin(2x) = 2sinx cosx. cos(2x) = cos²x - sin²x = 2cos²x - 1 = 1 - 2sin²x. tan(2x) = 2tanx/(1-tan²x).",
        "type": "formula",
    },
    {
        "topic": "trigonometric_identities",
        "title": "Compound Angle Formulas",
        "content": "sin(A±B) = sinA cosB ± cosA sinB. cos(A±B) = cosA cosB ∓ sinA sinB. tan(A±B) = (tanA ± tanB)/(1 ∓ tanA tanB). sinA + sinB = 2sin((A+B)/2)cos((A-B)/2). cosA + cosB = 2cos((A+B)/2)cos((A-B)/2).",
        "type": "formula",
    },
    {
        "topic": "trigonometric_equations",
        "title": "General Solutions",
        "content": "sin(x) = sin(α) → x = nπ + (-1)ⁿα. cos(x) = cos(α) → x = 2nπ ± α. tan(x) = tan(α) → x = nπ + α. Principal values: sin⁻¹ in [-π/2,π/2], cos⁻¹ in [0,π], tan⁻¹ in (-π/2,π/2).",
        "type": "formula",
    },
    # === COORDINATE GEOMETRY ===
    {
        "topic": "straight_lines",
        "title": "Line Formulas",
        "content": "Distance between (x₁,y₁) and (x₂,y₂): √((x₂-x₁)²+(y₂-y₁)²). Slope m = (y₂-y₁)/(x₂-x₁). Point-slope: y-y₁ = m(x-x₁). Two-point form. Slope-intercept: y = mx+c. Distance from point (x₀,y₀) to line ax+by+c=0: |ax₀+by₀+c|/√(a²+b²).",
        "type": "formula",
    },
    {
        "topic": "circles",
        "title": "Circle Properties",
        "content": "Standard form: (x-h)² + (y-k)² = r². General: x²+y²+2gx+2fy+c=0, center (-g,-f), radius √(g²+f²-c). Tangent at (x₁,y₁): xx₁+yy₁+g(x+x₁)+f(y+y₁)+c=0. Length of tangent from (x₁,y₁): √(S₁₁). Common chord: S₁-S₂=0.",
        "type": "formula",
    },
    {
        "topic": "parabola",
        "title": "Parabola Properties",
        "content": "y² = 4ax: vertex (0,0), focus (a,0), directrix x=-a, latus rectum = 4a. Parametric: (at², 2at). Tangent at (x₁,y₁): yy₁ = 2a(x+x₁). Normal: y = mx - 2am - am³.",
        "type": "formula",
    },
    {
        "topic": "ellipse",
        "title": "Ellipse Properties",
        "content": "x²/a² + y²/b² = 1 (a>b): center (0,0), foci (±c,0) where c²=a²-b², eccentricity e=c/a, latus rectum = 2b²/a. Major axis = 2a, minor axis = 2b. Sum of focal distances = 2a.",
        "type": "formula",
    },
    {
        "topic": "hyperbola",
        "title": "Hyperbola Properties",
        "content": "x²/a² - y²/b² = 1: center (0,0), foci (±c,0) where c²=a²+b², eccentricity e=c/a, latus rectum = 2b²/a. Asymptotes: y = ±(b/a)x. Difference of focal distances = 2a. Rectangular hyperbola: a=b, e=√2, asymptotes perpendicular.",
        "type": "formula",
    },
    # === VECTORS ===
    {
        "topic": "vectors",
        "title": "Vector Operations",
        "content": "Dot product: a·b = |a||b|cosθ = a₁b₁+a₂b₂+a₃b₃. Cross product: |a×b| = |a||b|sinθ. Scalar triple product: [a b c] = a·(b×c) = determinant. Vector triple product: a×(b×c) = (a·c)b - (a·b)c.",
        "type": "formula",
    },
    {
        "topic": "vectors",
        "title": "Vector Properties",
        "content": "|a+b|² = |a|² + |b|² + 2a·b. |a-b|² = |a|² + |b|² - 2a·b. If a⊥b then a·b=0. If a∥b then a×b=0. Unit vector: â = a/|a|. Projection of a on b: (a·b)/|b|.",
        "type": "formula",
    },
    # === 3D GEOMETRY ===
    {
        "topic": "three_d_geometry",
        "title": "3D Geometry Formulas",
        "content": "Distance from (x₀,y₀,z₀) to plane ax+by+cz+d=0: |ax₀+by₀+cz₀+d|/√(a²+b²+c²). Angle between planes = angle between normals. Direction cosines: l²+m²+n²=1. Shortest distance between skew lines: |(a₂-a₁)·(b₁×b₂)|/|b₁×b₂|.",
        "type": "formula",
    },
    # === STATISTICS ===
    {
        "topic": "statistics",
        "title": "Statistics Formulas",
        "content": "Mean = Σxᵢ/n. Variance = Σ(xᵢ-mean)²/n = (Σxᵢ²)/n - mean². SD = √variance. For grouped data: mean = Σfᵢxᵢ/Σfᵢ. Combined mean = (n₁x̄₁+n₂x̄₂)/(n₁+n₂). Correlation: r = cov(X,Y)/(σₓσᵧ).",
        "type": "formula",
    },
    # === COMMON PITFALLS ===
    {
        "topic": "limits",
        "title": "Common Limit Mistakes",
        "content": "Mistake 1: Applying standard limit sin(x)/x = 1 when x is in degrees (must be radians). Mistake 2: Using L'Hopital's rule without checking 0/0 or ∞/∞ form. Mistake 3: Forgetting that lim(x→0) |x|/x does not exist (left ≠ right). Mistake 4: Incorrectly expanding to insufficient order in Taylor series.",
        "type": "pitfall",
    },
    {
        "topic": "integration",
        "title": "Common Integration Mistakes",
        "content": "Mistake 1: Forgetting +C for indefinite integrals. Mistake 2: Wrong substitution choice leading to more complex integral. Mistake 3: Not checking if integrand has singularities in definite integral bounds. Mistake 4: Incorrectly applying integration by parts with wrong u/dv choice.",
        "type": "pitfall",
    },
    {
        "topic": "differentiation",
        "title": "Common Differentiation Mistakes",
        "content": "Mistake 1: Forgetting chain rule for composite functions. Mistake 2: Incorrect sign in derivative of cos(x), cot(x), csc(x). Mistake 3: Not simplifying before differentiating (e.g., ln((1+x)/(1-x)) should be simplified first). Mistake 4: Wrong assumption about differentiability at sharp corners or cusps.",
        "type": "pitfall",
    },
    {
        "topic": "complex_numbers",
        "title": "Common Complex Number Mistakes",
        "content": "Mistake 1: Assuming √(z₁z₂) = √z₁·√z₂ for complex numbers (false in general). Mistake 2: Forgetting that |z|² = z·z̄, not z². Mistake 3: Incorrectly computing argument without checking quadrant. Mistake 4: Confusing |z₁+z₂| with |z₁|+|z₂| (triangle inequality: ≤, not =).",
        "type": "pitfall",
    },
    {
        "topic": "probability",
        "title": "Common Probability Mistakes",
        "content": "Mistake 1: Treating dependent events as independent. Mistake 2: Forgetting to subtract P(A∩B) in P(A∪B). Mistake 3: Confusing P(A|B) with P(B|A). Mistake 4: Not adjusting for 'without replacement' in drawing problems. Mistake 5: Double-counting outcomes in permutation problems.",
        "type": "pitfall",
    },
    {
        "topic": "matrices_determinants",
        "title": "Common Matrix Mistakes",
        "content": "Mistake 1: Assuming (A+B)⁻¹ = A⁻¹ + B⁻¹ (false). Mistake 2: Forgetting |kA| = kⁿ|A|, not k|A|. Mistake 3: Matrix multiplication is not commutative: AB ≠ BA in general. Mistake 4: Not checking |A|≠0 before computing A⁻¹. Mistake 5: Confusing adj(A) with Aᵀ.",
        "type": "pitfall",
    },
]


# ---------------------------------------------------------------------------
# JEE Solution Templates by Problem Type
# ---------------------------------------------------------------------------

JEE_TEMPLATES: list[dict] = [
    {
        "problem_type": "limits",
        "template": "1. Identify the form (0/0, ∞/∞, 1^∞, 0·∞, etc.)\n2. For 0/0 or ∞/∞: Apply L'Hopital's rule or factor/expand\n3. For 1^∞: Rewrite as e^{lim g(x)(f(x)-1)}\n4. For trigonometric: Use standard limits or Taylor expansion\n5. Verify by numerical substitution",
    },
    {
        "problem_type": "differentiation",
        "template": "1. Identify function type (explicit, implicit, parametric, logarithmic)\n2. Apply appropriate rule (chain, product, quotient)\n3. For inverse trig: Consider substitution to simplify\n4. Simplify the derivative expression\n5. Substitute the given point if required",
    },
    {
        "problem_type": "integration",
        "template": "1. Identify integral type (standard form, by parts, substitution, partial fractions)\n2. For algebraic: Try completing square or substitution\n3. For trigonometric: Use identities or universal substitution t=tan(x/2)\n4. For product of functions: Consider integration by parts (LIATE)\n5. Verify by differentiating the result",
    },
    {
        "problem_type": "definite_integrals",
        "template": "1. Check for King's property: ∫₀^a f(x) = ∫₀^a f(a-x)\n2. Check symmetry: even/odd properties about origin or midpoint\n3. For trigonometric integrals 0 to π/2: Check if f(sin,cos) = f(cos,sin)\n4. Evaluate using antiderivative or properties\n5. Verify result is physically reasonable (area ≥ 0 for positive functions)",
    },
    {
        "problem_type": "differential_equations",
        "template": "1. Identify DE type (separable, linear, homogeneous, exact, Bernoulli)\n2. For linear: Find integrating factor IF = e^{∫P dx}\n3. For homogeneous: Substitute y = vx\n4. Apply initial conditions to find particular solution\n5. Verify by substituting back into original DE",
    },
    {
        "problem_type": "maxima_minima",
        "template": "1. Find f'(x) and solve f'(x) = 0 for critical points\n2. Classify using second derivative test or sign analysis\n3. Check boundary points if interval is closed\n4. Compare function values at critical and boundary points\n5. Verify answer makes sense (e.g., maxima should be larger than nearby values)",
    },
    {
        "problem_type": "complex_numbers",
        "template": "1. Convert to appropriate form (rectangular, polar, exponential)\n2. For modulus/argument: Use polar form z = r(cos θ + i sin θ)\n3. For equations: Equate real and imaginary parts separately\n4. Use properties: |z|² = z·z̄, arg(z₁z₂) = arg(z₁)+arg(z₂)\n5. Check quadrant for correct argument value",
    },
    {
        "problem_type": "quadratic",
        "template": "1. Identify what is given (roots, coefficients, conditions)\n2. Use Vieta's formulas: α+β = -b/a, αβ = c/a\n3. For nature of roots: Compute discriminant D = b²-4ac\n4. For conditions on roots: Express in terms of sum and product\n5. Verify by substituting back",
    },
    {
        "problem_type": "permutations_combinations",
        "template": "1. Identify if order matters (permutation) or not (combination)\n2. Check for restrictions (certain positions, adjacency, separation)\n3. For circular: Use (n-1)! for arrangements around a circle\n4. For repetition: Divide by factorial of repeated items\n5. Use complementary counting when 'at least' or 'at most' appears",
    },
    {
        "problem_type": "binomial_theorem",
        "template": "1. Identify what to find (specific term, coefficient, sum)\n2. General term: T_{r+1} = C(n,r) a^{n-r} b^r\n3. For independent term: Set power of x to 0\n4. For sum of coefficients: Substitute x=1 (and sometimes x=-1)\n5. For coefficient relations: Write two general terms and equate",
    },
    {
        "problem_type": "matrices_determinants",
        "template": "1. Identify operation needed (determinant, inverse, solve system, rank)\n2. For determinant: Use properties or row reduction\n3. For inverse: Check |A|≠0, then A⁻¹ = adj(A)/|A|\n4. For system: Use Cramer's rule, matrix inversion, or row reduction\n5. Verify by substituting solution back into equations",
    },
    {
        "problem_type": "probability",
        "template": "1. Define events clearly\n2. Identify if events are independent, mutually exclusive, or dependent\n3. Use appropriate formula: P(A∪B), P(A|B), Bayes' theorem\n4. For counting: Use nCr or nPr as appropriate\n5. Verify probabilities sum to 1 where applicable",
    },
    {
        "problem_type": "sequences_series",
        "template": "1. Identify sequence type (AP, GP, HP, or mixed)\n2. For AP: Use a, d, n formulas. For GP: Use a, r, n formulas\n3. For summation: Split into known sums (Σr, Σr², Σr³)\n4. For infinite series: Check |r|<1 for convergence\n5. Verify first few terms match given information",
    },
    {
        "problem_type": "trigonometric_identities",
        "template": "1. Start from one side (usually more complex)\n2. Express everything in terms of sin and cos if stuck\n3. Use fundamental identities: sin²+cos²=1, 1+tan²=sec²\n4. Factor or combine fractions as needed\n5. Verify with specific angle value",
    },
    {
        "problem_type": "trigonometric_equations",
        "template": "1. Simplify using identities to single trig function if possible\n2. Factor the equation (quadratic in sin, cos, tan, etc.)\n3. Find principal solutions first\n4. Write general solution using n\n5. Check if solutions are within given interval",
    },
    {
        "problem_type": "inverse_trig",
        "template": "1. Check principal value range for each inverse trig function\n2. Use substitution: let θ = sin⁻¹(x), then sin(θ) = x\n3. Draw right triangle to find other trig ratios\n4. For sums: Use tan⁻¹(x) ± tan⁻¹(y) = tan⁻¹((x±y)/(1∓xy)) with quadrant check\n5. Verify result is in correct principal range",
    },
    {
        "problem_type": "straight_lines",
        "template": "1. Identify given information (points, slope, intercepts, angle)\n2. Choose appropriate form: point-slope, two-point, slope-intercept\n3. For distance/angle problems: Use standard formulas\n4. For locus problems: Use geometric properties\n5. Verify by checking if given points satisfy the equation",
    },
    {
        "problem_type": "circles",
        "template": "1. Identify given information (center, radius, points, tangents)\n2. Use standard form or general form as appropriate\n3. For tangents: Distance from center to line = radius\n4. For common chord: S₁ - S₂ = 0\n5. Verify by substituting points back into equation",
    },
    {
        "problem_type": "vectors",
        "template": "1. Write vectors in component form\n2. Identify operation needed (dot, cross, triple product, angle)\n3. Apply formula component-wise\n4. For geometric conditions: Translate to algebraic equations\n5. Verify magnitude and direction make sense",
    },
    {
        "problem_type": "three_d_geometry",
        "template": "1. Identify given information (points, planes, lines, directions)\n2. For plane: Use point-normal form or intercept form\n3. For line: Use symmetric or parametric form\n4. For distances/angles: Apply standard formulas\n5. Verify by substituting coordinates back",
    },
]


# ---------------------------------------------------------------------------
# RAG Knowledge Base Class
# ---------------------------------------------------------------------------

@dataclass
class RAGConfig:
    """Configuration for the RAG knowledge base."""
    embedding_model: str = "all-MiniLM-L6-v2"
    collection_name: str = "jee_knowledge"
    persist_directory: str = field(default_factory=lambda: os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "chroma_db"
    ))
    top_k: int = 3


class JEEKnowledgeBase:
    """RAG Knowledge Base for JEE Mathematics.
    
    Provides semantic retrieval of:
    - Similar solved problems
    - Relevant formulas and theorems
    - Common pitfalls for the topic
    - Solution templates
    
    Usage:
        kb = JEEKnowledgeBase(config)
        kb.build(problems_list)  # Build from JEE problem dataset
        
        # Retrieve context for a new problem
        context = kb.retrieve("Evaluate lim x->0 sin(3x)/x")
        # Returns top-3 documents with similarity scores
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        self.config = config or RAGConfig()
        self._embedding_model: Optional[SentenceTransformer] = None
        self._chroma_client: Optional[chromadb.Client] = None
        self._collection = None
        self._is_available = _HAS_SENTENCE_TRANSFORMERS and _HAS_CHROMADB
        
        if not self._is_available:
            missing = []
            if not _HAS_SENTENCE_TRANSFORMERS:
                missing.append("sentence-transformers")
            if not _HAS_CHROMADB:
                missing.append("chromadb")
            print(f"[RAG] Warning: Missing dependencies {missing}. RAG will return empty results.")
            return
        
        self._init_embedding_model()
        self._init_chroma()
    
    def _init_embedding_model(self):
        """Initialize the sentence transformer embedding model."""
        try:
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            print(f"[RAG] Loaded embedding model: {self.config.embedding_model}")
        except Exception as e:
            print(f"[RAG] Failed to load embedding model: {e}")
            self._is_available = False
    
    def _init_chroma(self):
        """Initialize ChromaDB client and collection."""
        try:
            os.makedirs(self.config.persist_directory, exist_ok=True)
            self._chroma_client = chromadb.Client(Settings(
                persist_directory=self.config.persist_directory,
                anonymized_telemetry=False,
                is_persistent=True,
            ))
            
            # Get or create collection
            self._collection = self._chroma_client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"description": "JEE Mathematics knowledge base"}
            )
            print(f"[RAG] ChromaDB collection ready: {self.config.collection_name}")
        except Exception as e:
            print(f"[RAG] Failed to initialize ChromaDB: {e}")
            self._is_available = False
    
    def build(self, problems: list[dict]) -> "JEEKnowledgeBase":
        """Build the knowledge base from JEE problems, formulas, and templates.
        
        Args:
            problems: List of problem dicts with 'question', 'problem_type', etc.
            
        Returns:
            Self for chaining.
        """
        if not self._is_available:
            print("[RAG] Knowledge base not available, skipping build.")
            return self
        
        # Check if already populated
        count = self._collection.count()
        if count > 0:
            print(f"[RAG] Collection already has {count} documents. Use rebuild() to reset.")
            return self
        
        print(f"[RAG] Building knowledge base...")
        
        documents = []
        metadatas = []
        ids = []
        
        # 1. Add JEE problems
        for i, prob in enumerate(problems):
            doc_text = self._format_problem_document(prob)
            documents.append(doc_text)
            metadatas.append({
                "source": "jee_problem",
                "problem_type": prob.get("problem_type", ""),
                "topic": prob.get("topic", ""),
                "difficulty": prob.get("difficulty", ""),
                "solution_method": prob.get("solution_method", ""),
            })
            ids.append(f"problem_{i}")
        
        # 2. Add formulas/theorems
        for i, formula in enumerate(JEE_FORMULAS):
            doc_text = self._format_formula_document(formula)
            documents.append(doc_text)
            metadatas.append({
                "source": "formula",
                "problem_type": formula.get("topic", ""),
                "topic": formula.get("title", ""),
                "doc_type": formula.get("type", ""),
            })
            ids.append(f"formula_{i}")
        
        # 3. Add solution templates
        for i, template in enumerate(JEE_TEMPLATES):
            doc_text = self._format_template_document(template)
            documents.append(doc_text)
            metadatas.append({
                "source": "template",
                "problem_type": template.get("problem_type", ""),
                "topic": "solution_template",
            })
            ids.append(f"template_{i}")
        
        # Batch add to collection
        batch_size = 100
        for start in range(0, len(documents), batch_size):
            end = min(start + batch_size, len(documents))
            self._collection.add(
                documents=documents[start:end],
                metadatas=metadatas[start:end],
                ids=ids[start:end],
            )
        
        print(f"[RAG] Added {len(documents)} documents to knowledge base.")
        return self
    
    def rebuild(self, problems: list[dict]) -> "JEEKnowledgeBase":
        """Clear and rebuild the knowledge base."""
        if not self._is_available:
            return self
        
        # Delete and recreate collection
        try:
            self._chroma_client.delete_collection(self.config.collection_name)
        except Exception:
            pass
        
        self._collection = self._chroma_client.create_collection(
            name=self.config.collection_name,
            metadata={"description": "JEE Mathematics knowledge base"}
        )
        
        return self.build(problems)
    
    def retrieve(self, query: str, top_k: Optional[int] = None, 
                 filter_by_type: Optional[str] = None) -> dict:
        """Retrieve top-k relevant documents for a query.
        
        Args:
            query: The problem text or query string.
            top_k: Number of results to return (default: config.top_k = 3).
            filter_by_type: Optional filter by problem_type metadata.
            
        Returns:
            Dictionary with:
                - "documents": list of retrieved document texts
                - "metadatas": list of metadata dicts
                - "distances": list of similarity distances (lower = more similar)
                - "formatted_context": string with top-3 results formatted for LLM prompt
        """
        if not self._is_available or self._collection is None:
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "formatted_context": "",
            }
        
        top_k = top_k or self.config.top_k
        
        # Build where filter if needed
        where_filter = None
        if filter_by_type:
            where_filter = {"problem_type": filter_by_type}
        
        try:
            results = self._collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
            
            documents = results.get("documents", [[]])[0]
            metadatas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]
            
            formatted = self._format_retrieved_context(documents, metadatas, distances)
            
            return {
                "documents": documents,
                "metadatas": metadatas,
                "distances": distances,
                "formatted_context": formatted,
            }
        except Exception as e:
            print(f"[RAG] Retrieval error: {e}")
            return {
                "documents": [],
                "metadatas": [],
                "distances": [],
                "formatted_context": "",
            }
    
    def retrieve_by_problem_type(self, query: str, problem_type: str, 
                                  top_k: Optional[int] = None) -> dict:
        """Retrieve documents filtered by problem type.
        
        This is useful when the problem type is already known from analysis.
        """
        return self.retrieve(query, top_k=top_k, filter_by_type=problem_type)
    
    def _format_problem_document(self, prob: dict) -> str:
        """Format a problem dict into a document string for embedding."""
        parts = [
            f"Problem Type: {prob.get('problem_type', '')}",
            f"Topic: {prob.get('topic', '')}",
            f"Difficulty: {prob.get('difficulty', '')}",
            f"Question: {prob.get('question', '')}",
            f"Solution Method: {prob.get('solution_method', '')}",
            f"Answer: {prob.get('answer', '')}",
        ]
        return "\n".join(parts)
    
    def _format_formula_document(self, formula: dict) -> str:
        """Format a formula dict into a document string."""
        parts = [
            f"Topic: {formula.get('topic', '')}",
            f"Title: {formula.get('title', '')}",
            f"Type: {formula.get('type', '')}",
            f"Content: {formula.get('content', '')}",
        ]
        return "\n".join(parts)
    
    def _format_template_document(self, template: dict) -> str:
        """Format a solution template into a document string."""
        parts = [
            f"Problem Type: {template.get('problem_type', '')}",
            f"Solution Template:\n{template.get('template', '')}",
        ]
        return "\n".join(parts)
    
    def _format_retrieved_context(self, documents: list[str], 
                                   metadatas: list[dict],
                                   distances: list[float]) -> str:
        """Format retrieved documents into a context string for LLM prompts."""
        if not documents:
            return ""
        
        sections = []
        
        # Group by source type
        problems = []
        formulas = []
        templates = []
        
        for doc, meta, dist in zip(documents, metadatas, distances):
            source = meta.get("source", "unknown")
            if source == "jee_problem":
                problems.append((doc, meta, dist))
            elif source == "formula":
                formulas.append((doc, meta, dist))
            elif source == "template":
                templates.append((doc, meta, dist))
        
        # Helper to extract multi-line field values from a formatted problem doc
        def _extract_problem_fields(doc: str) -> dict:
            """Extract Question, Solution Method, and Answer from a problem document.
            
            Handles multi-line values by finding section headers and capturing everything
            between them.
            """
            lines = doc.split("\n")
            fields = {"question": "", "solution_method": "", "answer": "", 
                      "problem_type": "", "topic": "", "difficulty": ""}
            current_field = None
            buffer = []
            
            for line in lines:
                if line.startswith("Problem Type:"):
                    if current_field:
                        fields[current_field] = "\n".join(buffer).strip()
                    fields["problem_type"] = line[len("Problem Type:"):].strip()
                    current_field = None
                    buffer = []
                elif line.startswith("Topic:"):
                    if current_field:
                        fields[current_field] = "\n".join(buffer).strip()
                    fields["topic"] = line[len("Topic:"):].strip()
                    current_field = None
                    buffer = []
                elif line.startswith("Difficulty:"):
                    if current_field:
                        fields[current_field] = "\n".join(buffer).strip()
                    fields["difficulty"] = line[len("Difficulty:"):].strip()
                    current_field = None
                    buffer = []
                elif line.startswith("Question:"):
                    if current_field:
                        fields[current_field] = "\n".join(buffer).strip()
                    current_field = "question"
                    buffer = [line[len("Question:"):].strip()]
                elif line.startswith("Solution Method:"):
                    if current_field:
                        fields[current_field] = "\n".join(buffer).strip()
                    current_field = "solution_method"
                    buffer = [line[len("Solution Method:"):].strip()]
                elif line.startswith("Answer:"):
                    if current_field:
                        fields[current_field] = "\n".join(buffer).strip()
                    current_field = "answer"
                    buffer = [line[len("Answer:"):].strip()]
                elif current_field:
                    buffer.append(line)
            
            if current_field:
                fields[current_field] = "\n".join(buffer).strip()
            
            return fields
        
        # Format similar problems section
        if problems:
            sections.append("=== SIMILAR SOLVED PROBLEMS ===")
            for i, (doc, meta, dist) in enumerate(problems[:2], 1):
                fields = _extract_problem_fields(doc)
                sections.append(f"\nSimilar Problem {i}:")
                if fields["question"]:
                    sections.append(f"Question: {fields['question']}")
                if fields["solution_method"]:
                    # Only include full solution for the FIRST (best) match
                    if i == 1:
                        sections.append(f"\nFull Solution:\n{fields['solution_method']}")
                    else:
                        # For secondary matches, include first 200 chars as preview
                        preview = fields["solution_method"][:200].strip()
                        if len(fields["solution_method"]) > 200:
                            preview += " ..."
                        sections.append(f"Solution Preview: {preview}")
                if fields["answer"] and fields["answer"].lower() not in ("", "none", "null"):
                    sections.append(f"Answer: {fields['answer']}")
        
        # Format relevant formulas section
        if formulas:
            sections.append("\n=== RELEVANT FORMULAS & THEOREMS ===")
            for doc, meta, dist in formulas[:2]:
                lines = doc.split("\n")
                title_line = [l for l in lines if l.startswith("Title:")]
                content_line = [l for l in lines if l.startswith("Content:")]
                if title_line:
                    sections.append(f"\n{title_line[0].replace('Title: ', '')}:")
                if content_line:
                    sections.append(content_line[0].replace("Content: ", ""))
        
        # Format solution template
        if templates:
            sections.append("\n=== SOLUTION APPROACH TEMPLATE ===")
            doc, meta, dist = templates[0]
            template_text = doc.split("Solution Template:")[-1].strip()
            sections.append(template_text)
        
        return "\n".join(sections)
    
    @property
    def is_available(self) -> bool:
        """Check if RAG is available and initialized."""
        return self._is_available and self._collection is not None
    
    def get_stats(self) -> dict:
        """Return statistics about the knowledge base."""
        if not self._is_available or self._collection is None:
            return {"available": False, "count": 0}
        
        return {
            "available": True,
            "count": self._collection.count(),
            "embedding_model": self.config.embedding_model,
            "collection_name": self.config.collection_name,
        }


# ---------------------------------------------------------------------------
# Singleton instance for reuse
# ---------------------------------------------------------------------------

_knowledge_base_instance: Optional[JEEKnowledgeBase] = None

def get_knowledge_base(config: Optional[RAGConfig] = None, 
                        problems: Optional[list[dict]] = None) -> JEEKnowledgeBase:
    """Get or create the singleton knowledge base instance.
    
    Args:
        config: Optional RAGConfig override.
        problems: Optional list of problems to build from. If None and KB
                  is not built, it will be empty until build() is called.
    
    Returns:
        JEEKnowledgeBase instance.
    """
    global _knowledge_base_instance
    
    if _knowledge_base_instance is None:
        _knowledge_base_instance = JEEKnowledgeBase(config)
        if problems:
            _knowledge_base_instance.build(problems)
    
    return _knowledge_base_instance


def reset_knowledge_base():
    """Reset the singleton instance."""
    global _knowledge_base_instance
    _knowledge_base_instance = None
