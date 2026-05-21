"""Prompt templates for the math reasoning agent."""

SYSTEM_PROMPT: str = (
    "You are an expert mathematical problem solver. You think step by step and use Python code for precise calculations."
)

THINK_PROMPT: str = """You are solving a math problem. Think step by step and write Python code to compute the final answer.

**Problem:** {problem}

Instructions:
1. Break down the problem into clear reasoning steps.
2. Write Python code that prints the final answer.
3. The code MUST include a `print()` statement with the final numerical answer.
4. Use only basic math operations and the math module if needed.

Return ONLY valid JSON with exactly these keys:
- "thought": A string with your step-by-step reasoning.
- "code": A string with the Python code (the code MUST print the final answer).

Return ONLY valid JSON. No markdown, no explanations outside JSON."""

VERIFY_PROMPT: str = """Verify the correctness of this math problem solution.

**Problem:** {problem}

**Reasoning:** {thought}

**Code:**
```python
{code}
```

**Code Execution Result:** {code_result}

Instructions:
1. Check if the reasoning is sound and the code correctly implements the solution.
2. Verify that the final answer matches what the problem asks for.
3. Check for any calculation errors or logical mistakes.

Return ONLY valid JSON with exactly these keys:
- "is_correct": A boolean (true if the solution is correct, false otherwise).
- "confidence": A float between 0 and 1 indicating your confidence.
- "issues": A list of strings describing any issues found (empty list if correct).

Return ONLY valid JSON. No markdown, no explanations outside JSON."""

REFLECT_PROMPT: str = """The previous solution had issues. Please fix it.

**Problem:** {problem}

**Previous Reasoning:** {thought}

**Previous Code:**
```python
{code}
```

**Verification Issues:** {issues}

Instructions:
1. Analyze what went wrong based on the verification issues.
2. Provide corrected reasoning and Python code.
3. The corrected code MUST print the final answer.

Return ONLY valid JSON with exactly these keys:
- "reflection": A string explaining what went wrong and how you fixed it.
- "corrected_code": A string with the corrected Python code (MUST print the final answer).

Return ONLY valid JSON. No markdown, no explanations outside JSON."""

EXTRACT_PROMPT: str = """Extract the final numerical answer from the following text. The text is the result of executing Python code.

**Text:** {text}

**Problem:** {problem}

Instructions:
1. Find the final numerical answer in the text.
2. Return only the number (integer, decimal, or fraction).
3. If there are multiple numbers, return the one that answers the problem.
4. Clean the answer: remove units, dollar signs, commas, etc.

Return ONLY valid JSON with exactly this key:
- "answer": A string containing ONLY the clean numerical answer (e.g., "42", "3.14", "1080").

Return ONLY valid JSON. No markdown, no explanations outside JSON."""
