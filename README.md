# JEE Advanced Math Reasoning Agent

> A production-grade, LangGraph-based math reasoning agent for JEE Advanced calculus problems. Uses multi-step symbolic reasoning, sandboxed SymPy execution, and self-verification with reflection to outperform raw LLMs.

---

## Architecture

The agent follows a **decomposed reasoning pipeline** built with LangGraph:

```
          ┌──────────┐
          │  Analyze │  ← Identify problem type and key concepts
          └────┬─────┘
               │
          ┌────▼─────┐
          │   Plan   │  ← Build step-by-step solution plan
          └────┬─────┘
               │
          ┌────▼───────────┐
          │ Solve Symbolic │  ← Execute planned steps with SymPy
          └────┬───────────┘
               │
          ┌────▼───────────┐
          │  Consolidate   │  ← Merge step results into final answer
          └────┬───────────┘
               │
          ┌────▼─────┐
          │  Verify  │  ← LLM verifies correctness and flags issues
          └────┬─────┘
               │
         ┌─────┴─────┐
    ┌────┤  Correct? ├────┐
    │    └───────────┘    │
    │ Yes                 │ No
    │                     │
┌───▼────┐          ┌────▼────┐
│ Format │          │ Reflect │  ← Analyze errors, correct code
└────────┘          └────┬────┘
                         │
                    ┌────▼───────────┐
                    │ Solve Symbolic │  ← Retry with corrected plan
                    └────────────────┘
```

## Key Features

1. **Multi-Step Symbolic Reasoning** -- Decomposes calculus problems into explicit analytical steps.
2. **Sandboxed Code Execution** -- Python/SymPy code runs in a restricted environment with timeout protection and module whitelisting.
3. **Self-Verification** -- An LLM verifies each solution for correctness, confidence, and identifies specific issues.
4. **Automatic Reflection** -- On verification failure, the agent analyzes errors and regenerates corrected code (up to configurable max retries).
5. **JEE Advanced Benchmarking** -- Built-in evaluation against a curated JEE problem bank with per-type and per-difficulty metrics.
6. **Interactive Streamlit UI** -- Three-tab interface: single problem testing, benchmark runner, and results dashboard with Plotly visualizations.
7. **Zero-Database Design** -- All results stored as JSON files; no database setup required.

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd math-agent

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Environment Setup

Set your API key as an environment variable:

```bash
export LLM_API_KEY="your-api-key"
export LLM_BASE_URL="https://api.openai.com/v1"  # optional
export LLM_MODEL="gpt-4o-mini"  # optional
```

Or provide it in the Streamlit sidebar.

### CLI Usage

```bash
# Solve a single problem
python run.py --problem "Evaluate: lim x->0 sin(3x)/x"

# Run JEE benchmark (20 problems, compare mode)
python run.py --jee --count 20 --mode both

# Filter by problem type and difficulty
python run.py --jee --jee-type limits --jee-difficulty medium

# Compare latest two benchmark runs
python run.py --compare

# Launch Streamlit dashboard
python run.py --dashboard
```

### Streamlit Dashboard

```bash
streamlit run interface/jee_app.py
```

Then open http://localhost:8501 in your browser.

## How the Agent Outperforms Raw LLM

| Capability | Raw LLM | Math Agent |
|---|---|---|
| **Symbolic Computation** | Hallucinates algebra/calculus | Executes SymPy code for exact results |
| **Multi-Step Problems** | Forgets intermediate values, loses track | Maintains explicit reasoning trace with planned steps |
| **Self-Correction** | Single-shot, no retry | Verifies + reflects up to N times |
| **Structured Output** | Free-form text, hard to parse | JSON-structured reasoning and answers |
| **Code Execution** | Cannot run code | Sandboxed Python REPL with timeout |
| **Error Recovery** | No recovery mechanism | Automatic reflection and code correction |

## Tech Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph |
| LLM Integration | LangChain + ChatOpenAI |
| Code Execution | Sandboxed Python REPL (threading + timeout) |
| Symbolic Math | SymPy |
| Benchmark Dataset | Curated JEE Advanced problem bank |
| Web UI | Streamlit |
| Visualization | Plotly |
| Evaluation | Custom accuracy metrics with fuzzy matching |
| Data Storage | JSON files (zero database) |

## Project Structure

```
math-agent/
  config.py                  # Central configuration with env var support
  requirements.txt           # Python dependencies
  run.py                     # CLI entry point with argparse
  README.md                  # This file
  .gitignore                 # Standard Python gitignore
  data/                      # Benchmark results storage
    .gitkeep
  agent/                     # LangGraph agent implementation
    __init__.py
    state.py                 # TypedDict state definitions
    tools.py                 # Sandboxed Python REPL
    prompts.py               # LLM prompt templates
    nodes.py                 # Graph node functions
    graph.py                 # LangGraph builder and MathAgent class
    jee_state.py             # JEE-specific state definitions
    jee_nodes.py             # JEE-specific node functions
    jee_graph.py             # JEE agent graph builder
    jee_prompts.py           # JEE-specific prompts
    sympy_tools.py           # SymPy execution utilities
  benchmark/                 # JEE benchmarking suite
    __init__.py
    jee_loader.py            # JEE problem bank loader
    jee_evaluator.py         # Benchmark runner and evaluator
    results.py               # JSON-based result storage
    comparison.py            # Agent vs LLM comparison analyzer
  interface/                 # Streamlit web interface
    __init__.py
    jee_app.py               # Three-tab Streamlit app
```

## Future Improvements

- [ ] Expand problem bank to additional JEE topics (algebra, coordinate geometry)
- [ ] Add support for parallel problem solving with async execution
- [ ] Implement human-in-the-loop feedback for incorrect solutions
- [ ] Add tool calling for Wolfram Alpha / calculator APIs
- [ ] Fine-tune a small model specifically for the reflection step
- [ ] Cache LLM calls to reduce API costs during repeated runs
- [ ] Deploy as a containerized service with REST API endpoints
