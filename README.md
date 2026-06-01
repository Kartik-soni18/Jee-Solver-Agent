# JEE Advanced Math Solver

A LangGraph-powered AI agent that solves JEE Advanced mathematics problems using multi-step symbolic reasoning with SymPy verification.

## Features

- **Structured Agent vs Raw LLM comparison** — see how decomposed reasoning + SymPy execution outperforms a plain LLM
- **3 curated example problems** — click to auto-fill and watch the agent solve them
- **Custom problem input** — paste any JEE problem, with or without multiple-choice options
- **Internal reasoning logs** — inspect the agent's analyze → plan → solve → verify pipeline
- **SymPy sandboxed execution** — all symbolic math runs safely in a restricted environment

## Quick Start

### 1. Set your Together AI API key

```bash
export TOGETHER_API_KEY="your-key-here"
```

### 2. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Launch the Streamlit app

```bash
streamlit run interface/jee_app.py
```

Then open http://localhost:8501.

## How It Works

The **JEEAgent** follows a decomposed reasoning pipeline:

```
analyze → plan → solve_symbolic (self-loop per step) → consolidate → verify → format → END
                          ↑___________|
                          └ reflect ←─┘ (on failure, up to 3 retries)
```

| Stage | What Happens |
|-------|--------------|
| **Analyze** | Classifies problem type (limits, differentiation, integration, etc.) |
| **Plan** | Generates a step-by-step SymPy execution plan |
| **Solve Symbolic** | Executes each planned step iteratively |
| **Consolidate** | Merges step results into a final answer |
| **Verify** | LLM checks correctness and assigns confidence |
| **Reflect** | On failure, analyzes errors and regenerates corrected code |
| **Format** | Produces LaTeX + plain-text final answer |

The **Raw LLM** simply sends the problem to the model with a "return only the answer" prompt — no tooling, no verification.

## Project Structure

```
.
├── config.py              # Together AI configuration
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── AGENTS.md              # Agent guide for AI coding assistants
├── data/                  # Result storage (JSON)
├── agent/                 # Core JEE agent
│   ├── jee_graph.py       # JEEAgent LangGraph builder
│   ├── jee_nodes.py       # Graph node functions
│   ├── jee_prompts.py     # Prompt templates
│   ├── jee_state.py       # State TypedDict
│   └── sympy_tools.py     # SymPy sandboxed execution toolkit
├── benchmark/             # Dataset & evaluation
│   ├── jee_loader.py      # ~140 embedded JEE problems
│   └── jee_evaluator.py   # Agent vs raw LLM evaluator
└── interface/
    └── jee_app.py         # Streamlit web UI
```

## Requirements

- Python 3.10+
- Together AI API key
- See `requirements.txt` for full dependency list

## License

MIT
