# JEE Advanced Math Reasoning Agent — Agent Guide

> This file is intended for AI coding agents. It describes the project architecture, conventions, and workflows.

---

## 1. Project Overview

A **LangGraph-based AI agent** that solves JEE Advanced mathematics problems. It outperforms raw LLMs by combining:

- **Multi-step symbolic reasoning** — decomposes problems into explicit analytical stages
- **Sandboxed SymPy execution** — runs Python/SymPy code in a restricted environment with timeout protection
- **Self-verification with reflection** — an LLM verifies each solution; on failure, the agent analyzes errors and retries with corrected code (up to `MAX_RETRIES`)
- **Image-to-LaTeX OCR** — accepts image uploads via the UI; extracts LaTeX using local pix2tex (LatexOCR) and feeds it into the text pipeline
- **Streamlit UI** — single-page app with example problems, custom input, image upload, and side-by-side Agent vs Raw LLM comparison

The project uses a **zero-database design** — no persistent storage required.

---

## 2. Technology Stack

| Component | Technology |
|---|---|
| Agent Framework | LangGraph (state-machine graphs with conditional edges) |
| LLM Integration | LangChain + `ChatOpenAI` (Together AI API) |
| Symbolic Math | SymPy |
| Code Execution | Sandboxed Python REPL (threading + timeout) |
| Image OCR | pix2tex (LatexOCR) — local, free |
| Web UI | Streamlit |
| Data Storage | JSON files (optional) |
| Configuration | `dataclasses` with hardcoded Together AI settings |

---

## 3. Directory Structure

```
.
├── config.py              # Together AI configuration (hardcoded)
├── requirements.txt       # Python dependencies
├── README.md              # Human-facing documentation
├── AGENTS.md              # This file
├── data/                  # Optional result storage (JSON)
│   └── .gitkeep
├── agent/                 # Core JEE agent implementation
│   ├── __init__.py        # Exports JEEAgent, JEEAgentState, SymPyTool, MathImageOCR
│   ├── jee_state.py       # JEEAgentState TypedDict
│   ├── jee_prompts.py     # JEE-specific prompts + TOOL_DESCRIPTIONS
│   ├── jee_nodes.py       # JEE-specific graph nodes
│   ├── jee_graph.py       # JEEAgent class
│   ├── sympy_tools.py     # SymPy sandboxed execution toolkit
│   └── image_ocr.py       # pix2tex LaTeX OCR wrapper
├── benchmark/             # Evaluation suite
│   ├── __init__.py        # Exports JEELoader, JEEEValuator
│   ├── jee_loader.py      # Embedded dataset of ~140 JEE problems
│   └── jee_evaluator.py   # Agent vs raw LLM evaluator
└── interface/
    └── jee_app.py         # Streamlit web interface
```

---

## 4. Architecture

### JEEAgent Reasoning Pipeline

```
analyze → plan → solve_symbolic (self-loop) → consolidate → verify → format → END
                          ↑___________|
                          └ reflect ←─┘ (on failure, up to MAX_RETRIES)
```

| Stage | Node | Purpose |
|---|---|---|
| 1 | `analyze` | Classify problem type, identify key concepts |
| 2 | `plan` | Generate step-by-step solution plan with SymPy tool calls |
| 3 | `solve_symbolic` | Execute each planned step iteratively (self-loop in graph) |
| 4 | `consolidate` | Merge step results into final answer |
| 5 | `verify` | LLM checks correctness, confidence, issues |
| 6 | `reflect` | On failure, analyze errors and regenerate corrected code |
| 7 | `format` | Produce LaTeX + plain-text final answer |

### Sandboxed Code Execution

- **`SymPyTool`** (`agent/sympy_tools.py`): Pre-injects SymPy symbols/functions into namespace, strips `import sympy` lines, 10s timeout, captures `result` variable automatically.

### JSON-Mode LLM Communication

- LLM initialized with `model_kwargs={"response_format": {"type": "json_object"}}` when `JSON_MODE` is enabled.
- `parse_json_response()` (in `agent/jee_nodes.py`) handles markdown fences, trailing commas, regex extraction, and manual key-value parsing as fallbacks.

---

## 5. Build and Run Commands

### Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export TOGETHER_API_KEY="your-key"
```

### Launch Streamlit App

```bash
streamlit run interface/jee_app.py
```

Then open http://localhost:8501.

---

## 6. Code Style and Conventions

### Configuration

- Centralized in `config.py` via a `Config` dataclass.
- Hardcoded for Together AI: `LLM_BASE_URL="https://api.together.xyz/v1"`, `LLM_MODEL="meta-llama/Llama-3.3-70B-Instruct-Turbo"`.
- API key read from `TOGETHER_API_KEY` environment variable.

### State Definitions

- `JEEAgentState` uses `TypedDict` with **plain `list` fields** (not `Annotated[list, operator.add]`).
- **Important**: `steps`, `symbolic_steps`, and `reasoning_trace` are plain lists. Nodes mutate them in-place (e.g., `trace.append(...)`) and return the same list object. Using `Annotated[list, operator.add]` would cause catastrophic unbounded duplication on every graph self-loop, leading to exponential memory growth (O(n²) state size per loop).
- State fields are documented with docstrings in `jee_state.py`.

### Node Functions

- Each node takes `(state, llm, config)` and returns a `dict` of state updates.
- LLM and config are injected via lambdas in graph builders (see `jee_graph.py`).

### Error Handling

- **Fallback-heavy**: Every LLM call has try/except with sensible fallbacks.
- Never let an unhandled exception crash the graph; return error messages in state fields.

### Prompts

- Prompt templates are module-level plain Python strings with `{placeholder}` formatting.
- `TOOL_DESCRIPTIONS` is duplicated in both `jee_prompts.py` and `sympy_tools.py` so both the LLM context and the tool class can reference them.

### Imports

- Prefer absolute imports from the project root (e.g., `from config import Config`, `from agent.jee_graph import JEEAgent`).
- `interface/jee_app.py` manually adds the project root to `sys.path` so it works when run directly via `streamlit run`.

---

## 7. Testing Strategy

**There is no formal test suite.** Validation is done through:

1. **Streamlit app** — `streamlit run interface/jee_app.py`
2. **Example problems** — click a curated example, verify agent produces correct answer + reasoning trace
3. **Custom problems** — paste a problem, compare agent vs raw LLM side-by-side

---

## 8. Key Integration Points

```
interface/jee_app.py (Streamlit)
  ├── config.Config ──► hardcoded Together AI settings
  ├── agent.jee_graph.JEEAgent ──► single problem solve
  ├── agent.image_ocr.MathImageOCR ──► pix2tex LaTeX OCR (local, free)
  ├── benchmark.jee_loader.JEELoader ──► 3 curated example problems
  ├── benchmark.jee_evaluator.JEEEValuator ──► raw LLM baseline
  └── streamlit ──► UI rendering

agent/jee_graph.py (JEEAgent)
  ├── agent.jee_state.JEEAgentState ──► state schema
  ├── agent.jee_nodes.* ──► graph node functions
  ├── agent.jee_prompts.* ──► prompt templates
  ├── agent.sympy_tools.SymPyTool ──► symbolic execution
  ├── agent.image_ocr.MathImageOCR ──► image-to-LaTeX OCR
  └── langchain_openai.ChatOpenAI ──► LLM client (Together AI)
```

---

## 9. Security Considerations

- **Code execution is sandboxed but not fully isolated** — `SymPyTool` uses restricted builtins and module whitelisting, but it still runs `exec()` in the same process. Do not expose to untrusted user input in production without additional containerization.
- **API keys are read from environment variables** — Never commit keys to git. The `.gitignore` covers `.env` files.
- **No input sanitization on problem text** — The agent passes user-provided problem strings directly to the LLM. While the LLM is the first line of defense, the SymPy sandbox is the second.

---

## 10. How to Extend

### Add a new SymPy tool

1. Add the method to `agent/sympy_tools.py` in the appropriate section.
2. Add the description to the `TOOL_DESCRIPTIONS` constant at the bottom of `sympy_tools.py`.
3. Add the corresponding description to `TOOL_DESCRIPTIONS` in `agent/jee_prompts.py`.
4. Add dispatch logic in `agent/jee_nodes.py` `_dispatch_sympy_op()` if the tool should be callable from plan steps.

### Add image support to a problem

1. The UI (`interface/jee_app.py`) already supports image upload.
2. When an image is uploaded, `MathImageOCR` (`agent/image_ocr.py`) extracts LaTeX using pix2tex locally.
3. The extracted LaTeX is appended to the problem text before entering the agent graph.
4. No graph node changes are required — the image is converted to text upstream.

### Add new JEE problems to the dataset

1. Append problem dictionaries to `JEE_PROBLEMS` in `benchmark/jee_loader.py`.
2. Each problem dict should have keys: `question`, `answer`, `answer_type`, `problem_type`, `topic`, `difficulty`, `solution_method`.

---

*Last updated: 2026-05-29*
